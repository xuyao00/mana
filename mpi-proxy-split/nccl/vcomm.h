/*
 * vcomm -- virtual communicator registry.
 *
 * The plugin destroys every NCCL communicator before a checkpoint (so no CUDA
 * IPC/GDR mappings are live when the CUDA plugin snapshots the process) and
 * creates brand-new ones afterwards. The real ncclComm_t value therefore
 * changes across a checkpoint, but the application still holds its old handle
 * in its own variables.
 *
 * So the application never sees a real handle. It gets VCOMM_HANDLE(i), a
 * stable synthetic value naming a slot in g_vcomms[], and every wrapper
 * translates that to the currently-live real handle on the way in.
 *
 * The registry is a fixed static array: it is checkpointed as ordinary memory
 * with no heap or pointer fixup, and iteration is a plain loop. Only the
 * `real` field goes stale across a checkpoint; everything else is restored
 * as-is and is exactly what rebuild needs.
 *
 * This translation unit deliberately depends on nothing but libc, pthreads and
 * <nccl.h> -- no DMTCP, no CUDA runtime, no dlsym -- so the host-only unit test
 * (make tests/cc) can link it and exercise the identity and ordering logic
 * without a GPU.
 */

#ifndef VCOMM_H
#define VCOMM_H

#include <nccl.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_VCOMMS 16
#define VCOMM_MAX_RANKS 64  /* max ranks in ONE communicator */
#define VCOMM_IDENT_LEN 16  /* 128-bit identity */
#define VCOMM_IDENT_HEX 33  /* 32 hex chars + NUL */
#define VCOMM_NAME_MAX 64   /* bounded copy of config.netName / config.commName */

/*
 * Handle encoding:
 *      bits 63..32  VCOMM_TAG
 *      bits 31..16  reserved (future use-after-destroy generation counter)
 *      bits 15..0   index into g_vcomms[]
 *
 * VCOMM_TAG makes the result non-canonical on x86-64 (bits 63:48 != bit 47), so
 * a wrapper that forgets to translate faults immediately, at an address that
 * names the bug, instead of handing this struct to real NCCL.
 */
#define VCOMM_TAG 0xC0FFEE00u
#define VCOMM_HANDLE(idx)                                                                          \
    ((ncclComm_t)(uintptr_t)(((uint64_t)VCOMM_TAG << 32) | (uint64_t)(uint16_t)(idx)))

typedef enum {
    VCOMM_FREE = 0,     /* slot unused                                          */
    VCOMM_REMOTE,       /* rebuilt via ncclCommInitRank with a fresh uniqueId.
                         * Covers InitRank, InitRankConfig and CommSplit alike:
                         * membership is implicit in the rendezvous, so no
                         * parent/color/key is needed to reconstruct.           */
    VCOMM_LOCAL_CLIQUE  /* ncclCommInitAll -- N ranks inside THIS process, so it
                         * is rebuilt by one ncclCommInitAll call and needs no
                         * cross-process rendezvous at all.                     */
} vcomm_origin_t;

typedef enum {
    VCOMM_OK = 0,
    VCOMM_ERR_FULL,     /* registry exhausted                                   */
    VCOMM_ERR_DUPLICATE /* identity already live: hash collision, or the app
                         * illegally reused one ncclUniqueId for two comms       */
} vcomm_status_t;

typedef struct {
    vcomm_origin_t origin;

    /* --- identity: written once at creation, never recomputed. Agreed by every
     *     member of the comm with zero communication. --- */
    uint8_t ident[VCOMM_IDENT_LEN];

    /* --- the rebuild arguments. `rank`/`nranks` are in THIS comm's rank space,
     *     which is not the job-global one: rank 1 of comm_world may be rank 0
     *     of a sub-communicator. --- */
    int nranks;
    int rank;
    int cuda_dev; /* from ncclCommCuDevice -- never assume device == rank */

    int nsplits; /* splits taken OF this comm, to distinguish two
                  * same-color splits of the same parent            */
    int clique;  /* ncclCommInitAll group index, -1 if none         */

    /* --- effective config, so a restart does not silently change comm
     *     behaviour. netName/commName point into application memory, so hold
     *     bounded copies and repoint at rebuild (see vcomm_config). --- */
    int has_config;
    ncclConfig_t config;
    char net_name[VCOMM_NAME_MAX];
    char comm_name[VCOMM_NAME_MAX];

    /* --- collective checkpoint boundary --- */
    int64_t seq;
    int64_t target;

    /* --- p2p drain accounting, indexed by peer rank IN THIS COMM'S rank space --- */
    int64_t sent_to[VCOMM_MAX_RANKS];
    int64_t recvd_from[VCOMM_MAX_RANKS];

    /* --- the only field that goes stale across a checkpoint. NULL while torn
     *     down or mid-rebuild; published last so readers never see a
     *     half-built entry. --- */
    ncclComm_t volatile real;
} vcomm_t;

extern vcomm_t g_vcomms[MAX_VCOMMS];

/*
 * Translate an application handle. Returns NULL when the handle is NOT one of
 * ours -- a real NCCL handle, or NULL -- and callers must then forward the
 * argument verbatim. This is what lets plugin-internal code call an interposed
 * symbol with a real handle, and makes ncclXxx(..., NULL, ...) behave exactly
 * as unwrapped NCCL does. It must never abort on a miss.
 */
static inline vcomm_t *vcomm_lookup(ncclComm_t handle) {
    uint64_t u = (uint64_t)(uintptr_t)handle;
    uint16_t i;
    if ((u >> 32) != VCOMM_TAG)
        return NULL;
    i = (uint16_t)u;
    if (i >= MAX_VCOMMS || g_vcomms[i].origin == VCOMM_FREE)
        return NULL;
    return &g_vcomms[i];
}

int vcomm_index(const vcomm_t *v);
ncclComm_t vcomm_handle(const vcomm_t *v);

/* Allocate a slot. Does not populate anything but origin and ident. */
vcomm_status_t vcomm_new(vcomm_origin_t origin, const uint8_t ident[VCOMM_IDENT_LEN],
                         vcomm_t **out);
void vcomm_free(vcomm_t *v);

vcomm_t *vcomm_find(const uint8_t ident[VCOMM_IDENT_LEN]);
int vcomm_live_count(void);

/*
 * Fill order[] with the indices of live entries in ascending identity order and
 * return the count. Every member of a comm derives the same identity, so all
 * ranks agree on this order, and each rank's list is a subsequence of one global
 * total order -- which is what makes the collective rebuild deadlock-free.
 */
int vcomm_order(int order[MAX_VCOMMS]);

/* seq = 0, target = -1 on every live entry. Identity is NOT touched. */
void vcomm_reset_seq(void);

/* --- identity derivation (MurmurHash3-x64-128) --- */

void vcomm_digest128(const void *key, size_t len, uint8_t out[VCOMM_IDENT_LEN]);

/* ncclCommInitRank / ncclCommInitRankConfig: the uniqueId is already a
 * universal name for the comm -- every member passes byte-identical bytes. */
void vcomm_ident_from_id(const ncclUniqueId *id, uint8_t out[VCOMM_IDENT_LEN]);

/* ncclCommSplit: no uniqueId exists. All members of a child share the parent
 * and the color, and ncclCommSplit is collective on the parent, so the split
 * index is agreed too. */
void vcomm_ident_from_split(const uint8_t parent[VCOMM_IDENT_LEN], int color, int split_index,
                            uint8_t out[VCOMM_IDENT_LEN]);

/* ncclCommInitAll: process-local, never published, so this only needs local
 * uniqueness plus stability across restart. */
void vcomm_ident_local(int clique_idx, int rank, uint8_t out[VCOMM_IDENT_LEN]);

int vcomm_ident_cmp(const uint8_t a[VCOMM_IDENT_LEN], const uint8_t b[VCOMM_IDENT_LEN]);
void vcomm_ident_hex(const uint8_t ident[VCOMM_IDENT_LEN], char out[VCOMM_IDENT_HEX]);

/* --- config --- */

void vcomm_store_config(vcomm_t *v, const ncclConfig_t *config);

/*
 * Returns a config pointer suitable for handing to ncclCommInitRankConfig, with
 * netName/commName repointed at this entry's own bounded copies (the app's
 * strings are long gone by rebuild time), or NULL if the comm was created
 * without a config.
 */
ncclConfig_t *vcomm_config(vcomm_t *v);

#ifdef __cplusplus
}
#endif

#endif /* VCOMM_H */
