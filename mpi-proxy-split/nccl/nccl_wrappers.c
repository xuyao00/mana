/*
 * NCCL interposition (from the nccl-mana-ckpt prototype, src/nccl_wrappers.c).
 *
 * In MANA, NCCL lives in the lower half, like MPI: libmana.so defines these
 * nccl* functions in the upper half, and the "real" functions are thunks
 * (generated, nccl_lh_thunks.cpp) that call the lower half's NCCL with the
 * lower half's FS.  The functions of nccl.h not defined here forward to the
 * lower half unchanged (generated, nccl_forward.c).
 *
 * The application never holds a real ncclComm_t -- it holds a virtual handle
 * naming a registry slot (see vcomm.h), because MANA destroys and recreates
 * communicators around a checkpoint and the real handle value changes.
 *
 * Rules every wrapper follows:
 *
 *   R1  A real handle never reaches application-visible memory. Comm-creating
 *       wrappers keep the real handle in a local and store a virtual handle.
 *   R2  vcomm_lookup() runs first. A miss means "not one of ours" -- a real
 *       handle, or NULL -- and the argument is forwarded verbatim.
 *   R3  Nothing reads v->real directly; it goes through enter_nccl()/wait_real(),
 *       which block while the comm is torn down or mid-rebuild.
 *   R4  Collectives pass the sequence gate before taking a live handle.
 *   R5  Checkpoint code uses the call_real_* helpers, never the interposed
 *       names, so wrapper bookkeeping is not re-entered.
 *
 * Interposing ALL of these is not optional polish. Once the application holds a
 * virtual handle, any comm-taking entry point left unwrapped receives a
 * deliberately non-canonical pointer and faults.
 */

#define _GNU_SOURCE
#include "nccl_wrappers.h"
#include "dmtcp.h"
#include "inflight.h"
#include "vcomm.h"
#include "nccl_cuda.h"
#include "nccl_lh.h"
#include <dlfcn.h>
#include <link.h>
#include <nccl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Newer entry points are wrapped only when the NCCL headers in use declare
 * them, so the plugin still builds against older NCCL. */
#define NCCL_HAS_2_30 (defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= 23000)
/* ncclGroupSimulateEnd and ncclSimInfo_t arrived in NCCL 2.22. */
#define NCCL_HAS_SIM_INFO (defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= 22200)

#define GATE_POLL_US 200

static ncclResult_t (*real_ncclSend)(const void *, size_t, ncclDataType_t, int, ncclComm_t,
                                     cudaStream_t) = NULL;
static ncclResult_t (*real_ncclRecv)(void *, size_t, ncclDataType_t, int, ncclComm_t,
                                     cudaStream_t) = NULL;
static ncclResult_t (*real_ncclAllReduce)(const void *, void *, size_t, ncclDataType_t, ncclRedOp_t,
                                          ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclBroadcast)(const void *, void *, size_t, ncclDataType_t, int,
                                          ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclBcast)(void *, size_t, ncclDataType_t, int, ncclComm_t,
                                      cudaStream_t) = NULL;
static ncclResult_t (*real_ncclReduce)(const void *, void *, size_t, ncclDataType_t, ncclRedOp_t,
                                       int, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclAllGather)(const void *, void *, size_t, ncclDataType_t, ncclComm_t,
                                          cudaStream_t) = NULL;
static ncclResult_t (*real_ncclReduceScatter)(const void *, void *, size_t, ncclDataType_t,
                                              ncclRedOp_t, ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclGather)(const void *, void *, size_t, ncclDataType_t, int,
                                       ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclScatter)(const void *, void *, size_t, ncclDataType_t, int,
                                        ncclComm_t, cudaStream_t) = NULL;
static ncclResult_t (*real_ncclAlltoAll)(const void *, void *, size_t, ncclDataType_t, ncclComm_t,
                                         cudaStream_t) = NULL;
static ncclResult_t (*real_ncclCommInitRank)(ncclComm_t *, int, ncclUniqueId, int) = NULL;
static ncclResult_t (*real_ncclCommInitRankConfig)(ncclComm_t *, int, ncclUniqueId, int,
                                                   ncclConfig_t *) = NULL;
static ncclResult_t (*real_ncclCommInitAll)(ncclComm_t *, int, const int *) = NULL;
static ncclResult_t (*real_ncclCommSplit)(ncclComm_t, int, int, ncclComm_t *,
                                          ncclConfig_t *) = NULL;
static ncclResult_t (*real_ncclCommDestroy)(ncclComm_t) = NULL;
static ncclResult_t (*real_ncclCommFinalize)(ncclComm_t) = NULL;
static ncclResult_t (*real_ncclCommAbort)(ncclComm_t) = NULL;
static ncclResult_t (*real_ncclCommCount)(const ncclComm_t, int *) = NULL;
static ncclResult_t (*real_ncclCommUserRank)(const ncclComm_t, int *) = NULL;
static ncclResult_t (*real_ncclCommCuDevice)(const ncclComm_t, int *) = NULL;
static ncclResult_t (*real_ncclCommGetAsyncError)(ncclComm_t, ncclResult_t *) = NULL;
static ncclResult_t (*real_ncclCommRegister)(const ncclComm_t, void *, size_t, void **) = NULL;
static ncclResult_t (*real_ncclCommDeregister)(const ncclComm_t, void *) = NULL;
static ncclResult_t (*real_ncclRedOpCreatePreMulSum)(ncclRedOp_t *, void *, ncclDataType_t,
                                                     ncclScalarResidence_t, ncclComm_t) = NULL;
static ncclResult_t (*real_ncclRedOpDestroy)(ncclRedOp_t, ncclComm_t) = NULL;
static ncclResult_t (*real_ncclGroupStart)(void) = NULL;
static ncclResult_t (*real_ncclGroupEnd)(void) = NULL;
#if NCCL_HAS_SIM_INFO
static ncclResult_t (*real_ncclGroupSimulateEnd)(ncclSimInfo_t *) = NULL;
#endif
static const char *(*real_ncclGetLastError)(ncclComm_t) = NULL;
#if NCCL_HAS_2_30
static ncclResult_t (*real_ncclCommGetUniqueId)(ncclComm_t, ncclUniqueId *) = NULL;
static ncclResult_t (*real_ncclCommMemStats)(ncclComm_t, ncclCommMemStat_t, uint64_t *) = NULL;
static ncclResult_t (*real_ncclCommShrink)(ncclComm_t, int *, int, ncclComm_t *, ncclConfig_t *,
                                           int) = NULL;
static ncclResult_t (*real_ncclCommGrow)(ncclComm_t, int, const ncclUniqueId *, int, ncclComm_t *,
                                         ncclConfig_t *) = NULL;
static ncclResult_t (*real_ncclCommRevoke)(ncclComm_t, int) = NULL;
static ncclResult_t (*real_ncclCommSuspend)(ncclComm_t, int) = NULL;
static ncclResult_t (*real_ncclCommResume)(ncclComm_t) = NULL;
static ncclResult_t (*real_ncclCommWindowRegister)(ncclComm_t, void *, size_t, ncclWindow_t *,
                                                   int) = NULL;
static ncclResult_t (*real_ncclCommWindowDeregister)(ncclComm_t, ncclWindow_t) = NULL;
static ncclResult_t (*real_ncclWinGetUserPtr)(ncclComm_t, ncclWindow_t, void **) = NULL;
static ncclResult_t (*real_ncclSignal)(int, int, int, unsigned int, ncclComm_t,
                                       cudaStream_t) = NULL;
static ncclResult_t (*real_ncclWaitSignal)(int, ncclWaitSignalDesc_t *, ncclComm_t,
                                           cudaStream_t) = NULL;
static ncclResult_t (*real_ncclPutSignal)(const void *, size_t, ncclDataType_t, int, ncclWindow_t,
                                          size_t, int, int, unsigned int, ncclComm_t,
                                          cudaStream_t) = NULL;
#endif

/*
 * The lower half's NCCL function 'name', called through its thunk, or NULL if
 * the lower half's NCCL does not have it.  The upper half's libnccl is MANA's
 * stub, so this never looks in the upper half.
 */
void *nccl_real_dlsym(const char *name) {
    return mana_nccl_real_dlsym(name);
}

#define LOAD_SYM(name)                                                                             \
    do {                                                                                           \
        if (!real_##name)                                                                          \
            *(void **)(&real_##name) = nccl_real_dlsym(#name);                                     \
    } while (0)

/*
 * Resolve a symbol we are about to call, and refuse rather than jump to NULL if
 * it is missing. Without this an unresolvable symbol becomes a call to address
 * 0, which is indistinguishable in a core file from a bad handle -- the two most
 * confusing failure modes in this plugin looking identical.
 */
#define REQUIRE_SYM(name, retval)                                                                  \
    do {                                                                                           \
        LOAD_SYM(name);                                                                            \
        if (real_##name == NULL) {                                                                 \
            fprintf(stderr,                                                                        \
                    "[NCCL PLUGIN] cannot resolve real \"%s\" in the lower half's NCCL;"         \
                    " refusing to call through NULL\n",                                            \
                    #name);                                                                        \
            fflush(stderr);                                                                        \
            return retval;                                                                         \
        }                                                                                          \
    } while (0)

/*
 * A comm-taking entry point we cannot make survive a checkpoint. Refusing here,
 * loudly, beats letting the application run and then failing at checkpoint time
 * -- or, worse, handing our non-canonical handle to real NCCL.
 */
#define REFUSE_UNSUPPORTED(fn)                                                                     \
    do {                                                                                           \
        fprintf(stderr,                                                                            \
                "[NCCL PLUGIN] %s() is not supported by this checkpointing plugin.\n"               \
                "              Its communicator cannot be reconstructed after restart.\n",         \
                (fn));                                                                             \
        fflush(stderr);                                                                            \
        return ncclInvalidUsage;                                                                   \
    } while (0)

/* ── sequence gate ────────────────────────────────────────────────── */

/*
 * Guards seq/target across every registry entry. The gate decision and the
 * seq++ it authorizes must be one critical section: otherwise a rank can be
 * judged "settled" by the checkpoint thread and then slip one more collective
 * through before the barrier, leaving an enqueued collective its peers will
 * never match -- silent numerical corruption rather than a hang.
 */
static pthread_mutex_t g_seq_lock = PTHREAD_MUTEX_INITIALIZER;

/* Threads inside a real NCCL call or an open group region. */
static volatile int g_nccl_activity;

/* Depth of ncclGroupStart nesting, per thread. */
static __thread int t_group_depth;

void nccl_seq_lock(void) {
    pthread_mutex_lock(&g_seq_lock);
}

void nccl_seq_unlock(void) {
    pthread_mutex_unlock(&g_seq_lock);
}

int nccl_activity(void) {
    return __atomic_load_n(&g_nccl_activity, __ATOMIC_SEQ_CST);
}

int mana_mpi_behind_target(void); /* seq_num.cpp */

int nccl_gate_closed_locked(void) {
    int i;
    int live = 0;
    if (!ckpt_pending)
        return 0;
    /* One cut with MPI's: a thread may owe an MPI collective that a peer
     * waits for (seq_num.h, g_other_settled). */
    if (mana_mpi_behind_target())
        return 0;
    for (i = 0; i < MAX_VCOMMS; i++) {
        vcomm_t *v = &g_vcomms[i];
        if (v->origin == VCOMM_FREE)
            continue;
        live++;
        /* target < 0 means "no consensus yet", which counts as at-target: the
         * frontier must freeze the instant a checkpoint is requested. */
        if (v->target >= 0 && v->seq < v->target)
            return 0; /* this rank still owes progress somewhere */
    }
    /* No live comm means no frontier to freeze, and the call this thread is
     * about to make is a creation rendezvous. Holding it deadlocks against a
     * peer already inside real_ncclGroupEnd(), whose raised activity count then
     * spins quiesce_nccl() forever. The empty loop used to return "closed". */
    return live > 0;
}

/*
 * Wait until v->real is live. Safe to block here even while holding the DMTCP
 * wrapper lock: comms are rebuilt at DMTCP_EVENT_RUNNING_AFTER, which runs
 * before WorkerState becomes RUNNING, so no new checkpoint can begin while we
 * wait.
 */
static ncclComm_t wait_real(vcomm_t *v) {
    ncclComm_t real;
    while ((real = v->real) == NULL)
        usleep(GATE_POLL_US);
    return real;
}

/*
 * Pass the gate and take a live handle. Returns with checkpointing DISABLED and
 * g_nccl_activity incremented; the caller must call leave_nccl().
 *
 * Activity is bumped under the same lock as the gate test, so a thread that got
 * through is immediately visible to the checkpoint thread's quiesce and cannot
 * be mistaken for an idle process.
 */
static ncclComm_t enter_nccl(vcomm_t *v, int bump_seq) {
    for (;;) {
        DMTCP_PLUGIN_DISABLE_CKPT();
        pthread_mutex_lock(&g_seq_lock);
        if (!nccl_gate_closed_locked()) {
            if (bump_seq)
                v->seq++;
            __atomic_add_fetch(&g_nccl_activity, 1, __ATOMIC_SEQ_CST);
            pthread_mutex_unlock(&g_seq_lock);
            break;
        }
        pthread_mutex_unlock(&g_seq_lock);
        /* Never park holding the DMTCP wrapper lock: the checkpoint thread
         * needs it exclusively to suspend threads. */
        DMTCP_PLUGIN_ENABLE_CKPT();
        usleep(GATE_POLL_US);
    }
    return wait_real(v);
}

static void leave_nccl(void) {
    __atomic_sub_fetch(&g_nccl_activity, 1, __ATOMIC_SEQ_CST);
    DMTCP_PLUGIN_ENABLE_CKPT();
}

/*
 * Inside an open group region the enclosing ncclGroupStart already holds the
 * gate, the activity count and the wrapper lock, so an individual op must not
 * take them again -- and must never park, since peers are waiting for the batch
 * to close.
 */
static int in_group_region(void) {
    return t_group_depth > 0;
}

/*
 * Communicators registered while a group region was open.
 *
 * Inside a region NCCL only queues the init job: it keeps the ncclComm_t* it was
 * handed and writes through it at ncclGroupEnd, so that destination must outlive
 * the call -- a local, as every creation path used to pass, is a write into a
 * returned stack frame. The registry slot is stable and is also the right
 * publication point: v->real stays NULL, so wait_real() parks, until the comm
 * exists. Anything unknowable until then is filled in here.
 */
static void record_comm_ranks(vcomm_t *v, int nranks, int rank);
static void record_comm_device(vcomm_t *v, ncclComm_t real);

typedef struct {
    int index;
    int query_ranks; /* rank/size must be read off the comm; a split child */
} pending_create_t;

static __thread pending_create_t t_pending[MAX_VCOMMS];
static __thread int t_npending;

static void pending_create_add(vcomm_t *v, int query_ranks) {
    if (t_npending >= MAX_VCOMMS) {
        fprintf(stderr,
                "[NCCL PLUGIN] more than %d communicators created in one group region.\n"
                "              Raise MAX_VCOMMS in src/vcomm.h and rebuild.\n",
                MAX_VCOMMS);
        fflush(stderr);
        abort();
    }
    t_pending[t_npending].index = vcomm_index(v);
    t_pending[t_npending].query_ranks = query_ranks;
    t_npending++;
}

static void finish_pending_creates(int ok) {
    int i;

    for (i = 0; i < t_npending; i++) {
        vcomm_t *v = &g_vcomms[t_pending[i].index];
        ncclComm_t real = v->real;

        if (!ok || real == NULL) {
            /* A registered comm that does not exist would still be published
             * for rebuild, blocking every peer in ncclCommInitRank. */
            v->real = NULL;
            vcomm_free(v);
            continue;
        }

        if (t_pending[i].query_ranks) {
            int nranks = 0, rank = 0;
            LOAD_SYM(ncclCommCount);
            LOAD_SYM(ncclCommUserRank);
            if (real_ncclCommCount == NULL || real_ncclCommUserRank == NULL ||
                real_ncclCommCount(real, &nranks) != ncclSuccess ||
                real_ncclCommUserRank(real, &rank) != ncclSuccess) {
                fprintf(stderr,
                        "[NCCL PLUGIN] could not read rank/size of a communicator created in a"
                        " group region; it cannot be rebuilt after a checkpoint\n");
                fflush(stderr);
                v->real = NULL;
                vcomm_free(v);
                continue;
            }
            record_comm_ranks(v, nranks, rank);
        }
        record_comm_device(v, real);
        init_rank_info(v);
    }
    t_npending = 0;
}

/* ── accounting ───────────────────────────────────────────────────── */

static size_t get_nccl_type_size(ncclDataType_t type) {
    switch (type) {
    case ncclInt8:
    case ncclUint8:
        return 1;
    case ncclFloat16:
        return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
        return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
        return 8;
    default:
        return 4;
    }
}

/*
 * p2p byte counters, indexed by peer rank in THIS comm's rank space. Written
 * only by application threads; read by the checkpoint thread after the quiesce
 * has driven activity to zero, so no lock is needed.
 */
static int peer_ok(const vcomm_t *v, int peer, const char *what) {
    if (peer >= 0 && peer < VCOMM_MAX_RANKS && peer < v->nranks)
        return 1;
    /* The previous code indexed a fixed table with no check at all, so a peer
     * beyond the bound silently corrupted adjacent memory. */
    fprintf(stderr,
            "[NCCL PLUGIN] %s: peer %d outside this communicator (nranks=%d, cap=%d);"
            " drain accounting would be corrupted\n",
            what, peer, v->nranks, VCOMM_MAX_RANKS);
    fflush(stderr);
    return 0;
}

/* ── real-symbol access for the plugin (R5) ───────────────────────── */

ncclResult_t call_real_ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype,
                                int peer, ncclComm_t real_comm, cudaStream_t stream) {
    REQUIRE_SYM(ncclSend, ncclInternalError);
    return real_ncclSend(sendbuff, count, datatype, peer, real_comm, stream);
}

ncclResult_t call_real_ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype, int peer,
                                ncclComm_t real_comm, cudaStream_t stream) {
    REQUIRE_SYM(ncclRecv, ncclInternalError);
    return real_ncclRecv(recvbuff, count, datatype, peer, real_comm, stream);
}

ncclResult_t call_real_ncclCommDestroy(ncclComm_t real_comm) {
    REQUIRE_SYM(ncclCommDestroy, ncclInternalError);
    return real_ncclCommDestroy(real_comm);
}

ncclResult_t call_real_ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId commId,
                                              int rank, ncclConfig_t *config) {
    if (config == NULL) {
        REQUIRE_SYM(ncclCommInitRank, ncclInternalError);
        return real_ncclCommInitRank(comm, nranks, commId, rank);
    }
    LOAD_SYM(ncclCommInitRankConfig);
    if (real_ncclCommInitRankConfig == NULL) {
        REQUIRE_SYM(ncclCommInitRank, ncclInternalError);
        return real_ncclCommInitRank(comm, nranks, commId, rank);
    }
    return real_ncclCommInitRankConfig(comm, nranks, commId, rank, config);
}

ncclResult_t call_real_ncclCommInitAll(ncclComm_t *comms, int ndev, const int *devlist) {
    REQUIRE_SYM(ncclCommInitAll, ncclInternalError);
    return real_ncclCommInitAll(comms, ndev, devlist);
}

ncclResult_t call_real_ncclCommGetAsyncError(ncclComm_t real_comm, ncclResult_t *asyncError) {
    REQUIRE_SYM(ncclCommGetAsyncError, ncclInternalError);
    return real_ncclCommGetAsyncError(real_comm, asyncError);
}

/* ── registration ─────────────────────────────────────────────────── */

/*
 * Populate a freshly allocated entry from the live communicator. rank/nranks are
 * read from NCCL rather than assumed, and the CUDA device is recorded because
 * NCCL binds a comm to the current device at creation -- device == rank is false
 * whenever CUDA_VISIBLE_DEVICES is set per rank, which is the normal Slurm case.
 */
static void record_comm_ranks(vcomm_t *v, int nranks, int rank) {
    v->nranks = nranks;
    v->rank = rank;

    if (v->nranks > VCOMM_MAX_RANKS) {
        fprintf(stderr,
                "[NCCL PLUGIN] communicator has %d ranks but the drain tables hold %d.\n"
                "              Raise VCOMM_MAX_RANKS in src/vcomm.h and rebuild.\n",
                v->nranks, VCOMM_MAX_RANKS);
        fflush(stderr);
        abort();
    }
}

/*
 * `real` is NULL for a comm created inside a group region. cudaGetDevice is not a
 * guess there: NCCL binds the comm to the creating thread's current device. The
 * probe is redone with the live handle once the region closes.
 */
static void record_comm_device(vcomm_t *v, ncclComm_t real) {
    int dev = -1;

    LOAD_SYM(ncclCommCuDevice);
    if (real != NULL && real_ncclCommCuDevice != NULL &&
        real_ncclCommCuDevice(real, &dev) == ncclSuccess) {
        v->cuda_dev = dev;
    } else if (cudaGetDevice(&dev) == cudaSuccess) {
        v->cuda_dev = dev;
    } else {
        v->cuda_dev = 0;
    }
}

static void record_comm(vcomm_t *v, ncclComm_t real, int nranks, int rank,
                        const ncclConfig_t *config) {
    record_comm_ranks(v, nranks, rank);
    record_comm_device(v, real);
    vcomm_store_config(v, config);

    /* Published last: an application thread must never observe a half-built
     * entry, and a non-NULL `real` is what releases wait_real(). */
    v->real = real;
}

static void registry_fatal(vcomm_status_t st, const char *fn) {
    if (st == VCOMM_ERR_FULL) {
        fprintf(stderr,
                "[NCCL PLUGIN] %s: communicator registry full (MAX_VCOMMS=%d).\n"
                "              Raise MAX_VCOMMS in src/vcomm.h and rebuild.\n",
                fn, MAX_VCOMMS);
    } else {
        fprintf(stderr,
                "[NCCL PLUGIN] %s: duplicate communicator identity.\n"
                "              Either one ncclUniqueId was reused for two communicators\n"
                "              (which NCCL does not permit), or a 128-bit digest collided.\n",
                fn);
    }
    fflush(stderr);
    abort();
}

/*
 * With ncclConfig_t.blocking = 0, creation succeeds by returning ncclInProgress
 * and the caller polls ncclCommGetAsyncError. The old `!= ncclSuccess` test read
 * that as failure: nothing registered, *comm unwritten, real handle leaked.
 */
static int init_ok(ncclResult_t res) {
    return res == ncclSuccess || res == ncclInProgress;
}

/* ── comm creation ────────────────────────────────────────────────── */

static ncclResult_t init_rank_common(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank,
                                     ncclConfig_t *config, const char *fn) {
    uint8_t ident[VCOMM_IDENT_LEN];
    ncclComm_t real = NULL; /* R1: the real handle stays in a local */
    vcomm_t *v = NULL;
    vcomm_status_t st;
    ncclResult_t res;

    int grouped = in_group_region();

    DMTCP_PLUGIN_DISABLE_CKPT();

    /* The uniqueId is already a universal name for this communicator: every
     * member passed byte-identical bytes. Digest it to a fixed width and
     * discard it -- keeping it would invite reusing a bootstrap address that is
     * dead after restart. */
    vcomm_ident_from_id(&commId, ident);

    if (grouped) {
        /* The handle is written at ncclGroupEnd, so the slot must exist first
         * and be the destination. */
        st = vcomm_new(VCOMM_REMOTE, ident, &v);
        if (st != VCOMM_OK)
            registry_fatal(st, fn);
        record_comm_ranks(v, nranks, rank);
        record_comm_device(v, NULL);
        vcomm_store_config(v, config);

        res = call_real_ncclCommInitRankConfig((ncclComm_t *)&v->real, nranks, commId, rank,
                                               config);
        if (!init_ok(res)) {
            v->real = NULL;
            vcomm_free(v);
            DMTCP_PLUGIN_ENABLE_CKPT();
            return res;
        }
        pending_create_add(v, 0);
    } else {
        res = call_real_ncclCommInitRankConfig(&real, nranks, commId, rank, config);
        if (!init_ok(res) || real == NULL) {
            DMTCP_PLUGIN_ENABLE_CKPT();
            return res;
        }
        st = vcomm_new(VCOMM_REMOTE, ident, &v);
        if (st != VCOMM_OK)
            registry_fatal(st, fn);
        record_comm(v, real, nranks, rank, config);
        init_rank_info(v);
    }
    *comm = vcomm_handle(v); /* R1 */

    DMTCP_PLUGIN_ENABLE_CKPT();
    return res;
}

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank) {
    return init_rank_common(comm, nranks, commId, rank, NULL, "ncclCommInitRank");
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank,
                                    ncclConfig_t *config) {
    return init_rank_common(comm, nranks, commId, rank, config, "ncclCommInitRankConfig");
}

/*
 * ncclCommInitAll creates a clique of ndev communicators, all inside THIS
 * process, one per device. Each is a separate communicator with its own rank, so
 * each gets its own registry entry, tied together by a clique index so the
 * rebuild can reproduce them with a single ncclCommInitAll call.
 *
 * The previous implementation passed &g_comm as the output array, which never
 * wrote the caller's array and, for ndev > 1, overran a single ncclComm_t into
 * the adjacent globals.
 */
ncclResult_t ncclCommInitAll(ncclComm_t *comm, int ndev, const int *devlist) {
    static int g_next_clique;
    ncclComm_t real[MAX_VCOMMS];
    vcomm_t *entries[MAX_VCOMMS];
    ncclResult_t res;
    int clique;
    int i;

    if (comm == NULL || ndev <= 0)
        return ncclInvalidArgument;
    if (ndev > MAX_VCOMMS) {
        fprintf(stderr,
                "[NCCL PLUGIN] ncclCommInitAll: %d devices exceeds MAX_VCOMMS=%d.\n"
                "              Raise MAX_VCOMMS in src/vcomm.h and rebuild.\n",
                ndev, MAX_VCOMMS);
        fflush(stderr);
        return ncclInvalidArgument;
    }

    DMTCP_PLUGIN_DISABLE_CKPT();

    memset(real, 0, sizeof(real));
    res = call_real_ncclCommInitAll(real, ndev, devlist);
    if (!init_ok(res)) {
        DMTCP_PLUGIN_ENABLE_CKPT();
        return res;
    }

    clique = g_next_clique++;
    for (i = 0; i < ndev; i++) {
        uint8_t ident[VCOMM_IDENT_LEN];
        vcomm_status_t st;
        /* Process-local and never published, so this identity only needs local
         * uniqueness plus stability across restart. */
        vcomm_ident_local(clique, i, ident);
        st = vcomm_new(VCOMM_LOCAL_CLIQUE, ident, &entries[i]);
        if (st != VCOMM_OK)
            registry_fatal(st, "ncclCommInitAll");
        entries[i]->clique = clique;
        record_comm(entries[i], real[i], ndev, i, NULL);
        comm[i] = vcomm_handle(entries[i]); /* R1 */
    }

    DMTCP_PLUGIN_ENABLE_CKPT();
    return res;
}

/*
 * ncclCommSplit. The child is rebuilt like any other communicator, via
 * ncclCommInitRank with a fresh uniqueId -- membership is implicit in the
 * rendezvous, so no parent/color/key needs to be stored. They are used here,
 * once, only to derive an identity all members of the child agree on.
 *
 * The split index matters: splitting the same parent twice with the same color
 * yields two independent communicators, and parent-plus-color alone would give
 * them the same identity. ncclCommSplit is collective on the parent, so every
 * member counts the same index -- which is why nsplits is advanced even for a
 * NCCL_SPLIT_NOCOLOR caller that gets no child back.
 */
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm,
                           ncclConfig_t *config) {
    vcomm_t *parent = vcomm_lookup(comm);
    uint8_t ident[VCOMM_IDENT_LEN];
    ncclComm_t real = NULL;
    ncclComm_t parent_real;
    vcomm_t *v = NULL;
    vcomm_status_t st;
    ncclResult_t res;
    int split_index;
    int grouped;
    int nranks = 0, rank = 0;

    LOAD_SYM(ncclCommSplit);
    if (real_ncclCommSplit == NULL)
        REFUSE_UNSUPPORTED("ncclCommSplit");
    if (parent == NULL) /* R2 */
        return real_ncclCommSplit(comm, color, key, newcomm, config);

    /* The enclosing ncclGroupStart already holds the gate, the activity count
     * and the wrapper lock. Taking them again could park a thread inside an open
     * batch its peers are waiting to see closed. */
    grouped = in_group_region();
    parent_real = grouped ? wait_real(parent) : enter_nccl(parent, 0);

    /* Take the index before the call, so all members agree even if some get
     * NULL back. */
    pthread_mutex_lock(&g_seq_lock);
    split_index = parent->nsplits++;
    pthread_mutex_unlock(&g_seq_lock);

    vcomm_ident_from_split(parent->ident, color, split_index, ident);

    if (grouped) {
        /* As in init_rank_common. rank and size cannot be read off the child
         * until the region closes either, so they are deferred rather than
         * recorded as the 0/0 that used to fail the rebuild. */
        st = vcomm_new(VCOMM_REMOTE, ident, &v);
        if (st != VCOMM_OK)
            registry_fatal(st, "ncclCommSplit");
        record_comm_device(v, NULL);
        vcomm_store_config(v, config ? config : vcomm_config(parent));

        res = real_ncclCommSplit(parent_real, color, key, (ncclComm_t *)&v->real, config);
        if (!init_ok(res)) {
            v->real = NULL;
            vcomm_free(v);
            return res;
        }
        /* NCCL_SPLIT_NOCOLOR resolves at group end: a NULL handle there means
         * "no child" and the slot is released. */
        pending_create_add(v, 1);
        if (newcomm != NULL)
            *newcomm = vcomm_handle(v); /* R1 */
        return res;
    }

    res = real_ncclCommSplit(parent_real, color, key, &real, config);
    if (!init_ok(res)) {
        leave_nccl();
        return res;
    }

    if (real == NULL) {
        /* NCCL_SPLIT_NOCOLOR: this rank is in no child communicator. Nothing to
         * register, and nothing to rebuild -- under uniform reconstruction a
         * non-member never participates in the child's rebuild at all. */
        if (newcomm != NULL)
            *newcomm = NULL;
        leave_nccl();
        return res;
    }

    st = vcomm_new(VCOMM_REMOTE, ident, &v);
    if (st != VCOMM_OK)
        registry_fatal(st, "ncclCommSplit");

    LOAD_SYM(ncclCommCount);
    LOAD_SYM(ncclCommUserRank);
    if (real_ncclCommCount == NULL || real_ncclCommUserRank == NULL ||
        real_ncclCommCount(real, &nranks) != ncclSuccess ||
        real_ncclCommUserRank(real, &rank) != ncclSuccess) {
        /* 0/0 would pass every check here, kill the drain via peer_ok, and
         * only then fail the rebuild. */
        fprintf(stderr, "[NCCL PLUGIN] ncclCommSplit: could not read the child's rank/size;"
                        " it cannot be rebuilt after a checkpoint\n");
        fflush(stderr);
        vcomm_free(v);
        leave_nccl();
        return ncclInternalError;
    }

    /* A child with no config inherits the parent's, so record the parent's copy
     * rather than defaulting -- otherwise a restart silently changes behaviour. */
    record_comm(v, real, nranks, rank, config ? config : vcomm_config(parent));

    if (newcomm != NULL)
        *newcomm = vcomm_handle(v); /* R1 */

    leave_nccl();
    return res;
}

/* ── comm teardown ────────────────────────────────────────────────── */

/*
 * Application-facing destroy. Distinct from the checkpoint teardown, which uses
 * call_real_ncclCommDestroy: this one also releases the registry slot, which at
 * checkpoint time would throw away the metadata needed to rebuild.
 *
 * The previous version ignored its argument entirely and always destroyed the
 * one global comm, so destroying three communicators destroyed the first one
 * once and leaked the other two.
 */
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    vcomm_t *v = vcomm_lookup(comm);
    ncclComm_t real;
    ncclResult_t res;

    REQUIRE_SYM(ncclCommDestroy, ncclInternalError);
    if (v == NULL) /* R2 */
        return real_ncclCommDestroy(comm);

    real = v->real;
    v->real = NULL;
    res = (real != NULL) ? real_ncclCommDestroy(real) : ncclSuccess;
    vcomm_free(v);
    return res;
}

ncclResult_t ncclCommFinalize(ncclComm_t comm) {
    vcomm_t *v = vcomm_lookup(comm);
    LOAD_SYM(ncclCommFinalize);
    if (real_ncclCommFinalize == NULL)
        return ncclSuccess;
    if (v == NULL) /* R2 */
        return real_ncclCommFinalize(comm);
    /* Leave the slot in place: ncclCommDestroy still has to run after this. */
    return real_ncclCommFinalize(wait_real(v));
}

ncclResult_t ncclCommAbort(ncclComm_t comm) {
    vcomm_t *v = vcomm_lookup(comm);
    ncclComm_t real;
    ncclResult_t res;

    LOAD_SYM(ncclCommAbort);
    if (real_ncclCommAbort == NULL)
        return ncclInvalidUsage;
    if (v == NULL) /* R2 */
        return real_ncclCommAbort(comm);

    real = v->real;
    v->real = NULL;
    res = (real != NULL) ? real_ncclCommAbort(real) : ncclSuccess;
    vcomm_free(v);
    return res;
}

/* ── answered from the registry ───────────────────────────────────── */

/*
 * No NCCL call at all, so these keep working while the communicator is torn
 * down for a checkpoint -- which matters because PyTorch's watchdog and plenty
 * of library code query rank and size at arbitrary moments.
 */

ncclResult_t ncclCommCount(const ncclComm_t comm, int *count) {
    vcomm_t *v = vcomm_lookup(comm);
    if (count == NULL)
        return ncclInvalidArgument;
    if (v == NULL) { /* R2 */
        REQUIRE_SYM(ncclCommCount, ncclInternalError);
        return real_ncclCommCount(comm, count);
    }
    *count = v->nranks;
    return ncclSuccess;
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *rank) {
    vcomm_t *v = vcomm_lookup(comm);
    if (rank == NULL)
        return ncclInvalidArgument;
    if (v == NULL) { /* R2 */
        REQUIRE_SYM(ncclCommUserRank, ncclInternalError);
        return real_ncclCommUserRank(comm, rank);
    }
    *rank = v->rank;
    return ncclSuccess;
}

ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int *device) {
    vcomm_t *v = vcomm_lookup(comm);
    if (device == NULL)
        return ncclInvalidArgument;
    if (v == NULL) { /* R2 */
        REQUIRE_SYM(ncclCommCuDevice, ncclInternalError);
        return real_ncclCommCuDevice(comm, device);
    }
    *device = v->cuda_dev;
    return ncclSuccess;
}

/* ── translate and forward ────────────────────────────────────────── */

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
    vcomm_t *v = vcomm_lookup(comm);
    REQUIRE_SYM(ncclCommGetAsyncError, ncclInternalError);
    if (v == NULL) /* R2 */
        return real_ncclCommGetAsyncError(comm, asyncError);
    /* Do not block: a watchdog thread polls this, and reporting success while
     * the comm is torn down for a checkpoint is the honest answer. */
    if (v->real == NULL) {
        if (asyncError != NULL)
            *asyncError = ncclSuccess;
        return ncclSuccess;
    }
    return real_ncclCommGetAsyncError(v->real, asyncError);
}

ncclResult_t ncclCommRegister(const ncclComm_t comm, void *buff, size_t size, void **handle) {
    vcomm_t *v = vcomm_lookup(comm);
    LOAD_SYM(ncclCommRegister);
    if (real_ncclCommRegister == NULL)
        return ncclInvalidUsage;
    if (v == NULL) /* R2 */
        return real_ncclCommRegister(comm, buff, size, handle);
    return real_ncclCommRegister(wait_real(v), buff, size, handle);
}

ncclResult_t ncclCommDeregister(const ncclComm_t comm, void *handle) {
    vcomm_t *v = vcomm_lookup(comm);
    LOAD_SYM(ncclCommDeregister);
    if (real_ncclCommDeregister == NULL)
        return ncclInvalidUsage;
    if (v == NULL) /* R2 */
        return real_ncclCommDeregister(comm, handle);
    return real_ncclCommDeregister(wait_real(v), handle);
}

ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar, ncclDataType_t datatype,
                                      ncclScalarResidence_t residence, ncclComm_t comm) {
    vcomm_t *v = vcomm_lookup(comm);
    LOAD_SYM(ncclRedOpCreatePreMulSum);
    if (real_ncclRedOpCreatePreMulSum == NULL)
        return ncclInvalidUsage;
    if (v == NULL) /* R2 */
        return real_ncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm);
    /* NOTE: the resulting ncclRedOp_t is tied to the real communicator and goes
     * stale when that comm is rebuilt. Custom reduction ops are not yet
     * virtualized. */
    return real_ncclRedOpCreatePreMulSum(op, scalar, datatype, residence, wait_real(v));
}

ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
    vcomm_t *v = vcomm_lookup(comm);
    LOAD_SYM(ncclRedOpDestroy);
    if (real_ncclRedOpDestroy == NULL)
        return ncclInvalidUsage;
    if (v == NULL) /* R2 */
        return real_ncclRedOpDestroy(op, comm);
    if (v->real == NULL)
        return ncclSuccess; /* the op died with the comm */
    return real_ncclRedOpDestroy(op, v->real);
}

const char *ncclGetLastError(ncclComm_t comm) {
    vcomm_t *v = vcomm_lookup(comm);
    LOAD_SYM(ncclGetLastError);
    if (real_ncclGetLastError == NULL)
        return "";
    if (v == NULL) /* R2 */
        return real_ncclGetLastError(comm);
    return real_ncclGetLastError(v->real); /* NULL is valid here: "no comm" */
}

#if NCCL_HAS_2_30
ncclResult_t ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId *uniqueId) {
    vcomm_t *v = vcomm_lookup(comm);
    LOAD_SYM(ncclCommGetUniqueId);
    if (real_ncclCommGetUniqueId == NULL)
        return ncclInvalidUsage;
    if (v == NULL) /* R2 */
        return real_ncclCommGetUniqueId(comm, uniqueId);
    return real_ncclCommGetUniqueId(wait_real(v), uniqueId);
}

ncclResult_t ncclCommMemStats(ncclComm_t comm, ncclCommMemStat_t stat, uint64_t *value) {
    vcomm_t *v = vcomm_lookup(comm);
    LOAD_SYM(ncclCommMemStats);
    if (real_ncclCommMemStats == NULL)
        return ncclInvalidUsage;
    if (v == NULL) /* R2 */
        return real_ncclCommMemStats(comm, stat, value);
    return real_ncclCommMemStats(wait_real(v), stat, value);
}
#endif

/* ── collectives ──────────────────────────────────────────────────── */

/*
 * Each of these counts one step on its OWN communicator's sequence, which is the
 * whole point of the change: a program doing one AllReduce on comm_world and two
 * on a sub-communicator per iteration cannot be described by a single counter.
 *
 * Inside a group region the enclosing ncclGroupStart already holds the gate, the
 * activity count and the wrapper lock, so none is taken again -- but the step
 * still counts, hence the seq++ on both paths. Advancing only on the ungated
 * path pinned seq at 0 for every batched collective, leaving
 * sequence_consensus() to settle on a cut that described nothing.
 */

#define COLLECTIVE_PROLOGUE(sym, forward)                                                          \
    vcomm_t *v = vcomm_lookup(comm);                                                               \
    ncclComm_t real;                                                                               \
    ncclResult_t ret;                                                                              \
    REQUIRE_SYM(sym, ncclInternalError);                                                           \
    if (v == NULL) /* R2 */                                                                        \
        return forward;                                                                            \
    add_stream(stream);                                                                            \
    if (in_group_region()) {                                                                       \
        nccl_seq_lock();                                                                           \
        v->seq++;                                                                                  \
        nccl_seq_unlock();                                                                         \
        real = wait_real(v);                                                                       \
    } else {                                                                                       \
        real = enter_nccl(v, 1); /* R4 */                                                          \
    }

#define COLLECTIVE_EPILOGUE()                                                                      \
    if (!in_group_region())                                                                         \
        leave_nccl();                                                                              \
    return ret;

ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                           cudaStream_t stream) {
    COLLECTIVE_PROLOGUE(ncclAllReduce, real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op,
                                                          comm, stream))
    ret = real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, real, stream);
    COLLECTIVE_EPILOGUE()
}

ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, int root, ncclComm_t comm,
                           cudaStream_t stream) {
    COLLECTIVE_PROLOGUE(ncclBroadcast, real_ncclBroadcast(sendbuff, recvbuff, count, datatype, root,
                                                          comm, stream))
    ret = real_ncclBroadcast(sendbuff, recvbuff, count, datatype, root, real, stream);
    COLLECTIVE_EPILOGUE()
}

/* The legacy spelling, and the most commonly called entry point left unwrapped
 * before this change. */
ncclResult_t ncclBcast(void *buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm,
                       cudaStream_t stream) {
    COLLECTIVE_PROLOGUE(ncclBcast, real_ncclBcast(buff, count, datatype, root, comm, stream))
    ret = real_ncclBcast(buff, count, datatype, root, real, stream);
    COLLECTIVE_EPILOGUE()
}

ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count, ncclDataType_t datatype,
                        ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
    COLLECTIVE_PROLOGUE(ncclReduce, real_ncclReduce(sendbuff, recvbuff, count, datatype, op, root,
                                                    comm, stream))
    ret = real_ncclReduce(sendbuff, recvbuff, count, datatype, op, root, real, stream);
    COLLECTIVE_EPILOGUE()
}

ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff, size_t sendcount,
                           ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
    COLLECTIVE_PROLOGUE(ncclAllGather,
                        real_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream))
    ret = real_ncclAllGather(sendbuff, recvbuff, sendcount, datatype, real, stream);
    COLLECTIVE_EPILOGUE()
}

ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                               ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                               cudaStream_t stream) {
    COLLECTIVE_PROLOGUE(ncclReduceScatter, real_ncclReduceScatter(sendbuff, recvbuff, recvcount,
                                                                  datatype, op, comm, stream))
    ret = real_ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, real, stream);
    COLLECTIVE_EPILOGUE()
}

ncclResult_t ncclAlltoAll(const void *sendbuff, void *recvbuff, size_t count,
                          ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
    COLLECTIVE_PROLOGUE(ncclAlltoAll,
                        real_ncclAlltoAll(sendbuff, recvbuff, count, datatype, comm, stream))
    if (real_ncclAlltoAll == NULL) {
        if (!in_group_region())
            leave_nccl();
        REFUSE_UNSUPPORTED("ncclAlltoAll");
    }
    ret = real_ncclAlltoAll(sendbuff, recvbuff, count, datatype, real, stream);
    COLLECTIVE_EPILOGUE()
}

ncclResult_t ncclGather(const void *sendbuff, void *recvbuff, size_t count,
                        ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) {
    COLLECTIVE_PROLOGUE(ncclGather, real_ncclGather(sendbuff, recvbuff, count, datatype, root, comm,
                                                    stream))
    if (real_ncclGather == NULL) {
        if (!in_group_region())
            leave_nccl();
        REFUSE_UNSUPPORTED("ncclGather");
    }
    ret = real_ncclGather(sendbuff, recvbuff, count, datatype, root, real, stream);
    COLLECTIVE_EPILOGUE()
}

/* This NCCL build does not export ncclScatter; the wrapper detects the missing
 * symbol so callers get a diagnostic rather than a crash. */
ncclResult_t ncclScatter(const void *sendbuff, void *recvbuff, size_t count,
                         ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream) {
    COLLECTIVE_PROLOGUE(ncclScatter, real_ncclScatter(sendbuff, recvbuff, count, datatype, root,
                                                      comm, stream))
    if (real_ncclScatter == NULL) {
        if (!in_group_region())
            leave_nccl();
        REFUSE_UNSUPPORTED("ncclScatter");
    }
    ret = real_ncclScatter(sendbuff, recvbuff, count, datatype, root, real, stream);
    COLLECTIVE_EPILOGUE()
}

/* ── point to point ───────────────────────────────────────────────── */

ncclResult_t ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype, int peer,
                      ncclComm_t comm, cudaStream_t stream) {
    vcomm_t *v = vcomm_lookup(comm);
    ncclComm_t real;
    ncclResult_t ret;
    size_t bytes;

    REQUIRE_SYM(ncclSend, ncclInternalError);
    if (v == NULL) /* R2 */
        return real_ncclSend(sendbuff, count, datatype, peer, comm, stream);

    add_stream(stream);
    bytes = count * get_nccl_type_size(datatype);

    if (in_group_region()) {
        real = wait_real(v);
        if (peer_ok(v, peer, "ncclSend"))
            v->sent_to[peer] += (int64_t)bytes;
        return real_ncclSend(sendbuff, count, datatype, peer, real, stream);
    }

    real = enter_nccl(v, 0);
    if (peer_ok(v, peer, "ncclSend"))
        v->sent_to[peer] += (int64_t)bytes;
    ret = real_ncclSend(sendbuff, count, datatype, peer, real, stream);
    leave_nccl();
    return ret;
}

ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype, int peer,
                      ncclComm_t comm, cudaStream_t stream) {
    vcomm_t *v = vcomm_lookup(comm);
    ncclComm_t real;
    ncclResult_t ret;
    size_t bytes;

    REQUIRE_SYM(ncclRecv, ncclInternalError);
    if (v == NULL) /* R2 */
        return real_ncclRecv(recvbuff, count, datatype, peer, comm, stream);

    bytes = count * get_nccl_type_size(datatype);

    /*
     * Replay a message the drain consumed from NCCL before a checkpoint. Nothing
     * will deliver it again, so this is the only path that can satisfy the recv.
     * Matched on (communicator identity, source rank) -- matching on the rank
     * alone would deliver a message drained on one comm to a recv on another.
     */
    if (__atomic_load_n(&g_drain_log_count, __ATOMIC_ACQUIRE) > 0) {
        drain_msg_t *m;
        DMTCP_PLUGIN_DISABLE_CKPT();
        m = drain_log_find(v->ident, peer);
        if (m != NULL) {
            if (m->bytes != bytes) {
                /* Truncating here would silently short-read; the byte-total
                 * accounting has already been violated if this happens. */
                fprintf(stderr,
                        "[NCCL PLUGIN] drained message from peer %d is %zu bytes but the recv"
                        " wants %zu; refusing to truncate\n",
                        peer, m->bytes, bytes);
                fflush(stderr);
                DMTCP_PLUGIN_ENABLE_CKPT();
                return ncclInvalidArgument;
            }
            /* Async on the caller's stream, so the replay keeps the ordering
             * semantics of the ncclRecv it stands in for. */
            if (cudaMemcpyAsync(recvbuff, m->data, bytes, cudaMemcpyHostToDevice, stream) !=
                cudaSuccess) {
                DMTCP_PLUGIN_ENABLE_CKPT();
                return ncclUnhandledCudaError;
            }
            drain_log_release(m);
            DMTCP_PLUGIN_ENABLE_CKPT();
            return ncclSuccess;
        }
        DMTCP_PLUGIN_ENABLE_CKPT();
    }

    add_stream(stream);

    if (in_group_region()) {
        real = wait_real(v);
        if (peer_ok(v, peer, "ncclRecv"))
            v->recvd_from[peer] += (int64_t)bytes;
        return real_ncclRecv(recvbuff, count, datatype, peer, real, stream);
    }

    real = enter_nccl(v, 0);
    if (peer_ok(v, peer, "ncclRecv"))
        v->recvd_from[peer] += (int64_t)bytes;
    ret = real_ncclRecv(recvbuff, count, datatype, peer, real, stream);
    leave_nccl();
    return ret;
}

/* ── group regions ────────────────────────────────────────────────── */

/*
 * A group region batches many ops, possibly across several communicators, and
 * peers block until it closes. So the gate is taken once at depth 0 and held for
 * the whole region: parking a thread inside an open batch would deadlock against
 * peers waiting for ncclGroupEnd. The activity count must cover the region too,
 * or the checkpoint would snapshot p2p counters mid-batch and destroy a
 * communicator from under it.
 *
 * Holding DMTCP_PLUGIN_DISABLE_CKPT across the region is safe: it is a
 * reentrant per-thread read lock, so the per-op toggles inside nest correctly.
 */
ncclResult_t ncclGroupStart(void) {
    ncclResult_t ret;
    int took_gate = 0;

    REQUIRE_SYM(ncclGroupStart, ncclInternalError);
    if (t_group_depth == 0) {
        for (;;) {
            int passed = 0;
            DMTCP_PLUGIN_DISABLE_CKPT();
            pthread_mutex_lock(&g_seq_lock);
            if (!nccl_gate_closed_locked()) {
                __atomic_add_fetch(&g_nccl_activity, 1, __ATOMIC_SEQ_CST);
                passed = 1;
            }
            pthread_mutex_unlock(&g_seq_lock);
            if (passed)
                break;
            DMTCP_PLUGIN_ENABLE_CKPT();
            usleep(GATE_POLL_US);
        }
        took_gate = 1;
    }

    ret = real_ncclGroupStart();
    if (ret != ncclSuccess) {
        /* No region was opened, so hold nothing: a raised activity count with
         * no thread left to drop it stalls every later checkpoint. */
        if (took_gate) {
            __atomic_sub_fetch(&g_nccl_activity, 1, __ATOMIC_SEQ_CST);
            DMTCP_PLUGIN_ENABLE_CKPT();
        }
        return ret;
    }
    t_group_depth++;
    return ret;
}

/* Shared by every way a region can close. */
static void group_region_close(ncclResult_t ret) {
    finish_pending_creates(init_ok(ret));
    __atomic_sub_fetch(&g_nccl_activity, 1, __ATOMIC_SEQ_CST);
    DMTCP_PLUGIN_ENABLE_CKPT();
}

ncclResult_t ncclGroupEnd(void) {
    ncclResult_t ret;
    REQUIRE_SYM(ncclGroupEnd, ncclInternalError);
    ret = real_ncclGroupEnd();
    if (t_group_depth > 0 && --t_group_depth == 0)
        group_region_close(ret);
    return ret;
}

/*
 * Closes NCCL's group just as ncclGroupEnd does, so it must unwind identically.
 * Left unwrapped, the region stays open on this thread forever: quiesce_nccl()
 * spins, and every later collective takes the in-region path and skips the gate.
 */
#if NCCL_HAS_SIM_INFO
ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t *simInfo) {
    ncclResult_t ret;
    REQUIRE_SYM(ncclGroupSimulateEnd, ncclInternalError);
    ret = real_ncclGroupSimulateEnd(simInfo);
    if (t_group_depth > 0 && --t_group_depth == 0)
        group_region_close(ret);
    return ret;
}
#endif

/* ── synthesized collectives ──────────────────────────────────────── */

/*
 * Built out of the wrapped ncclSend/ncclRecv, so they take the caller's virtual
 * handle and let those wrappers do the translation and accounting. The caller
 * owns ncclGroupStart/ncclGroupEnd.
 */
ncclResult_t ncclGather_p2p(const void *sendbuff, void *recvbuff, size_t sendcount,
                            ncclDataType_t datatype, int root, ncclComm_t comm,
                            cudaStream_t stream) {
    vcomm_t *v = vcomm_lookup(comm);
    size_t elem_bytes = get_nccl_type_size(datatype);
    ncclResult_t ret;
    int rank, nranks, r;

    if (v == NULL)
        return ncclInvalidArgument;
    rank = v->rank;
    nranks = v->nranks;

    if (elem_bytes == 0 || root < 0 || root >= nranks)
        return ncclInvalidArgument;
    if (rank == root && recvbuff == NULL)
        return ncclInvalidArgument;

    if (rank == root) {
        /* Root posts a recv slot for every rank, including itself. */
        for (r = 0; r < nranks; r++) {
            char *slot = (char *)recvbuff + (size_t)r * sendcount * elem_bytes;
            ret = ncclRecv(slot, sendcount, datatype, r, comm, stream);
            if (ret != ncclSuccess)
                return ret;
        }
    }
    return ncclSend(sendbuff, sendcount, datatype, root, comm, stream);
}

ncclResult_t ncclScatter_p2p(const void *sendbuff, void *recvbuff, size_t sendcount,
                             ncclDataType_t datatype, int root, ncclComm_t comm,
                             cudaStream_t stream) {
    vcomm_t *v = vcomm_lookup(comm);
    size_t elem_bytes = get_nccl_type_size(datatype);
    ncclResult_t ret;
    int rank, nranks, r;

    if (v == NULL)
        return ncclInvalidArgument;
    rank = v->rank;
    nranks = v->nranks;

    if (elem_bytes == 0 || root < 0 || root >= nranks)
        return ncclInvalidArgument;
    if (recvbuff == NULL)
        return ncclInvalidArgument;
    if (rank == root && sendbuff == NULL)
        return ncclInvalidArgument;

    ret = ncclRecv(recvbuff, sendcount, datatype, root, comm, stream);
    if (ret != ncclSuccess)
        return ret;

    if (rank == root) {
        for (r = 0; r < nranks; r++) {
            const char *slot = (const char *)sendbuff + (size_t)r * sendcount * elem_bytes;
            ret = ncclSend(slot, sendcount, datatype, r, comm, stream);
            if (ret != ncclSuccess)
                return ret;
        }
    }
    return ncclSuccess;
}

/* ── refused: cannot be reconstructed yet ─────────────────────────── */

/*
 * Wrapped purely so a virtual handle never reaches real NCCL. Each of these
 * either produces a communicator the rebuild path does not yet derive an agreed
 * identity for, or changes a communicator's membership in a way the registry
 * does not model.
 *
 * Crucially they still obey R2 and refuse ONLY for a handle of ours. NCCL calls
 * some of these public symbols internally -- commReclaim() calls
 * ncclCommWindowDeregister() on its way through ncclCommDestroy() -- so a
 * wrapper that refuses unconditionally hijacks NCCL's own cleanup and makes
 * every destroy fail.
 */

#if NCCL_HAS_2_30
/* Every one of these names its communicator parameter `comm`. */
#define FORWARD_OR_REFUSE(fn, sym, call)                                                           \
    do {                                                                                           \
        vcomm_t *_v = vcomm_lookup(comm);                                                          \
        LOAD_SYM(sym);                                                                             \
        if (_v == NULL) { /* R2: not ours, e.g. an internal NCCL call */                            \
            if (real_##sym == NULL)                                                                \
                return ncclInvalidUsage;                                                           \
            return call;                                                                           \
        }                                                                                          \
        REFUSE_UNSUPPORTED(fn);                                                                    \
    } while (0)

ncclResult_t ncclCommShrink(ncclComm_t comm, int *excludeRanksList, int excludeRanksCount,
                            ncclComm_t *newcomm, ncclConfig_t *config, int shrinkFlags) {
    FORWARD_OR_REFUSE("ncclCommShrink", ncclCommShrink,
                      real_ncclCommShrink(comm, excludeRanksList, excludeRanksCount, newcomm,
                                          config, shrinkFlags));
}

ncclResult_t ncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId *uniqueId, int rank,
                          ncclComm_t *newcomm, ncclConfig_t *config) {
    FORWARD_OR_REFUSE("ncclCommGrow", ncclCommGrow,
                      real_ncclCommGrow(comm, nRanks, uniqueId, rank, newcomm, config));
}

ncclResult_t ncclCommRevoke(ncclComm_t comm, int revokeFlags) {
    FORWARD_OR_REFUSE("ncclCommRevoke", ncclCommRevoke, real_ncclCommRevoke(comm, revokeFlags));
}

ncclResult_t ncclCommSuspend(ncclComm_t comm, int flags) {
    FORWARD_OR_REFUSE("ncclCommSuspend", ncclCommSuspend, real_ncclCommSuspend(comm, flags));
}

ncclResult_t ncclCommResume(ncclComm_t comm) {
    FORWARD_OR_REFUSE("ncclCommResume", ncclCommResume, real_ncclCommResume(comm));
}

ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void *buff, size_t size, ncclWindow_t *win,
                                    int winFlags) {
    FORWARD_OR_REFUSE("ncclCommWindowRegister", ncclCommWindowRegister,
                      real_ncclCommWindowRegister(comm, buff, size, win, winFlags));
}

ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win) {
    FORWARD_OR_REFUSE("ncclCommWindowDeregister", ncclCommWindowDeregister,
                      real_ncclCommWindowDeregister(comm, win));
}

ncclResult_t ncclWinGetUserPtr(ncclComm_t comm, ncclWindow_t win, void **outUserPtr) {
    FORWARD_OR_REFUSE("ncclWinGetUserPtr", ncclWinGetUserPtr,
                      real_ncclWinGetUserPtr(comm, win, outUserPtr));
}

ncclResult_t ncclSignal(int peer, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm,
                        cudaStream_t stream) {
    FORWARD_OR_REFUSE("ncclSignal", ncclSignal,
                      real_ncclSignal(peer, sigIdx, ctx, flags, comm, stream));
}

ncclResult_t ncclWaitSignal(int nDesc, ncclWaitSignalDesc_t *signalDescs, ncclComm_t comm,
                            cudaStream_t stream) {
    FORWARD_OR_REFUSE("ncclWaitSignal", ncclWaitSignal,
                      real_ncclWaitSignal(nDesc, signalDescs, comm, stream));
}

ncclResult_t ncclPutSignal(const void *localbuff, size_t count, ncclDataType_t datatype, int peer,
                           ncclWindow_t peerWin, size_t peerWinOffset, int sigIdx, int ctx,
                           unsigned int flags, ncclComm_t comm, cudaStream_t stream) {
    FORWARD_OR_REFUSE("ncclPutSignal", ncclPutSignal,
                      real_ncclPutSignal(localbuff, count, datatype, peer, peerWin, peerWinOffset,
                                         sigIdx, ctx, flags, comm, stream));
}

/* No input communicator to inspect, so there is nothing to forward: this one
 * only ever produces a comm the rebuild path cannot yet name. */
ncclResult_t ncclCommInitRankScalable(ncclComm_t *newcomm, int nranks, int myrank, int nId,
                                      ncclUniqueId *commIds, ncclConfig_t *config) {
    (void)newcomm;
    (void)nranks;
    (void)myrank;
    (void)nId;
    (void)commIds;
    (void)config;
    REFUSE_UNSUPPORTED("ncclCommInitRankScalable");
}
#endif /* NCCL_HAS_2_30 */

/* ── restart ──────────────────────────────────────────────────────── */

/*
 * Resolve every real symbol up front.
 *
 * Called on the checkpoint thread right after clear_wrapper_pointers() at
 * restart. Doing it here rather than lazily matters: the alternative is that the
 * first dlsym() after a restart happens from whichever thread first calls into
 * NCCL -- for the rebuild that is a freshly spawned helper thread, moments after
 * the address space was reconstructed -- and resolving there faults. Resolving
 * on this thread means every later call finds a cached pointer and dlsym is
 * never reached from that context.
 *
 * A missing symbol is not fatal here: some of these do not exist in older NCCL,
 * and REQUIRE_SYM reports it at the actual call site if anything needs one.
 */
void resolve_wrapper_pointers(void) {
    LOAD_SYM(ncclSend);
    LOAD_SYM(ncclRecv);
    LOAD_SYM(ncclAllReduce);
    LOAD_SYM(ncclBroadcast);
    LOAD_SYM(ncclBcast);
    LOAD_SYM(ncclReduce);
    LOAD_SYM(ncclAllGather);
    LOAD_SYM(ncclReduceScatter);
    LOAD_SYM(ncclGather);
    LOAD_SYM(ncclScatter);
    LOAD_SYM(ncclAlltoAll);
    LOAD_SYM(ncclCommInitRank);
    LOAD_SYM(ncclCommInitRankConfig);
    LOAD_SYM(ncclCommInitAll);
    LOAD_SYM(ncclCommSplit);
    LOAD_SYM(ncclCommDestroy);
    LOAD_SYM(ncclCommFinalize);
    LOAD_SYM(ncclCommAbort);
    LOAD_SYM(ncclCommCount);
    LOAD_SYM(ncclCommUserRank);
    LOAD_SYM(ncclCommCuDevice);
    LOAD_SYM(ncclCommGetAsyncError);
    LOAD_SYM(ncclCommRegister);
    LOAD_SYM(ncclCommDeregister);
    LOAD_SYM(ncclRedOpCreatePreMulSum);
    LOAD_SYM(ncclRedOpDestroy);
    LOAD_SYM(ncclGroupStart);
    LOAD_SYM(ncclGroupEnd);
#if NCCL_HAS_SIM_INFO
    LOAD_SYM(ncclGroupSimulateEnd);
#endif
    LOAD_SYM(ncclGetLastError);
#if NCCL_HAS_2_30
    LOAD_SYM(ncclCommGetUniqueId);
    LOAD_SYM(ncclCommMemStats);
    LOAD_SYM(ncclCommShrink);
    LOAD_SYM(ncclCommGrow);
    LOAD_SYM(ncclCommRevoke);
    LOAD_SYM(ncclCommSuspend);
    LOAD_SYM(ncclCommResume);
    LOAD_SYM(ncclCommWindowRegister);
    LOAD_SYM(ncclCommWindowDeregister);
    LOAD_SYM(ncclWinGetUserPtr);
    LOAD_SYM(ncclSignal);
    LOAD_SYM(ncclWaitSignal);
    LOAD_SYM(ncclPutSignal);
#endif
}

void clear_wrapper_pointers(void) {
    /* At restart, the lower half (and its NCCL) is a new one. The real_*
     * pointers are the upper half's thunks, which stay valid: only the
     * thunks' lower-half functions are forgotten. Nulling the pointers would
     * race with application threads, which DMTCP releases before the rebuild
     * (DMTCP_EVENT_RUNNING_AFTER). They wait in the thunks, at the lower
     * half's gate, until the rebuild is done. */
    mana_nccl_reset_lh_fns();
}
