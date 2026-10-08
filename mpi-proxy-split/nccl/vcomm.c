/*
 * vcomm -- virtual communicator registry. See vcomm.h for the design.
 *
 * Deliberately free of DMTCP, CUDA and dlsym dependencies, and of any I/O or
 * abort(): a caller that hits VCOMM_ERR_FULL or VCOMM_ERR_DUPLICATE decides how
 * loudly to fail. That keeps this file linkable into the host-only unit test.
 */

#include "vcomm.h"
#include <pthread.h>
#include <string.h>

vcomm_t g_vcomms[MAX_VCOMMS];

/*
 * Guards slot allocation and release only -- never held across a blocking NCCL
 * call. DMTCP_PLUGIN_DISABLE_CKPT is a per-thread read lock, not mutual
 * exclusion, so it does not serialize two application threads creating
 * communicators concurrently (PyTorch does exactly that).
 *
 * Field updates on an already-allocated entry need no lock: the only fields
 * written by one thread and read by another are `real` (ckpt thread writes,
 * app threads read, declared volatile and published last) and seq/target,
 * which the sequence-consensus code guards with its own lock.
 */
static pthread_mutex_t g_alloc_lock = PTHREAD_MUTEX_INITIALIZER;

int vcomm_index(const vcomm_t *v) {
    return (int)(v - g_vcomms);
}

ncclComm_t vcomm_handle(const vcomm_t *v) {
    return VCOMM_HANDLE(vcomm_index(v));
}

vcomm_t *vcomm_find(const uint8_t ident[VCOMM_IDENT_LEN]) {
    int i;
    for (i = 0; i < MAX_VCOMMS; i++) {
        if (g_vcomms[i].origin != VCOMM_FREE &&
            memcmp(g_vcomms[i].ident, ident, VCOMM_IDENT_LEN) == 0)
            return &g_vcomms[i];
    }
    return NULL;
}

vcomm_status_t vcomm_new(vcomm_origin_t origin, const uint8_t ident[VCOMM_IDENT_LEN],
                         vcomm_t **out) {
    int i;
    vcomm_status_t st = VCOMM_ERR_FULL;

    *out = NULL;
    pthread_mutex_lock(&g_alloc_lock);

    if (vcomm_find(ident) != NULL) {
        st = VCOMM_ERR_DUPLICATE;
        goto done;
    }

    for (i = 0; i < MAX_VCOMMS; i++) {
        if (g_vcomms[i].origin == VCOMM_FREE) {
            vcomm_t *v = &g_vcomms[i];
            memset(v, 0, sizeof(*v));
            v->origin = origin;
            memcpy(v->ident, ident, VCOMM_IDENT_LEN);
            v->clique = -1;
            v->target = -1;
            v->real = NULL;
            *out = v;
            st = VCOMM_OK;
            goto done;
        }
    }

done:
    pthread_mutex_unlock(&g_alloc_lock);
    return st;
}

void vcomm_free(vcomm_t *v) {
    if (v == NULL)
        return;
    pthread_mutex_lock(&g_alloc_lock);
    memset(v, 0, sizeof(*v));
    v->origin = VCOMM_FREE;
    pthread_mutex_unlock(&g_alloc_lock);
}

int vcomm_live_count(void) {
    int i, n = 0;
    for (i = 0; i < MAX_VCOMMS; i++) {
        if (g_vcomms[i].origin != VCOMM_FREE)
            n++;
    }
    return n;
}

int vcomm_order(int order[MAX_VCOMMS]) {
    int i, j, n = 0;

    for (i = 0; i < MAX_VCOMMS; i++) {
        if (g_vcomms[i].origin != VCOMM_FREE)
            order[n++] = i;
    }

    /* Insertion sort on identity: n <= 16, and it keeps the comparison
     * explicit rather than hiding it behind a qsort callback. */
    for (i = 1; i < n; i++) {
        int cur = order[i];
        for (j = i - 1; j >= 0 && vcomm_ident_cmp(g_vcomms[order[j]].ident,
                                                  g_vcomms[cur].ident) > 0;
             j--) {
            order[j + 1] = order[j];
        }
        order[j + 1] = cur;
    }
    return n;
}

void vcomm_reset_seq(void) {
    int i;
    for (i = 0; i < MAX_VCOMMS; i++) {
        if (g_vcomms[i].origin != VCOMM_FREE) {
            g_vcomms[i].seq = 0;
            g_vcomms[i].target = -1;
        }
    }
}

/* ── identity ─────────────────────────────────────────────────────── */

int vcomm_ident_cmp(const uint8_t a[VCOMM_IDENT_LEN], const uint8_t b[VCOMM_IDENT_LEN]) {
    return memcmp(a, b, VCOMM_IDENT_LEN);
}

void vcomm_ident_hex(const uint8_t ident[VCOMM_IDENT_LEN], char out[VCOMM_IDENT_HEX]) {
    static const char hex[] = "0123456789abcdef";
    int i;
    for (i = 0; i < VCOMM_IDENT_LEN; i++) {
        out[i * 2] = hex[(ident[i] >> 4) & 0xf];
        out[i * 2 + 1] = hex[ident[i] & 0xf];
    }
    out[VCOMM_IDENT_LEN * 2] = '\0';
}

/*
 * MurmurHash3_x64_128 by Austin Appleby -- public domain, reproduced here so the
 * plugin picks up no dependency for it. Deterministic given identical input
 * bytes, which is the whole requirement: every member of a communicator must
 * derive bit-identical output with no communication.
 */

static inline uint64_t rotl64(uint64_t x, int8_t r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t getblock64(const uint8_t *p, int i) {
    /* Explicit little-endian load rather than a cast: keeps the digest
     * independent of alignment and of the host's word order. */
    return ((uint64_t)p[i * 8 + 0]) | ((uint64_t)p[i * 8 + 1] << 8) |
           ((uint64_t)p[i * 8 + 2] << 16) | ((uint64_t)p[i * 8 + 3] << 24) |
           ((uint64_t)p[i * 8 + 4] << 32) | ((uint64_t)p[i * 8 + 5] << 40) |
           ((uint64_t)p[i * 8 + 6] << 48) | ((uint64_t)p[i * 8 + 7] << 56);
}

static inline uint64_t fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

static void put64le(uint8_t *p, uint64_t v) {
    int i;
    for (i = 0; i < 8; i++)
        p[i] = (uint8_t)(v >> (i * 8));
}

void vcomm_digest128(const void *key, size_t len, uint8_t out[VCOMM_IDENT_LEN]) {
    const uint8_t *data = (const uint8_t *)key;
    const size_t nblocks = len / 16;
    const uint8_t *tail = data + nblocks * 16;

    const uint64_t c1 = 0x87c37b91114253d5ULL;
    const uint64_t c2 = 0x4cf5ad432745937fULL;

    /* Fixed seed: the digest must be reproducible across processes and runs. */
    uint64_t h1 = 0x6e63636c6d616e61ULL; /* "ncclmana" */
    uint64_t h2 = h1;

    uint64_t k1 = 0, k2 = 0;
    size_t i;

    for (i = 0; i < nblocks; i++) {
        uint64_t b1 = getblock64(data, (int)(i * 2 + 0));
        uint64_t b2 = getblock64(data, (int)(i * 2 + 1));

        b1 *= c1;
        b1 = rotl64(b1, 31);
        b1 *= c2;
        h1 ^= b1;
        h1 = rotl64(h1, 27);
        h1 += h2;
        h1 = h1 * 5 + 0x52dce729;

        b2 *= c2;
        b2 = rotl64(b2, 33);
        b2 *= c1;
        h2 ^= b2;
        h2 = rotl64(h2, 31);
        h2 += h1;
        h2 = h2 * 5 + 0x38495ab5;
    }

    switch (len & 15) {
    case 15: k2 ^= (uint64_t)tail[14] << 48; /* fallthrough */
    case 14: k2 ^= (uint64_t)tail[13] << 40; /* fallthrough */
    case 13: k2 ^= (uint64_t)tail[12] << 32; /* fallthrough */
    case 12: k2 ^= (uint64_t)tail[11] << 24; /* fallthrough */
    case 11: k2 ^= (uint64_t)tail[10] << 16; /* fallthrough */
    case 10: k2 ^= (uint64_t)tail[9] << 8;   /* fallthrough */
    case 9:
        k2 ^= (uint64_t)tail[8] << 0;
        k2 *= c2;
        k2 = rotl64(k2, 33);
        k2 *= c1;
        h2 ^= k2;
        /* fallthrough */
    case 8: k1 ^= (uint64_t)tail[7] << 56; /* fallthrough */
    case 7: k1 ^= (uint64_t)tail[6] << 48; /* fallthrough */
    case 6: k1 ^= (uint64_t)tail[5] << 40; /* fallthrough */
    case 5: k1 ^= (uint64_t)tail[4] << 32; /* fallthrough */
    case 4: k1 ^= (uint64_t)tail[3] << 24; /* fallthrough */
    case 3: k1 ^= (uint64_t)tail[2] << 16; /* fallthrough */
    case 2: k1 ^= (uint64_t)tail[1] << 8;  /* fallthrough */
    case 1:
        k1 ^= (uint64_t)tail[0] << 0;
        k1 *= c1;
        k1 = rotl64(k1, 31);
        k1 *= c2;
        h1 ^= k1;
        /* fallthrough */
    default: break;
    }

    h1 ^= (uint64_t)len;
    h2 ^= (uint64_t)len;
    h1 += h2;
    h2 += h1;
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 += h2;
    h2 += h1;

    put64le(out, h1);
    put64le(out + 8, h2);
}

void vcomm_ident_from_id(const ncclUniqueId *id, uint8_t out[VCOMM_IDENT_LEN]) {
    /* Digest all of it, padding included: the id is distributed as raw bytes
     * (cc_complex.c uses MPI_Bcast of the struct), so every member sees the
     * same padding too, and there is no "meaningful prefix" to guess at. */
    vcomm_digest128(id, sizeof(*id), out);
}

void vcomm_ident_from_split(const uint8_t parent[VCOMM_IDENT_LEN], int color, int split_index,
                            uint8_t out[VCOMM_IDENT_LEN]) {
    /* Build the pre-image byte-by-byte rather than memcpy-ing a struct, so no
     * padding or word-order difference can creep between ranks. */
    uint8_t buf[VCOMM_IDENT_LEN + 8];
    memcpy(buf, parent, VCOMM_IDENT_LEN);
    put64le(buf + VCOMM_IDENT_LEN, ((uint64_t)(uint32_t)color << 32) | (uint32_t)split_index);
    vcomm_digest128(buf, sizeof(buf), out);
}

void vcomm_ident_local(int clique_idx, int rank, uint8_t out[VCOMM_IDENT_LEN]) {
    static const char tag[] = "nccl-mana-local-clique";
    uint8_t buf[sizeof(tag) + 8];
    memcpy(buf, tag, sizeof(tag));
    put64le(buf + sizeof(tag), ((uint64_t)(uint32_t)clique_idx << 32) | (uint32_t)rank);
    vcomm_digest128(buf, sizeof(buf), out);
}

/* ── config ───────────────────────────────────────────────────────── */

static void store_name(char *dst, const char *src) {
    if (src == NULL) {
        dst[0] = '\0';
        return;
    }
    strncpy(dst, src, VCOMM_NAME_MAX - 1);
    dst[VCOMM_NAME_MAX - 1] = '\0';
}

void vcomm_store_config(vcomm_t *v, const ncclConfig_t *config) {
    if (config == NULL) {
        v->has_config = 0;
        return;
    }
    v->config = *config;
    store_name(v->net_name, config->netName);
    store_name(v->comm_name, config->commName);
    /* Do not leave the app's pointers in the copy -- they are meaningless by
     * rebuild time. vcomm_config() repoints them at our own buffers. */
    v->config.netName = NULL;
    v->config.commName = NULL;
    v->has_config = 1;
}

ncclConfig_t *vcomm_config(vcomm_t *v) {
    if (!v->has_config)
        return NULL;
    v->config.netName = v->net_name[0] ? v->net_name : NULL;
    v->config.commName = v->comm_name[0] ? v->comm_name : NULL;
    return &v->config;
}
