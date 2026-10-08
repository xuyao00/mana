/*
 * The checkpoint side of MANA's NCCL support (from the nccl-mana-ckpt
 * prototype, src/libncclplugin.c).  MANA's plugin (mpi_plugin.cpp) calls these
 * from its event hook; see nccl_ckpt.h for where.
 *
 * NCCL is in the lower half, so the NCCL calls made here go through the thunks
 * like the application's (nccl_wrappers.c, R5), but as MANA's own calls
 * (mana_nccl_internal_begin()): the lower half is closed to the application
 * by then.  The CUDA calls are the application's CUDA (nccl_cuda.h).
 */
#include "dmtcp.h"
#include "dmtcp_kvdb_c.h"
#include "inflight.h"
#include "nccl_ckpt.h"
#include "nccl_cuda.h"
#include "nccl_lh.h"
#include "nccl_wrappers.h"
#include "vcomm.h"
#include <dlfcn.h>
#include <nccl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#define CUDA_CHECK(cmd)                                                                            \
    do {                                                                                           \
        cudaError_t e = cmd;                                                                       \
        if (e != cudaSuccess) {                                                                    \
            printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));           \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define MAX_STREAMS 64
#define POLL_US 200
#define MAX_CHASE_ROUNDS 32

static cudaStream_t g_active_streams[MAX_STREAMS];
static int g_num_streams = 0;

int g_nranks = 0;
int g_my_rank = -1;
volatile int ckpt_pending = 0;

/*
 * Checkpoint epoch, bumped at the top of every presuspend. Every peer receives
 * every DMT_DO_CHECKPOINT, so these counters stay in lockstep, and a restart
 * restores them all from images taken at the same epoch.
 *
 * This tag is not cosmetic. The coordinator wipes the KVDB only when the first
 * RESTARTING worker connects -- not on a plain checkpoint-and-resume -- while
 * presuspend destroys communicators on EVERY checkpoint. Without the epoch, the
 * accumulating MAX target keys would still hold the previous checkpoint's
 * high-water mark after seq resets to 0, and the second checkpoint of a run
 * would never converge.
 */
static unsigned g_epoch = 0;

static void pre_init_cuda(void);
void add_stream(cudaStream_t stream) {
    int i;
    for (i = 0; i < g_num_streams; i++) {
        if (g_active_streams[i] == stream)
            return;
    }
    if (g_num_streams < MAX_STREAMS) {
        g_active_streams[g_num_streams++] = stream;
    }
}

/*
 * Latch job-level identity from the first cross-process communicator. This is
 * only used for logging and for job-wide rank-0 roles; a communicator's own
 * rank and size live in its registry entry, because a sub-communicator has a
 * different rank space entirely.
 */
void init_rank_info(vcomm_t *v) {
    if (g_my_rank == -1 && v->origin == VCOMM_REMOTE) {
        g_my_rank = v->rank;
        g_nranks = v->nranks;
        printf("[NCCL Plugin] job rank %d of %d\n", g_my_rank, g_nranks);
        fflush(stdout);
    }
}

/*
 * The two NCCL entry points the plugin calls but does not interpose. Linking
 * them directly leaves undefined symbols the loader can only satisfy from the
 * global scope, which holds no NCCL when the application dlopened it
 * RTLD_LOCAL -- so the first rebuild dies on a symbol lookup error.
 */
static const char *nccl_err_str(ncclResult_t r) {
    const char *(*fn)(ncclResult_t);
    *(void **)&fn = nccl_real_dlsym("ncclGetErrorString");
    return (fn != NULL) ? fn(r) : "<ncclGetErrorString unavailable>";
}

static ncclResult_t nccl_get_unique_id(ncclUniqueId *id) {
    ncclResult_t (*fn)(ncclUniqueId *);
    *(void **)&fn = nccl_real_dlsym("ncclGetUniqueId");
    return (fn != NULL) ? fn(id) : ncclInternalError;
}

/* ── KVDB keys ────────────────────────────────────────────────────── */

/*
 * One flat "nccl-mana" namespace. Identity goes in as 32 fixed-width hex chars,
 * so keys are fixed-length and greppable in the coordinator's JSON dump.
 *
 * Every key is epoch-tagged, and every rank index is a rank IN THAT
 * COMMUNICATOR'S rank space -- never a job-global rank.
 */
#define KVDB_KEY_MAX 96

/*
 * Silent snprintf truncation would collapse two distinct keys into one, which is
 * the same cross-communicator aliasing this whole change exists to remove. So
 * truncation is fatal rather than ignored.
 */
#define KEYF(buf, ...)                                                                             \
    do {                                                                                           \
        int _n = snprintf((buf), KVDB_KEY_MAX, __VA_ARGS__);                                       \
        if (_n < 0 || _n >= KVDB_KEY_MAX) {                                                        \
            fprintf(stderr, "[NCCL PLUGIN] FATAL: kvdb key truncated at %s:%d\n", __FILE__,        \
                    __LINE__);                                                                     \
            fflush(stderr);                                                                        \
            abort();                                                                               \
        }                                                                                          \
    } while (0)

static void key_target(char *buf, const vcomm_t *v) {
    char hex[VCOMM_IDENT_HEX];
    vcomm_ident_hex(v->ident, hex);
    KEYF(buf, "C/%s/%u/t", hex, g_epoch);
}

static void key_p2p(char *buf, const vcomm_t *v, int src, int dst, char kind) {
    char hex[VCOMM_IDENT_HEX];
    vcomm_ident_hex(v->ident, hex);
    KEYF(buf, "C/%s/%u/p/%d/%d/%c", hex, g_epoch, src, dst, kind);
}

static void key_uid(char *buf, const vcomm_t *v) {
    char hex[VCOMM_IDENT_HEX];
    vcomm_ident_hex(v->ident, hex);
    KEYF(buf, "C/%s/%u/id", hex, g_epoch);
}

static void key_round(char *buf, int round) {
    KEYF(buf, "R/%u/%d/unsettled", g_epoch, round);
}

/* ── epoch agreement ──────────────────────────────────────────────── */

/*
 * Confirms every peer is on the same epoch. The one failure mode of a
 * process-local counter is a process that somehow missed a presuspend, which
 * would otherwise show up as an unexplainable non-convergence; this turns it
 * into a printed line.
 *
 * A shortfall is also the normal symptom of a DMTCP peer that does not load this
 * plugin (a launcher process, say). That configuration cannot work anyway -- the
 * barriers below need every peer -- so this reports rather than aborts.
 */
static void epoch_vote(void) {
    char key[KVDB_KEY_MAX];
    char barrier[64];
    int num_peers = 0, is_running = 0;
    int64_t votes;

    KEYF(key, "R/%u/votes", g_epoch);
    dmtcp_kvdb_incr(key, 1);

    snprintf(barrier, sizeof(barrier), "nccl-mana/%u/epoch-vote", g_epoch);
    dmtcp_global_barrier(barrier);

    dmtcp_get_coordinator_status(&num_peers, &is_running);
    votes = dmtcp_kvdb_get(key);
    if (votes != (int64_t)num_peers) {
        fprintf(stderr,
                "[NCCL PLUGIN] epoch %u: %ld of %d peers voted. Peers not running this\n"
                "              plugin cannot participate in its barriers.\n",
                g_epoch, (long)votes, num_peers);
        fflush(stderr);
    }
}

/* ── collective sequence consensus ────────────────────────────────── */

/*
 * Drive every communicator to a consistent cut.
 *
 * Each round: read peers' published targets, let this rank's application threads
 * catch up to them, then publish our own sequence and report whether we are
 * settled. Repeat until no rank anywhere was unsettled.
 *
 * Rounds are needed because catching up on one communicator can require issuing
 * collectives on another -- cc_complex does 1 AllReduce on comm_world, 2 on
 * comm_A and 1 on comm_B per iteration, so a rank told to advance comm_A may hit
 * comm_B first and overshoot its target. The overshoot is published in the next
 * round and the cut simply moves forward; targets are monotone, so this is the
 * fixpoint iteration of a monotone function and terminates.
 *
 * One MAX-accumulated key per communicator replaces the old per-rank key plus
 * enumeration, which also removes the question of whose rank space the enumerated
 * index lived in.
 */
static __attribute__((unused)) void sequence_consensus(void) {
    char key[KVDB_KEY_MAX];
    char barrier[80];
    int order[MAX_VCOMMS];
    int round;

    for (round = 0; round < MAX_CHASE_ROUNDS; round++) {
        int n, i, settled;

        /* 1. Adopt peers' targets. */
        nccl_seq_lock();
        n = vcomm_order(order);
        for (i = 0; i < n; i++) {
            vcomm_t *v = &g_vcomms[order[i]];
            int64_t published;
            if (v->origin == VCOMM_LOCAL_CLIQUE) {
                /* All ranks live in this process, so consensus is local. */
                v->target = v->seq;
                continue;
            }
            key_target(key, v);
            published = dmtcp_kvdb_get(key);
            /* 0 means "nobody published yet"; values carry a +1 bias so a
             * legitimate sequence of 0 is distinguishable from a missing key. */
            if (published > 0 && published - 1 > v->target)
                v->target = published - 1;
        }
        nccl_seq_unlock();

        /* 2. Wait for our own application threads to reach those targets. The
         *    gate lets them through precisely while some comm is behind. */
        for (;;) {
            int behind = 0;
            nccl_seq_lock();
            for (i = 0; i < n; i++) {
                vcomm_t *v = &g_vcomms[order[i]];
                if (v->target >= 0 && v->seq < v->target) {
                    behind = 1;
                    break;
                }
            }
            nccl_seq_unlock();
            if (!behind)
                break;
            usleep(POLL_US);
        }

        /* 3. Publish, and decide "settled" under the same lock -- otherwise a
         *    thread can slip one more collective through between the decision
         *    and the barrier, leaving an enqueued collective unmatched. */
        nccl_seq_lock();
        settled = 1;
        for (i = 0; i < n; i++) {
            vcomm_t *v = &g_vcomms[order[i]];
            if (v->origin == VCOMM_LOCAL_CLIQUE)
                continue;
            key_target(key, v);
            dmtcp_kvdb_max(key, v->seq + 1);
            if (v->target < 0 || v->seq != v->target)
                settled = 0;
        }
        nccl_seq_unlock();

        /* 4. Global vote. Reached unconditionally by every peer, including one
         *    holding no communicators at all, so the barrier always releases. */
        key_round(key, round);
        dmtcp_kvdb_incr(key, settled ? 0 : 1);
        snprintf(barrier, sizeof(barrier), "nccl-mana/%u/round/%d", g_epoch, round);
        dmtcp_global_barrier(barrier);

        if (dmtcp_kvdb_get(key) == 0) {
            printf("[NCCL Plugin] rank %d: sequence consensus converged in %d round(s)\n",
                   g_my_rank, round + 1);
            fflush(stdout);
            return;
        }
    }

    fprintf(stderr, "[NCCL PLUGIN] sequence consensus did not converge in %d rounds:\n",
            MAX_CHASE_ROUNDS);
    {
        int n = vcomm_order(order), i;
        for (i = 0; i < n; i++) {
            vcomm_t *v = &g_vcomms[order[i]];
            char hex[VCOMM_IDENT_HEX];
            vcomm_ident_hex(v->ident, hex);
            fprintf(stderr, "              comm %s rank %d/%d seq %ld target %ld\n", hex, v->rank,
                    v->nranks, (long)v->seq, (long)v->target);
        }
    }
    fflush(stderr);
    abort();
}

/* ── quiesce ──────────────────────────────────────────────────────── */

/*
 * Wait until no application thread is inside a NCCL call or an open group
 * region. The gate is already closed by this point, so no thread can enter, and
 * the checkpoint thread becomes the only submitter on every communicator --
 * which is what makes the drain's send/recv safe to post.
 */
static void quiesce_nccl(void) {
    while (nccl_activity() != 0)
        usleep(POLL_US);
}

/* ── p2p drain ────────────────────────────────────────────────────── */

static void publish_p2p_counters(void) {
    char key[KVDB_KEY_MAX];
    int order[MAX_VCOMMS];
    int n = vcomm_order(order);
    int i, peer;

    for (i = 0; i < n; i++) {
        vcomm_t *v = &g_vcomms[order[i]];
        if (v->origin == VCOMM_LOCAL_CLIQUE)
            continue;
        for (peer = 0; peer < v->nranks && peer < VCOMM_MAX_RANKS; peer++) {
            if (peer == v->rank)
                continue;
            /* "bytes src sent to dst" and "bytes dst received from src", both
             * biased by 1 so a missing key stays distinguishable from zero. */
            key_p2p(key, v, v->rank, peer, 's');
            dmtcp_kvdb_set(key, v->sent_to[peer] + 1);
            key_p2p(key, v, peer, v->rank, 'r');
            dmtcp_kvdb_set(key, v->recvd_from[peer] + 1);
        }
    }
}

/*
 * Drain one communicator.
 *
 * An imbalance is asymmetric by construction: owed_in > 0 means the peer's
 * application already enqueued a send we never received, and owed_out > 0 means
 * the peer already enqueued a recv we never satisfied. Exactly one side acts, and
 * its counterpart is an operation already sitting in NCCL rather than something
 * the peer's checkpoint thread has to do in step with us. So no per-communicator
 * rendezvous is needed here -- which is just as well, since dmtcp_global_barrier
 * cannot synchronize a subset of peers.
 */
static void drain_comm(vcomm_t *v) {
    char key[KVDB_KEY_MAX];
    ncclComm_t real = v->real;
    int peer;

    if (real == NULL)
        return;

    CUDA_CHECK(cudaSetDevice(v->cuda_dev));

    for (peer = 0; peer < v->nranks && peer < VCOMM_MAX_RANKS; peer++) {
        int64_t sent_by_peer, recvd_by_peer, owed_in, owed_out;

        if (peer == v->rank)
            continue;

        key_p2p(key, v, peer, v->rank, 's');
        sent_by_peer = dmtcp_kvdb_get(key);
        key_p2p(key, v, v->rank, peer, 'r');
        recvd_by_peer = dmtcp_kvdb_get(key);

        if (sent_by_peer == 0 || recvd_by_peer == 0) {
            /* Both were published before the barrier that precedes this read, so
             * a raw 0 is a protocol error. Treating it as a zero count would
             * fabricate an imbalance and drain a message that does not exist. */
            char hex[VCOMM_IDENT_HEX];
            vcomm_ident_hex(v->ident, hex);
            fprintf(stderr,
                    "[NCCL PLUGIN] comm %s rank %d: peer %d never published its p2p counters;"
                    " skipping drain for this pair\n",
                    hex, v->rank, peer);
            fflush(stderr);
            continue;
        }

        /* Undo the +1 publication bias. Our own side is read locally rather than
         * round-tripped through the coordinator. */
        owed_in = (sent_by_peer - 1) - v->recvd_from[peer];
        owed_out = (recvd_by_peer - 1) - v->sent_to[peer];

        if (owed_in > 0) {
            /* The peer enqueued this send; consume it now and stash the payload
             * so the application's eventual ncclRecv can be satisfied from the
             * host copy after restart. */
            void *dev_buf = NULL;
            void *host_buf = NULL;
            size_t bytes = (size_t)owed_in;

            printf("[NCCL Drain] comm rank %d: draining recv of %zu bytes from %d\n", v->rank,
                   bytes, peer);
            fflush(stdout);

            CUDA_CHECK(cudaMalloc(&dev_buf, bytes));
            call_real_ncclRecv(dev_buf, bytes, ncclInt8, peer, real, 0);
            CUDA_CHECK(cudaStreamSynchronize(0));

            host_buf = malloc(bytes);
            if (host_buf == NULL) {
                fprintf(stderr, "[NCCL PLUGIN] out of memory stashing a drained message\n");
                fflush(stderr);
                abort();
            }
            CUDA_CHECK(cudaMemcpy(host_buf, dev_buf, bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(dev_buf));

            if (!drain_log_push(v->ident, peer, host_buf, bytes)) {
                /* Dropping it would silently lose an application message. */
                fprintf(stderr,
                        "[NCCL PLUGIN] drain log full (MAX_DRAIN_MSGS=%d); cannot stash the"
                        " message from peer %d\n",
                        MAX_DRAIN_MSGS, peer);
                fflush(stderr);
                abort();
            }
            v->recvd_from[peer] += owed_in;
        }

        if (owed_out > 0) {
            /* The peer already enqueued a matching recv. It has to be satisfied
             * or the communicator cannot be destroyed.
             *
             * NOTE: the payload is zero-filled, so the peer's buffer receives
             * zeros rather than the data its application expected. That is a
             * pre-existing gap, not something introduced here: satisfying it
             * correctly needs the sender's real buffer, which by definition was
             * never handed to NCCL. */
            void *dev_buf = NULL;
            size_t bytes = (size_t)owed_out;

            printf("[NCCL Drain] comm rank %d: draining send of %zu bytes to %d\n", v->rank, bytes,
                   peer);
            fflush(stdout);

            CUDA_CHECK(cudaMalloc(&dev_buf, bytes));
            CUDA_CHECK(cudaMemset(dev_buf, 0, bytes));
            call_real_ncclSend(dev_buf, bytes, ncclInt8, peer, real, 0);
            CUDA_CHECK(cudaStreamSynchronize(0));
            CUDA_CHECK(cudaFree(dev_buf));
            v->sent_to[peer] += owed_out;
        }
    }
}

/* ── teardown ─────────────────────────────────────────────────────── */

/*
 * Destroy every communicator so the CUDA driver holds no CUDA IPC / P2P handles
 * (or GDR mappings from connection setup) when the CUDA plugin snapshots the
 * process.
 *
 * Reverse identity order, mirroring the rebuild. Uses the real symbol directly:
 * the interposed ncclCommDestroy also frees the registry slot, which here would
 * throw away the very metadata the rebuild needs.
 */
static void teardown_all_comms(void) {
    int order[MAX_VCOMMS];
    int n = vcomm_order(order);
    int k;

    for (k = n - 1; k >= 0; k--) {
        vcomm_t *v = &g_vcomms[order[k]];
        ncclComm_t real = v->real;
        ncclResult_t r;
        char hex[VCOMM_IDENT_HEX];

        if (real == NULL)
            continue;

        vcomm_ident_hex(v->ident, hex);
        printf("[NCCL Plugin] teardown: destroying comm %s (rank %d/%d)\n", hex, v->rank,
               v->nranks);
        fflush(stdout);

        /* Cleared first, so any application thread arriving at a wrapper blocks
         * in wait_real() instead of using a handle we are about to destroy. */
        v->real = NULL;
        r = call_real_ncclCommDestroy(real);
        if (r != ncclSuccess) {
            fprintf(stderr, "[NCCL PLUGIN] teardown: ncclCommDestroy failed: %s\n",
                    nccl_err_str(r));
            fflush(stderr);
        }
    }
}

/* ── rebuild ──────────────────────────────────────────────────────── */

/*
 * Rebuild a process-local clique with a single ncclCommInitAll.
 *
 * ncclCommInitRank "must be called by different threads/processes or use
 * ncclGroupStart/ncclGroupEnd", so reconstructing an N-rank clique as N
 * sequential InitRank calls from this one thread would hang. Being
 * process-local, it also needs no rendezvous at all.
 */
static void rebuild_clique(int clique) {
    ncclComm_t tmp[MAX_VCOMMS];
    int devlist[MAX_VCOMMS];
    vcomm_t *members[MAX_VCOMMS];
    int count = 0;
    int i;
    ncclResult_t r;

    for (i = 0; i < MAX_VCOMMS; i++) {
        vcomm_t *v = &g_vcomms[i];
        if (v->origin == VCOMM_LOCAL_CLIQUE && v->clique == clique)
            members[count++] = v;
    }
    if (count == 0)
        return;

    memset(devlist, 0, sizeof(devlist));
    for (i = 0; i < count; i++) {
        if (members[i]->rank < 0 || members[i]->rank >= count) {
            fprintf(stderr, "[NCCL PLUGIN] clique %d has an out-of-range rank %d\n", clique,
                    members[i]->rank);
            fflush(stderr);
            abort();
        }
        /* ncclCommInitAll returns comms in devlist order, so indexing by rank
         * reproduces the original rank-to-device mapping exactly. */
        devlist[members[i]->rank] = members[i]->cuda_dev;
    }

    r = call_real_ncclCommInitAll(tmp, count, devlist);
    if (r != ncclSuccess) {
        fprintf(stderr, "[NCCL PLUGIN] rebuild: ncclCommInitAll failed: %s\n",
                nccl_err_str(r));
        fflush(stderr);
        exit(1);
    }
    for (i = 0; i < count; i++)
        members[i]->real = tmp[members[i]->rank];
}

/*
 * Recreate every communicator after a checkpoint or restart.
 *
 * The registry survives as ordinary restored memory, so {identity, nranks, rank,
 * device, config} are all still here; only `real` went stale. Membership is
 * implicit in the ncclCommInitRank rendezvous, so nothing about how the
 * communicator was originally created needs replaying -- a comm made by
 * ncclCommSplit is rebuilt by exactly the same call as one made by
 * ncclCommInitRank.
 *
 * The old ncclUniqueId is useless: it encodes the bootstrap address of the
 * original root. Each comm's member 0 generates a fresh one and publishes it
 * under the comm's identity. Publishing is non-blocking, so ONE global barrier
 * covers every communicator at once, and no per-comm rendezvous is needed.
 */
static void rebuild_publish_ids(void) {
    char key[KVDB_KEY_MAX];
    int order[MAX_VCOMMS];
    int n, k;

    n = vcomm_order(order);

    /* Phase 1: publish a fresh id for each comm we are member 0 of. */
    for (k = 0; k < n; k++) {
        vcomm_t *v = &g_vcomms[order[k]];
        ncclUniqueId id;
        ncclResult_t r;

        if (v->origin != VCOMM_REMOTE || v->real != NULL || v->rank != 0)
            continue;

        r = nccl_get_unique_id(&id);
        if (r != ncclSuccess) {
            fprintf(stderr, "[NCCL PLUGIN] rebuild: ncclGetUniqueId failed: %s\n",
                    nccl_err_str(r));
            fflush(stderr);
            exit(1);
        }
        key_uid(key, v);
        if (!dmtcp_kvdb_set_blob(key, &id, sizeof(id))) {
            fprintf(stderr, "[NCCL PLUGIN] rebuild: could not publish a fresh ncclUniqueId\n");
            fflush(stderr);
            exit(1);
        }
    }
}

/*
 * The stored config is replayed verbatim, so a communicator the application
 * created with blocking = 0 is rebuilt non-blocking too and comes back
 * ncclInProgress: it exists but NCCL is still finishing it in the background.
 * v->real must not be published until that completes -- the store is what
 * releases application threads from wait_real(), and NCCL rejects a comm used
 * before its init returned ncclSuccess.
 *
 * The wait is unbounded, matching the blocking path: ncclCommInitRank is
 * collective, so a rebuild whose peers never arrive hangs either way.
 */
static ncclResult_t wait_comm_ready(ncclComm_t real) {
    for (;;) {
        ncclResult_t state = ncclSuccess;
        ncclResult_t r = call_real_ncclCommGetAsyncError(real, &state);
        if (r != ncclSuccess)
            return r;
        if (state != ncclInProgress)
            return state;
        usleep(POLL_US);
    }
}

/*
 * Fresh ids, fetched on the checkpoint thread.  dmtcp_kvdb_* is a coordinator
 * round-trip and is only legal there -- the same reason publication and the
 * barrier stay on that thread.  The worker only touches CUDA and NCCL.
 */
static ncclUniqueId g_rebuild_ids[MAX_VCOMMS];
static int g_rebuild_id_ok[MAX_VCOMMS];

static void rebuild_fetch_ids(void) {
    char key[KVDB_KEY_MAX];
    int order[MAX_VCOMMS];
    int n = vcomm_order(order);
    int k;

    for (k = 0; k < MAX_VCOMMS; k++)
        g_rebuild_id_ok[k] = 0;

    for (k = 0; k < n; k++) {
        int idx = order[k];
        vcomm_t *v = &g_vcomms[idx];

        if (v->real != NULL || v->origin != VCOMM_REMOTE)
            continue;
        key_uid(key, v);
        g_rebuild_id_ok[idx] =
            dmtcp_kvdb_get_blob(key, &g_rebuild_ids[idx], sizeof(g_rebuild_ids[idx]));
    }
}

static void rebuild_create_comms(void) {
    int order[MAX_VCOMMS];
    int n, k;

    n = vcomm_order(order);

    /*
     * Phase 2: rebuild in ascending identity order.
     *
     * ncclCommInitRank is collective and blocking, so ranks that share
     * communicators must agree on the order. Every member derives the same
     * identity, so each rank's list is a subsequence of one global total order,
     * and a rank can only ever be blocked on the minimal communicator whose
     * other members have therefore all reached it.
     */
    for (k = 0; k < n; k++) {
        vcomm_t *v = &g_vcomms[order[k]];
        ncclUniqueId id = g_rebuild_ids[order[k]];
        ncclComm_t real = NULL;
        ncclResult_t r;
        char hex[VCOMM_IDENT_HEX];

        if (v->real != NULL)
            continue; /* nothing was torn down, or a clique-mate did it */

        vcomm_ident_hex(v->ident, hex);

        if (v->origin == VCOMM_LOCAL_CLIQUE) {
            rebuild_clique(v->clique);
            continue;
        }

        /* NCCL binds a communicator to the CURRENT device, so this must be the
         * device the comm was originally created on -- not the job rank, which
         * is a different number whenever CUDA_VISIBLE_DEVICES is set per rank. */
        CUDA_CHECK(cudaSetDevice(v->cuda_dev));

        if (!g_rebuild_id_ok[order[k]]) {
            fprintf(stderr,
                    "[NCCL PLUGIN] rebuild: no fresh id published for comm %s"
                    " (rank %d/%d)\n",
                    hex, v->rank, v->nranks);
            fflush(stderr);
            exit(1);
        }

        printf("[NCCL Plugin] rebuild: comm %s rank %d/%d on device %d\n", hex, v->rank, v->nranks,
               v->cuda_dev);
        fflush(stdout);

        r = call_real_ncclCommInitRankConfig(&real, v->nranks, id, v->rank, vcomm_config(v));
        if (r == ncclInProgress && real != NULL)
            r = wait_comm_ready(real);
        if (r != ncclSuccess) {
            fprintf(stderr, "[NCCL PLUGIN] rebuild: ncclCommInitRank failed for %s: %s\n", hex,
                    nccl_err_str(r));
            fflush(stderr);
            exit(1);
        }

        /* Published last: this is what releases application threads parked in
         * wait_real(). */
        v->real = real;
    }
}

static void *rebuild_worker(void *arg) {
    (void)arg;
    /* Before any CUDA call: see mana_nccl_note_thread(). */
    mana_nccl_note_thread();
    mana_nccl_internal_begin();
    rebuild_create_comms();
    mana_nccl_internal_end();
    return NULL;
}

static void rebuild_all_comms(void) {
    char barrier[80];
    pthread_t worker;
    int rc;

    printf("[NCCL Plugin] rebuild: phase 1 cuda init\n");
    fflush(stdout);
    pre_init_cuda();

    printf("[NCCL Plugin] rebuild: phase 2 publish ids\n");
    fflush(stdout);
    rebuild_publish_ids();

    /* Unconditional on every peer: a process with no communicators still has to
     * reach this, or its peers block here forever. Barriers are only legal on
     * the checkpoint thread, which is why publication stays here. */
    printf("[NCCL Plugin] rebuild: phase 3 barrier\n");
    fflush(stdout);
    snprintf(barrier, sizeof(barrier), "nccl-mana/%u/ids-published", g_epoch);
    dmtcp_global_barrier(barrier);

    rebuild_fetch_ids();

    /* Nothing to create: no helper thread either (a thread's creation maps
     * its stack while the application's threads run). */
    if (vcomm_live_count() == 0)
        return;

    printf("[NCCL Plugin] rebuild: phase 4 worker thread\n");
    fflush(stdout);

    /*
     * Create the communicators on an ordinary thread rather than on this one.
     *
     * ncclCommInitRank crashes partway through its channel setup when called on
     * DMTCP's checkpoint thread after a restart, while the identical call from an
     * application thread succeeds -- verified by creating a communicator
     * post-restart with only the CUDA plugin loaded. So the checkpoint thread
     * keeps the parts that must run on it (KVDB publication and the barrier
     * above) and hands the NCCL work to a helper it joins.
     *
     * Joining matters: the rebuild has to finish before this event handler
     * returns, so that a checkpoint cannot begin against half-rebuilt state.
     */
    rc = pthread_create(&worker, NULL, rebuild_worker, NULL);
    if (rc != 0) {
        fprintf(stderr, "[NCCL PLUGIN] rebuild: pthread_create failed (%d)\n", rc);
        fflush(stderr);
        exit(1);
    }
    pthread_join(worker, NULL);
}

/* ── DMTCP events ─────────────────────────────────────────────────── */

static void resume(void) {
    ckpt_pending = 0;
    vcomm_reset_seq();
    /*
     * The p2p byte counters are deliberately NOT reset. They are cumulative, the
     * drain has just equalized them across ranks, and a reset would have to
     * happen identically everywhere to keep the differences meaningful. Leaving
     * them alone is symmetric by construction.
     *
     * The drain log is not reset either -- see inflight.h.
     */
    printf("[NCCL Plugin] rank %d: resumed after checkpoint\n", g_my_rank);
    fflush(stdout);
}

static void restart(void) {
    int i;
    ckpt_pending = 0;
    g_num_streams = 0;
    /* Drop the lower half's functions: the lower half is a new one. They are
     * resolved again in mana_nccl_rebuild(), on the checkpoint thread, before
     * the rebuild's helper thread calls NCCL. */
    clear_wrapper_pointers();
    /* Teardown already nulled these, but a checkpoint image taken before this
     * code existed, or an aborted teardown, could leave one set. */
    for (i = 0; i < MAX_VCOMMS; i++) {
        if (g_vcomms[i].origin != VCOMM_FREE)
            g_vcomms[i].real = NULL;
    }
    vcomm_reset_seq();
    printf("[NCCL Plugin] restarting from checkpoint with %d communicator(s)\n",
           vcomm_live_count());
    fflush(stdout);
}

static void pre_init_cuda(void) {
    int device_count = 0;
    cudaError_t err;

    /* The application's CUDA: in MANA, the CUDA plugin has restored it by
     * now (its RUNNING_AFTER runs before MANA's). */
    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        printf("[NCCL Plugin] pre_init_cuda: cudaGetDeviceCount failed: %s\n",
               cudaGetErrorString(err));
        fflush(stdout);
        return;
    }
    if (device_count > 0) {
        /* Do not disturb the application's current device: if a communicator is
         * registered, its own recorded device is authoritative and the rebuild
         * sets it per comm. */
        cudaFree(0);
    }
}

/* ── MANA's event hook (mpi_plugin.cpp) ───────────────────────────── */

/* Set when the communicators were torn down for a checkpoint; checkpointed
 * with it, so it is also set at restart. */
static int g_rebuild_pending = 0;

void mana_nccl_init(void) {
    drain_log_init();
}

void mana_nccl_ckpt_begin(void) {
    g_epoch++;
    ckpt_pending = 1;
    __sync_synchronize();

    /*
     * Every phase runs on every peer unconditionally, including a process
     * that has not created a communicator yet -- its loops are simply empty.
     */
    epoch_vote();
}

/*
 * One round of sequence_consensus(), as a round of MANA's MPI drain
 * (seq_num.cpp, g_drain_round_hook), which does the global vote and the
 * barrier. Unlike sequence_consensus(), it waits for this rank's threads only
 * a little: a thread may first have to get through an MPI collective whose
 * peers wait on another rank -- the next round sees the progress.
 */
#define ROUND_WAIT_US 20000

int mana_nccl_consensus_round(int round) {
    char key[KVDB_KEY_MAX];
    int order[MAX_VCOMMS];
    int n, i, settled, waited;

    (void)round;

    /* 1. Adopt peers' targets, published in earlier rounds. */
    nccl_seq_lock();
    n = vcomm_order(order);
    for (i = 0; i < n; i++) {
        vcomm_t *v = &g_vcomms[order[i]];
        int64_t published;
        if (v->origin == VCOMM_LOCAL_CLIQUE) {
            v->target = v->seq;
            continue;
        }
        key_target(key, v);
        published = dmtcp_kvdb_get(key);
        if (published > 0 && published - 1 > v->target)
            v->target = published - 1;
    }
    nccl_seq_unlock();

    /* 2. Let this rank's threads catch up, for a while. */
    for (waited = 0; waited < ROUND_WAIT_US; waited += POLL_US) {
        int behind = 0;
        nccl_seq_lock();
        for (i = 0; i < n; i++) {
            vcomm_t *v = &g_vcomms[order[i]];
            if (v->target >= 0 && v->seq < v->target) {
                behind = 1;
                break;
            }
        }
        nccl_seq_unlock();
        if (!behind)
            break;
        usleep(POLL_US);
    }

    /* 3. Publish, and decide "settled" under the same lock. */
    nccl_seq_lock();
    settled = 1;
    for (i = 0; i < n; i++) {
        vcomm_t *v = &g_vcomms[order[i]];
        if (v->origin == VCOMM_LOCAL_CLIQUE)
            continue;
        key_target(key, v);
        dmtcp_kvdb_max(key, v->seq + 1);
        if (v->target < 0 || v->seq != v->target)
            settled = 0;
    }
    nccl_seq_unlock();
    return settled;
}

/* For MANA's MPI gate (seq_num.cpp, g_other_settled): whether no NCCL
 * communicator of this rank is behind its target. */
int mana_nccl_settled(void) {
    int i, settled = 1;
    if (!ckpt_pending)
        return 1;
    nccl_seq_lock();
    for (i = 0; i < MAX_VCOMMS; i++) {
        vcomm_t *v = &g_vcomms[i];
        if (v->origin != VCOMM_FREE && v->target >= 0 && v->seq < v->target) {
            settled = 0;
            break;
        }
    }
    nccl_seq_unlock();
    return settled;
}

void mana_nccl_ckpt_quiesce(void) {
    /* From here the checkpoint thread is the only submitter. */
    quiesce_nccl();
}

void mana_nccl_ckpt_teardown(void) {
    struct timeval start, end;
    double elapsed;
    int order[MAX_VCOMMS];
    int n, k;
    char barrier[80];

    gettimeofday(&start, NULL);
    mana_nccl_internal_begin();

    if (vcomm_live_count() > 0)
        pre_init_cuda();

    /* Snapshot p2p state, then drain against it. */
    publish_p2p_counters();
    snprintf(barrier, sizeof(barrier), "nccl-mana/%u/p2p-published", g_epoch);
    dmtcp_global_barrier(barrier);

    n = vcomm_order(order);
    for (k = 0; k < n; k++)
        drain_comm(&g_vcomms[order[k]]);

    /* Only now can the streams complete: the drain has posted the counterparts
     * for every unmatched operation. */
    if (vcomm_live_count() > 0) {
        CUDA_CHECK(cudaStreamSynchronize(0));
        for (k = 0; k < g_num_streams; k++)
            cudaStreamSynchronize(g_active_streams[k]);
    }

    /* Nobody may destroy a communicator while a peer is still draining it. */
    snprintf(barrier, sizeof(barrier), "nccl-mana/%u/drained", g_epoch);
    dmtcp_global_barrier(barrier);

    teardown_all_comms();
    g_rebuild_pending = 1;

    mana_nccl_internal_end();
    gettimeofday(&end, NULL);
    elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("[NCCL PLUGIN] Checkpoint overhead (drain and teardown): %.6f seconds\n", elapsed);
    fflush(stdout);
}

void mana_nccl_resume(void) {
    resume();
}

void mana_nccl_restart(void) {
    restart();
}

void mana_nccl_rebuild(void) {
    if (!g_rebuild_pending)
        return; /* initial launch: nothing to rebuild */
    g_rebuild_pending = 0;
    mana_nccl_internal_begin();
    /* The lower half's functions, resolved here on the checkpoint thread (see
     * restart()). */
    resolve_wrapper_pointers();
    rebuild_all_comms();
    mana_nccl_internal_end();
}
