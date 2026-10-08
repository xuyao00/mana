#ifndef INFLIGHT_H
#define INFLIGHT_H

#include "vcomm.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Drained-message log.
 *
 * When a checkpoint finds an unmatched inbound message, the drain posts a real
 * ncclRecv for it, copies the payload to host memory, and parks it here. The
 * wrapped ncclRecv later finds it and replays the host copy instead of calling
 * into NCCL.
 *
 * This log is the one piece of state that deliberately SURVIVES restart -- the
 * message was consumed from NCCL before the checkpoint, so nothing will deliver
 * it again. That is exactly why entries are keyed by the communicator's
 * 128-bit identity rather than by a vcomm_t pointer or registry index: the real
 * handle is meaningless after restart and a slot index can be reassigned, but
 * the identity is stable.
 *
 * `source` is the sender's rank IN THAT COMMUNICATOR'S rank space, which is not
 * the job-global one. Keying on the rank alone (as the previous
 * drain_queues[MAX_RANKS] design did) means a message drained on one comm gets
 * replayed into the next ncclRecv from that rank on ANY comm.
 *
 * Drained messages are rare -- at most one per (comm, peer) per checkpoint, and
 * only on an imbalance -- so a flat array with a linear scan costs a few KB,
 * where a per-comm array of per-rank rings would cost hundreds.
 */

#define MAX_DRAIN_MSGS 64

typedef struct {
    uint8_t ident[VCOMM_IDENT_LEN]; /* communicator identity */
    uint64_t order;                 /* global push order: FIFO within (ident, source) */
    void *data;                     /* malloc'd host copy; DMTCP restores it for free */
    size_t bytes;
    int source; /* sender rank IN THAT COMM'S rank space */
    int in_use;
} drain_msg_t;

extern drain_msg_t g_drain_log[MAX_DRAIN_MSGS];

/*
 * Cheap hot-path guard: the wrapped ncclRecv checks this before scanning, so an
 * application that never drains pays one integer load per recv.
 */
extern volatile int g_drain_log_count;

void drain_log_init(void);

/*
 * Takes ownership of `data` (freed by drain_log_release). Returns 0 when the
 * log is full, which the caller must treat as fatal: silently dropping the
 * entry would lose an application message.
 */
int drain_log_push(const uint8_t ident[VCOMM_IDENT_LEN], int source, void *data, size_t bytes);

/* Oldest entry for (ident, source), or NULL. */
drain_msg_t *drain_log_find(const uint8_t ident[VCOMM_IDENT_LEN], int source);

void drain_log_release(drain_msg_t *m);

/* Frees every remaining entry. Not called on the restart path -- see above. */
void drain_log_reset(void);

#ifdef __cplusplus
}
#endif

#endif /* INFLIGHT_H */
