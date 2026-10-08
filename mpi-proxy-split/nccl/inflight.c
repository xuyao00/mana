#include "inflight.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

drain_msg_t g_drain_log[MAX_DRAIN_MSGS];
volatile int g_drain_log_count;

/*
 * Monotonic push counter, so two messages queued for the same (ident, source)
 * are replayed in the order NCCL would have delivered them. It is process-local
 * and only compared within this log, so wrapping is not a concern.
 */
static uint64_t g_next_order;

void drain_log_init(void) {
    memset(g_drain_log, 0, sizeof(g_drain_log));
    g_drain_log_count = 0;
    g_next_order = 0;
}

int drain_log_push(const uint8_t ident[VCOMM_IDENT_LEN], int source, void *data, size_t bytes) {
    int i;
    for (i = 0; i < MAX_DRAIN_MSGS; i++) {
        drain_msg_t *m = &g_drain_log[i];
        if (m->in_use)
            continue;
        memcpy(m->ident, ident, VCOMM_IDENT_LEN);
        m->order = g_next_order++;
        m->data = data;
        m->bytes = bytes;
        m->source = source;
        m->in_use = 1;
        g_drain_log_count++;
        return 1;
    }
    return 0;
}

drain_msg_t *drain_log_find(const uint8_t ident[VCOMM_IDENT_LEN], int source) {
    drain_msg_t *best = NULL;
    int i;
    for (i = 0; i < MAX_DRAIN_MSGS; i++) {
        drain_msg_t *m = &g_drain_log[i];
        if (!m->in_use || m->source != source)
            continue;
        if (memcmp(m->ident, ident, VCOMM_IDENT_LEN) != 0)
            continue;
        if (best == NULL || m->order < best->order)
            best = m;
    }
    return best;
}

void drain_log_release(drain_msg_t *m) {
    if (m == NULL || !m->in_use)
        return;
    free(m->data);
    memset(m, 0, sizeof(*m));
    g_drain_log_count--;
}

void drain_log_reset(void) {
    int i;
    for (i = 0; i < MAX_DRAIN_MSGS; i++) {
        if (g_drain_log[i].in_use)
            drain_log_release(&g_drain_log[i]);
    }
    drain_log_init();
}
