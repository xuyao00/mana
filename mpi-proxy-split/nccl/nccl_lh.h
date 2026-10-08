// NCCL in the lower half: the upper half's calls into it (see README).
#ifndef MANA_NCCL_LH_H
#define MANA_NCCL_LH_H

#include <nccl.h>
#include <string.h>

#ifdef __cplusplus
# include "lower-half-api.h"
# include "switch-context.h"
extern "C" {
#endif

// lh_fn()'s mark of a function that the lower half's NCCL does not have.
#define MANA_NCCL_MISSING ((void *)-1)

// The lower half's NCCL function 'name' (lh_info->lh_nccl_dlsym), or NULL.
void *mana_nccl_lh_dlsym(const char *name);
// The thunk (mana_lh_ncclX) of 'name' if the lower half's NCCL has 'name'.
void *mana_nccl_real_dlsym(const char *name);
// Forgets the lower half's functions: at restart, the lower half is new.
void mana_nccl_reset_lh_fns(void);

// A thunk's call into the lower half is part of the application's call, and
// waits while the lower half is closed for a checkpoint
// (LOWER_HALF_DISABLE_CKPT), unless MANA itself makes it: the checkpoint
// thread's drain and teardown, and the rebuild, run between
// mana_nccl_internal_begin() and mana_nccl_internal_end().
int mana_nccl_gate_enter(void);
void mana_nccl_gate_leave(int gated);
void mana_nccl_internal_begin(void);
void mana_nccl_internal_end(void);
// Registers the calling thread with the CUDA forwarding shims
// (mana_fwd_note_thread()).  A thread that MANA's NCCL code creates itself,
// like the rebuild's helper, is otherwise taken for a thread of the lower half:
// its CUDA calls from the lower half (NCCL's cudaGetDevice()) run with a
// donor's TLS, whose current device is 0 whatever the thread set.
void mana_nccl_note_thread(void);

void mana_nccl_missing(const char *name);
void mana_nccl_refuse(const char *name);

#ifdef __cplusplus
}
#endif

#endif
