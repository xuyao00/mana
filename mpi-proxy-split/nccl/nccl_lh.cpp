// NCCL in the lower half: the upper half's side of the calls into it (see
// README and nccl_lh.h).  The thunks themselves are generated
// (nccl_lh_thunks.cpp).

#include <stdio.h>
#include <stdlib.h>

#include "nccl_lh.h"
#include "lower_half_ckpt.h"

extern void initialize_wrappers();
extern void mana_fwd_provide_donors();

// Nesting depth of mana_nccl_internal_begin() on this thread.
static __thread int t_internal ATTR_TLS_INITIAL_EXEC;

extern "C" void *
mana_nccl_lh_dlsym(const char *name)
{
  initialize_wrappers();
  if (lh_info == NULL || lh_info->lh_nccl_dlsym == NULL) {
    return NULL;
  }
  // NCCL's threads in the lower half call CUDA.
  mana_fwd_provide_donors();
  void *(*lh_dlsym)(const char *) =
    (void *(*)(const char *))lh_info->lh_nccl_dlsym;
  void *fn;
  JUMP_TO_LOWER_HALF(lh_info->fsaddr);
  fn = lh_dlsym(name);
  RETURN_TO_UPPER_HALF();
  return fn;
}

extern "C" int
mana_nccl_gate_enter(void)
{
  if (t_internal > 0) {
    return 0;
  }
  LOWER_HALF_DISABLE_CKPT();
  return 1;
}

extern "C" void
mana_nccl_gate_leave(int gated)
{
  if (gated) {
    LOWER_HALF_ENABLE_CKPT();
  }
}

extern "C" void
mana_nccl_internal_begin(void)
{
  t_internal++;
}

extern "C" void
mana_nccl_internal_end(void)
{
  t_internal--;
}

extern "C" void
mana_nccl_missing(const char *name)
{
  fprintf(stderr, "MANA: the lower half's NCCL has no %s()\n", name);
}

extern "C" void
mana_nccl_refuse(const char *name)
{
  fprintf(stderr, "MANA: %s() is not supported on a checkpointable NCCL "
          "communicator\n", name);
}
