/* Dynamic trampolines: upper-half function pointers handed to the lower half
 * (cuGetProcAddress, cudaGetDriverEntryPoint); see mana_fwd_wrap_ptr(). */
#include "fwd-tramp.inc"

FWD_COMMON mana_fwd_dyn_lib, fwd_dyn_common, fwd_dyn_ret

  .text
  .p2align 4
  .globl mana_fwd_dyn_tramp
  .hidden mana_fwd_dyn_tramp
mana_fwd_dyn_tramp:
  .set slot, 0
  .rept 4096
  .p2align 4
  movl $slot, %r11d
  jmp fwd_dyn_common
  .set slot, slot + 1
  .endr

  .section .note.GNU-stack,"",@progbits
