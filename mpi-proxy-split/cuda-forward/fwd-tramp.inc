/*
 * Trampolines of MANA's forwarding shims (see fwd-runtime.c).
 *
 *   FWD_COMMON desc, common, ret   -- defines the shared entry 'common' and
 *                                     return path 'ret' for library 'desc'
 *   FWD_SLOT name, idx, common     -- a function 'name' forwarding slot 'idx'
 *
 * A slot loads its index into r11 and jumps to 'common', which saves the
 * argument registers, calls mana_fwd_enter(desc, idx, &return_address) (lower
 * half, lower FS) for a FwdRecord, keeps the record in r12 (callee-saved: the
 * caller's r12 is saved in the record), restores the arguments, replaces the
 * return address with 'ret', switches FS to the upper half's and jumps to the
 * target.  The stack arguments stay where the caller put them.  'ret' switches
 * FS back, pops the record (mana_fwd_exit), restores r12 and returns to the
 * caller with the target's rax/rdx/xmm0.
 * FwdRecord: target 0, uhfs 8, lhfs 16, ret 24, saved_r12 32.
 */

.macro FWD_SETFS field
  movq mana_fwd_fsgsbase@GOTPCREL(%rip), %r11
  cmpl $0, (%r11)
  je 1f
  movq \field(%r12), %r11
  wrfsbase %r11
  jmp 2f
1:
  pushq %rax
  pushq %rcx
  pushq %rdi
  pushq %rsi
  movl $158, %eax          /* SYS_arch_prctl */
  movl $0x1002, %edi       /* ARCH_SET_FS */
  movq \field(%r12), %rsi
  syscall
  popq %rsi
  popq %rdi
  popq %rcx
  popq %rax
2:
.endm

.macro FWD_COMMON desc, common, ret
  .text
  .p2align 4
  .type \common, @function
\common:
  subq $200, %rsp
  movdqu %xmm0, 0(%rsp)
  movdqu %xmm1, 16(%rsp)
  movdqu %xmm2, 32(%rsp)
  movdqu %xmm3, 48(%rsp)
  movdqu %xmm4, 64(%rsp)
  movdqu %xmm5, 80(%rsp)
  movdqu %xmm6, 96(%rsp)
  movdqu %xmm7, 112(%rsp)
  movq %rdi, 128(%rsp)
  movq %rsi, 136(%rsp)
  movq %rdx, 144(%rsp)
  movq %rcx, 152(%rsp)
  movq %r8, 160(%rsp)
  movq %r9, 168(%rsp)
  movq %rax, 176(%rsp)
  leaq \desc(%rip), %rdi
  movq %r11, %rsi
  leaq 200(%rsp), %rdx
  call mana_fwd_enter@PLT
  movq %r12, 32(%rax)
  movq %rax, %r12
  movdqu 0(%rsp), %xmm0
  movdqu 16(%rsp), %xmm1
  movdqu 32(%rsp), %xmm2
  movdqu 48(%rsp), %xmm3
  movdqu 64(%rsp), %xmm4
  movdqu 80(%rsp), %xmm5
  movdqu 96(%rsp), %xmm6
  movdqu 112(%rsp), %xmm7
  movq 128(%rsp), %rdi
  movq 136(%rsp), %rsi
  movq 144(%rsp), %rdx
  movq 152(%rsp), %rcx
  movq 160(%rsp), %r8
  movq 168(%rsp), %r9
  movq 176(%rsp), %rax
  addq $200, %rsp
  leaq \ret(%rip), %r11
  movq %r11, (%rsp)
  FWD_SETFS 8
  movq 0(%r12), %r11
  jmp *%r11
  .size \common, .-\common

  .p2align 4
  .type \ret, @function
\ret:
  subq $48, %rsp
  movq %rax, 0(%rsp)
  movq %rdx, 8(%rsp)
  movdqu %xmm0, 16(%rsp)
  movq 32(%r12), %r11
  movq %r11, 32(%rsp)
  movq 24(%r12), %r11
  movq %r11, 40(%rsp)
  FWD_SETFS 16
  movq %r12, %rdi
  call mana_fwd_exit@PLT
  movq 40(%rsp), %r11
  movq 32(%rsp), %r12
  movq 0(%rsp), %rax
  movq 8(%rsp), %rdx
  movdqu 16(%rsp), %xmm0
  addq $48, %rsp
  jmp *%r11
  .size \ret, .-\ret
.endm

.macro FWD_SLOT name, idx, common
  .globl \name
  .type \name, @function
  .p2align 4
\name:
  movl $\idx, %r11d
  jmp \common
  .size \name, .-\name
.endm
