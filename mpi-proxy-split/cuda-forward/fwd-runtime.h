#ifndef MANA_FWD_RUNTIME_H
#define MANA_FWD_RUNTIME_H

#include <stddef.h>

// See fwd-runtime.c.  The offsets of FwdRecord's fields are used by the
// trampolines (fwd-tramp.inc): keep them in sync.
typedef struct {
  void *target;               //  0: the upper-half function
  unsigned long uhfs;         //  8: FS to call it with
  unsigned long lhfs;         // 16: FS to return with
  void *ret;                  // 24: the caller's return address
  unsigned long saved_r12;    // 32: the caller's r12 (the trampoline's own)
} FwdRecord;

typedef struct {
  const char *soname;         // library in the upper half, e.g. libcuda.so.1
  const char **names;         // symbol of each slot
  void **targets;             // resolved upper-half function of each slot
  int n;
  void (*on_upper)(void);     // Called once, when the upper half is there.
  int upper_seen;
} FwdLib;

// Shared between the lower half (libmana_fwd.so) and MANA's upper half
// (lh_info->fwd_ctl points here).
typedef struct {
  // Set by the upper half: dlopen()s 'soname' in the upper half if needed,
  // and returns its 'symbol'.  Must be called with the upper half's FS.
  void *(*uh_dlsym)(const char *soname, const char *symbol);
  // Called by the upper half (with the lower half's FS): this thread's FS.
  void (*register_thread)(int tid, unsigned long uhfs);
  void (*forget_threads)(void);
  int *fsgsbase;              // Set to 1 by the lower half if wrfsbase works.
  // Called by the upper half (with any FS) on the checkpoint thread: before
  // the CUDA plugin checkpoints the GPU (after MANA's drain), and after it
  // restored the GPU on resume.  The lower half's CUDA state that refers to
  // lower-half memory (host-memory registrations) must not be in the
  // checkpoint: at restart, the lower half is a new one.
  void (*before_ckpt)(void);
  void (*after_resume)(void);
  // On resume, after every process ran after_resume (a global barrier):
  // the state shared between processes (CUDA IPC imports).
  void (*after_resume_peers)(void);
  // As register_thread, with the thread's stack [lo, hi) (no gettid then).
  void (*register_thread_stack)(int tid, unsigned long uhfs,
                                unsigned long lo, unsigned long hi);
} FwdCtl;

#ifdef __cplusplus
extern "C" {
#endif
extern FwdCtl mana_fwd_ctl;
FwdCtl *mana_fwd_get_ctl(void);
void mana_fwd_register_thread(int tid, unsigned long uhfs);
void mana_fwd_register_thread_stack(int tid, unsigned long uhfs,
                                    unsigned long lo, unsigned long hi);
void mana_fwd_forget_threads(void);
unsigned long mana_fwd_uhfs(void);
void mana_fwd_switch(unsigned long fs);
unsigned long mana_fwd_current_fs(void);
void *mana_fwd_wrap_ptr(void *upper_fn, const char *name);
void mana_fwd_note_host_register(void *ptr, size_t size, unsigned flags,
                                 void *ctx);
void mana_fwd_note_host_unregister(void *ptr);
extern void (*mana_fwd_cuda_before_ckpt)(void);
extern void (*mana_fwd_cuda_after_resume)(void);
extern void (*mana_fwd_cuda_after_resume_peers)(void);
const void *mana_fwd_wrap_table(const void *table, void *lib_fn,
                                const void *uuid);
#ifdef __cplusplus
}
#endif

// For the C wrappers in the shims: call an upper-half function pointer;
// 'result' is 'fallback' if this thread has no upper half.
#define FWD_CALL_UPPER(result, fallback, call) \
  do { \
    unsigned long uh_ = mana_fwd_uhfs(); \
    unsigned long lh_ = mana_fwd_current_fs(); \
    if (uh_ == 0) { \
      result = (fallback); \
    } else { \
      mana_fwd_switch(uh_); \
      result = (call); \
      mana_fwd_switch(lh_); \
    } \
  } while (0)

#endif
