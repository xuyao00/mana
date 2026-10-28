/****************************************************************************
 *  Copyright (C) 2026 by Yao Xu                                            *
 *                                                                          *
 *  This file is part of DMTCP.                                             *
 *                                                                          *
 *  DMTCP is free software: you can redistribute it and/or                  *
 *  modify it under the terms of the GNU Lesser General Public License as   *
 *  published by the Free Software Foundation, either version 3 of the      *
 *  License, or (at your option) any later version.                         *
 *                                                                          *
 *  DMTCP is distributed in the hope that it will be useful,                *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *  GNU Lesser General Public License for more details.                     *
 *                                                                          *
 *  You should have received a copy of the GNU Lesser General Public        *
 *  License along with DMTCP:dmtcp/src.  If not, see                        *
 *  <http://www.gnu.org/licenses/>.                                         *
 ****************************************************************************/

// The front's runtime: per-thread lower-half TLS, binding of lower-half
// functions, and the checkpoint events.

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <dlfcn.h>
#include <mutex>
#include <set>
#include "front_objs.h"

void initialize_wrappers();  // libmana.so: sets lh_info

unsigned front_gen = 1;
__thread FrontThread front_thread
  __attribute__((tls_model("initial-exec"))) = { NULL, 0, -1, 0, {}, 0, {} };

// The process's initial thread, whose lower-half TLS the MPI calls share.
static FrontThread *initial_thread;

// The threads that have called CUDA, for the checkpoint.
static std::mutex threads_lock;
static std::set<FrontThread *> threads;

void
front_log(const char *fmt, ...)
{
  static int verbose = -1;
  if (verbose < 0) {
    const char *v = getenv("MANA_CUDA_VERBOSE");
    verbose = v != NULL && v[0] != '\0' && v[0] != '0';
  }
  if (!verbose) {
    return;
  }
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "MANA cuda[%d]: ", getpid());
  vfprintf(stderr, fmt, ap);
  va_end(ap);
}

void
front_fatal(const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "MANA: --cuda: ");
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  abort();
}

void
front_unsupported(const char *name)
{
  front_fatal("%s is not supported (cuda/DESIGN-lower-half-cuda.md)\n", name);
}

void
front_bad_handle(const void *handle, int kind)
{
  static const char *const kinds[] = { "", "stream", "event", "kernel",
                                       "fatbin", "context", "module",
                                       "function", "device", "CUDA array",
                                       "mipmapped array", "texture object" };
  front_fatal("%p is not a %s that the front made\n", handle,
              kind > 0 && kind < FRONT_N_KINDS ? kinds[kind] : "handle");
}

static void
front_need_lower_half()
{
  if (lh_info == NULL) {
    initialize_wrappers();
  }
  if (lh_info == NULL || !lh_info->cuda_enabled) {
    front_fatal("the lower half has no CUDA; run with mana_launch --cuda\n");
  }
}

// The process's initial thread uses the lower half's main TLS, as MPI calls
// do, so that the MPI's own CUDA calls see the application's current
// context.  (Its virtual tid is the virtual pid, also after a restart.)
static bool
front_is_initial_thread()
{
  return syscall(SYS_gettid) == getpid();
}

// The lower half's TLS factory takes one caller at a time.
static std::mutex tls_lock;

static void *
front_new_tls()
{
  std::lock_guard<std::mutex> guard(tls_lock);
  void *fs = NULL;
  typedef void *(*new_tls_t)(void);
  new_tls_t new_tls = (new_tls_t)lh_info->cuda_new_tls;
  JUMP_TO_LOWER_HALF(lh_info->cuda_tls_factory_fs);
  fs = new_tls();
  RETURN_TO_UPPER_HALF();
  if (fs == NULL) {
    front_fatal("cannot create a lower-half thread for a TLS\n");
  }
  return fs;
}

// The thread exits: so does the lower-half thread whose TLS it has.
static void
front_thread_exit()
{
  FrontThread *t = &front_thread;
  {
    std::lock_guard<std::mutex> guard(threads_lock);
    threads.erase(t);
  }
  if (t->gen != front_gen || t->fs == NULL || lh_info == NULL ||
      t->fs == lh_info->fsaddr) {
    return;
  }
  std::lock_guard<std::mutex> guard(tls_lock);
  typedef void (*free_tls_t)(void *);
  free_tls_t free_tls = (free_tls_t)lh_info->cuda_free_tls;
  void *fs = t->fs;
  JUMP_TO_LOWER_HALF(lh_info->cuda_tls_factory_fs);
  free_tls(fs);
  RETURN_TO_UPPER_HALF();
  t->fs = NULL;
  t->gen = 0;
}

void *
front_lh_fs_slow()
{
  front_need_lower_half();
  FrontThread *t = &front_thread;
  bool restarted = t->gen != 0 && t->gen != front_gen;
  bool initial = front_is_initial_thread();
  t->fs = initial ? lh_info->fsaddr : front_new_tls();
  t->gen = front_gen;
  if (initial) {
    initial_thread = t;
  }
  {
    std::lock_guard<std::mutex> guard(threads_lock);
    threads.insert(t);
  }
  // After a restart, this thread's contexts and current device come back on
  // its first call into the lower half.
  if (restarted) {
    front_bind(&front_cudart_lib);
    front_bind(&front_cuda_lib);
    front_stack_restore(t);
  }
  return t->fs;
}

static void *
front_lh_dlsym(const char *library, const char *name)
{
  typedef void *(*dlsym_t)(const char *, const char *);
  dlsym_t lh_dlsym = (dlsym_t)lh_info->cuda_dlsym;
  void *fs = front_lh_fs();
  void *fn = NULL;
  JUMP_TO_LOWER_HALF(fs);
  fn = lh_dlsym(library, name);
  RETURN_TO_UPPER_HALF();
  return fn;
}

// NULL for a function that this lower half lacks: the front libraries are
// made from the lower half's libraries, so only for one of an older driver,
// which the application would not have found there either.
void
front_bind(FrontLib *lib)
{
  static std::recursive_mutex lock;
  std::lock_guard<std::recursive_mutex> guard(lock);
  if (lib->bound_gen == front_gen) {
    return;
  }
  front_need_lower_half();
  for (int i = 0; i < lib->n; i++) {
    lib->real[i] = front_lh_dlsym(lib->soname, lib->names[i]);
  }
  __atomic_store_n(&lib->bound_gen, front_gen, __ATOMIC_RELEASE);
}

extern "C" void mana_cuda_event(int event);

unsigned front_slot_gen;

void
front_open_slot()
{
  static std::mutex lock;
  std::lock_guard<std::mutex> guard(lock);
  if (front_slot_gen == front_gen) {
    return;
  }
  front_need_lower_half();
  typedef void (*open_slot_t)(void);
  open_slot_t open_slot = (open_slot_t)lh_info->cuda_open_slot;
  void *fs = front_lh_fs();
  JUMP_TO_LOWER_HALF(fs);
  open_slot();
  RETURN_TO_UPPER_HALF();
  __atomic_store_n(&front_slot_gen, front_gen, __ATOMIC_RELEASE);
}

// libmana.so calls mana_cuda_event() through the function that it is given
// here, also if the application dlopen()s a front library.
__attribute__((constructor)) static void
front_register()
{
  typedef void (*register_t)(mana_cuda_event_t);
  register_t reg = (register_t)dlsym(RTLD_DEFAULT, "mana_cuda_register");
  if (reg != NULL) {
    reg(mana_cuda_event);
  }
}

// Called by libmana.so (mpi_plugin.cpp) on the checkpoint thread, with the
// lower half closed to the application's threads; MANA_CUDA_THREAD_EXIT, on
// an exiting thread; MANA_CUDA_DRAIN_MEMOPS, in an MPI call.
extern "C" void
mana_cuda_event(int event)
{
  switch (event) {
    case MANA_CUDA_QUIESCE: {
      // The application's threads are out of the lower half, so no new
      // work can start, and their context stacks stay as they are: the
      // copies learn what the MPI made current.  Then wait for the work
      // that was started.
      {
        std::lock_guard<std::mutex> guard(threads_lock);
        for (FrontThread *t : threads) {
          if (t != &front_thread && t->gen == front_gen) {
            front_stack_sync(t);
          }
        }
      }
      cudaError_t rc = cudaSuccess;
      for (int dev = 0; rc == cudaSuccess && dev < 64; dev++) {
        if (!(front_devices_used & (1ULL << dev))) {
          continue;
        }
        FRONT_LH(rc = REAL(cudaSetDevice)(dev));
        if (rc == cudaSuccess) {
          FRONT_LH(rc = REAL(cudaDeviceSynchronize)());
        }
      }
      if (rc != cudaSuccess) {
        front_fatal("synchronizing the GPUs before a checkpoint: error %d\n",
                    rc);
      }
      front_ctx_quiesce();
      front_events_quiesce();
      break;
    }
    case MANA_CUDA_STASH:
      front_objs_stash();
      front_vmm_stash();
      break;
    case MANA_CUDA_RESUME:
      front_vmm_drop_shadows();
      front_objs_drop_shadows();
      front_events_resume();
      break;
    case MANA_CUDA_RESTART:
      if (lh_info == NULL || !lh_info->cuda_enabled) {
        front_fatal("the new lower half has no CUDA; restart with "
                    "mana_restart --cuda\n");
      }
      __atomic_add_fetch(&front_gen, 1, __ATOMIC_ACQ_REL);
      // On the lower half's main TLS, where the initial thread made most
      // objects at launch: its per-thread state in the driver and in cudart
      // exists already, as it did then (see front_note_cudart()).  The MPI
      // calls of this thread use it too; the application's threads are not
      // running.
      front_thread.fs = lh_info->fsaddr;
      front_thread.gen = front_gen;
      front_objs_restore_devices();
      front_vmm_map();
      front_objs_restore();
      front_vmm_fill();
      front_events_restart();
      front_vmm_drop_shadows();
      front_objs_drop_shadows();
      // Now, not on its first call: the MPI's calls on the checkpoint thread
      // (the replay) use the initial thread's TLS too.  The application's
      // threads are not running yet.
      if (initial_thread != NULL) {
        initial_thread->fs = lh_info->fsaddr;
        initial_thread->gen = front_gen;
        front_stack_restore(initial_thread);
      } else {
        CUresult rc;
        FRONT_LH(rc = DRV(cuCtxSetCurrent)(NULL));
        (void)rc;
      }
      front_ipc_publish();
      // This thread gets a TLS of its own at its next call.
      if (&front_thread != initial_thread) {
        front_thread.fs = NULL;
        front_thread.gen = 0;
      }
      break;
    case MANA_CUDA_RESTART_IPC:
      front_ipc_reimport();
      break;
    case MANA_CUDA_THREAD_EXIT:
      front_thread_exit();
      break;
    case MANA_CUDA_DRAIN_MEMOPS:
      front_drain_memops();
      break;
    default:
      front_fatal("unknown event %d\n", event);
  }
}
