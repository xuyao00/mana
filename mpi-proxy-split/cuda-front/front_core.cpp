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
#include <mutex>
#include "front.h"

void initialize_wrappers();  // libmana.so: sets lh_info

unsigned front_gen = 1;
__thread FrontThread front_thread
  __attribute__((tls_model("initial-exec"))) = { NULL, 0, -1, 0, {} };

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

static void *
front_new_tls()
{
  static std::mutex lock;
  std::lock_guard<std::mutex> guard(lock);
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

void *
front_lh_fs_slow()
{
  front_need_lower_half();
  FrontThread *t = &front_thread;
  bool restarted = t->gen != 0 && t->gen != front_gen;
  t->fs = front_is_initial_thread() ? lh_info->fsaddr : front_new_tls();
  t->gen = front_gen;
  // After a restart, this thread's current device comes back on its first
  // call into the lower half.
  if (restarted && t->device >= 0) {
    int device = t->device;
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaSetDevice)(device));
    if (rc != cudaSuccess) {
      front_fatal("restoring the current device %d: error %d\n", device, rc);
    }
  }
  return t->fs;
}

static void *
front_lh_dlsym(const char *library, const char *name)
{
  typedef void *(*dlsym_t)(const char *, const char *);
  dlsym_t lh_dlsym = (dlsym_t)lh_info->cuda_dlsym;
  void *fn = NULL;
  FRONT_LH(fn = lh_dlsym(library, name));
  return fn;
}

void *
front_sym_slow(FrontSym *sym)
{
  front_need_lower_half();
  void *fn = front_lh_dlsym(sym->library, sym->name);
  if (fn == NULL) {
    front_fatal("the lower half has no %s in %s\n", sym->name, sym->library);
  }
  sym->fn = fn;
  __atomic_store_n(&sym->gen, front_gen, __ATOMIC_RELEASE);
  return fn;
}

void
front_bind(FrontLib *lib)
{
  static std::mutex lock;
  std::lock_guard<std::mutex> guard(lock);
  if (lib->bound_gen == front_gen) {
    return;
  }
  front_need_lower_half();
  for (int i = 0; i < lib->n; i++) {
    // NULL for a function that this lower half lacks; its call aborts in
    // the lower half, so check here.
    lib->real[i] = front_lh_dlsym(lib->soname, lib->names[i]);
  }
  __atomic_store_n(&lib->bound_gen, front_gen, __ATOMIC_RELEASE);
}

// Called by libmana.so (mpi_plugin.cpp) on the checkpoint thread, with the
// lower half closed to the application's threads.
extern "C" void
mana_cuda_event(int event)
{
  switch (event) {
    case MANA_CUDA_QUIESCE: {
      // The application's threads are out of the lower half, so no new
      // work can start.  Wait for the work that was started.
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
      break;
    }
    case MANA_CUDA_STASH:
      front_objs_stash();
      front_vmm_stash();
      break;
    case MANA_CUDA_RESUME:
      front_vmm_drop_shadows();
      front_objs_drop_shadows();
      break;
    case MANA_CUDA_RESTART:
      if (lh_info == NULL || !lh_info->cuda_enabled) {
        front_fatal("the new lower half has no CUDA; restart with "
                    "mana_restart --cuda\n");
      }
      __atomic_add_fetch(&front_gen, 1, __ATOMIC_ACQ_REL);
      front_objs_restore_devices();
      front_vmm_restore();
      front_objs_restore();
      front_vmm_drop_shadows();
      front_objs_drop_shadows();
      break;
    default:
      front_fatal("unknown event %d\n", event);
  }
}
