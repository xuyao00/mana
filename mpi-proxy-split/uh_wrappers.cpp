/****************************************************************************
 *  Copyright (C) 2019-2020 by Twinkle Jain, Rohan garg, and Gene Cooperman *
 *  jain.t@husky.neu.edu, rohgarg@ccs.neu.edu, gene@ccs.neu.edu             *
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

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string.h>

#include "lower-half-api.h"
#include "dmtcp.h"
#include "mpi_plugin.h"
#include "cuda-forward/fwd-runtime.h"
#include "switch-context.h"
#include <dlfcn.h>
#include <pthread.h>

int initialized = 0;

void initialize_wrappers();
void reset_wrappers();
static void readLhInfoAddr();
extern "C" pid_t dmtcp_get_real_pid();

LowerHalfInfo_t *lh_info;
proxyDlsym_t pdlsym;

void initialize_wrappers() {
  if (!initialized) {
    readLhInfoAddr();
    initialized = 1;
  }
}

void reset_wrappers() {
  initialized = 0;
}

void* mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
  static __typeof__(&mmap) lowerHalfMmapWrapper = (__typeof__(&mmap)) - 1;
  if (!initialized) {
    initialize_wrappers();
  }
  if (lowerHalfMmapWrapper == (__typeof__(&mmap)) - 1) {
    lowerHalfMmapWrapper = (__typeof__(&mmap))lh_info->mmap;
  }
  void *ret;
  // Read mana_state once: a checkpoint can change it during the call, and
  // the lock taken must be the lock released.
  bool lock = mana_state == RUNNING;
  if (lock) {
    DMTCP_PLUGIN_DISABLE_CKPT();
  }
  ret = lowerHalfMmapWrapper(addr, length, prot, flags, fd, offset);
  if (lock) {
    DMTCP_PLUGIN_ENABLE_CKPT();
  }
  return ret;
}

int munmap(void *addr, size_t length) {
  static __typeof__(&munmap) lowerHalfMunmapWrapper = (__typeof__(&munmap)) - 1;
  if (!initialized) {
    initialize_wrappers();
  }
  if (lowerHalfMunmapWrapper == (__typeof__(&munmap)) - 1) {
    lowerHalfMunmapWrapper = (__typeof__(&munmap))lh_info->munmap;
  }
  int ret;
  bool lock = mana_state == RUNNING;
  if (lock) {
    DMTCP_PLUGIN_DISABLE_CKPT();
  }
  ret = lowerHalfMunmapWrapper(addr, length);
  if (lock) {
    DMTCP_PLUGIN_ENABLE_CKPT();
  }
  return ret;
}

static void readLhInfoAddr() {
  char *addr_str = getenv("MANA_LH_INFO_ADDR");
  // If the env var is set, MANA is launching. Otherwise, MANA is restarting
  if (addr_str != NULL) {
    lh_info = (LowerHalfInfo_t*) strtol(addr_str, NULL, 16);
  } else {
    // File name format: mana_tmp_lh_info_[hostname]_[pid]
    char filename[100] = "/tmp/mana_tmp_lh_info_";
    gethostname(filename + strlen(filename), 100 - strlen(filename));
    filename[strlen(filename)] = '_';
    // Convert real pid to char* without calling snprintf
    // During process startup, avoid directly or indirectly
    // calling a libc function that is a DMTCP wrapper.
    int real_pid = dmtcp_get_real_pid();
    static char buf[32] = {0};
    int i = 30;
    for(; real_pid && i ; --i, real_pid /= 10) {
      buf[i] = "0123456789"[real_pid % 10];
    }
    memcpy(filename + strlen(filename), &buf[i+1], strlen(&buf[i+1]));
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
      printf("Could not open %s for reading.\n", filename);
      exit(-1);
    }
    ssize_t rc = read(fd, &lh_info, sizeof(lh_info));
    if (rc != (ssize_t)sizeof(lh_info)) {
      perror("Read fewer bytes than expected from addr.bin.\n");
      exit(-1);
    }
    close(fd);
    if (remove(filename) != 0) {
      fprintf(stderr, "Cannot remove MANA tmp file %s\n", filename);
    }
  }
  pdlsym = (proxyDlsym_t)lh_info->lh_dlsym;
}

// CUDA-aware MPI (MANA_CUDA_FORWARD; see cuda-forward/fwd-runtime.c): the
// lower half's libcuda/libcudart are shims forwarding to these libraries of
// the upper half.  Runs in the upper half, with its FS.
static void *
mana_uh_dlsym(const char *soname, const char *symbol)
{
  void *handle = dlopen(soname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
  if (handle == NULL) {
    handle = dlopen(soname, RTLD_NOW | RTLD_GLOBAL);
  }
  return handle != NULL ? dlsym(handle, symbol) : NULL;
}

// Tells the forwarding shims the upper half's FS of thread 'real_tid'.
void
mana_fwd_note_thread(pid_t real_tid, unsigned long fs)
{
  initialize_wrappers();
  FwdCtl *ctl = (FwdCtl *)lh_info->fwd_ctl;
  if (ctl == NULL) {
    return;
  }
  ctl->uh_dlsym = mana_uh_dlsym;
  // The thread's stack, by which the shims find the thread without a
  // system call.  Called on the thread itself.
  unsigned long lo = 0, hi = 0;
  if (real_tid == dmtcp_get_real_pid()) {
    lo = (unsigned long)lh_info->uh_stack_start;
    hi = (unsigned long)lh_info->uh_stack_end;
  } else {
    pthread_attr_t attr;
    if (pthread_getattr_np(pthread_self(), &attr) == 0) {
      void *addr;
      size_t size;
      if (pthread_attr_getstack(&attr, &addr, &size) == 0) {
        lo = (unsigned long)addr;
        hi = lo + size;
      }
      pthread_attr_destroy(&attr);
    }
  }
  if (ctl->register_thread_stack != NULL) {
    ctl->register_thread_stack(real_tid, fs, lo, hi);
  } else {
    ctl->register_thread(real_tid, fs);
  }
}

// MANA_CUDA_FORWARD: makes the CUDA context that owns 'buf' current on this
// thread, if 'buf' is device memory.  The checkpoint thread calls MPI (the
// restart replay of pending receives) with no context or another device's,
// and the MPI's CUDA calls on the buffer would fail (libfabric:
// cuMemGetAddressRange -> CUDA_ERROR_NOT_FOUND).  Runs in the upper half,
// with its FS.
void
mana_fwd_use_context_of(const void *buf)
{
  // From cuda.h, which MANA does not need to build.
  const int CU_POINTER_ATTRIBUTE_CONTEXT = 1;
  const int CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9;
  typedef int (*attr_fn)(void *, int, unsigned long long);
  typedef int (*set_fn)(void *);
  typedef int (*retain_fn)(void **, int);
  static attr_fn attr = NULL;
  static set_fn set = NULL;
  static retain_fn retain = NULL;
  static int looked = 0;
  if (lh_info == NULL || lh_info->fwd_ctl == NULL || buf == NULL) {
    return;
  }
  if (!looked) {
    looked = 1;
    // The application's driver, if it has one: never load it here.
    void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (cuda != NULL) {
      attr = (attr_fn)dlsym(cuda, "cuPointerGetAttribute");
      set = (set_fn)dlsym(cuda, "cuCtxSetCurrent");
      retain = (retain_fn)dlsym(cuda, "cuDevicePrimaryCtxRetain");
    }
  }
  if (attr == NULL || set == NULL) {
    return;
  }
  void *ctx = NULL;
  int dev = -1;
  if (attr(&ctx, CU_POINTER_ATTRIBUTE_CONTEXT, (uintptr_t)buf) == 0 &&
      ctx != NULL) {
    set(ctx);
  } else if (retain != NULL &&
             attr(&dev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                  (uintptr_t)buf) == 0 &&
             retain(&ctx, dev) == 0) {
    // VMM memory (libmana_cuda_vmm.so) belongs to no context: use its
    // device's primary context, the CUDA runtime's.
    set(ctx);
  }
}

// MANA_CUDA_FORWARD: whether 'buf' is device memory of the application's
// CUDA.  Runs in the upper half, with its FS.
int
mana_fwd_is_device_memory(const void *buf)
{
  // From cuda.h, which MANA does not need to build.
  const int CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2;
  const unsigned CU_MEMORYTYPE_DEVICE = 2;
  typedef int (*attr_fn)(void *, int, unsigned long long);
  static attr_fn attr = NULL;
  static int looked = 0;
  if (lh_info == NULL || lh_info->fwd_ctl == NULL || buf == NULL) {
    return 0;
  }
  if (!looked) {
    looked = 1;
    void *cuda = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (cuda != NULL) {
      attr = (attr_fn)dlsym(cuda, "cuPointerGetAttribute");
    }
  }
  unsigned type = 0;
  return attr != NULL &&
         attr(&type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (uintptr_t)buf) == 0 &&
         type == CU_MEMORYTYPE_DEVICE;
}

// With MANA_CUDA_FORWARD, the lower half initializes MPI when the upper half
// asks (lh_info->lh_mpi_init): at the application's MPI_Init(), and at
// restart after the CUDA plugin restored the GPU (DMTCP_EVENT_RESTART runs in
// reverse plugin order), since the MPI's CUDA calls go to this half's CUDA.
extern "C" pid_t dmtcp_get_real_tid() __attribute((weak));

void
mana_lower_half_mpi_init()
{
  initialize_wrappers();
  if (lh_info->lh_mpi_init != NULL) {
    mana_fwd_note_thread(dmtcp_get_real_tid(), getFS());
    JUMP_TO_LOWER_HALF(lh_info->fsaddr);
    ((void (*)(void))lh_info->lh_mpi_init)();
    RETURN_TO_UPPER_HALF();
  }
}

// MANA_CUDA_FORWARD: see FwdCtl::before_ckpt/after_resume.  Run with the
// lower half's FS (they are lower-half code that uses its libc).
void
mana_fwd_before_ckpt()
{
  FwdCtl *ctl = lh_info != NULL ? (FwdCtl *)lh_info->fwd_ctl : NULL;
  if (ctl != NULL && ctl->before_ckpt != NULL) {
    JUMP_TO_LOWER_HALF(lh_info->fsaddr);
    ctl->before_ckpt();
    RETURN_TO_UPPER_HALF();
  }
}

void
mana_fwd_after_resume()
{
  FwdCtl *ctl = lh_info != NULL ? (FwdCtl *)lh_info->fwd_ctl : NULL;
  if (ctl != NULL && ctl->after_resume != NULL) {
    JUMP_TO_LOWER_HALF(lh_info->fsaddr);
    ctl->after_resume();
    RETURN_TO_UPPER_HALF();
  }
}

void
mana_fwd_after_resume_peers()
{
  FwdCtl *ctl = lh_info != NULL ? (FwdCtl *)lh_info->fwd_ctl : NULL;
  if (ctl != NULL && ctl->after_resume_peers != NULL) {
    JUMP_TO_LOWER_HALF(lh_info->fsaddr);
    ctl->after_resume_peers();
    RETURN_TO_UPPER_HALF();
  }
}
