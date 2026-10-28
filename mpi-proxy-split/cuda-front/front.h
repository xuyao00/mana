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

// The CUDA front: MANA's libcudart in the upper half under --cuda.  The real
// CUDA runs in the lower half; see cuda/DESIGN-lower-half-cuda.md.
//
// Every call into the lower half switches the FS register to the calling
// thread's own lower-half TLS (front_lh_fs()), because cudart and the driver
// keep per-thread state there.  Calls from the application take MANA's
// checkpoint gate (FRONT_ENTER/FRONT_LEAVE); the front's internal calls into
// the lower half (FRONT_LH) do not, so that the checkpoint thread can make
// them while the lower half is closed.  Never take the gate while holding a
// lock of the front.

#ifndef _MANA_CUDA_FRONT_H
#define _MANA_CUDA_FRONT_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdint.h>
#include "lower-half-api.h"
#include "lower-half/switch-context.h"
#include "lower_half_ckpt.h"
#include "mana_cuda.h"

#define FRONT_HIDDEN __attribute__((visibility("hidden")))

// A lower-half library whose functions the generated part of the front
// forwards (gen-cuda-front.py).
struct FrontLib {
  const char *soname;
  const char *const *names;
  void **real;
  int n;
  unsigned bound_gen;
};

// Bumped at each restart: lower-half handles, function pointers and TLS of an
// older generation are stale.
extern unsigned front_gen FRONT_HIDDEN;

#define FRONT_MAX_LAUNCH_CONFIGS 4

struct FrontLaunchConfig {
  dim3 grid;
  dim3 block;
  size_t shared_mem;
  cudaStream_t stream;   // virtual
};

// Per upper-half thread.  Saved and restored with the thread.
struct FrontThread {
  void *fs;              // this thread's lower-half TLS (of generation gen)
  unsigned gen;
  int device;            // the last cudaSetDevice(), or -1
  int n_configs;         // __cudaPushCallConfiguration() stack
  FrontLaunchConfig configs[FRONT_MAX_LAUNCH_CONFIGS];
};

extern __thread FrontThread front_thread
  __attribute__((tls_model("initial-exec"))) FRONT_HIDDEN;

void *front_lh_fs_slow() FRONT_HIDDEN;

static inline void *
front_lh_fs()
{
  if (__builtin_expect(front_thread.gen == front_gen && front_gen != 0, 1)) {
    return front_thread.fs;
  }
  return front_lh_fs_slow();
}

// A call into the lower half, without the gate.
#define FRONT_LH(stmt)                     \
  do {                                     \
    void *front_fs_ = front_lh_fs();       \
    JUMP_TO_LOWER_HALF(front_fs_);         \
    stmt;                                  \
    RETURN_TO_UPPER_HALF();                \
  } while (0)

#define FRONT_ENTER() LOWER_HALF_DISABLE_CKPT()
#define FRONT_LEAVE() LOWER_HALF_ENABLE_CKPT()

[[noreturn]] void front_unsupported(const char *name) FRONT_HIDDEN;
[[noreturn]] void front_fatal(const char *fmt, ...) FRONT_HIDDEN
  __attribute__((format(printf, 1, 2)));
void front_log(const char *fmt, ...) FRONT_HIDDEN
  __attribute__((format(printf, 1, 2)));

// Binds the generated table of 'lib' to the current lower half.
void front_bind(FrontLib *lib) FRONT_HIDDEN;

static inline void
front_check_bound(FrontLib *lib)
{
  if (__builtin_expect(lib->bound_gen != front_gen || front_gen == 0, 0)) {
    front_bind(lib);
  }
}

// A lower-half function by name, for the hand-written parts of the front.
struct FrontSym {
  const char *library;
  const char *name;
  void *fn;
  unsigned gen;
};

void *front_sym_slow(FrontSym *sym) FRONT_HIDDEN;

static inline void *
front_sym(FrontSym *sym)
{
  if (__builtin_expect(sym->gen == front_gen && front_gen != 0, 1)) {
    return sym->fn;
  }
  return front_sym_slow(sym);
}

// The lower half's libcudart, which the front stands for.
#ifndef FRONT_CUDART_SONAME
# define FRONT_CUDART_SONAME "libcudart.so.13"
#endif

// REAL(f): the lower half's function f of libcudart; DRV(f), of libcuda.
// Their types are the ones that the headers declare for f.
#define FRONT_SYM_OF(lib, name, type)                              \
  ([]() {                                                          \
    static FrontSym front_sym_ = { lib, name, NULL, 0 };           \
    return (type)front_sym(&front_sym_);                           \
  }())
#define REAL(f) FRONT_SYM_OF(FRONT_CUDART_SONAME, #f, decltype(&::f))
#define DRV(f) FRONT_SYM_OF("libcuda.so.1", #f, decltype(&::f))

// Virtual handles: pointers to records in upper-half memory.  Streams and
// events are translated in every call (front_auto.h); special stream values
// (0, cudaStreamLegacy, cudaStreamPerThread) are not handles.
enum FrontKind {
  FRONT_STREAM = 1,
  FRONT_EVENT,
  FRONT_KERNEL,
  FRONT_FATBIN,
};

#define FRONT_MAGIC 0x6d616e61u  // "mana"

struct FrontObj {
  uint32_t magic;
  uint32_t kind;
  void *real;            // the lower half's handle, of generation front_gen
  FrontObj *prev;        // in the list of its kind, in creation order
  FrontObj *next;
};

static inline cudaStream_t
front_real_stream(cudaStream_t stream)
{
  if ((uintptr_t)stream <= 2) {
    return stream;
  }
  return (cudaStream_t)((FrontObj *)stream)->real;
}

static inline cudaEvent_t
front_real_event(cudaEvent_t event)
{
  if (event == NULL) {
    return NULL;
  }
  return (cudaEvent_t)((FrontObj *)event)->real;
}

// The VMM allocator (front_vmm.cpp).
cudaError_t front_vmm_alloc(void **ptr, size_t size) FRONT_HIDDEN;
// Returns false if 'ptr' is not the allocator's.
bool front_vmm_free(void *ptr, cudaError_t *result) FRONT_HIDDEN;
void front_vmm_stash() FRONT_HIDDEN;
void front_vmm_drop_shadows() FRONT_HIDDEN;
void front_vmm_restore() FRONT_HIDDEN;

// Records of objects (front_objs.cpp).
void front_objs_stash() FRONT_HIDDEN;
void front_objs_drop_shadows() FRONT_HIDDEN;
void front_objs_restore_devices() FRONT_HIDDEN;
void front_objs_restore() FRONT_HIDDEN;

// Bit d: the application used device d (device 0 always counts).
extern uint64_t front_devices_used FRONT_HIDDEN;

#endif  // ifndef _MANA_CUDA_FRONT_H
