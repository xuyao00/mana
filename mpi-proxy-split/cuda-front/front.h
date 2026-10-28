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
#include "front_reg.h"
#include "gen/cuda-index.h"
#include "gen/cudart-index.h"
#include "lower-half-api.h"
#include "lower-half/switch-context.h"
#include "lower_half_ckpt.h"
#include "mana_cuda.h"

#define FRONT_HIDDEN __attribute__((visibility("hidden")))

// A lower-half library that a front library stands for: its functions, as
// the generated dispatch table lists them (gen-cuda-front.py).
struct FrontLib {
  const char *soname;
  const char *const *names;
  void **real;           // the lower half's, of generation bound_gen
  int n;
  unsigned bound_gen;
};

extern FrontLib front_cudart_lib FRONT_HIDDEN;
extern FrontLib front_cuda_lib FRONT_HIDDEN;

// Bumped at each restart: lower-half handles, function pointers and TLS of an
// older generation are stale.
extern unsigned front_gen FRONT_HIDDEN;

#define FRONT_MAX_LAUNCH_CONFIGS 4
#define FRONT_MAX_CTX_STACK 8

struct Ctx;

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
  int n_ctx;             // the driver's context stack of this thread
  Ctx *ctx_stack[FRONT_MAX_CTX_STACK];
  struct MemopEvents *memops;  // front_memops.cpp
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

// Binds the tables of 'lib' to the current lower half.
void front_bind(FrontLib *lib) FRONT_HIDDEN;

static inline void
front_check_bound(FrontLib *lib)
{
  if (__builtin_expect(lib->bound_gen != front_gen || front_gen == 0, 0)) {
    front_bind(lib);
  }
}

// A call into the lower half, without the gate.  'stmt' runs on the lower
// half's TLS: it must not touch the upper half's (no front_thread, no
// errno, no front function that may), so REAL() and DRV() are bound before.
#define FRONT_LH(stmt)                     \
  do {                                     \
    void *front_fs_ = front_lh_fs();       \
    front_check_bound(&front_cudart_lib);  \
    front_check_bound(&front_cuda_lib);    \
    JUMP_TO_LOWER_HALF(front_fs_);         \
    stmt;                                  \
    RETURN_TO_UPPER_HALF();                \
  } while (0)

// The lower half's slot for the first context (lh_info->cuda_open_slot) is
// released at the first call that may make a context: any but the fatbin
// registrations of the application's constructors (FRONT_ENTER_REGISTER).
extern unsigned front_slot_gen FRONT_HIDDEN;
void front_open_slot() FRONT_HIDDEN;

// The gate without LOWER_HALF_DISABLE_CKPT()'s wait for the memops, which
// is for MPI calls.
#define FRONT_ENTER()                                       \
  do {                                                      \
    lower_half_disable_ckpt();                              \
    if (__builtin_expect(front_slot_gen != front_gen, 0)) { \
      front_open_slot();                                    \
    }                                                       \
  } while (0)
#define FRONT_ENTER_REGISTER() lower_half_disable_ckpt()
#define FRONT_LEAVE() LOWER_HALF_ENABLE_CKPT()

[[noreturn]] void front_unsupported(const char *name) FRONT_HIDDEN;
[[noreturn]] void front_fatal(const char *fmt, ...) FRONT_HIDDEN
  __attribute__((format(printf, 1, 2)));
void front_log(const char *fmt, ...) FRONT_HIDDEN
  __attribute__((format(printf, 1, 2)));

// REAL(f): the lower half's function f of libcudart; DRV(f), of libcuda;
// with the types that the headers declare for f.  Inside FRONT_LH() only
// (or after front_check_bound()): they read the bound tables.
#define REAL(f) \
  ((decltype(&::f))front_cudart_lib.real[FRONT_IDX_cudart_##f])
#define DRV(f) ((decltype(&::f))front_cuda_lib.real[FRONT_IDX_cuda_##f])
#define DRV_AS(f, type) ((type)front_cuda_lib.real[FRONT_IDX_cuda_##f])

// Virtual handles: pointers to records in upper-half memory.  Streams and
// events are translated in every call (front_auto.h); special stream values
// (0, cudaStreamLegacy, cudaStreamPerThread) are not handles.
enum FrontKind {
  FRONT_STREAM = 1,
  FRONT_EVENT,
  FRONT_KERNEL,
  FRONT_FATBIN,
  FRONT_CTX,
  FRONT_MODULE,
  FRONT_FUNCTION,
  FRONT_DEVICE,          // cudart's context of a device was made
  FRONT_N_KINDS
};

#define FRONT_MAGIC 0x6d616e61u  // "mana"

struct FrontObj {
  uint32_t magic;
  uint32_t kind;
  void *real;            // the lower half's handle, of generation front_gen
  uint64_t seq;          // creation order, over all kinds
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

[[noreturn]] void front_bad_handle(const void *handle, int kind) FRONT_HIDDEN;

// Contexts, modules and functions are checked: the application may hold a
// real one that the front did not make (cudaGetFuncBySymbol, ...).
static inline void *
front_real_obj(const void *handle, int kind)
{
  const FrontObj *obj = (const FrontObj *)handle;
  if (obj == NULL) {
    return NULL;
  }
  if (__builtin_expect(obj->magic != FRONT_MAGIC || (int)obj->kind != kind,
                       0)) {
    front_bad_handle(handle, kind);
  }
  return obj->real;
}

// The VMM allocator (front_vmm.cpp).
cudaError_t front_vmm_alloc(void **ptr, size_t size) FRONT_HIDDEN;
cudaError_t front_vmm_alloc_pitch(void **ptr, size_t *pitch, size_t width,
                                  size_t height) FRONT_HIDDEN;
// Returns false if 'ptr' is not the allocator's.
bool front_vmm_free(void *ptr, cudaError_t *result) FRONT_HIDDEN;
// cudaFreeAsync, cuMemFreeAsync: free once 'real_stream' gets there.
bool front_vmm_free_async(void *ptr, cudaStream_t real_stream, int *result)
  FRONT_HIDDEN;
void front_vmm_stash() FRONT_HIDDEN;
void front_vmm_drop_shadows() FRONT_HIDDEN;
// At restart: the mappings, before anything else; their contents, last.
void front_vmm_map() FRONT_HIDDEN;
void front_vmm_fill() FRONT_HIDDEN;

// CUDA IPC (front_ipc.cpp): the chunk of a live allocation that starts at
// 'ptr', exported as a POSIX fd (once per generation).
struct FrontExport {
  CUdeviceptr base;
  size_t size;
  int fd;
};
bool front_vmm_export(void *ptr, FrontExport *out) FRONT_HIDDEN;
// At restart: exports what was exported, and publishes each.
void front_vmm_reexport(void (*publish)(const FrontExport &)) FRONT_HIDDEN;
// Addresses in the range for imported memory.
CUdeviceptr front_vmm_take_va(size_t size) FRONT_HIDDEN;
void front_vmm_give_back_va(CUdeviceptr va, size_t size) FRONT_HIDDEN;
void front_vmm_peer_access(int device, int peer, bool on) FRONT_HIDDEN;
void front_ipc_publish() FRONT_HIDDEN;
void front_ipc_reimport() FRONT_HIDDEN;

// cudart made its context of 'device' (front_objs.cpp): a restart makes it
// again at the same point among the other creations.  The driver places
// module data by what it allocated before, so this order gives the same
// addresses.
void front_note_cudart(int device) FRONT_HIDDEN;

// Records of objects (front_objs.cpp).
void front_objs_stash() FRONT_HIDDEN;
void front_objs_drop_shadows() FRONT_HIDDEN;
void front_objs_restore_devices() FRONT_HIDDEN;
void front_objs_restore() FRONT_HIDDEN;

// After a synchronous copy or set of the application, on the legacy (or
// with 'ptds', the per-thread) default stream; within the gate.
void front_order_memop(bool ptds) FRONT_HIDDEN;
// MANA_CUDA_DRAIN_MEMOPS: libmana.so is about to call MPI.
void front_drain_memops() FRONT_HIDDEN;

// Bit d: cudart has a context of device d.
extern uint64_t front_devices_used FRONT_HIDDEN;

#endif  // ifndef _MANA_CUDA_FRONT_H
