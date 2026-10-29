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

// The front's records of the application's live CUDA objects, shared by the
// runtime's own functions (front_objs.cpp) and the driver's (front_drv.cpp).
// A record is the virtual handle that the application holds; it keeps what
// a restart needs to make the object again.

#ifndef _MANA_CUDA_FRONT_OBJS_H
#define _MANA_CUDA_FRONT_OBJS_H

#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "front.h"

// A device's primary context (cudart's, and cuDevicePrimaryCtxRetain's), or
// one of cuCtxCreate.
struct Ctx : FrontObj {
  int device;
  bool primary;
  bool destroyed;        // kept: a thread's context stack may still name it
  unsigned flags;        // cuCtxCreate's (not primary)
  int retains;           // primary: the application's retains, not released
  std::map<int, size_t> limits;  // cuCtxSetLimit, cudaDeviceSetLimit
  int cache_config;      // cuCtxSetCacheConfig, cudaDeviceSetCacheConfig
};

struct Stream : FrontObj {
  unsigned flags;
  int priority;
  Ctx *ctx;
};

struct Event : FrontObj {
  unsigned flags;
  Ctx *ctx;
  // Recorded before the last restart, and not since: before_ref ms before
  // the last checkpoint's reference event (front_events_quiesce()).
  bool time_saved;
  float before_ref;
  // By the checkpoint in progress: 1 saved, 2 moved to its reference (the
  // value for a resume is resume_before_ref).
  int ckpt_state;
  float resume_before_ref;
};

struct Pinned {
  size_t size;
  unsigned flags;        // cudaHostRegister/cuMemHostRegister flags
  bool ours;             // cudaHostAlloc/cuMemHostAlloc: the front mapped it
  Ctx *ctx;              // current at registration
  bool managed;          // cudaMallocManaged/cuMemAllocManaged (ours too)
};

// Protects the records and the lists.  Never held across the gate.
extern std::mutex objs_lock FRONT_HIDDEN;
extern FrontObj *obj_heads[FRONT_N_KINDS] FRONT_HIDDEN;
extern std::map<void *, Pinned> pinned FRONT_HIDDEN;

// The file mapped at 'addr' (/proc/self/maps).
bool front_mapped_file(const void *addr, std::string *path) FRONT_HIDDEN;

void obj_link(FrontObj *obj, FrontKind kind) FRONT_HIDDEN;
void obj_unlink(FrontObj *obj) FRONT_HIDDEN;

// The checkpoint thread uses cudart's context of 'device'.
void front_use_device(int device) FRONT_HIDDEN;
// cudaSetDeviceFlags, cuDevicePrimaryCtxSetFlags: set again at restart.
void front_set_device_flags(int device, unsigned flags) FRONT_HIDDEN;
// cuCtxEnablePeerAccess, cuCtxDisablePeerAccess (front_objs.cpp).
void front_peer_access(int device, int peer, unsigned flags, bool on)
  FRONT_HIDDEN;

// Contexts (front_drv.cpp).  These call into the lower half: never with
// objs_lock held.
//
// The record of a device's primary context, made on first use.
Ctx *front_ctx_primary(int device) FRONT_HIDDEN;
// The record of a real context, or NULL (not with objs_lock held).
Ctx *front_ctx_of_real(CUcontext real) FRONT_HIDDEN;
// The calling thread's current context, or NULL.
Ctx *front_ctx_current() FRONT_HIDDEN;
// Makes 'ctx' current on the checkpoint thread.
void front_ctx_use(Ctx *ctx) FRONT_HIDDEN;
// The calling thread's context stack, as the driver keeps it: a copy in the
// upper half, from which a restart makes it again (front_core.cpp).
void front_stack_set_top(Ctx *ctx) FRONT_HIDDEN;
void front_stack_push(Ctx *ctx) FRONT_HIDDEN;
void front_stack_pop() FRONT_HIDDEN;
void front_stack_restore(FrontThread *t) FRONT_HIDDEN;
// At the checkpoint, for another thread outside the lower half: the top of
// its copy becomes what is current there now.  The MPI's own calls on that
// thread's TLS may have changed it.
void front_stack_sync(FrontThread *t) FRONT_HIDDEN;

// Shared by cudaStreamCreate* and cuStreamCreate*, ...: the result is a
// cudaError_t, or a CUresult with 'drv'.
int front_create_stream(cudaStream_t *stream, unsigned flags, int priority,
                        bool drv) FRONT_HIDDEN;
int front_destroy_stream(cudaStream_t stream, bool drv) FRONT_HIDDEN;
int front_create_event(cudaEvent_t *event, unsigned flags, bool drv)
  FRONT_HIDDEN;
int front_destroy_event(cudaEvent_t event, bool drv) FRONT_HIDDEN;
int front_host_alloc(void **ptr, size_t size, unsigned register_flags,
                     bool drv) FRONT_HIDDEN;
int front_host_free(void *ptr, bool drv) FRONT_HIDDEN;
// Managed memory (cudaMallocManaged, cuMemAllocManaged): mapped pinned memory
// of the upper half, which a restart registers again.  front_managed_free()
// frees it and returns true if 'ptr' is the front's managed memory.
int front_managed_alloc(void **ptr, size_t size, bool drv) FRONT_HIDDEN;
bool front_managed_free(void *ptr, int *rc, bool drv) FRONT_HIDDEN;
// Whether [ptr, ptr + count) is in the front's managed memory.
bool front_is_managed(const void *ptr, size_t count) FRONT_HIDDEN;
int front_host_register(void *ptr, size_t size, unsigned flags, bool drv)
  FRONT_HIDDEN;
int front_host_unregister(void *ptr, bool drv) FRONT_HIDDEN;

// The driver's records (front_drv.cpp).  At restart, front_objs_restore()
// forgets the old contexts, then makes each record again in creation order.
void front_ctx_quiesce() FRONT_HIDDEN;
void front_ctx_forget() FRONT_HIDDEN;
void front_drv_restore(FrontObj *obj) FRONT_HIDDEN;  // context, module,
                                                     // function
void front_modules_stash() FRONT_HIDDEN;
void front_modules_drop_shadows() FRONT_HIDDEN;

// Arrays, texture and surface objects (front_arrays.cpp).
void front_arrays_stash() FRONT_HIDDEN;
void front_arrays_drop_shadows() FRONT_HIDDEN;
void front_arrays_restore(FrontObj *obj) FRONT_HIDDEN;  // in the replay
void front_arrays_fill() FRONT_HIDDEN;
void front_arrays_forget(Ctx *ctx) FRONT_HIDDEN;  // with objs_lock held

// The context of an object that the calling thread has just made, for its
// record; the checkpoint waits for the device's work.
Ctx *front_object_ctx() FRONT_HIDDEN;
// Whether 'stream' is a live stream of the front, or a special one.
bool front_stream_live(cudaStream_t stream) FRONT_HIDDEN;

// A library's records (front_blas.cpp, front_fft.cpp), of one kind.  At
// restart, 'restore' makes one again at its place in the replay; 'settle'
// then sets what names other records (streams), once all exist.  'forget'
// (with objs_lock held) is for a destroyed context's.
struct FrontLibRecords {
  void (*restore)(FrontObj *obj);
  void (*settle)();
  void (*forget)(Ctx *ctx);
  void (*stash)();       // or NULL: at the checkpoint, after the GPUs sync
  void (*drop)();        // or NULL: after a resume or a restart
};
void front_lib_records(FrontKind kind, const FrontLibRecords *records)
  FRONT_HIDDEN;
void front_libs_forget(Ctx *ctx) FRONT_HIDDEN;  // with objs_lock held
// Whether 'host' is the host stub of a kernel that the application
// registered (with objs_lock held).
bool front_registered_kernel(const void *host) FRONT_HIDDEN;
// Stream capture (front_graphs.cpp): the captures begun and not ended.  A
// thread holds the checkpoint gate during its capture.
extern int front_captures FRONT_HIDDEN;
// Whether 'real' (a stream of the lower half's) is capturing.
bool front_capturing(cudaStream_t real) FRONT_HIDDEN;
// A host node's function and data for the driver (front_async.cpp): a
// trampoline that hands each call to the runner, and a target that lives as
// long as the process, since a graph may run it again and keep it across a
// restart.
void front_host_node(cudaHostFn_t fn, void *user_data, void **driver_fn,
                     void **driver_data) FRONT_HIDDEN;
// The application's function and data of a host node made so; false if the
// driver's are not the front's.
bool front_host_node_target(void *driver_fn, void *driver_data,
                            cudaHostFn_t *fn, void **user_data) FRONT_HIDDEN;

// Event times (front_async.cpp).
void front_events_quiesce() FRONT_HIDDEN;
void front_events_resume() FRONT_HIDDEN;
void front_events_restart() FRONT_HIDDEN;
void front_event_restore(Event *e) FRONT_HIDDEN;

#endif  // ifndef _MANA_CUDA_FRONT_OBJS_H
