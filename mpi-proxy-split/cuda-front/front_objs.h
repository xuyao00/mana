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
};

// Protects the records and the lists.  Never held across the gate.
extern std::mutex objs_lock FRONT_HIDDEN;
extern FrontObj *obj_heads[FRONT_N_KINDS] FRONT_HIDDEN;
extern std::map<void *, Pinned> pinned FRONT_HIDDEN;

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

// Event times (front_async.cpp).
void front_events_quiesce() FRONT_HIDDEN;
void front_events_resume() FRONT_HIDDEN;
void front_events_restart() FRONT_HIDDEN;
void front_event_restore(Event *e) FRONT_HIDDEN;

#endif  // ifndef _MANA_CUDA_FRONT_OBJS_H
