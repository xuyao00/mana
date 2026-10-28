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

// Synchronous copies and sets, and the MPI (as in cuda-aware-pr's
// vmm-alloc.c).  A cudaMemcpy between device buffers or from pageable
// memory, or a cudaMemset, may return before it is done.  For cudaMalloc
// memory, the MPI prevents that with CU_POINTER_ATTRIBUTE_SYNC_MEMOPS
// (libfabric sets it on the buffers that it registers), but the driver
// refuses that attribute for VMM memory.  Only the MPI accesses device
// memory outside CUDA's stream order (a NIC, or a peer GPU through CUDA
// IPC), and only in an MPI call.  So the application's copy or set records
// an event on its default stream, and the next MPI call first waits for the
// pending events (LOWER_HALF_DISABLE_CKPT(), lower_half_ckpt.h).  The MPI's
// own copies are its business, in the lower half.

#include "front_objs.h"

// libmana.so (lower_half_ckpt.cpp): nonzero while events are pending.
extern int mana_cuda_memops_pending;

// A thread's events, after its last copy on the legacy (0) and on the
// per-thread (1) default stream.  Events of the lower half: a checkpoint
// waits for them, and a restart forgets them (gen).
struct MemopEvents {
  CUcontext ctx;
  CUevent event[2];
  bool pending[2];
  unsigned gen;
};

static std::mutex memops_lock;
static std::vector<MemopEvents *> all_memops;

void
front_order_memop(bool ptds)
{
  // Not into a stream capture (the per-thread stream can be captured): the
  // event would be the graph's.
  if (ptds && __atomic_load_n(&front_captures, __ATOMIC_ACQUIRE) > 0 &&
      front_capturing(cudaStreamPerThread)) {
    return;
  }
  FrontThread *t = &front_thread;
  CUcontext ctx = NULL;
  CUresult rc;
  FRONT_LH(rc = DRV(cuCtxGetCurrent)(&ctx));
  if (rc != CUDA_SUCCESS || ctx == NULL) {
    return;
  }
  MemopEvents *m = t->memops;
  if (m == NULL) {
    m = t->memops = new MemopEvents();
    std::lock_guard<std::mutex> guard(memops_lock);
    all_memops.push_back(m);
  }
  int k = ptds ? 1 : 0;
  std::lock_guard<std::mutex> guard(memops_lock);
  if (m->gen != front_gen || m->ctx != ctx) {
    // The first copy, or one in another context: events of this one.  Wait
    // for the old ones first.  They are not destroyed: a drain may be
    // waiting for them.
    for (int j = 0; j < 2; j++) {
      if (m->gen == front_gen && m->pending[j]) {
        FRONT_LH(DRV(cuEventSynchronize)(m->event[j]));
      }
      m->event[j] = NULL;
      m->pending[j] = false;
      FRONT_LH(rc = DRV(cuEventCreate)(&m->event[j],
                                       CU_EVENT_DISABLE_TIMING));
      if (rc != CUDA_SUCCESS) {
        front_fatal("creating an event for the order of copies: error %d\n",
                    rc);
      }
    }
    m->ctx = ctx;
    m->gen = front_gen;
  }
  CUstream stream = ptds ? CU_STREAM_PER_THREAD : CU_STREAM_LEGACY;
  FRONT_LH(rc = DRV(cuEventRecord)(m->event[k], stream));
  if (rc != CUDA_SUCCESS) {
    // No event: wait now.
    FRONT_LH(DRV(cuStreamSynchronize)(stream));
    return;
  }
  m->pending[k] = true;
  __atomic_store_n(&mana_cuda_memops_pending, 1, __ATOMIC_RELEASE);
}

void
front_drain_memops()
{
  std::vector<CUevent> events;
  {
    std::lock_guard<std::mutex> guard(memops_lock);
    __atomic_store_n(&mana_cuda_memops_pending, 0, __ATOMIC_RELEASE);
    for (MemopEvents *m : all_memops) {
      for (int k = 0; k < 2; k++) {
        if (m->pending[k] && m->gen == front_gen) {
          events.push_back(m->event[k]);
        }
        m->pending[k] = false;
      }
    }
  }
  // An error shows at the application's next CUDA call.
  for (CUevent event : events) {
    FRONT_LH(DRV(cuEventSynchronize)(event));
  }
}

// cuCtxDestroy: the context's events go with it (the destroy waits for its
// work), so no drain waits for them.
void
front_memops_forget(CUcontext ctx)
{
  std::lock_guard<std::mutex> guard(memops_lock);
  for (MemopEvents *m : all_memops) {
    if (m->ctx == ctx) {
      m->ctx = NULL;
      m->gen = 0;
      m->pending[0] = m->pending[1] = false;
      m->event[0] = m->event[1] = NULL;
    }
  }
}
