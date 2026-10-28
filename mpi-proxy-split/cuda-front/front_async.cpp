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

// Work that the driver finishes later, in both APIs: host functions and
// stream callbacks, stream-ordered allocation (with the device's default
// pool), and event times across a restart.
//
// Host functions.  The driver calls them on a thread of its own, in the
// lower half, where the upper half's TLS (errno, malloc, ...) is not there.
// So the driver calls a trampoline, which hands the call to the runner, an
// upper-half thread, and waits.  The trampoline runs with the lower half's
// FS: it uses no libc, only atomics and raw futex system calls.  The
// checkpoint's GPU synchronization waits for host functions too; the runner
// is an ordinary thread then, and does not take the checkpoint gate.

#include <linux/futex.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include "front_objs.h"
#include "front_reg.h"

namespace {

enum JobKind { JOB_HOST_FN, JOB_RUNTIME_CALLBACK, JOB_DRIVER_CALLBACK };

struct Job {
  Job *next;
  int kind;
  void *fn;
  void *user_data;
  cudaStream_t stream;     // virtual, for the stream callbacks
  int status;
  int state;               // 0 queued, 1 run (by the runner), 2 returned
};

Job *queue;                // pushed by the trampolines, newest first
int queue_seq;             // futex word: bumped at each push
pthread_once_t runner_once = PTHREAD_ONCE_INIT;

long
raw_futex(int *addr, int op, int val)
{
  long ret;
  register long timeout asm("r10") = 0;
  asm volatile("syscall"
               : "=a"(ret)
               : "0"((long)SYS_futex), "D"(addr), "S"((long)op),
                 "d"((long)val), "r"(timeout)
               : "rcx", "r11", "memory");
  return ret;
}

// On a driver thread, with the lower half's FS.
void
hand_to_runner(Job *job)
{
  Job *head = __atomic_load_n(&queue, __ATOMIC_RELAXED);
  do {
    job->next = head;
  } while (!__atomic_compare_exchange_n(&queue, &head, job, true,
                                        __ATOMIC_RELEASE, __ATOMIC_RELAXED));
  __atomic_add_fetch(&queue_seq, 1, __ATOMIC_RELEASE);
  raw_futex(&queue_seq, FUTEX_WAKE_PRIVATE, 1);
  while (__atomic_load_n(&job->state, __ATOMIC_ACQUIRE) == 0) {
    raw_futex(&job->state, FUTEX_WAIT_PRIVATE, 0);
  }
  // The runner may free the job from now on.
  __atomic_store_n(&job->state, 2, __ATOMIC_RELEASE);
}

void CUDART_CB
host_fn_trampoline(void *arg)
{
  hand_to_runner((Job *)arg);
}

void CUDART_CB
runtime_callback_trampoline(cudaStream_t, cudaError_t status, void *arg)
{
  ((Job *)arg)->status = (int)status;
  hand_to_runner((Job *)arg);
}

void CUDA_CB
driver_callback_trampoline(CUstream, CUresult status, void *arg)
{
  ((Job *)arg)->status = (int)status;
  hand_to_runner((Job *)arg);
}

void *
runner_main(void *)
{
  std::vector<Job *> done;  // until their trampolines return
  for (;;) {
    int seq = __atomic_load_n(&queue_seq, __ATOMIC_ACQUIRE);
    Job *list = __atomic_exchange_n(&queue, (Job *)NULL, __ATOMIC_ACQ_REL);
    if (list == NULL) {
      done.erase(std::remove_if(done.begin(), done.end(),
                                [](Job *job) {
                                  if (__atomic_load_n(&job->state,
                                                      __ATOMIC_ACQUIRE) != 2) {
                                    return false;
                                  }
                                  delete job;
                                  return true;
                                }),
                 done.end());
      syscall(SYS_futex, &queue_seq, FUTEX_WAIT_PRIVATE, seq, NULL, NULL, 0);
      continue;
    }
    std::vector<Job *> jobs;
    for (Job *job = list; job != NULL; job = job->next) {
      jobs.push_back(job);
    }
    for (auto it = jobs.rbegin(); it != jobs.rend(); ++it) {
      Job *job = *it;
      switch (job->kind) {
        case JOB_HOST_FN:
          ((cudaHostFn_t)job->fn)(job->user_data);
          break;
        case JOB_RUNTIME_CALLBACK:
          ((cudaStreamCallback_t)job->fn)(job->stream,
                                          (cudaError_t)job->status,
                                          job->user_data);
          break;
        case JOB_DRIVER_CALLBACK:
          ((CUstreamCallback)job->fn)((CUstream)job->stream,
                                      (CUresult)job->status, job->user_data);
          break;
      }
      __atomic_store_n(&job->state, 1, __ATOMIC_RELEASE);
      syscall(SYS_futex, &job->state, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
      done.push_back(job);
    }
  }
  return NULL;
}

void
start_runner()
{
  pthread_t thread;
  if (pthread_create(&thread, NULL, runner_main, NULL) != 0) {
    front_fatal("cannot start the thread for host functions\n");
  }
  pthread_detach(thread);
}

Job *
new_job(int kind, void *fn, void *user_data, cudaStream_t stream)
{
  pthread_once(&runner_once, start_runner);
  Job *job = new Job();
  job->kind = kind;
  job->fn = fn;
  job->user_data = user_data;
  job->stream = stream;
  return job;
}

// The function and the trampoline for a host function (fn) or a stream
// callback, of the runtime (cudaLaunchHostFunc, cudaStreamAddCallback) or
// the driver.
template <typename Enqueue>
int
enqueue(int kind, void *fn, void *user_data, cudaStream_t stream,
        Enqueue call)
{
  if (fn == NULL) {
    return cudaErrorInvalidValue;  // CUDA_ERROR_INVALID_VALUE too
  }
  Job *job = new_job(kind, fn, user_data, stream);
  FRONT_ENTER();
  int rc = call(front_real_stream(stream), job);
  FRONT_LEAVE();
  if (rc != 0) {
    delete job;
  }
  return rc;
}

// The device's default pool: the allocator's memory is VMM memory of its
// own, so a pool keeps only the attributes that the application sets.
struct Pool {
  uint32_t magic;
  int device;
  std::map<int, uint64_t> attrs;
};

#define POOL_MAGIC 0x706f6f6cu  // "pool"

std::mutex pools_lock;
Pool *default_pools[64];

Pool *
default_pool(int device)
{
  if (device < 0 || device >= 64) {
    return NULL;
  }
  std::lock_guard<std::mutex> guard(pools_lock);
  if (default_pools[device] == NULL) {
    default_pools[device] = new Pool();
    default_pools[device]->magic = POOL_MAGIC;
    default_pools[device]->device = device;
  }
  return default_pools[device];
}

Pool *
pool_of(void *pool)
{
  Pool *p = (Pool *)pool;
  if (p == NULL || p->magic != POOL_MAGIC) {
    front_unsupported("memory pools other than a device's default pool");
  }
  return p;
}

// cudaMemPoolAttr and CUmemPool_attribute have the same values: 1 to 3 are
// ints, the others 64-bit sizes.
bool
pool_attr_is_int(int attr)
{
  return attr >= 1 && attr <= 3;
}

int
pool_set(void *pool, int attr, void *value)
{
  if (value == NULL) {
    return cudaErrorInvalidValue;
  }
  Pool *p = pool_of(pool);
  std::lock_guard<std::mutex> guard(pools_lock);
  p->attrs[attr] = pool_attr_is_int(attr) ? *(int *)value
                                          : *(uint64_t *)value;
  return 0;
}

int
pool_get(void *pool, int attr, void *value)
{
  if (value == NULL) {
    return cudaErrorInvalidValue;
  }
  Pool *p = pool_of(pool);
  std::lock_guard<std::mutex> guard(pools_lock);
  auto it = p->attrs.find(attr);
  uint64_t v = it != p->attrs.end() ? it->second : 0;
  if (pool_attr_is_int(attr)) {
    *(int *)value = (int)v;
  } else {
    *(uint64_t *)value = v;
  }
  return 0;
}

// Stream-ordered allocation: the memory is the allocator's at once, which
// is valid in stream order too.  A free waits for the stream's work so far,
// through an event, before the memory is used again.
int
malloc_async(void **ptr, size_t size)
{
  FRONT_ENTER();
  int rc = front_vmm_alloc(ptr, size);
  FRONT_LEAVE();
  return rc;
}

int
free_async(void *ptr, cudaStream_t stream, bool drv)
{
  FRONT_ENTER();
  int rc = 0;
  if (!front_vmm_free_async(ptr, front_real_stream(stream), &rc)) {
    if (drv) {
      FRONT_LH(rc = DRV(cuMemFreeAsync)((CUdeviceptr)ptr,
                                        front_real_stream(stream)));
    } else {
      FRONT_LH(rc = REAL(cudaFreeAsync)(ptr, front_real_stream(stream)));
    }
  }
  FRONT_LEAVE();
  return rc;
}

Event *
event_of(cudaEvent_t event)
{
  return (Event *)event;
}

// Recording an event again makes its time real again.
template <typename Record>
int
record_event(cudaEvent_t event, Record call)
{
  FRONT_ENTER();
  int rc = call(front_real_event(event));
  if (rc == 0 && event != NULL) {
    event_of(event)->time_saved = false;
    event_of(event)->ckpt_state = 0;
  }
  FRONT_LEAVE();
  return rc;
}

// See front_events_quiesce().
template <typename Elapsed>
int
elapsed_time(float *ms, cudaEvent_t start, cudaEvent_t end, Elapsed call)
{
  if (ms == NULL || start == NULL || end == NULL) {
    return cudaErrorInvalidValue;
  }
  Event *s = event_of(start);
  Event *e = event_of(end);
  // Within the gate: a checkpoint changes the saved times.
  FRONT_ENTER();
  int rc = 0;
  if (s->time_saved && e->time_saved) {
    *ms = s->before_ref - e->before_ref;
  } else {
    rc = call(ms, front_real_event(start), front_real_event(end));
    // A restored event stands where the restart recorded it: as if the
    // checkpoint took no time.
    if (rc == 0 && s->time_saved) {
      *ms += s->before_ref;
    } else if (rc == 0 && e->time_saved) {
      *ms -= e->before_ref;
    }
  }
  FRONT_LEAVE();
  return rc;
}

}  // namespace

// At restart: per context, an event recorded before the restored ones,
// which stand where the restart recorded them.
std::map<Ctx *, CUevent> restore_refs;
unsigned restore_refs_gen;

// At restart, an event again: recorded once, so that waits and queries
// behave as before the checkpoint (an event never recorded completes at
// once, as a recorded one does after the checkpoint's synchronization).
void
front_event_restore(Event *e)
{
  CUresult rc;
  FRONT_LH(rc = DRV(cuEventCreate)((CUevent *)&e->real, e->flags));
  if (rc == CUDA_SUCCESS && !(e->flags & cudaEventDisableTiming)) {
    if (restore_refs_gen != front_gen) {
      restore_refs.clear();
      restore_refs_gen = front_gen;
    }
    CUevent &ref = restore_refs[e->ctx];
    if (ref == NULL) {
      FRONT_LH(rc = DRV(cuEventCreate)(&ref, CU_EVENT_DEFAULT));
      if (rc == CUDA_SUCCESS) {
        FRONT_LH(rc = DRV(cuEventRecord)(ref, NULL));
      }
    }
  }
  if (rc == CUDA_SUCCESS) {
    FRONT_LH(rc = DRV(cuEventRecord)((CUevent)e->real, NULL));
  }
  if (rc != CUDA_SUCCESS) {
    front_fatal("creating an event again: error %d\n", rc);
  }
}

// At the checkpoint, after the GPUs are synchronized: each event's time
// before a reference event of its context, for cudaEventElapsedTime() after
// a restart.  An event restored by an earlier restart keeps its time, moved
// to this reference by the time since that restart's reference (all events
// of a context by the same amount), as if the checkpoint took no time.
void
front_events_quiesce()
{
  std::map<Ctx *, CUevent> refs;
  std::map<Ctx *, float> since_restore;
  for (FrontObj *o = obj_heads[FRONT_EVENT]; o != NULL; o = o->next) {
    Event *e = (Event *)o;
    if (e->flags & cudaEventDisableTiming) {
      continue;
    }
    CUresult rc;
    CUevent &ref = refs[e->ctx];
    if (ref == NULL) {
      front_ctx_use(e->ctx);
      FRONT_LH(rc = DRV(cuEventCreate)(&ref, CU_EVENT_DEFAULT));
      if (rc == CUDA_SUCCESS) {
        FRONT_LH(rc = DRV(cuEventRecord)(ref, NULL));
      }
      if (rc == CUDA_SUCCESS) {
        FRONT_LH(rc = DRV(cuEventSynchronize)(ref));
      }
      if (rc != CUDA_SUCCESS) {
        front_fatal("making a reference event: error %d\n", rc);
      }
    }
    float ms = 0;
    auto restored = restore_refs.find(e->ctx);
    if (e->time_saved && restore_refs_gen == front_gen &&
        restored != restore_refs.end()) {
      auto known = since_restore.find(e->ctx);
      if (known == since_restore.end()) {
        FRONT_LH(rc = DRV(cuEventElapsedTime_v2)(&ms, restored->second, ref));
        if (rc != CUDA_SUCCESS) {
          front_fatal("timing the restart's reference event: error %d\n",
                      rc);
        }
        known = since_restore.emplace(e->ctx, ms).first;
      }
      ms = known->second;
      rc = CUDA_SUCCESS;
    } else {
      FRONT_LH(rc = DRV(cuEventElapsedTime_v2)(&ms, (CUevent)e->real, ref));
    }
    front_log("event %p: %d, %f ms before the reference%s\n", (void *)e, rc,
              ms, e->time_saved ? " (restored)" : "");
    if (rc != CUDA_SUCCESS) {
      continue;  // not recorded yet: no time
    }
    if (e->time_saved) {
      e->resume_before_ref = e->before_ref;
      e->before_ref += ms;
      e->ckpt_state = 2;
    } else {
      e->before_ref = ms;
      e->time_saved = true;
      e->ckpt_state = 1;
    }
  }
  for (auto &entry : refs) {
    FRONT_LH(DRV(cuEventDestroy_v2)(entry.second));
  }
}

// After a resume, the events are the same ones: what the checkpoint saved
// or moved goes back.
void
front_events_resume()
{
  for (FrontObj *o = obj_heads[FRONT_EVENT]; o != NULL; o = o->next) {
    Event *e = (Event *)o;
    if (e->ckpt_state == 1) {
      e->time_saved = false;
    } else if (e->ckpt_state == 2) {
      e->before_ref = e->resume_before_ref;
    }
    e->ckpt_state = 0;
  }
}

// After a restart, what the checkpoint saved stays.
void
front_events_restart()
{
  for (FrontObj *o = obj_heads[FRONT_EVENT]; o != NULL; o = o->next) {
    ((Event *)o)->ckpt_state = 0;
  }
}

// ---- Host functions and stream callbacks ---------------------------------

extern "C" cudaError_t
cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData)
{
  return (cudaError_t)enqueue(
    JOB_HOST_FN, (void *)fn, userData, stream,
    [](cudaStream_t real, Job *job) {
      cudaError_t rc;
      FRONT_LH(rc = REAL(cudaLaunchHostFunc)(real, host_fn_trampoline, job));
      return (int)rc;
    });
}

extern "C" cudaError_t
cudaLaunchHostFunc_ptsz(cudaStream_t stream, cudaHostFn_t fn, void *userData)
{
  return (cudaError_t)enqueue(
    JOB_HOST_FN, (void *)fn, userData, stream,
    [](cudaStream_t real, Job *job) {
      cudaError_t rc;
      FRONT_LH(rc = REAL(cudaLaunchHostFunc_ptsz)(real, host_fn_trampoline,
                                                  job));
      return (int)rc;
    });
}

extern "C" cudaError_t
cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback,
                      void *userData, unsigned int flags)
{
  return (cudaError_t)enqueue(
    JOB_RUNTIME_CALLBACK, (void *)callback, userData, stream,
    [flags](cudaStream_t real, Job *job) {
      cudaError_t rc;
      FRONT_LH(rc = REAL(cudaStreamAddCallback)(
                 real, runtime_callback_trampoline, job, flags));
      return (int)rc;
    });
}

extern "C" cudaError_t
cudaStreamAddCallback_ptsz(cudaStream_t stream, cudaStreamCallback_t callback,
                           void *userData, unsigned int flags)
{
  return (cudaError_t)enqueue(
    JOB_RUNTIME_CALLBACK, (void *)callback, userData, stream,
    [flags](cudaStream_t real, Job *job) {
      cudaError_t rc;
      FRONT_LH(rc = REAL(cudaStreamAddCallback_ptsz)(
                 real, runtime_callback_trampoline, job, flags));
      return (int)rc;
    });
}

extern "C" CUresult
cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void *userData)
{
  return (CUresult)enqueue(
    JOB_HOST_FN, (void *)fn, userData, hStream,
    [](cudaStream_t real, Job *job) {
      CUresult rc;
      FRONT_LH(rc = DRV(cuLaunchHostFunc)(real, host_fn_trampoline, job));
      return (int)rc;
    });
}

extern "C" CUresult
cuLaunchHostFunc_ptsz(CUstream hStream, CUhostFn fn, void *userData)
{
  return (CUresult)enqueue(
    JOB_HOST_FN, (void *)fn, userData, hStream,
    [](cudaStream_t real, Job *job) {
      CUresult rc;
      FRONT_LH(rc = DRV(cuLaunchHostFunc_ptsz)(real, host_fn_trampoline,
                                               job));
      return (int)rc;
    });
}

extern "C" CUresult
cuStreamAddCallback(CUstream hStream, CUstreamCallback callback,
                    void *userData, unsigned int flags)
{
  return (CUresult)enqueue(
    JOB_DRIVER_CALLBACK, (void *)callback, userData, hStream,
    [flags](cudaStream_t real, Job *job) {
      CUresult rc;
      FRONT_LH(rc = DRV(cuStreamAddCallback)(
                 real, driver_callback_trampoline, job, flags));
      return (int)rc;
    });
}

extern "C" CUresult
cuStreamAddCallback_ptsz(CUstream hStream, CUstreamCallback callback,
                         void *userData, unsigned int flags)
{
  return (CUresult)enqueue(
    JOB_DRIVER_CALLBACK, (void *)callback, userData, hStream,
    [flags](cudaStream_t real, Job *job) {
      CUresult rc;
      FRONT_LH(rc = DRV(cuStreamAddCallback_ptsz)(
                 real, driver_callback_trampoline, job, flags));
      return (int)rc;
    });
}

// ---- Stream-ordered allocation and default pools -------------------------

extern "C" cudaError_t
cudaMallocAsync(void **devPtr, size_t size, cudaStream_t)
{
  return (cudaError_t)malloc_async(devPtr, size);
}

extern "C" cudaError_t
cudaMallocAsync_ptsz(void **devPtr, size_t size, cudaStream_t)
{
  return (cudaError_t)malloc_async(devPtr, size);
}

extern "C" cudaError_t
cudaMallocFromPoolAsync(void **ptr, size_t size, cudaMemPool_t memPool,
                        cudaStream_t)
{
  pool_of(memPool);
  return (cudaError_t)malloc_async(ptr, size);
}

extern "C" cudaError_t
cudaMallocFromPoolAsync_ptsz(void **ptr, size_t size, cudaMemPool_t memPool,
                             cudaStream_t)
{
  pool_of(memPool);
  return (cudaError_t)malloc_async(ptr, size);
}

extern "C" cudaError_t
cudaFreeAsync(void *devPtr, cudaStream_t hStream)
{
  return (cudaError_t)free_async(devPtr, hStream, false);
}

extern "C" cudaError_t
cudaFreeAsync_ptsz(void *devPtr, cudaStream_t hStream)
{
  return (cudaError_t)free_async(devPtr, hStream, false);
}

extern "C" CUresult
cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream)
{
  return (CUresult)malloc_async((void **)dptr, bytesize);
}

extern "C" CUresult
cuMemAllocAsync_ptsz(CUdeviceptr *dptr, size_t bytesize, CUstream)
{
  return (CUresult)malloc_async((void **)dptr, bytesize);
}

extern "C" CUresult
cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool,
                        CUstream)
{
  pool_of(pool);
  return (CUresult)malloc_async((void **)dptr, bytesize);
}

extern "C" CUresult
cuMemAllocFromPoolAsync_ptsz(CUdeviceptr *dptr, size_t bytesize,
                             CUmemoryPool pool, CUstream)
{
  pool_of(pool);
  return (CUresult)malloc_async((void **)dptr, bytesize);
}

extern "C" CUresult
cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream)
{
  return (CUresult)free_async((void *)dptr, hStream, true);
}

extern "C" CUresult
cuMemFreeAsync_ptsz(CUdeviceptr dptr, CUstream hStream)
{
  return (CUresult)free_async((void *)dptr, hStream, true);
}

extern "C" cudaError_t
cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool, int device)
{
  if (memPool == NULL) {
    return cudaErrorInvalidValue;
  }
  Pool *p = default_pool(device);
  if (p == NULL) {
    return cudaErrorInvalidDevice;
  }
  *memPool = (cudaMemPool_t)p;
  return cudaSuccess;
}

// The current pool is the default one: cudaDeviceSetMemPool is not
// supported.
extern "C" cudaError_t
cudaDeviceGetMemPool(cudaMemPool_t *memPool, int device)
{
  return cudaDeviceGetDefaultMemPool(memPool, device);
}

extern "C" cudaError_t
cudaMemPoolSetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr,
                        void *value)
{
  return (cudaError_t)pool_set(memPool, (int)attr, value);
}

extern "C" cudaError_t
cudaMemPoolGetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr,
                        void *value)
{
  return (cudaError_t)pool_get(memPool, (int)attr, value);
}

extern "C" cudaError_t
cudaMemPoolTrimTo(cudaMemPool_t memPool, size_t)
{
  pool_of(memPool);
  return cudaSuccess;
}

extern "C" CUresult
cuDeviceGetDefaultMemPool(CUmemoryPool *pool_out, CUdevice dev)
{
  return cudaDeviceGetDefaultMemPool((cudaMemPool_t *)pool_out, dev) ==
         cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

extern "C" CUresult
cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev)
{
  return cuDeviceGetDefaultMemPool(pool, dev);
}

extern "C" CUresult
cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr,
                      void *value)
{
  return (CUresult)pool_set(pool, (int)attr, value);
}

extern "C" CUresult
cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr,
                      void *value)
{
  return (CUresult)pool_get(pool, (int)attr, value);
}

extern "C" CUresult
cuMemPoolTrimTo(CUmemoryPool pool, size_t)
{
  pool_of(pool);
  return CUDA_SUCCESS;
}

// ---- Event times ---------------------------------------------------------

extern "C" cudaError_t
cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
  return (cudaError_t)record_event(event, [stream](cudaEvent_t real) {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaEventRecord)(real, front_real_stream(stream)));
    return (int)rc;
  });
}

extern "C" cudaError_t
cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream)
{
  return (cudaError_t)record_event(event, [stream](cudaEvent_t real) {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaEventRecord_ptsz)(real,
                                             front_real_stream(stream)));
    return (int)rc;
  });
}

extern "C" cudaError_t
cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream,
                         unsigned int flags)
{
  return (cudaError_t)record_event(event, [stream, flags](cudaEvent_t real) {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaEventRecordWithFlags)(
               real, front_real_stream(stream), flags));
    return (int)rc;
  });
}

extern "C" cudaError_t
cudaEventRecordWithFlags_ptsz(cudaEvent_t event, cudaStream_t stream,
                              unsigned int flags)
{
  return (cudaError_t)record_event(event, [stream, flags](cudaEvent_t real) {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaEventRecordWithFlags_ptsz)(
               real, front_real_stream(stream), flags));
    return (int)rc;
  });
}

extern "C" CUresult
cuEventRecord(CUevent hEvent, CUstream hStream)
{
  return (CUresult)record_event(hEvent, [hStream](cudaEvent_t real) {
    CUresult rc;
    FRONT_LH(rc = DRV(cuEventRecord)(real, front_real_stream(hStream)));
    return (int)rc;
  });
}

extern "C" CUresult
cuEventRecord_ptsz(CUevent hEvent, CUstream hStream)
{
  return (CUresult)record_event(hEvent, [hStream](cudaEvent_t real) {
    CUresult rc;
    FRONT_LH(rc = DRV(cuEventRecord_ptsz)(real, front_real_stream(hStream)));
    return (int)rc;
  });
}

extern "C" CUresult
cuEventRecordWithFlags(CUevent hEvent, CUstream hStream, unsigned int flags)
{
  return (CUresult)record_event(hEvent, [hStream, flags](cudaEvent_t real) {
    CUresult rc;
    FRONT_LH(rc = DRV(cuEventRecordWithFlags)(
               real, front_real_stream(hStream), flags));
    return (int)rc;
  });
}

extern "C" CUresult
cuEventRecordWithFlags_ptsz(CUevent hEvent, CUstream hStream,
                            unsigned int flags)
{
  return (CUresult)record_event(hEvent, [hStream, flags](cudaEvent_t real) {
    CUresult rc;
    FRONT_LH(rc = DRV(cuEventRecordWithFlags_ptsz)(
               real, front_real_stream(hStream), flags));
    return (int)rc;
  });
}

extern "C" cudaError_t
cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
  return (cudaError_t)elapsed_time(
    ms, start, end, [](float *out, cudaEvent_t s, cudaEvent_t e) {
      cudaError_t rc;
      FRONT_LH(rc = REAL(cudaEventElapsedTime)(out, s, e));
      return (int)rc;
    });
}

extern "C" CUresult
cuEventElapsedTime_v2(float *pMilliseconds, CUevent hStart, CUevent hEnd)
{
  return (CUresult)elapsed_time(
    pMilliseconds, hStart, hEnd, [](float *out, cudaEvent_t s, cudaEvent_t e) {
      CUresult rc;
      FRONT_LH(rc = DRV(cuEventElapsedTime_v2)(out, s, e));
      return (int)rc;
    });
}
