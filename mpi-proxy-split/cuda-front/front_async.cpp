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
//
// A host node of a graph runs at every launch, and the graph may outlive a
// restart: its data for the driver is a target that lives as long as the
// process, and each call puts its job on the driver thread's stack.

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
  bool on_stack;           // a host node's: not the runner's to free
};

// A host node's target (front_host_node()).
struct HostTarget {
  cudaHostFn_t fn;
  void *user_data;
};

std::mutex targets_lock;
std::map<std::pair<void *, void *>, HostTarget *> targets;

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

// A host node's: the job on this stack, filled field by field (no libc
// here).
void CUDART_CB
host_node_trampoline(void *arg)
{
  HostTarget *target = (HostTarget *)arg;
  Job job;
  job.next = NULL;
  job.kind = JOB_HOST_FN;
  job.fn = (void *)target->fn;
  job.user_data = target->user_data;
  job.stream = NULL;
  job.status = 0;
  job.state = 0;
  job.on_stack = true;
  hand_to_runner(&job);
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
      bool keep = !job->on_stack;  // the job is gone once its state is 1
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
      if (keep) {
        done.push_back(job);
      }
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

// cudaLaunchHostFunc, cuLaunchHostFunc: into a stream capture, the call
// becomes a host node, which gets a target (front_host_node()).
template <typename Launch>
int
launch_host_fn(void *fn, void *user_data, cudaStream_t stream, Launch call)
{
  if (fn != NULL && __atomic_load_n(&front_captures, __ATOMIC_ACQUIRE) > 0) {
    FRONT_ENTER();
    cudaStream_t real = front_real_stream(stream);
    bool capturing = front_capturing(real);
    int rc = 0;
    if (capturing) {
      void *driver_fn, *driver_data;
      front_host_node((cudaHostFn_t)fn, user_data, &driver_fn, &driver_data);
      rc = call(real, driver_fn, driver_data);
    }
    FRONT_LEAVE();
    if (capturing) {
      return rc;
    }
  }
  return enqueue(JOB_HOST_FN, fn, user_data, stream,
                 [&](cudaStream_t real, Job *job) {
                   return call(real, (void *)host_fn_trampoline, job);
                 });
}

// A device's default pool, or one of cudaMemPoolCreate: the allocator's
// memory is VMM memory of its own, so a pool keeps only the attributes and
// the access that the application sets, and a restart has nothing to make.
struct Pool {
  uint32_t magic;
  int device;
  bool created;                  // by cudaMemPoolCreate, not a default pool
  std::map<int, uint64_t> attrs;
  std::map<int, int> access;     // by device: cudaMemAccessFlags
};

#define POOL_MAGIC 0x706f6f6cu  // "pool"

std::mutex pools_lock;
Pool *default_pools[64];
Pool *current_pools[64];         // cudaDeviceSetMemPool; NULL: the default

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
    front_unsupported("a memory pool that the front did not make (imported, "
                      "or destroyed)");
  }
  return p;
}

// cudaMemPoolCreate, cuMemPoolCreate: pinned device memory without IPC
// handles; the driver's and cudart's enums have the same values.
int
pool_create(void **pool, int alloc_type, int handle_types, int location_type,
            int device)
{
  if (pool == NULL) {
    return cudaErrorInvalidValue;
  }
  if (alloc_type != cudaMemAllocationTypePinned ||
      location_type != cudaMemLocationTypeDevice) {
    front_unsupported("a memory pool of other than pinned device memory");
  }
  if (handle_types != cudaMemHandleTypeNone) {
    front_unsupported("a memory pool for CUDA IPC (handleTypes)");
  }
  if (device < 0 || device >= 64) {
    return cudaErrorInvalidDevice;
  }
  Pool *p = new Pool();
  p->magic = POOL_MAGIC;
  p->device = device;
  p->created = true;
  *pool = p;
  return 0;
}

int
pool_destroy(void *pool)
{
  Pool *p = pool_of(pool);
  if (!p->created) {
    return cudaErrorInvalidValue;
  }
  std::lock_guard<std::mutex> guard(pools_lock);
  if (current_pools[p->device] == p) {
    current_pools[p->device] = NULL;
  }
  p->magic = 0;  // kept: a later use is refused, not undefined
  return 0;
}

int
set_current_pool(int device, void *pool)
{
  Pool *p = pool_of(pool);
  if (device < 0 || device >= 64 || p->device != device) {
    return cudaErrorInvalidValue;
  }
  std::lock_guard<std::mutex> guard(pools_lock);
  current_pools[device] = p->created ? p : NULL;
  return 0;
}

Pool *
current_pool(int device)
{
  if (device < 0 || device >= 64) {
    return NULL;
  }
  {
    std::lock_guard<std::mutex> guard(pools_lock);
    if (current_pools[device] != NULL) {
      return current_pools[device];
    }
  }
  return default_pool(device);
}

// cudaMemPoolSetAccess, cuMemPoolSetAccess (cudaMemAccessDesc and
// CUmemAccessDesc are the same).  Access to the pool's device is the
// allocator's peer access to all of that device's memory, as
// cuCtxEnablePeerAccess gives; it only grows, since taking it back would take
// back that of cuCtxEnablePeerAccess too.
int
pool_set_access(void *pool, const cudaMemAccessDesc *descs, size_t count)
{
  Pool *p = pool_of(pool);
  if (descs == NULL && count > 0) {
    return cudaErrorInvalidValue;
  }
  for (size_t i = 0; i < count; i++) {
    if (descs[i].location.type != cudaMemLocationTypeDevice ||
        descs[i].location.id < 0 || descs[i].location.id >= 64) {
      return cudaErrorInvalidValue;
    }
  }
  for (size_t i = 0; i < count; i++) {
    int device = descs[i].location.id;
    if (device == p->device) {
      continue;
    }
    {
      std::lock_guard<std::mutex> guard(pools_lock);
      p->access[device] = descs[i].flags;
    }
    if (descs[i].flags != cudaMemAccessFlagsProtNone) {
      front_vmm_peer_access(device, p->device, true);
    }
  }
  return 0;
}

int
pool_get_access(int *flags, void *pool, const cudaMemLocation *location)
{
  Pool *p = pool_of(pool);
  if (flags == NULL || location == NULL ||
      location->type != cudaMemLocationTypeDevice) {
    return cudaErrorInvalidValue;
  }
  if (location->id == p->device) {
    *flags = cudaMemAccessFlagsProtReadWrite;
    return 0;
  }
  std::lock_guard<std::mutex> guard(pools_lock);
  auto it = p->access.find(location->id);
  *flags = it != p->access.end() ? it->second : cudaMemAccessFlagsProtNone;
  return 0;
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
// In a stream capture, it would be a graph's memory node, which the front
// does not support.
void
check_not_capturing(cudaStream_t stream)
{
  if (__atomic_load_n(&front_captures, __ATOMIC_ACQUIRE) > 0 &&
      front_capturing(front_real_stream(stream))) {
    front_fatal("stream-ordered allocation in a stream capture (a graph's "
                "memory node) is not supported\n");
  }
}

// On 'device', a pool's, or on the current device if it is -1.
int
malloc_async(void **ptr, size_t size, cudaStream_t stream, int device = -1)
{
  FRONT_ENTER();
  check_not_capturing(stream);
  int rc = front_vmm_alloc_on(ptr, size, device);
  FRONT_LEAVE();
  return rc;
}

int
free_async(void *ptr, cudaStream_t stream, bool drv)
{
  FRONT_ENTER();
  check_not_capturing(stream);
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

void
front_host_node(cudaHostFn_t fn, void *user_data, void **driver_fn,
                void **driver_data)
{
  pthread_once(&runner_once, start_runner);
  std::lock_guard<std::mutex> guard(targets_lock);
  HostTarget *&target = targets[std::make_pair((void *)fn, user_data)];
  if (target == NULL) {
    target = new HostTarget();
    target->fn = fn;
    target->user_data = user_data;
  }
  *driver_fn = (void *)host_node_trampoline;
  *driver_data = target;
}

bool
front_host_node_target(void *driver_fn, void *driver_data, cudaHostFn_t *fn,
                       void **user_data)
{
  if (driver_fn != (void *)host_node_trampoline || driver_data == NULL) {
    return false;
  }
  *fn = ((HostTarget *)driver_data)->fn;
  *user_data = ((HostTarget *)driver_data)->user_data;
  return true;
}

extern "C" cudaError_t
cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData)
{
  return (cudaError_t)launch_host_fn(
    (void *)fn, userData, stream,
    [](cudaStream_t real, void *driver_fn, void *driver_data) {
      cudaError_t rc;
      FRONT_LH(rc = REAL(cudaLaunchHostFunc)(
                 real, (cudaHostFn_t)driver_fn, driver_data));
      return (int)rc;
    });
}

extern "C" cudaError_t
cudaLaunchHostFunc_ptsz(cudaStream_t stream, cudaHostFn_t fn, void *userData)
{
  return (cudaError_t)launch_host_fn(
    (void *)fn, userData, stream,
    [](cudaStream_t real, void *driver_fn, void *driver_data) {
      cudaError_t rc;
      FRONT_LH(rc = REAL(cudaLaunchHostFunc_ptsz)(
                 real, (cudaHostFn_t)driver_fn, driver_data));
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
  return (CUresult)launch_host_fn(
    (void *)fn, userData, hStream,
    [](cudaStream_t real, void *driver_fn, void *driver_data) {
      CUresult rc;
      FRONT_LH(rc = DRV(cuLaunchHostFunc)(
                 real, (CUhostFn)driver_fn, driver_data));
      return (int)rc;
    });
}

extern "C" CUresult
cuLaunchHostFunc_ptsz(CUstream hStream, CUhostFn fn, void *userData)
{
  return (CUresult)launch_host_fn(
    (void *)fn, userData, hStream,
    [](cudaStream_t real, void *driver_fn, void *driver_data) {
      CUresult rc;
      FRONT_LH(rc = DRV(cuLaunchHostFunc_ptsz)(
                 real, (CUhostFn)driver_fn, driver_data));
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
cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream)
{
  return (cudaError_t)malloc_async(devPtr, size, stream);
}

extern "C" cudaError_t
cudaMallocAsync_ptsz(void **devPtr, size_t size, cudaStream_t stream)
{
  return (cudaError_t)malloc_async(devPtr, size, stream);
}

extern "C" cudaError_t
cudaMallocFromPoolAsync(void **ptr, size_t size, cudaMemPool_t memPool,
                        cudaStream_t stream)
{
  return (cudaError_t)malloc_async(ptr, size, stream,
                                   pool_of(memPool)->device);
}

extern "C" cudaError_t
cudaMallocFromPoolAsync_ptsz(void **ptr, size_t size, cudaMemPool_t memPool,
                             cudaStream_t stream)
{
  return (cudaError_t)malloc_async(ptr, size, stream,
                                   pool_of(memPool)->device);
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
cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream stream)
{
  return (CUresult)malloc_async((void **)dptr, bytesize, stream);
}

extern "C" CUresult
cuMemAllocAsync_ptsz(CUdeviceptr *dptr, size_t bytesize, CUstream stream)
{
  return (CUresult)malloc_async((void **)dptr, bytesize, stream);
}

extern "C" CUresult
cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize, CUmemoryPool pool,
                        CUstream stream)
{
  return (CUresult)malloc_async((void **)dptr, bytesize, stream,
                                pool_of(pool)->device);
}

extern "C" CUresult
cuMemAllocFromPoolAsync_ptsz(CUdeviceptr *dptr, size_t bytesize,
                             CUmemoryPool pool, CUstream stream)
{
  return (CUresult)malloc_async((void **)dptr, bytesize, stream,
                                pool_of(pool)->device);
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

extern "C" cudaError_t
cudaDeviceGetMemPool(cudaMemPool_t *memPool, int device)
{
  if (memPool == NULL) {
    return cudaErrorInvalidValue;
  }
  Pool *p = current_pool(device);
  if (p == NULL) {
    return cudaErrorInvalidDevice;
  }
  *memPool = (cudaMemPool_t)p;
  return cudaSuccess;
}

extern "C" cudaError_t
cudaDeviceSetMemPool(int device, cudaMemPool_t memPool)
{
  return (cudaError_t)set_current_pool(device, memPool);
}

extern "C" cudaError_t
cudaMemPoolCreate(cudaMemPool_t *memPool, const struct cudaMemPoolProps *props)
{
  if (props == NULL) {
    return cudaErrorInvalidValue;
  }
  return (cudaError_t)pool_create((void **)memPool, props->allocType,
                                  props->handleTypes, props->location.type,
                                  props->location.id);
}

extern "C" cudaError_t
cudaMemPoolDestroy(cudaMemPool_t memPool)
{
  return (cudaError_t)pool_destroy(memPool);
}

extern "C" cudaError_t
cudaMemPoolSetAccess(cudaMemPool_t memPool,
                     const struct cudaMemAccessDesc *descList, size_t count)
{
  return (cudaError_t)pool_set_access(memPool, descList, count);
}

extern "C" cudaError_t
cudaMemPoolGetAccess(enum cudaMemAccessFlags *flags, cudaMemPool_t memPool,
                     struct cudaMemLocation *location)
{
  return (cudaError_t)pool_get_access((int *)flags, memPool, location);
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
  return cudaDeviceGetMemPool((cudaMemPool_t *)pool, dev) == cudaSuccess
         ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

extern "C" CUresult
cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool)
{
  return set_current_pool(dev, pool) == 0 ? CUDA_SUCCESS
                                          : CUDA_ERROR_INVALID_VALUE;
}

extern "C" CUresult
cuMemPoolCreate(CUmemoryPool *pool, const CUmemPoolProps *poolProps)
{
  if (poolProps == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int rc = pool_create((void **)pool, poolProps->allocType,
                       poolProps->handleTypes, poolProps->location.type,
                       poolProps->location.id);
  return rc == 0 ? CUDA_SUCCESS : rc == cudaErrorInvalidDevice
                                    ? CUDA_ERROR_INVALID_DEVICE
                                    : CUDA_ERROR_INVALID_VALUE;
}

extern "C" CUresult
cuMemPoolDestroy(CUmemoryPool pool)
{
  return pool_destroy(pool) == 0 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

extern "C" CUresult
cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc *map,
                   size_t count)
{
  return pool_set_access(pool, (const cudaMemAccessDesc *)map, count) == 0
         ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

extern "C" CUresult
cuMemPoolGetAccess(CUmemAccess_flags *flags, CUmemoryPool memPool,
                   CUmemLocation *location)
{
  return pool_get_access((int *)flags, memPool,
                         (const cudaMemLocation *)location) == 0
         ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
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
