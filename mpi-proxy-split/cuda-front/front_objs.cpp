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

// The front's hand-written functions ("own" in cudart-own.txt), and the
// records from which a restart rebuilds what was alive at the checkpoint:
// one record per live object, never a history.  The kernel registrations
// that the application's constructors make are the one fixed set of calls
// that a restart makes again.

#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <map>
#include <mutex>
#include <vector>
#include "front.h"

// cudart's registration interface (crt/host_runtime.h, crt/device_functions.h).
extern "C" {
void **__cudaRegisterFatBinary(void *fatCubin);
void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
void __cudaUnregisterFatBinary(void **fatCubinHandle);
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize);
void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                       char *deviceAddress, const char *deviceName, int ext,
                       size_t size, int constant, int global);
void __cudaRegisterManagedVar(void **fatCubinHandle, void **hostVarPtrAddress,
                              char *deviceAddress, const char *deviceName,
                              int ext, size_t size, int constant, int global);
char __cudaInitModule(void **fatCubinHandle);
unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                     size_t sharedMem,
                                     struct CUstream_st *stream);
cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                       size_t *sharedMem, void *stream);
cudaError_t __cudaGetKernel(cudaKernel_t *kernel, const void *hostFun);
cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 gridDim,
                               dim3 blockDim, void **args, size_t sharedMem,
                               cudaStream_t stream);
cudaError_t __cudaLaunchKernel_ptsz(cudaKernel_t kernel, dim3 gridDim,
                                    dim3 blockDim, void **args,
                                    size_t sharedMem, cudaStream_t stream);
}

uint64_t front_devices_used = 1;

namespace {

struct Stream : FrontObj {
  unsigned flags;
  int priority;
  int device;
};

struct Event : FrontObj {
  unsigned flags;
  int device;
};

struct Kernel : FrontObj {
  const void *host_fun;
};

struct Registration {
  bool is_var;
  const char *host;      // host stub, or host shadow of the variable
  char *device;          // deviceFun, or deviceAddress
  const char *name;
  int thread_limit;
  int ext;
  size_t size;
  int constant;
  int global;
  std::map<int, void *> shadows;  // by device, during a checkpoint
};

struct Fatbin : FrontObj {
  void *fat_cubin;
  bool ended;
  std::vector<Registration> regs;
};

struct FuncSetting {
  int device;
  bool cache_config;     // cudaFuncSetCacheConfig, else cudaFuncSetAttribute
  int value;
};

struct Pinned {
  size_t size;
  unsigned flags;        // cudaHostRegister flags
  bool ours;             // cudaHostAlloc: we mapped it
};

std::mutex objs_lock;
FrontObj *heads[FRONT_FATBIN + 1];
FrontObj *tails[FRONT_FATBIN + 1];
std::map<const void *, Kernel *> kernels;      // by host stub
// (function, attribute) -> setting; cudaFuncSetCacheConfig uses -1.
std::map<std::pair<const void *, int>, FuncSetting> func_settings;
std::map<void *, Pinned> pinned;
std::map<int, unsigned> device_flags;

void
link(FrontObj *obj, FrontKind kind)
{
  obj->magic = FRONT_MAGIC;
  obj->kind = kind;
  obj->prev = tails[kind];
  obj->next = NULL;
  if (tails[kind] != NULL) {
    tails[kind]->next = obj;
  } else {
    heads[kind] = obj;
  }
  tails[kind] = obj;
}

void
unlink(FrontObj *obj)
{
  if (obj->prev != NULL) {
    obj->prev->next = obj->next;
  } else {
    heads[obj->kind] = obj->next;
  }
  if (obj->next != NULL) {
    obj->next->prev = obj->prev;
  } else {
    tails[obj->kind] = obj->prev;
  }
  obj->magic = 0;
}

int
current_device()
{
  int device = 0;
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGetDevice)(&device));
  return rc == cudaSuccess ? device : 0;
}

void
mark_device(int device)
{
  if (device >= 0 && device < 64) {
    __atomic_or_fetch(&front_devices_used, 1ULL << device, __ATOMIC_RELAXED);
  }
}

// Makes 'device' current for the checkpoint thread, with cudart's context.
void
use_device(int device)
{
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaSetDevice)(device));
  if (rc == cudaSuccess) {
    FRONT_LH(rc = REAL(cudaFree)(NULL));
  }
  if (rc != cudaSuccess) {
    front_fatal("using device %d: error %d\n", device, rc);
  }
}

cudaError_t
create_stream(cudaStream_t *stream, unsigned flags, int priority)
{
  if (stream == NULL) {
    return cudaErrorInvalidValue;
  }
  FRONT_ENTER();
  Stream *s = new Stream();
  s->flags = flags;
  s->priority = priority;
  s->device = current_device();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaStreamCreateWithPriority)(
             (cudaStream_t *)&s->real, flags, priority));
  if (rc == cudaSuccess) {
    std::lock_guard<std::mutex> guard(objs_lock);
    link(s, FRONT_STREAM);
    *stream = (cudaStream_t)s;
  } else {
    delete s;
  }
  FRONT_LEAVE();
  return rc;
}

cudaError_t
create_event(cudaEvent_t *event, unsigned flags)
{
  if (event == NULL) {
    return cudaErrorInvalidValue;
  }
  if (flags & cudaEventInterprocess) {
    front_unsupported("cudaEventCreateWithFlags(cudaEventInterprocess)");
  }
  FRONT_ENTER();
  Event *e = new Event();
  e->flags = flags;
  e->device = current_device();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaEventCreateWithFlags)((cudaEvent_t *)&e->real,
                                               flags));
  if (rc == cudaSuccess) {
    std::lock_guard<std::mutex> guard(objs_lock);
    link(e, FRONT_EVENT);
    *event = (cudaEvent_t)e;
  } else {
    delete e;
  }
  FRONT_LEAVE();
  return rc;
}

Fatbin *
fatbin_of(void **handle)
{
  Fatbin *f = (Fatbin *)handle;
  if (f == NULL || f->magic != FRONT_MAGIC || f->kind != FRONT_FATBIN) {
    front_fatal("not a fatbin handle: %p\n", (void *)handle);
  }
  return f;
}

cudaError_t
launch_kernel(bool ptsz, cudaKernel_t kernel, dim3 grid, dim3 block,
              void **args, size_t shared_mem, cudaStream_t stream)
{
  FRONT_ENTER();
  cudaKernel_t real =
    kernel != NULL ? (cudaKernel_t)((Kernel *)kernel)->real : NULL;
  cudaStream_t real_stream = front_real_stream(stream);
  cudaError_t rc;
  if (ptsz) {
    FRONT_LH(rc = REAL(__cudaLaunchKernel_ptsz)(real, grid, block, args,
                                                shared_mem, real_stream));
  } else {
    FRONT_LH(rc = REAL(__cudaLaunchKernel)(real, grid, block, args,
                                           shared_mem, real_stream));
  }
  FRONT_LEAVE();
  return rc;
}

void
set_func(const void *func, int attr, const FuncSetting &setting)
{
  cudaError_t rc;
  if (setting.cache_config) {
    FRONT_LH(rc = REAL(cudaFuncSetCacheConfig)(
               func, (enum cudaFuncCache)setting.value));
  } else {
    FRONT_LH(rc = REAL(cudaFuncSetAttribute)(
               func, (enum cudaFuncAttribute)attr, setting.value));
  }
  if (rc != cudaSuccess) {
    front_fatal("restoring attribute %d of function %p: error %d\n", attr,
                func, rc);
  }
}

}  // namespace

// ---- Devices -------------------------------------------------------------

extern "C" cudaError_t
cudaSetDevice(int device)
{
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaSetDevice)(device));
  if (rc == cudaSuccess) {
    front_thread.device = device;
    mark_device(device);
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaSetDeviceFlags(unsigned int flags)
{
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaSetDeviceFlags)(flags));
  if (rc == cudaSuccess) {
    int device = current_device();
    std::lock_guard<std::mutex> guard(objs_lock);
    device_flags[device] = flags;
  }
  FRONT_LEAVE();
  return rc;
}

// ---- Device memory -------------------------------------------------------

extern "C" cudaError_t
cudaMalloc(void **devPtr, size_t size)
{
  FRONT_ENTER();
  cudaError_t rc = front_vmm_alloc(devPtr, size);
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height)
{
  if (pitch == NULL) {
    return cudaErrorInvalidValue;
  }
  FRONT_ENTER();
  static size_t alignment = 0;
  if (alignment == 0) {
    int value = 0;
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaDeviceGetAttribute)(
               &value, cudaDevAttrTexturePitchAlignment, current_device()));
    alignment = (rc == cudaSuccess && value > 0) ? value : 512;
  }
  *pitch = (width + alignment - 1) / alignment * alignment;
  cudaError_t rc = front_vmm_alloc(devPtr, *pitch * height);
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaFree(void *devPtr)
{
  FRONT_ENTER();
  cudaError_t rc = cudaSuccess;
  if (!front_vmm_free(devPtr, &rc)) {
    // NULL (which initializes the context), or not ours.
    FRONT_LH(rc = REAL(cudaFree)(devPtr));
  }
  FRONT_LEAVE();
  return rc;
}

// ---- Pinned host memory: upper-half memory, registered again at restart --

extern "C" cudaError_t
cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
  if (pHost == NULL) {
    return cudaErrorInvalidValue;
  }
  if (size == 0) {
    *pHost = NULL;
    return cudaSuccess;
  }
  // cudaHostAllocWriteCombined cannot be reproduced on registered memory.
  unsigned reg = 0;
  if (flags & cudaHostAllocPortable) {
    reg |= cudaHostRegisterPortable;
  }
  if (flags & cudaHostAllocMapped) {
    reg |= cudaHostRegisterMapped;
  }
  void *p = mmap(NULL, size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    return cudaErrorMemoryAllocation;
  }
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaHostRegister)(p, size, reg));
  if (rc == cudaSuccess) {
    std::lock_guard<std::mutex> guard(objs_lock);
    pinned[p] = { size, reg, true };
  }
  FRONT_LEAVE();
  if (rc != cudaSuccess) {
    munmap(p, size);
    return rc;
  }
  *pHost = p;
  return cudaSuccess;
}

extern "C" cudaError_t
cudaMallocHost(void **ptr, size_t size)
{
  return cudaHostAlloc(ptr, size, cudaHostAllocDefault);
}

extern "C" cudaError_t
cudaFreeHost(void *ptr)
{
  FRONT_ENTER();
  Pinned p = { 0, 0, false };
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    auto it = pinned.find(ptr);
    if (it != pinned.end() && it->second.ours) {
      p = it->second;
      pinned.erase(it);
    }
  }
  cudaError_t rc;
  if (p.ours) {
    FRONT_LH(rc = REAL(cudaHostUnregister)(ptr));
  } else {
    FRONT_LH(rc = REAL(cudaFreeHost)(ptr));
  }
  FRONT_LEAVE();
  if (p.ours) {
    munmap(ptr, p.size);
  }
  return rc;
}

extern "C" cudaError_t
cudaHostRegister(void *ptr, size_t size, unsigned int flags)
{
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaHostRegister)(ptr, size, flags));
  if (rc == cudaSuccess) {
    std::lock_guard<std::mutex> guard(objs_lock);
    pinned[ptr] = { size, flags, false };
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaHostUnregister(void *ptr)
{
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaHostUnregister)(ptr));
  if (rc == cudaSuccess) {
    std::lock_guard<std::mutex> guard(objs_lock);
    pinned.erase(ptr);
  }
  FRONT_LEAVE();
  return rc;
}

// ---- Streams and events --------------------------------------------------

extern "C" cudaError_t
cudaStreamCreate(cudaStream_t *pStream)
{
  return create_stream(pStream, cudaStreamDefault, 0);
}

extern "C" cudaError_t
cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
{
  return create_stream(pStream, flags, 0);
}

extern "C" cudaError_t
cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags,
                             int priority)
{
  return create_stream(pStream, flags, priority);
}

extern "C" cudaError_t
cudaStreamDestroy(cudaStream_t stream)
{
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaStreamDestroy)(front_real_stream(stream)));
  if (rc == cudaSuccess && (uintptr_t)stream > 2) {
    std::lock_guard<std::mutex> guard(objs_lock);
    unlink((FrontObj *)stream);
    delete (Stream *)stream;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaEventCreate(cudaEvent_t *event)
{
  return create_event(event, cudaEventDefault);
}

extern "C" cudaError_t
cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
  return create_event(event, flags);
}

extern "C" cudaError_t
cudaEventDestroy(cudaEvent_t event)
{
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaEventDestroy)(front_real_event(event)));
  if (rc == cudaSuccess && event != NULL) {
    std::lock_guard<std::mutex> guard(objs_lock);
    unlink((FrontObj *)event);
    delete (Event *)event;
  }
  FRONT_LEAVE();
  return rc;
}

// ---- Kernels -------------------------------------------------------------

extern "C" void **
__cudaRegisterFatBinary(void *fatCubin)
{
  FRONT_ENTER();
  Fatbin *f = new Fatbin();
  f->fat_cubin = fatCubin;
  f->ended = false;
  FRONT_LH(f->real = REAL(__cudaRegisterFatBinary)(fatCubin));
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    link(f, FRONT_FATBIN);
  }
  FRONT_LEAVE();
  return (void **)f;
}

extern "C" void
__cudaRegisterFatBinaryEnd(void **fatCubinHandle)
{
  FRONT_ENTER();
  Fatbin *f = fatbin_of(fatCubinHandle);
  FRONT_LH(REAL(__cudaRegisterFatBinaryEnd)((void **)f->real));
  f->ended = true;
  FRONT_LEAVE();
}

extern "C" void
__cudaUnregisterFatBinary(void **fatCubinHandle)
{
  FRONT_ENTER();
  Fatbin *f = fatbin_of(fatCubinHandle);
  FRONT_LH(REAL(__cudaUnregisterFatBinary)((void **)f->real));
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    unlink(f);
  }
  delete f;
  FRONT_LEAVE();
}

extern "C" void
__cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                       char *deviceFun, const char *deviceName,
                       int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim,
                       dim3 *gDim, int *wSize)
{
  // nvcc passes NULL for tid, bid, bDim, gDim and wSize.
  if (tid != NULL || bid != NULL || bDim != NULL || gDim != NULL ||
      wSize != NULL) {
    front_unsupported("__cudaRegisterFunction with launch bounds pointers");
  }
  FRONT_ENTER();
  Fatbin *f = fatbin_of(fatCubinHandle);
  Registration r = { false, hostFun, deviceFun, deviceName, thread_limit,
                     0, 0, 0, 0, {} };
  f->regs.push_back(r);
  FRONT_LH(REAL(__cudaRegisterFunction)((void **)f->real, hostFun, deviceFun,
                                        deviceName, thread_limit, NULL, NULL,
                                        NULL, NULL, NULL));
  FRONT_LEAVE();
}

extern "C" void
__cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
                  const char *deviceName, int ext, size_t size, int constant,
                  int global)
{
  FRONT_ENTER();
  Fatbin *f = fatbin_of(fatCubinHandle);
  Registration r = { true, hostVar, deviceAddress, deviceName, 0, ext, size,
                     constant, global, {} };
  f->regs.push_back(r);
  FRONT_LH(REAL(__cudaRegisterVar)((void **)f->real, hostVar, deviceAddress,
                                   deviceName, ext, size, constant, global));
  FRONT_LEAVE();
}

extern "C" void
__cudaRegisterManagedVar(void **, void **, char *, const char *, int, size_t,
                         int, int)
{
  front_unsupported("__managed__ variables (__cudaRegisterManagedVar)");
}

extern "C" char
__cudaInitModule(void **fatCubinHandle)
{
  FRONT_ENTER();
  Fatbin *f = fatbin_of(fatCubinHandle);
  char rc;
  FRONT_LH(rc = REAL(__cudaInitModule)((void **)f->real));
  FRONT_LEAVE();
  return rc;
}

// The launch configuration of <<<...>>> stays in the upper half, with the
// virtual stream: the lower half sees only the launch.
extern "C" unsigned
__cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                            struct CUstream_st *stream)
{
  FrontThread *t = &front_thread;
  if (t->n_configs == FRONT_MAX_LAUNCH_CONFIGS) {
    front_fatal("too many nested launch configurations\n");
  }
  t->configs[t->n_configs++] = { gridDim, blockDim, sharedMem, stream };
  return 0;
}

extern "C" cudaError_t
__cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim, size_t *sharedMem,
                           void *stream)
{
  FrontThread *t = &front_thread;
  if (t->n_configs == 0) {
    return cudaErrorMissingConfiguration;
  }
  FrontLaunchConfig *c = &t->configs[--t->n_configs];
  *gridDim = c->grid;
  *blockDim = c->block;
  *sharedMem = c->shared_mem;
  *(cudaStream_t *)stream = c->stream;
  return cudaSuccess;
}

// nvcc keeps the kernel handle in a static variable of the host stub: the
// handle must survive a restart, so it is a record.
extern "C" cudaError_t
__cudaGetKernel(cudaKernel_t *kernel, const void *hostFun)
{
  if (kernel == NULL) {
    return cudaErrorInvalidValue;
  }
  FRONT_ENTER();
  cudaError_t rc = cudaSuccess;
  Kernel *k = NULL;
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    auto it = kernels.find(hostFun);
    if (it != kernels.end()) {
      k = it->second;
    }
  }
  if (k == NULL) {
    cudaKernel_t real = NULL;
    FRONT_LH(rc = REAL(__cudaGetKernel)(&real, hostFun));
    if (rc == cudaSuccess) {
      std::lock_guard<std::mutex> guard(objs_lock);
      auto it = kernels.find(hostFun);
      if (it != kernels.end()) {
        k = it->second;
      } else {
        k = new Kernel();
        k->real = real;
        k->host_fun = hostFun;
        link(k, FRONT_KERNEL);
        kernels[hostFun] = k;
      }
    }
  }
  if (rc == cudaSuccess) {
    *kernel = (cudaKernel_t)k;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
__cudaLaunchKernel(cudaKernel_t kernel, dim3 gridDim, dim3 blockDim,
                   void **args, size_t sharedMem, cudaStream_t stream)
{
  return launch_kernel(false, kernel, gridDim, blockDim, args, sharedMem,
                       stream);
}

extern "C" cudaError_t
__cudaLaunchKernel_ptsz(cudaKernel_t kernel, dim3 gridDim, dim3 blockDim,
                        void **args, size_t sharedMem, cudaStream_t stream)
{
  return launch_kernel(true, kernel, gridDim, blockDim, args, sharedMem,
                       stream);
}

extern "C" cudaError_t
cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr,
                     int value)
{
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaFuncSetAttribute)(func, attr, value));
  if (rc == cudaSuccess) {
    std::lock_guard<std::mutex> guard(objs_lock);
    func_settings[{ func, (int)attr }] = { current_device(), false, value };
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache cacheConfig)
{
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaFuncSetCacheConfig)(func, cacheConfig));
  if (rc == cudaSuccess) {
    std::lock_guard<std::mutex> guard(objs_lock);
    func_settings[{ func, -1 }] = { current_device(), true,
                                    (int)cacheConfig };
  }
  FRONT_LEAVE();
  return rc;
}

// ---- Checkpoint and restart ----------------------------------------------

// __device__ and __constant__ contents, on each device used (each context
// has its own copy).  Their addresses move at restart.
void
front_objs_stash()
{
  std::lock_guard<std::mutex> guard(objs_lock);
  for (int device = 0; device < 64; device++) {
    if (!(front_devices_used & (1ULL << device))) {
      continue;
    }
    use_device(device);
    for (FrontObj *o = heads[FRONT_FATBIN]; o != NULL; o = o->next) {
      for (Registration &r : ((Fatbin *)o)->regs) {
        if (!r.is_var || r.size == 0) {
          continue;
        }
        void *shadow = malloc(r.size);
        cudaError_t rc;
        FRONT_LH(rc = REAL(cudaMemcpyFromSymbol)(shadow, r.host, r.size, 0,
                                                 cudaMemcpyDeviceToHost));
        if (rc != cudaSuccess) {
          front_fatal("saving %s on device %d: error %d\n", r.name, device,
                      rc);
        }
        r.shadows[device] = shadow;
      }
    }
  }
}

void
front_objs_drop_shadows()
{
  std::lock_guard<std::mutex> guard(objs_lock);
  for (FrontObj *o = heads[FRONT_FATBIN]; o != NULL; o = o->next) {
    for (Registration &r : ((Fatbin *)o)->regs) {
      for (auto &entry : r.shadows) {
        free(entry.second);
      }
      r.shadows.clear();
    }
  }
}

// Before any context exists in the new lower half.
void
front_objs_restore_devices()
{
  std::lock_guard<std::mutex> guard(objs_lock);
  for (auto &entry : device_flags) {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaSetDevice)(entry.first));
    if (rc == cudaSuccess) {
      FRONT_LH(rc = REAL(cudaSetDeviceFlags)(entry.second));
    }
    if (rc != cudaSuccess) {
      front_fatal("restoring the flags of device %d: error %d\n",
                  entry.first, rc);
    }
  }
}

// In the dependency order: kernels, their attributes and globals, streams,
// events, pinned memory.  (Device memory is already back: front_vmm.cpp.)
void
front_objs_restore()
{
  std::lock_guard<std::mutex> guard(objs_lock);
  cudaError_t rc;
  for (FrontObj *o = heads[FRONT_FATBIN]; o != NULL; o = o->next) {
    Fatbin *f = (Fatbin *)o;
    FRONT_LH(f->real = REAL(__cudaRegisterFatBinary)(f->fat_cubin));
    for (Registration &r : f->regs) {
      if (r.is_var) {
        FRONT_LH(REAL(__cudaRegisterVar)((void **)f->real, (char *)r.host,
                                         r.device, r.name, r.ext, r.size,
                                         r.constant, r.global));
      } else {
        FRONT_LH(REAL(__cudaRegisterFunction)((void **)f->real, r.host,
                                              r.device, r.name,
                                              r.thread_limit, NULL, NULL,
                                              NULL, NULL, NULL));
      }
    }
    if (f->ended) {
      FRONT_LH(REAL(__cudaRegisterFatBinaryEnd)((void **)f->real));
    }
  }
  for (FrontObj *o = heads[FRONT_KERNEL]; o != NULL; o = o->next) {
    Kernel *k = (Kernel *)o;
    FRONT_LH(rc = REAL(__cudaGetKernel)((cudaKernel_t *)&k->real,
                                        k->host_fun));
    if (rc != cudaSuccess) {
      front_fatal("finding kernel %p again: error %d\n", k->host_fun, rc);
    }
  }
  for (auto &entry : func_settings) {
    use_device(entry.second.device);
    set_func(entry.first.first, entry.first.second, entry.second);
  }
  for (int device = 0; device < 64; device++) {
    if (!(front_devices_used & (1ULL << device))) {
      continue;
    }
    use_device(device);
    for (FrontObj *o = heads[FRONT_FATBIN]; o != NULL; o = o->next) {
      for (Registration &r : ((Fatbin *)o)->regs) {
        auto it = r.shadows.find(device);
        if (!r.is_var || it == r.shadows.end()) {
          continue;
        }
        FRONT_LH(rc = REAL(cudaMemcpyToSymbol)(r.host, it->second, r.size, 0,
                                               cudaMemcpyHostToDevice));
        if (rc != cudaSuccess) {
          front_fatal("restoring %s on device %d: error %d\n", r.name,
                      device, rc);
        }
      }
    }
  }
  for (FrontObj *o = heads[FRONT_STREAM]; o != NULL; o = o->next) {
    Stream *s = (Stream *)o;
    use_device(s->device);
    FRONT_LH(rc = REAL(cudaStreamCreateWithPriority)(
               (cudaStream_t *)&s->real, s->flags, s->priority));
    if (rc != cudaSuccess) {
      front_fatal("creating a stream again: error %d\n", rc);
    }
  }
  // An event that was never recorded completes at once, as a recorded one
  // does after the checkpoint's synchronization.  So record each event
  // once: waits and queries then behave as before the checkpoint.
  for (FrontObj *o = heads[FRONT_EVENT]; o != NULL; o = o->next) {
    Event *e = (Event *)o;
    use_device(e->device);
    FRONT_LH(rc = REAL(cudaEventCreateWithFlags)((cudaEvent_t *)&e->real,
                                                 e->flags));
    if (rc == cudaSuccess) {
      FRONT_LH(rc = REAL(cudaEventRecord)((cudaEvent_t)e->real, NULL));
    }
    if (rc != cudaSuccess) {
      front_fatal("creating an event again: error %d\n", rc);
    }
  }
  for (auto &entry : pinned) {
    FRONT_LH(rc = REAL(cudaHostRegister)(entry.first, entry.second.size,
                                         entry.second.flags));
    if (rc != cudaSuccess) {
      front_fatal("registering host memory %p again: error %d\n",
                  entry.first, rc);
    }
  }
  front_log("restored the CUDA objects\n");
}
