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

// The runtime's own functions ("own" in cudart-own.txt), and the records
// from which a restart rebuilds what was alive at the checkpoint: one record
// per live object, never a history.  The kernel registrations that the
// application's constructors make are the one fixed set of calls that a
// restart makes again.

#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <algorithm>
#include "front_objs.h"
#include "front_reg.h"

uint64_t front_devices_used;
std::mutex objs_lock;
FrontObj *obj_heads[FRONT_N_KINDS];
std::map<void *, Pinned> pinned;

namespace {

struct Kernel : FrontObj {
  const void *host_fun;
  Ctx *ctx;
};

struct DeviceInit : FrontObj {
  int device;
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

FrontObj *tails[FRONT_N_KINDS];
uint64_t next_seq;
bool cudart_noted[64];
std::map<const void *, Kernel *> kernels;      // by host stub
// (function, attribute) -> setting; cudaFuncSetCacheConfig uses -1.
std::map<std::pair<const void *, int>, FuncSetting> func_settings;
std::map<int, unsigned> device_flags;
std::map<int, std::string> error_strings;      // (name? 1 : 0) + 2 * error
// (device, peer) with peer access, as cudaDeviceEnablePeerAccess made it.
std::map<std::pair<int, int>, unsigned> peer_access;
// cudaGetSymbolAddress: what the application has, by (device, symbol).
std::map<std::pair<int, const void *>, void *> symbol_addresses;

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

// The calling thread's context, for a record of an object made in it.
Ctx *
creating_ctx()
{
  Ctx *ctx = front_ctx_current();
  if (ctx == NULL) {
    ctx = front_ctx_primary(current_device());
  }
  return ctx;
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

// A string of the lower half's libcudart, copied into upper-half memory: the
// application may keep it across a restart.
const char *
error_string(cudaError_t error, bool name)
{
  int key = 2 * (int)error + (name ? 1 : 0);
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    auto it = error_strings.find(key);
    if (it != error_strings.end()) {
      return it->second.c_str();
    }
  }
  const char *s;
  FRONT_ENTER();
  if (name) {
    FRONT_LH(s = REAL(cudaGetErrorName)(error));
  } else {
    FRONT_LH(s = REAL(cudaGetErrorString)(error));
  }
  FRONT_LEAVE();
  std::lock_guard<std::mutex> guard(objs_lock);
  return error_strings.emplace(key, s != NULL ? s : "").first->second.c_str();
}

}  // namespace

void
obj_link(FrontObj *obj, FrontKind kind)
{
  obj->magic = FRONT_MAGIC;
  obj->kind = kind;
  obj->seq = ++next_seq;
  obj->prev = tails[kind];
  obj->next = NULL;
  if (tails[kind] != NULL) {
    tails[kind]->next = obj;
  } else {
    obj_heads[kind] = obj;
  }
  tails[kind] = obj;
}

void
obj_unlink(FrontObj *obj)
{
  if (obj->prev != NULL) {
    obj->prev->next = obj->next;
  } else {
    obj_heads[obj->kind] = obj->next;
  }
  if (obj->next != NULL) {
    obj->next->prev = obj->prev;
  } else {
    tails[obj->kind] = obj->prev;
  }
  obj->magic = 0;
}

Ctx *
front_object_ctx()
{
  Ctx *ctx = creating_ctx();
  mark_device(ctx->device);
  return ctx;
}

bool
front_stream_live(cudaStream_t stream)
{
  if ((uintptr_t)stream <= 2) {
    return true;
  }
  std::lock_guard<std::mutex> guard(objs_lock);
  for (FrontObj *o = obj_heads[FRONT_STREAM]; o != NULL; o = o->next) {
    if (o == (FrontObj *)stream) {
      return true;
    }
  }
  return false;
}

static const FrontLibRecords *lib_records[FRONT_N_KINDS];

void
front_lib_records(FrontKind kind, const FrontLibRecords *records)
{
  lib_records[kind] = records;
}

void
front_libs_forget(Ctx *ctx)
{
  for (const FrontLibRecords *records : lib_records) {
    if (records != NULL) {
      records->forget(ctx);
    }
  }
}

static void
libs_stash()
{
  for (const FrontLibRecords *records : lib_records) {
    if (records != NULL && records->stash != NULL) {
      records->stash();
    }
  }
}

static void
libs_drop()
{
  for (const FrontLibRecords *records : lib_records) {
    if (records != NULL && records->drop != NULL) {
      records->drop();
    }
  }
}

bool
front_registered_kernel(const void *host)
{
  for (FrontObj *o = obj_heads[FRONT_FATBIN]; o != NULL; o = o->next) {
    for (const Registration &r : ((Fatbin *)o)->regs) {
      if (!r.is_var && r.host == host) {
        return true;
      }
    }
  }
  return false;
}

// Loads the modules of the fatbins registered before 'seq' into cudart's
// context of the current device, by a variable or function of each.  CUDA
// loads a module at its first use, and so does a library its kernels; the
// driver places module data by what it allocated before.  So a library's
// kernels would move the application's module data, by their timing, which
// a restart does not repeat.  Instead, the application's modules are loaded
// when cudart starts on a device, at launch and at that place in the replay
// alike, and before a library makes a handle (front_before_library()).
static void
load_modules(uint64_t seq)
{
  std::lock_guard<std::mutex> guard(objs_lock);
  cudaError_t pending;
  FRONT_LH(pending = REAL(cudaPeekAtLastError)());
  for (FrontObj *o = obj_heads[FRONT_FATBIN]; o != NULL && o->seq < seq;
       o = o->next) {
    Fatbin *f = (Fatbin *)o;
    const Registration *use = NULL;
    for (const Registration &r : f->regs) {
      if (use == NULL || (r.is_var && !use->is_var)) {
        use = &r;
      }
    }
    if (!f->ended || use == NULL) {
      continue;
    }
    // A module that cannot load fails as it would at its first use.
    cudaError_t rc;
    if (use->is_var) {
      void *address;
      FRONT_LH(rc = REAL(cudaGetSymbolAddress)(&address, use->host));
    } else {
      cudaFuncAttributes attr;
      FRONT_LH(rc = REAL(cudaFuncGetAttributes)(&attr, use->host));
    }
    (void)rc;
  }
  if (pending == cudaSuccess) {
    FRONT_LH((void)REAL(cudaGetLastError)());
  }
}

void
front_note_cudart(int device)
{
  if (device < 0 || device >= 64 ||
      __atomic_load_n(&cudart_noted[device], __ATOMIC_ACQUIRE)) {
    return;
  }
  mark_device(device);
  DeviceInit *d = NULL;
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    if (!cudart_noted[device]) {
      d = new DeviceInit();
      d->device = device;
      obj_link(d, FRONT_DEVICE);
      __atomic_store_n(&cudart_noted[device], true, __ATOMIC_RELEASE);
    }
  }
  if (d != NULL) {
    load_modules(d->seq);
  }
}

void
front_before_library()
{
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    if (obj_heads[FRONT_FATBIN] == NULL) {
      return;
    }
  }
  front_note_cudart(current_device());
}

// Makes 'device' current for the checkpoint thread, with cudart's context.
void
front_use_device(int device)
{
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaSetDevice)(device));
  if (rc == cudaSuccess) {
    FRONT_LH(rc = REAL(cudaFree)(NULL));
  }
  if (rc != cudaSuccess) {
    front_fatal("using device %d: error %d\n", device, rc);
  }
  front_note_cudart(device);
}

// ---- Shared by the runtime and the driver --------------------------------

int
front_create_stream(cudaStream_t *stream, unsigned flags, int priority,
                    bool drv)
{
  if (stream == NULL) {
    return cudaErrorInvalidValue;  // CUDA_ERROR_INVALID_VALUE too
  }
  FRONT_ENTER();
  Stream *s = new Stream();
  s->flags = flags;
  s->priority = priority;
  int rc;
  if (drv) {
    FRONT_LH(rc = DRV(cuStreamCreateWithPriority)((CUstream *)&s->real,
                                                  flags, priority));
  } else {
    FRONT_LH(rc = REAL(cudaStreamCreateWithPriority)(
               (cudaStream_t *)&s->real, flags, priority));
  }
  if (rc == 0) {
    s->ctx = creating_ctx();
    if (!drv) {
      front_note_cudart(s->ctx->device);
    }
    std::lock_guard<std::mutex> guard(objs_lock);
    obj_link(s, FRONT_STREAM);
    *stream = (cudaStream_t)s;
  } else {
    delete s;
  }
  FRONT_LEAVE();
  return rc;
}

int
front_destroy_stream(cudaStream_t stream, bool drv)
{
  FRONT_ENTER();
  int rc;
  if (drv) {
    FRONT_LH(rc = DRV(cuStreamDestroy_v2)(front_real_stream(stream)));
  } else {
    FRONT_LH(rc = REAL(cudaStreamDestroy)(front_real_stream(stream)));
  }
  if (rc == 0 && (uintptr_t)stream > 2) {
    std::lock_guard<std::mutex> guard(objs_lock);
    obj_unlink((FrontObj *)stream);
    delete (Stream *)stream;
  }
  FRONT_LEAVE();
  return rc;
}

int
front_create_event(cudaEvent_t *event, unsigned flags, bool drv)
{
  if (event == NULL) {
    return cudaErrorInvalidValue;
  }
  if (flags & cudaEventInterprocess) {  // CU_EVENT_INTERPROCESS too
    front_unsupported("interprocess events");
  }
  FRONT_ENTER();
  Event *e = new Event();
  e->flags = flags;
  int rc;
  if (drv) {
    FRONT_LH(rc = DRV(cuEventCreate)((CUevent *)&e->real, flags));
  } else {
    FRONT_LH(rc = REAL(cudaEventCreateWithFlags)((cudaEvent_t *)&e->real,
                                                 flags));
  }
  if (rc == 0) {
    e->ctx = creating_ctx();
    if (!drv) {
      front_note_cudart(e->ctx->device);
    }
    std::lock_guard<std::mutex> guard(objs_lock);
    obj_link(e, FRONT_EVENT);
    *event = (cudaEvent_t)e;
  } else {
    delete e;
  }
  FRONT_LEAVE();
  return rc;
}

int
front_destroy_event(cudaEvent_t event, bool drv)
{
  FRONT_ENTER();
  int rc;
  if (drv) {
    FRONT_LH(rc = DRV(cuEventDestroy_v2)(front_real_event(event)));
  } else {
    FRONT_LH(rc = REAL(cudaEventDestroy)(front_real_event(event)));
  }
  if (rc == 0 && event != NULL) {
    std::lock_guard<std::mutex> guard(objs_lock);
    obj_unlink((FrontObj *)event);
    delete (Event *)event;
  }
  FRONT_LEAVE();
  return rc;
}

// Pinned host memory is upper-half memory, registered again at restart.  The
// register flags are the same in both APIs.
int
front_host_alloc(void **ptr, size_t size, unsigned register_flags, bool drv)
{
  if (ptr == NULL) {
    return cudaErrorInvalidValue;
  }
  if (size == 0) {
    *ptr = NULL;
    return 0;
  }
  void *p = mmap(NULL, size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    return cudaErrorMemoryAllocation;  // CUDA_ERROR_OUT_OF_MEMORY too
  }
  int rc = front_host_register(p, size, register_flags, drv);
  if (rc != 0) {
    munmap(p, size);
    return rc;
  }
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    pinned[p].ours = true;
  }
  *ptr = p;
  return 0;
}

int
front_host_free(void *ptr, bool drv)
{
  size_t size = 0;
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    auto it = pinned.find(ptr);
    if (it != pinned.end() && it->second.ours) {
      size = it->second.size;
    }
  }
  if (size == 0) {
    // Not the front's (or NULL): the lower half's error, or success.
    FRONT_ENTER();
    int rc;
    if (drv) {
      FRONT_LH(rc = DRV(cuMemFreeHost)(ptr));
    } else {
      FRONT_LH(rc = REAL(cudaFreeHost)(ptr));
    }
    FRONT_LEAVE();
    return rc;
  }
  int rc = front_host_unregister(ptr, drv);
  if (rc == 0) {
    munmap(ptr, size);
  }
  return rc;
}

int
front_host_register(void *ptr, size_t size, unsigned flags, bool drv)
{
  FRONT_ENTER();
  int rc;
  if (drv) {
    FRONT_LH(rc = DRV(cuMemHostRegister_v2)(ptr, size, flags));
  } else {
    FRONT_LH(rc = REAL(cudaHostRegister)(ptr, size, flags));
  }
  if (rc == 0) {
    Ctx *ctx = creating_ctx();
    std::lock_guard<std::mutex> guard(objs_lock);
    pinned[ptr] = { size, flags, false, ctx };
  }
  FRONT_LEAVE();
  return rc;
}

int
front_host_unregister(void *ptr, bool drv)
{
  FRONT_ENTER();
  int rc;
  if (drv) {
    FRONT_LH(rc = DRV(cuMemHostUnregister)(ptr));
  } else {
    FRONT_LH(rc = REAL(cudaHostUnregister)(ptr));
  }
  if (rc == 0) {
    std::lock_guard<std::mutex> guard(objs_lock);
    pinned.erase(ptr);
  }
  FRONT_LEAVE();
  return rc;
}

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
    // cudart makes the device's primary context, and makes it current.
    front_stack_set_top(front_ctx_primary(device));
    front_note_cudart(device);
  }
  FRONT_LEAVE();
  return rc;
}

// A reset of 'device': its init record and cudart_noted go, so that the
// next call on it starts cudart and loads the application's modules again,
// at its place in the creation order (front_note_cudart()).  Symbols may
// then be elsewhere, and its peer access went with its context.
void
front_cudart_forget(int device)
{
  if (device < 0 || device >= 64) {
    return;
  }
  std::lock_guard<std::mutex> guard(objs_lock);
  for (FrontObj *o = obj_heads[FRONT_DEVICE], *next; o != NULL; o = next) {
    next = o->next;
    if (((DeviceInit *)o)->device == device) {
      obj_unlink(o);
    }
  }
  __atomic_store_n(&cudart_noted[device], false, __ATOMIC_RELEASE);
  for (auto it = symbol_addresses.begin(); it != symbol_addresses.end();) {
    it = it->first.first == device ? symbol_addresses.erase(it) : std::next(it);
  }
  for (auto it = peer_access.begin(); it != peer_access.end();) {
    it = it->first.first == device ? peer_access.erase(it) : std::next(it);
  }
}

extern "C" cudaError_t
cudaDeviceReset(void)
{
  return (cudaError_t)front_reset_primary(current_device(), true);
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

// cuDevicePrimaryCtxSetFlags (front_drv.cpp) sets the same flags.
void
front_set_device_flags(int device, unsigned flags)
{
  std::lock_guard<std::mutex> guard(objs_lock);
  device_flags[device] = flags;
}

// The limits and the cache configuration belong to the current context;
// cudaLimit and CUlimit, cudaFuncCache and CUfunc_cache have the same values.
extern "C" cudaError_t
cudaDeviceSetLimit(enum cudaLimit limit, size_t value)
{
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaDeviceSetLimit)(limit, value));
  if (rc == cudaSuccess) {
    Ctx *ctx = creating_ctx();
    std::lock_guard<std::mutex> guard(objs_lock);
    ctx->limits[(int)limit] = value;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig)
{
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaDeviceSetCacheConfig)(cacheConfig));
  if (rc == cudaSuccess) {
    Ctx *ctx = creating_ctx();
    std::lock_guard<std::mutex> guard(objs_lock);
    ctx->cache_config = (int)cacheConfig;
  }
  FRONT_LEAVE();
  return rc;
}

// Peer access, from the current device; it stays on across restarts.
static cudaError_t
peer_access_on(int peer, unsigned flags, bool on)
{
  FRONT_ENTER();
  cudaError_t rc;
  int device = current_device();
  if (on) {
    FRONT_LH(rc = REAL(cudaDeviceEnablePeerAccess)(peer, flags));
  } else {
    FRONT_LH(rc = REAL(cudaDeviceDisablePeerAccess)(peer));
  }
  if (rc == cudaSuccess) {
    front_vmm_peer_access(device, peer, on);
    std::lock_guard<std::mutex> guard(objs_lock);
    if (on) {
      peer_access[{ device, peer }] = flags;
    } else {
      peer_access.erase({ device, peer });
    }
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
{
  return peer_access_on(peerDevice, flags, true);
}

extern "C" cudaError_t
cudaDeviceDisablePeerAccess(int peerDevice)
{
  return peer_access_on(peerDevice, 0, false);
}

// The context API's form (front_drv.cpp).
void
front_peer_access(int device, int peer, unsigned flags, bool on)
{
  front_vmm_peer_access(device, peer, on);
  std::lock_guard<std::mutex> guard(objs_lock);
  if (on) {
    peer_access[{ device, peer }] = flags;
  } else {
    peer_access.erase({ device, peer });
  }
}

// A __device__ address that the application has must not move at restart:
// it may be stored in device memory, out of the front's sight.
extern "C" cudaError_t
cudaGetSymbolAddress(void **devPtr, const void *symbol)
{
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGetSymbolAddress)(devPtr, symbol));
  if (rc == cudaSuccess && devPtr != NULL) {
    int device = current_device();
    std::lock_guard<std::mutex> guard(objs_lock);
    symbol_addresses[{ device, symbol }] = *devPtr;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" const char *
cudaGetErrorString(cudaError_t error)
{
  return error_string(error, false);
}

extern "C" const char *
cudaGetErrorName(cudaError_t error)
{
  return error_string(error, true);
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
  cudaError_t rc = front_vmm_alloc_pitch(devPtr, pitch, width, height);
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaFree(void *devPtr)
{
  FRONT_ENTER();
  cudaError_t rc = cudaSuccess;
  int managed_rc = 0;
  if (front_managed_free(devPtr, &managed_rc, false)) {
    rc = (cudaError_t)managed_rc;
  } else if (!front_vmm_free(devPtr, &rc)) {
    // NULL (which initializes the context), or not ours.
    FRONT_LH(rc = REAL(cudaFree)(devPtr));
  }
  FRONT_LEAVE();
  return rc;
}

// ---- Managed memory ------------------------------------------------------
//
// The driver places managed memory itself, so a restart could not put it
// back at its address.  Instead it is memory of the upper half, saved with
// the image:
// - Where the GPUs can use pageable memory (HMM, ATS), anonymous memory: the
//   driver migrates it to a GPU as it does managed memory, and the CPU's
//   reads bring it back for the checkpoint.  Prefetches and advice work on it.
// - Elsewhere, or with MANA_CUDA_MANAGED=pinned, mapped pinned memory,
//   registered again at restart: the same address on the host and on the
//   devices (UVA), where a device can use host pointers for registered
//   memory.  It is not migrated, so kernels reach it over the bus: correct,
//   but slower.  Prefetches and advice on it do nothing.

namespace {
std::map<void *, size_t> pageable_managed;  // under objs_lock
}

// The current device's attribute, through the API that the caller uses.
static int
current_device_attribute(cudaDeviceAttr rt_attr, CUdevice_attribute drv_attr,
                         bool drv, int *value)
{
  int rc;
  FRONT_ENTER();
  if (drv) {
    CUdevice device = 0;
    FRONT_LH(rc = DRV(cuCtxGetDevice)(&device));
    if (rc == 0) {
      FRONT_LH(rc = DRV(cuDeviceGetAttribute)(value, drv_attr, device));
    }
  } else {
    int device = 0;
    FRONT_LH(rc = REAL(cudaGetDevice)(&device));
    if (rc == 0) {
      FRONT_LH(rc = REAL(cudaDeviceGetAttribute)(value, rt_attr, device));
    }
  }
  FRONT_LEAVE();
  return rc;
}

int
front_managed_alloc(void **ptr, size_t size, bool drv)
{
  if (ptr == NULL) {
    return cudaErrorInvalidValue;
  }
  if (size == 0) {
    *ptr = NULL;
    return 0;
  }
  const char *kind = getenv("MANA_CUDA_MANAGED");
  int pageable = 0;
  int rc = 0;
  if (kind == NULL || strcmp(kind, "pinned") != 0) {
    rc = current_device_attribute(cudaDevAttrPageableMemoryAccess,
                                  CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
                                  drv, &pageable);
    if (rc != 0) {
      return rc;
    }
  }
  if (pageable) {
    void *p = mmap(NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) {
      return cudaErrorMemoryAllocation;
    }
    std::lock_guard<std::mutex> guard(objs_lock);
    pageable_managed[p] = size;
    *ptr = p;
    return 0;
  }
  int same = 0;
  rc = current_device_attribute(
    cudaDevAttrCanUseHostPointerForRegisteredMem,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM, drv, &same);
  if (rc != 0) {
    return rc;
  }
  if (!same) {
    front_unsupported("managed memory on a device without pageable memory "
                      "access that cannot use host pointers for registered "
                      "memory");
  }
  static std::once_flag notice;
  std::call_once(notice, [] {
    fprintf(stderr, "MANA: --cuda: managed memory is mapped host memory "
            "under MANA (no HMM), which a restart can make again; kernels "
            "reach it over the bus, more slowly than managed memory.\n");
  });
  rc = front_host_alloc(ptr, size,
                        cudaHostRegisterMapped | cudaHostRegisterPortable, drv);
  if (rc == 0 && *ptr != NULL) {
    std::lock_guard<std::mutex> guard(objs_lock);
    pinned[*ptr].managed = true;
  }
  return rc;
}

bool
front_managed_free(void *ptr, int *rc, bool drv)
{
  if (ptr == NULL) {
    return false;
  }
  size_t pageable_size = 0;
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    auto pageable = pageable_managed.find(ptr);
    if (pageable != pageable_managed.end()) {
      pageable_size = pageable->second;
      pageable_managed.erase(pageable);
    } else {
      auto it = pinned.find(ptr);
      if (it == pinned.end() || !it->second.managed) {
        return false;
      }
    }
  }
  if (pageable_size != 0) {
    // The GPUs may still use it: wait for them, as cudaFree does.
    FRONT_ENTER();
    FRONT_LH(*rc = REAL(cudaDeviceSynchronize)());
    FRONT_LEAVE();
    munmap(ptr, pageable_size);
    return true;
  }
  *rc = front_host_free(ptr, drv);
  return true;
}

bool
front_is_pinned_managed(const void *ptr, size_t count)
{
  std::lock_guard<std::mutex> guard(objs_lock);
  auto it = pinned.upper_bound((void *)ptr);
  if (it == pinned.begin()) {
    return false;
  }
  --it;
  const char *base = (const char *)it->first;
  return it->second.managed && (const char *)ptr >= base &&
         (const char *)ptr + count <= base + it->second.size;
}

void
front_managed_check_restart()
{
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    if (pageable_managed.empty()) {
      return;
    }
  }
  for (int device = 0; device < 64; device++) {
    if (!(front_devices_used & (1ULL << device))) {
      continue;
    }
    int pageable = 0;
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaDeviceGetAttribute)(
      &pageable, cudaDevAttrPageableMemoryAccess, device));
    if (rc != cudaSuccess || !pageable) {
      front_fatal("managed memory was pageable memory (HMM) at the "
                  "checkpoint, but device %d here cannot access pageable "
                  "memory\n", device);
    }
  }
}

extern "C" cudaError_t
cudaMallocManaged(void **devPtr, size_t size, unsigned int flags)
{
  if (flags != cudaMemAttachGlobal && flags != cudaMemAttachHost) {
    return cudaErrorInvalidValue;
  }
  return (cudaError_t)front_managed_alloc(devPtr, size, false);
}

extern "C" cudaError_t
cudaMemPrefetchAsync(const void *devPtr, size_t count,
                     struct cudaMemLocation location, unsigned int flags,
                     cudaStream_t stream)
{
  if (front_is_pinned_managed(devPtr, count)) {
    return cudaSuccess;
  }
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaMemPrefetchAsync)(devPtr, count, location, flags,
                                           front_real_stream(stream)));
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaMemPrefetchAsync_ptsz(const void *devPtr, size_t count,
                          struct cudaMemLocation location, unsigned int flags,
                          cudaStream_t stream)
{
  if (front_is_pinned_managed(devPtr, count)) {
    return cudaSuccess;
  }
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaMemPrefetchAsync_ptsz)(devPtr, count, location,
                                                flags,
                                                front_real_stream(stream)));
  FRONT_LEAVE();
  return rc;
}

extern "C" cudaError_t
cudaMemAdvise(const void *devPtr, size_t count, enum cudaMemoryAdvise advice,
              struct cudaMemLocation location)
{
  if (front_is_pinned_managed(devPtr, count)) {
    return cudaSuccess;
  }
  FRONT_ENTER();
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaMemAdvise)(devPtr, count, advice, location));
  FRONT_LEAVE();
  return rc;
}

// ---- Pinned host memory --------------------------------------------------

extern "C" cudaError_t
cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
  // cudaHostAllocWriteCombined cannot be reproduced on registered memory.
  unsigned reg = 0;
  if (flags & cudaHostAllocPortable) {
    reg |= cudaHostRegisterPortable;
  }
  if (flags & cudaHostAllocMapped) {
    reg |= cudaHostRegisterMapped;
  }
  return (cudaError_t)front_host_alloc(pHost, size, reg, false);
}

extern "C" cudaError_t
cudaMallocHost(void **ptr, size_t size)
{
  return cudaHostAlloc(ptr, size, cudaHostAllocDefault);
}

extern "C" cudaError_t
cudaFreeHost(void *ptr)
{
  return (cudaError_t)front_host_free(ptr, false);
}

extern "C" cudaError_t
cudaHostRegister(void *ptr, size_t size, unsigned int flags)
{
  return (cudaError_t)front_host_register(ptr, size, flags, false);
}

extern "C" cudaError_t
cudaHostUnregister(void *ptr)
{
  return (cudaError_t)front_host_unregister(ptr, false);
}

// ---- Streams and events --------------------------------------------------

extern "C" cudaError_t
cudaStreamCreate(cudaStream_t *pStream)
{
  return (cudaError_t)front_create_stream(pStream, cudaStreamDefault, 0,
                                          false);
}

extern "C" cudaError_t
cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
{
  return (cudaError_t)front_create_stream(pStream, flags, 0, false);
}

extern "C" cudaError_t
cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags,
                             int priority)
{
  return (cudaError_t)front_create_stream(pStream, flags, priority, false);
}

extern "C" cudaError_t
cudaStreamDestroy(cudaStream_t stream)
{
  return (cudaError_t)front_destroy_stream(stream, false);
}

extern "C" cudaError_t
cudaEventCreate(cudaEvent_t *event)
{
  return (cudaError_t)front_create_event(event, cudaEventDefault, false);
}

extern "C" cudaError_t
cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
  return (cudaError_t)front_create_event(event, flags, false);
}

extern "C" cudaError_t
cudaEventDestroy(cudaEvent_t event)
{
  return (cudaError_t)front_destroy_event(event, false);
}

// ---- Kernels -------------------------------------------------------------

extern "C" void **
__cudaRegisterFatBinary(void *fatCubin)
{
  FRONT_ENTER_REGISTER();
  Fatbin *f = new Fatbin();
  f->fat_cubin = fatCubin;
  f->ended = false;
  FRONT_LH(f->real = REAL(__cudaRegisterFatBinary)(fatCubin));
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    obj_link(f, FRONT_FATBIN);
  }
  FRONT_LEAVE();
  return (void **)f;
}

extern "C" void
__cudaRegisterFatBinaryEnd(void **fatCubinHandle)
{
  FRONT_ENTER_REGISTER();
  Fatbin *f = fatbin_of(fatCubinHandle);
  FRONT_LH(REAL(__cudaRegisterFatBinaryEnd)((void **)f->real));
  f->ended = true;
  FRONT_LEAVE();
}

extern "C" void
__cudaUnregisterFatBinary(void **fatCubinHandle)
{
  FRONT_ENTER_REGISTER();
  Fatbin *f = fatbin_of(fatCubinHandle);
  FRONT_LH(REAL(__cudaUnregisterFatBinary)((void **)f->real));
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    obj_unlink(f);
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
  FRONT_ENTER_REGISTER();
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
  FRONT_ENTER_REGISTER();
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
  FRONT_ENTER_REGISTER();
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
      Ctx *ctx = creating_ctx();
      front_note_cudart(ctx->device);
      std::lock_guard<std::mutex> guard(objs_lock);
      auto it = kernels.find(hostFun);
      if (it != kernels.end()) {
        k = it->second;
      } else {
        k = new Kernel();
        k->real = real;
        k->host_fun = hostFun;
        k->ctx = ctx;
        obj_link(k, FRONT_KERNEL);
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
    int device = current_device();
    std::lock_guard<std::mutex> guard(objs_lock);
    func_settings[{ func, (int)attr }] = { device, false, value };
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
    int device = current_device();
    std::lock_guard<std::mutex> guard(objs_lock);
    func_settings[{ func, -1 }] = { device, true, (int)cacheConfig };
  }
  FRONT_LEAVE();
  return rc;
}

// ---- Checkpoint and restart ----------------------------------------------

// __device__ and __constant__ contents, on each device used (each context
// has its own copy), and the driver's module globals.  Their addresses move
// at restart.
void
front_objs_stash()
{
  for (int device = 0; device < 64; device++) {
    if (!(front_devices_used & (1ULL << device))) {
      continue;
    }
    // Before the lock: front_note_cudart() takes it, for a device whose
    // cudart has not started since a reset (or ever: a driver context's).
    front_use_device(device);
    {
      std::lock_guard<std::mutex> guard(objs_lock);
      for (FrontObj *o = obj_heads[FRONT_FATBIN]; o != NULL; o = o->next) {
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
  front_modules_stash();
  front_arrays_stash();
  libs_stash();
}

void
front_objs_drop_shadows()
{
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    for (FrontObj *o = obj_heads[FRONT_FATBIN]; o != NULL; o = o->next) {
      for (Registration &r : ((Fatbin *)o)->regs) {
        for (auto &entry : r.shadows) {
          free(entry.second);
        }
        r.shadows.clear();
      }
    }
  }
  front_modules_drop_shadows();
  front_arrays_drop_shadows();
  libs_drop();
}

// Before any context exists in the new lower half.  The driver's call makes
// no context, so the contexts come at their place in the creation order.
void
front_objs_restore_devices()
{
  std::lock_guard<std::mutex> guard(objs_lock);
  for (auto &entry : device_flags) {
    CUresult rc;
    FRONT_LH(rc = DRV(cuDevicePrimaryCtxSetFlags_v2)(entry.first,
                                                     entry.second));
    if (rc != CUDA_SUCCESS) {
      front_fatal("restoring the flags of device %d: error %d\n",
                  entry.first, rc);
    }
  }
}

static void
restore_obj(FrontObj *o)
{
  CUresult rc = CUDA_SUCCESS;
  switch (o->kind) {
    case FRONT_DEVICE:
      front_use_device(((DeviceInit *)o)->device);
      load_modules(o->seq);
      break;
    case FRONT_KERNEL: {
      Kernel *k = (Kernel *)o;
      front_ctx_use(k->ctx);
      cudaError_t rrc;
      FRONT_LH(rrc = REAL(__cudaGetKernel)((cudaKernel_t *)&k->real,
                                           k->host_fun));
      if (rrc != cudaSuccess) {
        front_fatal("finding kernel %p again: error %d\n", k->host_fun, rrc);
      }
      break;
    }
    case FRONT_STREAM: {
      Stream *s = (Stream *)o;
      front_ctx_use(s->ctx);
      FRONT_LH(rc = DRV(cuStreamCreateWithPriority)((CUstream *)&s->real,
                                                    s->flags, s->priority));
      if (rc != CUDA_SUCCESS) {
        front_fatal("creating a stream again: error %d\n", rc);
      }
      break;
    }
    case FRONT_EVENT:
      front_ctx_use(((Event *)o)->ctx);
      front_event_restore((Event *)o);
      break;
    case FRONT_ARRAY:
    case FRONT_MIPMAP:
    case FRONT_TEXOBJ:
      front_arrays_restore(o);
      break;
    case FRONT_BLAS:
    case FRONT_BLASLT:
    case FRONT_FFT:
    case FRONT_RAND:
      lib_records[o->kind]->restore(o);
      break;
    default:
      front_drv_restore(o);
  }
}

// The fatbins' registrations first, as the application's constructors made
// them.  Then every other record, in creation order over all kinds: so the
// driver places module data where it was, and a record comes after the
// context that it belongs to (texture objects: see front_arrays.cpp;
// libraries' handles, plans and generators: front_blas.cpp, front_fft.cpp,
// front_rand.cpp).  Then the settings and contents.  (Device memory is
// already mapped, and is filled last: front_vmm.cpp.)  objs_lock is not
// held: the checkpoint thread is the only one running.
void
front_objs_restore()
{
  cudaError_t rc;
  front_ctx_forget();
  for (FrontObj *o = obj_heads[FRONT_FATBIN]; o != NULL; o = o->next) {
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
  front_open_slot();
  std::vector<FrontObj *> order;
  for (int kind : { FRONT_CTX, FRONT_DEVICE, FRONT_MODULE, FRONT_FUNCTION,
                    FRONT_KERNEL, FRONT_STREAM, FRONT_EVENT, FRONT_ARRAY,
                    FRONT_MIPMAP, FRONT_TEXOBJ, FRONT_BLAS, FRONT_BLASLT,
                    FRONT_FFT, FRONT_RAND }) {
    for (FrontObj *o = obj_heads[kind]; o != NULL; o = o->next) {
      order.push_back(o);
    }
  }
  std::sort(order.begin(), order.end(),
            [](FrontObj *a, FrontObj *b) { return a->seq < b->seq; });
  for (FrontObj *o : order) {
    restore_obj(o);
  }
  for (const FrontLibRecords *records : lib_records) {
    if (records != NULL) {
      records->settle();
    }
  }
  front_arrays_fill();
  for (auto &entry : func_settings) {
    front_use_device(entry.second.device);
    set_func(entry.first.first, entry.first.second, entry.second);
  }
  for (int device = 0; device < 64; device++) {
    if (!(front_devices_used & (1ULL << device))) {
      continue;
    }
    front_use_device(device);
    for (FrontObj *o = obj_heads[FRONT_FATBIN]; o != NULL; o = o->next) {
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
  for (auto &entry : symbol_addresses) {
    front_use_device(entry.first.first);
    void *now = NULL;
    FRONT_LH(rc = REAL(cudaGetSymbolAddress)(&now, entry.first.second));
    if (rc != cudaSuccess || now != entry.second) {
      front_fatal("the __device__ variable at host address %p moved from %p "
                  "to %p on device %d, and the application has its "
                  "address\n", entry.first.second, entry.second, now,
                  entry.first.first);
    }
  }
  for (auto &entry : peer_access) {
    front_use_device(entry.first.first);
    FRONT_LH(rc = REAL(cudaDeviceEnablePeerAccess)(entry.first.second,
                                                   entry.second));
    if (rc != cudaSuccess && rc != cudaErrorPeerAccessAlreadyEnabled) {
      front_fatal("enabling peer access from device %d to %d again: error "
                  "%d\n", entry.first.first, entry.first.second, rc);
    }
  }
  CUresult drc;
  for (auto &entry : pinned) {
    front_ctx_use(entry.second.ctx);
    FRONT_LH(drc = DRV(cuMemHostRegister_v2)(entry.first, entry.second.size,
                                             entry.second.flags));
    if (drc != CUDA_SUCCESS) {
      front_fatal("registering host memory %p again: error %d\n",
                  entry.first, drc);
    }
  }
  front_log("restored the CUDA objects\n");
}
