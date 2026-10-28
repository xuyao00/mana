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

// The application's CUDA IPC (cuda/DESIGN-lower-half-cuda.md, section 9),
// between processes of the MANA job on one node.  The handle names the
// exporter by its DMTCP virtual pid and the allocator's chunk by its address,
// which stay the same across restarts; the chunk is a VMM allocation,
// shared as a POSIX fd that the importer takes with pidfd_getfd().  The real
// pid and the fd in the handle are valid in its epoch (the restart
// generation) only.  At restart, each exporter exports again and publishes
// (virtual pid, chunk) -> (host, real pid, fd) in DMTCP's key-value store;
// after a global barrier, importers map the chunks again, at the same
// addresses.  Imported memory is not saved: its exporter saves it.

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "front_objs.h"

// libmana.so (mpi_plugin.cpp): DMTCP's key-value store, which takes one
// caller at a time from the application's threads.
extern "C" int mana_kv_set(const char *db, const char *key, const char *val);
extern "C" int mana_kv_get(const char *db, const char *key, char *val,
                           size_t size);

namespace {

#define IPC_MAGIC 0x4d414e41u  // "MANA"

// In the 64 bytes of a cudaIpcMemHandle_t or CUipcMemHandle.
struct IpcHandle {
  uint32_t magic;
  uint32_t epoch;          // the exporter's generation when it made it
  uint64_t vpid;           // the exporter's DMTCP virtual pid
  uint64_t chunk;          // the chunk's address
  uint64_t chunk_size;
  uint64_t offset;         // of the allocation in the chunk
  uint64_t host;           // the exporter's host (boot id)
  int32_t real_pid;        // in 'epoch' only
  int32_t fd;              // in the exporter, in 'epoch' only
};

static_assert(sizeof(IpcHandle) <= sizeof(cudaIpcMemHandle_t),
              "an IPC handle fits");
static_assert(sizeof(cudaIpcMemHandle_t) == sizeof(CUipcMemHandle),
              "both APIs' handles are alike");

struct Import {
  CUdeviceptr va;
  size_t size;
  int device;
  CUmemGenericAllocationHandle handle;  // of this generation
  int refs;
};

typedef std::pair<uint64_t, uint64_t> ImportKey;  // (vpid, chunk)

std::mutex ipc_lock;
std::map<ImportKey, Import> imports;
std::map<CUdeviceptr, ImportKey> imports_by_va;
std::mutex kv_lock;

// Without DMTCP's wrappers, which would take these pids for virtual ones.
long
raw_syscall(long nr, long a, long b, long c)
{
  long ret;
  asm volatile("syscall"
               : "=a"(ret)
               : "0"(nr), "D"(a), "S"(b), "d"(c)
               : "rcx", "r11", "memory");
  return ret;
}

int
real_pid()
{
  return (int)raw_syscall(SYS_getpid, 0, 0, 0);
}

uint64_t
host_id()
{
  static uint64_t id = 0;
  if (id == 0) {
    char boot[64] = "";
    int fd = open("/proc/sys/kernel/random/boot_id", O_RDONLY);
    if (fd >= 0) {
      ssize_t n = read(fd, boot, sizeof boot - 1);
      boot[n > 0 ? n : 0] = '\0';
      close(fd);
    }
    id = 1469598103934665603ULL;  // FNV-1a
    for (const char *p = boot; *p != '\0'; p++) {
      id = (id ^ (unsigned char)*p) * 1099511628211ULL;
    }
  }
  return id;
}

void
kv_db(char *db, size_t size)
{
  snprintf(db, size, "mana-cuda-ipc-%u", front_gen);
}

void
kv_key(char *key, size_t size, uint64_t vpid, uint64_t chunk)
{
  snprintf(key, size, "%llu:%#llx", (unsigned long long)vpid,
           (unsigned long long)chunk);
}

// The exporter of (vpid, chunk) now, as it published it at the last restart.
bool
look_up(uint64_t vpid, uint64_t chunk, uint64_t *host, int *pid, int *fd)
{
  char db[64], key[64], val[128];
  kv_db(db, sizeof db);
  kv_key(key, sizeof key, vpid, chunk);
  std::lock_guard<std::mutex> guard(kv_lock);
  unsigned long long h;
  return mana_kv_get(db, key, val, sizeof val) == 0 &&
         sscanf(val, "%llu:%d:%d", &h, pid, fd) == 3 && (*host = h, true);
}

// Maps the exporter's chunk at 'va' (0: a new address), for 'device'.
CUresult
import_chunk(int pid, int fd, size_t size, int device, CUdeviceptr *va,
             CUmemGenericAllocationHandle *handle)
{
  int pidfd = (int)raw_syscall(SYS_pidfd_open, pid, 0, 0);
  if (pidfd < 0) {
    front_log("pidfd_open(%d): %d\n", pid, -pidfd);
    return CUDA_ERROR_INVALID_VALUE;
  }
  int local = (int)raw_syscall(SYS_pidfd_getfd, pidfd, fd, 0);
  raw_syscall(SYS_close, pidfd, 0, 0);
  if (local < 0) {
    // EPERM: Yama's ptrace_scope (cuda/DESIGN-lower-half-cuda.md, Risks).
    front_log("pidfd_getfd(%d, %d): %d\n", pid, fd, -local);
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUresult rc;
  FRONT_LH(rc = DRV(cuMemImportFromShareableHandle)(
             handle, (void *)(uintptr_t)local,
             CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  raw_syscall(SYS_close, local, 0, 0);
  if (rc != CUDA_SUCCESS) {
    return rc;
  }
  bool new_va = *va == 0;
  if (new_va && (*va = front_vmm_take_va(size)) == 0) {
    FRONT_LH(DRV(cuMemRelease)(*handle));
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  FRONT_LH(rc = DRV(cuMemMap)(*va, size, 0, *handle, 0));
  if (rc == CUDA_SUCCESS) {
    CUmemAccessDesc access;
    memset(&access, 0, sizeof access);
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    FRONT_LH(rc = DRV(cuMemSetAccess)(*va, size, &access, 1));
    if (rc != CUDA_SUCCESS) {
      FRONT_LH(DRV(cuMemUnmap)(*va, size));
    }
  }
  if (rc != CUDA_SUCCESS) {
    FRONT_LH(DRV(cuMemRelease)(*handle));
    if (new_va) {
      front_vmm_give_back_va(*va, size);
    }
  }
  return rc;
}

int
get_handle(void *handle, void *ptr)
{
  if (handle == NULL) {
    return cudaErrorInvalidValue;
  }
  FRONT_ENTER();
  FrontExport e;
  bool ours = front_vmm_export(ptr, &e);
  FRONT_LEAVE();
  if (!ours) {
    // Not the base of a live allocation of the application.
    return cudaErrorInvalidValue;
  }
  IpcHandle h;
  memset(&h, 0, sizeof h);
  h.magic = IPC_MAGIC;
  h.epoch = front_gen;
  h.vpid = (uint64_t)getpid();
  h.chunk = e.base;
  h.chunk_size = e.size;
  h.offset = (CUdeviceptr)ptr - e.base;
  h.host = host_id();
  h.real_pid = real_pid();
  h.fd = e.fd;
  memset(handle, 0, sizeof(cudaIpcMemHandle_t));
  memcpy(handle, &h, sizeof h);
  return 0;
}

int
open_handle(void **ptr, const void *handle)
{
  if (ptr == NULL) {
    return cudaErrorInvalidValue;
  }
  IpcHandle h;
  memcpy(&h, handle, sizeof h);
  if (h.magic != IPC_MAGIC) {
    front_unsupported("CUDA IPC with a process outside the MANA job");
  }
  if (h.vpid == (uint64_t)getpid()) {
    return cudaErrorInvalidValue;  // as natively, for the process's own
  }
  ImportKey key(h.vpid, h.chunk);
  FRONT_ENTER();
  std::lock_guard<std::mutex> guard(ipc_lock);
  int rc = 0;
  auto it = imports.find(key);
  if (it == imports.end()) {
    uint64_t host = h.host;
    int pid = h.real_pid, fd = h.fd;
    if (h.epoch != front_gen && !look_up(h.vpid, h.chunk, &host, &pid, &fd)) {
      front_log("no exporter of %#llx in pid %llu now\n",
                (unsigned long long)h.chunk, (unsigned long long)h.vpid);
      rc = cudaErrorInvalidValue;
    } else if (host != host_id()) {
      rc = cudaErrorInvalidValue;  // another node
    } else {
      Import imp = { 0, h.chunk_size, 0, 0, 0 };
      FRONT_LH(REAL(cudaGetDevice)(&imp.device));
      rc = import_chunk(pid, fd, imp.size, imp.device, &imp.va, &imp.handle);
      if (rc == 0) {
        it = imports.emplace(key, imp).first;
        imports_by_va[imp.va] = key;
      }
    }
  }
  if (rc == 0) {
    it->second.refs++;
    *ptr = (void *)(it->second.va + h.offset);
  }
  FRONT_LEAVE();
  return rc;
}

int
close_handle(void *ptr)
{
  CUdeviceptr p = (CUdeviceptr)ptr;
  FRONT_ENTER();
  std::lock_guard<std::mutex> guard(ipc_lock);
  int rc = cudaErrorInvalidValue;
  auto by_va = imports_by_va.upper_bound(p);
  if (by_va != imports_by_va.begin()) {
    --by_va;
    auto it = imports.find(by_va->second);
    Import &imp = it->second;
    if (p < imp.va + imp.size) {
      rc = 0;
      if (--imp.refs == 0) {
        FRONT_LH(DRV(cuMemUnmap)(imp.va, imp.size));
        FRONT_LH(DRV(cuMemRelease)(imp.handle));
        front_vmm_give_back_va(imp.va, imp.size);
        imports_by_va.erase(by_va);
        imports.erase(it);
      }
    }
  }
  FRONT_LEAVE();
  return rc;
}

void
publish(const FrontExport &e)
{
  char db[64], key[64], val[128];
  kv_db(db, sizeof db);
  kv_key(key, sizeof key, (uint64_t)getpid(), e.base);
  snprintf(val, sizeof val, "%llu:%d:%d", (unsigned long long)host_id(),
           real_pid(), e.fd);
  std::lock_guard<std::mutex> guard(kv_lock);
  if (mana_kv_set(db, key, val) != 0) {
    front_fatal("publishing the CUDA IPC export %s\n", key);
  }
}

}  // namespace

// At restart, before the global barrier: the chunks that were exported.
void
front_ipc_publish()
{
  front_vmm_reexport(publish);
}

// After it: the imported chunks, at their addresses.  A rank that shared
// memory with this one must be on this node again.
void
front_ipc_reimport()
{
  std::lock_guard<std::mutex> guard(ipc_lock);
  for (auto &entry : imports) {
    Import &imp = entry.second;
    uint64_t host;
    int pid, fd;
    if (!look_up(entry.first.first, entry.first.second, &host, &pid, &fd)) {
      front_fatal("rank with pid %llu did not export %#llx again\n",
                  (unsigned long long)entry.first.first,
                  (unsigned long long)entry.first.second);
    }
    if (host != host_id()) {
      front_fatal("the restart put the ranks that share memory through CUDA "
                  "IPC on different nodes (pid %llu)\n",
                  (unsigned long long)entry.first.first);
    }
    CUresult rc = (CUresult)import_chunk(pid, fd, imp.size, imp.device,
                                         &imp.va, &imp.handle);
    if (rc != CUDA_SUCCESS) {
      front_fatal("mapping CUDA IPC memory of pid %llu again: error %d\n",
                  (unsigned long long)entry.first.first, rc);
    }
  }
}

// ---- The runtime's and the driver's functions ----------------------------

extern "C" cudaError_t
cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr)
{
  return (cudaError_t)get_handle(handle, devPtr);
}

extern "C" cudaError_t
cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, unsigned int)
{
  return (cudaError_t)open_handle(devPtr, &handle);
}

extern "C" cudaError_t
cudaIpcCloseMemHandle(void *devPtr)
{
  return (cudaError_t)close_handle(devPtr);
}

extern "C" CUresult
cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr)
{
  return (CUresult)get_handle(pHandle, (void *)dptr);
}

extern "C" CUresult
cuIpcOpenMemHandle_v2(CUdeviceptr *pdptr, CUipcMemHandle handle,
                      unsigned int)
{
  return (CUresult)open_handle((void **)pdptr, &handle);
}

extern "C" CUresult
cuIpcCloseMemHandle(CUdeviceptr dptr)
{
  return (CUresult)close_handle((void *)dptr);
}
