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

// The device allocator.  Every device allocation of the application is VMM
// memory in the range that the lower half reserved at a fixed address
// (lh_info->cuda_vmm_base), so that a restart maps new physical memory at the
// same addresses and no allocation history is replayed (from crac-old's
// vmm-ckpt.cpp).  Its records are upper-half memory, saved with the image.
//
// Allocations under SMALL_MAX are blocks of a size class (powers of two from
// 512 bytes) in chunks of the granularity (2 MB), as in cuda-aware-pr's
// vmm-alloc.c: a chunk is an allocation of its own, which stays mapped.  So
// CU_POINTER_ATTRIBUTE_RANGE_START/SIZE, and a CUDA IPC export, cover the
// chunk.
//
// At a checkpoint, the contents go into shadows in the upper half, which
// DMTCP saves with the image, or with MANA_CUDA_DEVICE_DATA=file into
// cuda_memory.mana in the checkpoint directory, through a pinned buffer of
// the lower half: then no host memory holds them (Perlmutter's 4 GPUs have
// 320 GB, its nodes 256 GB).

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <map>
#include <mutex>
#include <string>
#include "front.h"

namespace {

struct Alloc {
  size_t size;                          // mapped size, granularity multiple
  size_t requested;
  int device;
  CUmemGenericAllocationHandle handle;  // of generation 'gen'
  void *shadow;                         // device contents during a checkpoint
  bool in_file;                         // or they are in the device file
  bool live;                            // false: freed, kept mapped for reuse
  int export_fd;                        // CUDA IPC: of generation export_gen
  unsigned export_gen;                  // 0: never exported
};

#define SMALL_MIN 512
#define SMALL_MAX (1 << 20)

// The blocks of one chunk of small allocations.
struct Chunk {
  size_t block;                  // the size class
  int device;
  uint32_t n_blocks;
  uint32_t n_free;
  std::vector<uint64_t> used;    // a bit per block
  std::vector<size_t> requested; // per block, for the checkpoint
};

#define DEVICE_FILE "cuda_memory.mana"
#define DEVICE_FILE_MAGIC 0x414455434e414d00ULL  // "\0MANCUDA"
#define DEVICE_FILE_BUFFER (64 << 20)

// One allocation's contents in the device file, after this header.
struct FileRecord {
  uint64_t addr;
  uint64_t size;
};

// cudaFreeAsync: a block is free once its stream has passed this event (of
// the lower half; none is left at a checkpoint).
struct PendingFree {
  CUdeviceptr ptr;
  CUevent event;
};

std::mutex vmm_lock;
std::vector<PendingFree> pending_frees;
CUdeviceptr va_base;
size_t va_size;
size_t va_top;                                 // bump offset into the range
std::map<CUdeviceptr, Alloc> allocs;           // by address
std::map<CUdeviceptr, size_t> holes;           // free VA below va_top
std::map<CUdeviceptr, Chunk> chunks;           // by address; also in allocs
size_t granularity[64];
int rdma_capable[64];                          // 1 yes, 0 no, -1 not known
unsigned device_ready_gen[64];                 // cudart has a context

#define DRV_CHECK(call)                                                   \
  do {                                                                    \
    CUresult drv_rc_;                                                     \
    FRONT_LH(drv_rc_ = (call));                                           \
    if (drv_rc_ != CUDA_SUCCESS) {                                        \
      front_fatal("%s failed: %d (%s:%d)\n", #call, drv_rc_, __FILE__,    \
                  __LINE__);                                              \
    }                                                                     \
  } while (0)

void
init_range()
{
  if (va_size == 0) {
    va_base = lh_info->cuda_vmm_base;
    va_size = lh_info->cuda_vmm_size;
    for (int i = 0; i < 64; i++) {
      rdma_capable[i] = -1;
    }
  }
}

CUmemAllocationProp
alloc_prop(int device)
{
  CUmemAllocationProp prop;
  memset(&prop, 0, sizeof prop);
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  // UCX's cuda_ipc shares such memory between processes of a node.
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  if (rdma_capable[device] < 0) {
    int value = 0;
    CUresult rc;
    FRONT_LH(rc = DRV(cuDeviceGetAttribute)(
      &value, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
      device));
    rdma_capable[device] = rc == CUDA_SUCCESS && value;
  }
  prop.allocFlags.gpuDirectRDMACapable = rdma_capable[device];
  return prop;
}

size_t
device_granularity(int device)
{
  if (granularity[device] == 0) {
    CUmemAllocationProp prop = alloc_prop(device);
    DRV_CHECK(DRV(cuMemGetAllocationGranularity)(
      &granularity[device], &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  }
  return granularity[device];
}

// Bit d of peers[x]: device d has peer access to device x's memory
// (cudaDeviceEnablePeerAccess), so the allocator grants it to d too.
uint64_t peers[64];

// The owner and its peers may read and write [va, va + size).
void
set_access(CUdeviceptr va, size_t size, int device)
{
  CUmemAccessDesc access[64];
  int n = 0;
  for (int d = 0; d < 64; d++) {
    if (d == device || (peers[device] & (1ULL << d))) {
      memset(&access[n], 0, sizeof access[n]);
      access[n].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      access[n].location.id = d;
      access[n].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      n++;
    }
  }
  DRV_CHECK(DRV(cuMemSetAccess)(va, size, access, n));
}

// Creates physical memory on 'device' and maps it at 'va'.
CUresult
map_physical(CUdeviceptr va, size_t size, int device,
             CUmemGenericAllocationHandle *handle)
{
  CUmemAllocationProp prop = alloc_prop(device);
  CUresult rc;
  FRONT_LH(rc = DRV(cuMemCreate)(handle, size, &prop, 0));
  if (rc != CUDA_SUCCESS) {
    return rc;
  }
  DRV_CHECK(DRV(cuMemMap)(va, size, 0, *handle, 0));
  set_access(va, size, device);
  return CUDA_SUCCESS;
}

// First fit over the holes, otherwise bump.
CUdeviceptr
take_va(size_t size)
{
  for (auto it = holes.begin(); it != holes.end(); ++it) {
    if (it->second >= size) {
      CUdeviceptr va = it->first;
      size_t rest = it->second - size;
      holes.erase(it);
      if (rest) {
        holes[va + size] = rest;
      }
      return va;
    }
  }
  if (va_top + size > va_size) {
    return 0;
  }
  CUdeviceptr va = va_base + va_top;
  va_top += size;
  return va;
}

void
give_back_va(CUdeviceptr va, size_t size)
{
  auto next = holes.lower_bound(va);
  if (next != holes.end() && va + size == next->first) {
    size += next->second;
    holes.erase(next);
  }
  auto prev = holes.lower_bound(va);
  if (prev != holes.begin()) {
    --prev;
    if (prev->first + prev->second == va) {
      prev->second += size;
      return;
    }
  }
  holes[va] = size;
}

// cudart's context for the current device, which the copies need.
void
ready_device(int device)
{
  if (device_ready_gen[device] != front_gen) {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaFree)(NULL));
    if (rc != cudaSuccess) {
      front_fatal("initializing device %d: error %d\n", device, rc);
    }
    device_ready_gen[device] = front_gen;
    front_note_cudart(device);
  }
}

}  // namespace

static cudaError_t big_alloc_locked(void **ptr, size_t size, int device);
static cudaError_t small_alloc_locked(void **ptr, size_t size, int device);
static void reap_locked(bool all);

cudaError_t
front_vmm_alloc(void **ptr, size_t size)
{
  if (ptr == NULL) {
    return cudaErrorInvalidValue;
  }
  if (size == 0) {
    *ptr = NULL;
    return cudaSuccess;
  }
  int device = 0;
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGetDevice)(&device));
  if (rc != cudaSuccess) {
    return rc;
  }
  if (device < 0 || device >= 64) {
    return cudaErrorInvalidDevice;
  }
  std::lock_guard<std::mutex> guard(vmm_lock);
  init_range();
  ready_device(device);
  if (!pending_frees.empty()) {
    reap_locked(false);
  }
  if (size < SMALL_MAX) {
    return small_alloc_locked(ptr, size, device);
  }
  return big_alloc_locked(ptr, size, device);
}

// cudaMallocPitch, cuMemAllocPitch.
cudaError_t
front_vmm_alloc_pitch(void **ptr, size_t *pitch, size_t width, size_t height)
{
  static size_t alignment = 0;
  if (alignment == 0) {
    int device = 0, value = 0;
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaGetDevice)(&device));
    if (rc == cudaSuccess) {
      FRONT_LH(rc = REAL(cudaDeviceGetAttribute)(
                 &value, cudaDevAttrTexturePitchAlignment, device));
    }
    alignment = (rc == cudaSuccess && value > 0) ? value : 512;
  }
  *pitch = (width + alignment - 1) / alignment * alignment;
  return front_vmm_alloc(ptr, *pitch * height);
}

cudaError_t
big_alloc_locked(void **ptr, size_t size, int device)
{
  size_t rounded = (size + device_granularity(device) - 1) /
                   device_granularity(device) * device_granularity(device);
  // A freed chunk of the same size and device.  Freed chunks stay mapped:
  // the MPI's registration caches may hold their physical memory.
  for (auto &entry : allocs) {
    Alloc &a = entry.second;
    if (!a.live && a.size == rounded && a.device == device) {
      a.live = true;
      a.requested = size;
      *ptr = (void *)entry.first;
      return cudaSuccess;
    }
  }
  CUdeviceptr va = take_va(rounded);
  if (va == 0) {
    return cudaErrorMemoryAllocation;
  }
  Alloc a = { rounded, size, device, 0, NULL, false, true, -1, 0 };
  if (map_physical(va, rounded, device, &a.handle) != CUDA_SUCCESS) {
    give_back_va(va, rounded);
    return cudaErrorMemoryAllocation;
  }
  allocs[va] = a;
  *ptr = (void *)va;
  front_log("alloc %#llx: %zu bytes on device %d\n", (unsigned long long)va,
            size, device);
  return cudaSuccess;
}

static cudaError_t
small_alloc_locked(void **ptr, size_t size, int device)
{
  size_t block = SMALL_MIN;
  while (block < size) {
    block <<= 1;
  }
  Chunk *chunk = NULL;
  CUdeviceptr base = 0;
  for (auto &entry : chunks) {
    if (entry.second.block == block && entry.second.device == device &&
        entry.second.n_free > 0) {
      chunk = &entry.second;
      base = entry.first;
      break;
    }
  }
  if (chunk == NULL) {
    // Always a new mapping: a freed big allocation may be reused only whole.
    size_t chunk_size = device_granularity(device);
    base = take_va(chunk_size);
    if (base == 0) {
      return cudaErrorMemoryAllocation;
    }
    Alloc a = { chunk_size, chunk_size, device, 0, NULL, false, true, -1,
                0 };
    if (map_physical(base, chunk_size, device, &a.handle) != CUDA_SUCCESS) {
      give_back_va(base, chunk_size);
      return cudaErrorMemoryAllocation;
    }
    allocs[base] = a;
    Chunk c;
    c.block = block;
    c.device = device;
    c.n_blocks = chunk_size / block;
    c.n_free = c.n_blocks;
    c.used.assign((c.n_blocks + 63) / 64, 0);
    c.requested.assign(c.n_blocks, 0);
    chunk = &(chunks[base] = c);
    front_log("chunk %#llx for %zu-byte blocks on device %d\n",
              (unsigned long long)base, block, device);
  }
  for (uint32_t i = 0; i < chunk->n_blocks; i++) {
    if (!(chunk->used[i / 64] & (1ULL << (i % 64)))) {
      chunk->used[i / 64] |= 1ULL << (i % 64);
      chunk->requested[i] = size;
      chunk->n_free--;
      *ptr = (void *)(base + i * block);
      return cudaSuccess;
    }
  }
  front_fatal("chunk %#llx has no free block\n", (unsigned long long)base);
}

// Whether 'p' is a live allocation of the allocator, not being freed.
static bool
is_live_locked(CUdeviceptr p)
{
  auto it = allocs.upper_bound(p);
  if (it == allocs.begin()) {
    return false;
  }
  --it;
  bool live;
  auto chunk = chunks.find(it->first);
  if (chunk != chunks.end()) {
    Chunk &c = chunk->second;
    if (p >= it->first + it->second.size || (p - it->first) % c.block != 0) {
      return false;
    }
    uint32_t index = (p - it->first) / c.block;
    live = c.used[index / 64] & (1ULL << (index % 64));
  } else {
    live = it->first == p && it->second.live;
  }
  for (const PendingFree &f : pending_frees) {
    live = live && f.ptr != p;
  }
  return live;
}

static void
release_locked(CUdeviceptr p)
{
  auto it = --allocs.upper_bound(p);
  auto chunk = chunks.find(it->first);
  if (chunk != chunks.end()) {
    Chunk &c = chunk->second;
    uint32_t index = (p - it->first) / c.block;
    c.used[index / 64] &= ~(1ULL << (index % 64));
    c.requested[index] = 0;
    c.n_free++;
  } else {
    it->second.live = false;
  }
}

// The blocks of cudaFreeAsync that their streams have passed (all, after the
// checkpoint's synchronization).
static void
reap_locked(bool all)
{
  for (auto it = pending_frees.begin(); it != pending_frees.end();) {
    CUresult rc = CUDA_SUCCESS;
    if (!all) {
      FRONT_LH(rc = DRV(cuEventQuery)(it->event));
    }
    if (rc == CUDA_ERROR_NOT_READY) {
      ++it;
      continue;
    }
    FRONT_LH(DRV(cuEventDestroy_v2)(it->event));
    release_locked(it->ptr);
    it = pending_frees.erase(it);
  }
}

bool
front_vmm_free(void *ptr, cudaError_t *result)
{
  CUdeviceptr p = (CUdeviceptr)ptr;
  {
    std::lock_guard<std::mutex> guard(vmm_lock);
    if (!is_live_locked(p)) {
      return false;  // not ours, or freed already: the driver's error
    }
  }
  // cudaFree() waits for the device's work.
  FRONT_LH(*result = REAL(cudaDeviceSynchronize)());
  std::lock_guard<std::mutex> guard(vmm_lock);
  if (is_live_locked(p)) {
    release_locked(p);
  }
  return true;
}

bool
front_vmm_free_async(void *ptr, cudaStream_t real_stream, int *result)
{
  CUdeviceptr p = (CUdeviceptr)ptr;
  std::unique_lock<std::mutex> guard(vmm_lock);
  if (!is_live_locked(p)) {
    return false;
  }
  CUevent event = NULL;
  CUresult rc;
  FRONT_LH(rc = DRV(cuEventCreate)(&event, CU_EVENT_DISABLE_TIMING));
  if (rc == CUDA_SUCCESS) {
    FRONT_LH(rc = DRV(cuEventRecord)(event, real_stream));
  }
  if (rc != CUDA_SUCCESS) {
    // No event: wait now.
    if (event != NULL) {
      FRONT_LH(DRV(cuEventDestroy_v2)(event));
    }
    guard.unlock();
    cudaError_t sync_rc;
    bool ours = front_vmm_free(ptr, &sync_rc);
    *result = sync_rc;
    return ours;
  }
  pending_frees.push_back({ p, event });
  *result = 0;
  return true;
}

static bool
device_data_to_file()
{
  const char *where = getenv("MANA_CUDA_DEVICE_DATA");
  return where != NULL && strcmp(where, "file") == 0;
}

void
front_vmm_stash()
{
  std::lock_guard<std::mutex> guard(vmm_lock);
  reap_locked(true);
  bool to_file = device_data_to_file();
  size_t bytes = 0;
  for (auto &entry : allocs) {
    Alloc &a = entry.second;
    if (!a.live) {
      continue;
    }
    auto chunk = chunks.find(entry.first);
    if (chunk != chunks.end()) {
      // Up to the end of the last block in use.
      size_t end = 0;
      for (uint32_t i = 0; i < chunk->second.n_blocks; i++) {
        if (chunk->second.requested[i] != 0) {
          end = i * chunk->second.block + chunk->second.requested[i];
        }
      }
      a.requested = end;
      if (end == 0) {
        continue;
      }
    }
    if (to_file) {
      a.in_file = true;  // front_vmm_save(); the GPU is idle until resume
      bytes += a.requested;
      continue;
    }
    a.shadow = malloc(a.requested);
    if (a.shadow == NULL) {
      front_fatal("no host memory for a %zu-byte device shadow\n",
                  a.requested);
    }
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaMemcpy)(a.shadow, (void *)entry.first,
                                   a.requested, cudaMemcpyDeviceToHost));
    if (rc != cudaSuccess) {
      front_fatal("saving device memory at %#llx: error %d\n",
                  (unsigned long long)entry.first, rc);
    }
    bytes += a.requested;
  }
  front_log("%s %zu bytes of device memory\n", to_file ? "to save" : "saved",
            bytes);
}

static void
write_all(int fd, const void *buf, size_t size, const std::string &path)
{
  for (size_t done = 0; done < size;) {
    ssize_t n = write(fd, (const char *)buf + done, size - done);
    if (n <= 0) {
      front_fatal("writing %s: %s\n", path.c_str(), strerror(errno));
    }
    done += n;
  }
}

static void
read_all(int fd, void *buf, size_t size, const std::string &path)
{
  for (size_t done = 0; done < size;) {
    ssize_t n = read(fd, (char *)buf + done, size - done);
    if (n <= 0) {
      front_fatal("reading %s: %s\n", path.c_str(),
                  n == 0 ? "too short" : strerror(errno));
    }
    done += n;
  }
}

static void *
pinned_buffer()
{
  void *buf = NULL;
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaHostAlloc)(&buf, DEVICE_FILE_BUFFER,
                                    cudaHostAllocPortable));
  if (rc != cudaSuccess) {
    front_fatal("no pinned buffer for the device file: error %d\n", rc);
  }
  return buf;
}

static void
free_pinned_buffer(void *buf)
{
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaFreeHost)(buf));
  (void)rc;
}

// MANA_STEP_SAVE, with MANA_CUDA_DEVICE_DATA=file: the contents that
// front_vmm_stash() left on the device, into the checkpoint directory.
void
front_vmm_save()
{
  std::lock_guard<std::mutex> guard(vmm_lock);
  uint64_t count = 0;
  for (auto &entry : allocs) {
    count += entry.second.in_file;
  }
  if (count == 0) {
    return;
  }
  std::string path = std::string(mana_ckpt_dir()) + "/" DEVICE_FILE;
  int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (fd < 0) {
    front_fatal("cannot create %s: %s\n", path.c_str(), strerror(errno));
  }
  uint64_t header[2] = { DEVICE_FILE_MAGIC, count };
  write_all(fd, header, sizeof header, path);
  void *buf = pinned_buffer();
  size_t bytes = 0;
  for (auto &entry : allocs) {
    Alloc &a = entry.second;
    if (!a.in_file) {
      continue;
    }
    FileRecord record = { entry.first, a.requested };
    write_all(fd, &record, sizeof record, path);
    for (size_t done = 0; done < a.requested;) {
      size_t n = a.requested - done < DEVICE_FILE_BUFFER
                   ? a.requested - done : DEVICE_FILE_BUFFER;
      cudaError_t rc;
      FRONT_LH(rc = REAL(cudaMemcpy)(buf, (void *)(entry.first + done), n,
                                     cudaMemcpyDeviceToHost));
      if (rc != cudaSuccess) {
        front_fatal("saving device memory at %#llx: error %d\n",
                    (unsigned long long)entry.first, rc);
      }
      write_all(fd, buf, n, path);
      done += n;
    }
    bytes += a.requested;
  }
  free_pinned_buffer(buf);
  if (close(fd) != 0) {
    front_fatal("writing %s: %s\n", path.c_str(), strerror(errno));
  }
  front_log("saved %zu bytes of device memory in %s\n", bytes, path.c_str());
}

void
front_vmm_drop_shadows()
{
  std::lock_guard<std::mutex> guard(vmm_lock);
  for (auto &entry : allocs) {
    free(entry.second.shadow);
    entry.second.shadow = NULL;
    entry.second.in_file = false;
  }
}

// At restart, in the new lower half, before any context exists: the same
// range, and new physical memory at each live allocation (VMM calls need no
// context).  Freed chunks are not kept.
void
front_vmm_map()
{
  std::lock_guard<std::mutex> guard(vmm_lock);
  for (int i = 0; i < 64; i++) {
    rdma_capable[i] = -1;
  }
  if (va_size == 0) {
    return;  // nothing was allocated
  }
  if (lh_info->cuda_vmm_base != va_base || lh_info->cuda_vmm_size != va_size) {
    front_fatal("the device range moved: %#llx (%#llx bytes) at checkpoint, "
                "%#llx (%#llx bytes) now\n", (unsigned long long)va_base,
                (unsigned long long)va_size, lh_info->cuda_vmm_base,
                lh_info->cuda_vmm_size);
  }
  for (auto it = allocs.begin(); it != allocs.end();) {
    if (!it->second.live) {
      give_back_va(it->first, it->second.size);
      it = allocs.erase(it);
      continue;
    }
    Alloc &a = it->second;
    if (map_physical(it->first, a.size, a.device, &a.handle) !=
        CUDA_SUCCESS) {
      front_fatal("no device memory for %zu bytes at %#llx on device %d\n",
                  a.size, (unsigned long long)it->first, a.device);
    }
    ++it;
  }
}

static bool
export_locked(CUdeviceptr base, Alloc &a)
{
  if (a.export_gen == front_gen) {
    return true;
  }
  int fd = -1;
  CUresult rc;
  FRONT_LH(rc = DRV(cuMemExportToShareableHandle)(
             &fd, a.handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
  if (rc != CUDA_SUCCESS) {
    front_log("exporting %#llx: error %d\n", (unsigned long long)base, rc);
    return false;
  }
  a.export_fd = fd;
  a.export_gen = front_gen;
  return true;
}

// A live allocation, or the base of a chunk of small ones: as natively, a
// handle stands for the allocation that holds the pointer, which
// cuMemGetAddressRange gives.
bool
front_vmm_export(void *ptr, FrontExport *out)
{
  std::lock_guard<std::mutex> guard(vmm_lock);
  CUdeviceptr p = (CUdeviceptr)ptr;
  auto it = allocs.find(p);
  bool chunk_base = it != allocs.end() && chunks.count(p) != 0;
  if (!chunk_base && !is_live_locked(p)) {
    return false;
  }
  it = --allocs.upper_bound(p);
  if (!export_locked(it->first, it->second)) {
    return false;
  }
  *out = { it->first, it->second.size, it->second.export_fd };
  return true;
}

void
front_vmm_reexport(void (*publish)(const FrontExport &))
{
  std::lock_guard<std::mutex> guard(vmm_lock);
  for (auto &entry : allocs) {
    Alloc &a = entry.second;
    if (a.export_gen != 0 && export_locked(entry.first, a)) {
      publish({ entry.first, a.size, a.export_fd });
    }
  }
}

CUdeviceptr
front_vmm_take_va(size_t size)
{
  std::lock_guard<std::mutex> guard(vmm_lock);
  init_range();
  return take_va(size);
}

void
front_vmm_give_back_va(CUdeviceptr va, size_t size)
{
  std::lock_guard<std::mutex> guard(vmm_lock);
  give_back_va(va, size);
}

// Peer access from 'device' to 'peer''s memory, on or off: also for the
// memory that is mapped already.
void
front_vmm_peer_access(int device, int peer, bool on)
{
  if (device < 0 || device >= 64 || peer < 0 || peer >= 64) {
    return;
  }
  std::lock_guard<std::mutex> guard(vmm_lock);
  if (on) {
    peers[peer] |= 1ULL << device;
  } else {
    peers[peer] &= ~(1ULL << device);
  }
  for (auto &entry : allocs) {
    if (entry.second.device != peer) {
      continue;
    }
    set_access(entry.first, entry.second.size, peer);
    if (!on) {
      CUmemAccessDesc none;
      memset(&none, 0, sizeof none);
      none.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      none.location.id = device;
      none.flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
      DRV_CHECK(DRV(cuMemSetAccess)(entry.first, entry.second.size, &none,
                                    1));
    }
  }
}

// The contents that front_vmm_save() wrote, from the device file next to the
// image being restored.
static void
fill_from_file()
{
  uint64_t count = 0;
  for (auto &entry : allocs) {
    count += entry.second.in_file;
  }
  if (count == 0) {
    return;
  }
  if (lh_info->restart_image == NULL) {
    front_fatal("device memory is in a file, but the lower half names no "
                "image\n");
  }
  std::string path = lh_info->restart_image;
  path = path.substr(0, path.rfind('/') + 1) + DEVICE_FILE;
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    front_fatal("cannot open %s: %s\n", path.c_str(), strerror(errno));
  }
  uint64_t header[2];
  read_all(fd, header, sizeof header, path);
  if (header[0] != DEVICE_FILE_MAGIC || header[1] != count) {
    front_fatal("%s is not this checkpoint's device memory\n", path.c_str());
  }
  void *buf = pinned_buffer();
  for (uint64_t i = 0; i < count; i++) {
    FileRecord record;
    read_all(fd, &record, sizeof record, path);
    auto it = allocs.find(record.addr);
    if (it == allocs.end() || !it->second.in_file ||
        it->second.requested != record.size) {
      front_fatal("%s: no allocation of %llu bytes at %#llx\n", path.c_str(),
                  (unsigned long long)record.size,
                  (unsigned long long)record.addr);
    }
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaSetDevice)(it->second.device));
    if (rc == cudaSuccess) {
      ready_device(it->second.device);
    }
    for (size_t done = 0; rc == cudaSuccess && done < record.size;) {
      size_t n = record.size - done < DEVICE_FILE_BUFFER
                   ? record.size - done : DEVICE_FILE_BUFFER;
      read_all(fd, buf, n, path);
      FRONT_LH(rc = REAL(cudaMemcpy)((void *)(record.addr + done), buf, n,
                                     cudaMemcpyHostToDevice));
      done += n;
    }
    if (rc != cudaSuccess) {
      front_fatal("restoring device memory at %#llx: error %d\n",
                  (unsigned long long)record.addr, rc);
    }
  }
  free_pinned_buffer(buf);
  close(fd);
}

// Last at restart: the contents, once cudart's contexts are back.
void
front_vmm_fill()
{
  std::lock_guard<std::mutex> guard(vmm_lock);
  fill_from_file();
  for (auto &entry : allocs) {
    Alloc &a = entry.second;
    if (a.shadow == NULL) {
      continue;
    }
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaSetDevice)(a.device));
    if (rc == cudaSuccess) {
      ready_device(a.device);
      FRONT_LH(rc = REAL(cudaMemcpy)((void *)entry.first, a.shadow,
                                     a.requested, cudaMemcpyHostToDevice));
    }
    if (rc != cudaSuccess) {
      front_fatal("restoring device memory at %#llx: error %d\n",
                  (unsigned long long)entry.first, rc);
    }
  }
  front_log("restored %zu allocations\n", allocs.size());
}
