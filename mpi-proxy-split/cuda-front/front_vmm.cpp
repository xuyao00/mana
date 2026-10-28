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
// chunk.  Not yet: peer access.

#include <stdlib.h>
#include <string.h>
#include <map>
#include <mutex>
#include "front.h"

namespace {

struct Alloc {
  size_t size;                          // mapped size, granularity multiple
  size_t requested;
  int device;
  CUmemGenericAllocationHandle handle;  // of generation 'gen'
  void *shadow;                         // device contents during a checkpoint
  bool live;                            // false: freed, kept mapped for reuse
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

std::mutex vmm_lock;
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
  CUmemAccessDesc access;
  memset(&access, 0, sizeof access);
  access.location = prop.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  DRV_CHECK(DRV(cuMemSetAccess)(va, size, &access, 1));
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
  }
}

}  // namespace

static cudaError_t big_alloc_locked(void **ptr, size_t size, int device);
static cudaError_t small_alloc_locked(void **ptr, size_t size, int device);

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
  if (size < SMALL_MAX) {
    return small_alloc_locked(ptr, size, device);
  }
  return big_alloc_locked(ptr, size, device);
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
  Alloc a = { rounded, size, device, 0, NULL, true };
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
    Alloc a = { chunk_size, chunk_size, device, 0, NULL, true };
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

bool
front_vmm_free(void *ptr, cudaError_t *result)
{
  std::unique_lock<std::mutex> guard(vmm_lock);
  CUdeviceptr p = (CUdeviceptr)ptr;
  auto it = allocs.upper_bound(p);
  if (it == allocs.begin()) {
    return false;
  }
  --it;
  auto chunk = chunks.find(it->first);
  uint32_t index = 0;
  if (chunk != chunks.end()) {
    Chunk &c = chunk->second;
    if (p >= it->first + it->second.size || (p - it->first) % c.block != 0) {
      return false;
    }
    index = (p - it->first) / c.block;
    if (!(c.used[index / 64] & (1ULL << (index % 64)))) {
      return false;  // freed already: the driver's error, as for cudaMalloc
    }
  } else if (it->first != p || !it->second.live) {
    return false;
  }
  guard.unlock();
  // cudaFree() waits for the device's work.
  FRONT_LH(*result = REAL(cudaDeviceSynchronize)());
  guard.lock();
  if (chunk != chunks.end()) {
    Chunk &c = chunk->second;
    c.used[index / 64] &= ~(1ULL << (index % 64));
    c.requested[index] = 0;
    c.n_free++;
  } else {
    it->second.live = false;
  }
  return true;
}

void
front_vmm_stash()
{
  std::lock_guard<std::mutex> guard(vmm_lock);
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
  front_log("saved %zu bytes of device memory\n", bytes);
}

void
front_vmm_drop_shadows()
{
  std::lock_guard<std::mutex> guard(vmm_lock);
  for (auto &entry : allocs) {
    free(entry.second.shadow);
    entry.second.shadow = NULL;
  }
}

// At restart, in the new lower half: the same range, then new physical
// memory at each live allocation, filled from its shadow.  Freed chunks are
// not kept.
void
front_vmm_restore()
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
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaSetDevice)(a.device));
    if (rc != cudaSuccess) {
      front_fatal("restoring device memory on device %d: error %d\n",
                  a.device, rc);
    }
    ready_device(a.device);
    if (map_physical(it->first, a.size, a.device, &a.handle) !=
        CUDA_SUCCESS) {
      front_fatal("no device memory for %zu bytes at %#llx on device %d\n",
                  a.size, (unsigned long long)it->first, a.device);
    }
    if (a.shadow != NULL) {
      FRONT_LH(rc = REAL(cudaMemcpy)((void *)it->first, a.shadow,
                                     a.requested, cudaMemcpyHostToDevice));
    }
    if (rc != cudaSuccess) {
      front_fatal("restoring device memory at %#llx: error %d\n",
                  (unsigned long long)it->first, rc);
    }
    ++it;
  }
  front_log("restored %zu allocations\n", allocs.size());
}
