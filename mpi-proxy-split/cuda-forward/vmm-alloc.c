/*
 * MANA's device-memory allocator for CUDA-aware MPI (libmana_cuda_vmm.so,
 * preloaded in the UPPER half by mana_launch --cuda).
 *
 * The lower half's MPI (UCX's cuda_ipc, Cray's GTL) shares device buffers
 * between processes with CUDA IPC.  A legacy IPC export (cuIpcGetMemHandle of
 * cudaMalloc memory) makes the process impossible to checkpoint, but VMM
 * memory exported as a POSIX fd can be checkpointed, and the forwarding
 * shim's virtual IPC handles (cuda-special.c) share it.  So this library
 * makes the application's (and the MPI's) device allocations VMM memory:
 * cuMemAlloc_v2, cuMemAllocPitch_v2 and cuMemFree_v2 of the real libcuda
 * become cuMemCreate (POSIX-fd shareable) + cuMemAddressReserve + cuMemMap.
 * cudaMalloc (shared or static cudart) and pointers from cuGetProcAddress
 * come here too: the driver's exported functions are stubs that call the
 * implementation through a pointer slot, and we redirect the slot.
 *
 * Small allocations are carved out of VMM chunks (the allocation granularity
 * is 2 MB), first fit, 512-byte aligned.  An IPC export then shares the whole
 * chunk; cuMemGetAddressRange reports the chunk, as UCX expects.
 *
 * Two orderings of cudaMalloc memory that VMM memory lacks are kept: cudaFree
 * waits for the device's work (wrap_free()), and the synchronous copies and
 * sets complete before the MPI may access their memory (order_memop()).
 */
#define _GNU_SOURCE
#include <cuda.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define ALIGN 512
#define SMALL_LIMIT (1 << 20)     // smaller ones are carved out of chunks
#define MAX_ALLOCS 65536
#define MAX_CHUNKS 4096
#define MAX_BLOCKS 64             // per chunk

typedef CUresult (*alloc_fn)(CUdeviceptr *, size_t);
typedef CUresult (*pitch_fn)(CUdeviceptr *, size_t *, size_t, size_t, unsigned);
typedef CUresult (*free_fn)(CUdeviceptr);
typedef CUresult (*setattr_fn)(const void *, CUpointer_attribute, CUdeviceptr);
static alloc_fn real_alloc;
static pitch_fn real_pitch;
static free_fn real_free;
static setattr_fn real_setattr;
static CUresult (*ctx_sync)(void);
static CUresult (*stream_sync)(CUstream);

typedef struct {                  // A VMM allocation of its own.
  CUdeviceptr base;
  size_t size;
  CUmemGenericAllocationHandle h;
  int on;
} Alloc;

typedef struct { size_t off, size; int used; } Block;
typedef struct {                  // A VMM chunk for small allocations.
  CUdeviceptr base;
  size_t size;
  CUmemGenericAllocationHandle h;
  CUdevice dev;
  Block blocks[MAX_BLOCKS];
  int n_blocks;
  int on;
} Chunk;

static Alloc allocs[MAX_ALLOCS];
static int n_allocs;
static Chunk chunks[MAX_CHUNKS];
static int n_chunks;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static int verbose;
static struct {
  unsigned long vmm, small, fallback, frees, deferred, immediate, drained;
} stats;

// Whether VMM memory of 'dev' can be made GPUDirect RDMA capable.  A NIC
// (e.g. Slingshot's, through libfabric's CXI provider) can register VMM
// memory only if it was created so; cudaMalloc memory always can be.
static int
rdma_capable(CUdevice dev)
{
  static int known[64];             // 0: not asked yet, 1: no, 2: yes
  if (dev < 0 || dev >= 64) {
    return 0;
  }
  if (known[dev] == 0) {
    int v = 0;
    CUresult rc = cuDeviceGetAttribute(
      &v, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, dev);
    known[dev] = rc == CUDA_SUCCESS && v ? 2 : 1;
  }
  return known[dev] == 2;
}

// A VMM allocation of 'size' (a multiple of the granularity) on the current
// context's device, shareable as a POSIX fd.
static CUresult
vmm_create(size_t size, CUdeviceptr *base, CUmemGenericAllocationHandle *h,
           CUdevice *devp)
{
  CUdevice dev;
  CUresult rc = cuCtxGetDevice(&dev);
  if (rc != CUDA_SUCCESS) {
    return rc;
  }
  CUmemAllocationProp prop;
  memset(&prop, 0, sizeof prop);
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = dev;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  prop.allocFlags.gpuDirectRDMACapable = rdma_capable(dev);
  if ((rc = cuMemCreate(h, size, &prop, 0)) != CUDA_SUCCESS) {
    return rc;
  }
  if ((rc = cuMemAddressReserve(base, size, 0, 0, 0)) != CUDA_SUCCESS) {
    cuMemRelease(*h);
    return rc;
  }
  CUmemAccessDesc acc;
  memset(&acc, 0, sizeof acc);
  acc.location = prop.location;
  acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if ((rc = cuMemMap(*base, size, 0, *h, 0)) != CUDA_SUCCESS ||
      (rc = cuMemSetAccess(*base, size, &acc, 1)) != CUDA_SUCCESS) {
    cuMemUnmap(*base, size);
    cuMemAddressFree(*base, size);
    cuMemRelease(*h);
    return rc;
  }
  if (devp != NULL) {
    *devp = dev;
  }
  return CUDA_SUCCESS;
}

static void
vmm_destroy(CUdeviceptr base, size_t size, CUmemGenericAllocationHandle h)
{
  cuMemUnmap(base, size);
  cuMemAddressFree(base, size);
  cuMemRelease(h);
}

static size_t
granularity(void)
{
  static size_t g = 0;
  if (g == 0) {
    CUdevice dev = 0;
    cuCtxGetDevice(&dev);
    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof prop);
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    if (cuMemGetAllocationGranularity(&g, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)
        != CUDA_SUCCESS || g == 0) {
      g = 2 << 20;
    }
  }
  return g;
}

// First fit in the chunks of the current device.
static CUresult
small_alloc(CUdeviceptr *dptr, size_t size)
{
  size = (size + ALIGN - 1) / ALIGN * ALIGN;
  CUdevice dev;
  CUresult rc = cuCtxGetDevice(&dev);
  if (rc != CUDA_SUCCESS) {
    return rc;
  }
  for (int c = 0; c < n_chunks; c++) {
    Chunk *ch = &chunks[c];
    if (!ch->on || ch->dev != dev) {
      continue;
    }
    for (int b = 0; b < ch->n_blocks; b++) {
      Block *bl = &ch->blocks[b];
      if (bl->used || bl->size < size) {
        continue;
      }
      if (bl->size > size && ch->n_blocks < MAX_BLOCKS) {   // split
        memmove(&ch->blocks[b + 2], &ch->blocks[b + 1],
                (ch->n_blocks - b - 1) * sizeof(Block));
        ch->blocks[b + 1] = (Block){ bl->off + size, bl->size - size, 0 };
        ch->n_blocks++;
        bl->size = size;
      }
      bl->used = 1;
      *dptr = ch->base + bl->off;
      return CUDA_SUCCESS;
    }
  }
  // A new chunk.
  int c;
  for (c = 0; c < n_chunks && chunks[c].on; c++) {
  }
  if (c == MAX_CHUNKS) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  Chunk *ch = &chunks[c];
  memset(ch, 0, sizeof *ch);
  ch->size = granularity();
  if ((rc = vmm_create(ch->size, &ch->base, &ch->h, &ch->dev)) != CUDA_SUCCESS) {
    return rc;
  }
  ch->on = 1;
  ch->blocks[0] = (Block){ 0, size, 1 };
  ch->blocks[1] = (Block){ size, ch->size - size, 0 };
  ch->n_blocks = 2;
  if (c == n_chunks) {
    n_chunks++;
  }
  *dptr = ch->base;
  return CUDA_SUCCESS;
}

// 1: freed a block of a chunk; 0: not in a chunk.
static int
small_free(CUdeviceptr p)
{
  for (int c = 0; c < n_chunks; c++) {
    Chunk *ch = &chunks[c];
    if (!ch->on || p < ch->base || p >= ch->base + ch->size) {
      continue;
    }
    for (int b = 0; b < ch->n_blocks; b++) {
      Block *bl = &ch->blocks[b];
      if (ch->base + bl->off != p || !bl->used) {
        continue;
      }
      bl->used = 0;
      // Merge with free neighbors.
      if (b + 1 < ch->n_blocks && !ch->blocks[b + 1].used) {
        bl->size += ch->blocks[b + 1].size;
        memmove(&ch->blocks[b + 1], &ch->blocks[b + 2],
                (ch->n_blocks - b - 2) * sizeof(Block));
        ch->n_blocks--;
      }
      if (b > 0 && !ch->blocks[b - 1].used) {
        ch->blocks[b - 1].size += bl->size;
        memmove(&ch->blocks[b], &ch->blocks[b + 1],
                (ch->n_blocks - b - 1) * sizeof(Block));
        ch->n_blocks--;
      }
      if (ch->n_blocks == 1 && !ch->blocks[0].used) {   // empty: give back
        vmm_destroy(ch->base, ch->size, ch->h);
        ch->on = 0;
      }
      return 1;
    }
    return 1;  // Inside a chunk but not a block start: ignore, like CUDA.
  }
  return 0;
}

static CUresult
vmm_alloc(CUdeviceptr *dptr, size_t bytesize)
{
  if (bytesize < SMALL_LIMIT) {
    CUresult rc = small_alloc(dptr, bytesize);
    if (rc == CUDA_SUCCESS) {
      stats.small++;
    }
    return rc;
  }
  int i;
  for (i = 0; i < n_allocs && allocs[i].on; i++) {
  }
  if (i == MAX_ALLOCS) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  size_t g = granularity();
  size_t size = (bytesize + g - 1) / g * g;
  Alloc *a = &allocs[i];
  CUresult rc = vmm_create(size, &a->base, &a->h, NULL);
  if (rc != CUDA_SUCCESS) {
    return rc;
  }
  a->size = size;
  a->on = 1;
  if (i == n_allocs) {
    n_allocs++;
  }
  stats.vmm++;
  *dptr = a->base;
  return CUDA_SUCCESS;
}

static CUresult
wrap_alloc(CUdeviceptr *dptr, size_t bytesize)
{
  if (bytesize == 0 || dptr == NULL) {
    return real_alloc(dptr, bytesize);
  }
  pthread_mutex_lock(&lock);
  CUresult rc = vmm_alloc(dptr, bytesize);
  pthread_mutex_unlock(&lock);
  if (rc != CUDA_SUCCESS) {
    stats.fallback++;
    rc = real_alloc(dptr, bytesize);   // e.g. no VMM support: as before
  }
  return rc;
}

static CUresult
wrap_pitch(CUdeviceptr *dptr, size_t *pitch, size_t width, size_t height,
           unsigned elem)
{
  if (dptr == NULL || pitch == NULL || width == 0 || height == 0) {
    return real_pitch(dptr, pitch, width, height, elem);
  }
  size_t p = (width + ALIGN - 1) / ALIGN * ALIGN;
  CUresult rc = wrap_alloc(dptr, p * height);
  if (rc == CUDA_SUCCESS) {
    *pitch = p;
  }
  return rc;
}

// The allocation of ours that contains 'p': 1 and its range, or 0.  With the
// lock.
static int
find_alloc(CUdeviceptr p, CUdeviceptr *base, size_t *size)
{
  for (int i = 0; i < n_allocs; i++) {
    if (allocs[i].on && p - allocs[i].base < allocs[i].size) {
      *base = allocs[i].base;
      *size = allocs[i].size;
      return 1;
    }
  }
  for (int c = 0; c < n_chunks; c++) {
    Chunk *ch = &chunks[c];
    if (!ch->on || p - ch->base >= ch->size) {
      continue;
    }
    for (int b = 0; b < ch->n_blocks; b++) {
      Block *bl = &ch->blocks[b];
      if (bl->used && p - (ch->base + bl->off) < bl->size) {
        *base = ch->base + bl->off;
        *size = bl->size;
        return 1;
      }
    }
    return 0;
  }
  return 0;
}

static CUresult
wrap_free(CUdeviceptr dptr)
{
  if (dptr == 0) {
    return real_free(dptr);
  }
  // cudaFree waits for the device's work (the driver's cuMemFree_v2 does),
  // so that no kernel or copy still uses memory that a new allocation may
  // get.  Wait as well.
  ctx_sync();
  pthread_mutex_lock(&lock);
  stats.frees++;
  for (int i = 0; i < n_allocs; i++) {
    if (allocs[i].on && allocs[i].base == dptr) {
      vmm_destroy(allocs[i].base, allocs[i].size, allocs[i].h);
      allocs[i].on = 0;
      pthread_mutex_unlock(&lock);
      return CUDA_SUCCESS;
    }
  }
  int done = small_free(dptr);
  pthread_mutex_unlock(&lock);
  return done ? CUDA_SUCCESS : real_free(dptr);
}

/*
 * The synchronous copies and sets (cudaMemcpy between device buffers,
 * cudaMemset, ...) may return before they are done: the driver refuses
 * CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, which makes them wait, for VMM memory.
 * In a CUDA-aware MPI program, only the MPI accesses device memory outside
 * CUDA's stream order (the NIC, or a peer GPU through CUDA IPC), and only
 * after a call has entered MANA's lower half.  So (order_memop()):
 *   - one that the application issues records an event on its default
 *     stream, and MANA waits for the pending events before an MPI call
 *     enters the lower half (mana_vmm_drain(), from lower_half_disable_ckpt());
 *   - one that the MPI library issues inside an MPI call (through the lower
 *     half's forwarding shims) waits at once, as with SYNC_MEMOPS.
 * Until MANA connects (mana_vmm_use_drain()), e.g. without MANA, each waits
 * at once.
 */
typedef struct {
  CUcontext ctx;                // the context of ev[]
  CUevent ev[2];                // after the last copy: legacy, per-thread stream
  int pending[2];
} Pending;
#define MAX_PENDING 256         // threads
static Pending *pendings[MAX_PENDING];
static int n_pendings;
static __thread Pending *my_pending;
static pthread_mutex_t pending_lock = PTHREAD_MUTEX_INITIALIZER;
static int (*forwarding)(void); // NULL until MANA connects
int mana_vmm_pending;           // events pending (for MANA)

void
mana_vmm_use_drain(int (*forwarding_fn)(void))
{
  __atomic_store_n(&forwarding, forwarding_fn, __ATOMIC_RELEASE);
}

void
mana_vmm_drain(void)
{
  CUevent evs[2 * MAX_PENDING];
  int n = 0;
  pthread_mutex_lock(&pending_lock);
  for (int i = 0; i < n_pendings; i++) {
    for (int k = 0; k < 2; k++) {
      if (pendings[i]->pending[k]) {
        pendings[i]->pending[k] = 0;
        evs[n++] = pendings[i]->ev[k];
      }
    }
  }
  __atomic_store_n(&mana_vmm_pending, 0, __ATOMIC_RELEASE);
  stats.drained += n;
  pthread_mutex_unlock(&pending_lock);
  for (int i = 0; i < n; i++) {
    cuEventSynchronize(evs[i]);  // An error shows at the next CUDA call.
  }
}

// Records an event after a copy on default stream k (0: legacy, 1:
// per-thread) of this thread.  0 if it cannot (then wait at once).
static int
defer_memop(int k)
{
  CUcontext ctx = NULL;
  if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == NULL) {
    return 0;
  }
  Pending *p = my_pending;
  pthread_mutex_lock(&pending_lock);
  if (p == NULL) {
    if (n_pendings == MAX_PENDING || (p = calloc(1, sizeof *p)) == NULL) {
      pthread_mutex_unlock(&pending_lock);
      return 0;
    }
    pendings[n_pendings++] = p;
    my_pending = p;
  }
  if (p->ctx != ctx) {
    // The first copy, or one on another device: events of this context.
    // The old ones are not destroyed: a drain may be waiting on them.
    CUevent ev[2];
    if (cuEventCreate(&ev[0], CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS ||
        cuEventCreate(&ev[1], CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) {
      pthread_mutex_unlock(&pending_lock);
      return 0;
    }
    for (int j = 0; j < 2; j++) {
      if (p->pending[j]) {
        cuEventSynchronize(p->ev[j]);
        p->pending[j] = 0;
      }
      p->ev[j] = ev[j];
    }
    p->ctx = ctx;
  }
  int ok = cuEventRecord(p->ev[k], k ? CU_STREAM_PER_THREAD : CU_STREAM_LEGACY)
           == CUDA_SUCCESS;
  if (ok) {
    p->pending[k] = 1;
    __atomic_store_n(&mana_vmm_pending, 1, __ATOMIC_RELEASE);
    stats.deferred++;
  }
  pthread_mutex_unlock(&pending_lock);
  return ok;
}

// After a synchronous copy or set on default stream k (0: legacy, 1:
// per-thread).
static void
order_memop(int k)
{
  int (*mpi)(void) = __atomic_load_n(&forwarding, __ATOMIC_ACQUIRE);
  if (mpi != NULL && !mpi() && defer_memop(k)) {
    return;                     // the application's: until the next MPI call
  }
  __atomic_fetch_add(&stats.immediate, 1, __ATOMIC_RELAXED);
  stream_sync(k ? CU_STREAM_PER_THREAD : CU_STREAM_LEGACY);
}

// The synchronous copies and sets, on the legacy (k = 0) and the per-thread
// (k = 1) default stream.
#define SYNC_MEMOP(name, k, params, args)                                     \
  static CUresult (*real_##name) params;                                      \
  static CUresult wrap_##name params                                          \
  {                                                                           \
    CUresult rc = real_##name args;                                           \
    if (rc == CUDA_SUCCESS) {                                                 \
      order_memop(k);                                                         \
    }                                                                         \
    return rc;                                                                \
  }
#define SYNC_MEMOPS(name, params, args)                                       \
  SYNC_MEMOP(name, 0, params, args)                                           \
  SYNC_MEMOP(name##_ptds, 1, params, args)

SYNC_MEMOPS(cuMemcpy, (CUdeviceptr d, CUdeviceptr s, size_t n), (d, s, n))
SYNC_MEMOPS(cuMemcpyDtoD_v2, (CUdeviceptr d, CUdeviceptr s, size_t n),
            (d, s, n))
SYNC_MEMOPS(cuMemcpyHtoD_v2, (CUdeviceptr d, const void *s, size_t n),
            (d, s, n))
SYNC_MEMOPS(cuMemcpyPeer, (CUdeviceptr d, CUcontext dc, CUdeviceptr s,
                           CUcontext sc, size_t n),
            (d, dc, s, sc, n))
SYNC_MEMOPS(cuMemcpy2D_v2, (const CUDA_MEMCPY2D *p), (p))
SYNC_MEMOPS(cuMemcpy2DUnaligned_v2, (const CUDA_MEMCPY2D *p), (p))
SYNC_MEMOPS(cuMemcpy3D_v2, (const CUDA_MEMCPY3D *p), (p))
SYNC_MEMOPS(cuMemsetD8_v2, (CUdeviceptr d, unsigned char v, size_t n),
            (d, v, n))
SYNC_MEMOPS(cuMemsetD16_v2, (CUdeviceptr d, unsigned short v, size_t n),
            (d, v, n))
SYNC_MEMOPS(cuMemsetD32_v2, (CUdeviceptr d, unsigned int v, size_t n),
            (d, v, n))
SYNC_MEMOPS(cuMemsetD2D8_v2, (CUdeviceptr d, size_t pitch, unsigned char v,
                              size_t w, size_t h),
            (d, pitch, v, w, h))
SYNC_MEMOPS(cuMemsetD2D16_v2, (CUdeviceptr d, size_t pitch, unsigned short v,
                               size_t w, size_t h),
            (d, pitch, v, w, h))
SYNC_MEMOPS(cuMemsetD2D32_v2, (CUdeviceptr d, size_t pitch, unsigned int v,
                               size_t w, size_t h),
            (d, pitch, v, w, h))

// CU_POINTER_ATTRIBUTE_SYNC_MEMOPS on our memory (libfabric sets it on the
// buffers it registers): accepted, as for cudaMalloc memory, and kept by
// order_memop() instead.
static CUresult
wrap_setattr(const void *value, CUpointer_attribute attribute, CUdeviceptr ptr)
{
  if (attribute == CU_POINTER_ATTRIBUTE_SYNC_MEMOPS) {
    CUdeviceptr base;
    size_t size;
    pthread_mutex_lock(&lock);
    int ours = find_alloc(ptr, &base, &size);
    pthread_mutex_unlock(&lock);
    if (ours) {
      return CUDA_SUCCESS;
    }
  }
  return real_setattr(value, attribute, ptr);
}

#define MEMOP(name) { #name, (void *)wrap_##name, (void **)&real_##name }
#define MEMOPS(name) MEMOP(name), MEMOP(name##_ptds)
static const struct { const char *name; void *wrapper; void **real; } memops[] = {
  MEMOPS(cuMemcpy), MEMOPS(cuMemcpyDtoD_v2), MEMOPS(cuMemcpyHtoD_v2),
  MEMOPS(cuMemcpyPeer), MEMOPS(cuMemcpy2D_v2), MEMOPS(cuMemcpy2DUnaligned_v2),
  MEMOPS(cuMemcpy3D_v2), MEMOPS(cuMemsetD8_v2), MEMOPS(cuMemsetD16_v2),
  MEMOPS(cuMemsetD32_v2), MEMOPS(cuMemsetD2D8_v2), MEMOPS(cuMemsetD2D16_v2),
  MEMOPS(cuMemsetD2D32_v2),
};

/*
 * Finds the call through the implementation slot in the entry code of an
 * exported driver function:
 *          [frame setup]               (driver 580: before the cmpl)
 *          cmpl $imm32, disp32(%rip)   81 3d d d d d i i i i   (10 bytes)
 *          je   ...                    74 xx  or  0f 84 xx xx xx xx
 *          [frame setup]               (driver 595: after the je)
 *   call:  call *disp32(%rip)          ff 15 d d d d -> the slot
 * Returns the address of the call, or NULL if the code is not of that form.
 */
static unsigned char *
slot_call(unsigned char *f)
{
  for (int i = 0; i < 0x20; i++) {
    if (f[i] == 0x81 && f[i + 1] == 0x3d) {
      unsigned char *p = f + i + 10;
      if (p[0] == 0x74) {
        p += 2;
      } else if (p[0] == 0x0f && p[1] == 0x84) {
        p += 6;
      } else {
        return NULL;
      }
      for (int j = 0; j < 0x10; j++) {
        if (p[j] == 0xff && p[j + 1] == 0x15) {
          return p + j;
        }
      }
      return NULL;
    }
  }
  return NULL;
}

/*
 * Redirects the implementation slot of exported driver function 'name'
 * (see slot_call()) to 'wrapper', and stores the old implementation in *real
 * before: the wrapper calls it.  Returns 0 if the entry code is not of the
 * expected form (*real is unchanged), else 1.
 */
static int
redirect(void *lib, const char *name, void *wrapper, void **real)
{
  unsigned char *f = (unsigned char *)dlsym(lib, name);
  unsigned char *call = f != NULL ? slot_call(f) : NULL;
  if (call == NULL) {
    fprintf(stderr, "[mana-vmm] %s: unexpected entry code; not redirected\n",
            name);
    return 0;
  }
  int32_t disp;
  memcpy(&disp, call + 2, sizeof disp);
  void **slot = (void **)(call + 6 + disp);
  void *old = *slot;
  long page = sysconf(_SC_PAGESIZE);
  void *pg = (void *)((uintptr_t)slot & ~(uintptr_t)(page - 1));
  if (mprotect(pg, page, PROT_READ | PROT_WRITE) != 0) {
    fprintf(stderr, "[mana-vmm] %s: cannot write its slot\n", name);
    return 0;
  }
  __atomic_store_n(real, old, __ATOMIC_SEQ_CST);
  __atomic_store_n(slot, wrapper, __ATOMIC_SEQ_CST);
  if (verbose) {
    fprintf(stderr, "[mana-vmm] %s: slot %p, %p -> %p\n", name, (void *)slot,
            old, wrapper);
  }
  return 1;
}

static void
report(void)
{
  if (verbose) {
    fprintf(stderr, "[mana-vmm %d] VMM allocations: %lu large, %lu small, "
            "%lu fallbacks, %lu frees; synchronous copies: %lu deferred to "
            "an MPI call (%lu waits), %lu waited at once\n", getpid(),
            stats.vmm, stats.small, stats.fallback, stats.frees,
            stats.deferred, stats.drained, stats.immediate);
  }
}

__attribute__((constructor)) static void
vmm_alloc_init(void)
{
  if (getenv("MANA_CUDA_VMM_ALLOC") == NULL) {
    return;
  }
  verbose = getenv("MANA_FWD_VERBOSE") != NULL;
  void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
  if (lib == NULL) {
    return;                      // No CUDA driver: nothing to do.
  }
  ctx_sync = (CUresult (*)(void))dlsym(lib, "cuCtxSynchronize");
  stream_sync = (CUresult (*)(CUstream))dlsym(lib, "cuStreamSynchronize");
  // The real functions are the exported ones until their slots are
  // redirected.  cuMemFree first: VMM allocations must never reach the real
  // cuMemFree.
  real_free = (free_fn)dlsym(lib, "cuMemFree_v2");
  real_alloc = (alloc_fn)dlsym(lib, "cuMemAlloc_v2");
  real_pitch = (pitch_fn)dlsym(lib, "cuMemAllocPitch_v2");
  real_setattr = (setattr_fn)dlsym(lib, "cuPointerSetAttribute");
  int on = ctx_sync != NULL && stream_sync != NULL &&
           redirect(lib, "cuMemFree_v2", (void *)wrap_free,
                    (void **)&real_free) &&
           redirect(lib, "cuMemAlloc_v2", (void *)wrap_alloc,
                    (void **)&real_alloc);
  if (!on) {
    fprintf(stderr, "[mana-vmm] device allocations stay cudaMalloc memory: "
            "no checkpointable CUDA IPC\n");
    return;
  }
  redirect(lib, "cuMemAllocPitch_v2", (void *)wrap_pitch, (void **)&real_pitch);
  // SYNC_MEMOPS is emulated only if every synchronous copy and set waits.
  int memops_on = 1;
  for (size_t i = 0; i < sizeof memops / sizeof memops[0]; i++) {
    *memops[i].real = dlsym(lib, memops[i].name);
    memops_on = memops_on && *memops[i].real != NULL &&
                redirect(lib, memops[i].name, memops[i].wrapper, memops[i].real);
  }
  if (!memops_on || !redirect(lib, "cuPointerSetAttribute",
                              (void *)wrap_setattr, (void **)&real_setattr)) {
    fprintf(stderr, "[mana-vmm] synchronous copies to device memory may "
            "return before they complete: not ordered with MPI\n");
  }
  atexit(report);
}
