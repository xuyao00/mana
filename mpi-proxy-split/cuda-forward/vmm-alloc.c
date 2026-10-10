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
static alloc_fn real_alloc;
static pitch_fn real_pitch;
static free_fn real_free;

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
static struct { unsigned long vmm, small, fallback, frees; } stats;

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

static CUresult
wrap_free(CUdeviceptr dptr)
{
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
 * (see slot_call()).  Returns the old implementation, or NULL if the entry
 * code is not of the expected form.
 */
static void *
redirect(void *lib, const char *name, void *wrapper)
{
  unsigned char *f = (unsigned char *)dlsym(lib, name);
  unsigned char *call = f != NULL ? slot_call(f) : NULL;
  if (call == NULL) {
    fprintf(stderr, "[mana-vmm] %s: unexpected entry code; not redirected\n",
            name);
    return NULL;
  }
  int32_t disp;
  memcpy(&disp, call + 2, sizeof disp);
  void **slot = (void **)(call + 6 + disp);
  void *old = *slot;
  long page = sysconf(_SC_PAGESIZE);
  void *pg = (void *)((uintptr_t)slot & ~(uintptr_t)(page - 1));
  if (mprotect(pg, page, PROT_READ | PROT_WRITE) != 0) {
    fprintf(stderr, "[mana-vmm] %s: cannot write its slot\n", name);
    return NULL;
  }
  __atomic_store_n(slot, wrapper, __ATOMIC_SEQ_CST);
  if (verbose) {
    fprintf(stderr, "[mana-vmm] %s: slot %p, %p -> %p\n", name, (void *)slot,
            old, wrapper);
  }
  return old;
}

static void
report(void)
{
  if (verbose) {
    fprintf(stderr, "[mana-vmm %d] VMM allocations: %lu large, %lu small, "
            "%lu fallbacks, %lu frees\n", getpid(), stats.vmm, stats.small,
            stats.fallback, stats.frees);
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
  // cuMemFree first: VMM allocations must never reach the real cuMemFree.
  // The real functions are set before their slots point to the wrappers.
  real_free = (free_fn)dlsym(lib, "cuMemFree_v2");      // until redirected
  void *f = redirect(lib, "cuMemFree_v2", (void *)wrap_free);
  void *a = NULL;
  if (f != NULL) {
    real_free = (free_fn)f;
    real_alloc = (alloc_fn)dlsym(lib, "cuMemAlloc_v2");
    real_pitch = (pitch_fn)dlsym(lib, "cuMemAllocPitch_v2");
    a = redirect(lib, "cuMemAlloc_v2", (void *)wrap_alloc);
    if (a != NULL) {
      real_alloc = (alloc_fn)a;
      void *p = redirect(lib, "cuMemAllocPitch_v2", (void *)wrap_pitch);
      if (p != NULL) {
        real_pitch = (pitch_fn)p;
      }
    }
  }
  if (a == NULL) {
    fprintf(stderr, "[mana-vmm] device allocations stay cudaMalloc memory: "
            "no checkpointable CUDA IPC\n");
  }
  atexit(report);
}
