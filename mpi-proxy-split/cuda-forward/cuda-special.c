/*
 * Hand-written functions of the libcuda.so.1 shim (the others are generated
 * trampolines): those that hand out upper-half function pointers, which the
 * lower half must call through trampolines.
 */
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <cuda.h>

#include "fwd-runtime.h"

static void *
upper_sym(const char *symbol)
{
  void *p = NULL;
  FWD_CALL_UPPER(p, NULL, mana_fwd_ctl.uh_dlsym("libcuda.so.1", symbol));
  return p;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                             cuuint64_t flags,
                             CUdriverProcAddressQueryResult *symbolStatus);

// The hand-written functions of this shim, which cuGetProcAddress must hand
// out too (the others are reached through trampolines).  The API version
// each one implements is the current one (e.g. cuIpcOpenMemHandle_v2).
static void *own_function(const char *symbol);

static void *
wrap(const char *symbol, void *fn)
{
  void *own = own_function(symbol);
  return own != NULL ? own : mana_fwd_wrap_ptr(fn, symbol);
}

CUresult
cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                    cuuint64_t flags,
                    CUdriverProcAddressQueryResult *symbolStatus)
{
  static CUresult (*real)(const char *, void **, int, cuuint64_t,
                          CUdriverProcAddressQueryResult *) = NULL;
  if (real == NULL) {
    real = upper_sym("cuGetProcAddress_v2");
    if (real == NULL) {
      return CUDA_ERROR_NO_DEVICE;
    }
  }
  CUresult rc;
  FWD_CALL_UPPER(rc, CUDA_ERROR_NO_DEVICE, real(symbol, pfn, cudaVersion, flags, symbolStatus));
  if (rc == CUDA_SUCCESS && pfn != NULL && *pfn != NULL) {
    *pfn = wrap(symbol, *pfn);
  }
  return rc;
}

#undef cuGetProcAddress
CUresult
cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
                 cuuint64_t flags)
{
  static CUresult (*real)(const char *, void **, int, cuuint64_t) = NULL;
  if (real == NULL) {
    real = upper_sym("cuGetProcAddress");
    if (real == NULL) {
      return CUDA_ERROR_NO_DEVICE;
    }
  }
  CUresult rc;
  FWD_CALL_UPPER(rc, CUDA_ERROR_NO_DEVICE, real(symbol, pfn, cudaVersion, flags));
  if (rc == CUDA_SUCCESS && pfn != NULL && *pfn != NULL) {
    *pfn = wrap(symbol, *pfn);
  }
  return rc;
}

/*
 * cuGetExportTable: the driver's private function tables, used by cudart
 * (e.g. the static cudart in libnccl) instead of the public entry points.
 * Their functions must be called through trampolines, too
 * (mana_fwd_wrap_table()).
 */
CUresult
cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId)
{
  static CUresult (*real)(const void **, const CUuuid *) = NULL;
  if (real == NULL) {
    real = upper_sym("cuGetExportTable");
    if (real == NULL) {
      return CUDA_ERROR_NO_DEVICE;
    }
  }
  const void *t = NULL;
  CUresult rc;
  FWD_CALL_UPPER(rc, CUDA_ERROR_NO_DEVICE, real(&t, pExportTableId));
  if (rc == CUDA_SUCCESS && t != NULL) {
    t = mana_fwd_wrap_table(t, (void *)real, pExportTableId);
  }
  *ppExportTable = t;
  return rc;
}

/*
 * Host-memory registrations of the lower half (cuMemHostRegister and
 * cudaHostRegister, by UCX or the MPI).  The CUDA checkpoint saves them by
 * address, and restores them at restart in a new lower half whose memory is
 * not there: cuCheckpointProcessRestore fails.  So unregister them before a
 * checkpoint, and register them again on resume (the lower half then sees no
 * change); at restart, the new lower half has registered nothing yet.
 */
typedef struct { void *ptr; size_t size; unsigned flags; CUcontext ctx; int on; } HostReg;
static HostReg hostregs[4096];
static int n_hostregs;

void
mana_fwd_note_host_register(void *ptr, size_t size, unsigned flags, void *ctx)
{
  for (int i = 0; i < n_hostregs; i++) {
    if (!hostregs[i].on) {
      hostregs[i] = (HostReg){ ptr, size, flags, (CUcontext)ctx, 1 };
      return;
    }
  }
  if (n_hostregs < 4096) {
    hostregs[n_hostregs++] = (HostReg){ ptr, size, flags, (CUcontext)ctx, 1 };
  }
}

void
mana_fwd_note_host_unregister(void *ptr)
{
  for (int i = 0; i < n_hostregs; i++) {
    if (hostregs[i].on && hostregs[i].ptr == ptr) {
      hostregs[i].on = 0;
      return;
    }
  }
}

static CUcontext
current_ctx(void)
{
  static CUresult (*getCur)(CUcontext *) = NULL;
  if (getCur == NULL) {
    getCur = upper_sym("cuCtxGetCurrent");
  }
  CUcontext ctx = NULL;
  CUresult rc;
  FWD_CALL_UPPER(rc, CUDA_ERROR_NO_DEVICE, getCur(&ctx));
  return rc == CUDA_SUCCESS ? ctx : NULL;
}

// (Un)registers hostregs[i] in its context, on this (the checkpoint) thread.
static CUresult
host_reg_call(int i, int reg)
{
  static CUresult (*push)(CUcontext), (*pop)(CUcontext *);
  static CUresult (*hreg)(void *, size_t, unsigned), (*hunreg)(void *);
  if (push == NULL) {
    push = upper_sym("cuCtxPushCurrent_v2");
    pop = upper_sym("cuCtxPopCurrent_v2");
    hreg = upper_sym("cuMemHostRegister_v2");
    hunreg = upper_sym("cuMemHostUnregister");
  }
  CUresult rc, ignored;
  CUcontext old;
  FWD_CALL_UPPER(ignored, CUDA_ERROR_NO_DEVICE, push(hostregs[i].ctx));
  if (reg) {
    FWD_CALL_UPPER(rc, CUDA_ERROR_NO_DEVICE,
                   hreg(hostregs[i].ptr, hostregs[i].size, hostregs[i].flags));
  } else {
    FWD_CALL_UPPER(rc, CUDA_ERROR_NO_DEVICE, hunreg(hostregs[i].ptr));
  }
  FWD_CALL_UPPER(ignored, CUDA_ERROR_NO_DEVICE, pop(&old));
  (void)ignored;
  return rc;
}

static void vipc_before_ckpt(void);
static void vipc_after_resume_exports(void);
static void vipc_after_resume_imports(void);

static void
cuda_before_ckpt(void)
{
  vipc_before_ckpt();
  int n = 0;
  for (int i = 0; i < n_hostregs; i++) {
    if (hostregs[i].on) {
      CUresult rc = host_reg_call(i, 0);
      if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[mana-fwd] cuMemHostUnregister(%p) -> %d\n",
                hostregs[i].ptr, rc);
      }
      n++;
    }
  }
  if (getenv("MANA_FWD_VERBOSE")) {
    fprintf(stderr, "[mana-fwd] before checkpoint: %d host registrations "
            "released\n", n);
  }
}

static unsigned long long ckpt_generation;

static void
cuda_after_resume(void)
{
  ckpt_generation++;              // See adjust_buffer_id().
  vipc_after_resume_exports();
  for (int i = 0; i < n_hostregs; i++) {
    if (hostregs[i].on) {
      CUresult rc = host_reg_call(i, 1);
      if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[mana-fwd] cuMemHostRegister(%p) -> %d\n",
                hostregs[i].ptr, rc);
      }
    }
  }
}

__attribute__((constructor)) static void
cuda_special_init(void)
{
  mana_fwd_cuda_before_ckpt = cuda_before_ckpt;
  mana_fwd_cuda_after_resume = cuda_after_resume;
  mana_fwd_cuda_after_resume_peers = vipc_after_resume_imports;
}

#define HOST_REGISTER_WRAPPER(name, ...)                                      \
CUresult                                                                      \
name(void *p, size_t bytesize, unsigned int Flags)                            \
{                                                                             \
  static CUresult (*real)(void *, size_t, unsigned) = NULL;                   \
  if (real == NULL) {                                                         \
    real = upper_sym(#name);                                                  \
    if (real == NULL) {                                                       \
      return CUDA_ERROR_NO_DEVICE;                                            \
    }                                                                         \
  }                                                                           \
  CUresult rc;                                                                \
  FWD_CALL_UPPER(rc, CUDA_ERROR_NO_DEVICE, real(p, bytesize, Flags));         \
  if (rc == CUDA_SUCCESS) {                                                   \
    mana_fwd_note_host_register(p, bytesize, Flags, current_ctx());           \
  }                                                                           \
  return rc;                                                                  \
}

#undef cuMemHostRegister
HOST_REGISTER_WRAPPER(cuMemHostRegister)
HOST_REGISTER_WRAPPER(cuMemHostRegister_v2)

CUresult
cuMemHostUnregister(void *p)
{
  static CUresult (*real)(void *) = NULL;
  if (real == NULL) {
    real = upper_sym("cuMemHostUnregister");
    if (real == NULL) {
      return CUDA_ERROR_NO_DEVICE;
    }
  }
  CUresult rc;
  FWD_CALL_UPPER(rc, CUDA_ERROR_NO_DEVICE, real(p));
  mana_fwd_note_host_unregister(p);
  return rc;
}

/*
 * Virtual CUDA IPC handles.
 *
 * A legacy IPC export (cuIpcGetMemHandle) makes a process un-checkpointable
 * until the buffer is freed, and no IPC handle or mapping survives a
 * checkpoint: the restored allocation is new memory, and a handle exported
 * before maps the old one.  VMM memory exported as a POSIX fd can be
 * checkpointed, if the importers hold no mapping.  So the lower half (UCX,
 * the MPI) gets virtual legacy IPC handles of VMM memory:
 *  - VMM memory that can be exported as a POSIX fd is reported as legacy-IPC
 *    capable (cuPointerGetAttribute(s)), so that the lower half (UCX's
 *    cuda_ipc) shares it with the legacy API;
 *  - cuIpcGetMemHandle exports the allocation as a POSIX fd (once) and returns
 *    a handle naming (exporter pid, fd number, size): the virtual id;
 *  - cuIpcOpenMemHandle gets the fd with pidfd_getfd, imports it, and maps it
 *    in a VA range reserved for it: the address the importer keeps (UCX's IPC
 *    cache stays valid);
 *  - before a checkpoint, importers unmap and release (the VA stays
 *    reserved) and exporters close their exports, keeping the fd numbers;
 *  - on resume, exporters export again on the same fd numbers; then (after a
 *    barrier) importers import again and map at the same VA;
 *  - at restart, the lower half is new: nothing to do.
 */
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>

#define VIPC_MAGIC 0x3143504941414e4dULL      // "MANAIPC1"

typedef struct {
  unsigned long long magic;
  int pid;                    // exporter (real pid)
  int fd;                     // its export fd, stable across checkpoints
  unsigned long long size;
  unsigned long long base;    // exporter's address (for diagnostics)
} VirtIpcHandle;              // fits in CUipcMemHandle (64 bytes)

typedef struct { CUdeviceptr base; size_t size; int fd; int on; } Export;
typedef struct {
  CUdeviceptr va; size_t size; int pid, fd; CUdevice dev;
  CUmemGenericAllocationHandle h; int mapped, on;
} Import;
static Export exports[1024];
static int n_exports;
static Import imports[1024];
static int n_imports;

#define UPPER(var, name) \
  static __typeof__(&name) var = NULL; \
  if (var == NULL) { var = (__typeof__(&name))upper_sym(#name); }
#define CALL(rc, call) FWD_CALL_UPPER(rc, CUDA_ERROR_NO_DEVICE, call)

static int
vipc_enabled(void)
{
  static int on = -1;
  if (on < 0) {
    const char *v = getenv("MANA_FWD_VIRTUAL_IPC");
    on = v == NULL || v[0] != '0';
  }
  return on;
}

static void
vlog(const char *fmt, long a, long b, long c)
{
  if (getenv("MANA_FWD_VERBOSE")) {
    fprintf(stderr, "[mana-fwd-ipc %d] ", getpid());
    fprintf(stderr, fmt, a, b, c);
  }
}

CUresult
cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle, void *addr)
{
  UPPER(real, cuMemRetainAllocationHandle);
  CUresult rc;
  CALL(rc, real(handle, addr));
  return rc;
}

// A fresh POSIX-fd export of the VMM allocation at 'base', or -1.
static int
export_fd(CUdeviceptr base)
{
  UPPER(retain, cuMemRetainAllocationHandle);
  UPPER(exportfd, cuMemExportToShareableHandle);
  UPPER(release, cuMemRelease);
  CUmemGenericAllocationHandle h;
  CUresult rc;
  CALL(rc, retain(&h, (void *)base));
  if (rc != CUDA_SUCCESS) {
    return -1;                // Not VMM memory.
  }
  int fd = -1;
  CALL(rc, exportfd(&fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
  CUresult ignored;
  CALL(ignored, release(h));
  (void)ignored;
  return rc == CUDA_SUCCESS ? fd : -1;
}

CUresult
cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr)
{
  UPPER(real, cuIpcGetMemHandle);
  UPPER(range, cuMemGetAddressRange_v2);
  if (!vipc_enabled()) {
    CUresult rc;
    CALL(rc, real(pHandle, dptr));
    return rc;
  }
  CUdeviceptr base;
  size_t size;
  CUresult rc;
  CALL(rc, range(&base, &size, dptr));
  if (rc != CUDA_SUCCESS) {
    return rc;
  }
  Export *e = NULL;
  for (int i = 0; i < n_exports; i++) {
    if (exports[i].on && exports[i].base == base) {
      e = &exports[i];
    }
  }
  if (e == NULL) {
    int fd = export_fd(base);
    if (fd < 0 || n_exports == 1024) {
      // cudaMalloc memory: a real legacy export would make the process
      // un-checkpointable.  The MPI falls back to a copy protocol.
      vlog("no virtual IPC for %#lx (not VMM POSIX-fd memory)\n", base, 0, 0);
      return CUDA_ERROR_NOT_SUPPORTED;
    }
    e = &exports[n_exports++];
    *e = (Export){ base, size, fd, 1 };
    vlog("export %#lx (%ld bytes) as fd %ld\n", base, size, fd);
  }
  VirtIpcHandle v = { VIPC_MAGIC, getpid(), e->fd, e->size, e->base };
  memset(pHandle, 0, sizeof *pHandle);
  memcpy(pHandle, &v, sizeof v);
  return CUDA_SUCCESS;
}

// Imports and maps imports[i] at its VA.
static CUresult
map_import(Import *im)
{
  UPPER(importfd, cuMemImportFromShareableHandle);
  UPPER(map, cuMemMap);
  UPPER(access, cuMemSetAccess);
  int pidfd = syscall(SYS_pidfd_open, im->pid, 0);
  int fd = pidfd >= 0 ? syscall(SYS_pidfd_getfd, pidfd, im->fd, 0) : -1;
  if (pidfd >= 0) {
    close(pidfd);
  }
  if (fd < 0) {
    vlog("pidfd_getfd(pid %ld, fd %ld) failed: errno %ld\n", im->pid, im->fd, errno);
    return CUDA_ERROR_INVALID_HANDLE;
  }
  CUresult rc;
  CALL(rc, importfd(&im->h, (void *)(long)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  close(fd);
  if (rc != CUDA_SUCCESS) {
    return rc;
  }
  CALL(rc, map(im->va, im->size, 0, im->h, 0));
  if (rc == CUDA_SUCCESS) {
    CUmemAccessDesc acc;
    memset(&acc, 0, sizeof acc);
    acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    acc.location.id = im->dev;
    acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CALL(rc, access(im->va, im->size, &acc, 1));
  }
  im->mapped = rc == CUDA_SUCCESS;
  return rc;
}

static void
unmap_import(Import *im)
{
  UPPER(unmap, cuMemUnmap);
  UPPER(release, cuMemRelease);
  CUresult ignored;
  if (im->mapped) {
    CALL(ignored, unmap(im->va, im->size));
    CALL(ignored, release(im->h));
    im->mapped = 0;
  }
  (void)ignored;
}

CUresult
cuIpcOpenMemHandle_v2(CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags)
{
  VirtIpcHandle v;
  memcpy(&v, &handle, sizeof v);
  if (v.magic != VIPC_MAGIC) {
    UPPER(real, cuIpcOpenMemHandle_v2);
    CUresult rc;
    CALL(rc, real(pdptr, handle, Flags));
    return rc;
  }
  UPPER(reserve, cuMemAddressReserve);
  UPPER(getdev, cuCtxGetDevice);
  for (int i = 0; i < n_imports; i++) {
    if (imports[i].on && imports[i].pid == v.pid && imports[i].fd == v.fd) {
      *pdptr = imports[i].va;      // Opened already.
      return CUDA_SUCCESS;
    }
  }
  if (n_imports == 1024) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  Import *im = &imports[n_imports];
  memset(im, 0, sizeof *im);
  im->pid = v.pid;
  im->fd = v.fd;
  im->size = v.size;
  CUresult rc;
  CALL(rc, getdev(&im->dev));
  if (rc == CUDA_SUCCESS) {
    CALL(rc, reserve(&im->va, im->size, 0, 0, 0));
  }
  if (rc == CUDA_SUCCESS) {
    rc = map_import(im);
  }
  if (rc != CUDA_SUCCESS) {
    return rc;
  }
  im->on = 1;
  n_imports++;
  vlog("import pid %ld fd %ld at %#lx\n", v.pid, v.fd, im->va);
  *pdptr = im->va;
  return CUDA_SUCCESS;
}

#undef cuIpcOpenMemHandle
CUresult
cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle, unsigned int Flags)
{
  return cuIpcOpenMemHandle_v2(pdptr, handle, Flags);
}

CUresult
cuIpcCloseMemHandle(CUdeviceptr dptr)
{
  for (int i = 0; i < n_imports; i++) {
    Import *im = &imports[i];
    if (im->on && im->va == dptr) {
      UPPER(freeva, cuMemAddressFree);
      unmap_import(im);
      CUresult ignored;
      CALL(ignored, freeva(im->va, im->size));
      (void)ignored;
      im->on = 0;
      return CUDA_SUCCESS;
    }
  }
  UPPER(real, cuIpcCloseMemHandle);
  CUresult rc;
  CALL(rc, real(dptr));
  return rc;
}

// Before a checkpoint: no mapped import, no export of pre-checkpoint memory
// (the fd numbers stay taken, by /dev/null).
static void
vipc_before_ckpt(void)
{
  int ni = 0, ne = 0;
  for (int i = 0; i < n_imports; i++) {
    if (imports[i].on && imports[i].mapped) {
      unmap_import(&imports[i]);
      ni++;
    }
  }
  int null = open("/dev/null", O_RDONLY);
  for (int i = 0; i < n_exports; i++) {
    if (exports[i].on) {
      dup2(null, exports[i].fd);
      ne++;
    }
  }
  close(null);
  vlog("before checkpoint: %ld imports unmapped, %ld exports closed%s\n",
       ni, ne, (long)"");
}

// On resume, step 1 (every rank): export again, on the same fd numbers.
static void
vipc_after_resume_exports(void)
{
  for (int i = 0; i < n_exports; i++) {
    Export *e = &exports[i];
    if (e->on) {
      int fd = export_fd(e->base);
      if (fd < 0) {
        vlog("cannot export %#lx again\n", e->base, 0, 0);
        e->on = 0;
      } else {
        dup2(fd, e->fd);
        close(fd);
      }
    }
  }
}

// On resume, step 2 (after all ranks did step 1): import again, same VA.
static void
vipc_after_resume_imports(void)
{
  for (int i = 0; i < n_imports; i++) {
    Import *im = &imports[i];
    if (im->on && !im->mapped) {
      CUresult rc = map_import(im);
      if (rc != CUDA_SUCCESS) {
        fprintf(stderr, "[mana-fwd-ipc] cannot import pid %d fd %d again: %d\n",
                im->pid, im->fd, rc);
      }
    }
  }
}

/*
 * Pointer attributes, for virtual IPC: UCX shares memory with the legacy IPC
 * API if CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE.  Report it for VMM
 * memory that can be exported as a POSIX fd (our cuIpcGetMemHandle shares
 * it), and not for real legacy memory (cudaMalloc): its export would make
 * the process un-checkpointable, and the MPI must use a copy protocol.
 */
static int
vmm_posix_fd_memory(CUdeviceptr p)
{
  UPPER(attr, cuPointerGetAttribute);
  unsigned long long types = 0;
  CUresult rc;
  CALL(rc, attr(&types, CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES, p));
  return rc == CUDA_SUCCESS && (types & CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
}

// MANA_FWD_TRACE_ATTR: log the pointer attributes.  Read once: UCX queries
// the attributes of each message's buffers.
static int
trace_attr(void)
{
  static int on = -1;
  if (on < 0) {
    on = getenv("MANA_FWD_TRACE_ATTR") != NULL;
  }
  return on;
}

/*
 * Buffer ids across a resume.  The CUDA plugin checkpoints and restores the
 * GPU in place, which can move device memory to other physical pages, but
 * the driver keeps each allocation's buffer id.  The MPI's caches validate
 * entries by buffer id: libfabric's CUDA monitor keeps NIC registrations
 * (Slingshot/CXI), so after a resume the NIC would still DMA to the old
 * pages.  So the buffer ids that the lower half sees change on each resume.
 * (At restart, the lower half and its caches are new.)  ckpt_generation is
 * bumped by cuda_after_resume().
 */
static void
adjust_buffer_id(CUpointer_attribute attribute, void *data)
{
  if (attribute == CU_POINTER_ATTRIBUTE_BUFFER_ID && data != NULL) {
    *(unsigned long long *)data ^= ckpt_generation << 48;
  }
}

CUresult
cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr)
{
  UPPER(real, cuPointerGetAttribute);
  CUresult rc;
  CALL(rc, real(data, attribute, ptr));
  if (vipc_enabled() && rc == CUDA_SUCCESS &&
      attribute == CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE) {
    *(int *)data = vmm_posix_fd_memory(ptr);
  }
  if (rc == CUDA_SUCCESS) {
    adjust_buffer_id(attribute, data);
  }
  if (trace_attr()) {
    fprintf(stderr, "[mana-fwd] cuPointerGetAttribute(%d, %#llx) -> %d\n",
            (int)attribute, (unsigned long long)ptr, (int)rc);
  }
  return rc;
}

CUresult
cuPointerGetAttributes(unsigned int numAttributes, CUpointer_attribute *attributes,
                       void **data, CUdeviceptr ptr)
{
  UPPER(real, cuPointerGetAttributes);
  CUresult rc;
  CALL(rc, real(numAttributes, attributes, data, ptr));
  for (unsigned i = 0; rc == CUDA_SUCCESS && i < numAttributes; i++) {
    if (trace_attr()) {
      fprintf(stderr, "[mana-fwd] cuPointerGetAttributes[%u] = %d (%#llx)\n",
              i, (int)attributes[i], (unsigned long long)ptr);
    }
    if (vipc_enabled() && data[i] != NULL &&
        attributes[i] == CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE) {
      *(int *)data[i] = vmm_posix_fd_memory(ptr);
    }
    adjust_buffer_id(attributes[i], data[i]);
  }
  return rc;
}

static void *
own_function(const char *symbol)
{
  static const struct { const char *name; void *fn; } own[] = {
    { "cuGetProcAddress", (void *)cuGetProcAddress_v2 },
    { "cuGetExportTable", (void *)cuGetExportTable },
    { "cuMemHostRegister", (void *)cuMemHostRegister_v2 },
    { "cuMemHostUnregister", (void *)cuMemHostUnregister },
    { "cuMemRetainAllocationHandle", (void *)cuMemRetainAllocationHandle },
    { "cuIpcGetMemHandle", (void *)cuIpcGetMemHandle },
    { "cuIpcOpenMemHandle", (void *)cuIpcOpenMemHandle_v2 },
    { "cuIpcCloseMemHandle", (void *)cuIpcCloseMemHandle },
    { "cuPointerGetAttribute", (void *)cuPointerGetAttribute },
    { "cuPointerGetAttributes", (void *)cuPointerGetAttributes },
  };
  for (unsigned i = 0; i < sizeof own / sizeof own[0]; i++) {
    if (strcmp(symbol, own[i].name) == 0) {
      return own[i].fn;
    }
  }
  return NULL;
}
