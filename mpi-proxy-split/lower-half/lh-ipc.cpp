// Virtual legacy CUDA IPC, for an MPI that knows only legacy IPC (Cray's
// GTL; cuda/DESIGN-lower-half-cuda.md, section 9).
//
// Under --cuda, the application's device memory is VMM memory
// (cuda-front/front_vmm.cpp), for which the driver gives no legacy IPC handle:
// GTL's cuIpcGetMemHandle fails, and Cray MPICH then aborts the send.  The
// allocations can be exported as a POSIX fd, so bin/lower-half defines these
// functions and exports them (lh-ipc.list), ahead of libcuda's, for the MPI:
//  - cuPointerGetAttribute(s) report such memory as legacy-IPC capable;
//  - cuIpcGetMemHandle exports the allocation as a POSIX fd, once, and returns
//    a handle that names it: the exporter's pid, the fd and the allocation's
//    buffer id;
//  - cuIpcOpenMemHandle takes the fd with pidfd_getfd(), imports it, and maps
//    it at an address that the driver picks;
//  - cuIpcCloseMemHandle unmaps it.
// Other memory, such as the MPI's own cudaMalloc buffers, keeps the real
// legacy IPC.  The front reaches libcuda through its own handle
// (lh_cuda_dlsym()), so the application's calls never come here.
//
// This is cuda-aware-pr's cuda-forward/cuda-special.c without its checkpoint
// handling: the lower half, with these exports and imports and the MPI's IPC
// cache, is not saved, and on resume the memory has not moved.
//
// MANA_CUDA_VIRTUAL_IPC=1 or 0 turns it on or off under --cuda.  By default it
// is on when the lower half has Cray's GTL (libmpi_gtl_cuda); UCX shares VMM
// memory without it, through its own POSIX-fd path.

#include <cuda.h>
#include <dlfcn.h>
#include <errno.h>
#include <link.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#define VIPC_MAGIC 0x3343504941414e4dULL      // "MANAIPC3"
#define MAX_EXPORTS 1024
#define MAX_IMPORTS 1024

typedef struct {
  unsigned long long magic;
  int pid;                    // the exporter's real pid
  int fd;                     // its export
  unsigned long long size;
  unsigned long long id;      // CU_POINTER_ATTRIBUTE_BUFFER_ID
} VirtIpcHandle;              // fits in CUipcMemHandle (64 bytes)

typedef struct {
  CUdeviceptr base;
  size_t size;
  unsigned long long id;
  int fd;
  int on;
} Export;

typedef struct {
  CUdeviceptr va;
  size_t size;
  int pid, fd;
  unsigned long long id;
  CUmemGenericAllocationHandle h;
  int refs;                   // opens not closed yet; 0: a free slot
} Import;

// In .bss: not malloc'ed memory, which the upper half's restore could map
// over.
static Export exports[MAX_EXPORTS];
static int n_exports;
static Import imports[MAX_IMPORTS];
static int n_imports;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static int enabled;
static int verbose;
// The real pid, for the handles: GTL asks for a handle per message, and a
// syscall costs 0.4 us on Perlmutter.
static int my_pid;

// libcuda's functions: the next definitions after this executable's.
static struct {
  decltype(&cuIpcGetMemHandle) ipc_get;
  decltype(&cuIpcOpenMemHandle_v2) ipc_open;
  decltype(&cuIpcCloseMemHandle) ipc_close;
  decltype(&cuPointerGetAttribute) attr;
  decltype(&cuPointerGetAttributes) attrs;
  decltype(&cuMemRetainAllocationHandle) retain;
  decltype(&cuMemExportToShareableHandle) export_fd;
  decltype(&cuMemImportFromShareableHandle) import_fd;
  decltype(&cuMemRelease) release;
  decltype(&cuMemAddressReserve) reserve;
  decltype(&cuMemAddressFree) free_va;
  decltype(&cuMemMap) map;
  decltype(&cuMemUnmap) unmap;
  decltype(&cuMemSetAccess) access;
  decltype(&cuCtxGetDevice) ctx_device;
} real;

static void
vlog(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

static void
vlog(const char *fmt, ...)
{
  if (!verbose) {
    return;
  }
  char buf[256];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  fprintf(stderr, "MANA lh-ipc[%d]: %s", getpid(), buf);
}

static int
has_gtl(struct dl_phdr_info *info, size_t size, void *data)
{
  (void)size;
  (void)data;
  return info->dlpi_name != NULL &&
         strstr(info->dlpi_name, "libmpi_gtl_cuda") != NULL;
}

// Called by init_lower_half_cuda(), under --cuda only, at launch and at
// restart: in main(), before the upper half exists.
extern "C" void
lh_ipc_init()
{
  const char *v = getenv("MANA_CUDA_VERBOSE");
  verbose = v != NULL && v[0] != '\0' && v[0] != '0';
  my_pid = (int)syscall(SYS_getpid);
#define REAL(field, name) \
  real.field = (decltype(real.field))dlsym(RTLD_NEXT, #name)
  REAL(ipc_get, cuIpcGetMemHandle);
  REAL(ipc_open, cuIpcOpenMemHandle_v2);
  REAL(ipc_close, cuIpcCloseMemHandle);
  REAL(attr, cuPointerGetAttribute);
  REAL(attrs, cuPointerGetAttributes);
  REAL(retain, cuMemRetainAllocationHandle);
  REAL(export_fd, cuMemExportToShareableHandle);
  REAL(import_fd, cuMemImportFromShareableHandle);
  REAL(release, cuMemRelease);
  REAL(reserve, cuMemAddressReserve);
  REAL(free_va, cuMemAddressFree);
  REAL(map, cuMemMap);
  REAL(unmap, cuMemUnmap);
  REAL(access, cuMemSetAccess);
  REAL(ctx_device, cuCtxGetDevice);
#undef REAL
  const char *on = getenv("MANA_CUDA_VIRTUAL_IPC");
  if (on != NULL && on[0] != '\0') {
    enabled = strcmp(on, "0") != 0;
  } else {
    enabled = dl_iterate_phdr(has_gtl, NULL) != 0;
  }
  vlog("virtual legacy IPC %s\n", enabled ? "on" : "off");
}

// The calls of the MPI that come before lh_ipc_init() (in MPI_Init), or
// without --cuda, go to libcuda.
static void *
next(void **fn, const char *name)
{
  if (*fn == NULL) {
    *fn = dlsym(RTLD_NEXT, name);
  }
  return *fn;
}
#define NEXT(field, name) \
  ((decltype(real.field))next((void **)&real.field, #name))

// Whether 'p' is VMM memory that can be exported as a POSIX fd.
static int
vmm_posix_fd_memory(CUdeviceptr p)
{
  unsigned long long types = 0;
  return real.attr(&types, CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES, p) ==
         CUDA_SUCCESS && (types & CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
}

// A fresh POSIX-fd export of the VMM allocation at 'base', or -1.
static int
export_fd(CUdeviceptr base)
{
  CUmemGenericAllocationHandle h;
  if (real.retain(&h, (void *)base) != CUDA_SUCCESS) {
    return -1;                // not VMM memory
  }
  int fd = -1;
  CUresult rc = real.export_fd(&fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                               0);
  real.release(h);
  return rc == CUDA_SUCCESS ? fd : -1;
}

extern "C" CUresult
cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr)
{
  if (!enabled || pHandle == NULL) {
    return NEXT(ipc_get, cuIpcGetMemHandle)(pHandle, dptr);
  }
  // GTL asks for a handle per message: one driver call.  For VMM memory, the
  // MAPPING attributes give the allocation's range, as cuMemGetAddressRange
  // does; the RANGE attributes give the whole reservation.  For a pointer
  // that the driver does not know, the call succeeds with zeros.
  CUpointer_attribute which[4] = { CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR,
                                   CU_POINTER_ATTRIBUTE_MAPPING_SIZE,
                                   CU_POINTER_ATTRIBUTE_BUFFER_ID,
                                   CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES };
  CUdeviceptr base = 0;
  size_t size = 0;
  unsigned long long id = 0;
  unsigned long long types = 0;
  void *data[4] = { &base, &size, &id, &types };
  CUresult rc = real.attrs(4, which, data, dptr);
  if (rc != CUDA_SUCCESS) {
    return rc;
  }
  if (size == 0 || !(types & CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)) {
    return real.ipc_get(pHandle, dptr);
  }
  pthread_mutex_lock(&lock);
  // The application may have freed the memory of an export, and the front
  // may have given its address to another allocation: the buffer id differs.
  Export *e = NULL;
  for (int i = 0; i < n_exports && e == NULL; i++) {
    if (exports[i].on && exports[i].base == base) {
      if (exports[i].id == id) {
        e = &exports[i];
      } else {
        close(exports[i].fd);
        exports[i].on = 0;
      }
    }
  }
  if (e == NULL) {
    int i;
    for (i = 0; i < n_exports && exports[i].on; i++) {
    }
    int fd = i < MAX_EXPORTS ? export_fd(base) : -1;
    if (fd < 0) {
      pthread_mutex_unlock(&lock);
      vlog("cannot export %#llx\n", (unsigned long long)base);
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    e = &exports[i];
    *e = (Export){ base, size, id, fd, 1 };
    if (i == n_exports) {
      n_exports++;
    }
    vlog("export %#llx (%zu bytes) as fd %d\n", (unsigned long long)base, size,
         fd);
  }
  VirtIpcHandle v = { VIPC_MAGIC, my_pid, e->fd, e->size, e->id };
  pthread_mutex_unlock(&lock);
  memset(pHandle, 0, sizeof *pHandle);
  memcpy(pHandle, &v, sizeof v);
  return CUDA_SUCCESS;
}

// Imports 'im' and maps it at a new address, for the current device.
static CUresult
map_import(Import *im)
{
  CUdevice dev;
  CUresult rc = real.ctx_device(&dev);
  if (rc != CUDA_SUCCESS) {
    return rc;
  }
  int pidfd = syscall(SYS_pidfd_open, im->pid, 0);
  int fd = pidfd >= 0 ? syscall(SYS_pidfd_getfd, pidfd, im->fd, 0) : -1;
  if (fd < 0) {
    vlog("pidfd_getfd(pid %d, fd %d): %s\n", im->pid, im->fd, strerror(errno));
  }
  if (pidfd >= 0) {
    close(pidfd);
  }
  if (fd < 0) {
    return CUDA_ERROR_INVALID_HANDLE;
  }
  rc = real.import_fd(&im->h, (void *)(long)fd,
                      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  close(fd);
  if (rc != CUDA_SUCCESS) {
    return rc;
  }
  rc = real.reserve(&im->va, im->size, 0, 0, 0);
  if (rc == CUDA_SUCCESS) {
    rc = real.map(im->va, im->size, 0, im->h, 0);
    if (rc == CUDA_SUCCESS) {
      CUmemAccessDesc acc;
      memset(&acc, 0, sizeof acc);
      acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      acc.location.id = dev;
      acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      rc = real.access(im->va, im->size, &acc, 1);
      if (rc != CUDA_SUCCESS) {
        real.unmap(im->va, im->size);
      }
    }
    if (rc != CUDA_SUCCESS) {
      real.free_va(im->va, im->size);
    }
  }
  if (rc != CUDA_SUCCESS) {
    real.release(im->h);
  }
  return rc;
}

extern "C" CUresult
cuIpcOpenMemHandle_v2(CUdeviceptr *pdptr, CUipcMemHandle handle,
                      unsigned int Flags)
{
  VirtIpcHandle v;
  memcpy(&v, &handle, sizeof v);
  if (v.magic != VIPC_MAGIC || pdptr == NULL) {
    return NEXT(ipc_open, cuIpcOpenMemHandle_v2)(pdptr, handle, Flags);
  }
  pthread_mutex_lock(&lock);
  int i;
  for (i = 0; i < n_imports; i++) {
    Import *im = &imports[i];
    if (im->refs > 0 && im->pid == v.pid && im->fd == v.fd && im->id == v.id) {
      im->refs++;
      *pdptr = im->va;
      pthread_mutex_unlock(&lock);
      return CUDA_SUCCESS;
    }
  }
  for (i = 0; i < n_imports && imports[i].refs > 0; i++) {
  }
  if (i == MAX_IMPORTS) {
    pthread_mutex_unlock(&lock);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  Import *im = &imports[i];
  *im = (Import){ 0, v.size, v.pid, v.fd, v.id, 0, 0 };
  CUresult rc = map_import(im);
  if (rc == CUDA_SUCCESS) {
    im->refs = 1;
    if (i == n_imports) {
      n_imports++;
    }
    *pdptr = im->va;
    vlog("import pid %d fd %d (%llu bytes) at %#llx\n", v.pid, v.fd, v.size,
         (unsigned long long)im->va);
  }
  pthread_mutex_unlock(&lock);
  return rc;
}

// cuda.h names cuIpcOpenMemHandle_v2 so; libcuda exports both names.
#undef cuIpcOpenMemHandle
extern "C" CUresult
cuIpcOpenMemHandle(CUdeviceptr *pdptr, CUipcMemHandle handle,
                   unsigned int Flags)
{
  return cuIpcOpenMemHandle_v2(pdptr, handle, Flags);
}

extern "C" CUresult
cuIpcCloseMemHandle(CUdeviceptr dptr)
{
  pthread_mutex_lock(&lock);
  for (int i = 0; i < n_imports; i++) {
    Import *im = &imports[i];
    if (im->refs > 0 && im->va == dptr) {
      if (--im->refs == 0) {
        real.unmap(im->va, im->size);
        real.release(im->h);
        real.free_va(im->va, im->size);
        vlog("close pid %d fd %d at %#llx\n", im->pid, im->fd,
             (unsigned long long)im->va);
      }
      pthread_mutex_unlock(&lock);
      return CUDA_SUCCESS;
    }
  }
  pthread_mutex_unlock(&lock);
  return NEXT(ipc_close, cuIpcCloseMemHandle)(dptr);
}

// The MPI asks whether memory has legacy IPC (UCX does, before it chooses
// its path).
static void
adjust_attribute(CUpointer_attribute attribute, void *data, CUdeviceptr p)
{
  if (attribute == CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE &&
      data != NULL && *(int *)data == 0 && vmm_posix_fd_memory(p)) {
    *(int *)data = 1;
  }
}

extern "C" CUresult
cuPointerGetAttribute(void *data, CUpointer_attribute attribute,
                      CUdeviceptr ptr)
{
  CUresult rc = NEXT(attr, cuPointerGetAttribute)(data, attribute, ptr);
  if (enabled && rc == CUDA_SUCCESS) {
    adjust_attribute(attribute, data, ptr);
  }
  return rc;
}

extern "C" CUresult
cuPointerGetAttributes(unsigned int numAttributes,
                       CUpointer_attribute *attributes, void **data,
                       CUdeviceptr ptr)
{
  CUresult rc = NEXT(attrs, cuPointerGetAttributes)(numAttributes, attributes,
                                                    data, ptr);
  for (unsigned i = 0; enabled && rc == CUDA_SUCCESS && i < numAttributes;
       i++) {
    adjust_attribute(attributes[i], data[i], ptr);
  }
  return rc;
}
