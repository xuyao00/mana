/*
 * Hand-written functions of the libcudart.so.13 shim (the others are
 * generated trampolines): those that hand out upper-half function pointers.
 */
#include <cuda_runtime_api.h>

#include "fwd-runtime.h"

static void *
upper_sym(const char *symbol)
{
  void *p = NULL;
  FWD_CALL_UPPER(p, NULL, mana_fwd_ctl.uh_dlsym("libcudart.so.13", symbol));
  return p;
}

cudaError_t
cudaGetDriverEntryPoint(const char *symbol, void **funcPtr,
                        unsigned long long flags,
                        enum cudaDriverEntryPointQueryResult *driverStatus)
{
  static cudaError_t (*real)(const char *, void **, unsigned long long,
                             enum cudaDriverEntryPointQueryResult *) = NULL;
  if (real == NULL) {
    real = upper_sym("cudaGetDriverEntryPoint");
    if (real == NULL) {
      return cudaErrorNoDevice;
    }
  }
  cudaError_t rc;
  FWD_CALL_UPPER(rc, cudaErrorNoDevice, real(symbol, funcPtr, flags, driverStatus));
  if (rc == cudaSuccess && funcPtr != NULL && *funcPtr != NULL) {
    *funcPtr = mana_fwd_wrap_ptr(*funcPtr, symbol);
  }
  return rc;
}

cudaError_t
cudaGetDriverEntryPointByVersion(const char *symbol, void **funcPtr,
                        unsigned int cudaVersion, unsigned long long flags,
                        enum cudaDriverEntryPointQueryResult *driverStatus)
{
  static cudaError_t (*real)(const char *, void **, unsigned int,
                             unsigned long long,
                             enum cudaDriverEntryPointQueryResult *) = NULL;
  if (real == NULL) {
    real = upper_sym("cudaGetDriverEntryPointByVersion");
    if (real == NULL) {
      return cudaErrorNoDevice;
    }
  }
  cudaError_t rc;
  FWD_CALL_UPPER(rc, cudaErrorNoDevice, real(symbol, funcPtr, cudaVersion, flags, driverStatus));
  if (rc == cudaSuccess && funcPtr != NULL && *funcPtr != NULL) {
    *funcPtr = mana_fwd_wrap_ptr(*funcPtr, symbol);
  }
  return rc;
}

/*
 * Kernel registration.  Libraries of the lower half with device code (yaksa
 * in libmpi, NCCL, ...) register it from their constructors, before the upper
 * half exists.  So record the registrations, give the libraries handles of
 * our own, and replay them to the upper half's cudart once it is there
 * (mana_fwd_cudart_on_upper(), called by the runtime).  Afterwards, launches
 * from the lower half (cudaLaunchKernel with a lower-half host stub) find the
 * stubs registered in the application's cudart, as in a native process.
 */
#include <stdlib.h>
#include <string.h>

typedef struct {
  void *fatCubin;
  void **real;                // The upper half's handle, once replayed.
  int ended;
} FatHandle;

enum { REG_FUNCTION, REG_VAR, REG_MANAGED_VAR };
typedef struct {
  int kind;
  FatHandle *h;
  const char *hostFun;        // Function: host stub; var: host variable.
  char *deviceFun;
  const char *deviceName;
  int thread_limit;
  size_t size;
  int ext, constant, global;
} Registration;

static FatHandle **handles;
static int n_handles, cap_handles;
static Registration *regs;
static int n_regs, cap_regs;
static int replayed;

static void *(*real_sym(const char *name))(void)
{
  return (void *(*)(void))upper_sym(name);
}

static void
replay_handle(FatHandle *h)
{
  void **(*regFat)(void *) = (void **(*)(void *))real_sym("__cudaRegisterFatBinary");
  void (*regEnd)(void **) = (void (*)(void **))real_sym("__cudaRegisterFatBinaryEnd");
  void (*regFun)(void **, const char *, char *, const char *, int, void *,
                 void *, void *, void *, int *) =
    (void (*)(void **, const char *, char *, const char *, int, void *, void *,
              void *, void *, int *))real_sym("__cudaRegisterFunction");
  void (*regVar)(void **, char *, char *, const char *, int, size_t, int, int) =
    (void (*)(void **, char *, char *, const char *, int, size_t, int, int))
      real_sym("__cudaRegisterVar");
  void (*regMVar)(void **, void **, char *, const char *, int, size_t, int, int) =
    (void (*)(void **, void **, char *, const char *, int, size_t, int, int))
      real_sym("__cudaRegisterManagedVar");
  FWD_CALL_UPPER(h->real, NULL, regFat(h->fatCubin));
  int ignored = 0;
  for (int i = 0; i < n_regs; i++) {
    Registration *r = &regs[i];
    if (r->h != h) {
      continue;
    }
    switch (r->kind) {
    case REG_FUNCTION:
      FWD_CALL_UPPER(ignored, 0,
        (regFun(h->real, r->hostFun, r->deviceFun, r->deviceName,
                r->thread_limit, NULL, NULL, NULL, NULL, NULL), 0));
      break;
    case REG_VAR:
      FWD_CALL_UPPER(ignored, 0,
        (regVar(h->real, (char *)r->hostFun, r->deviceFun, r->deviceName,
                r->ext, r->size, r->constant, r->global), 0));
      break;
    case REG_MANAGED_VAR:
      FWD_CALL_UPPER(ignored, 0,
        (regMVar(h->real, (void **)r->hostFun, r->deviceFun, r->deviceName,
                 r->ext, r->size, r->constant, r->global), 0));
      break;
    }
  }
  if (h->ended) {
    FWD_CALL_UPPER(ignored, 0, (regEnd(h->real), 0));
  }
  (void)ignored;
}

void
mana_fwd_cudart_on_upper(void)
{
  if (replayed) {
    return;
  }
  replayed = 1;
  for (int i = 0; i < n_handles; i++) {
    replay_handle(handles[i]);
  }
}

static Registration *
new_reg(int kind, void **handle)
{
  if (n_regs == cap_regs) {
    cap_regs = cap_regs ? 2 * cap_regs : 1024;
    regs = (Registration *)realloc(regs, cap_regs * sizeof *regs);
  }
  Registration *r = &regs[n_regs++];
  memset(r, 0, sizeof *r);
  r->kind = kind;
  r->h = (FatHandle *)handle;
  return r;
}

void **
__cudaRegisterFatBinary(void *fatCubin)
{
  FatHandle *h = (FatHandle *)calloc(1, sizeof *h);
  h->fatCubin = fatCubin;
  if (n_handles == cap_handles) {
    cap_handles = cap_handles ? 2 * cap_handles : 64;
    handles = (FatHandle **)realloc(handles, cap_handles * sizeof *handles);
  }
  handles[n_handles++] = h;
  return (void **)h;
}

void
__cudaRegisterFatBinaryEnd(void **fatCubinHandle)
{
  FatHandle *h = (FatHandle *)fatCubinHandle;
  h->ended = 1;
  if (replayed) {                     // Registered after the replay.
    replay_handle(h);
  }
}

void
__cudaUnregisterFatBinary(void **fatCubinHandle)
{
  // At exit; the upper half's cudart cleans up its own modules.
  (void)fatCubinHandle;
}

void
__cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                       char *deviceFun, const char *deviceName,
                       int thread_limit, void *tid, void *bid, void *bDim,
                       void *gDim, int *wSize)
{
  Registration *r = new_reg(REG_FUNCTION, fatCubinHandle);
  r->hostFun = hostFun;
  r->deviceFun = deviceFun;
  r->deviceName = deviceName;
  r->thread_limit = thread_limit;
}

void
__cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
                  const char *deviceName, int ext, size_t size, int constant,
                  int global)
{
  Registration *r = new_reg(REG_VAR, fatCubinHandle);
  r->hostFun = hostVar;
  r->deviceFun = deviceAddress;
  r->deviceName = deviceName;
  r->ext = ext;
  r->size = size;
  r->constant = constant;
  r->global = global;
}

void
__cudaRegisterManagedVar(void **fatCubinHandle, void **hostVarPtrAddress,
                         char *deviceAddress, const char *deviceName, int ext,
                         size_t size, int constant, int global)
{
  Registration *r = new_reg(REG_MANAGED_VAR, fatCubinHandle);
  r->hostFun = (const char *)hostVarPtrAddress;
  r->deviceFun = deviceAddress;
  r->deviceName = deviceName;
  r->ext = ext;
  r->size = size;
  r->constant = constant;
  r->global = global;
}

char
__cudaInitModule(void **fatCubinHandle)
{
  FatHandle *h = (FatHandle *)fatCubinHandle;
  char (*init)(void **) = (char (*)(void **))real_sym("__cudaInitModule");
  char rc = 0;
  if (h->real != NULL && init != NULL) {
    FWD_CALL_UPPER(rc, 0, init(h->real));
  }
  return rc;
}

// Host-memory registrations: see cuda-special.c.
#include <cuda.h>

cudaError_t
cudaHostRegister(void *ptr, size_t size, unsigned int flags)
{
  static cudaError_t (*real)(void *, size_t, unsigned) = NULL;
  static CUresult (*getCur)(CUcontext *) = NULL;
  if (real == NULL) {
    real = upper_sym("cudaHostRegister");
    void *p = NULL;
    FWD_CALL_UPPER(p, NULL, mana_fwd_ctl.uh_dlsym("libcuda.so.1", "cuCtxGetCurrent"));
    getCur = p;
    if (real == NULL) {
      return cudaErrorNoDevice;
    }
  }
  cudaError_t rc;
  FWD_CALL_UPPER(rc, cudaErrorNoDevice, real(ptr, size, flags));
  if (rc == cudaSuccess) {
    CUcontext ctx = NULL;
    CUresult ignored;
    FWD_CALL_UPPER(ignored, CUDA_ERROR_NO_DEVICE, getCur(&ctx));
    (void)ignored;
    mana_fwd_note_host_register(ptr, size, flags, ctx);
  }
  return rc;
}

cudaError_t
cudaHostUnregister(void *ptr)
{
  static cudaError_t (*real)(void *) = NULL;
  if (real == NULL) {
    real = upper_sym("cudaHostUnregister");
    if (real == NULL) {
      return cudaErrorNoDevice;
    }
  }
  cudaError_t rc;
  FWD_CALL_UPPER(rc, cudaErrorNoDevice, real(ptr));
  mana_fwd_note_host_unregister(ptr);
  return rc;
}
