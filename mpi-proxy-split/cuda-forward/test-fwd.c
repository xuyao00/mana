// Standalone test of the forwarding shim of libcuda.so.1 (no MANA): the
// "upper half" is the real driver loaded by path, and this thread registers
// its own FS as the upper half's (so the FS switch is a no-op).  Checks that
// arguments (incl. the stack arguments of cuLaunchKernel), return values and
// cuGetProcAddress go through the trampolines.
//   LD_LIBRARY_PATH=lh-shims ./test-fwd
#define _GNU_SOURCE
#include <asm/prctl.h>
#include <cuda.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/auxv.h>
#include <time.h>
#include <pthread.h>

#include "fwd-runtime.h"

static void *real_handle;

static void *
test_uh_dlsym(const char *soname, const char *symbol)
{
  if (real_handle == NULL) {
    real_handle = dlopen(getenv("REAL_LIBCUDA"), RTLD_NOW | RTLD_LOCAL);
  }
  return real_handle ? dlsym(real_handle, symbol) : NULL;
}

static const char *ptx =
  ".version 7.0\n.target sm_52\n.address_size 64\n"
  ".visible .entry add(.param .u64 a, .param .u32 v, .param .u32 n) {\n"
  "  .reg .b32 %r<6>; .reg .b64 %rd<5>; .reg .pred %p;\n"
  "  ld.param.u64 %rd1, [a]; ld.param.u32 %r1, [v]; ld.param.u32 %r2, [n];\n"
  "  mov.u32 %r3, %tid.x; setp.ge.u32 %p, %r3, %r2; @%p bra DONE;\n"
  "  cvta.to.global.u64 %rd2, %rd1; mul.wide.u32 %rd3, %r3, 4;\n"
  "  add.s64 %rd4, %rd2, %rd3; ld.global.u32 %r4, [%rd4];\n"
  "  add.s32 %r5, %r4, %r1; st.global.u32 [%rd4], %r5;\n"
  "DONE: ret;\n}\n";

#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { \
  printf("FAIL %s -> %d\n", #x, r_); return 1; } } while (0)

int
main(void)
{
  unsigned long fs;
  syscall(SYS_arch_prctl, ARCH_GET_FS, &fs);
  FwdCtl *ctl = mana_fwd_get_ctl();
  ctl->uh_dlsym = test_uh_dlsym;
  ctl->register_thread(syscall(SYS_gettid), fs);

  CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn; CUdeviceptr d;
  CK(cuInit(0));
  CK(cuDeviceGet(&dev, 0));
  char name[128];
  CK(cuDeviceGetName(name, sizeof name, dev));
  printf("device: %s\n", name);
  CK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CK(cuCtxSetCurrent(ctx));
  CK(cuModuleLoadData(&mod, ptx));
  CK(cuModuleGetFunction(&fn, mod, "add"));
  unsigned n = 256, v = 7, host[256];
  CK(cuMemAlloc(&d, n * 4));
  CK(cuMemsetD32(d, 35, n));
  void *args[] = { &d, &v, &n };
  CK(cuLaunchKernel(fn, 1, 1, 1, n, 1, 1, 0, NULL, args, NULL));
  CK(cuCtxSynchronize());
  CK(cuMemcpyDtoH(host, d, n * 4));
  printf("kernel result: %u (want 42)\n", host[n - 1]);

  // A pointer from cuGetProcAddress must be a trampoline, and work.
  CUresult (*getCount)(int *) = NULL;
  CUdriverProcAddressQueryResult st;
  CK(cuGetProcAddress("cuDeviceGetCount", (void **)&getCount, 12000, 0, &st));
  int count = -1;
  CK(getCount(&count));
  printf("cuGetProcAddress: %p -> count %d\n", (void *)getCount, count);
  const char *err = NULL;
  cuGetErrorName(CUDA_ERROR_INVALID_VALUE, &err);
  printf("cuGetErrorName: %s\n", err);
  CK(cuMemFree(d));

  // Cost of a forwarded call (FS switched with wrfsbase if possible).
  unsigned long hw = getauxval(AT_HWCAP2);
  *ctl->fsgsbase = (hw & 2) != 0;
  {   // with the stack range, as MANA registers its threads
    pthread_attr_t a; void *lo; size_t sz;
    pthread_getattr_np(pthread_self(), &a); pthread_attr_getstack(&a, &lo, &sz);
    ctl->register_thread_stack(syscall(SYS_gettid), fs, (unsigned long)lo,
                               (unsigned long)lo + sz);
  }
  CUresult (*direct)(CUcontext *) =
    (CUresult (*)(CUcontext *))test_uh_dlsym(NULL, "cuCtxGetCurrent");
  const int N = 1000000;
  CUcontext c;
  struct timespec t0, t1, t2;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (int i = 0; i < N; i++) direct(&c);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  for (int i = 0; i < N; i++) cuCtxGetCurrent(&c);
  clock_gettime(CLOCK_MONOTONIC, &t2);
  double td = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
  double tf = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
  printf("cuCtxGetCurrent: direct %.1f ns, forwarded %.1f ns (fsgsbase %d): "
         "%.1f ns per forwarded call\n", td / N, tf / N, *ctl->fsgsbase,
         (tf - td) / N);
  printf("%s\n", host[n - 1] == 42 && count >= 1 ? "PASS" : "FAIL");
  return 0;
}
