// Two independent libcuda instances in one process (as in MANA's split
// process), then an external cuCheckpointProcess* lock/ckpt/restore/unlock.
//   ./two_drivers N   -- N = 1 or 2 driver instances
// It prints its pid, waits for "go" files, and verifies device memory.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>

typedef struct {
  CUresult (*Init)(unsigned);
  CUresult (*DeviceGet)(CUdevice*, int);
  CUresult (*PrimaryCtxRetain)(CUcontext*, CUdevice);
  CUresult (*CtxSetCurrent)(CUcontext);
  CUresult (*MemAlloc)(CUdeviceptr*, size_t);
  CUresult (*MemsetD32)(CUdeviceptr, unsigned, size_t);
  CUresult (*MemcpyDtoH)(void*, CUdeviceptr, size_t);
  CUcontext ctx; CUdeviceptr buf;
} Drv;

#define SYM(d, h, f, n) *(void**)&d->f = dlsym(h, n); if (!d->f) { printf("no %s\n", n); exit(1); }

static void load(Drv *d, int i) {
  void *h = dlmopen(LM_ID_NEWLM, "libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!h) { printf("dlmopen: %s\n", dlerror()); exit(1); }
  SYM(d, h, Init, "cuInit"); SYM(d, h, DeviceGet, "cuDeviceGet");
  SYM(d, h, PrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
  SYM(d, h, CtxSetCurrent, "cuCtxSetCurrent"); SYM(d, h, MemAlloc, "cuMemAlloc_v2");
  SYM(d, h, MemsetD32, "cuMemsetD32_v2"); SYM(d, h, MemcpyDtoH, "cuMemcpyDtoH_v2");
  CUdevice dev;
  printf("inst %d: init %d", i, d->Init(0)); d->DeviceGet(&dev, 0);
  printf(" ctx %d", d->PrimaryCtxRetain(&d->ctx, dev)); d->CtxSetCurrent(d->ctx);
  printf(" alloc %d", d->MemAlloc(&d->buf, 1 << 20));
  printf(" memset %d buf=%llx\n", d->MemsetD32(d->buf, 1000 + i, (1 << 20) / 4), (unsigned long long)d->buf);
}

static void verify(Drv *d, int i, const char *when) {
  unsigned v = 0;
  d->CtxSetCurrent(d->ctx);
  CUresult r = d->MemcpyDtoH(&v, d->buf + 4096, 4);
  printf("inst %d %s: memcpy %d value %u (%s)\n", i, when, r, v, v == 1000u + i ? "OK" : "BAD");
  fflush(stdout);
}

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 2;
  Drv d[2];
  for (int i = 0; i < n; i++) load(&d[i], i);
  for (int i = 0; i < n; i++) verify(&d[i], i, "before");
  printf("pid %d ready\n", getpid()); fflush(stdout);
  while (access("go", F_OK) != 0) usleep(100000);
  for (int i = 0; i < n; i++) verify(&d[i], i, "after");
  return 0;
}
