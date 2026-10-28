// The CUDA driver API (run with MANA's --cuda), without the runtime: a
// context of its own (cuCtxCreate) next to the device's primary one, a
// module JIT-compiled from PTX with options and one from a fatbin, a module
// global that only kernels update, a function whose attribute (more than
// 48 KB of dynamic shared memory) a restart must set again, streams, events,
// pinned memory, and a function pointer from cuGetProcAddress kept across
// checkpoints.  MPI reduces a device buffer of the created context.

#include <cuda.h>

#include "cuda_driver_kernels.h"  // driver_ptx[], driver_fatbin[] (bin2c)
#include "mana_test.h"

#define N 4096
#define SHARED_INTS (64 * 1024 / 4)

#define CU_CHECK(call)                                                       \
  do {                                                                       \
    CUresult r_ = (call);                                                    \
    const char *s_ = "?";                                                    \
    cuGetErrorName(r_, &s_);                                                 \
    MT_CHECK(r_ == CUDA_SUCCESS, "%s: %s", #call, s_);                       \
  } while (0)

static void
launch(CUfunction f, unsigned grid, unsigned block, unsigned shared,
       CUstream stream, void **params)
{
  CU_CHECK(cuLaunchKernel(f, grid, 1, 1, block, 1, 1, shared, stream, params,
                          NULL));
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "cuda_driver");
  CU_CHECK(cuInit(0));
  int ndev = 0;
  CU_CHECK(cuDeviceGetCount(&ndev));
  CUdevice dev;
  CU_CHECK(cuDeviceGet(&dev, mt_rank % ndev));

  // The primary context, with a module from the fatbin.
  CUcontext primary;
  CU_CHECK(cuDevicePrimaryCtxRetain(&primary, dev));
  CU_CHECK(cuCtxPushCurrent(primary));
  CUmodule fat_module;
  CUfunction fat_add_one;
  CU_CHECK(cuModuleLoadFatBinary(&fat_module, driver_fatbin));
  CU_CHECK(cuModuleGetFunction(&fat_add_one, fat_module, "add_one"));
  CUdeviceptr pbuf;
  CU_CHECK(cuMemAlloc(&pbuf, N * sizeof(int)));
  CU_CHECK(cuMemsetD32(pbuf, 0, N));
  CUcontext popped;
  CU_CHECK(cuCtxPopCurrent(&popped));
  MT_CHECK(popped == primary, "%s", "cuCtxPopCurrent: not the primary");

  // A context of its own, current from now on, with a module from PTX.
  CUcontext ctx;
  CU_CHECK(cuCtxCreate(&ctx, NULL, 0, dev));
  char *ptx = malloc(sizeof driver_ptx + 1);
  memcpy(ptx, driver_ptx, sizeof driver_ptx);
  ptx[sizeof driver_ptx] = '\0';
  char log[4096];
  CUjit_option options[] = { CU_JIT_OPTIMIZATION_LEVEL,
                             CU_JIT_ERROR_LOG_BUFFER,
                             CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES };
  void *values[] = { (void *)(uintptr_t)3, log, (void *)sizeof log };
  CUmodule module;
  CU_CHECK(cuModuleLoadDataEx(&module, ptx, 3, options, values));
  free(ptx);  // The front keeps its own copy.
  CUfunction add_one, bump, through_shared;
  CU_CHECK(cuModuleGetFunction(&add_one, module, "add_one"));
  CU_CHECK(cuModuleGetFunction(&bump, module, "bump"));
  CU_CHECK(cuModuleGetFunction(&through_shared, module, "through_shared"));
  CU_CHECK(cuFuncSetAttribute(through_shared,
                              CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                              SHARED_INTS * sizeof(int)));
  CUdeviceptr counter0;
  size_t counter_size;
  CU_CHECK(cuModuleGetGlobal(&counter0, &counter_size, module, "counter"));
  MT_CHECK(counter_size == sizeof(int), "counter: %zu bytes", counter_size);
  CU_CHECK(cuMemsetD32(counter0, 0, 1));

  CUstream stream;
  CU_CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CUevent done;
  CU_CHECK(cuEventCreate(&done, CU_EVENT_DISABLE_TIMING));
  CUdeviceptr state, out, dsum;
  CU_CHECK(cuMemAlloc(&state, SHARED_INTS * sizeof(int)));
  CU_CHECK(cuMemAlloc(&out, sizeof(int)));
  CU_CHECK(cuMemAlloc(&dsum, sizeof(int)));
  CU_CHECK(cuMemsetD32(state, 0, SHARED_INTS));
  int *host;
  CU_CHECK(cuMemAllocHost((void **)&host, 4 * sizeof(int)));

  // Called through this pointer after restarts too.
  CUresult (*mem_alloc)(CUdeviceptr *, size_t) = NULL;
  CUdriverProcAddressQueryResult status;
  CU_CHECK(cuGetProcAddress("cuMemAlloc", (void **)&mem_alloc, 13000,
                            CU_GET_PROC_ADDRESS_DEFAULT, &status));
  MT_CHECK(mem_alloc != NULL && status == CU_GET_PROC_ADDRESS_SUCCESS,
           "%s", "cuGetProcAddress(cuMemAlloc)");

  long it;
  for (it = 0; mt_continue(it); it++) {
    CUcontext cur;
    CU_CHECK(cuCtxGetCurrent(&cur));
    MT_CHECK(cur == ctx, "iteration %ld: current context %p, not %p", it,
             (void *)cur, (void *)ctx);
    CUdeviceptr counter;
    CU_CHECK(cuModuleGetGlobal(&counter, NULL, module, "counter"));
    MT_CHECK(counter == counter0, "iteration %ld: counter moved", it);

    int n = SHARED_INTS;
    void *state_params[] = { &state, &n };
    void *out_params[] = { &out };
    launch(add_one, (n + 255) / 256, 256, 0, stream, state_params);
    launch(through_shared, 1, 256, SHARED_INTS * sizeof(int), stream,
           state_params);
    launch(bump, 1, 1, 0, stream, out_params);
    CU_CHECK(cuEventRecord(done, stream));
    CU_CHECK(cuMemcpyDtoHAsync(&host[0], state, sizeof(int), stream));
    CU_CHECK(cuMemcpyDtoHAsync(&host[1], state + (n - 1) * sizeof(int),
                               sizeof(int), stream));
    CU_CHECK(cuMemcpyDtoHAsync(&host[2], out, sizeof(int), stream));
    CU_CHECK(cuEventSynchronize(done));
    CU_CHECK(cuStreamSynchronize(stream));
    MT_CHECK(host[0] == it + 1 && host[1] == it + 1,
             "iteration %ld: state %d %d", it, host[0], host[1]);
    MT_CHECK(host[2] == it + 1, "iteration %ld: counter %d", it, host[2]);

    int mine = mt_value(mt_rank, it, 0);
    CU_CHECK(cuMemcpyHtoD(dsum, &mine, sizeof(int)));
    CU_CHECK(cuCtxSynchronize());
    MT_MPI(MPI_Allreduce(MPI_IN_PLACE, (void *)dsum, 1, MPI_INT, MPI_SUM,
                         MPI_COMM_WORLD));
    // The MPI makes the primary context current, as it does natively.
    CU_CHECK(cuCtxSetCurrent(ctx));
    int sum = 0, want = 0;
    CU_CHECK(cuMemcpyDtoH(&sum, dsum, sizeof(int)));
    for (int r = 0; r < mt_size; r++) {
      want += mt_value(r, it, 0);
    }
    MT_CHECK(sum == want, "iteration %ld: MPI_Allreduce: %d, not %d", it, sum,
             want);

    // The primary context's module, on its legacy stream.
    CU_CHECK(cuCtxPushCurrent(primary));
    int pn = N;
    void *pparams[] = { &pbuf, &pn };
    launch(fat_add_one, (N + 255) / 256, 256, 0, CU_STREAM_LEGACY, pparams);
    CU_CHECK(cuMemcpyDtoH(&host[3], pbuf + (N - 1) * sizeof(int),
                          sizeof(int)));
    CU_CHECK(cuCtxPopCurrent(&popped));
    MT_CHECK(popped == primary, "iteration %ld: popped %p", it,
             (void *)popped);
    MT_CHECK(host[3] == it + 1, "iteration %ld: primary %d", it, host[3]);

    if (it % 100 == 0) {
      CUdeviceptr p;
      CU_CHECK(mem_alloc(&p, 1 << 20));
      CU_CHECK(cuMemFree(p));
    }
  }

  CU_CHECK(cuMemFreeHost(host));
  CU_CHECK(cuEventDestroy(done));
  CU_CHECK(cuStreamDestroy(stream));
  CU_CHECK(cuModuleUnload(module));
  CU_CHECK(cuCtxDestroy(ctx));
  CU_CHECK(cuDevicePrimaryCtxRelease(dev));
  mt_finish(it);
  return 0;
}
