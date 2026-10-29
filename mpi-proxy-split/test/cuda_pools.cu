// Memory pools (run with --cuda --no-mpi).  A pool of cudaMemPoolCreate is
// the device's current pool, with a release threshold that a restart must
// keep; device state lives in an allocation from it and in one from the
// default pool; each iteration also allocates from the current pool with
// cudaMallocAsync and frees in stream order, and makes, uses and destroys a
// second pool.  The ranks come from the launcher and never talk.

#include <cuda_runtime.h>
#include <stdint.h>

#define MT_NO_MPI
#include "mana_test.h"

#define N (1 << 18)

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t e_ = (call);                                                 \
    MT_CHECK(e_ == cudaSuccess, "%s: %s", #call, cudaGetErrorString(e_));    \
  } while (0)

__global__ void
add(int *a, int n, int v)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += v;
}

static cudaMemPool_t
make_pool(int device)
{
  cudaMemPoolProps props = {};
  props.allocType = cudaMemAllocationTypePinned;
  props.location.type = cudaMemLocationTypeDevice;
  props.location.id = device;
  cudaMemPool_t pool;
  CUDA_CHECK(cudaMemPoolCreate(&pool, &props));
  return pool;
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "cuda_pools");
  int ndev;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  int device = mt_rank % ndev;
  CUDA_CHECK(cudaSetDevice(device));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cudaMemPool_t pool = make_pool(device);
  uint64_t threshold = 64 << 20;
  CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                     &threshold));
  CUDA_CHECK(cudaDeviceSetMemPool(device, pool));
  cudaMemPool_t default_pool;
  CUDA_CHECK(cudaDeviceGetDefaultMemPool(&default_pool, device));
  MT_CHECK(default_pool != pool, "%s", "the created pool is the default");

  int *kept, *in_default;
  CUDA_CHECK(cudaMallocFromPoolAsync((void **)&kept, N * sizeof(int), pool,
                                     stream));
  CUDA_CHECK(cudaMallocFromPoolAsync((void **)&in_default, N * sizeof(int),
                                     default_pool, stream));
  CUDA_CHECK(cudaMemsetAsync(kept, 0, N * sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(in_default, 0, N * sizeof(int), stream));
  int *host;
  CUDA_CHECK(cudaMallocHost(&host, N * sizeof(int)));

  long it;
  for (it = 0; mt_continue(it); it++) {
    cudaMemPool_t current;
    CUDA_CHECK(cudaDeviceGetMemPool(&current, device));
    MT_CHECK(current == pool, "iteration %ld: another current pool", it);
    uint64_t got = 0;
    CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                       &got));
    MT_CHECK(got == threshold, "iteration %ld: release threshold %llu", it,
             (unsigned long long)got);
    cudaMemLocation here = {cudaMemLocationTypeDevice, device};
    cudaMemAccessFlags access;
    CUDA_CHECK(cudaMemPoolGetAccess(&access, pool, &here));
    MT_CHECK(access == cudaMemAccessFlagsProtReadWrite,
             "iteration %ld: the device's access to its pool is %d", it,
             (int)access);

    add<<<(N + 255) / 256, 256, 0, stream>>>(kept, N, 1);
    add<<<(N + 255) / 256, 256, 0, stream>>>(in_default, N, 2);
    // From the current pool, freed in stream order.
    int *tmp;
    CUDA_CHECK(cudaMallocAsync((void **)&tmp, N * sizeof(int), stream));
    CUDA_CHECK(cudaMemcpyAsync(tmp, kept, N * sizeof(int),
                               cudaMemcpyDeviceToDevice, stream));
    add<<<(N + 255) / 256, 256, 0, stream>>>(tmp, N, 1);
    CUDA_CHECK(cudaMemcpyAsync(host, tmp, N * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaFreeAsync(tmp, stream));
    // A second pool, made and destroyed in the iteration.
    cudaMemPool_t other = make_pool(device);
    int *in_other;
    CUDA_CHECK(cudaMallocFromPoolAsync((void **)&in_other, 4096, other,
                                       stream));
    CUDA_CHECK(cudaMemsetAsync(in_other, 1, 4096, stream));
    CUDA_CHECK(cudaFreeAsync(in_other, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemPoolDestroy(other));
    CUDA_CHECK(cudaGetLastError());
    for (int i = 0; i < N; i++) {
      MT_CHECK(host[i] == (int)it + 2, "iteration %ld: tmp[%d] is %d", it, i,
               host[i]);
    }
    if (it % 10 == 0) {
      CUDA_CHECK(cudaMemcpyAsync(host, in_default, N * sizeof(int),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      for (int i = 0; i < N; i++) {
        MT_CHECK(host[i] == 2 * ((int)it + 1),
                 "iteration %ld: in_default[%d] is %d", it, i, host[i]);
      }
    }
  }
  mt_finish(it);
  return 0;
}
