// CUDA without MPI (run with mana_launch --cuda --no-mpi): the CUDA front
// and MANA's core, with the application's CUDA in bin/lower-half-nompi.  Each
// iteration a kernel updates device state that must survive checkpoint and
// restart, and another fills a buffer that goes to pinned host memory on a
// stream of the application's, behind an event.  The ranks come from the
// launcher and never talk.

#include <cuda_runtime.h>

#define MT_NO_MPI
#include "mana_test.h"

#define N 65536

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t e_ = (call);                                                 \
    MT_CHECK(e_ == cudaSuccess, "%s: %s", #call, cudaGetErrorString(e_));    \
  } while (0)

// a[i] += 1: the device state.
__global__ void
increment(int *a, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += 1;
}

// buf[i] = mt_value(rank, it, i) + state[i]
__global__ void
fill(int *buf, const int *state, int n, int rank, long it)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] = (int)(rank * 1000003L + it * 7919L + i) + state[i];
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "cuda_nompi");
  int ndev;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  CUDA_CHECK(cudaSetDevice(mt_rank % ndev));
  int *state, *buf, *host;
  CUDA_CHECK(cudaMalloc(&state, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&buf, N * sizeof(int)));
  CUDA_CHECK(cudaMallocHost(&host, N * sizeof(int)));
  CUDA_CHECK(cudaMemset(state, 0, N * sizeof(int)));
  cudaStream_t stream;
  cudaEvent_t done;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&done));

  long it;
  for (it = 0; mt_continue(it); it++) {
    increment<<<(N + 255) / 256, 256, 0, stream>>>(state, N);
    fill<<<(N + 255) / 256, 256, 0, stream>>>(buf, state, N, mt_rank, it);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(host, buf, N * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(done, stream));
    CUDA_CHECK(cudaEventSynchronize(done));
    for (int i = 0; i < N; i++) {
      // One increment per iteration: state[i] == it + 1.
      MT_CHECK(host[i] == mt_value(mt_rank, it, i) + (int)(it + 1),
               "iteration %ld: host[%d] is %d", it, i, host[i]);
    }
  }
  mt_finish(it);
  return 0;
}
