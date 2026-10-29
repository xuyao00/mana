// Managed memory (run with --cuda --no-mpi), which MANA gives as mapped host
// memory.  State in cudaMallocManaged memory is updated by the host and by a
// kernel in turn each iteration, with prefetches and advice in between; a
// temporary managed buffer is set with cudaMemset, read by the host, and
// freed.  The ranks come from the launcher and never talk.

#include <cuda_runtime.h>

#define MT_NO_MPI
#include "mana_test.h"

#define N (1 << 16)

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t e_ = (call);                                                 \
    MT_CHECK(e_ == cudaSuccess, "%s: %s", #call, cudaGetErrorString(e_));    \
  } while (0)

__global__ void
twice(int *a, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] *= 2;
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "cuda_managed");
  int ndev;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  int device = mt_rank % ndev;
  CUDA_CHECK(cudaSetDevice(device));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  int *state;
  CUDA_CHECK(cudaMallocManaged(&state, N * sizeof(int)));
  for (int i = 0; i < N; i++) {
    state[i] = i;
  }
  cudaMemLocation gpu = {cudaMemLocationTypeDevice, device};
  cudaMemLocation cpu = {cudaMemLocationTypeHost, 0};
  CUDA_CHECK(cudaMemAdvise(state, N * sizeof(int),
                           cudaMemAdviseSetPreferredLocation, gpu));

  long it;
  for (it = 0; mt_continue(it); it++) {
    // The host adds 1, the GPU doubles: state[i] = 2 * (old + 1).
    for (int i = 0; i < N; i++) {
      state[i] += 1;
    }
    CUDA_CHECK(cudaMemPrefetchAsync(state, N * sizeof(int), gpu, 0, stream));
    twice<<<(N + 255) / 256, 256, 0, stream>>>(state, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemPrefetchAsync(state, N * sizeof(int), cpu, 0, stream));
    int *tmp;
    CUDA_CHECK(cudaMallocManaged(&tmp, 4096));
    CUDA_CHECK(cudaMemsetAsync(tmp, (int)(it & 0x7f), 4096, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MT_CHECK(((unsigned char *)tmp)[4095] == (unsigned char)(it & 0x7f),
             "iteration %ld: cudaMemset on managed memory", it);
    CUDA_CHECK(cudaFree(tmp));
    // Values stay small: state[i] is kept below 2^20 by a reset every 8.
    for (int i = 0; i < N; i += 997) {
      MT_CHECK(state[i] % 2 == 0, "iteration %ld: state[%d] is %d", it, i,
               state[i]);
    }
    if (it % 8 == 7) {
      for (int i = 0; i < N; i++) {
        long want = i;
        for (int k = 0; k < 8; k++) {
          want = 2 * (want + 1);
        }
        MT_CHECK(state[i] == (int)want, "iteration %ld: state[%d] is %d, "
                 "not %ld", it, i, state[i], want);
        state[i] = i;
      }
    }
  }
  mt_finish(it);
  return 0;
}
