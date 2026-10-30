// cudaDeviceReset in the middle of a run and at its end, after MPI_Finalize,
// as OSU does (run with MANA's --cuda).  Every RESET iterations a rank resets
// its device and makes its buffers, pinned memory, stream and event again;
// a checkpoint may come between resets, and the restart must then make only
// the objects made since the last one.  The state that resets, checkpoints
// and restarts must keep is on the host, and copied to the device after each
// reset.  MPI works on host values: natively too, a reset tears down the
// MPI's own CUDA state on the device.

#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>

#include "mana_test.h"

#define N 4096
#define RESET 200

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t e_ = (call);                                                 \
    MT_CHECK(e_ == cudaSuccess, "%s: %s", #call, cudaGetErrorString(e_));    \
  } while (0)

// a[i] += v
__global__ void
add(int *a, int n, int v)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += v;
}

struct Objects {
  int *dev;
  int *pinned;
  cudaStream_t stream;
  cudaEvent_t done;
};

// The objects of a device after a reset (or at the start), with dev[i] =
// state + i.
static void
make_objects(Objects *o, long state)
{
  CUDA_CHECK(cudaMalloc(&o->dev, N * sizeof(int)));
  CUDA_CHECK(cudaMallocHost(&o->pinned, N * sizeof(int)));
  CUDA_CHECK(cudaStreamCreate(&o->stream));
  CUDA_CHECK(cudaEventCreate(&o->done));
  for (int i = 0; i < N; i++) {
    o->pinned[i] = (int)(state + i);
  }
  CUDA_CHECK(cudaMemcpyAsync(o->dev, o->pinned, N * sizeof(int),
                             cudaMemcpyHostToDevice, o->stream));
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "cuda_reset");
  int ndev;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  CUDA_CHECK(cudaSetDevice(mt_rank % ndev));
  long state = 0, resets = 0;
  Objects o;
  make_objects(&o, state);

  long it;
  for (it = 0; mt_continue(it); it++) {
    if (it > 0 && it % RESET == 0) {
      CUDA_CHECK(cudaDeviceReset());
      resets++;
      make_objects(&o, state);
    }
    add<<<(N + 255) / 256, 256, 0, o.stream>>>(o.dev, N, 1);
    CUDA_CHECK(cudaGetLastError());
    state++;
    CUDA_CHECK(cudaMemcpyAsync(o.pinned, o.dev, N * sizeof(int),
                               cudaMemcpyDeviceToHost, o.stream));
    CUDA_CHECK(cudaEventRecord(o.done, o.stream));
    CUDA_CHECK(cudaEventSynchronize(o.done));
    for (int i : { 0, N / 2, N - 1 }) {
      MT_CHECK(o.pinned[i] == (int)(state + i),
               "iteration %ld (%ld resets): element %d is %d, not %d", it,
               resets, i, o.pinned[i], (int)(state + i));
    }
    long mine[2] = { state, resets }, sum[2];
    MT_MPI(MPI_Allreduce(mine, sum, 2, MPI_LONG, MPI_SUM, MPI_COMM_WORLD));
    MT_CHECK(sum[0] == state * mt_size && sum[1] == resets * mt_size,
             "iteration %ld: the ranks disagree (state %ld, resets %ld)", it,
             sum[0], sum[1]);
    usleep(1000);
  }
  mt_finish(it);
  // As OSU: after MPI_Finalize, so no MPI_Abort on failure.
  cudaError_t rc = cudaDeviceReset();
  if (rc != cudaSuccess) {
    fprintf(stderr, "cuda_reset: rank %d: cudaDeviceReset at exit: %s\n",
            mt_rank, cudaGetErrorString(rc));
    return 1;
  }
  return 0;
}
