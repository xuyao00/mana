// CUDA-aware MPI: MPI on device buffers (run with MANA's --cuda).  Each
// iteration a kernel updates device state that must survive checkpoint and
// restart; the ranks then exchange device buffers around a ring (blocking
// MPI_Sendrecv, and MPI_Irecv/MPI_Isend that rank 0 posts late, so that the
// others wait on a pending device receive when a checkpoint comes) and
// reduce one in place (MPI_Allreduce).  All checks run on the device data.

#include <cuda_runtime.h>
#include <unistd.h>

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

// Counts the i with buf[i] != mt_value(from, it, i) + state[i] (the state
// of all ranks is the same: one increment per iteration).
__global__ void
check(const int *buf, const int *state, int n, int from, long it, int *bad)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && buf[i] != (int)(from * 1000003L + it * 7919L + i) + state[i]) {
    atomicAdd(bad, 1);
  }
}

static int
count_bad(const int *buf, const int *state, int from, long it, int *d_bad)
{
  int bad;
  CUDA_CHECK(cudaMemset(d_bad, 0, sizeof(int)));
  check<<<(N + 255) / 256, 256>>>(buf, state, N, from, it, d_bad);
  CUDA_CHECK(cudaMemcpy(&bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost));
  return bad;
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "cuda_aware");
  int ndev;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  CUDA_CHECK(cudaSetDevice(mt_rank % ndev));
  int *state, *send, *recv, *recv2, *sum, *d_bad;
  CUDA_CHECK(cudaMalloc(&state, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&send, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&recv, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&recv2, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&sum, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_bad, sizeof(int)));
  CUDA_CHECK(cudaMemset(state, 0, N * sizeof(int)));
  int right = (mt_rank + 1) % mt_size;
  int left = (mt_rank + mt_size - 1) % mt_size;

  long it;
  for (it = 0; mt_continue(it); it++) {
    increment<<<(N + 255) / 256, 256>>>(state, N);
    fill<<<(N + 255) / 256, 256>>>(send, state, N, mt_rank, it);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Status st;
    MT_MPI(MPI_Sendrecv(send, N, MPI_INT, right, 1, recv, N, MPI_INT, left, 1,
                        MPI_COMM_WORLD, &st));
    MT_CHECK(count_bad(recv, state, left, it, d_bad) == 0,
             "iteration %ld: MPI_Sendrecv: wrong data from rank %d", it, left);

    MPI_Request reqs[2];
    MT_MPI(MPI_Irecv(recv2, N, MPI_INT, left, 2, MPI_COMM_WORLD, &reqs[0]));
    if (mt_rank == 0) {
      usleep(2000);  // The others wait on their device receive meanwhile.
    }
    MT_MPI(MPI_Isend(send, N, MPI_INT, right, 2, MPI_COMM_WORLD, &reqs[1]));
    MT_MPI(MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE));
    MT_CHECK(count_bad(recv2, state, left, it, d_bad) == 0,
             "iteration %ld: MPI_Irecv: wrong data from rank %d", it, left);

    // sum = sum over the ranks of send[0] = mt_value(r, it, 0) + state[0]
    CUDA_CHECK(cudaMemcpy(sum, send, sizeof(int), cudaMemcpyDeviceToDevice));
    MT_MPI(MPI_Allreduce(MPI_IN_PLACE, sum, 1, MPI_INT, MPI_SUM,
                         MPI_COMM_WORLD));
    int got, s0;
    CUDA_CHECK(cudaMemcpy(&got, sum, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&s0, state, sizeof(int), cudaMemcpyDeviceToHost));
    MT_CHECK(s0 == it + 1, "iteration %ld: device state %d", it, s0);
    long want = 0;
    for (int r = 0; r < mt_size; r++) {
      want += mt_value(r, it, 0) + s0;
    }
    MT_CHECK(got == (int)want, "iteration %ld: MPI_Allreduce: %d, not %d", it,
             got, (int)want);
  }
  mt_finish(it);
  return 0;
}
