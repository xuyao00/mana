// NCCL with NCCL in MANA's lower half (run with MANA's --cuda; see
// ../nccl/README).  Each iteration a kernel updates device state that must
// survive checkpoint and restart; then NCCL reduces it (ncclAllReduce), moves
// it point to point in a group (ncclSend/ncclRecv), and CUDA-aware MPI reduces
// it too (MPI_Allreduce on device memory).  All checks run on device data.
//
// NCCL accepts one rank per GPU in a communicator.  With a GPU per rank, the
// communicator spans all ranks (rank 0's ncclUniqueId, sent with MPI_Bcast);
// otherwise each rank has a communicator of its own, and the send/receive go
// to itself.

#include <cuda_runtime.h>
#include <nccl.h>

#include "mana_test.h"

#define N 65536

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t e_ = (call);                                                 \
    MT_CHECK(e_ == cudaSuccess, "%s: %s", #call, cudaGetErrorString(e_));    \
  } while (0)

#define NCCL_CHECK(call)                                                     \
  do {                                                                       \
    ncclResult_t r_ = (call);                                                \
    MT_CHECK(r_ == ncclSuccess, "%s: %s", #call, ncclGetErrorString(r_));    \
  } while (0)

// a[i] += 1: the device state.
__global__ void
increment(int *a, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += 1;
}

// buf[i] = rank * 7 + i + state[i]
__global__ void
fill(int *buf, const int *state, int n, int rank)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] = rank * 7 + i + state[i];
}

// Counts the i with buf[i] != sum over 'ranks' ranks from 'first' of
// (r * 7 + i + state[i]) (the state of all ranks is the same).
__global__ void
check_sum(const int *buf, const int *state, int n, int first, int ranks,
          int *bad)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int want = 0;
    for (int r = first; r < first + ranks; r++) {
      want += r * 7 + i + state[i];
    }
    if (buf[i] != want) atomicAdd(bad, 1);
  }
}

static int
count_bad(const int *buf, const int *state, int first, int ranks, int *d_bad,
          cudaStream_t stream)
{
  int bad;
  CUDA_CHECK(cudaMemsetAsync(d_bad, 0, sizeof(int), stream));
  check_sum<<<(N + 255) / 256, 256, 0, stream>>>(buf, state, N, first, ranks,
                                                 d_bad);
  CUDA_CHECK(cudaMemcpyAsync(&bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return bad;
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "nccl_lower_half");
  int ndev;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  CUDA_CHECK(cudaSetDevice(mt_rank % ndev));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // A communicator of all ranks if each has a GPU, else of this rank alone.
  int shared = ndev >= mt_size;
  int nranks = shared ? mt_size : 1;
  int rank = shared ? mt_rank : 0;
  int first = shared ? 0 : mt_rank;   // job rank of the communicator's rank 0
  ncclUniqueId id;
  if (!shared || mt_rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&id));
  }
  if (shared) {
    MT_MPI(MPI_Bcast(&id, sizeof id, MPI_BYTE, 0, MPI_COMM_WORLD));
  }
  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));
  int count, me;
  NCCL_CHECK(ncclCommCount(comm, &count));
  NCCL_CHECK(ncclCommUserRank(comm, &me));
  MT_CHECK(count == nranks && me == rank, "communicator: rank %d of %d", me,
           count);

  int *state, *send, *sum, *recv, *mpisum, *d_bad;
  CUDA_CHECK(cudaMalloc(&state, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&send, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&sum, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&recv, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&mpisum, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_bad, sizeof(int)));
  CUDA_CHECK(cudaMemset(state, 0, N * sizeof(int)));
  int right = (rank + 1) % nranks;
  int left = (rank + nranks - 1) % nranks;

  long it;
  for (it = 0; mt_continue(it); it++) {
    increment<<<(N + 255) / 256, 256, 0, stream>>>(state, N);
    fill<<<(N + 255) / 256, 256, 0, stream>>>(send, state, N, mt_rank);
    CUDA_CHECK(cudaGetLastError());

    // NCCL collective.
    NCCL_CHECK(ncclAllReduce(send, sum, N, ncclInt32, ncclSum, comm, stream));
    MT_CHECK(count_bad(sum, state, first, nranks, d_bad, stream) == 0,
             "iteration %ld: ncclAllReduce: wrong sum", it);

    // NCCL point to point, around the ring of the communicator.
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(send, N, ncclInt32, right, comm, stream));
    NCCL_CHECK(ncclRecv(recv, N, ncclInt32, left, comm, stream));
    NCCL_CHECK(ncclGroupEnd());
    MT_CHECK(count_bad(recv, state, first + left, 1, d_bad, stream) == 0,
             "iteration %ld: ncclRecv: wrong data from rank %d", it, left);

    // CUDA-aware MPI on the same device memory.
    MT_MPI(MPI_Allreduce(send, mpisum, N, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
    MT_CHECK(count_bad(mpisum, state, 0, mt_size, d_bad, stream) == 0,
             "iteration %ld: MPI_Allreduce: wrong sum", it);

    int s0;
    CUDA_CHECK(cudaMemcpy(&s0, state, sizeof(int), cudaMemcpyDeviceToHost));
    MT_CHECK(s0 == it + 1, "iteration %ld: device state %d", it, s0);
  }
  NCCL_CHECK(ncclCommDestroy(comm));
  mt_finish(it);
  return 0;
}
