// NCCL and MPI calls interleaved (run with MANA's --cuda;
// cuda/PLAN-nccl-front.md), in the pattern that argv[1] names:
//   spmd:  every rank calls in the same order: an NCCL all-reduce, an MPI
//          all-reduce, an MPI broadcast from a root that moves, an MPI ring
//          by MPI_Sendrecv, and a second NCCL all-reduce;
//   order: the ranks call in different orders: even ranks call MPI_Barrier
//          between the two NCCL all-reduces, odd ranks after both;
//   send:  rank 0 posts both NCCL all-reduces, then sends each other rank
//          the iteration by MPI_Send, which they receive before posting
//          theirs.
// NCCL calls only queue work, so each order runs natively; the stream is
// synchronized, and every result checked, at the end of each iteration.  A
// rank sleeps 1 ms between its two NCCL all-reduces (in order, the odd
// ranks; in send, rank 0), where a checkpoint most often finds it.  NCCL
// needs a GPU per rank on a node: rank r uses device r % (devices).

#include <cuda_runtime.h>
#include <nccl.h>
#include <string.h>

#include "mana_test.h"

#define N 1024

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

// buf[i] = mt_value(rank, it, i)
__global__ void
fill(unsigned *buf, int n, int rank, long it)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] = (unsigned)(rank * 1000003L + it * 7919L + i);
}

// The sum over the ranks of mt_value(r, it, i), in dev.
static void
check_sum(unsigned *host, const unsigned *dev, int size, long it,
          const char *what)
{
  CUDA_CHECK(cudaMemcpy(host, dev, N * sizeof(unsigned),
                        cudaMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
    unsigned want = 0;
    for (int r = 0; r < size; r++) want += (unsigned)mt_value(r, it, i);
    MT_CHECK(host[i] == want, "iteration %ld: %s [%d] = %u, not %u", it, what,
             i, host[i], want);
  }
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "nccl_mpi");
  int rank = mt_rank, size = mt_size;
  MT_CHECK(argc == 2 && (!strcmp(argv[1], "spmd") ||
                         !strcmp(argv[1], "order") ||
                         !strcmp(argv[1], "send")),
           "usage: nccl_mpi spmd|order|send");
  const char *mode = argv[1];
  int ndev;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  CUDA_CHECK(cudaSetDevice(rank % ndev));
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  ncclUniqueId id;
  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&id));
  }
  MT_MPI(MPI_Bcast(&id, sizeof id, MPI_BYTE, 0, MPI_COMM_WORLD));
  ncclComm_t world;
  NCCL_CHECK(ncclCommInitRank(&world, size, id, rank));

  unsigned *send, *sum1, *sum2;
  CUDA_CHECK(cudaMalloc(&send, N * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&sum1, N * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&sum2, N * sizeof(unsigned)));
  // NCCL connects the ranks at a communicator's first all-reduce, which then
  // waits for every rank; later ones only queue work.
  NCCL_CHECK(ncclAllReduce(send, sum1, N, ncclUint32, ncclSum, world, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  unsigned *host = (unsigned *)malloc(N * sizeof(unsigned));
  int next = (rank + 1) % size, prev = (rank + size - 1) % size;

  long it;
  for (it = 0; mt_continue(it); it++) {
    fill<<<(N + 255) / 256, 256, 0, s>>>(send, N, rank, it);
    CUDA_CHECK(cudaGetLastError());
    if (!strcmp(mode, "spmd")) {
      NCCL_CHECK(ncclAllReduce(send, sum1, N, ncclUint32, ncclSum, world, s));
      long mine = it + rank, total;
      MT_MPI(MPI_Allreduce(&mine, &total, 1, MPI_LONG, MPI_SUM,
                           MPI_COMM_WORLD));
      MT_CHECK(total == size * it + (long)size * (size - 1) / 2,
               "iteration %ld: MPI all-reduce %ld", it, total);
      long root_value = rank == it % size ? it : -1;
      MT_MPI(MPI_Bcast(&root_value, 1, MPI_LONG, it % size, MPI_COMM_WORLD));
      MT_CHECK(root_value == it, "iteration %ld: MPI broadcast %ld", it,
               root_value);
      long from_prev = -1;
      MT_MPI(MPI_Sendrecv(&mine, 1, MPI_LONG, next, 0, &from_prev, 1,
                          MPI_LONG, prev, 0, MPI_COMM_WORLD,
                          MPI_STATUS_IGNORE));
      MT_CHECK(from_prev == it + prev, "iteration %ld: MPI ring %ld", it,
               from_prev);
      NCCL_CHECK(ncclAllReduce(send, sum2, N, ncclUint32, ncclSum, world, s));
    } else if (!strcmp(mode, "order")) {
      NCCL_CHECK(ncclAllReduce(send, sum1, N, ncclUint32, ncclSum, world, s));
      if (rank % 2 == 0) {
        MT_MPI(MPI_Barrier(MPI_COMM_WORLD));
      } else {
        usleep(1000);
      }
      NCCL_CHECK(ncclAllReduce(send, sum2, N, ncclUint32, ncclSum, world, s));
      if (rank % 2 == 1) {
        MT_MPI(MPI_Barrier(MPI_COMM_WORLD));
      }
    } else {  // send
      long got = it;
      if (rank != 0) {
        MT_MPI(MPI_Recv(&got, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE));
        MT_CHECK(got == it, "iteration %ld: got %ld from rank 0", it, got);
      }
      NCCL_CHECK(ncclAllReduce(send, sum1, N, ncclUint32, ncclSum, world, s));
      if (rank == 0) {
        usleep(1000);
      }
      NCCL_CHECK(ncclAllReduce(send, sum2, N, ncclUint32, ncclSum, world, s));
      if (rank == 0) {
        for (int r = 1; r < size; r++) {
          MT_MPI(MPI_Send(&got, 1, MPI_LONG, r, 0, MPI_COMM_WORLD));
        }
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
    check_sum(host, sum1, size, it, "first all-reduce");
    check_sum(host, sum2, size, it, "second all-reduce");
  }

  NCCL_CHECK(ncclCommDestroy(world));
  free(host);
  mt_finish(it);
  return 0;
}
