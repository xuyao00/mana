// NCCL (run with MANA's --cuda; cuda/PLAN-nccl-front.md): communicators that
// each checkpoint drains and each restart makes again.  MPI gives the unique
// ids.  The communicators: 'world', of every rank; 'reversed', a split of it
// with the ranks in reverse order; 'evens', a split that gives the odd ranks
// none (NCCL_SPLIT_NOCOLOR).  Each iteration: an all-reduce on world, a
// broadcast on reversed from a root that moves, a send/recv ring in a group
// region on world, an all-gather on world into memory of ncclMemAlloc, and
// on the even ranks an all-reduce on evens.  Every 50th iteration, a new
// communicator by a new unique id, for one all-reduce, then destroyed.  The
// stream is synchronized and every result checked each iteration.  NCCL
// needs a GPU per rank on a node: rank r uses device r % (devices).

#include <cuda_runtime.h>
#include <nccl.h>

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

// Unsigned, so that sums wrap without overflow.
static unsigned
value(int rank, long it, int i)
{
  return (unsigned)mt_value(rank, it, i);
}

// buf[i] = value(rank, it, i)
__global__ void
fill(unsigned *buf, int n, int rank, long it)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) buf[i] = (unsigned)(rank * 1000003L + it * 7919L + i);
}

static void
fill_on(cudaStream_t s, unsigned *buf, int rank, long it)
{
  fill<<<(N + 255) / 256, 256, 0, s>>>(buf, N, rank, it);
  CUDA_CHECK(cudaGetLastError());
}

static void
to_host(unsigned *host, const unsigned *dev, size_t count)
{
  CUDA_CHECK(cudaMemcpy(host, dev, count * sizeof(unsigned),
                        cudaMemcpyDeviceToHost));
}

static ncclComm_t
new_comm(int rank, int size)
{
  ncclUniqueId id;
  if (rank == 0) {
    NCCL_CHECK(ncclGetUniqueId(&id));
  }
  MT_MPI(MPI_Bcast(&id, sizeof id, MPI_BYTE, 0, MPI_COMM_WORLD));
  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, size, id, rank));
  return comm;
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "nccl");
  int rank = mt_rank, size = mt_size;
  int ndev;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  CUDA_CHECK(cudaSetDevice(rank % ndev));
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  ncclComm_t world = new_comm(rank, size);
  ncclComm_t evens, reversed;
  NCCL_CHECK(ncclCommSplit(world, rank % 2 == 0 ? 0 : NCCL_SPLIT_NOCOLOR,
                           rank, &evens, NULL));
  MT_CHECK((evens == NULL) == (rank % 2 == 1), "evens is %p", (void *)evens);
  NCCL_CHECK(ncclCommSplit(world, 0, size - 1 - rank, &reversed, NULL));
  int rev_rank;
  NCCL_CHECK(ncclCommUserRank(reversed, &rev_rank));
  MT_CHECK(rev_rank == size - 1 - rank, "rank %d in reversed", rev_rank);

  unsigned *send, *sum, *bcast, *ring, *evensum, *gathered;
  CUDA_CHECK(cudaMalloc(&send, N * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&sum, N * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&bcast, N * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&ring, N * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&evensum, N * sizeof(unsigned)));
  NCCL_CHECK(ncclMemAlloc((void **)&gathered, size * N * sizeof(unsigned)));
  unsigned *host = (unsigned *)malloc(size * N * sizeof(unsigned));
  int next = (rank + 1) % size, prev = (rank + size - 1) % size;

  long it;
  for (it = 0; mt_continue(it); it++) {
    fill_on(s, send, rank, it);
    NCCL_CHECK(ncclAllReduce(send, sum, N, ncclUint32, ncclSum, world, s));
    // reversed's rank k is world's size - 1 - k.
    int root = it % size;
    if (rev_rank == root) {
      CUDA_CHECK(cudaMemcpyAsync(bcast, send, N * sizeof(unsigned),
                                 cudaMemcpyDeviceToDevice, s));
    }
    NCCL_CHECK(ncclBroadcast(bcast, bcast, N, ncclUint32, root, reversed, s));
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(send, N, ncclUint32, next, world, s));
    NCCL_CHECK(ncclRecv(ring, N, ncclUint32, prev, world, s));
    NCCL_CHECK(ncclGroupEnd());
    NCCL_CHECK(ncclAllGather(send, gathered, N, ncclUint32, world, s));
    if (evens != NULL) {
      NCCL_CHECK(ncclAllReduce(send, evensum, N, ncclUint32, ncclSum, evens,
                               s));
    }
    CUDA_CHECK(cudaStreamSynchronize(s));

    to_host(host, sum, N);
    for (int i = 0; i < N; i++) {
      unsigned want = 0;
      for (int r = 0; r < size; r++) want += value(r, it, i);
      MT_CHECK(host[i] == want, "iteration %ld: all-reduce [%d] = %u, not %u",
               it, i, host[i], want);
    }
    to_host(host, bcast, N);
    for (int i = 0; i < N; i++) {
      unsigned want = value(size - 1 - root, it, i);
      MT_CHECK(host[i] == want, "iteration %ld: broadcast [%d] = %u, not %u",
               it, i, host[i], want);
    }
    to_host(host, ring, N);
    for (int i = 0; i < N; i++) {
      MT_CHECK(host[i] == value(prev, it, i),
               "iteration %ld: ring [%d] = %u, not %u", it, i, host[i],
               value(prev, it, i));
    }
    to_host(host, gathered, size * N);
    for (int r = 0; r < size; r++) {
      for (int i = 0; i < N; i++) {
        MT_CHECK(host[r * N + i] == value(r, it, i),
                 "iteration %ld: all-gather [%d][%d] = %u, not %u", it, r, i,
                 host[r * N + i], value(r, it, i));
      }
    }
    if (evens != NULL) {
      to_host(host, evensum, N);
      for (int i = 0; i < N; i++) {
        unsigned want = 0;
        for (int r = 0; r < size; r += 2) want += value(r, it, i);
        MT_CHECK(host[i] == want, "iteration %ld: evens [%d] = %u, not %u",
                 it, i, host[i], want);
      }
    }

    if (it % 50 == 0) {
      ncclComm_t once = new_comm(rank, size);
      unsigned mine = rank + 1 + (unsigned)it;
      CUDA_CHECK(cudaMemcpy(sum, &mine, sizeof mine, cudaMemcpyHostToDevice));
      NCCL_CHECK(ncclAllReduce(sum, sum, 1, ncclUint32, ncclSum, once, s));
      CUDA_CHECK(cudaStreamSynchronize(s));
      to_host(host, sum, 1);
      unsigned want = size * (size + 1) / 2 + size * (unsigned)it;
      MT_CHECK(host[0] == want, "iteration %ld: new communicator: %u, not %u",
               it, host[0], want);
      NCCL_CHECK(ncclCommDestroy(once));
    }
  }

  NCCL_CHECK(ncclCommDestroy(reversed));
  if (evens != NULL) {
    NCCL_CHECK(ncclCommDestroy(evens));
  }
  NCCL_CHECK(ncclCommDestroy(world));
  NCCL_CHECK(ncclMemFree(gathered));
  free(host);
  mt_finish(it);
  return 0;
}
