// NCCL (run with MANA's --cuda; cuda/PLAN-nccl-front.md): communicators that
// each checkpoint drains and each restart makes again.  MPI gives the unique
// ids.  nccl_nompi is this test without MPI (MT_NO_MPI, run with --no-mpi):
// the first unique ids go through a file, the others by ncclBroadcast on
// world.  The communicators: 'world', of every rank; 'reversed', a split of it
// with the ranks in reverse order; 'evens', a split that gives the odd ranks
// none (NCCL_SPLIT_NOCOLOR).  Each iteration: an all-reduce on world, a
// broadcast on reversed from a root that moves, a send/recv ring in a group
// region on world, an all-gather on world into memory of ncclMemAlloc, and
// on the even ranks an all-reduce on evens.  Every 50th iteration, a new
// communicator by a new unique id, for one all-reduce, then destroyed.  The
// stream is synchronized and every result checked each iteration.  NCCL
// needs a GPU per rank on a node: rank r uses device r % (devices).
//
// With NCCL 2.29 or later, what a restart must make again on a
// communicator, too: world is made by ncclCommInitRankScalable with two
// unique ids, and has a registered buffer and two PreMulSum operators, one
// with a host scalar and one with a device scalar, each used every
// iteration; 'shrunk', world without its last rank, by ncclCommShrink; with
// NCCL_SUSPEND_MEM, 'sleeper', a split of world that is suspended for 20
// iterations out of 40.  Within a node, a symmetric window on world: each
// iteration puts with a signal into the next rank's window, which that rank
// waits for and checks at the start of its next iteration, and an
// all-reduce from the window into it.  Every 10th iteration, rank 0 alone
// simulates a group region with an all-reduce, which runs nothing.

#include <cuda_runtime.h>
#include <nccl.h>

#include "mana_test.h"

#ifndef TEST_NAME
#define TEST_NAME "nccl"
#endif

#define N 1024

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
#define STEP6 1
#endif

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

#ifdef MT_NO_MPI
// Rank 0 writes the unique ids to MT_CONTROL/nccl_id (or ./nccl_id), by a
// rename so that no rank reads half of them; the others wait for them.
static void
share_by_file(ncclUniqueId *id, int nids)
{
  const char *dir = getenv("MT_CONTROL");
  char path[4096], tmp[4096];
  snprintf(path, sizeof path, "%s/nccl_id", dir != NULL ? dir : ".");
  if (mt_rank == 0) {
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "w");
    MT_CHECK(f != NULL && fwrite(id, sizeof *id, nids, f) == (size_t)nids,
             "writing %s", tmp);
    fclose(f);
    MT_CHECK(rename(tmp, path) == 0, "renaming %s", tmp);
    return;
  }
  for (int tries = 0;; tries++) {
    FILE *f = fopen(path, "r");
    if (f != NULL) {
      size_t n = fread(id, sizeof *id, nids, f);
      fclose(f);
      if (n == (size_t)nids) {
        return;
      }
    }
    MT_CHECK(tries < 60000, "no %s after 60 s", path);
    usleep(1000);
  }
}
#endif

// A communicator of every rank, by new unique ids from rank 0 (nids > 1:
// ncclCommInitRankScalable): with MPI, by MPI_Bcast; without, the first
// through a file, the others by ncclBroadcast on 'world'.
static ncclComm_t
new_comm(int rank, int size, ncclComm_t world, cudaStream_t s, int nids)
{
  ncclUniqueId id[2];
  MT_CHECK(nids >= 1 && nids <= 2, "%d unique ids", nids);
  if (rank == 0) {
    for (int i = 0; i < nids; i++) {
      NCCL_CHECK(ncclGetUniqueId(&id[i]));
    }
  }
#ifndef MT_NO_MPI
  MT_MPI(MPI_Bcast(id, nids * sizeof *id, MPI_BYTE, 0, MPI_COMM_WORLD));
#else
  if (world == NULL) {
    share_by_file(id, nids);
  } else {
    static void *dev_id;
    if (dev_id == NULL) {
      CUDA_CHECK(cudaMalloc(&dev_id, sizeof id));
    }
    CUDA_CHECK(cudaMemcpyAsync(dev_id, id, nids * sizeof *id,
                               cudaMemcpyHostToDevice, s));
    NCCL_CHECK(ncclBroadcast(dev_id, dev_id, nids * sizeof *id, ncclUint8, 0,
                             world, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaMemcpy(id, dev_id, nids * sizeof *id,
                          cudaMemcpyDeviceToHost));
  }
#endif
  ncclComm_t comm;
  if (nids == 1) {
    NCCL_CHECK(ncclCommInitRank(&comm, size, id[0], rank));
  } else {
#ifdef STEP6
    NCCL_CHECK(ncclCommInitRankScalable(&comm, size, rank, nids, id, NULL));
#endif
  }
  return comm;
}

// The sum over ranks [0, nranks) of value(r, it, i), times mul.
static void
check_sum(unsigned *host, const unsigned *dev, int nranks, unsigned mul,
          long it, const char *what)
{
  to_host(host, dev, N);
  for (int i = 0; i < N; i++) {
    unsigned want = 0;
    for (int r = 0; r < nranks; r++) want += mul * value(r, it, i);
    MT_CHECK(host[i] == want, "iteration %ld: %s [%d] = %u, not %u", it, what,
             i, host[i], want);
  }
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, TEST_NAME);
  int rank = mt_rank, size = mt_size;
  int ndev;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  CUDA_CHECK(cudaSetDevice(rank % ndev));
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

#ifdef STEP6
  ncclComm_t world = new_comm(rank, size, NULL, s, 2);
#else
  ncclComm_t world = new_comm(rank, size, NULL, s, 1);
#endif
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

#ifdef STEP6
  // A registered buffer; PreMulSum operators: times 3, a host scalar, and
  // times 5, in device memory.
  void *reg;
  NCCL_CHECK(ncclCommRegister(world, send, N * sizeof(unsigned), &reg));
  unsigned three = 3, five = 5, *dev_five, *scaled, *shrunksum, *sleepsum;
  CUDA_CHECK(cudaMalloc(&dev_five, sizeof five));
  CUDA_CHECK(cudaMemcpy(dev_five, &five, sizeof five, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&scaled, N * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&shrunksum, N * sizeof(unsigned)));
  CUDA_CHECK(cudaMalloc(&sleepsum, N * sizeof(unsigned)));
  ncclRedOp_t times3, times5;
  NCCL_CHECK(ncclRedOpCreatePreMulSum(&times3, &three, ncclUint32,
                                      ncclScalarHostImmediate, world));
  NCCL_CHECK(ncclRedOpCreatePreMulSum(&times5, dev_five, ncclUint32,
                                      ncclScalarDevice, world));
  ncclComm_t shrunk = NULL;
  int last = size - 1;
  if (rank != last) {
    NCCL_CHECK(ncclCommShrink(world, &last, 1, &shrunk, NULL,
                              NCCL_SHRINK_DEFAULT));
  }
#ifdef NCCL_SUSPEND_MEM
  ncclComm_t sleeper;
  NCCL_CHECK(ncclCommSplit(world, 0, rank, &sleeper, NULL));
  bool asleep = false;
#endif
  // The window: [0, N) for the previous rank's put, [N, 2N) this rank's
  // values, [2N, 3N) their all-reduce.  NCCL makes none across nodes.
  unsigned *wbuf;
  ncclWindow_t win;
  NCCL_CHECK(ncclMemAlloc((void **)&wbuf, 3 * N * sizeof(unsigned)));
  NCCL_CHECK(ncclCommWindowRegister(world, wbuf, 3 * N * sizeof(unsigned),
                                    &win, NCCL_WIN_COLL_SYMMETRIC));
  ncclWaitSignalDesc_t from_prev = { 1, prev, 0, 0 };
#endif

  long it;
  for (it = 0; mt_continue(it); it++) {
#ifdef STEP6
    if (win != NULL && it > 0) {
      // The previous rank's put of the last iteration.
      NCCL_CHECK(ncclWaitSignal(1, &from_prev, world, s));
      CUDA_CHECK(cudaStreamSynchronize(s));
      to_host(host, wbuf, N);
      for (int i = 0; i < N; i++) {
        MT_CHECK(host[i] == value(prev, it - 1, i),
                 "iteration %ld: put [%d] = %u, not %u", it, i, host[i],
                 value(prev, it - 1, i));
      }
    }
#endif
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
#ifdef STEP6
    NCCL_CHECK(ncclAllReduce(send, scaled, N, ncclUint32,
                             it % 2 ? times5 : times3, world, s));
    if (shrunk != NULL) {
      NCCL_CHECK(ncclAllReduce(send, shrunksum, N, ncclUint32, ncclSum,
                               shrunk, s));
    }
#ifdef NCCL_SUSPEND_MEM
    if (asleep && it % 40 == 0) {
      NCCL_CHECK(ncclCommResume(sleeper));
      asleep = false;
    }
    if (!asleep) {
      NCCL_CHECK(ncclAllReduce(send, sleepsum, N, ncclUint32, ncclSum,
                               sleeper, s));
    }
#endif
    if (win != NULL) {
      fill_on(s, wbuf + N, rank, it);
      NCCL_CHECK(ncclAllReduce(wbuf + N, wbuf + 2 * N, N, ncclUint32, ncclSum,
                               world, s));
    }
    if (rank == 0 && it % 10 == 0) {
      // Runs nothing; the other ranks post no such all-reduce.
      ncclSimInfo_t sim = NCCL_SIM_INFO_INITIALIZER;
      NCCL_CHECK(ncclGroupStart());
      NCCL_CHECK(ncclAllReduce(send, sum, N, ncclUint32, ncclSum, world, s));
      NCCL_CHECK(ncclGroupSimulateEnd(&sim));
    }
#endif
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

#ifdef STEP6
    check_sum(host, scaled, size, it % 2 ? 5 : 3, it, "PreMulSum all-reduce");
    if (shrunk != NULL) {
      check_sum(host, shrunksum, size - 1, 1, it, "shrunk all-reduce");
    }
#ifdef NCCL_SUSPEND_MEM
    if (!asleep) {
      check_sum(host, sleepsum, size, 1, it, "sleeper all-reduce");
    }
    if (it % 40 == 20) {
      NCCL_CHECK(ncclCommSuspend(sleeper, NCCL_SUSPEND_MEM));
      asleep = true;
    }
#endif
    if (win != NULL) {
      check_sum(host, wbuf + 2 * N, size, 1, it, "window all-reduce");
      // Into the next rank's [0, N), which it checks at the start of its next
      // iteration: it has checked the last one, since this rank's all-reduces
      // of this iteration are done.
      NCCL_CHECK(ncclPutSignal(wbuf + N, N, ncclUint32, next, win, 0, 0, 0, 0,
                               world, s));
    }
#endif

    if (it % 50 == 0) {
#ifdef STEP6
      // An operator for one all-reduce.
      unsigned seven = 7;
      ncclRedOp_t times7;
      NCCL_CHECK(ncclRedOpCreatePreMulSum(&times7, &seven, ncclUint32,
                                          ncclScalarHostImmediate, world));
      NCCL_CHECK(ncclAllReduce(send, scaled, N, ncclUint32, times7, world, s));
      CUDA_CHECK(cudaStreamSynchronize(s));
      check_sum(host, scaled, size, 7, it, "times-7 all-reduce");
      NCCL_CHECK(ncclRedOpDestroy(times7, world));
#endif
      ncclComm_t once = new_comm(rank, size, world, s, 1);
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

#ifdef STEP6
  if (win != NULL && it > 0) {
    NCCL_CHECK(ncclWaitSignal(1, &from_prev, world, s));  // the last put
    CUDA_CHECK(cudaStreamSynchronize(s));
  }
  if (win != NULL) {
    NCCL_CHECK(ncclCommWindowDeregister(world, win));
  }
  NCCL_CHECK(ncclMemFree(wbuf));
#ifdef NCCL_SUSPEND_MEM
  if (asleep) {
    NCCL_CHECK(ncclCommResume(sleeper));
  }
  NCCL_CHECK(ncclCommDestroy(sleeper));
#endif
  if (shrunk != NULL) {
    NCCL_CHECK(ncclCommDestroy(shrunk));
  }
  NCCL_CHECK(ncclRedOpDestroy(times3, world));
  NCCL_CHECK(ncclRedOpDestroy(times5, world));
  NCCL_CHECK(ncclCommDeregister(world, reg));
#endif
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
