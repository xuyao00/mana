// The application's CUDA IPC (run with MANA's --cuda), between pairs of
// ranks on one node.  Each rank exports a large buffer and a small one (a
// block of the allocator's chunk), sends the handles over MPI, and opens its
// peer's.  As natively, a handle stands for a whole allocation, so the
// handle is the one of the buffer's base (cuMemGetAddressRange), sent with
// the buffer's offset there.  Each iteration a kernel writes the peer's buffers through the
// imported pointers, for long enough that a checkpoint may come while it
// runs, and each rank checks what its peer wrote into its own memory.  A
// handle sent each iteration is opened, read and closed too, also when a
// checkpoint comes between the send and the open.

#include <cuda.h>
#include <cuda_runtime.h>

#include "mana_test.h"

#define N (1 << 20)
#define SMALL 64

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t e_ = (call);                                                 \
    MT_CHECK(e_ == cudaSuccess, "%s: %s", #call, cudaGetErrorString(e_));    \
  } while (0)

__device__ int
value(int rank, long it, int i)
{
  return (int)(rank * 1000003L + it * 7919L + i);
}

__global__ void
write_peer(int *peer, int n, int rank, long it, long spin)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  long start = clock64();
  while (clock64() - start < spin) {
  }
  if (i < n) peer[i] = value(rank, it, i);
}

__global__ void
count_bad(const int *mine, int n, int peer, long it, int *bad)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && mine[i] != value(peer, it, i)) atomicAdd(bad, 1);
}

// What the peer needs to open 'ptr'.
struct Shared {
  cudaIpcMemHandle_t handle;
  size_t offset;
};

static Shared
share(void *ptr)
{
  CUdeviceptr base;
  size_t size;
  MT_CHECK(cuMemGetAddressRange(&base, &size, (CUdeviceptr)ptr) ==
           CUDA_SUCCESS, "%s", "cuMemGetAddressRange");
  Shared s;
  CUDA_CHECK(cudaIpcGetMemHandle(&s.handle, (void *)base));
  s.offset = (CUdeviceptr)ptr - base;
  return s;
}

static int *
open_shared(const Shared &s, void **base)
{
  CUDA_CHECK(cudaIpcOpenMemHandle(base, s.handle,
                                  cudaIpcMemLazyEnablePeerAccess));
  return (int *)((char *)*base + s.offset);
}

static int
bad_values(const int *mine, int n, int peer, long it, int *d_bad)
{
  int bad = 0;
  CUDA_CHECK(cudaMemset(d_bad, 0, sizeof(int)));
  count_bad<<<(n + 255) / 256, 256>>>(mine, n, peer, it, d_bad);
  CUDA_CHECK(cudaMemcpy(&bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost));
  return bad;
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "cuda_ipc");
  MT_CHECK(mt_size % 2 == 0, "%d ranks: pairs are needed", mt_size);
  int peer = mt_rank ^ 1;
  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  CUDA_CHECK(cudaSetDevice(mt_rank % ndev));

  int *big, *small, *fresh, *d_bad;
  CUDA_CHECK(cudaMalloc(&big, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&small, SMALL * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&fresh, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_bad, sizeof(int)));

  Shared mine[2] = { share(big), share(small) }, theirs[2];
  MT_MPI(MPI_Sendrecv(mine, sizeof mine, MPI_BYTE, peer, 0, theirs,
                      sizeof theirs, MPI_BYTE, peer, 0, MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE));
  void *peer_big_base, *peer_small_base;
  int *peer_big = open_shared(theirs[0], &peer_big_base);
  int *peer_small = open_shared(theirs[1], &peer_small_base);

  long it;
  for (it = 0; mt_continue(it); it++) {
    write_peer<<<(N + 255) / 256, 256>>>(peer_big, N, mt_rank, it, 20000);
    write_peer<<<1, SMALL>>>(peer_small, SMALL, mt_rank, it, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    MT_MPI(MPI_Barrier(MPI_COMM_WORLD));
    int bad = bad_values(big, N, peer, it, d_bad);
    MT_CHECK(bad == 0, "iteration %ld: %d bad values in the large buffer", it,
             bad);
    bad = bad_values(small, SMALL, peer, it, d_bad);
    MT_CHECK(bad == 0, "iteration %ld: %d bad values in the small buffer", it,
             bad);

    // A handle that the peer opens, reads and closes.
    int v = mt_rank * 100000 + (int)it;
    CUDA_CHECK(cudaMemcpy(fresh, &v, sizeof v, cudaMemcpyHostToDevice));
    // The copy from pageable memory may still be in flight, and the peer's
    // read is in another process.
    CUDA_CHECK(cudaDeviceSynchronize());
    Shared h = share(fresh), hp;
    MT_MPI(MPI_Sendrecv(&h, sizeof h, MPI_BYTE, peer, 1, &hp, sizeof hp,
                        MPI_BYTE, peer, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    void *pf_base;
    int *pf = open_shared(hp, &pf_base), got = -1;
    CUDA_CHECK(cudaMemcpy(&got, pf, sizeof got, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaIpcCloseMemHandle(pf_base));
    MT_CHECK(got == peer * 100000 + (int)it, "iteration %ld: fresh %d", it,
             got);
    // The peer must not overwrite this rank's memory before it checked.
    MT_MPI(MPI_Barrier(MPI_COMM_WORLD));
  }
  CUDA_CHECK(cudaIpcCloseMemHandle(peer_big_base));
  if (peer_small_base != peer_big_base) {
    CUDA_CHECK(cudaIpcCloseMemHandle(peer_small_base));
  }
  mt_finish(it);
  return 0;
}
