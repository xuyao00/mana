// The orderings of cudaMalloc memory that a CUDA-aware MPI program relies on
// (run with MANA's --cuda, whose VMM allocator must keep them):
//   - CU_POINTER_ATTRIBUTE_SYNC_MEMOPS: with it, a cudaMemcpy between device
//     buffers and a cudaMemset return only when they are done, so MPI sends
//     what they wrote.  A CUDA-aware MPI sets it on the buffers it registers
//     (libfabric does; Cray's GTL, within a node, does not), so the test sets
//     it on its send buffers itself: the guarantee then holds on every path.
//     Each iteration queues the copy and the set behind a busy kernel and
//     sends at once, without synchronizing.
//   - cudaFree waits for the device's work: a busy kernel still writes a
//     buffer when it is freed, and the next allocation, which may reuse the
//     memory, must not see that write.
// All buffers are allocated with cudaMalloc.

#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

#include "mana_test.h"

#define N (1 << 18)            // ints: 1 MB messages, sent with RDMA
#define N_FREED (1 << 16)      // ints: a small allocation, reused at once

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t e_ = (call);                                                 \
    MT_CHECK(e_ == cudaSuccess, "%s: %s", #call, cudaGetErrorString(e_));    \
  } while (0)

// Spins for 'cycles', then sets a[i] = v + i: still running when the host
// goes on.
__global__ void
busy_fill(int *a, int n, int v, long long cycles)
{
  long long t = clock64();
  while (clock64() - t < cycles) {
  }
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = v + i;
}

#define CU_CHECK(call)                                                       \
  do {                                                                       \
    CUresult r_ = (call);                                                    \
    MT_CHECK(r_ == CUDA_SUCCESS, "%s: %d", #call, (int)r_);                  \
  } while (0)

static int *h;                 // host copy for the checks

// Checks every 997th int of device buffer 'd' (n ints): want(i).
#define CHECK_DATA(d, n, want, ...)                                          \
  do {                                                                       \
    CUDA_CHECK(cudaMemcpy(h, (d), (n) * sizeof(int), cudaMemcpyDeviceToHost)); \
    for (int i_ = 0; i_ < (n); i_ += 997) {                                  \
      MT_CHECK(h[i_] == (want), __VA_ARGS__);                                \
    }                                                                        \
  } while (0)

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "cuda_ordering");
  int ndev, khz;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  CUDA_CHECK(cudaSetDevice(mt_rank % ndev));
  CUDA_CHECK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, mt_rank % ndev));
  long long cycles = khz;      // 1 ms
  int *state, *send, *recv, *send2, *recv2;
  CUDA_CHECK(cudaMalloc(&state, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&send, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&recv, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&send2, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&recv2, N * sizeof(int)));
  h = (int *)malloc(N * sizeof(int));
  cudaStream_t other;          // does not wait for the legacy stream
  CUDA_CHECK(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking));
  int right = (mt_rank + 1) % mt_size;
  int left = (mt_rank + mt_size - 1) % mt_size;

  unsigned int one = 1;
  CU_CHECK(cuPointerSetAttribute(&one, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                 (CUdeviceptr)send));
  CU_CHECK(cuPointerSetAttribute(&one, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                 (CUdeviceptr)send2));
  // First use: the MPI registers the buffers.
  CUDA_CHECK(cudaDeviceSynchronize());
  MT_MPI(MPI_Sendrecv(send, N, MPI_INT, right, 0, recv, N, MPI_INT, left, 0,
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE));
  MT_MPI(MPI_Sendrecv(send2, N, MPI_INT, right, 1, recv2, N, MPI_INT, left, 1,
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE));

  long it;
  for (it = 0; mt_continue(it); it++) {
    // cudaMemcpy behind a busy kernel, then MPI right away.
    int v = mt_value(mt_rank, it, 0);
    busy_fill<<<N / 256, 256>>>(state, N, v, cycles);
    CUDA_CHECK(cudaMemcpy(send, state, N * sizeof(int),
                          cudaMemcpyDeviceToDevice));
    MT_MPI(MPI_Sendrecv(send, N, MPI_INT, right, 0, recv, N, MPI_INT, left, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    int from = mt_value(left, it, 0);
    CHECK_DATA(recv, N, from + i_,
               "iteration %ld: cudaMemcpy then MPI_Sendrecv: [%d] = %d, not %d",
               it, i_, h[i_], from + i_);

    // cudaMemset behind a busy kernel, then MPI right away.
    int byte = (int)(it % 251) + 1;
    busy_fill<<<N / 256, 256>>>(state, N, 0, cycles);
    CUDA_CHECK(cudaMemset(send2, byte, N * sizeof(int)));
    MT_MPI(MPI_Sendrecv(send2, N, MPI_INT, right, 1, recv2, N, MPI_INT, left, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    int want = byte * 0x01010101;
    CHECK_DATA(recv2, N, want,
               "iteration %ld: cudaMemset then MPI_Sendrecv: [%d] = %#x, not %#x",
               it, i_, h[i_], want);

    // cudaFree while a busy kernel writes the buffer; the next allocation
    // is cleared on another stream and must stay clear.
    int *freed, *fresh;
    CUDA_CHECK(cudaMalloc(&freed, N_FREED * sizeof(int)));
    busy_fill<<<N_FREED / 256, 256>>>(freed, N_FREED, 0x7eadbeef, cycles);
    CUDA_CHECK(cudaFree(freed));
    CUDA_CHECK(cudaMalloc(&fresh, N_FREED * sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(fresh, 0, N_FREED * sizeof(int), other));
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_DATA(fresh, N_FREED, 0,
               "iteration %ld: cudaFree did not wait for the kernel: "
               "[%d] = %#x", it, i_, h[i_]);
    CUDA_CHECK(cudaFree(fresh));
  }
  mt_finish(it);
  return 0;
}
