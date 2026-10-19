// CUDA-aware MPI on buffers that come and go (run with MANA's --cuda): each
// iteration allocates a buffer, sends it around a ring, checks what it
// received and frees it, so that later allocations reuse the addresses.  An
// MPI library that caches what it knows about a buffer (UCX's memory-type
// and registration caches; see MANA_UCX_MEM_EVENTS in cuda-forward/README.md)
// must forget it when the buffer is freed.  The sizes cover small and large
// allocations (MANA's VMM allocator puts those under 1 MB in shared 2 MB
// chunks) and, every third round, large host buffers (mmap).

#include <cuda_runtime.h>
#include <stdlib.h>

#include "mana_test.h"

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t e_ = (call);                                                 \
    MT_CHECK(e_ == cudaSuccess, "%s: %s", #call, cudaGetErrorString(e_));    \
  } while (0)

static const size_t sizes[] = { 16, 1024, 65536, 300000, 1 << 20, 3 << 20,
                                16 << 20 };   // bytes
#define N_SIZES (sizeof sizes / sizeof sizes[0])

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "cuda_churn");
  int ndev;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  CUDA_CHECK(cudaSetDevice(mt_rank % ndev));
  int right = (mt_rank + 1) % mt_size;
  int left = (mt_rank + mt_size - 1) % mt_size;
  int *host = (int *)malloc(16 << 20);
  MT_CHECK(host != NULL, "malloc");

  long it;
  for (it = 0; mt_continue(it); it++) {
    size_t n = sizes[it % N_SIZES] / sizeof(int);
    int on_host = (it / N_SIZES) % 3 == 2 && n >= 16384;
    int *send, *recv;
    if (on_host) {
      send = (int *)malloc(n * sizeof(int));
      recv = (int *)malloc(n * sizeof(int));
      MT_CHECK(send != NULL && recv != NULL, "malloc");
    } else {
      CUDA_CHECK(cudaMalloc(&send, n * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&recv, n * sizeof(int)));
    }
    for (size_t i = 0; i < n; i++) {
      host[i] = mt_value(mt_rank, it, i);
    }
    if (on_host) {
      memcpy(send, host, n * sizeof(int));
      memset(recv, 0, n * sizeof(int));
    } else {
      CUDA_CHECK(cudaMemcpy(send, host, n * sizeof(int),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemset(recv, 0, n * sizeof(int)));
    }
    MT_MPI(MPI_Sendrecv(send, n, MPI_INT, right, 1, recv, n, MPI_INT, left, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    if (on_host) {
      memcpy(host, recv, n * sizeof(int));
    } else {
      CUDA_CHECK(cudaMemcpy(host, recv, n * sizeof(int),
                            cudaMemcpyDeviceToHost));
    }
    for (size_t i = 0; i < n; i++) {
      MT_CHECK(host[i] == mt_value(left, it, i),
               "iteration %ld (%zu bytes, %s): element %zu from rank %d: "
               "%d, not %d", it, n * sizeof(int), on_host ? "host" : "device",
               i, left, host[i], mt_value(left, it, i));
    }
    if (on_host) {
      free(send);
      free(recv);
    } else {
      CUDA_CHECK(cudaFree(send));
      CUDA_CHECK(cudaFree(recv));
    }
  }
  mt_finish(it);
  return 0;
}
