// Checkpoint cost with GPU memory: each rank holds MB megabytes of device
// memory (filled with a pattern), and loops: kernel update, MPI_Allreduce of
// a checksum on device memory.  After restart the pattern is checked.
//   ckpt_mem MB
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__global__ void fill(unsigned *a, size_t n, unsigned v) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) a[i] = (unsigned)i * 2654435761u + v;
}
__global__ void check(const unsigned *a, size_t n, unsigned v, int *bad) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n && a[i] != (unsigned)i * 2654435761u + v) atomicAdd(bad, 1);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
  size_t mb = atol(argv[1]), n = mb << 18;   // unsigned words
  cudaSetDevice(0);
  unsigned *a; int *bad, *sum;
  cudaMalloc(&a, n * sizeof(unsigned)); cudaMalloc(&bad, sizeof(int)); cudaMalloc(&sum, sizeof(int));
  unsigned blocks = (n + 255) / 256;
  for (long it = 1; ; it++) {
    fill<<<blocks, 256>>>(a, n, (unsigned)it);
    cudaMemset(bad, 0, sizeof(int));
    check<<<blocks, 256>>>(a, n, (unsigned)it, bad);
    MPI_Allreduce(bad, sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    int s; cudaMemcpy(&s, sum, sizeof(int), cudaMemcpyDeviceToHost);
    if (s != 0) { printf("FAIL at %ld\n", it); fflush(stdout); MPI_Abort(MPI_COMM_WORLD, 1); }
    if (rank == 0 && it % 20 == 0) { printf("progress %ld\n", it); fflush(stdout); }
    usleep(50000);
  }
}
