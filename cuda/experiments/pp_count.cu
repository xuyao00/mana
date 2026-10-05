// Ping-pong of N messages of SIZE bytes on device buffers (for counting the
// forwarded CUDA calls per message).  pp_count SIZE N
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  size_t n = atol(argv[1]); int iters = atoi(argv[2]);
  cudaSetDevice(0); char *b; cudaMalloc(&b, n);
  for (int i = 0; i < iters; i++) {
    if (rank == 0) { MPI_Send(b, n, MPI_BYTE, 1, 0, MPI_COMM_WORLD); MPI_Recv(b, n, MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); }
    else { MPI_Recv(b, n, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); MPI_Send(b, n, MPI_BYTE, 0, 0, MPI_COMM_WORLD); }
  }
  MPI_Finalize();
  return 0;
}
