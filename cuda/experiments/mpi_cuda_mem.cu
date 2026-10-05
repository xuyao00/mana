// MPI + CUDA checkpoint test for other CUDA memory kinds:
//   pinned host memory (cudaHostAlloc) and managed memory (cudaMallocManaged).
//   mpi_cuda_mem ITERS [SLEEP_US] [both|pinned|managed|mapped]
//   (pinned = cudaHostAllocDefault, mapped = cudaHostAllocMapped)
// Each iteration a kernel increments both buffers (through the device), the
// host verifies them directly, and MPI_Allreduce runs on the pinned and on
// the managed buffer (both are host-accessible, so no staging is needed).
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#define N 4096
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "rank %d: %s:%d %s: %s\n", rank, __FILE__, __LINE__, #x, \
          cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

__global__ void incr(float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += 1.0f;
}

int rank, size;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int iters = argc > 1 ? atoi(argv[1]) : 1000;
  int sleep_us = argc > 2 ? atoi(argv[2]) : 10000;
  const char *mode = argc > 3 ? argv[3] : "both";
  int use_managed = !strcmp(mode, "both") || !strcmp(mode, "managed");
  int use_pinned = strcmp(mode, "managed") != 0;
  unsigned pin_flags = !strcmp(mode, "pinned") ? cudaHostAllocDefault : cudaHostAllocMapped;
  int ndev; CK(cudaGetDeviceCount(&ndev));
  CK(cudaSetDevice(rank % ndev));

  float *pinned, *managed, *red;
  float *dev;
  CK(cudaMalloc(&dev, N * sizeof(float)));
  if (use_pinned) CK(cudaHostAlloc(&pinned, N * sizeof(float), pin_flags));
  else pinned = (float*)malloc(N * sizeof(float));
  if (use_managed) CK(cudaMallocManaged(&managed, N * sizeof(float)));
  else managed = (float*)malloc(N * sizeof(float));
  CK(cudaHostAlloc(&red, 2 * sizeof(float), cudaHostAllocDefault));
  for (int i = 0; i < N; i++) { pinned[i] = 0; managed[i] = 0; }


  for (int it = 1; it <= iters; it++) {
    // Through the device: copy in, increment, copy out (works for any kind).
    CK(cudaMemcpy(dev, pinned, N * sizeof(float), cudaMemcpyDefault));
    incr<<<(N + 255) / 256, 256>>>(dev, N);
    CK(cudaMemcpy(pinned, dev, N * sizeof(float), cudaMemcpyDefault));
    if (use_managed) {
      incr<<<(N + 255) / 256, 256>>>(managed, N);
    } else {
      for (int i = 0; i < N; i++) managed[i] += 1.0f;
    }
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    for (int i = 0; i < N; i += 512) {
      if (pinned[i] != (float)it || managed[i] != (float)it) {
        fprintf(stderr, "rank %d: FAIL: iteration %d: pinned %f managed %f\n",
                rank, it, pinned[i], managed[i]);
        MPI_Abort(MPI_COMM_WORLD, 3);
      }
    }
    red[0] = pinned[N - 1]; red[1] = managed[N - 1];
    MPI_Allreduce(MPI_IN_PLACE, red, 2, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    float msum;
    MPI_Allreduce(&managed[0], &msum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    if (red[0] != (float)(it * size) || red[1] != (float)(it * size) ||
        msum != (float)(it * size)) {
      fprintf(stderr, "rank %d: FAIL: allreduce at %d: %f %f %f\n",
              rank, it, red[0], red[1], msum);
      MPI_Abort(MPI_COMM_WORLD, 4);
    }
    if (it % 100 == 0) { printf("rank %d: iteration %d\n", rank, it); fflush(stdout); }
    usleep(sleep_us);
  }
  printf("rank %d: PASS (%d iterations, %s)\n", rank, iters, mode);
  fflush(stdout);
  MPI_Finalize();
  return 0;
}
