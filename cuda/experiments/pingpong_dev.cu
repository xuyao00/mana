// Ping-pong between 2 ranks on device buffers (CUDA-aware MPI):
// latency (8 B) and bandwidth (up to 16 MB).  pingpong_dev [iters]
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

// MEM=vmm: a VMM allocation exportable as a POSIX fd (UCX's cuda_ipc then
// shares it with cuMemExportToShareableHandle instead of cuIpcGetMemHandle).
static void *dev_alloc(size_t size) {
  const char *mem = getenv("MEM");
  if (mem != NULL && strcmp(mem, "host") == 0) {   // host buffers: no CUDA
    void *p = malloc(size); memset(p, 0, size); return p;
  }
  if (mem == NULL || strcmp(mem, "vmm") != 0) {
    void *p; cudaMalloc(&p, size); return p;
  }
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = 0;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  size_t gran; cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  size = (size + gran - 1) / gran * gran;
  CUmemGenericAllocationHandle h; cuMemCreate(&h, size, &prop, 0);
  CUdeviceptr p; cuMemAddressReserve(&p, size, 0, 0, 0); cuMemMap(p, size, 0, h, 0);
  CUmemAccessDesc acc = {}; acc.location = prop.location; acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cuMemSetAccess(p, size, &acc, 1);
  return (void *)p;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int iters = argc > 1 ? atoi(argv[1]) : 1000;
  if (getenv("MEM") == NULL || strcmp(getenv("MEM"), "host") != 0) cudaSetDevice(0);
  size_t max = 16 << 20;
  char *buf = (char *)dev_alloc(max);
  if (getenv("MEM") == NULL || strcmp(getenv("MEM"), "host") != 0) cudaMemset(buf, rank, max);
  size_t sizes[] = { 8, 64, 512, 4096, 32768, 262144, 1 << 20, 4 << 20, 16 << 20 };
  for (size_t n : sizes) {
    int it = n >= (1 << 20) ? iters / 10 + 1 : iters;
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < it; i++) {
      if (rank == 0) {
        MPI_Send(buf, n, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(buf, n, MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      } else {
        MPI_Recv(buf, n, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(buf, n, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
      }
    }
    double dt = (MPI_Wtime() - t0) / it / 2;
    if (rank == 0) {
      printf("%9zu bytes: %8.2f us  %8.1f MB/s\n", n, dt * 1e6, n / dt / 1e6);
      if (getenv("PP_CSV")) fprintf(stderr, "PPCSV,%zu,%.3f,%.1f\n", n, dt * 1e6, n / dt / 1e6);
      fflush(stdout);
    }
  }
  MPI_Finalize();
  return 0;
}
