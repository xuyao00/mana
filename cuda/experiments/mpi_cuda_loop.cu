// MPI + CUDA checkpoint test.
//   mpi_cuda_loop host   ITERS : MPI on host buffers (cudaMemcpy staging)
//   mpi_cuda_loop device ITERS : CUDA-aware MPI on device buffers
// Each iteration a kernel increments a device array (GPU state that must
// survive checkpoint/restart), then ranks exchange it with MPI and verify.
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>

// MEM=vmm: a VMM allocation exportable as a POSIX fd (UCX's cuda_ipc then
// shares it with cuMemExportToShareableHandle instead of cuIpcGetMemHandle).
static void *dev_alloc(size_t size) {
  const char *mem = getenv("MEM");
  if (mem == NULL || strcmp(mem, "vmm") != 0) {
    void *p; cudaMalloc(&p, size); return p;
  }
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = 0;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  // A NIC (e.g. Slingshot) can register VMM memory only if made RDMA capable.
  int rdma = 0;
  cuDeviceGetAttribute(&rdma, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, 0);
  prop.allocFlags.gpuDirectRDMACapable = rdma;
  size_t gran; cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  size = (size + gran - 1) / gran * gran;
  CUmemGenericAllocationHandle h; cuMemCreate(&h, size, &prop, 0);
  CUdeviceptr p; cuMemAddressReserve(&p, size, 0, 0, 0); cuMemMap(p, size, 0, h, 0);
  CUmemAccessDesc acc = {}; acc.location = prop.location; acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cuMemSetAccess(p, size, &acc, 1);
  return (void *)p;
}


#define N 4096
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "rank %d: %s:%d %s: %s\n", rank, __FILE__, __LINE__, #x, \
          cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

__global__ void incr(float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += 1.0f;
}
__global__ void check(const float *a, int n, float want, int *bad) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && a[i] != want) atomicAdd(bad, 1);
}

int rank, size;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int device = argc > 1 && strcmp(argv[1], "device") == 0;
  int iters = argc > 2 ? atoi(argv[2]) : 1000;
  int ndev; CK(cudaGetDeviceCount(&ndev));
  CK(cudaSetDevice(rank % ndev));

  float *d_state, *d_send, *d_recv; int *d_bad;
  d_state = (float *)dev_alloc(N * sizeof(float));
  d_send = (float *)dev_alloc(N * sizeof(float));
  d_recv = (float *)dev_alloc(N * sizeof(float));
  CK(cudaMalloc(&d_bad, sizeof(int)));
  CK(cudaMemset(d_state, 0, N * sizeof(float)));
  float *h_send = (float*)malloc(N * sizeof(float));
  float *h_recv = (float*)malloc(N * sizeof(float));
  int right = (rank + 1) % size, left = (rank + size - 1) % size;

  for (int it = 1; it <= iters; it++) {
    incr<<<(N + 255) / 256, 256>>>(d_state, N);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    // GPU state check: every element must equal 'it' (catches lost state).
    CK(cudaMemset(d_bad, 0, sizeof(int)));
    check<<<(N + 255) / 256, 256>>>(d_state, N, (float)it, d_bad);
    int bad; CK(cudaMemcpy(&bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost));
    if (bad) {
      fprintf(stderr, "rank %d: FAIL: GPU state wrong at iteration %d\n", rank, it);
      MPI_Abort(MPI_COMM_WORLD, 3);
    }
    // Ring exchange + allreduce, payload = state + rank.
    float got, sum;
    if (device) {
      CK(cudaMemcpy(d_send, d_state, N * sizeof(float), cudaMemcpyDeviceToDevice));
      MPI_Sendrecv(d_send, N, MPI_FLOAT, right, 0, d_recv, N, MPI_FLOAT, left, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Allreduce(MPI_IN_PLACE, d_send, N, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      CK(cudaMemcpy(&got, d_recv + N - 1, sizeof(float), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(&sum, d_send, sizeof(float), cudaMemcpyDeviceToHost));
    } else {
      CK(cudaMemcpy(h_send, d_state, N * sizeof(float), cudaMemcpyDeviceToHost));
      MPI_Sendrecv(h_send, N, MPI_FLOAT, right, 0, h_recv, N, MPI_FLOAT, left, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Allreduce(MPI_IN_PLACE, h_send, N, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      CK(cudaMemcpy(d_recv, h_recv, N * sizeof(float), cudaMemcpyHostToDevice));
      got = h_recv[N - 1]; sum = h_send[0];
    }
    if (got != (float)it || sum != (float)(it * size)) {
      fprintf(stderr, "rank %d: FAIL: iteration %d: got %f sum %f\n", rank, it, got, sum);
      MPI_Abort(MPI_COMM_WORLD, 4);
    }
    if (it % (argc > 3 && atoi(argv[3]) == 0 ? 5000 : 100) == 0) { printf("rank %d: iteration %d\n", rank, it); fflush(stdout); }
    usleep(argc > 3 ? atoi(argv[3]) : 10000);
  }
  printf("rank %d: PASS (%d iterations, %s buffers)\n", rank, iters, device ? "device" : "host");
  MPI_Finalize();
  return 0;
}
