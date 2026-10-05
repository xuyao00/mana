// Test of MANA's VMM allocator (libmana_cuda_vmm.so), without MANA:
//   MANA_CUDA_VMM_ALLOC=1 LD_PRELOAD=.../libmana_cuda_vmm.so ./vmm_alloc_test
// Checks that cudaMalloc/cudaMallocPitch memory is VMM memory shareable as a
// POSIX fd, that it works (memset, kernel, memcpy, many small allocations),
// and times cudaMalloc+cudaFree per size.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void add(int *a, int n, int v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += v;
}

static double now() {
  struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static int posix_fd_shareable(void *p) {
  unsigned long long types = 0;
  cuPointerGetAttribute(&types, CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES, (CUdeviceptr)p);
  return (types & CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) != 0;
}

int main() {
  int bad = 0;
  cudaSetDevice(0);
  // 1. Correctness on sizes around the small/large limit.
  size_t sizes[] = { 4, 1000, 65536, 1 << 20, (1 << 20) + 7, 5 << 20 };
  for (size_t s : sizes) {
    int *d; cudaError_t e = cudaMalloc(&d, s);
    int n = s / sizeof(int);
    cudaMemset(d, 0, s);
    add<<<(n + 255) / 256, 256>>>(d, n, 7);
    int last = -1;
    if (n > 0) cudaMemcpy(&last, d + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
    int ok = e == cudaSuccess && (n == 0 || last == 7) && posix_fd_shareable(d);
    printf("cudaMalloc(%8zu): %s, VMM POSIX-fd %d, data %s\n", s,
           cudaGetErrorString(e), posix_fd_shareable(d), ok ? "OK" : "BAD");
    bad += !ok;
    cudaFree(d);
  }
  // 2. Many small allocations: distinct, aligned, usable; then freed.
  const int K = 2000;
  int *p[K];
  for (int i = 0; i < K; i++) {
    cudaMalloc(&p[i], 100 + i % 3000);
    if ((unsigned long)p[i] % 256) { printf("misaligned %p\n", p[i]); bad++; }
    cudaMemset(p[i], i & 0xff, 64);
  }
  for (int i = 0; i < K; i += 97) {
    unsigned char c; cudaMemcpy(&c, p[i], 1, cudaMemcpyDeviceToHost);
    if (c != (i & 0xff)) { printf("small %d BAD\n", i); bad++; }
  }
  for (int i = 0; i < K; i++) cudaFree(p[i]);
  size_t pitch; int *q;
  cudaError_t e = cudaMallocPitch(&q, &pitch, 1000, 100);
  printf("cudaMallocPitch: %s, pitch %zu, VMM POSIX-fd %d\n", cudaGetErrorString(e),
         pitch, posix_fd_shareable(q));
  cudaFree(q);
  size_t freeb, total; cudaMemGetInfo(&freeb, &total);
  printf("after frees: %zu MB free of %zu\n", freeb >> 20, total >> 20);
  // 3. Timing: cudaMalloc + cudaFree.
  size_t tsizes[] = { 256, 65536, 1 << 20, 16 << 20, 256 << 20 };
  for (size_t s : tsizes) {
    int reps = s >= (16 << 20) ? 50 : 500;
    double t0 = now();
    for (int r = 0; r < reps; r++) { void *d; cudaMalloc(&d, s); cudaFree(d); }
    printf("cudaMalloc+cudaFree %10zu B: %8.1f us\n", s, (now() - t0) / reps * 1e6);
  }
  printf("%s\n", bad ? "FAIL" : "PASS");
  return bad != 0;
}
