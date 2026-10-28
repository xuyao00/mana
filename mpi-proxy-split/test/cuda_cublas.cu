// cuBLAS and cuBLASLt (run with MANA's --cuda).  Two cuBLAS handles: one
// with the defaults, and one whose stream (made after the handle),
// workspace, pointer mode and math mode a restart must set again.  A
// cuBLASLt matmul uses descriptors and an algorithm chosen before the first
// iteration.  The matrices hold small integers, so the results in double are
// exact.  A dot product goes to MPI from device memory, and a __device__
// variable's address is taken after cuBLAS has run its kernels: it must come
// back at the same address.

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <string.h>

#include "mana_test.h"

#define N 64

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t e_ = (call);                                                 \
    MT_CHECK(e_ == cudaSuccess, "%s: %s", #call, cudaGetErrorString(e_));    \
  } while (0)

#define BLAS_CHECK(call)                                                     \
  do {                                                                       \
    cublasStatus_t s_ = (call);                                              \
    MT_CHECK(s_ == CUBLAS_STATUS_SUCCESS, "%s: %s", #call,                   \
             cublasGetStatusString(s_));                                     \
  } while (0)

__device__ long counter;

__global__ void
bump()
{
  counter++;
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "cuda_cublas");
  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  CUDA_CHECK(cudaSetDevice(mt_rank % ndev));

  // Column-major A and B, and C = A B.
  static double a[N * N], b[N * N], c[N * N], out[N * N];
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      a[i + N * j] = (i + 2 * j + mt_rank) % 7 - 3;
      b[i + N * j] = (3 * i + j) % 5 - 2;
    }
  }
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N; i++) {
      double sum = 0;
      for (int k = 0; k < N; k++) {
        sum += a[i + N * k] * b[k + N * j];
      }
      c[i + N * j] = sum;
    }
  }
  double x[N], y[N], x2 = 0;
  for (int i = 0; i < N; i++) {
    x[i] = i % 8 + 1;
    x2 += x[i] * x[i];
  }

  cublasHandle_t plain, tuned;
  BLAS_CHECK(cublasCreate(&plain));
  BLAS_CHECK(cublasCreate(&tuned));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  void *workspace;
  size_t workspace_size = 4 << 20;
  CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
  BLAS_CHECK(cublasSetStream(tuned, stream));
  BLAS_CHECK(cublasSetWorkspace(tuned, workspace, workspace_size));
  BLAS_CHECK(cublasSetPointerMode(tuned, CUBLAS_POINTER_MODE_DEVICE));
  BLAS_CHECK(cublasSetMathMode(tuned, CUBLAS_PEDANTIC_MATH));
  const char *success = cublasGetStatusString(CUBLAS_STATUS_SUCCESS);
  char saved[256];
  snprintf(saved, sizeof saved, "%s", success);

  double *da, *db, *dc, *dd, *dx, *dy, *d_alpha, *d_dot, *d_sum;
  CUDA_CHECK(cudaMalloc(&da, sizeof a));
  CUDA_CHECK(cudaMalloc(&db, sizeof b));
  CUDA_CHECK(cudaMalloc(&dc, sizeof c));
  CUDA_CHECK(cudaMalloc(&dd, sizeof c));
  CUDA_CHECK(cudaMalloc(&dx, sizeof x));
  CUDA_CHECK(cudaMalloc(&dy, sizeof y));
  CUDA_CHECK(cudaMalloc(&d_alpha, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_dot, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
  BLAS_CHECK(cublasSetMatrix(N, N, sizeof(double), a, N, da, N));
  BLAS_CHECK(cublasSetMatrix(N, N, sizeof(double), b, N, db, N));
  BLAS_CHECK(cublasSetVector(N, sizeof(double), x, 1, dx, 1));
  CUDA_CHECK(cudaMemset(dy, 0, sizeof y));
  double one = 1, zero = 0;
  CUDA_CHECK(cudaMemcpy(d_alpha, &one, sizeof one, cudaMemcpyHostToDevice));

  // cuBLAS's kernels run, then the application's module is used.
  BLAS_CHECK(cublasDgemm(plain, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, da,
                         N, db, N, &zero, dc, N));
  CUDA_CHECK(cudaDeviceSynchronize());
  long *counter_addr;
  CUDA_CHECK(cudaGetSymbolAddress((void **)&counter_addr, counter));

  // cuBLASLt: D = A B, with an algorithm of the heuristic.
  cublasLtHandle_t lt;
  BLAS_CHECK(cublasLtCreate(&lt));
  cublasLtMatmulDesc_t op;
  BLAS_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_64F, CUDA_R_64F));
  cublasLtMatrixLayout_t layout;
  BLAS_CHECK(cublasLtMatrixLayoutCreate(&layout, CUDA_R_64F, N, N, N));
  cublasLtMatmulPreference_t pref;
  BLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  void *lt_workspace;
  size_t lt_workspace_size = 1 << 20;
  CUDA_CHECK(cudaMalloc(&lt_workspace, lt_workspace_size));
  BLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
    pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt_workspace_size,
    sizeof lt_workspace_size));
  cublasLtMatmulHeuristicResult_t heuristic;
  int found = 0;
  BLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, op, layout, layout, layout,
                                            layout, pref, 1, &heuristic,
                                            &found));
  MT_CHECK(found == 1, "%s", "no cuBLASLt algorithm");

  long it;
  for (it = 0; mt_continue(it); it++) {
    BLAS_CHECK(cublasDgemm(plain, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one,
                           da, N, db, N, &zero, dc, N));
    BLAS_CHECK(cublasLtMatmul(lt, op, &one, da, layout, db, layout, &zero, dd,
                              layout, dd, layout, &heuristic.algo,
                              lt_workspace, lt_workspace_size, 0));
    // y += x, then x . y, on the tuned handle's stream and pointers.
    BLAS_CHECK(cublasDaxpy(tuned, N, d_alpha, dx, 1, dy, 1));
    BLAS_CHECK(cublasDdot(tuned, N, dx, 1, dy, 1, d_dot));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    bump<<<1, 1>>>();

    BLAS_CHECK(cublasGetMatrix(N, N, sizeof(double), dc, N, out, N));
    MT_CHECK(memcmp(out, c, sizeof c) == 0, "iteration %ld: cublasDgemm",
             it);
    BLAS_CHECK(cublasGetMatrix(N, N, sizeof(double), dd, N, out, N));
    MT_CHECK(memcmp(out, c, sizeof c) == 0, "iteration %ld: cublasLtMatmul",
             it);
    BLAS_CHECK(cublasGetVector(N, sizeof(double), dy, 1, y, 1));
    for (int i = 0; i < N; i++) {
      MT_CHECK(y[i] == (it + 1) * x[i], "iteration %ld: y[%d] = %f", it, i,
               y[i]);
    }
    MT_MPI(MPI_Allreduce(d_dot, d_sum, 1, MPI_DOUBLE, MPI_SUM,
                         MPI_COMM_WORLD));
    double sum = 0;
    CUDA_CHECK(cudaMemcpy(&sum, d_sum, sizeof sum, cudaMemcpyDeviceToHost));
    MT_CHECK(sum == mt_size * (it + 1) * x2, "iteration %ld: dot %f", it,
             sum);
    long count = 0;
    CUDA_CHECK(cudaMemcpy(&count, counter_addr, sizeof count,
                          cudaMemcpyDeviceToHost));
    MT_CHECK(count == it + 1, "iteration %ld: counter %ld", it, count);

    cublasPointerMode_t mode;
    BLAS_CHECK(cublasGetPointerMode(tuned, &mode));
    MT_CHECK(mode == CUBLAS_POINTER_MODE_DEVICE, "iteration %ld: mode %d", it,
             mode);
    cublasMath_t math;
    BLAS_CHECK(cublasGetMathMode(tuned, &math));
    MT_CHECK(math == CUBLAS_PEDANTIC_MATH, "iteration %ld: math %d", it,
             math);
    cudaStream_t got;
    BLAS_CHECK(cublasGetStream(tuned, &got));
    MT_CHECK(got == stream, "iteration %ld: stream %p", it, (void *)got);
    MT_CHECK(strcmp(success, saved) == 0, "iteration %ld: '%s'", it,
             success);
  }
  BLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
  BLAS_CHECK(cublasLtMatrixLayoutDestroy(layout));
  BLAS_CHECK(cublasLtMatmulDescDestroy(op));
  BLAS_CHECK(cublasLtDestroy(lt));
  BLAS_CHECK(cublasDestroy(tuned));
  BLAS_CHECK(cublasDestroy(plain));
  mt_finish(it);
  return 0;
}
