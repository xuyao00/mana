// cuFFT (run with MANA's --cuda), with plans made in each of the ways that a
// restart must repeat: cufftPlan1d; cufftCreate with a property and without
// auto-allocation, cufftMakePlanMany, then the application's work area and a
// stream made after the plan; cufftXtMakePlanMany in double, run by
// cufftXtExec.  Each iteration transforms signals of one frequency, which
// must give one peak, and transforms one back.  A peak goes to MPI from
// device memory.

#include <cuda_runtime.h>
#include <cufftXt.h>
#include <math.h>

#include "mana_test.h"

#define N 256
#define BATCH 4
#define ROWS 32
#define COLS 64

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t e_ = (call);                                                 \
    MT_CHECK(e_ == cudaSuccess, "%s: %s", #call, cudaGetErrorString(e_));    \
  } while (0)

#define FFT_CHECK(call)                                                      \
  do {                                                                       \
    cufftResult r_ = (call);                                                 \
    MT_CHECK(r_ == CUFFT_SUCCESS, "%s: result %d", #call, (int)r_);          \
  } while (0)

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "cuda_cufft");
  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  CUDA_CHECK(cudaSetDevice(mt_rank % ndev));

  // A batch of complex signals.
  cufftHandle batched;
  FFT_CHECK(cufftPlan1d(&batched, N, CUFFT_C2C, BATCH));

  // 2D real to complex, in the application's work area, on its stream.
  cufftHandle real2d;
  FFT_CHECK(cufftCreate(&real2d));
  FFT_CHECK(cufftSetPlanPropertyInt64(
    real2d, NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT, 1));
  FFT_CHECK(cufftSetAutoAllocation(real2d, 0));
  int dims[2] = { ROWS, COLS };
  size_t work_size = 0;
  FFT_CHECK(cufftMakePlanMany(real2d, 2, dims, NULL, 1, 0, NULL, 1, 0,
                              CUFFT_R2C, 1, &work_size));
  void *work = NULL;
  CUDA_CHECK(cudaMalloc(&work, work_size > 0 ? work_size : 1));
  FFT_CHECK(cufftSetWorkArea(real2d, work));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  FFT_CHECK(cufftSetStream(real2d, stream));

  // Double precision, through the Xt interface.
  cufftHandle xt;
  FFT_CHECK(cufftCreate(&xt));
  long long n64[1] = { N };
  size_t xt_work = 0;
  FFT_CHECK(cufftXtMakePlanMany(xt, 1, n64, NULL, 1, 0, CUDA_C_64F, NULL, 1,
                                0, CUDA_C_64F, 1, &xt_work, CUDA_C_64F));

  cufftComplex *d_batched;
  cufftReal *d_real;
  cufftComplex *d_spectrum;
  cufftDoubleComplex *d_xt;
  double *d_sum;
  CUDA_CHECK(cudaMalloc(&d_batched, sizeof(cufftComplex) * N * BATCH));
  CUDA_CHECK(cudaMalloc(&d_real, sizeof(cufftReal) * ROWS * COLS));
  CUDA_CHECK(cudaMalloc(&d_spectrum,
                        sizeof(cufftComplex) * ROWS * (COLS / 2 + 1)));
  CUDA_CHECK(cudaMalloc(&d_xt, sizeof(cufftDoubleComplex) * N));
  CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));

  static cufftComplex signal[BATCH][N], spectrum[BATCH][N];
  static cufftReal plane[ROWS][COLS];
  static cufftComplex plane_spectrum[ROWS][COLS / 2 + 1];
  static cufftDoubleComplex wave[N], wave_spectrum[N];
  long it;
  for (it = 0; mt_continue(it); it++) {
    int k = (int)(it % N);

    // One frequency per signal of the batch: one peak of N; and back.
    for (int b = 0; b < BATCH; b++) {
      for (int n = 0; n < N; n++) {
        double phase = 2 * M_PI * ((k + b) % N) * n / N;
        signal[b][n].x = (float)cos(phase);
        signal[b][n].y = (float)sin(phase);
      }
    }
    CUDA_CHECK(cudaMemcpy(d_batched, signal, sizeof signal,
                          cudaMemcpyHostToDevice));
    FFT_CHECK(cufftExecC2C(batched, d_batched, d_batched, CUFFT_FORWARD));
    CUDA_CHECK(cudaMemcpy(spectrum, d_batched, sizeof spectrum,
                          cudaMemcpyDeviceToHost));
    for (int b = 0; b < BATCH; b++) {
      for (int f = 0; f < N; f++) {
        double want = f == (k + b) % N ? N : 0;
        MT_CHECK(fabs(spectrum[b][f].x - want) < 1e-2 &&
                   fabs(spectrum[b][f].y) < 1e-2,
                 "iteration %ld: signal %d: bin %d: %f%+fi", it, b, f,
                 spectrum[b][f].x, spectrum[b][f].y);
      }
    }
    FFT_CHECK(cufftExecC2C(batched, d_batched, d_batched, CUFFT_INVERSE));
    CUDA_CHECK(cudaMemcpy(spectrum, d_batched, sizeof spectrum,
                          cudaMemcpyDeviceToHost));
    for (int n = 0; n < N; n++) {
      MT_CHECK(fabs(spectrum[0][n].x / N - signal[0][n].x) < 1e-4 &&
                 fabs(spectrum[0][n].y / N - signal[0][n].y) < 1e-4,
               "iteration %ld: back at %d", it, n);
    }

    // A cosine along the rows: a peak of ROWS * COLS / 2 at (0, f).
    int f = 1 + (int)(it % (COLS / 2 - 1));
    for (int r = 0; r < ROWS; r++) {
      for (int c = 0; c < COLS; c++) {
        plane[r][c] = (float)cos(2 * M_PI * f * c / COLS);
      }
    }
    CUDA_CHECK(cudaMemcpyAsync(d_real, plane, sizeof plane,
                               cudaMemcpyHostToDevice, stream));
    FFT_CHECK(cufftExecR2C(real2d, d_real, d_spectrum));
    CUDA_CHECK(cudaMemcpyAsync(plane_spectrum, d_spectrum,
                               sizeof plane_spectrum, cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int r = 0; r < ROWS; r++) {
      for (int c = 0; c <= COLS / 2; c++) {
        double want = r == 0 && c == f ? ROWS * COLS / 2 : 0;
        MT_CHECK(fabs(plane_spectrum[r][c].x - want) < 1e-2 &&
                   fabs(plane_spectrum[r][c].y) < 1e-2,
                 "iteration %ld: 2D bin (%d, %d): %f%+fi", it, r, c,
                 plane_spectrum[r][c].x, plane_spectrum[r][c].y);
      }
    }

    // In double; the peaks of all ranks summed from device memory.
    for (int n = 0; n < N; n++) {
      wave[n].x = cos(2 * M_PI * k * n / N);
      wave[n].y = sin(2 * M_PI * k * n / N);
    }
    CUDA_CHECK(cudaMemcpy(d_xt, wave, sizeof wave, cudaMemcpyHostToDevice));
    FFT_CHECK(cufftXtExec(xt, d_xt, d_xt, CUFFT_FORWARD));
    CUDA_CHECK(cudaMemcpy(wave_spectrum, d_xt, sizeof wave_spectrum,
                          cudaMemcpyDeviceToHost));
    MT_CHECK(fabs(wave_spectrum[k].x - N) < 1e-9 &&
               fabs(wave_spectrum[(k + 1) % N].x) < 1e-9,
             "iteration %ld: double peak %f", it, wave_spectrum[k].x);
    MT_MPI(MPI_Allreduce(&d_xt[k].x, d_sum, 1, MPI_DOUBLE, MPI_SUM,
                         MPI_COMM_WORLD));
    double sum = 0;
    CUDA_CHECK(cudaMemcpy(&sum, d_sum, sizeof sum, cudaMemcpyDeviceToHost));
    MT_CHECK(fabs(sum - (double)mt_size * N) < 1e-6,
             "iteration %ld: MPI sum %f", it, sum);

    // What the plans were made with.
    long long patient = 0;
    FFT_CHECK(cufftGetPlanPropertyInt64(
      real2d, NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT, &patient));
    MT_CHECK(patient == 1, "iteration %ld: property %lld", it, patient);
    size_t size = 0;
    FFT_CHECK(cufftGetSize(real2d, &size));
    MT_CHECK(size == work_size, "iteration %ld: work size %zu, not %zu", it,
             size, work_size);
  }
  FFT_CHECK(cufftDestroy(xt));
  FFT_CHECK(cufftDestroy(real2d));
  FFT_CHECK(cufftDestroy(batched));
  mt_finish(it);
  return 0;
}
