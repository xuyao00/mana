// Work that CUDA finishes later (run with MANA's --cuda): host functions and
// stream callbacks, which must see the application's own stream handle;
// stream-ordered allocation with the default pool's release threshold; event
// times across checkpoints; and a cudaMemset whose buffer goes to MPI right
// away, which the next MPI call must wait for.

#include <cuda_runtime.h>
#include <math.h>

#include "mana_test.h"

#define N 65536

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t e_ = (call);                                                 \
    MT_CHECK(e_ == cudaSuccess, "%s: %s", #call, cudaGetErrorString(e_));    \
  } while (0)

__global__ void
fill(int *a, int n, int value)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = value;
}

__global__ void
spin(long cycles)
{
  long start = clock64();
  while (clock64() - start < cycles) {
  }
}

static cudaStream_t stream;
static volatile long host_fn_calls;
static volatile long callback_calls;
static volatile int callback_ok = 1;

static void CUDART_CB
count_host_fn(void *arg)
{
  host_fn_calls += (long)arg;
}

static void CUDART_CB
check_callback(cudaStream_t s, cudaError_t status, void *arg)
{
  if (s != stream || status != cudaSuccess || arg != &callback_calls) {
    callback_ok = 0;
  }
  callback_calls++;
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "cuda_async");
  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  int device = mt_rank % ndev;
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cudaMemPool_t pool;
  CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, device));
  uint64_t threshold = 64 << 20, got_threshold = 0;
  CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                     &threshold));

  // e1 and e2 are recorded once: their elapsed time must stay the same.
  cudaEvent_t e1, e2, e_start, e_now;
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventCreate(&e2));
  CUDA_CHECK(cudaEventCreate(&e_start));
  CUDA_CHECK(cudaEventCreate(&e_now));
  CUDA_CHECK(cudaEventRecord(e_start, stream));
  CUDA_CHECK(cudaEventRecord(e1, stream));
  spin<<<1, 1, 0, stream>>>(1000000);
  CUDA_CHECK(cudaEventRecord(e2, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float e1_e2 = 0;
  CUDA_CHECK(cudaEventElapsedTime(&e1_e2, e1, e2));
  MT_CHECK(e1_e2 > 0, "elapsed time %f", e1_e2);

  int *send, *recv;
  CUDA_CHECK(cudaMalloc(&send, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&recv, N * sizeof(int)));
  int right = (mt_rank + 1) % mt_size;
  int left = (mt_rank + mt_size - 1) % mt_size;
  float last_elapsed = 0;

  long it;
  for (it = 0; mt_continue(it); it++) {
    // Stream-ordered memory, used and freed on the stream.
    int *tmp;
    CUDA_CHECK(cudaMallocAsync(&tmp, N * sizeof(int), stream));
    fill<<<(N + 255) / 256, 256, 0, stream>>>(tmp, N, (int)it);
    int last = -1;
    CUDA_CHECK(cudaMemcpyAsync(&last, tmp + N - 1, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaFreeAsync(tmp, stream));

    // Host functions and callbacks run in stream order.
    CUDA_CHECK(cudaLaunchHostFunc(stream, count_host_fn, (void *)2));
    CUDA_CHECK(cudaStreamAddCallback(stream, check_callback,
                                     (void *)&callback_calls, 0));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MT_CHECK(last == (int)it, "iteration %ld: cudaMallocAsync memory: %d", it,
             last);
    MT_CHECK(host_fn_calls == 2 * (it + 1), "iteration %ld: %ld host calls",
             it, host_fn_calls);
    MT_CHECK(callback_calls == it + 1 && callback_ok,
             "iteration %ld: callbacks %ld, ok %d", it, callback_calls,
             callback_ok);

    // Event times.
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e1, e2));
    MT_CHECK(fabsf(ms - e1_e2) < 1e-3, "iteration %ld: e1-e2 %f, not %f", it,
             ms, e1_e2);
    CUDA_CHECK(cudaEventRecord(e_now, stream));
    CUDA_CHECK(cudaEventSynchronize(e_now));
    CUDA_CHECK(cudaEventElapsedTime(&ms, e_start, e_now));
    MT_CHECK(ms >= last_elapsed, "iteration %ld: elapsed %f after %f", it, ms,
             last_elapsed);
    last_elapsed = ms;

    // No synchronization between the set and the MPI call.
    CUDA_CHECK(cudaMemset(send, mt_rank + (int)(it & 0x3f), N * sizeof(int)));
    MT_MPI(MPI_Sendrecv(send, N, MPI_INT, right, 0, recv, N, MPI_INT, left, 0,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    int got[2];
    CUDA_CHECK(cudaMemcpy(&got[0], recv, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&got[1], recv + N - 1, sizeof(int),
                          cudaMemcpyDeviceToHost));
    int byte = (left + (int)(it & 0x3f)) & 0xff;
    int want = byte * 0x01010101;
    MT_CHECK(got[0] == want && got[1] == want,
             "iteration %ld: from rank %d: %#x %#x, not %#x", it, left, got[0],
             got[1], want);
  }
  CUDA_CHECK(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold,
                                     &got_threshold));
  MT_CHECK(got_threshold == threshold, "release threshold %lu",
           (unsigned long)got_threshold);
  mt_finish(it);
  return 0;
}
