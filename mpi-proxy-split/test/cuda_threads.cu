// CUDA from several threads (run with MANA's --cuda), built with nvcc's
// --default-stream per-thread.  Each of NT workers owns a stream, an event,
// device state and pinned memory, made before the first checkpoint.  Each
// iteration, a worker computes on its per-thread default stream, makes its
// stream wait for that work, advances its state on its stream, and checks the
// state.  The main thread then reduces the workers' states over MPI on a
// device buffer.  A short-lived thread per iteration also calls CUDA and
// exits, so the lower-half TLS donors come and go, also across checkpoints;
// the number of threads must stay bounded.

#include <cuda_runtime.h>
#include <dirent.h>
#include <pthread.h>

#include "mana_test.h"

#define N 4096
#define NT 4

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t e_ = (call);                                                 \
    MT_CHECK(e_ == cudaSuccess, "%s: %s", #call, cudaGetErrorString(e_));    \
  } while (0)

// In a worker: report the first error to the main thread, which aborts.
#define W_CHECK(w, cond, ...)                                                \
  do {                                                                       \
    if (!(cond) && (w)->error[0] == '\0') {                                  \
      snprintf((w)->error, sizeof((w)->error), __VA_ARGS__);                 \
    }                                                                        \
  } while (0)
#define W_CUDA(w, call)                                                      \
  do {                                                                       \
    cudaError_t e_ = (call);                                                 \
    W_CHECK(w, e_ == cudaSuccess, "%s: %s", #call, cudaGetErrorString(e_));  \
  } while (0)

__global__ void
fill(int *a, int n, int value)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = value + i % 7;
}

__global__ void
add(int *a, const int *b, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += b[i];
}

struct Worker {
  int id;
  cudaStream_t stream;
  cudaEvent_t event;
  int *state;      // state[i] = sum over k <= it of (id + k + i % 7)
  int *scratch;
  int *host;       // pinned: state[0] and state[N - 1]
  char error[256];
};

static Worker workers[NT];
static pthread_barrier_t start, done;
static volatile int stopping;
static int device;
static int *churn_buf;

static long
expected(int id, long it, int i)
{
  return (it + 1) * (id + i % 7) + it * (it + 1) / 2;
}

static void *
work(void *arg)
{
  Worker *w = (Worker *)arg;
  W_CUDA(w, cudaSetDevice(device));
  W_CUDA(w, cudaStreamCreateWithFlags(&w->stream, cudaStreamNonBlocking));
  W_CUDA(w, cudaEventCreateWithFlags(&w->event, cudaEventDisableTiming));
  W_CUDA(w, cudaMalloc(&w->state, N * sizeof(int)));
  W_CUDA(w, cudaMalloc(&w->scratch, N * sizeof(int)));
  W_CUDA(w, cudaMallocHost(&w->host, 2 * sizeof(int)));
  W_CUDA(w, cudaMemsetAsync(w->state, 0, N * sizeof(int), w->stream));
  W_CUDA(w, cudaStreamSynchronize(w->stream));
  for (long it = 0;; it++) {
    pthread_barrier_wait(&start);
    if (stopping) {
      break;
    }
    // The per-thread default stream (stream 0 in this build).
    fill<<<(N + 255) / 256, 256>>>(w->scratch, N, w->id + (int)it);
    W_CUDA(w, cudaGetLastError());
    W_CUDA(w, cudaEventRecord(w->event, cudaStreamPerThread));
    W_CUDA(w, cudaStreamWaitEvent(w->stream, w->event, 0));
    add<<<(N + 255) / 256, 256, 0, w->stream>>>(w->state, w->scratch, N);
    W_CUDA(w, cudaGetLastError());
    W_CUDA(w, cudaMemcpyAsync(&w->host[0], &w->state[0], sizeof(int),
                              cudaMemcpyDeviceToHost, w->stream));
    W_CUDA(w, cudaMemcpyAsync(&w->host[1], &w->state[N - 1], sizeof(int),
                              cudaMemcpyDeviceToHost, w->stream));
    W_CUDA(w, cudaStreamSynchronize(w->stream));
    W_CHECK(w, w->host[0] == expected(w->id, it, 0),
            "worker %d: iteration %ld: state[0] %d, not %ld", w->id, it,
            w->host[0], expected(w->id, it, 0));
    W_CHECK(w, w->host[1] == expected(w->id, it, N - 1),
            "worker %d: iteration %ld: state[N-1] %d, not %ld", w->id, it,
            w->host[1], expected(w->id, it, N - 1));
    pthread_barrier_wait(&done);
  }
  return NULL;
}

// A thread that lives for one iteration.
static void *
churn(void *arg)
{
  long it = *(long *)arg;
  static char error[256];
  int got[2] = { 0, 0 };
  cudaError_t rc = cudaSetDevice(device);
  if (rc == cudaSuccess) {
    rc = cudaMemsetAsync(churn_buf, (int)(it & 0x7f), 2 * sizeof(int),
                         cudaStreamPerThread);
  }
  if (rc == cudaSuccess) {
    rc = cudaMemcpyAsync(got, churn_buf, sizeof got, cudaMemcpyDeviceToHost,
                         cudaStreamPerThread);
  }
  if (rc == cudaSuccess) {
    rc = cudaStreamSynchronize(cudaStreamPerThread);
  }
  int want = (int)(it & 0x7f) * 0x01010101;
  if (rc != cudaSuccess) {
    snprintf(error, sizeof error, "churn: %s", cudaGetErrorString(rc));
    return error;
  }
  if (got[0] != want || got[1] != want) {
    snprintf(error, sizeof error, "churn: iteration %ld: %#x, not %#x", it,
             got[0], want);
    return error;
  }
  return NULL;
}

static int
count_threads()
{
  DIR *dir = opendir("/proc/self/task");
  int n = 0;
  if (dir != NULL) {
    for (struct dirent *e; (e = readdir(dir)) != NULL;) {
      n += e->d_name[0] != '.';
    }
    closedir(dir);
  }
  return n;
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "cuda_threads");
  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  device = mt_rank % ndev;
  CUDA_CHECK(cudaSetDevice(device));
  int *d_sum;
  CUDA_CHECK(cudaMalloc(&d_sum, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&churn_buf, 2 * sizeof(int)));

  pthread_barrier_init(&start, NULL, NT + 1);
  pthread_barrier_init(&done, NULL, NT + 1);
  pthread_t threads[NT];
  for (int t = 0; t < NT; t++) {
    workers[t].id = t;
    MT_CHECK(pthread_create(&threads[t], NULL, work, &workers[t]) == 0,
             "%s", "pthread_create");
  }

  int baseline = -1;
  long it;
  for (it = 0; mt_continue(it); it++) {
    pthread_barrier_wait(&start);
    pthread_barrier_wait(&done);
    long sum = 0;
    for (int t = 0; t < NT; t++) {
      MT_CHECK(workers[t].error[0] == '\0', "%s", workers[t].error);
      sum += workers[t].host[0];
    }

    int local = (int)sum;
    CUDA_CHECK(cudaMemcpy(d_sum, &local, sizeof(int), cudaMemcpyHostToDevice));
    // The copy may still be in flight (from pageable memory), and the MPI's
    // streams do not wait for the per-thread default stream.
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    MT_MPI(MPI_Allreduce(MPI_IN_PLACE, d_sum, 1, MPI_INT, MPI_SUM,
                         MPI_COMM_WORLD));
    int total = 0;
    CUDA_CHECK(cudaMemcpy(&total, d_sum, sizeof(int), cudaMemcpyDeviceToHost));
    long want = 0;
    for (int t = 0; t < NT; t++) {
      want += expected(t, it, 0);
    }
    MT_CHECK(total == (int)(want * mt_size),
             "iteration %ld: MPI_Allreduce: %d, not %ld", it, total,
             want * mt_size);

    pthread_t c;
    void *result = NULL;
    MT_CHECK(pthread_create(&c, NULL, churn, &it) == 0, "%s",
             "pthread_create");
    pthread_join(c, &result);
    MT_CHECK(result == NULL, "%s", (char *)result);

    if (it == 20) {
      baseline = count_threads();
    } else if (baseline > 0 && it % 50 == 0) {
      int n = count_threads();
      MT_CHECK(n <= baseline + 4, "iteration %ld: %d threads, %d at "
               "iteration 20", it, n, baseline);
    }
  }

  stopping = 1;
  pthread_barrier_wait(&start);
  for (int t = 0; t < NT; t++) {
    pthread_join(threads[t], NULL);
  }
  mt_finish(it);
  return 0;
}
