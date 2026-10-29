// A kernel that waits for a host thread (run with --cuda --no-mpi): it spins
// until a flag in mapped pinned memory is set, and a host thread sets the
// flag only after a CUDA call of its own, half a second after the launch.  A
// checkpoint that comes meanwhile stops that thread at MANA's gate, so the
// kernel never ends, and the checkpoint's wait for the GPUs must time out:
// autotest runs this with MANA_CUDA_QUIESCE_TIMEOUT, and the process must end
// with its message.  Without a checkpoint, every iteration ends.

#include <cuda_runtime.h>
#include <pthread.h>

#define MT_NO_MPI
#include "mana_test.h"

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t e_ = (call);                                                 \
    MT_CHECK(e_ == cudaSuccess, "%s: %s", #call, cudaGetErrorString(e_));    \
  } while (0)

__global__ void
spin(volatile int *flag)
{
  while (*flag == 0) {
  }
}

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static long launched, released;
static volatile int *flag;
static cudaEvent_t event;

// Sets the flag for each launch, half a second after it, after a CUDA call.
static void *
releaser(void *arg)
{
  (void)arg;
  for (;;) {
    pthread_mutex_lock(&lock);
    while (released == launched) {
      pthread_cond_wait(&cond, &lock);
    }
    pthread_mutex_unlock(&lock);
    usleep(500000);
    cudaError_t rc = cudaEventQuery(event);
    MT_CHECK(rc == cudaSuccess || rc == cudaErrorNotReady, "%s",
             "cudaEventQuery");
    *flag = 1;
    pthread_mutex_lock(&lock);
    released++;
    pthread_mutex_unlock(&lock);
  }
  return NULL;
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "cuda_quiesce");
  int ndev;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  CUDA_CHECK(cudaSetDevice(mt_rank % ndev));
  int *host;
  CUDA_CHECK(cudaHostAlloc(&host, sizeof(int), cudaHostAllocMapped));
  flag = host;
  int *device_flag;
  CUDA_CHECK(cudaHostGetDevicePointer(&device_flag, host, 0));
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&event));
  CUDA_CHECK(cudaEventRecord(event, stream));
  pthread_t thread;
  MT_CHECK(pthread_create(&thread, NULL, releaser, NULL) == 0, "%s",
           "pthread_create");

  long it;
  for (it = 0; mt_continue(it); it++) {
    *flag = 0;
    spin<<<1, 1, 0, stream>>>(device_flag);
    CUDA_CHECK(cudaGetLastError());
    pthread_mutex_lock(&lock);
    launched++;
    pthread_cond_signal(&cond);
    pthread_mutex_unlock(&lock);
    // Not cudaStreamSynchronize(): this thread leaves the lower half between
    // the polls, so the checkpoint can close it.
    cudaError_t rc;
    while ((rc = cudaStreamQuery(stream)) == cudaErrorNotReady) {
      usleep(1000);
    }
    CUDA_CHECK(rc);
    MT_CHECK(*flag == 1, "iteration %ld: the kernel ended early", it);
  }
  mt_finish(it);
  return 0;
}
