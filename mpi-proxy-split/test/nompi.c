// A program without MPI (run with mana_launch --no-mpi): MANA's core alone,
// in bin/lower-half-nompi.  Each iteration updates state in the heap, in an
// mmap()ed region and in a second thread, all of which must survive
// checkpoint and restart.  The ranks come from the launcher and never talk:
// each one's checkpoint is its own ckpt_rank_N.

#include <pthread.h>
#include <sys/mman.h>

#define MT_NO_MPI
#include "mana_test.h"

#define N 65536

// The worker takes one step for each of the main thread's.
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static long main_steps, worker_steps;
static int stop;

static void *
worker(void *arg)
{
  (void)arg;
  pthread_mutex_lock(&lock);
  while (!stop) {
    if (worker_steps < main_steps) {
      worker_steps++;
      pthread_cond_broadcast(&cond);
    } else {
      pthread_cond_wait(&cond, &lock);
    }
  }
  pthread_mutex_unlock(&lock);
  return NULL;
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "nompi");
  int *heap = (int *)malloc(N * sizeof(int));
  int *mapped = (int *)mmap(NULL, N * sizeof(int), PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  MT_CHECK(heap != NULL && mapped != MAP_FAILED, "%s", "out of memory");
  for (int i = 0; i < N; i++) {
    heap[i] = mt_value(mt_rank, 0, i);
    mapped[i] = -mt_value(mt_rank, 0, i);
  }
  pthread_t thread;
  MT_CHECK(pthread_create(&thread, NULL, worker, NULL) == 0, "%s",
           "pthread_create");

  long it;
  for (it = 0; mt_continue(it); it++) {
    for (int i = 0; i < N; i++) {
      MT_CHECK(heap[i] == mt_value(mt_rank, it, i),
               "iteration %ld: heap[%d] is %d", it, i, heap[i]);
      MT_CHECK(mapped[i] == -mt_value(mt_rank, it, i),
               "iteration %ld: mapped[%d] is %d", it, i, mapped[i]);
      heap[i] = mt_value(mt_rank, it + 1, i);
      mapped[i] = -mt_value(mt_rank, it + 1, i);
    }
    pthread_mutex_lock(&lock);
    main_steps++;
    pthread_cond_broadcast(&cond);
    while (worker_steps < main_steps) {
      pthread_cond_wait(&cond, &lock);
    }
    MT_CHECK(worker_steps == it + 1, "iteration %ld: the worker took %ld steps",
             it, worker_steps);
    pthread_mutex_unlock(&lock);
  }

  pthread_mutex_lock(&lock);
  stop = 1;
  pthread_cond_broadcast(&cond);
  pthread_mutex_unlock(&lock);
  pthread_join(thread, NULL);
  mt_finish(it);
  return 0;
}
