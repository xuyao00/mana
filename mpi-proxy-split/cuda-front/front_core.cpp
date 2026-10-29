/****************************************************************************
 *  Copyright (C) 2026 by Yao Xu                                            *
 *                                                                          *
 *  This file is part of DMTCP.                                             *
 *                                                                          *
 *  DMTCP is free software: you can redistribute it and/or                  *
 *  modify it under the terms of the GNU Lesser General Public License as   *
 *  published by the Free Software Foundation, either version 3 of the      *
 *  License, or (at your option) any later version.                         *
 *                                                                          *
 *  DMTCP is distributed in the hope that it will be useful,                *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 *  GNU Lesser General Public License for more details.                     *
 *                                                                          *
 *  You should have received a copy of the GNU Lesser General Public        *
 *  License along with DMTCP:dmtcp/src.  If not, see                        *
 *  <http://www.gnu.org/licenses/>.                                         *
 ****************************************************************************/

// The front's runtime: per-thread lower-half TLS, binding of lower-half
// functions, and the checkpoint events.

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <dlfcn.h>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include "front_objs.h"

void initialize_wrappers();  // libmana.so: sets lh_info

unsigned front_gen = 1;
__thread FrontThread front_thread
  __attribute__((tls_model("initial-exec"))) = { NULL, 0, -1, 0, {}, 0, {} };

// The process's initial thread, whose lower-half TLS the MPI calls share.
static FrontThread *initial_thread;

// The threads that have called CUDA, for the checkpoint.
static std::mutex threads_lock;
static std::set<FrontThread *> threads;

void
front_log(const char *fmt, ...)
{
  static int verbose = -1;
  if (verbose < 0) {
    const char *v = getenv("MANA_CUDA_VERBOSE");
    verbose = v != NULL && v[0] != '\0' && v[0] != '0';
  }
  if (!verbose) {
    return;
  }
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "MANA cuda[%d]: ", getpid());
  vfprintf(stderr, fmt, ap);
  va_end(ap);
}

void
front_fatal(const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "MANA: --cuda: ");
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  abort();
}

void
front_unsupported(const char *name)
{
  front_fatal("%s is not supported (cuda/DESIGN-lower-half-cuda.md)\n", name);
}

void
front_bad_handle(const void *handle, int kind)
{
  static const char *const kinds[] = { "", "stream", "event", "kernel",
                                       "fatbin", "context", "module",
                                       "function", "device", "CUDA array",
                                       "mipmapped array", "texture object",
                                       "cuBLAS handle", "cuBLASLt handle",
                                       "cuFFT plan", "cuRAND generator",
                                       "graph", "graph node",
                                       "executable graph" };
  front_fatal("%p is not a %s that the front made\n", handle,
              kind > 0 && kind < FRONT_N_KINDS ? kinds[kind] : "handle");
}

void
front_missing(const FrontLib *lib, int index)
{
  front_fatal("%s is not in the lower half's %s\n", lib->names[index],
              lib->file != NULL ? lib->file : lib->soname);
}

bool
front_mapped_file(const void *addr, std::string *path)
{
  FILE *maps = fopen("/proc/self/maps", "r");
  if (maps == NULL) {
    return false;
  }
  uintptr_t at = (uintptr_t)addr;
  char line[4096];
  bool found = false;
  while (!found && fgets(line, sizeof line, maps) != NULL) {
    unsigned long start, end;
    char file[4096] = "";
    if (sscanf(line, "%lx-%lx %*s %*s %*s %*s %4095s", &start, &end, file) >=
          2 && at >= start && at < end && file[0] == '/') {
      *path = file;
      found = true;
    }
  }
  fclose(maps);
  return found;
}

static void
front_need_lower_half()
{
  if (lh_info == NULL) {
    initialize_wrappers();
  }
  if (lh_info == NULL || !lh_info->cuda_enabled) {
    front_fatal("the lower half has no CUDA; run with mana_launch --cuda\n");
  }
}

// The process's initial thread uses the lower half's main TLS, as MPI calls
// do, so that the MPI's own CUDA calls see the application's current
// context.  (Its virtual tid is the virtual pid, also after a restart.)
static bool
front_is_initial_thread()
{
  return syscall(SYS_gettid) == getpid();
}

// The lower half's TLS factory takes one caller at a time.
static std::mutex tls_lock;

static void *
front_new_tls()
{
  std::lock_guard<std::mutex> guard(tls_lock);
  void *fs = NULL;
  typedef void *(*new_tls_t)(void);
  new_tls_t new_tls = (new_tls_t)lh_info->cuda_new_tls;
  JUMP_TO_LOWER_HALF(lh_info->cuda_tls_factory_fs);
  fs = new_tls();
  RETURN_TO_UPPER_HALF();
  if (fs == NULL) {
    front_fatal("cannot create a lower-half thread for a TLS\n");
  }
  return fs;
}

// The thread exits: so does the lower-half thread whose TLS it has.
static void
front_thread_exit()
{
  FrontThread *t = &front_thread;
  {
    std::lock_guard<std::mutex> guard(threads_lock);
    threads.erase(t);
  }
  if (t->gen != front_gen || t->fs == NULL || lh_info == NULL ||
      t->fs == lh_info->fsaddr) {
    return;
  }
  std::lock_guard<std::mutex> guard(tls_lock);
  typedef void (*free_tls_t)(void *);
  free_tls_t free_tls = (free_tls_t)lh_info->cuda_free_tls;
  void *fs = t->fs;
  JUMP_TO_LOWER_HALF(lh_info->cuda_tls_factory_fs);
  free_tls(fs);
  RETURN_TO_UPPER_HALF();
  t->fs = NULL;
  t->gen = 0;
}

void *
front_lh_fs_slow()
{
  front_need_lower_half();
  FrontThread *t = &front_thread;
  bool restarted = t->gen != 0 && t->gen != front_gen;
  bool initial = front_is_initial_thread();
  t->fs = initial ? lh_info->fsaddr : front_new_tls();
  t->gen = front_gen;
  if (initial) {
    initial_thread = t;
  }
  {
    std::lock_guard<std::mutex> guard(threads_lock);
    threads.insert(t);
  }
  // After a restart, this thread's contexts and current device come back on
  // its first call into the lower half.
  if (restarted) {
    front_bind(&front_cudart_lib);
    front_bind(&front_cuda_lib);
    front_stack_restore(t);
  }
  return t->fs;
}

static void *
front_lh_dlsym(const char *library, const char *name)
{
  typedef void *(*dlsym_t)(const char *, const char *);
  dlsym_t lh_dlsym = (dlsym_t)lh_info->cuda_dlsym;
  void *fs = front_lh_fs();
  void *fn = NULL;
  JUMP_TO_LOWER_HALF(fs);
  fn = lh_dlsym(library, name);
  RETURN_TO_UPPER_HALF();
  return fn;
}

// A library that only the application uses (cuBLAS, cuFFT, ...) loads in
// the lower half by its soname, through the lower half's search path; else
// from the directory of the lower half's libcudart, where a toolkit keeps
// it.  The MPI loaded libcudart and libcuda already.
static const char *
front_lib_file(FrontLib *lib)
{
  if (lib == &front_cudart_lib || lib == &front_cuda_lib || lib->n == 0 ||
      front_lh_dlsym(lib->soname, lib->names[0]) != NULL) {
    return lib->soname;
  }
  front_check_bound(&front_cudart_lib);
  std::string cudart;
  if (front_mapped_file((void *)REAL(cudaGetErrorString), &cudart)) {
    std::string path = cudart.substr(0, cudart.rfind('/') + 1) + lib->soname;
    if (front_lh_dlsym(path.c_str(), lib->names[0]) != NULL) {
      return strdup(path.c_str());
    }
  }
  front_fatal("the lower half cannot load %s: put its directory on "
              "LD_LIBRARY_PATH\n", lib->soname);
}

// NULL for a function that this lower half lacks: the front libraries are
// made from the lower half's libraries, so only for one of an older driver,
// which the application would not have found there either, or of another
// version of a library (front_missing()).
void
front_bind(FrontLib *lib)
{
  static std::recursive_mutex lock;
  std::lock_guard<std::recursive_mutex> guard(lock);
  if (lib->bound_gen == front_gen) {
    return;
  }
  front_need_lower_half();
  if (lib->file == NULL) {
    lib->file = front_lib_file(lib);
  }
  for (int i = 0; i < lib->n; i++) {
    lib->real[i] = front_lh_dlsym(lib->file, lib->names[i]);
  }
  __atomic_store_n(&lib->bound_gen, front_gen, __ATOMIC_RELEASE);
}

extern "C" void mana_cuda_event(int event);

unsigned front_slot_gen;

void
front_open_slot()
{
  static std::mutex lock;
  std::lock_guard<std::mutex> guard(lock);
  if (front_slot_gen == front_gen) {
    return;
  }
  front_need_lower_half();
  typedef void (*open_slot_t)(void);
  open_slot_t open_slot = (open_slot_t)lh_info->cuda_open_slot;
  void *fs = front_lh_fs();
  JUMP_TO_LOWER_HALF(fs);
  open_slot();
  RETURN_TO_UPPER_HALF();
  __atomic_store_n(&front_slot_gen, front_gen, __ATOMIC_RELEASE);
}

// The checkpoint waits for the GPUs (MANA_STEP_QUIESCE).  A kernel can wait
// for a host thread that the checkpoint has stopped at the gate, such as a
// kernel that spins on a flag that the thread sets after a CUDA call; the
// checkpoint would then hang without a word.  So a thread of its own says so
// on stderr every MANA_CUDA_QUIESCE_WARN seconds (60), and ends the process
// after MANA_CUDA_QUIESCE_TIMEOUT seconds, if set.  It never enters the lower
// half.
class QuiesceWatchdog {
 public:
  QuiesceWatchdog() : thread_(&QuiesceWatchdog::watch, this) {}
  ~QuiesceWatchdog()
  {
    {
      std::lock_guard<std::mutex> guard(lock_);
      done_ = true;
    }
    done_cv_.notify_one();
    thread_.join();
  }

 private:
  static long seconds(const char *name, long dflt)
  {
    const char *value = getenv(name);
    return value != NULL && value[0] >= '0' && value[0] <= '9' ? atol(value)
                                                               : dflt;
  }

  void watch()
  {
    long warn = seconds("MANA_CUDA_QUIESCE_WARN", 60);
    long timeout = seconds("MANA_CUDA_QUIESCE_TIMEOUT", 0);
    auto start = std::chrono::steady_clock::now();
    long next_warn = warn;
    std::unique_lock<std::mutex> guard(lock_);
    while (!done_) {
      long next = next_warn;
      if (timeout > 0 && (next == 0 || timeout < next)) {
        next = timeout;
      }
      if (next == 0) {
        done_cv_.wait(guard, [this] { return done_; });
        break;
      }
      if (done_cv_.wait_until(guard, start + std::chrono::seconds(next),
                              [this] { return done_; })) {
        break;
      }
      if (next == timeout) {
        fprintf(stderr, "MANA: --cuda: the GPUs have not finished their work "
                "%ld s into the checkpoint (MANA_CUDA_QUIESCE_TIMEOUT): the "
                "process ends.  A kernel may be waiting for a host thread, "
                "which the checkpoint has stopped.\n", timeout);
        fflush(stderr);
        _exit(1);
      }
      fprintf(stderr, "MANA: --cuda: the checkpoint has waited %ld s for the "
              "GPUs.  A kernel may be waiting for a host thread, which the "
              "checkpoint has stopped; MANA_CUDA_QUIESCE_TIMEOUT=SECONDS ends "
              "the process instead.\n", next_warn);
      next_warn += warn;
    }
  }

  std::mutex lock_;
  std::condition_variable done_cv_;
  bool done_ = false;
  std::thread thread_;  // last: it uses the others
};

// libmana.so, the core, calls mana_cuda_event() through the function that
// it is given here, also if the application dlopen()s a front library.
__attribute__((constructor)) static void
front_register()
{
  typedef void (*register_t)(int, mana_layer_fn_t);
  register_t reg = (register_t)dlsym(RTLD_DEFAULT, "mana_register_layer");
  if (reg != NULL) {
    reg(MANA_LAYER_CUDA, mana_cuda_event);
  }
}

// The CUDA layer's steps (mana_core.h).  Called by libmana.so (mana_core.cpp)
// on the checkpoint thread, with the lower half closed to the application's
// threads; MANA_STEP_THREAD_EXIT, on an exiting thread;
// MANA_STEP_DRAIN_MEMOPS, in an MPI call.  The other steps are not the
// front's.
extern "C" void
mana_cuda_event(int event)
{
  switch (event) {
    case MANA_STEP_QUIESCE: {
      QuiesceWatchdog watchdog;
      // The application's threads are out of the lower half, so no new
      // work can start, and their context stacks stay as they are: the
      // copies learn what the MPI made current.  Then wait for the work
      // that was started.
      {
        std::lock_guard<std::mutex> guard(threads_lock);
        for (FrontThread *t : threads) {
          if (t != &front_thread && t->gen == front_gen) {
            front_stack_sync(t);
          }
        }
      }
      cudaError_t rc = cudaSuccess;
      for (int dev = 0; rc == cudaSuccess && dev < 64; dev++) {
        if (!(front_devices_used & (1ULL << dev))) {
          continue;
        }
        FRONT_LH(rc = REAL(cudaSetDevice)(dev));
        if (rc == cudaSuccess) {
          FRONT_LH(rc = REAL(cudaDeviceSynchronize)());
        }
      }
      if (rc != cudaSuccess) {
        front_fatal("synchronizing the GPUs before a checkpoint: error %d\n",
                    rc);
      }
      front_ctx_quiesce();
      front_events_quiesce();
      break;
    }
    case MANA_STEP_STASH:
      front_objs_stash();
      front_vmm_stash();
      break;
    case MANA_STEP_RESUME:
      front_vmm_drop_shadows();
      front_objs_drop_shadows();
      front_events_resume();
      break;
    case MANA_STEP_RESTART:
      if (lh_info == NULL || !lh_info->cuda_enabled) {
        front_fatal("the new lower half has no CUDA; restart with "
                    "mana_restart --cuda\n");
      }
      __atomic_add_fetch(&front_gen, 1, __ATOMIC_ACQ_REL);
      // On the lower half's main TLS, where the initial thread made most
      // objects at launch: its per-thread state in the driver and in cudart
      // exists already, as it did then (see front_note_cudart()).  The MPI
      // calls of this thread use it too; the application's threads are not
      // running.
      front_thread.fs = lh_info->fsaddr;
      front_thread.gen = front_gen;
      front_objs_restore_devices();
      front_vmm_map();
      front_objs_restore();
      front_vmm_fill();
      front_events_restart();
      front_vmm_drop_shadows();
      front_objs_drop_shadows();
      // Now, not on its first call: the MPI's calls on the checkpoint thread
      // (the replay) use the initial thread's TLS too.  The application's
      // threads are not running yet.
      if (initial_thread != NULL) {
        initial_thread->fs = lh_info->fsaddr;
        initial_thread->gen = front_gen;
        front_stack_restore(initial_thread);
      } else {
        CUresult rc;
        FRONT_LH(rc = DRV(cuCtxSetCurrent)(NULL));
        (void)rc;
      }
      front_ipc_publish();
      // This thread gets a TLS of its own at its next call.
      if (&front_thread != initial_thread) {
        front_thread.fs = NULL;
        front_thread.gen = 0;
      }
      break;
    case MANA_STEP_RESTART_PEERS:
      front_ipc_reimport();
      break;
    case MANA_STEP_THREAD_EXIT:
      front_thread_exit();
      break;
    case MANA_STEP_DRAIN_MEMOPS:
      front_drain_memops();
      break;
    default:
      break;
  }
}
