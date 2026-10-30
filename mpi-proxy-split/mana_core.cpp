/****************************************************************************
 *   Copyright (C) 2019-2021 by Gene Cooperman, Rohan Garg, Yao Xu          *
 *   gene@ccs.neu.edu, rohgarg@ccs.neu.edu, xu.yao1@northeastern.edu        *
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
 *  License in the files COPYING and COPYING.LESSER.  If not, see           *
 *  <http://www.gnu.org/licenses/>.                                         *
 ****************************************************************************/

// libmana.so, the core of MANA's upper half: the DMTCP plugin.  It keeps the
// split process (the upper half's memory, FS bases, signals and fds) and runs
// the checkpoint and the restart, calling the steps of the layers that have
// registered (mana_core.h): MPI (libmana_mpi.so, mpi_plugin.cpp) and CUDA
// (cuda-front/).

#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cxxabi.h>  /* For backtrace() */
#include <execinfo.h>  /* For backtrace() */

#include <algorithm>
#include <regex>

#include "jconvert.h"
#include "mana_core.h"
#include "lower-half-api.h"
#include "lower_half_ckpt.h"
#include "uh_wrappers.h"
#include "logging.h"

#include "config.h"
#include "dmtcp.h"
#include "kvdb.h"
#include "util.h"
#include "dmtcp_assert.h"
#include "jfilesystem.h"
#include "protectedfds.h"
#include "procselfmaps.h"

using namespace dmtcp;
using dmtcp::kvdb::KVDBRequest;
using dmtcp::kvdb::KVDBResponse;

/* Global variables */
void * lh_ckpt_mem_addr = NULL;
size_t lh_ckpt_mem_size = 0;
int pagesize = sysconf(_SC_PAGESIZE);

constexpr const char *MANA_FILE_REGEX_ENV = "MANA_FILE_REGEX";
constexpr const char *MANA_SEGV_DEBUG_LOOP = "MANA_SEGV_DEBUG_LOOP";
static std::regex *file_regex = NULL;
static map<string, int> *g_file_flags_map = NULL;
static vector<int> ckpt_fds;
static bool processingOpenCkpFileFds = false;

std::unordered_map<pid_t, unsigned long> *g_upper_half_fsbase = NULL;
DmtcpMutex g_upper_half_fsbase_lock;
int CheckAndEnableFsGsBase();

EXTERNC pid_t dmtcp_get_real_tid() __attribute((weak));

mana_state_t mana_state = UNKNOWN_STATE;
bool g_libmana_is_initialized = false;

void *heapAddr = nullptr;

// High memory could start at 0x7ffc00000000 on Perlmutter
const VA HIGH_ADDR_START = (VA)0x7ffc00000000;

void *old_brk;
void *old_end_of_brk;
char *uh_stack_start;
char *uh_stack_end;

// The launcher's rank (launcher_rank()), then the MPI layer's
// (mana_set_rank()).
static int mana_rank = -1;
// mana_set_header()'s lines of header.mana.
static map<string, string> *header_fields = NULL;

void recordOpenFds()
{
  ostringstream o;
  vector<int>fds = jalib::Filesystem::ListOpenFds();
  for (int fd : fds) {
    if (!Util::isValidFd(fd)) {
      continue;
    }

    string device = jalib::Filesystem::GetDeviceName(fd);
    o << fd << " -> " << device << "\n";
  }

  string workerPath("/worker/" + string(dmtcp_get_uniquepid_str()));
  kvdb::set(workerPath, "ProcSelfFds", o.str());
}

static void
processFileOpen(const char *path, int flags)
{
  if (file_regex == NULL ||
      processingOpenCkpFileFds ||
      Util::strStartsWith(path, "/proc") ||
      Util::strStartsWith(path, "/sys")) {
    return;
  }

  if ((flags & O_APPEND) ||
      ((flags & O_RDWR) || (flags & O_WRONLY))) {
    if (std::regex_search(path, *file_regex)) {
      g_file_flags_map->insert(std::make_pair(path, flags));
    }
  }
}

static void
openCkptFileFds()
{
  ostringstream o;
  ostringstream staleFilesStr;

  processingOpenCkpFileFds = true;
  ckpt_fds.clear();
  ckpt_fds.reserve(g_file_flags_map->size());

  vector<string> staleFiles;

  for (auto i : *g_file_flags_map) {
    string const& path = i.first;
    off64_t flags = i.second;

    struct stat statbuf;
    if (stat(path.c_str(), &statbuf) == 0 && S_ISREG(statbuf.st_mode)) {
      int fd = open(path.c_str(), O_APPEND | O_WRONLY);
      WARN_ERRNO(fd != -1, "path={} flags={}", path, flags);

      if (fd != -1) {
        o << path << ";size:" << statbuf.st_size << ";flags:" << std::oct << flags << std::dec << "\n";
        ckpt_fds.push_back(fd);
      }
    } else {
      staleFilesStr << path << ":" << flags<< "\n";
      staleFiles.push_back(path);
    }
  }

  for (string const& stale : staleFiles) {
    g_file_flags_map->erase(stale);
  }

  string workerPath("/worker/" + string(dmtcp_get_uniquepid_str()));
  kvdb::set(workerPath, "Ckpt_Files", o.str());
  kvdb::set(workerPath, "Ckpt_Files_Stale", staleFilesStr.str());
}

static void
closeCkptFileFds()
{
  for (int fd : ckpt_fds) {
    close(fd);
  }

  ckpt_fds.clear();
}

static void
logCkptFileFds()
{
  ostringstream o;
  struct stat statbuf;
  string workerPath("/worker/" + string(dmtcp_get_uniquepid_str()));

  for (int fd : ckpt_fds) {
    ASSERT_ERRNO(fstat(fd, &statbuf) == 0, "fd={}", fd);
    o << jalib::Filesystem::GetDeviceName(fd) << ";size:" << statbuf.st_size << "\n";
  }

  if (ckpt_fds.size() > 0) {
    kvdb::set(workerPath, "Ckpt_Files_Restart", o.str());
  }
}

// Finds the first range of the upper half's memory that ends above addr, and
// stores it in [*start, *end).  This range is either a run of the upper half's
// mappings, which the lower half's mmap wrapper records, or the upper-half
// stack.  Returns false if there is no such range.
static bool
nextUpperHalfRange(VA addr, VA *start, VA *end)
{
  int num;
  get_mmapped_list_fptr_t getMmappedList =
    (get_mmapped_list_fptr_t)lh_info->mmap_list_fptr;
  // Reads the live list, which includes the memory that DMTCP maps while it
  // writes the checkpoint image.
  const std::vector<MmapInfo_t> &regions = getMmappedList(&num);
  // The list is sorted and disjoint, and touching regions are merged.
  auto it = std::partition_point(regions.begin(), regions.end(),
                                 [addr](const MmapInfo_t &r) {
                                   return r.addr + r.len <= addr;
                                 });
  bool found = (it != regions.end());
  if (found) {
    *start = it->addr;
    *end = it->addr + it->len;
  }
  // The lower half maps the stack before its mmap wrapper exists.
  if (lh_info->uh_stack_end > addr &&
      (!found || lh_info->uh_stack_start < *start)) {
    *start = lh_info->uh_stack_start;
    *end = lh_info->uh_stack_end;
    found = true;
  }
  return found;
}

// Decides that the checkpoint saves the upper half's memory and skips the
// lower half's memory.
EXTERNC int
dmtcp_skip_memory_region_ckpting(const ProcMapsArea *area, VA *uncheckedAddr)
{
  VA start;
  VA end;
  if (!nextUpperHalfRange(area->addr, &start, &end) || start >= area->endAddr) {
    *uncheckedAddr = area->endAddr;
    return 1;
  }
  if (start > area->addr) {
    *uncheckedAddr = start;
    return 1;
  }
  *uncheckedAddr = std::min(end, area->endAddr);

  // The next definition may skip some of this range.
  __typeof__(&dmtcp_skip_memory_region_ckpting) next =
    NEXT_FNC(dmtcp_skip_memory_region_ckpting);
  if (next == NULL) {
    return 0;
  }
  ProcMapsArea range = *area;
  range.endAddr = *uncheckedAddr;
  range.size = range.endAddr - range.addr;
  return next(&range, uncheckedAddr);
}

#define BT_SIZE 50 /* Maximum size backtrace of stack */

// TODO(kapil): Replace with Jassert::PrintBackrace.
string GetBacktrace()
{
  ostringstream ss;
  void *buffer[BT_SIZE];
  int nptrs = backtrace(buffer, BT_SIZE);

  ss << "Backtrace:\n";
  char buf[1024];

  for (int i = 1; i < nptrs; i++) {
    Dl_info info;
    ss << "   " << i << ' ' ;
    if (dladdr1(buffer[i], &info, NULL, 0)) {
      int status;
      size_t buflen = sizeof(buf);
      buf[0] = '\0';
      if (info.dli_sname) {
        //  WAS: 'char *demangled =' (but compiler issues 'not used' warning.
        abi::__cxa_demangle(info.dli_sname, buf, &buflen, &status);
        if (status != 0) {
          strncpy(buf, info.dli_sname, sizeof(buf) - 1);
        }
      }

      ss << (buf[0] ? buf : "") << " in " << info.dli_fname << " ";
    }
    ss << jalib::XToHexString(buffer[i]) << "\n";
  }

  return ss.str();
}

constexpr int MaxSignals = 64;
static struct sigaction userSignalHandlers[MaxSignals] = {0};

void
mana_signal_sa_handler_wrapper(int signum)
{
  unsigned long fsbase = getFS();

  unsigned long uh_fsbase;
  DmtcpMutexLock(&g_upper_half_fsbase_lock);
  pid_t real_tid = dmtcp_get_real_tid();
  std::unordered_map<pid_t, unsigned long>::iterator it= g_upper_half_fsbase->find(real_tid);
  if (it == g_upper_half_fsbase->end()) {
    uh_fsbase = getFS();
    g_upper_half_fsbase->insert(std::make_pair(real_tid, uh_fsbase));
  } else {
    uh_fsbase = it->second;
  }
  DmtcpMutexUnlock(&g_upper_half_fsbase_lock);

  if (fsbase != uh_fsbase) {
    // We are in lower-half; switch to upper half.
    setFS(uh_fsbase);
  }

  userSignalHandlers[signum].sa_handler(signum);

  if (fsbase != uh_fsbase) {
    // go back to lower-half.
    setFS(fsbase);
  }
}

void
mana_signal_sa_sigaction_wrapper(int signum, siginfo_t *siginfo, void *context)
{
  unsigned long fsbase = getFS();
  unsigned long uh_fsbase;
  DmtcpMutexLock(&g_upper_half_fsbase_lock);
  pid_t real_tid = dmtcp_get_real_tid();
  std::unordered_map<pid_t, unsigned long>::iterator it= g_upper_half_fsbase->find(real_tid);
  if (it == g_upper_half_fsbase->end()) {
    uh_fsbase = getFS();
    g_upper_half_fsbase->insert(std::make_pair(real_tid, uh_fsbase));
  } else {
    uh_fsbase = it->second;
  }
  DmtcpMutexUnlock(&g_upper_half_fsbase_lock);

  if (fsbase != uh_fsbase) {
    // We are in lower-half; switch to upper half.
    setFS(uh_fsbase);
  }

  if (signum == SIGTERM) {
    NOTE("SIGTERM received. Exiting.\n{}", GetBacktrace());
    exit(1);
  }

  userSignalHandlers[signum].sa_sigaction(signum, siginfo, context);

  if (fsbase != uh_fsbase) {
    // go back to lower-half.
    setFS(fsbase);
  }
}

EXTERNC int
sigaction(int signum, const struct sigaction *act, struct sigaction *oldAct)
{
  int ret;
  if (signum == dmtcp_get_ckpt_signal() ||
      signum >= MaxSignals) {
    return NEXT_FNC(sigaction)(signum, act, oldAct);
  }

  if (act != NULL) {
    struct sigaction newAct = *act;

    if (newAct.sa_flags & SA_SIGINFO) {
      newAct.sa_sigaction = mana_signal_sa_sigaction_wrapper;
      ret = NEXT_FNC(sigaction)(signum, &newAct, oldAct);
    } else {
      if (newAct.sa_handler == SIG_IGN || newAct.sa_handler == SIG_DFL) {
        return NEXT_FNC(sigaction)(signum, act, oldAct);
      }

      newAct.sa_handler = mana_signal_sa_handler_wrapper;
      ret = NEXT_FNC(sigaction)(signum, &newAct, oldAct);
    }

    if (ret == 0) {
      userSignalHandlers[signum] = *act;
    }
  } else {
    ret = NEXT_FNC(sigaction)(signum, act, oldAct);
  }

  if (oldAct != NULL && ret == 0) {
    *oldAct = userSignalHandlers[signum];
  }

  return ret;
}

EXTERNC sighandler_t
signal(int signum, sighandler_t handler)
{
  sighandler_t ret;
  if (signum == dmtcp_get_ckpt_signal() ||
      signum >= MaxSignals) {
    return NEXT_FNC(signal)(signum, handler);
  }

  if (handler == SIG_IGN || handler == SIG_DFL) {
    ret = NEXT_FNC(signal)(signum, handler);
  } else {
    ret = NEXT_FNC(signal)(signum, mana_signal_sa_handler_wrapper);
  }

  if (ret != SIG_ERR) {
    userSignalHandlers[signum].sa_handler = handler;
  }

  return ret;
}

void
initialize_signal_handlers()
{
  for (int i = 0; i < MaxSignals; i++) {
    NEXT_FNC(sigaction)(i, NULL, &userSignalHandlers[i]);
  }
}

// Handler for SIGSEGV: forces the code into an infinite loop for attaching
// GDB and debugging
void
segvfault_handler(int signum, siginfo_t *siginfo, void *context)
{
  int dummy = 0;
  NOTE("Caught a segmentation fault. Attach gdb to inspect...");
  while (!dummy);
}

// Installs a handler for SIGSEGV; useful for debugging crashes
void
initialize_segv_handler()
{
  const char* segvLoop = getenv(MANA_SEGV_DEBUG_LOOP);
  if (!segvLoop || segvLoop[0] != '1') {
    return;
  }

  static struct sigaction action;
  memset(&action, 0, sizeof(action));
  action.sa_flags = SA_SIGINFO;
  action.sa_sigaction = segvfault_handler;
  sigemptyset(&action.sa_mask);

  ASSERT_ERRNO(sigaction(SIGSEGV, &action, NULL) != -1,
               "Could not set up the segfault handler");
}

const char *
get_mana_header_file_name()
{
  struct stat st;
  const char *ckptDir;
  dmtcp::ostringstream o;

  ckptDir = dmtcp_get_ckpt_dir();
  if (stat(ckptDir, &st) == -1) {
    mkdir(ckptDir, 0700); // Create directory if not already exist
  }
  o << ckptDir << "/header.mana";
  return strdup(o.str().c_str());
}

// After a restart, DMTCP has restored the upper half's fds over the
// placeholders that the lower half kept them free with; close the others.
static void
closeReservedFds()
{
  if (lh_info->reserved_fd_ino == 0) {
    return;
  }
  for (int fd : jalib::Filesystem::ListOpenFds()) {
    struct stat st;
    if (fstat(fd, &st) == 0 && st.st_dev == lh_info->reserved_fd_dev &&
        st.st_ino == lh_info->reserved_fd_ino) {
      close(fd);
    }
  }
  lh_info->reserved_fd_ino = 0;
}

void
save_mana_header(const char *filename)
{
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0755);
  if (fd == -1) {
    return;
  }

  // One "name=value" line per field: the layers' (mana_set_header()), then
  // fds.  mana_restart reads p2p_wait to start the new lower half in the same
  // mode, and fds to keep these fd numbers free for DMTCP's restore
  // (reserve_restart_fds() in the lower half).
  ostringstream o;
  if (header_fields != NULL) {
    for (auto &field : *header_fields) {
      o << field.first << "=" << field.second << "\n";
    }
  }
  o << "fds=";
  const char *sep = "";
  for (int open_fd : jalib::Filesystem::ListOpenFds()) {
    if (open_fd > 2 && open_fd != fd && !dmtcp_is_protected_fd(open_fd)) {
      o << sep << open_fd;
      sep = ",";
    }
  }
  o << "\n";
  string text = o.str();
  write(fd, text.c_str(), text.size());
  close(fd);
}

void printElapsedTime(time_t origin_time, const char *msg) {
  char time_string[30];
  time_t cur_time = time(NULL);
  time_t delta_time = cur_time - origin_time;
  strftime(time_string, sizeof(time_string),
           "%H:%M:%S", localtime(&cur_time));
  if (msg != NULL) {
    fprintf(stderr, "%s: *** MANA: %s\n", time_string, msg);
    if (origin_time == 0) { return; } // We only print msg, not elapsed time
    fprintf(stderr, "%*c      ", (int)strlen(time_string), ' ');
  } else {
    fprintf(stderr, "%s:     ", time_string);
  }
  fprintf(stderr, "Elapsed time since INIT/RESTART: %ld seconds\n", delta_time);
  return;
}

// FIXME:  Use 'getenv("MANA_TIMING")' instead in 'if' stmt.
//         In bin/mana_launch, add '--timing' flag that sets this env. var.
void printEventToStderr(const char *msg) {
  if (!getenv("MANA_TIMING")) { return; }

  static time_t init_time = 0;
  if (init_time == 0 && strstr(msg, "INIT")) {
    init_time = time(NULL);
    printElapsedTime(0, msg);
    return;  // Only one process should print, but we don't yet have a rank.
  }
  if (mana_rank == 0) {
    if (strstr(msg, "RESTART")) {
      init_time = time(NULL);
      printElapsedTime(0, msg);
    } else {
      printElapsedTime(init_time, msg);
    }
  }
}

// Names this rank's checkpoint directory ckpt_rank_<N>.
static void
updateCkptDirByRank()
{
  const char *ckptDir = dmtcp_get_ckpt_dir();
  dmtcp::string baseDir;

  if (strstr(ckptDir, "ckpt_rank_") != NULL) {
    baseDir = jalib::Filesystem::DirName(ckptDir);
  } else {
    baseDir = ckptDir;
  }
  TRACE("Updating checkpoint directory: ckptDir={} baseDir={}", ckptDir,
        baseDir);
  dmtcp::ostringstream o;
  o << baseDir << "/ckpt_rank_" << mana_rank;
  dmtcp_set_ckpt_dir(o.str().c_str());
}

// The launcher's rank of this process, as mana_restart and
// bin/lower-half-nompi read it, or 0 if there is none: a single process.
static int
launcher_rank()
{
  const char *vars[] = {"PMI_RANK", "PMIX_RANK", "SLURM_PROCID"};
  for (const char *var : vars) {
    const char *value = getenv(var);
    if (value != NULL && value[0] >= '0' && value[0] <= '9') {
      return atoi(value);
    }
  }
  return 0;
}

// The layers that have registered (mana_register_layer()), by ManaLayer.
static mana_layer_fn_t layers[MANA_NUM_LAYERS];

extern "C" void
mana_register_layer(int layer, mana_layer_fn_t fn)
{
  if (layer >= 0 && layer < MANA_NUM_LAYERS) {
    __atomic_store_n(&layers[layer], fn, __ATOMIC_RELEASE);
  }
}

extern "C" void
mana_set_rank(int rank)
{
  mana_rank = rank;
}

extern "C" void
mana_set_header(const char *name, const char *value)
{
  if (header_fields == NULL) {
    header_fields = new map<string, string>();
  }
  (*header_fields)[name] = value;
}

// Calls a layer's step, if the layer has registered.
static void
run_layer(int layer, int step)
{
  mana_layer_fn_t fn = __atomic_load_n(&layers[layer], __ATOMIC_ACQUIRE);
  if (fn != NULL) {
    fn(step);
  }
}

// Calls a step of every layer that has registered, in their order, or in
// reverse order.
static void
run_layers(int step, bool reverse = false)
{
  for (int i = 0; i < MANA_NUM_LAYERS; i++) {
    run_layer(reverse ? MANA_NUM_LAYERS - 1 - i : i, step);
  }
}

// --cuda (MANA_CUDA=1): the application's CUDA runs in the lower half, and
// the CUDA front (cuda-front/) saves and rebuilds it.  Whether the front has
// registered or not: every rank must take the CUDA steps' global barriers.
static bool
mana_cuda_enabled()
{
  const char *cuda = getenv("MANA_CUDA");
  return cuda != NULL && cuda[0] != '\0' && strcmp(cuda, "0") != 0;
}

// LOWER_HALF_DISABLE_CKPT() (lower_half_ckpt.h), in an MPI call.
void
mana_cuda_drain_memops()
{
  run_layer(MANA_LAYER_CUDA, MANA_STEP_DRAIN_MEMOPS);
}

extern "C" const char *
mana_ckpt_dir(void)
{
  return dmtcp_get_ckpt_dir();
}

// DMTCP's key-value store, for the front's CUDA IPC.  From the application's
// threads, one caller at a time: they share one socket to the coordinator.
extern "C" int
mana_kv_set(const char *db, const char *key, const char *val)
{
  return dmtcp::kvdb::set(db, key, val) == KVDBResponse::SUCCESS ? 0 : -1;
}

extern "C" int
mana_kv_get(const char *db, const char *key, char *val, size_t size)
{
  dmtcp::string found;
  if (dmtcp::kvdb::get(db, key, &found) != KVDBResponse::SUCCESS) {
    return -1;
  }
  snprintf(val, size, "%s", found.c_str());
  return 0;
}

// "/plugin/MANA/<what>-<start>-<generation>", as ckpt_kvdb_name() in
// seq_num.cpp names the MPI layer's databases.
static string
ckpt_kv_db(const char *what)
{
  DmtcpUniqueProcessId id = dmtcp_get_computation_id();
  char db[64];
  int n = snprintf(db, sizeof db, "/plugin/MANA/%s-%llx-%u", what,
                   (unsigned long long)id._time, id._computation_generation);
  // DMTCP silently truncates a KVDB name at 63 characters.
  ASSERT(n > 0 && (size_t)n < sizeof db, "KVDB name too long: {}", db);
  return db;
}

extern "C" void
mana_ckpt_kv_update(const char *what, const char *key, int op, int64_t val)
{
  KVDBRequest request =
    op == MANA_KV_MAX ? KVDBRequest::MAX : KVDBRequest::INCRBY;
  string db = ckpt_kv_db(what);
  ASSERT(kvdb::request64(request, db, key, val) == KVDBResponse::SUCCESS,
         "db={} key={}", db, key);
}

extern "C" int64_t
mana_ckpt_kv_get(const char *what, const char *key)
{
  int64_t val = 0;
  if (kvdb::get64(ckpt_kv_db(what), key, &val) != KVDBResponse::SUCCESS) {
    return 0;
  }
  return val;
}

extern "C" void
mana_global_barrier(const char *id)
{
  dmtcp_global_barrier(id);
}

static void
mana_event_hook(DmtcpEvent_t event, DmtcpEventData_t *data)
{
  switch (event) {
    case DMTCP_EVENT_INIT: {
      printEventToStderr("EVENT_INIT"); // We don't have a rank. So no printing.
      TRACE("*** DMTCP_EVENT_INIT");
      ASSERT_NOT_NULL(dmtcp_get_real_tid);
      initialize_signal_handlers();
      initialize_segv_handler();
      init_lower_half_ckpt();
      mana_state = RUNNING;
      if (mana_rank == -1) {
        mana_rank = launcher_rank();
      }

      DmtcpMutexInit(&g_upper_half_fsbase_lock, DMTCP_MUTEX_LLL);
      if (g_upper_half_fsbase != NULL) {
        delete g_upper_half_fsbase;
      }

      g_upper_half_fsbase = new std::unordered_map<pid_t, unsigned long>();
      pid_t real_tid = dmtcp_get_real_tid();
      unsigned long fs = getFS();
      g_upper_half_fsbase->insert(std::make_pair(real_tid, fs));

      if (g_file_flags_map != NULL) {
        delete g_file_flags_map;
      }
      g_file_flags_map = new map<string, int>();

      // TODO(kapil): Remove from final commit.
      setenv(MANA_FILE_REGEX_ENV, ".*", 1);

      if (file_regex == NULL && getenv(MANA_FILE_REGEX_ENV) != NULL) {
        file_regex = new std::regex(getenv(MANA_FILE_REGEX_ENV));
      }

      heapAddr = sbrk(0);
      ASSERT_NOT_NULL(heapAddr);
      // FIXME:  If we use PROT_NONE (preferred), then an older DMTCP
      //         will not restore this memory region.
      // By creating a memory page just beyond the end of the heap,
      // this will prevent glibc from extending the main malloc arena.
      // So, glibc will create a second arena.
      // NOTE:  This is needed for mtcp_restart, lh_proxy, and the upper half.
      // Rationale:  On restart, the end-of-heap is not here, but at
      //             the end-of-heap for mtcp_restart.  Luckily, glibc
      //             caches 'sbrk(0)', which is here.  So, glibc should
      //             detect that there is an mmap'ed region just beyond it,
      //             thus causing glibc to allocate a second arena elsewhere.
      mmap(heapAddr, 4096, PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);

      //set global flag to indiacte MANA is initialized
      // FIXME: Move the code from PMPI_Init and PMPI_init_thread wrappers to
      //        This section before setting the Flag for MANA's initialization
      g_libmana_is_initialized = true;
      break;
    }
    case DMTCP_EVENT_EXIT: {
      run_layers(MANA_STEP_EXIT, true);
      printEventToStderr("EVENT_EXIT");
      TRACE("*** DMTCP_EVENT_EXIT");
      break;
    }

    case DMTCP_EVENT_OPEN_FD: {
      processFileOpen(data->openFd.path, data->openFd.flags);
      break;
    }

    case DMTCP_EVENT_PTHREAD_START: {
      fprintf(stderr, "PTHREAD_START event triggered\n");
      // Do we need a mutex here?
      DmtcpMutexLock(&g_upper_half_fsbase_lock);
      pid_t real_tid = dmtcp_get_real_tid();
      unsigned long fs = getFS();
      g_upper_half_fsbase->insert(std::make_pair(real_tid, fs));
      DmtcpMutexUnlock(&g_upper_half_fsbase_lock);
      break;
    }

    case DMTCP_EVENT_PTHREAD_RETURN:
    case DMTCP_EVENT_PTHREAD_EXIT: {
      // Under DMTCP's wrapper lock: no checkpoint can come meanwhile.
      run_layers(MANA_STEP_THREAD_EXIT, true);
      unregister_lower_half_thread();
      // Do we need a mutex here?
      DmtcpMutexLock(&g_upper_half_fsbase_lock);
      pid_t real_tid = dmtcp_get_real_tid();
      g_upper_half_fsbase->erase(real_tid);
      DmtcpMutexUnlock(&g_upper_half_fsbase_lock);
      break;
    }

    case DMTCP_EVENT_PRESUSPEND: {
      printEventToStderr("EVENT_PRESUSPEND (finish collective op's)");
      run_layers(MANA_STEP_WAIT_FOR_INIT);
      mana_state = CKPT_COLLECTIVE;  // Not RUNNING: the checkpoint has begun.
      run_layers(MANA_STEP_DRAIN, true);
      openCkptFileFds();
      // No thread may be in the lower half when DMTCP suspends the threads.
      // The MPI layer's drain has closed it already.
      if (!__atomic_load_n(&lower_half_closed, __ATOMIC_ACQUIRE)) {
        wait_for_threads_to_leave_lower_half();  // lower_half_ckpt.cpp
      }
      if (mana_cuda_enabled()) {
        // The lower half is closed, so no thread starts GPU work any more:
        // wait for the GPUs, then copy device memory into the upper half.
        // Until every rank has synchronized, another rank's kernel may
        // still write this rank's memory through CUDA IPC.
        run_layers(MANA_STEP_QUIESCE);
        dmtcp_global_barrier("MANA:cuda-quiesced");
        run_layers(MANA_STEP_STASH);
      }
      printEventToStderr("EVENT_PRESUSPEND (done)");
      break;
    }

    case DMTCP_EVENT_PRECHECKPOINT: {
      printEventToStderr("EVENT_PRECHECKPOINT (drain send/recv)");
      // The threads are suspended now; let them back into the lower half
      // when they resume.
      allow_threads_to_enter_lower_half();
      recordOpenFds();
      run_layers(MANA_STEP_PRECHECKPOINT);
      dmtcp_global_barrier("MANA:update-ckpt-dir-by-rank");
      updateCkptDirByRank();
      // computeUnionOfCkptImageAddresses();
      dmtcp_global_barrier("MANA:save-mana-header");
      const char *file = get_mana_header_file_name();
      save_mana_header(file);
      run_layers(MANA_STEP_SAVE);
      printEventToStderr("EVENT_PRECHECKPOINT (done)");
      // Save a copy of the break address before checkpoint
      old_brk = sbrk(0);
      uh_stack_start = lh_info->uh_stack_start;
      uh_stack_end = lh_info->uh_stack_end;
      break;
    }

    case DMTCP_EVENT_RESUME: {
      printEventToStderr("EVENT_RESUME");
      processingOpenCkpFileFds = false;
      run_layers(MANA_STEP_RESUME, true);
      mana_state = RUNNING;
      printEventToStderr("EVENT_RESUME (done)");
      break;
    }

    case DMTCP_EVENT_RESTART: {
      reset_wrappers();
      initialize_wrappers();
      lh_info->uh_stack_start = uh_stack_start;
      lh_info->uh_stack_end = uh_stack_end;
      printEventToStderr("EVENT_RESTART");
      processingOpenCkpFileFds = false;
      logCkptFileFds();

      g_upper_half_fsbase->clear();
      g_upper_half_fsbase->insert(std::make_pair(dmtcp_get_real_tid(), getFS()));

      run_layers(MANA_STEP_RESTART);
      if (mana_cuda_enabled()) {
        // The application's CUDA IPC and NCCL: every rank exports again,
        // and makes new NCCL unique ids, before any maps a peer's memory or
        // joins a communicator.
        dmtcp_global_barrier("MANA:cuda-ipc-exported");
        run_layers(MANA_STEP_RESTART_PEERS);
      }
      // Device memory is back before the MPI replay reposts receives into
      // it.
      run_layers(MANA_STEP_REPLAY);
      mana_state = RUNNING;
      printEventToStderr("EVENT_RESTART (done)");
      break;
    }

    case DMTCP_EVENT_RUNNING: {
      closeCkptFileFds();
      closeReservedFds();
      break;
    }

    case DMTCP_EVENT_THREAD_RESUME: {
      DmtcpMutexLock(&g_upper_half_fsbase_lock);
      g_upper_half_fsbase->insert(std::make_pair(dmtcp_get_real_tid(), getFS()));
      DmtcpMutexUnlock(&g_upper_half_fsbase_lock);
      break;
    }

    default: {
      break;
    }
  }
}

DmtcpPluginDescriptor_t mana_plugin = {
  DMTCP_PLUGIN_API_VERSION,
  PACKAGE_VERSION,
  "mana",
  "DMTCP",
  "dmtcp@ccs.neu.edu",
  "MANA's split process",
  mana_event_hook
};

DMTCP_DECL_PLUGIN(mana_plugin);
