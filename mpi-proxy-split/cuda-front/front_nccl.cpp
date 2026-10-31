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

// NCCL (cuda/PLAN-nccl-front.md), in the lower half with its net plugin,
// libfabric and GDRCopy, as cuFFT is.  The application's ncclComm_t is the
// address of the communicator's record.  A record has what a restart needs
// to make the communicator again: its name, rank, size and device, and its
// config; and what the drain needs: the steps posted on it, and the bytes
// sent to and received from each peer.
//
// The drain (front_nccl_drain(), at MANA_STEP_DRAIN, before MPI's drain
// closes the lower half).  NCCL's kernels wait for every member of their
// communicator, so the GPUs quiesce only once the members have posted the
// same operations.  As in netsplit's drain (~/nccl-mana-ckpt, branch
// netsplit: nccl_wrappers.c and libncclplugin.c), the ranks agree on how
// many steps each communicator takes and how many bytes each pair sends;
// a rank that is behind lets its threads run until it is there, and the
// others' threads wait at their next NCCL call (the gate).  This assumes
// that the program does not interleave NCCL and MPI: a rank held at the
// gate before NCCL operation k+1 never reaches an MPI_Send that a peer
// waits for before it posts k.
//
// A resume changes nothing in the lower half, so the communicators stay.  A
// restart makes each again by ncclCommInitRank(Config), a split too (as
// netsplit does), with a new unique id: the old one names a bootstrap root,
// a thread of the old lower half.  Rank 0 of each communicator makes the new
// id and publishes it under the communicator's name before a global barrier
// (front_nccl_publish()); after it, every member makes the communicator
// again (front_nccl_rebuild()).  A unique id that this process made and has
// not used is made again too, for the ranks that call ncclCommInitRank with
// it after the restart.  Then what the program made on each: its buffer
// registrations, windows and PreMulSum operators; and a communicator that
// was suspended is suspended again.  The application's handles of these are
// the front's records, and its operators keep their values.

#include <nccl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>
#include "front_objs.h"
#include "front_nccl.h"
#include "gen/nccl-index.h"

#define NCCL(f) ((decltype(&::f))front_nccl_lib.real[FRONT_IDX_nccl_##f])

#define NCCL_LH(stmt)                   \
  do {                                  \
    front_check_bound(&front_nccl_lib); \
    FRONT_LH(stmt);                     \
  } while (0)

namespace {

enum Made {
  INIT_RANK,             // ncclCommInitRank(Config, Scalable)
  INIT_ALL,              // ncclCommInitAll: one record per device
  SPLIT,                 // ncclCommSplit
  SHRINK                 // ncclCommShrink
};

// Point-to-point traffic with each peer, in one unit: the bytes of ncclSend
// and ncclRecv, or the signals of ncclPutSignal and ncclSignal and the
// waits for them (ncclWaitSignal).  out[peer], what this rank has posted to
// the peer; in[peer], from it.  The targets are what the peers have
// published: out_target, what each has posted to take from this rank;
// in_target, to give it.
enum Kind { BYTES, SIGNALS, KINDS };

struct Traffic {
  std::vector<int64_t> out;
  std::vector<int64_t> in;
  std::vector<int64_t> out_target;
  std::vector<int64_t> in_target;
};

struct Comm;

// A buffer registration (ncclCommRegister), whose address is the
// application's handle.  real is NULL once the communicator is not made
// again.
struct Reg {
  Comm *c;
  void *buff;
  size_t size;
  void *real;
};

// A window (ncclCommWindowRegister), whose address is the application's
// ncclWindow_t; real is NCCL's (NCCL 2.27 or later).
struct Win {
  Comm *c;
  void *buff;
  size_t size;
  int flags;
  void *real;
};

// A PreMulSum operator.  Its value for the application is NCCL's when it was
// made, unless another operator of the communicator has that value now:
// NCCL's changes when a restart makes it again.  The scalar is copied if
// NCCL reads it at once, else it is the device address, which stays.
struct Op {
  ncclRedOp_t value;
  ncclRedOp_t real;
  ncclDataType_t datatype;
  ncclScalarResidence_t residence;
  void *scalar;
  uint64_t immediate;
};

struct Comm : FrontObj {
  Ctx *ctx;
  Made made;
  int nranks;
  int rank;
  int device;
  // INIT_RANK: the names of its unique ids, which are used now.
  std::vector<uint64_t> ids;
  int color;             // SPLIT
  int key;
  // SPLIT, SHRINK, until NCCL has made it (wait_made()).
  ncclComm_t parent_real;
  int clique;            // INIT_ALL: the call's number
  // The config, for the restart; a split without one has its parent's, as
  // NCCL gives it.
  bool has_config;
  ncclConfig_t config;
  std::string net_name;  // the config's strings, which are the application's
  std::string comm_name;
  bool nonblocking;      // config.blocking == 0
  // Not made again at a restart (real is NULL then): the application may
  // only destroy it.
  bool finalized;
  // Its name in the drain, the same in every member: a digest of the unique
  // ids; of the parent's name, the split's number on the parent and the
  // color; or of the parent's name, the ranks that a shrink left out and
  // its number among the parent's shrinks without them.  A clique of
  // ncclCommInitAll is in this process alone.
  uint64_t ident;
  int splits;            // the splits of this one so far
  // The shrinks of this one so far: a digest of the ranks left out, and how
  // many.
  std::vector<std::pair<uint64_t, int>> shrinks;
  // Under nccl_lock.  A step is a collective, a split, a window's
  // registration, or the first of ncclCommFinalize and ncclCommDestroy.  The
  // targets are what the members have published: target, the steps (-1:
  // not known yet), and each kind of traffic's.
  int64_t seq;
  int64_t target;
  Traffic traffic[KINDS];
  // Under nccl_lock: what the program made on it.
  std::vector<Reg *> regs;
  std::vector<Win *> wins;
  std::vector<Op> ops;
  bool suspended;        // by ncclCommSuspend, with these flags
  int suspend_flags;
  // Destroyed: kept for the drain, with its last step, until a drain
  // converges, when every member has destroyed it too.
  bool destroyed;
};

// The drain and the gate.  Never held with objs_lock.
std::mutex nccl_lock;
std::condition_variable gate_cv;
std::vector<Comm *> comms;     // in creation order
// From front_nccl_drain() to the resume or the restart, which clear it
// without the lock (see enter()).
std::atomic<bool> draining;
bool frozen;                   // settled in the drain's last round
bool creation_pending;         // a rank had a creation in flight at the vote
int activity;                  // threads in a call, or in a group region
int creating;                  // creations by unique id in flight here
int cliques;
// The names of the unique ids that this process made and has not used.
std::vector<uint64_t> unused_ids;

// The application's thread: its depth in group regions; the communicators
// and windows that NCCL makes at the region's end, a window into the
// application's variable; and what the region's calls added to the counts,
// which ncclGroupSimulateEnd takes back.
__thread int group_depth;
__thread std::vector<Comm *> *group_made;
__thread std::vector<std::pair<Win *, void **>> *group_wins;
__thread std::vector<std::pair<int64_t *, int64_t>> *group_counts;

const char *const KV = "nccl";  // the drain's database (mana_core.h)

uint64_t
digest(uint64_t h, const void *data, size_t size)
{
  const unsigned char *p = (const unsigned char *)data;
  for (size_t i = 0; i < size; i++) {
    h = (h ^ p[i]) * 0x100000001b3ULL;  // FNV-1a
  }
  return h;
}

const uint64_t DIGEST_START = 0xcbf29ce484222325ULL;

Comm *
comm_of(ncclComm_t comm)
{
  if (comm == NULL) {
    return NULL;
  }
  front_real_obj(comm, FRONT_NCCL);  // checks the handle
  return (Comm *)comm;
}

// Whether this communicator's members wait for this rank: a step that they
// have posted, or traffic.
bool
owes_locked(const Comm *c)
{
  if (c->target >= 0 && c->seq < c->target) {
    return true;
  }
  for (const Traffic &t : c->traffic) {
    for (size_t peer = 0; peer < t.out.size(); peer++) {
      if (t.out[peer] < t.out_target[peer] ||
          t.in[peer] < t.in_target[peer]) {
        return true;
      }
    }
  }
  return false;
}

bool
behind_locked()
{
  for (Comm *c : comms) {
    if (!c->destroyed && owes_locked(c)) {
      return true;
    }
  }
  return false;
}

// The gate holds a call during a drain unless this rank is behind, or a
// creation is in flight somewhere: a unique id names no members, so nobody
// knows who must still join it.  A rank without a communicator is not held
// either, since its call makes one, with peers that may be in the
// rendezvous already.  Once this rank settles, nothing passes until the
// next round.
bool
gate_closed_locked()
{
  if (!draining) {
    return false;
  }
  if (frozen) {
    return true;
  }
  if (creation_pending) {
    return false;
  }
  bool live = false;
  for (Comm *c : comms) {
    if (c->destroyed) {
      continue;
    }
    if (owes_locked(c)) {
      return false;
    }
    live = true;
  }
  return live;
}

// What a call adds to the counts: n to *at (a step, bytes or signals).
struct Count {
  int64_t *at;
  int64_t n;
};

// Before a call that posts on a communicator or makes one: waits at the
// gate, except in a group region, whose ncclGroupStart passed it.  Then
// counts the call in activity, and adds its counts, under the same lock:
// the drain sees a call either before the gate or posted.  create: a
// creation that every member must join (by unique id, or a shrink), in
// flight until leave().
void
enter_counts(const Count *counts, size_t ncounts, bool create)
{
  std::unique_lock<std::mutex> lock(nccl_lock);
  if (group_depth == 0) {
    // The resume and the restart open the gate without the lock, which a
    // thread that DMTCP suspended here may hold; so a thread may miss
    // their notify, and looks again now and then.
    while (gate_closed_locked()) {
      gate_cv.wait_for(lock, std::chrono::milliseconds(50));
    }
  }
  for (size_t i = 0; i < ncounts; i++) {
    if (counts[i].at != NULL) {
      *counts[i].at += counts[i].n;
    }
  }
  activity++;
  if (create) {
    creating++;
  }
}

// After the call; a call that failed takes back what enter_counts() added.
// In a group region, a creation stays in flight until the region ends, and
// the counts are kept for ncclGroupSimulateEnd.
void
leave_counts(bool ok, const Count *counts, size_t ncounts, bool create)
{
  std::lock_guard<std::mutex> guard(nccl_lock);
  for (size_t i = 0; i < ncounts; i++) {
    if (counts[i].at == NULL) {
      continue;
    }
    if (!ok) {
      *counts[i].at -= counts[i].n;
    } else if (group_depth > 0) {
      if (group_counts == NULL) {
        group_counts = new std::vector<std::pair<int64_t *, int64_t>>();
      }
      group_counts->push_back({ counts[i].at, counts[i].n });
    }
  }
  activity--;
  if (create && (group_depth == 0 || !ok)) {
    creating--;
  }
}

void
enter(int64_t *count, int64_t n, bool create)
{
  Count one = { count, n };
  enter_counts(&one, 1, create);
}

void
leave(bool ok, int64_t *count, int64_t n, bool create)
{
  Count one = { count, n };
  leave_counts(ok, &one, 1, create);
}

bool
posted(ncclResult_t rc)
{
  return rc == ncclSuccess || rc == ncclInProgress;
}

// A new communicator's record, once NCCL has made it (c->real).  Its size,
// rank and device are -1 where NCCL has them: a split's, or a non-blocking
// communicator's, which it has only once it is made.
void
link_comm(Comm *c)
{
  ncclComm_t real = (ncclComm_t)c->real;
  if (c->nranks < 0) {
    NCCL_LH(NCCL(ncclCommCount)(real, &c->nranks));
  }
  if (c->rank < 0) {
    NCCL_LH(NCCL(ncclCommUserRank)(real, &c->rank));
  }
  if (c->device < 0) {
    NCCL_LH(NCCL(ncclCommCuDevice)(real, &c->device));
  }
  c->ctx = front_object_ctx();
  c->parent_real = NULL;
  c->target = -1;
  size_t n = c->nranks > 0 ? c->nranks : 0;
  for (Traffic &t : c->traffic) {
    t.out.assign(n, 0);
    t.in.assign(n, 0);
    t.out_target.assign(n, 0);
    t.in_target.assign(n, 0);
  }
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    obj_link(c, FRONT_NCCL);
  }
  std::lock_guard<std::mutex> guard(nccl_lock);
  comms.push_back(c);
  // Its members have met at its unique ids' bootstrap roots.
  for (uint64_t ident : c->ids) {
    unused_ids.erase(std::remove(unused_ids.begin(), unused_ids.end(),
                                 ident),
                     unused_ids.end());
  }
}

// A non-blocking split returns ncclInProgress, and NCCL writes the new
// communicator later, from the job that makes it.  The front waits for it,
// and for it to be ready, since the record needs its size and rank: such a
// split blocks.  A non-blocking ncclCommInitRankConfig does not: NCCL
// writes its communicator at once, and the record knows the rest.  False
// if NCCL failed to make it.
bool
wait_made(Comm *c)
{
  auto start = std::chrono::steady_clock::now();
  for (;; usleep(100)) {
    ncclComm_t real = (ncclComm_t)__atomic_load_n(&c->real, __ATOMIC_ACQUIRE);
    ncclResult_t state = ncclInProgress;
    ncclResult_t rc = ncclSuccess;
    if (real != NULL) {
      NCCL_LH(rc = NCCL(ncclCommGetAsyncError)(real, &state));
      if (rc != ncclSuccess) {
        return false;
      }
      if (state != ncclInProgress) {
        return state == ncclSuccess;
      }
      continue;
    }
    if (c->parent_real != NULL) {
      NCCL_LH(rc = NCCL(ncclCommGetAsyncError)(c->parent_real, &state));
      if (rc != ncclSuccess ||
          (state != ncclSuccess && state != ncclInProgress)) {
        return false;
      }
    }
    // The job is done, and has written no communicator.
    if (state == ncclSuccess && std::chrono::steady_clock::now() - start >
                                  std::chrono::seconds(10)) {
      return false;
    }
  }
}

bool
pending(const Comm *c, ncclResult_t rc)
{
  return c->real == NULL ||
         (rc == ncclInProgress && (c->made == SPLIT || c->made == SHRINK));
}

// A creation's record: linked now, or at the end of the group region it is
// in, when NCCL makes the communicator.  NULL if it failed.
ncclComm_t
made_comm(Comm *c, ncclResult_t rc)
{
  if (group_depth > 0 && rc == ncclSuccess) {
    if (group_made == NULL) {
      group_made = new std::vector<Comm *>();
    }
    group_made->push_back(c);
    return (ncclComm_t)c;
  }
  if (!posted(rc)) {
    delete c;
    return NULL;
  }
  if (pending(c, rc) && !wait_made(c)) {
    return NULL;  // not deleted: NCCL may write into it yet
  }
  link_comm(c);
  return (ncclComm_t)c;
}

void win_made(Win *w, void **app);
bool wait_made(Comm *c);

// The outermost ncclGroupEnd, or ncclGroupSimulateEnd (simulated), which
// runs nothing: its calls' counts are taken back, and what they would have
// made is not.  The region's creations by unique id or shrink are no longer
// in flight; the others were steps on their parent.
void
end_group(ncclResult_t rc, bool simulated)
{
  std::vector<Comm *> *made = group_made;
  group_made = NULL;
  std::vector<std::pair<Win *, void **>> *wins = group_wins;
  group_wins = NULL;
  std::vector<std::pair<int64_t *, int64_t>> *counts = group_counts;
  group_counts = NULL;
  if (simulated) {
    rc = ncclInternalError;  // nothing made
  }
  int joined = 0;
  if (made != NULL) {
    for (Comm *c : *made) {
      joined += c->made == INIT_RANK || c->made == SHRINK;
      if (!posted(rc)) {
        delete c;
      } else if (!pending(c, rc) || wait_made(c)) {
        link_comm(c);
      }  // else not deleted: NCCL may write into it yet
    }
    delete made;
  }
  if (wins != NULL) {
    for (auto &w : *wins) {
      // A non-blocking communicator's window is there once it is ready.
      if (posted(rc) && (rc == ncclSuccess || wait_made(w.first->c))) {
        win_made(w.first, w.second);
      } else {
        delete w.first;
      }
    }
    delete wins;
  }
  std::lock_guard<std::mutex> guard(nccl_lock);
  if (counts != NULL) {
    if (simulated) {
      for (auto &count : *counts) {
        *count.first -= count.second;
      }
    }
    delete counts;
  }
  creating -= joined;
  activity--;
}

void
keep_config(Comm *c, const ncclConfig_t *config)
{
  c->has_config = config != NULL;
  if (config == NULL) {
    return;
  }
  c->config = *config;
  c->net_name = config->netName != NULL ? config->netName : "";
  c->comm_name = config->commName != NULL ? config->commName : "";
  c->nonblocking = config->blocking == 0;
}

// A split's or a shrink's config: its own, else its parent's, as NCCL gives
// it.
void
child_config(Comm *c, const Comm *parent, const ncclConfig_t *config)
{
  keep_config(c, config);
  if (config == NULL && parent != NULL) {
    c->has_config = parent->has_config;
    c->config = parent->config;
    c->net_name = parent->net_name;
    c->comm_name = parent->comm_name;
    c->nonblocking = parent->nonblocking;
  }
}

bool
custom_op(ncclRedOp_t op)
{
  return (int)op >= (int)ncclNumOps;
}

// NCCL's value of an operator of the communicator.
ncclRedOp_t
real_op(Comm *c, ncclRedOp_t op)
{
  if (c == NULL || !custom_op(op)) {
    return op;
  }
  std::lock_guard<std::mutex> guard(nccl_lock);
  for (const Op &o : c->ops) {
    if (o.value == op) {
      return o.real;
    }
  }
  return op;
}

bool
op_used_locked(const Comm *c, ncclRedOp_t value)
{
  for (const Op &o : c->ops) {
    if (o.value == value) {
      return true;
    }
  }
  return false;
}

// The record of the communicator's window that the application holds as
// win, if it is one.
Win *
win_of(Comm *c, const void *win)
{
  if (c == NULL || win == NULL) {
    return NULL;
  }
  std::lock_guard<std::mutex> guard(nccl_lock);
  for (Win *w : c->wins) {
    if (w == win) {
      return w;
    }
  }
  return NULL;
}

// NCCL's window for the application's.
void *
real_win(Comm *c, void *win)
{
  Win *w = win_of(c, win);
  return w != NULL ? w->real : win;
}

// A window that NCCL has written into the application's variable *app: the
// application gets the record instead.  NCCL makes none across nodes, and
// writes NULL, which the application then sees.
void
win_made(Win *w, void **app)
{
  w->real = *app;
  if (w->real == NULL) {
    delete w;
    return;
  }
  {
    std::lock_guard<std::mutex> guard(nccl_lock);
    w->c->wins.push_back(w);
  }
  *app = w;
}

// Where a call's traffic with a peer is counted.
int64_t *
traffic_count(Comm *c, Kind kind, bool out, int peer)
{
  if (c == NULL) {
    return NULL;
  }
  Traffic &t = c->traffic[kind];
  std::vector<int64_t> &v = out ? t.out : t.in;
  if (peer < 0 || peer >= (int)v.size()) {
    return NULL;
  }
  return &v[peer];
}

// What the program made on a communicator, which goes with it.
void
forget_made_locked(Comm *c)
{
  for (Reg *r : c->regs) {
    delete r;
  }
  c->regs.clear();
  for (Win *w : c->wins) {
    delete w;
  }
  c->wins.clear();
  c->ops.clear();
}

// The communicators are not in the restart's replay of the records: they
// are made again after a global barrier (front_nccl_rebuild()).
void
nccl_settle()
{
}

void
nccl_forget(Ctx *ctx)
{
  (void)ctx;
}

const FrontLibRecords nccl_records = { NULL, nccl_settle, nccl_forget, NULL,
                                       NULL };

__attribute__((constructor)) void
register_records()
{
  front_lib_records(FRONT_NCCL, &nccl_records);
}

size_t
type_size(ncclDataType_t type)
{
  switch (type) {
    case ncclInt8:
    case ncclUint8:
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 24, 0)
    case ncclFloat8e4m3:
    case ncclFloat8e5m2:
#endif
      return 1;
    case ncclFloat16:
#if defined(__CUDA_BF16_TYPES_EXIST__) || NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0)
    case ncclBfloat16:
#endif
      return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      return 4;
    default:
      return 8;
  }
}

// ---- The drain, on the checkpoint thread ----

// A communicator as the drain publishes it.
struct Published {
  uint64_t ident;
  int rank;
  int64_t seq;
  std::vector<int64_t> out[KINDS];
  std::vector<int64_t> in[KINDS];
};

// What the members published in the last round, for one communicator.
struct Read {
  Comm *c;
  int64_t seq;           // the largest step, plus 1; 0 if none
  std::vector<int64_t> out_target[KINDS];
  std::vector<int64_t> in_target[KINDS];
};

// Keys in KV: s/<ident>, the members' largest step plus 1.  For bytes:
// q/<ident>, 1 if a member has sent or received; p/<ident>/<src>/<dst>, the
// bytes that src has posted to send to dst; r/<ident>/<src>/<dst>, the bytes
// that dst has posted to receive from src.  For signals, the same with z, x
// and y: the signals that src has posted to dst, and the ones dst has
// posted to wait for from src.  Values only grow (MANA_KV_MAX).
const char USED_KEY[KINDS] = { 'q', 'z' };
const char OUT_KEY[KINDS] = { 'p', 'x' };
const char IN_KEY[KINDS] = { 'r', 'y' };

void
key(char *buf, size_t size, char kind, uint64_t ident, int src = -1,
    int dst = -1)
{
  if (src < 0) {
    snprintf(buf, size, "%c/%016llx", kind, (unsigned long long)ident);
  } else {
    snprintf(buf, size, "%c/%016llx/%d/%d", kind, (unsigned long long)ident,
             src, dst);
  }
}

// Step 1 of a round: the targets that the members published in the last
// one.  A clique of ncclCommInitAll takes the furthest of its own.
void
adopt_targets()
{
  std::vector<Read> reads;
  {
    std::lock_guard<std::mutex> guard(nccl_lock);
    for (Comm *c : comms) {
      if (!c->destroyed && c->made != INIT_ALL) {
        reads.push_back({ c, 0, {}, {} });
      }
    }
  }
  char k[96];
  for (Read &r : reads) {
    Comm *c = r.c;  // its identity, size and rank do not change
    key(k, sizeof k, 's', c->ident);
    r.seq = mana_ckpt_kv_get(KV, k);
    for (int kind = 0; kind < KINDS; kind++) {
      key(k, sizeof k, USED_KEY[kind], c->ident);
      if (mana_ckpt_kv_get(KV, k) == 0) {
        continue;
      }
      r.out_target[kind].assign(c->nranks, 0);
      r.in_target[kind].assign(c->nranks, 0);
      for (int peer = 0; peer < c->nranks; peer++) {
        if (peer == c->rank) {
          continue;
        }
        key(k, sizeof k, IN_KEY[kind], c->ident, c->rank, peer);
        r.out_target[kind][peer] = mana_ckpt_kv_get(KV, k);
        key(k, sizeof k, OUT_KEY[kind], c->ident, peer, c->rank);
        r.in_target[kind][peer] = mana_ckpt_kv_get(KV, k);
      }
    }
  }
  std::lock_guard<std::mutex> guard(nccl_lock);
  for (Read &r : reads) {
    Comm *c = r.c;
    if (r.seq > 0) {
      c->target = std::max(c->target, r.seq - 1);
    }
    for (int kind = 0; kind < KINDS; kind++) {
      Traffic &t = c->traffic[kind];
      for (size_t peer = 0; peer < r.out_target[kind].size(); peer++) {
        t.out_target[peer] = std::max(t.out_target[peer],
                                      r.out_target[kind][peer]);
        t.in_target[peer] = std::max(t.in_target[peer],
                                     r.in_target[kind][peer]);
      }
    }
  }
  for (Comm *c : comms) {
    if (c->destroyed || c->made != INIT_ALL) {
      continue;
    }
    for (Comm *d : comms) {
      if (!d->destroyed && d->made == INIT_ALL && d->clique == c->clique) {
        c->target = std::max(c->target, d->seq);
      }
    }
  }
  gate_cv.notify_all();
}

// Non-blocking communicators whose creation or operations are still in
// progress.
int
in_progress()
{
  std::vector<ncclComm_t> reals;
  {
    std::lock_guard<std::mutex> guard(nccl_lock);
    for (Comm *c : comms) {
      if (!c->destroyed && c->nonblocking && c->real != NULL) {
        reals.push_back((ncclComm_t)c->real);
      }
    }
  }
  int n = 0;
  for (ncclComm_t real : reals) {
    ncclResult_t state = ncclSuccess;
    ncclResult_t rc;
    NCCL_LH(rc = NCCL(ncclCommGetAsyncError)(real, &state));
    n += rc == ncclSuccess && state == ncclInProgress;
  }
  return n;
}

// Whether this rank has caught up with the targets and has nothing in
// flight.
bool
quiet()
{
  bool busy;
  {
    std::lock_guard<std::mutex> guard(nccl_lock);
    busy = activity > 0 || creating > 0 || behind_locked();
  }
  return !busy && in_progress() == 0;
}

// Step 3: what this rank publishes, and whether it is settled: at its
// targets, balanced with each peer, and nothing in flight.  Decided under
// the lock, which then closes the gate until the next round, so that no
// call slips through between the decision and the vote.  While a creation
// was in flight anywhere at the last vote, no rank settles: the gate stays
// open the whole round, for a rank that must still join the creation.
bool
settle(int progress, std::vector<Published> *out, bool *behind,
       int *in_flight)
{
  std::lock_guard<std::mutex> guard(nccl_lock);
  bool settled = !creation_pending && activity == 0 && creating == 0 &&
                 progress == 0;
  for (Comm *c : comms) {
    if (c->made == INIT_ALL) {
      settled = settled && (c->destroyed || c->seq == c->target);
      continue;
    }
    Published p = { c->ident, c->rank, c->seq, {}, {} };
    for (int kind = 0; kind < KINDS; kind++) {
      p.out[kind] = c->traffic[kind].out;
      p.in[kind] = c->traffic[kind].in;
    }
    out->push_back(std::move(p));
    if (c->destroyed) {
      continue;
    }
    if (c->target < 0 || c->seq != c->target) {
      settled = false;
    }
    for (const Traffic &t : c->traffic) {
      for (size_t peer = 0; peer < t.out.size(); peer++) {
        if (t.out[peer] != t.out_target[peer] ||
            t.in[peer] != t.in_target[peer]) {
          settled = false;
        }
      }
    }
  }
  *behind = behind_locked();
  *in_flight = creating + progress;
  if (settled) {
    frozen = true;
  }
  return settled;
}

void
publish(const std::vector<Published> &pubs)
{
  char k[96];
  for (const Published &p : pubs) {
    key(k, sizeof k, 's', p.ident);
    mana_ckpt_kv_update(KV, k, MANA_KV_MAX, p.seq + 1);
    for (int kind = 0; kind < KINDS; kind++) {
      bool used = false;
      for (size_t peer = 0; peer < p.out[kind].size(); peer++) {
        if (p.out[kind][peer] > 0) {
          key(k, sizeof k, OUT_KEY[kind], p.ident, p.rank, peer);
          mana_ckpt_kv_update(KV, k, MANA_KV_MAX, p.out[kind][peer]);
          used = true;
        }
        if (p.in[kind][peer] > 0) {
          key(k, sizeof k, IN_KEY[kind], p.ident, peer, p.rank);
          mana_ckpt_kv_update(KV, k, MANA_KV_MAX, p.in[kind][peer]);
          used = true;
        }
      }
      if (used) {
        key(k, sizeof k, USED_KEY[kind], p.ident);
        mana_ckpt_kv_update(KV, k, MANA_KV_MAX, 1);
      }
    }
  }
}

// The steps and the point-to-point bytes that this rank has posted.
void
posted_locked(int64_t *steps, int64_t *bytes)
{
  *steps = *bytes = 0;
  for (Comm *c : comms) {
    *steps += c->seq;
    const Traffic &t = c->traffic[BYTES];
    for (size_t peer = 0; peer < t.out.size(); peer++) {
      *bytes += t.out[peer] + t.in[peer];
    }
  }
}

// What this rank still owes its peers, while the checkpoint waits for it.
void
report_owed(long seconds)
{
  std::lock_guard<std::mutex> guard(nccl_lock);
  for (Comm *c : comms) {
    if (c->destroyed) {
      continue;
    }
    if (c->target >= 0 && c->seq < c->target) {
      fprintf(stderr, "MANA: --cuda: %ld s into the checkpoint, NCCL "
              "communicator %016llx: rank %d of %d is at step %lld of "
              "%lld\n", seconds, (unsigned long long)c->ident, c->rank,
              c->nranks, (long long)c->seq, (long long)c->target);
    }
    static const char *const out_what[KINDS] = { "sends of", "signals:" };
    static const char *const in_what[KINDS] = { "receives of", "waits:" };
    static const char *const unit[KINDS] = { " B", "" };
    for (int kind = 0; kind < KINDS; kind++) {
      const Traffic &t = c->traffic[kind];
      for (size_t peer = 0; peer < t.out.size(); peer++) {
        if (t.out[peer] < t.out_target[peer]) {
          fprintf(stderr, "MANA: --cuda: %ld s into the checkpoint, NCCL "
                  "communicator %016llx: rank %d owes rank %zu %s %lld%s\n",
                  seconds, (unsigned long long)c->ident, c->rank, peer,
                  out_what[kind], (long long)(t.out_target[peer] - t.out[peer]),
                  unit[kind]);
        }
        if (t.in[peer] < t.in_target[peer]) {
          fprintf(stderr, "MANA: --cuda: %ld s into the checkpoint, NCCL "
                  "communicator %016llx: rank %d owes rank %zu %s %lld%s\n",
                  seconds, (unsigned long long)c->ident, c->rank, peer,
                  in_what[kind], (long long)(t.in_target[peer] - t.in[peer]),
                  unit[kind]);
        }
      }
    }
  }
  if (activity > 0 || creating > 0) {
    fprintf(stderr, "MANA: --cuda: %ld s into the checkpoint, %d thread(s) "
            "in NCCL calls or group regions, %d creation(s) of NCCL "
            "communicators in flight\n", seconds, activity, creating);
  }
  fflush(stderr);
}

// ---- The restart ----

// The unique ids made at this restart, in DMTCP's key-value store: under
// u/<name>, a communicator's; under g/<name>, the one that replaces a unique
// id that its maker had not used.
void
restart_db(char *db, size_t size)
{
  snprintf(db, size, "mana-nccl-%u", front_gen);
}

void
to_hex(const ncclUniqueId &id, char *hex)
{
  const unsigned char *p = (const unsigned char *)&id;
  for (size_t i = 0; i < sizeof id; i++) {
    snprintf(hex + 2 * i, 3, "%02x", p[i]);
  }
}

bool
from_hex(const char *hex, ncclUniqueId *id)
{
  if (strlen(hex) != 2 * sizeof *id) {
    return false;
  }
  unsigned char *p = (unsigned char *)id;
  for (size_t i = 0; i < sizeof *id; i++) {
    unsigned byte;
    if (sscanf(hex + 2 * i, "%2x", &byte) != 1) {
      return false;
    }
    p[i] = byte;
  }
  return true;
}

void
publish_new_id(const char *db, char kind, uint64_t ident)
{
  ncclUniqueId id;
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclGetUniqueId)(&id));
  if (rc != ncclSuccess) {
    front_fatal("making an NCCL unique id again: error %d\n", rc);
  }
  char k[32], hex[2 * sizeof id + 1];
  key(k, sizeof k, kind, ident);
  to_hex(id, hex);
  if (mana_kv_set(db, k, hex) != 0) {
    front_fatal("publishing NCCL unique id %s\n", k);
  }
}

// In ncclCommInitRank(Config), within the checkpoint gate: the unique id
// that replaces one made before the last restart, which its maker had not
// used then.
void
renewed_id(uint64_t ident, ncclUniqueId *id)
{
  if (front_gen == 1) {
    return;
  }
  char db[32], k[32], hex[2 * sizeof *id + 1];
  restart_db(db, sizeof db);
  key(k, sizeof k, 'g', ident);
  std::lock_guard<std::mutex> guard(front_kv_lock);
  ncclUniqueId now;
  if (mana_kv_get(db, k, hex, sizeof hex) == 0 && from_hex(hex, &now)) {
    *id = now;
    front_log("NCCL: unique id %016llx was made again at the restart\n",
              (unsigned long long)ident);
  }
}

// The config that a communicator was made with, its strings the record's.
ncclConfig_t
config_of(const Comm *c)
{
  ncclConfig_t config = c->config;
  config.netName = c->net_name.empty() ? NULL : c->net_name.c_str();
  config.commName = c->comm_name.empty() ? NULL : c->comm_name.c_str();
  return config;
}

// A clique of ncclCommInitAll again, by one call: it is in this process
// alone.  The drain brings its communicators to the same step, so all of
// them are finalized, or none.  Returns how many it made.
size_t
rebuild_clique(int clique)
{
  std::vector<Comm *> members;
  for (Comm *c : comms) {
    if (!c->destroyed && c->made == INIT_ALL && c->clique == clique) {
      members.push_back(c);
    }
  }
  if (members.empty() || members[0]->finalized) {
    return 0;
  }
  std::sort(members.begin(), members.end(),
            [](Comm *a, Comm *b) { return a->rank < b->rank; });
  int n = (int)members.size();
  std::vector<int> devlist(n);
  for (int i = 0; i < n; i++) {
    if (members[i]->rank != i) {
      front_fatal("NCCL clique %d lacks rank %d\n", clique, i);
    }
    devlist[i] = members[i]->device;
  }
  std::vector<ncclComm_t> real(n);
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclCommInitAll)(real.data(), n, devlist.data()));
  if (rc != ncclSuccess) {
    front_fatal("making NCCL clique %d again: error %d\n", clique, rc);
  }
  for (int i = 0; i < n; i++) {
    members[i]->real = real[i];
  }
  return n;
}

// The communicators made by unique id or split, blocking or not, in one
// group region: NCCL makes them together at ncclGroupEnd, so the members
// need not agree on an order.  (NCCL refuses a region with both kinds.)
size_t
rebuild_group(const char *db, bool nonblocking)
{
  std::vector<Comm *> todo;
  for (Comm *c : comms) {
    if (!c->destroyed && !c->finalized && c->made != INIT_ALL &&
        c->nonblocking == nonblocking) {
      todo.push_back(c);
    }
  }
  if (todo.empty()) {
    return 0;
  }
  std::vector<ncclUniqueId> ids(todo.size());
  std::vector<ncclConfig_t> configs(todo.size());
  for (size_t i = 0; i < todo.size(); i++) {
    char k[32], hex[2 * sizeof(ncclUniqueId) + 1];
    key(k, sizeof k, 'u', todo[i]->ident);
    if (mana_kv_get(db, k, hex, sizeof hex) != 0 ||
        !from_hex(hex, &ids[i])) {
      front_fatal("NCCL communicator %016llx: its rank 0 did not make it "
                  "again\n", (unsigned long long)todo[i]->ident);
    }
    configs[i] = config_of(todo[i]);
  }
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclGroupStart)());
  for (size_t i = 0; i < todo.size() && posted(rc); i++) {
    Comm *c = todo[i];
    front_use_device(c->device);
    c->real = NULL;
    ncclComm_t *real = (ncclComm_t *)&c->real;
    if (c->has_config) {
      NCCL_LH(rc = NCCL(ncclCommInitRankConfig)(real, c->nranks, ids[i],
                                                c->rank, &configs[i]));
    } else {
      NCCL_LH(rc = NCCL(ncclCommInitRank)(real, c->nranks, ids[i], c->rank));
    }
  }
  ncclResult_t end_rc;
  NCCL_LH(end_rc = NCCL(ncclGroupEnd)());
  if (posted(rc)) {
    rc = end_rc;
  }
  if (!posted(rc)) {
    front_fatal("making %zu NCCL communicator(s) again: error %d\n",
                todo.size(), rc);
  }
  if (nonblocking) {
    for (Comm *c : todo) {
      if (!wait_made(c)) {  // made in the background
        front_fatal("making NCCL communicator %016llx again failed\n",
                    (unsigned long long)c->ident);
      }
    }
  }
  return todo.size();
}

bool
live(const Comm *c)
{
  return !c->destroyed && c->real != NULL;
}

// The buffer registrations, each local to its rank.  The buffers are at the
// same addresses, unless the program freed one without deregistering it,
// which NCCL lets pass: then the registration is gone, with a warning.
size_t
restore_regs()
{
  size_t n = 0;
  for (Comm *c : comms) {
    for (Reg *r : c->regs) {
      r->real = NULL;
#ifdef FRONT_IDX_nccl_ncclCommRegister
      if (!live(c)) {
        continue;
      }
      front_use_device(c->device);
      ncclResult_t rc;
      NCCL_LH(rc = NCCL(ncclCommRegister)((ncclComm_t)c->real, r->buff,
                                          r->size, &r->real));
      if (rc != ncclSuccess) {
        r->real = NULL;
        fprintf(stderr, "MANA: --cuda: registering %zu B at %p with NCCL "
                "communicator %016llx again failed (error %d); freed "
                "without ncclCommDeregister?\n", r->size, r->buff,
                (unsigned long long)c->ident, rc);
        continue;
      }
      n++;
#endif
    }
  }
  return n;
}

// The windows of the communicators, blocking or not, in one group region:
// registering a window is collective.
size_t
restore_wins(bool nonblocking)
{
  std::vector<Win *> todo;
  for (Comm *c : comms) {
    for (Win *w : c->wins) {
      if (!live(c)) {
        w->real = NULL;
      } else if (c->nonblocking == nonblocking) {
        todo.push_back(w);
      }
    }
  }
  if (todo.empty()) {
    return 0;
  }
#ifdef FRONT_IDX_nccl_ncclCommWindowRegister
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclGroupStart)());
  for (size_t i = 0; i < todo.size() && posted(rc); i++) {
    Win *w = todo[i];
    front_use_device(w->c->device);
    w->real = NULL;
    NCCL_LH(rc = NCCL(ncclCommWindowRegister)((ncclComm_t)w->c->real,
                                              w->buff, w->size,
                                              (ncclWindow_t *)&w->real,
                                              w->flags));
  }
  ncclResult_t end_rc;
  NCCL_LH(end_rc = NCCL(ncclGroupEnd)());
  if (posted(rc)) {
    rc = end_rc;
  }
  for (Win *w : todo) {
    if (posted(rc) && nonblocking && !wait_made(w->c)) {
      rc = ncclInternalError;
    }
    if (!posted(rc) || w->real == NULL) {
      front_fatal("registering NCCL window of %zu B at %p again: error %d\n",
                  w->size, w->buff, rc);
    }
  }
#endif
  return todo.size();
}

// The PreMulSum operators: NCCL's values change, the application's do not.
size_t
restore_ops()
{
  size_t n = 0;
#ifdef FRONT_IDX_nccl_ncclRedOpCreatePreMulSum
  for (Comm *c : comms) {
    if (!live(c)) {
      continue;
    }
    for (Op &o : c->ops) {
      front_use_device(c->device);
      void *scalar =
        o.residence == ncclScalarHostImmediate ? &o.immediate : o.scalar;
      ncclResult_t rc;
      NCCL_LH(rc = NCCL(ncclRedOpCreatePreMulSum)(&o.real, scalar, o.datatype,
                                                  o.residence,
                                                  (ncclComm_t)c->real));
      if (rc != ncclSuccess) {
        front_fatal("making NCCL operator %d of communicator %016llx again: "
                    "error %d\n", (int)o.value, (unsigned long long)c->ident,
                    rc);
      }
      n++;
    }
  }
#endif
  return n;
}

// The communicators that were suspended, last: a suspended one takes no
// other call.
size_t
restore_suspended()
{
  size_t n = 0;
#ifdef FRONT_IDX_nccl_ncclCommSuspend
  for (Comm *c : comms) {
    if (!live(c) || !c->suspended) {
      continue;
    }
    front_use_device(c->device);
    ncclResult_t rc;
    NCCL_LH(rc = NCCL(ncclCommSuspend)((ncclComm_t)c->real, c->suspend_flags));
    if (!posted(rc) || (rc == ncclInProgress && !wait_made(c))) {
      front_fatal("suspending NCCL communicator %016llx again: error %d\n",
                  (unsigned long long)c->ident, rc);
    }
    n++;
  }
#endif
  return n;
}

}  // namespace

// MANA_STEP_DRAIN, before MPI's drain, with the application's threads
// running.  In rounds: (1) adopt the targets that the members published;
// (2) let this rank's threads catch up, for one slice at most; (3) publish
// this rank's position and decide whether it is settled; (4) vote, with a
// barrier of every rank.  Done when no rank is unsettled.  Targets only
// grow, so the rounds converge: a rank told to catch up on one
// communicator may overshoot on another, and the next round takes it.  A
// slice is short (10 ms, doubling up to 1 s while a rank catches up),
// because catching up may need a peer that learns of it only from this
// rank's next publication.  Then the gate stays closed until the resume.
void
front_nccl_drain()
{
  int64_t steps0, bytes0;
  {
    std::lock_guard<std::mutex> guard(nccl_lock);
    posted_locked(&steps0, &bytes0);
    draining = true;
    frozen = false;
    creation_pending = false;
    for (Comm *c : comms) {
      c->target = -1;
      for (Traffic &t : c->traffic) {
        std::fill(t.out_target.begin(), t.out_target.end(), 0);
        std::fill(t.in_target.begin(), t.in_target.end(), 0);
      }
    }
  }
  long slice_ms = 10;
  long waited_ms = 0;
  long next_report_ms = 60000;
  for (int round = 0;; round++) {
    adopt_targets();

    auto start = std::chrono::steady_clock::now();
    while (!quiet() && std::chrono::steady_clock::now() - start <
                         std::chrono::milliseconds(slice_ms)) {
      usleep(1000);
    }

    std::vector<Published> pubs;
    bool behind = false;
    int in_flight = 0;
    bool settled = settle(in_progress(), &pubs, &behind, &in_flight);
    publish(pubs);

    char k[32];
    char barrier[48];
    snprintf(k, sizeof k, "u/%d", round);
    mana_ckpt_kv_update(KV, k, MANA_KV_ADD, !settled);
    snprintf(k, sizeof k, "w/%d", round);
    mana_ckpt_kv_update(KV, k, MANA_KV_ADD, behind);
    snprintf(k, sizeof k, "c/%d", round);
    mana_ckpt_kv_update(KV, k, MANA_KV_ADD, in_flight);
    snprintf(barrier, sizeof barrier, "MANA:NCCL-drain-%06d", round);
    mana_global_barrier(barrier);
    snprintf(k, sizeof k, "u/%d", round);
    int64_t unsettled = mana_ckpt_kv_get(KV, k);
    snprintf(k, sizeof k, "w/%d", round);
    int64_t waiting = mana_ckpt_kv_get(KV, k);
    snprintf(k, sizeof k, "c/%d", round);
    int64_t creations = mana_ckpt_kv_get(KV, k);

    if (unsettled == 0) {
      // Every member of a destroyed communicator has destroyed it now.
      std::lock_guard<std::mutex> guard(nccl_lock);
      int64_t steps, bytes;
      posted_locked(&steps, &bytes);
      auto gone = std::stable_partition(comms.begin(), comms.end(),
                                        [](Comm *c) { return !c->destroyed; });
      for (auto it = gone; it != comms.end(); ++it) {
        delete *it;
      }
      comms.erase(gone, comms.end());
      front_log("NCCL drain: %d round(s), %zu communicator(s); %lld step(s) "
                "and %lld B of sends and receives posted meanwhile\n",
                round + 1, comms.size(), (long long)(steps - steps0),
                (long long)(bytes - bytes0));
      return;
    }
    {
      std::lock_guard<std::mutex> guard(nccl_lock);
      frozen = false;
      creation_pending = creations > 0;
    }
    gate_cv.notify_all();
    // Every rank reads the same votes, so the slices agree.
    if (waiting > 0 || creations > 0) {
      waited_ms += slice_ms;
      slice_ms = std::min(2 * slice_ms, 1000L);
      if (waited_ms >= next_report_ms) {
        report_owed(waited_ms / 1000);
        next_report_ms += 60000;
      }
    }
  }
}

// MANA_STEP_RESUME: the gate opens.  Without the lock, which a thread that
// DMTCP suspended at the gate may hold (see enter()).
void
front_nccl_resume()
{
  draining = false;
  gate_cv.notify_all();
}

// MANA_STEP_RESTART, before the global barrier: the new unique ids, which
// this process makes for the communicators where it is rank 0, and for the
// unique ids that it made and had not used.  Without nccl_lock, as below:
// the application's threads are not running, and one may hold it.
void
front_nccl_publish()
{
  char db[32];
  restart_db(db, sizeof db);
  for (Comm *c : comms) {
    if (!c->destroyed && !c->finalized && c->made != INIT_ALL &&
        c->rank == 0) {
      publish_new_id(db, 'u', c->ident);
    }
  }
  for (uint64_t ident : unused_ids) {
    publish_new_id(db, 'g', ident);
  }
}

// MANA_STEP_RESTART_PEERS, after the barrier: the communicators again, then
// the gate opens.
void
front_nccl_rebuild()
{
  if (!comms.empty()) {
    auto start = std::chrono::steady_clock::now();
    char db[32];
    restart_db(db, sizeof db);
    for (Comm *c : comms) {
      if (c->finalized) {
        c->real = NULL;  // not made again
      }
    }
    size_t n = 0;
    for (int clique = 0; clique < cliques; clique++) {
      n += rebuild_clique(clique);
    }
    n += rebuild_group(db, false);
    n += rebuild_group(db, true);
    size_t regs = restore_regs();
    size_t wins = restore_wins(false) + restore_wins(true);
    size_t ops = restore_ops();
    size_t suspended = restore_suspended();
    std::chrono::duration<double> took =
      std::chrono::steady_clock::now() - start;
    front_log("NCCL: %zu of %zu communicator(s) made again in %.2f s; %zu "
              "buffer registration(s), %zu window(s), %zu operator(s), %zu "
              "suspended\n", n, comms.size(), took.count(), regs, wins, ops,
              suspended);
  }
  draining = false;
  gate_cv.notify_all();
}

// A unique id names its bootstrap root, a thread of this lower half; this
// process makes one again at a restart until it has used it.
extern "C" ncclResult_t
ncclGetUniqueId(ncclUniqueId *uniqueId)
{
  FRONT_ENTER();
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclGetUniqueId)(uniqueId));
  if (rc == ncclSuccess) {
    std::lock_guard<std::mutex> guard(nccl_lock);
    unused_ids.push_back(digest(DIGEST_START, uniqueId, sizeof *uniqueId));
  }
  FRONT_LEAVE();
  return rc;
}

// ncclCommInitRank(Config) with one unique id, ncclCommInitRankScalable with
// several.  A restart makes either again by one.
static ncclResult_t
init_rank(ncclComm_t *comm, int nranks, int nids, const ncclUniqueId *commIds,
          int rank, ncclConfig_t *config, bool scalable)
{
  if (comm == NULL || commIds == NULL || nids <= 0) {
    return ncclInvalidArgument;
  }
  enter(NULL, 0, true);
  FRONT_ENTER();
  front_before_library();
  Comm *c = new Comm();
  c->made = INIT_RANK;
  c->ident = digest(DIGEST_START, commIds, nids * sizeof *commIds);
  std::vector<ncclUniqueId> ids(commIds, commIds + nids);
  for (ncclUniqueId &id : ids) {
    c->ids.push_back(digest(DIGEST_START, &id, sizeof id));
    renewed_id(c->ids.back(), &id);
  }
  c->nranks = nranks;
  c->rank = rank;
  keep_config(c, config);
  cudaError_t drc;
  FRONT_LH(drc = REAL(cudaGetDevice)(&c->device));
  if (drc != cudaSuccess) {
    c->device = -1;
  }
  // In a group region, NCCL writes the communicator at ncclGroupEnd: into
  // the record, which lives on.
  ncclComm_t *real = (ncclComm_t *)&c->real;
  ncclResult_t rc = ncclInvalidUsage;
  if (scalable) {
#ifdef FRONT_IDX_nccl_ncclCommInitRankScalable
    NCCL_LH(rc = NCCL(ncclCommInitRankScalable)(real, nranks, rank, nids,
                                                ids.data(), config));
#endif
  } else if (config != NULL) {
    NCCL_LH(rc = NCCL(ncclCommInitRankConfig)(real, nranks, ids[0], rank,
                                              config));
  } else {
    NCCL_LH(rc = NCCL(ncclCommInitRank)(real, nranks, ids[0], rank));
  }
  ncclComm_t made = made_comm(c, rc);
  if (made != NULL) {
    *comm = made;
  }
  FRONT_LEAVE();
  leave(made != NULL, NULL, 0, true);
  return rc;
}

extern "C" ncclResult_t
ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId commId,
                       int rank, ncclConfig_t *config)
{
  return init_rank(comm, nranks, 1, &commId, rank, config, false);
}

extern "C" ncclResult_t
ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank)
{
  return init_rank(comm, nranks, 1, &commId, rank, NULL, false);
}

#ifdef FRONT_IDX_nccl_ncclCommInitRankScalable
extern "C" ncclResult_t
ncclCommInitRankScalable(ncclComm_t *newcomm, int nranks, int myrank, int nId,
                         ncclUniqueId *commIds, ncclConfig_t *config)
{
  return init_rank(newcomm, nranks, nId, commIds, myrank, config, true);
}
#endif

extern "C" ncclResult_t
ncclCommInitAll(ncclComm_t *comms, int ndev, const int *devlist)
{
  if (comms == NULL || ndev <= 0) {
    return ncclInvalidArgument;
  }
  enter(NULL, 0, false);
  FRONT_ENTER();
  front_before_library();
  std::vector<ncclComm_t> real(ndev);
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclCommInitAll)(real.data(), ndev, devlist));
  if (rc == ncclSuccess) {
    int clique;
    {
      std::lock_guard<std::mutex> guard(nccl_lock);
      clique = cliques++;
    }
    for (int i = 0; i < ndev; i++) {
      Comm *c = new Comm();
      c->made = INIT_ALL;
      c->clique = clique;
      c->real = real[i];
      c->nranks = ndev;
      c->rank = i;
      c->device = devlist != NULL ? devlist[i] : i;
      link_comm(c);
      comms[i] = (ncclComm_t)c;
    }
  }
  FRONT_LEAVE();
  leave(true, NULL, 0, false);
  return rc;
}

// A split is a step on its parent, whose members all take it.
extern "C" ncclResult_t
ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm,
              ncclConfig_t *config)
{
  if (newcomm == NULL) {
    return ncclInvalidArgument;
  }
  Comm *parent = comm_of(comm);
  enter(parent != NULL ? &parent->seq : NULL, 1, false);
  FRONT_ENTER();
  // Every split takes a number on its parent, also one that gives this rank
  // no communicator, so that the members' numbers agree.
  int index = parent != NULL
                ? __atomic_fetch_add(&parent->splits, 1, __ATOMIC_RELAXED)
                : 0;
  ncclResult_t rc;
  if (color == NCCL_SPLIT_NOCOLOR) {
    // No communicator for this rank; in a group region NCCL writes NULL at
    // ncclGroupEnd, into the application's own variable.
    NCCL_LH(rc = NCCL(ncclCommSplit)(front_translate(comm), color, key,
                                     newcomm, config));
    FRONT_LEAVE();
    leave(posted(rc), parent != NULL ? &parent->seq : NULL, 1, false);
    return rc;
  }
  Comm *c = new Comm();
  c->made = SPLIT;
  c->color = color;
  c->key = key;
  c->nranks = c->rank = c->device = -1;
  if (parent != NULL) {
    c->ident = digest(DIGEST_START, &parent->ident, sizeof parent->ident);
    c->ident = digest(c->ident, &index, sizeof index);
    c->ident = digest(c->ident, &color, sizeof color);
  }
  child_config(c, parent, config);
  c->parent_real = front_translate(comm);
  NCCL_LH(rc = NCCL(ncclCommSplit)(c->parent_real, color, key,
                                   (ncclComm_t *)&c->real, config));
  ncclComm_t made = made_comm(c, rc);
  if (made != NULL) {
    *newcomm = made;
  }
  FRONT_LEAVE();
  leave(posted(rc), parent != NULL ? &parent->seq : NULL, 1, false);
  return rc;
}

#ifdef FRONT_IDX_nccl_ncclCommShrink
// A shrink makes a communicator of the parent's members that call it, the
// ranks not left out: a creation that they must all join, but not a step
// on the parent, which the ranks left out do not take.  A restart makes it
// again as a split.
extern "C" ncclResult_t
ncclCommShrink(ncclComm_t comm, int *excludeRanksList, int excludeRanksCount,
               ncclComm_t *newcomm, ncclConfig_t *config, int shrinkFlags)
{
  if (newcomm == NULL || excludeRanksCount < 0 ||
      (excludeRanksCount > 0 && excludeRanksList == NULL)) {
    return ncclInvalidArgument;
  }
  Comm *parent = comm_of(comm);
  enter(NULL, 0, true);
  FRONT_ENTER();
  Comm *c = new Comm();
  c->made = SHRINK;
  c->nranks = c->rank = c->device = -1;
  if (parent != NULL) {
    std::vector<int> out(excludeRanksList,
                         excludeRanksList + excludeRanksCount);
    std::sort(out.begin(), out.end());
    uint64_t left_out = digest(DIGEST_START, out.data(),
                               out.size() * sizeof(int));
    int index;
    {
      std::lock_guard<std::mutex> guard(nccl_lock);
      auto it = std::find_if(parent->shrinks.begin(), parent->shrinks.end(),
                             [&](const std::pair<uint64_t, int> &s) {
                               return s.first == left_out;
                             });
      if (it == parent->shrinks.end()) {
        parent->shrinks.push_back({ left_out, 0 });
        it = parent->shrinks.end() - 1;
      }
      index = it->second++;
    }
    c->ident = digest(DIGEST_START, &parent->ident, sizeof parent->ident);
    c->ident = digest(c->ident, "shrink", 6);
    c->ident = digest(c->ident, &left_out, sizeof left_out);
    c->ident = digest(c->ident, &index, sizeof index);
  }
  child_config(c, parent, config);
  // NCCL_SHRINK_ABORT aborts the parent, so its error says nothing.
  if (!(shrinkFlags & NCCL_SHRINK_ABORT)) {
    c->parent_real = front_translate(comm);
  }
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclCommShrink)(front_translate(comm), excludeRanksList,
                                    excludeRanksCount, (ncclComm_t *)&c->real,
                                    config, shrinkFlags));
  ncclComm_t made = made_comm(c, rc);
  if (made != NULL) {
    *newcomm = made;
  }
  FRONT_LEAVE();
  leave(made != NULL, NULL, 0, true);
  return rc;
}
#endif

extern "C" ncclResult_t
ncclCommFinalize(ncclComm_t comm)
{
  Comm *c = comm_of(comm);
  int64_t *step = c != NULL && !c->finalized ? &c->seq : NULL;
  enter(step, 1, false);
  FRONT_ENTER();
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclCommFinalize)(front_translate(comm)));
  if (posted(rc) && c != NULL) {
    c->finalized = true;
  }
  FRONT_LEAVE();
  leave(posted(rc), step, 1, false);
  return rc;
}

// ncclCommDestroy, a step unless ncclCommFinalize took it; and
// ncclCommAbort, which is not collective.
static ncclResult_t
end_comm(ncclComm_t comm, bool abort)
{
  Comm *c = comm_of(comm);
  int64_t *step = !abort && c != NULL && !c->finalized ? &c->seq : NULL;
  enter(step, 1, false);
  FRONT_ENTER();
  ncclResult_t rc = ncclSuccess;
  if (c != NULL && c->real == NULL) {
    // Finalized before a restart, and not made again.
  } else if (abort) {
    NCCL_LH(rc = NCCL(ncclCommAbort)(front_translate(comm)));
  } else {
    NCCL_LH(rc = NCCL(ncclCommDestroy)(front_translate(comm)));
  }
  if (rc == ncclSuccess && c != NULL) {
    {
      std::lock_guard<std::mutex> guard(objs_lock);
      obj_unlink(c);
    }
    std::lock_guard<std::mutex> guard(nccl_lock);
    c->destroyed = true;
    c->real = NULL;
    forget_made_locked(c);
  }
  FRONT_LEAVE();
  leave(rc == ncclSuccess, step, 1, false);
  // Without a live communicator, this rank's gate may open.
  gate_cv.notify_all();
  return rc;
}

extern "C" ncclResult_t
ncclCommDestroy(ncclComm_t comm)
{
  return end_comm(comm, false);
}

extern "C" ncclResult_t
ncclCommAbort(ncclComm_t comm)
{
  return end_comm(comm, true);
}

// A communicator finalized before a restart is not made again: its
// finalization is complete.
extern "C" ncclResult_t
ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError)
{
  Comm *c = comm_of(comm);
  if (c != NULL && c->real == NULL && asyncError != NULL) {
    *asyncError = ncclSuccess;
    return ncclSuccess;
  }
  FRONT_ENTER();
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclCommGetAsyncError)(front_translate(comm),
                                           asyncError));
  FRONT_LEAVE();
  return rc;
}

// A group region passes the gate at its outermost ncclGroupStart, and
// counts as activity until its ncclGroupEnd: NCCL launches its operations
// only then, so the drain must not settle in between.
extern "C" ncclResult_t
ncclGroupStart()
{
  if (group_depth == 0) {
    enter(NULL, 0, false);
  }
  group_depth++;
  FRONT_ENTER();
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclGroupStart)());
  FRONT_LEAVE();
  if (rc != ncclSuccess && --group_depth == 0) {
    leave(true, NULL, 0, false);
  }
  return rc;
}

extern "C" ncclResult_t
ncclGroupEnd()
{
  FRONT_ENTER();
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclGroupEnd)());
  if (group_depth > 0 && --group_depth == 0) {
    end_group(rc, false);
  }
  FRONT_LEAVE();
  return rc;
}

#ifdef FRONT_IDX_nccl_ncclGroupSimulateEnd
// Ends the group region as ncclGroupEnd does, without running it.
extern "C" ncclResult_t
ncclGroupSimulateEnd(ncclSimInfo_t *simInfo)
{
  FRONT_ENTER();
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclGroupSimulateEnd)(simInfo));
  if (group_depth > 0 && --group_depth == 0) {
    end_group(rc, true);
  }
  FRONT_LEAVE();
  return rc;
}
#endif

// The collectives: a step each; the communicator, the stream and the
// operator translated.
#define NCCL_COLLECTIVE(name, params, args)                         \
  extern "C" ncclResult_t name params                               \
  {                                                                 \
    Comm *c = comm_of(comm);                                        \
    int64_t *step = c != NULL ? &c->seq : NULL;                     \
    enter(step, 1, false);                                          \
    FRONT_ENTER();                                                  \
    ncclResult_t rc;                                                \
    ncclComm_t real_comm = front_translate(comm);                   \
    cudaStream_t real_stream = front_real_stream(stream);           \
    NCCL_LH(rc = NCCL(name) args);                                  \
    FRONT_LEAVE();                                                  \
    leave(posted(rc), step, 1, false);                              \
    return rc;                                                      \
  }

NCCL_COLLECTIVE(ncclAllReduce,
                (const void *sendbuff, void *recvbuff, size_t count,
                 ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                 cudaStream_t stream),
                (sendbuff, recvbuff, count, datatype, real_op(c, op),
                 real_comm, real_stream))
NCCL_COLLECTIVE(ncclBroadcast,
                (const void *sendbuff, void *recvbuff, size_t count,
                 ncclDataType_t datatype, int root, ncclComm_t comm,
                 cudaStream_t stream),
                (sendbuff, recvbuff, count, datatype, root, real_comm,
                 real_stream))
NCCL_COLLECTIVE(ncclBcast,
                (void *buff, size_t count, ncclDataType_t datatype, int root,
                 ncclComm_t comm, cudaStream_t stream),
                (buff, count, datatype, root, real_comm, real_stream))
NCCL_COLLECTIVE(ncclReduce,
                (const void *sendbuff, void *recvbuff, size_t count,
                 ncclDataType_t datatype, ncclRedOp_t op, int root,
                 ncclComm_t comm, cudaStream_t stream),
                (sendbuff, recvbuff, count, datatype, real_op(c, op), root,
                 real_comm, real_stream))
NCCL_COLLECTIVE(ncclAllGather,
                (const void *sendbuff, void *recvbuff, size_t sendcount,
                 ncclDataType_t datatype, ncclComm_t comm,
                 cudaStream_t stream),
                (sendbuff, recvbuff, sendcount, datatype, real_comm,
                 real_stream))
NCCL_COLLECTIVE(ncclReduceScatter,
                (const void *sendbuff, void *recvbuff, size_t recvcount,
                 ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                 cudaStream_t stream),
                (sendbuff, recvbuff, recvcount, datatype, real_op(c, op),
                 real_comm, real_stream))
#ifdef FRONT_IDX_nccl_ncclAlltoAll
NCCL_COLLECTIVE(ncclAlltoAll,
                (const void *sendbuff, void *recvbuff, size_t count,
                 ncclDataType_t datatype, ncclComm_t comm,
                 cudaStream_t stream),
                (sendbuff, recvbuff, count, datatype, real_comm,
                 real_stream))
#endif
#ifdef FRONT_IDX_nccl_ncclGather
NCCL_COLLECTIVE(ncclGather,
                (const void *sendbuff, void *recvbuff, size_t sendcount,
                 ncclDataType_t datatype, int root, ncclComm_t comm,
                 cudaStream_t stream),
                (sendbuff, recvbuff, sendcount, datatype, root, real_comm,
                 real_stream))
#endif
#ifdef FRONT_IDX_nccl_ncclScatter
NCCL_COLLECTIVE(ncclScatter,
                (const void *sendbuff, void *recvbuff, size_t recvcount,
                 ncclDataType_t datatype, int root, ncclComm_t comm,
                 cudaStream_t stream),
                (sendbuff, recvbuff, recvcount, datatype, root, real_comm,
                 real_stream))
#endif

// Point-to-point: the bytes to or from the peer.  NCCL has no wildcard
// receive, and a pair's sends and receives match in the order posted, so
// the counts say which operations still lack their counterpart.
extern "C" ncclResult_t
ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype, int peer,
         ncclComm_t comm, cudaStream_t stream)
{
  Comm *c = comm_of(comm);
  int64_t *sent = traffic_count(c, BYTES, true, peer);
  int64_t bytes = (int64_t)(count * type_size(datatype));
  enter(sent, bytes, false);
  FRONT_ENTER();
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclSend)(sendbuff, count, datatype, peer,
                              front_translate(comm),
                              front_real_stream(stream)));
  FRONT_LEAVE();
  leave(posted(rc), sent, bytes, false);
  return rc;
}

extern "C" ncclResult_t
ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype, int peer,
         ncclComm_t comm, cudaStream_t stream)
{
  Comm *c = comm_of(comm);
  int64_t *received = traffic_count(c, BYTES, false, peer);
  int64_t bytes = (int64_t)(count * type_size(datatype));
  enter(received, bytes, false);
  FRONT_ENTER();
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclRecv)(recvbuff, count, datatype, peer,
                              front_translate(comm),
                              front_real_stream(stream)));
  FRONT_LEAVE();
  leave(posted(rc), received, bytes, false);
  return rc;
}

// The front's allocator: VMM memory at fixed addresses, which a restart
// maps again, and which the NIC can register.
extern "C" ncclResult_t
ncclMemAlloc(void **ptr, size_t size)
{
  if (ptr == NULL) {
    return ncclInvalidArgument;
  }
  FRONT_ENTER();
  cudaError_t rc = front_vmm_alloc(ptr, size);
  FRONT_LEAVE();
  return rc == cudaSuccess ? ncclSuccess : ncclUnhandledCudaError;
}

extern "C" ncclResult_t
ncclMemFree(void *ptr)
{
  if (ptr == NULL) {
    return ncclSuccess;
  }
  FRONT_ENTER();
  cudaError_t result = cudaSuccess;
  bool ours = front_vmm_free(ptr, &result);
  FRONT_LEAVE();
  if (!ours) {
    return ncclInvalidArgument;
  }
  return result == cudaSuccess ? ncclSuccess : ncclUnhandledCudaError;
}

#ifdef FRONT_IDX_nccl_ncclCommRegister
// A buffer registration is local to the rank.  The application's handle is
// the record, which a restart registers again.
extern "C" ncclResult_t
ncclCommRegister(const ncclComm_t comm, void *buff, size_t size, void **handle)
{
  if (handle == NULL) {
    return ncclInvalidArgument;
  }
  Comm *c = comm_of(comm);
  FRONT_ENTER();
  void *real = NULL;
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclCommRegister)(front_translate(comm), buff, size,
                                      &real));
  FRONT_LEAVE();
  if (rc != ncclSuccess || c == NULL) {
    *handle = real;
    return rc;
  }
  Reg *r = new Reg{ c, buff, size, real };
  std::lock_guard<std::mutex> guard(nccl_lock);
  c->regs.push_back(r);
  *handle = r;
  return rc;
}

extern "C" ncclResult_t
ncclCommDeregister(const ncclComm_t comm, void *handle)
{
  Comm *c = comm_of(comm);
  Reg *r = NULL;
  if (c != NULL) {
    std::lock_guard<std::mutex> guard(nccl_lock);
    auto it = std::find(c->regs.begin(), c->regs.end(), (Reg *)handle);
    if (it != c->regs.end()) {
      r = *it;
    }
  }
  ncclResult_t rc = ncclSuccess;
  FRONT_ENTER();
  if (r == NULL) {
    NCCL_LH(rc = NCCL(ncclCommDeregister)(front_translate(comm), handle));
  } else if (r->real != NULL) {  // else gone with a finalized communicator
    NCCL_LH(rc = NCCL(ncclCommDeregister)(front_translate(comm), r->real));
  }
  FRONT_LEAVE();
  if (r != NULL && rc == ncclSuccess) {
    std::lock_guard<std::mutex> guard(nccl_lock);
    c->regs.erase(std::remove(c->regs.begin(), c->regs.end(), r),
                  c->regs.end());
    delete r;
  }
  return rc;
}
#endif

#ifdef FRONT_IDX_nccl_ncclCommWindowRegister
// Registering a window is collective: a step on its communicator.  NCCL
// writes its window into the application's variable, at ncclGroupEnd in a
// group region; then the front puts its record there (win_made()).
extern "C" ncclResult_t
ncclCommWindowRegister(ncclComm_t comm, void *buff, size_t size,
                       ncclWindow_t *win, int winFlags)
{
  if (win == NULL) {
    return ncclInvalidArgument;
  }
  Comm *c = comm_of(comm);
  int64_t *step = c != NULL ? &c->seq : NULL;
  enter(step, 1, false);
  FRONT_ENTER();
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclCommWindowRegister)(front_translate(comm), buff, size,
                                            win, winFlags));
  if (posted(rc) && c != NULL) {
    Win *w = new Win{ c, buff, size, winFlags, NULL };
    if (group_depth > 0) {
      if (group_wins == NULL) {
        group_wins = new std::vector<std::pair<Win *, void **>>();
      }
      group_wins->push_back({ w, (void **)win });
    } else if (rc == ncclSuccess || wait_made(c)) {
      win_made(w, (void **)win);
    } else {
      delete w;
    }
  }
  FRONT_LEAVE();
  leave(posted(rc), step, 1, false);
  return rc;
}

// Deregistering a window is local.
extern "C" ncclResult_t
ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win)
{
  Comm *c = comm_of(comm);
  Win *w = win_of(c, win);
  ncclResult_t rc = ncclSuccess;
  FRONT_ENTER();
  if (w == NULL) {
    NCCL_LH(rc = NCCL(ncclCommWindowDeregister)(front_translate(comm), win));
  } else if (w->real != NULL) {  // else gone with a finalized communicator
    NCCL_LH(rc = NCCL(ncclCommWindowDeregister)(front_translate(comm),
                                                (ncclWindow_t)w->real));
  }
  FRONT_LEAVE();
  if (w != NULL && rc == ncclSuccess) {
    std::lock_guard<std::mutex> guard(nccl_lock);
    c->wins.erase(std::remove(c->wins.begin(), c->wins.end(), w),
                  c->wins.end());
    delete w;
  }
  return rc;
}
#endif

#ifdef FRONT_IDX_nccl_ncclWinGetUserPtr
extern "C" ncclResult_t
ncclWinGetUserPtr(ncclComm_t comm, ncclWindow_t win, void **outUserPtr)
{
  Comm *c = comm_of(comm);
  ncclWindow_t real = (ncclWindow_t)real_win(c, win);
  FRONT_ENTER();
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclWinGetUserPtr)(front_translate(comm), real,
                                       outUserPtr));
  FRONT_LEAVE();
  return rc;
}
#endif

#ifdef FRONT_IDX_nccl_ncclPutSignal
// One-sided operations: each signal to a peer, and each that a wait takes
// from it, is traffic for the drain, which needs every signal taken: NCCL's
// signals do not outlive a restart.  What a put writes is in the peer's
// window, which is the front's memory.
extern "C" ncclResult_t
ncclPutSignal(const void *localbuff, size_t count, ncclDataType_t datatype,
              int peer, ncclWindow_t peerWin, size_t peerWinOffset, int sigIdx,
              int ctx, unsigned int flags, ncclComm_t comm,
              cudaStream_t stream)
{
  Comm *c = comm_of(comm);
  int64_t *signals = traffic_count(c, SIGNALS, true, peer);
  ncclWindow_t real = (ncclWindow_t)real_win(c, peerWin);
  enter(signals, 1, false);
  FRONT_ENTER();
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclPutSignal)(localbuff, count, datatype, peer, real,
                                   peerWinOffset, sigIdx, ctx, flags,
                                   front_translate(comm),
                                   front_real_stream(stream)));
  FRONT_LEAVE();
  leave(posted(rc), signals, 1, false);
  return rc;
}

extern "C" ncclResult_t
ncclSignal(int peer, int sigIdx, int ctx, unsigned int flags, ncclComm_t comm,
           cudaStream_t stream)
{
  Comm *c = comm_of(comm);
  int64_t *signals = traffic_count(c, SIGNALS, true, peer);
  enter(signals, 1, false);
  FRONT_ENTER();
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclSignal)(peer, sigIdx, ctx, flags,
                                front_translate(comm),
                                front_real_stream(stream)));
  FRONT_LEAVE();
  leave(posted(rc), signals, 1, false);
  return rc;
}

extern "C" ncclResult_t
ncclWaitSignal(int nDesc, ncclWaitSignalDesc_t *signalDescs, ncclComm_t comm,
               cudaStream_t stream)
{
  Comm *c = comm_of(comm);
  std::vector<Count> waits;
  for (int i = 0; i < nDesc && signalDescs != NULL; i++) {
    waits.push_back({ traffic_count(c, SIGNALS, false, signalDescs[i].peer),
                      signalDescs[i].opCnt });
  }
  enter_counts(waits.data(), waits.size(), false);
  FRONT_ENTER();
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclWaitSignal)(nDesc, signalDescs, front_translate(comm),
                                    front_real_stream(stream)));
  FRONT_LEAVE();
  leave_counts(posted(rc), waits.data(), waits.size(), false);
  return rc;
}
#endif

#ifdef FRONT_IDX_nccl_ncclRedOpCreatePreMulSum
// A PreMulSum operator is local to the rank.  The application keeps the
// value it got; a restart makes the operator again (restore_ops()).
extern "C" ncclResult_t
ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar,
                         ncclDataType_t datatype,
                         ncclScalarResidence_t residence, ncclComm_t comm)
{
  if (op == NULL) {
    return ncclInvalidArgument;
  }
  Comm *c = comm_of(comm);
  FRONT_ENTER();
  ncclRedOp_t real;
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclRedOpCreatePreMulSum)(&real, scalar, datatype,
                                              residence,
                                              front_translate(comm)));
  FRONT_LEAVE();
  if (rc != ncclSuccess || c == NULL) {
    if (rc == ncclSuccess) {
      *op = real;
    }
    return rc;
  }
  Op o = {};
  o.real = real;
  o.datatype = datatype;
  o.residence = residence;
  o.scalar = scalar;
  if (residence == ncclScalarHostImmediate && scalar != NULL) {
    memcpy(&o.immediate, scalar,
           std::min(type_size(datatype), sizeof o.immediate));
  }
  std::lock_guard<std::mutex> guard(nccl_lock);
  o.value = real;
  while (op_used_locked(c, o.value)) {
    o.value = o.value == ncclMaxRedOp ? (ncclRedOp_t)ncclNumOps
                                      : (ncclRedOp_t)(o.value + 1);
  }
  c->ops.push_back(o);
  *op = o.value;
  return rc;
}

extern "C" ncclResult_t
ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm)
{
  Comm *c = comm_of(comm);
  ncclRedOp_t real = real_op(c, op);
  bool ours = false;
  if (c != NULL && custom_op(op)) {
    std::lock_guard<std::mutex> guard(nccl_lock);
    ours = op_used_locked(c, op);
  }
  ncclResult_t rc = ncclSuccess;
  FRONT_ENTER();
  if (!ours || c->real != NULL) {  // else gone with a finalized communicator
    NCCL_LH(rc = NCCL(ncclRedOpDestroy)(real, front_translate(comm)));
  }
  FRONT_LEAVE();
  if (ours && rc == ncclSuccess) {
    std::lock_guard<std::mutex> guard(nccl_lock);
    c->ops.erase(std::remove_if(c->ops.begin(), c->ops.end(),
                                [op](const Op &o) { return o.value == op; }),
                 c->ops.end());
  }
  return rc;
}
#endif

#ifdef FRONT_IDX_nccl_ncclCommSuspend
// Suspending a communicator is local; a restart suspends it again, so that
// the program's ncclCommResume finds it so.
extern "C" ncclResult_t
ncclCommSuspend(ncclComm_t comm, int flags)
{
  Comm *c = comm_of(comm);
  FRONT_ENTER();
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclCommSuspend)(front_translate(comm), flags));
  FRONT_LEAVE();
  if (posted(rc) && c != NULL) {
    std::lock_guard<std::mutex> guard(nccl_lock);
    c->suspended = true;
    c->suspend_flags = flags;
  }
  return rc;
}
#endif

#ifdef FRONT_IDX_nccl_ncclCommResume
extern "C" ncclResult_t
ncclCommResume(ncclComm_t comm)
{
  Comm *c = comm_of(comm);
  FRONT_ENTER();
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclCommResume)(front_translate(comm)));
  FRONT_LEAVE();
  if (posted(rc) && c != NULL) {
    std::lock_guard<std::mutex> guard(nccl_lock);
    c->suspended = false;
  }
  return rc;
}
#endif
