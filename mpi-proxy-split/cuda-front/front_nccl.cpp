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
// to make the communicator again (step 4 of the plan): how it was made, its
// rank, size and device, the unique id or the parent, color and key, and
// the config; and what the drain needs: the steps posted on it, and the
// bytes sent to and received from each peer.
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
// A resume changes nothing in the lower half, so the communicators stay.
// A restart with one is refused until the rebuild exists (step 4).

#include <nccl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
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
  INIT_RANK,             // ncclCommInitRank, ncclCommInitRankConfig
  INIT_ALL,              // ncclCommInitAll: one record per device
  SPLIT                  // ncclCommSplit
};

struct Comm : FrontObj {
  Ctx *ctx;
  Made made;
  int nranks;
  int rank;
  int device;
  ncclUniqueId id;       // INIT_RANK
  Comm *parent;          // SPLIT
  int color;
  int key;
  int clique;            // INIT_ALL: the call's number
  bool has_config;
  ncclConfig_t config;
  std::string net_name;  // the config's strings, which are the application's
  std::string comm_name;
  bool nonblocking;      // config.blocking == 0
  bool finalized;
  // Its name in the drain, the same in every member: a digest of the unique
  // id, or of the parent's name, the split's number on the parent and the
  // color.  A clique of ncclCommInitAll is in this process alone.
  uint64_t ident;
  int splits;            // the splits of this one so far
  // Under nccl_lock.  A step is a collective, a split, or the first of
  // ncclCommFinalize and ncclCommDestroy.  The targets are what the members
  // have published: target, the steps (-1: not known yet); send_target, the
  // bytes each peer has posted to receive from this rank; recv_target, to
  // send to it.
  int64_t seq;
  int64_t target;
  std::vector<int64_t> sent;
  std::vector<int64_t> received;
  std::vector<int64_t> send_target;
  std::vector<int64_t> recv_target;
  // Destroyed: kept for the drain, with its last step, until a drain
  // converges, when every member has destroyed it too.
  bool destroyed;
};

// The drain and the gate.  Never held with objs_lock.
std::mutex nccl_lock;
std::condition_variable gate_cv;
std::vector<Comm *> comms;     // in creation order
bool draining;                 // from front_nccl_drain() to the resume
bool frozen;                   // settled in the drain's last round
bool creation_pending;         // a rank had a creation in flight at the vote
int activity;                  // threads in a call, or in a group region
int creating;                  // creations by unique id in flight here
int cliques;

// The application's thread: its depth in group regions, and the
// communicators that NCCL makes at the region's end.
__thread int group_depth;
__thread std::vector<Comm *> *group_made;

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
// have posted, or a send or a receive.
bool
owes_locked(const Comm *c)
{
  if (c->target >= 0 && c->seq < c->target) {
    return true;
  }
  for (size_t peer = 0; peer < c->sent.size(); peer++) {
    if (c->sent[peer] < c->send_target[peer] ||
        c->received[peer] < c->recv_target[peer]) {
      return true;
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

// Before a call that posts on a communicator or makes one: waits at the
// gate, except in a group region, whose ncclGroupStart passed it.  Then
// counts the call in activity, and adds n to *count (a step, or bytes to a
// peer), under the same lock: the drain sees a call either before the gate
// or posted.  create: a creation by unique id, in flight until leave().
void
enter(int64_t *count, int64_t n, bool create)
{
  std::unique_lock<std::mutex> lock(nccl_lock);
  if (group_depth == 0) {
    gate_cv.wait(lock, [] { return !gate_closed_locked(); });
  }
  if (count != NULL) {
    *count += n;
  }
  activity++;
  if (create) {
    creating++;
  }
}

// After the call; a call that failed takes back what enter() added.  In a
// group region, a creation stays in flight until the region ends.
void
leave(bool ok, int64_t *count, int64_t n, bool create)
{
  std::lock_guard<std::mutex> guard(nccl_lock);
  if (!ok && count != NULL) {
    *count -= n;
  }
  activity--;
  if (create && (group_depth == 0 || !ok)) {
    creating--;
  }
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
  c->target = -1;
  size_t n = c->nranks > 0 ? c->nranks : 0;
  c->sent.assign(n, 0);
  c->received.assign(n, 0);
  c->send_target.assign(n, 0);
  c->recv_target.assign(n, 0);
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    obj_link(c, FRONT_NCCL);
  }
  std::lock_guard<std::mutex> guard(nccl_lock);
  comms.push_back(c);
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
  if (posted(rc) && c->real != NULL) {
    link_comm(c);
    return (ncclComm_t)c;
  }
  delete c;
  return NULL;
}

// The outermost ncclGroupEnd.  Its creations by unique id are no longer in
// flight; the others were steps on their parent.
void
end_group(ncclResult_t rc)
{
  std::vector<Comm *> *made = group_made;
  group_made = NULL;
  int unique = 0;
  if (made != NULL) {
    for (Comm *c : *made) {
      unique += c->made == INIT_RANK;
      if (posted(rc) && c->real != NULL) {
        link_comm(c);
      } else {
        delete c;
      }
    }
    delete made;
  }
  std::lock_guard<std::mutex> guard(nccl_lock);
  creating -= unique;
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

void
nccl_restore(FrontObj *obj)
{
  (void)obj;
  front_fatal("restarting with an NCCL communicator: not supported yet\n");
}

void
nccl_settle()
{
}

void
nccl_forget(Ctx *ctx)
{
  (void)ctx;
}

const FrontLibRecords nccl_records = { nccl_restore, nccl_settle,
                                       nccl_forget, NULL, NULL };

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
  std::vector<int64_t> sent;
  std::vector<int64_t> received;
};

// What the members published in the last round, for one communicator.
struct Read {
  Comm *c;
  int64_t seq;           // the largest step, plus 1; 0 if none
  std::vector<int64_t> send_target;
  std::vector<int64_t> recv_target;
};

// Keys in KV: s/<ident>, the members' largest step plus 1; q/<ident>, 1 if
// a member has sent or received; p/<ident>/<src>/<dst>, the bytes that
// src has posted to send to dst; r/<ident>/<src>/<dst>, the bytes that dst
// has posted to receive from src.  Values only grow (MANA_KV_MAX).
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
    key(k, sizeof k, 'q', c->ident);
    if (mana_ckpt_kv_get(KV, k) == 0) {
      continue;
    }
    r.send_target.assign(c->nranks, 0);
    r.recv_target.assign(c->nranks, 0);
    for (int peer = 0; peer < c->nranks; peer++) {
      if (peer == c->rank) {
        continue;
      }
      key(k, sizeof k, 'r', c->ident, c->rank, peer);
      r.send_target[peer] = mana_ckpt_kv_get(KV, k);
      key(k, sizeof k, 'p', c->ident, peer, c->rank);
      r.recv_target[peer] = mana_ckpt_kv_get(KV, k);
    }
  }
  std::lock_guard<std::mutex> guard(nccl_lock);
  for (Read &r : reads) {
    Comm *c = r.c;
    if (r.seq > 0) {
      c->target = std::max(c->target, r.seq - 1);
    }
    for (size_t peer = 0; peer < r.send_target.size(); peer++) {
      c->send_target[peer] = std::max(c->send_target[peer],
                                      r.send_target[peer]);
      c->recv_target[peer] = std::max(c->recv_target[peer],
                                      r.recv_target[peer]);
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
      if (!c->destroyed && c->nonblocking) {
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
    out->push_back({ c->ident, c->rank, c->seq, c->sent, c->received });
    if (c->destroyed) {
      continue;
    }
    if (c->target < 0 || c->seq != c->target) {
      settled = false;
    }
    for (size_t peer = 0; peer < c->sent.size(); peer++) {
      if (c->sent[peer] != c->send_target[peer] ||
          c->received[peer] != c->recv_target[peer]) {
        settled = false;
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
    bool p2p = false;
    for (size_t peer = 0; peer < p.sent.size(); peer++) {
      if (p.sent[peer] > 0) {
        key(k, sizeof k, 'p', p.ident, p.rank, peer);
        mana_ckpt_kv_update(KV, k, MANA_KV_MAX, p.sent[peer]);
        p2p = true;
      }
      if (p.received[peer] > 0) {
        key(k, sizeof k, 'r', p.ident, peer, p.rank);
        mana_ckpt_kv_update(KV, k, MANA_KV_MAX, p.received[peer]);
        p2p = true;
      }
    }
    if (p2p) {
      key(k, sizeof k, 'q', p.ident);
      mana_ckpt_kv_update(KV, k, MANA_KV_MAX, 1);
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
    for (size_t peer = 0; peer < c->sent.size(); peer++) {
      *bytes += c->sent[peer] + c->received[peer];
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
    for (size_t peer = 0; peer < c->sent.size(); peer++) {
      if (c->sent[peer] < c->send_target[peer]) {
        fprintf(stderr, "MANA: --cuda: %ld s into the checkpoint, NCCL "
                "communicator %016llx: rank %d owes rank %zu sends of %lld "
                "B\n", seconds, (unsigned long long)c->ident, c->rank, peer,
                (long long)(c->send_target[peer] - c->sent[peer]));
      }
      if (c->received[peer] < c->recv_target[peer]) {
        fprintf(stderr, "MANA: --cuda: %ld s into the checkpoint, NCCL "
                "communicator %016llx: rank %d owes rank %zu receives of "
                "%lld B\n", seconds, (unsigned long long)c->ident, c->rank,
                peer, (long long)(c->recv_target[peer] - c->received[peer]));
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
      std::fill(c->send_target.begin(), c->send_target.end(), 0);
      std::fill(c->recv_target.begin(), c->recv_target.end(), 0);
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

// MANA_STEP_RESUME: the gate opens.
void
front_nccl_resume()
{
  {
    std::lock_guard<std::mutex> guard(nccl_lock);
    draining = false;
    frozen = false;
    creation_pending = false;
  }
  gate_cv.notify_all();
}

extern "C" ncclResult_t
ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId commId,
                       int rank, ncclConfig_t *config)
{
  if (comm == NULL) {
    return ncclInvalidArgument;
  }
  enter(NULL, 0, true);
  FRONT_ENTER();
  front_before_library();
  Comm *c = new Comm();
  c->made = INIT_RANK;
  c->id = commId;
  c->ident = digest(DIGEST_START, &commId, sizeof commId);
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
  ncclResult_t rc;
  if (config != NULL) {
    NCCL_LH(rc = NCCL(ncclCommInitRankConfig)(real, nranks, commId, rank,
                                              config));
  } else {
    NCCL_LH(rc = NCCL(ncclCommInitRank)(real, nranks, commId, rank));
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
ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId, int rank)
{
  return ncclCommInitRankConfig(comm, nranks, commId, rank, NULL);
}

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
  c->parent = parent;
  c->color = color;
  c->key = key;
  c->nranks = c->rank = c->device = -1;
  if (parent != NULL) {
    int index = __atomic_fetch_add(&parent->splits, 1, __ATOMIC_RELAXED);
    c->ident = digest(DIGEST_START, &parent->ident, sizeof parent->ident);
    c->ident = digest(c->ident, &index, sizeof index);
    c->ident = digest(c->ident, &color, sizeof color);
  }
  keep_config(c, config);
  if (config == NULL && parent != NULL) {
    c->nonblocking = parent->nonblocking;  // NCCL gives it the parent's
  }
  NCCL_LH(rc = NCCL(ncclCommSplit)(front_translate(comm), color, key,
                                   (ncclComm_t *)&c->real, config));
  ncclComm_t made = made_comm(c, rc);
  if (made != NULL) {
    *newcomm = made;
  }
  FRONT_LEAVE();
  leave(posted(rc), parent != NULL ? &parent->seq : NULL, 1, false);
  return rc;
}

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
  ncclResult_t rc;
  if (abort) {
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
    end_group(rc);
  }
  FRONT_LEAVE();
  return rc;
}

// The collectives: a step each; the communicator and the stream
// translated.
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
                (sendbuff, recvbuff, count, datatype, op, real_comm,
                 real_stream))
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
                (sendbuff, recvbuff, count, datatype, op, root, real_comm,
                 real_stream))
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
                (sendbuff, recvbuff, recvcount, datatype, op, real_comm,
                 real_stream))
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
static int64_t *
p2p_count(Comm *c, std::vector<int64_t> Comm::*counts, int peer)
{
  if (c == NULL || peer < 0 || peer >= (int)(c->*counts).size()) {
    return NULL;
  }
  return &(c->*counts)[peer];
}

extern "C" ncclResult_t
ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype, int peer,
         ncclComm_t comm, cudaStream_t stream)
{
  Comm *c = comm_of(comm);
  int64_t *sent = p2p_count(c, &Comm::sent, peer);
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
  int64_t *received = p2p_count(c, &Comm::received, peer);
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
