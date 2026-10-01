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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <map>
#include <algorithm>
#include <vector>
#include "kvdb.h"
#include "jassert.h"
#include "p2p_drain_send_recv.h"
#include "p2p_log_replay.h"
#include "mpi_nextfunc.h"
#include "virtual_id.h"

using namespace dmtcp;
using dmtcp::kvdb::KVDBRequest;
using dmtcp::kvdb::KVDBResponse;

extern int MPI_Alltoall_internal(const void *sendbuf, int sendcount,
                                 MPI_Datatype sendtype, void *recvbuf,
                                 int recvcount, MPI_Datatype recvtype,
                                 MPI_Comm comm);
// Defined with C linkage in mpi-wrappers/mpi_request_wrappers.cpp.
extern "C" int MPI_Test_internal(MPI_Request *, int *flag, MPI_Status *status,
                                 bool isRealRequest);
// FIXME: These three internal functions were added to avoid record and replay.
// Since we no longer record MPI_Comm and MPI_Group related functions, these
// internal functions can be removed.
extern int MPI_Comm_create_group_internal(MPI_Comm comm, MPI_Group group,
                                          int tag, MPI_Comm *newcomm);
extern int MPI_Comm_free_internal(MPI_Comm *comm);
extern int MPI_Comm_group_internal(MPI_Comm comm, MPI_Group *group);
extern int MPI_Group_free_internal(MPI_Group *group);
#ifdef DEBUG_P2P
int *g_sendBytesByRank; // Number of bytes sent to other ranks
int *g_rsendBytesByRank; // Number of bytes sent to other ranks by MPI_rsend
int *g_bytesSentToUsByRank; // Number of bytes other ranks sent to us
int *g_recvBytesByRank; // Number of bytes received from other ranks
#endif
int64_t global_sent_messages = 0, global_recv_messages = 0;
int64_t local_sent_messages = 0, local_recv_messages = 0;
std::unordered_set<MPI_Comm> active_comms;
dmtcp::vector<mpi_message_t*> g_message_queue;

DrainStats g_drain_stats;

// This checkpoint's database in the coordinator (set by drainP2p()), and the
// round of the in-flight drain.  Keys are never reused: the counters and the
// bitmap of blocked ranks only add up (INCRBY, OR), and the coordinator
// keeps its databases from one computation to the next unless it restarts
// one, so no round has to reset them first.
static char g_drain_db[128];
static int g_drain_round;

uint64_t
drainStatsNow()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

void
resetDrainStats()
{
  memset(&g_drain_stats, 0, sizeof(g_drain_stats));
}

// The drain's requests to the coordinator, counted in g_drain_stats.
// kvGet() leaves *val unchanged if there is no such key.
static inline void
kvGet(const char *db, const char *key, int64_t *val)
{
  g_drain_stats.kvdb_requests++;
  kvdb::get64(db, key, val);
}

static inline void
kvIncr(const char *db, const char *key, int64_t val)
{
  g_drain_stats.kvdb_requests++;
  kvdb::request64(KVDBRequest::INCRBY, db, key, val);
}

static inline void
globalBarrier(const char *name)
{
  g_drain_stats.barriers++;
  dmtcp_global_barrier(name);
}

void
initialize_drain_send_recv()
{
  getLocalRankInfo();
#ifdef DEBUG_P2P
  g_sendBytesByRank = (int*)JALLOC_HELPER_MALLOC(g_world_size * sizeof(int));
  g_rsendBytesByRank = (int*)JALLOC_HELPER_MALLOC(g_world_size * sizeof(int));
  g_bytesSentToUsByRank =
    (int*)JALLOC_HELPER_MALLOC(g_world_size * sizeof(int));
  g_recvBytesByRank = (int*)JALLOC_HELPER_MALLOC(g_world_size * sizeof(int));
  memset(g_sendBytesByRank, 0, g_world_size * sizeof(int));
  memset(g_rsendBytesByRank, 0, g_world_size * sizeof(int));
  memset(g_bytesSentToUsByRank, 0, g_world_size * sizeof(int));
  memset(g_recvBytesByRank, 0, g_world_size * sizeof(int));
#endif
  active_comms.insert(MPI_COMM_WORLD);
  active_comms.insert(MPI_COMM_SELF);
}

// Sums the ranks' counts of sent and received messages, in this round's
// keys.  Since the next round uses other keys, a rank may go on as soon as
// it has read the sums: every rank reads the same ones, and so decides the
// same whether to drain again.
void
registerLocalSendsAndRecvs()
{
  char sent_key[32], recv_key[32];
  uint64_t t0 = drainStatsNow();
  snprintf(sent_key, sizeof(sent_key), "sent_%d", g_drain_round);
  snprintf(recv_key, sizeof(recv_key), "recv_%d", g_drain_round);
  g_drain_round++;
  kvIncr(g_drain_db, sent_key, local_sent_messages);
  kvIncr(g_drain_db, recv_key, local_recv_messages);
  globalBarrier("MPI:Register-p2p-send-recv");
  global_sent_messages = 0;
  global_recv_messages = 0;
  kvGet(g_drain_db, sent_key, &global_sent_messages);
  kvGet(g_drain_db, recv_key, &global_recv_messages);
  g_drain_stats.t_register += drainStatsNow() - t0;
}

// status was received by MPI_Iprobe
int
recvMsgIntoInternalBuffer(MPI_Status status, MPI_Comm comm)
{
  int count = 0;
  int size = 0;
  MPI_Get_count(&status, MPI_BYTE, &count);
  MPI_Type_size(MPI_BYTE, &size);
  JASSERT(size == 1);
  void *buf = JALLOC_HELPER_MALLOC(count);
  // Receive the probed message directly in the lower half: the MPI_Recv
  // wrapper would wait out the drain that is running now.
  MPI_Comm realComm = get_real_id((mana_mpi_handle){.comm = comm}).comm;
  JUMP_TO_LOWER_HALF(lh_info->fsaddr);
  int retval = NEXT_FUNC(Recv)(buf, count, MPI_BYTE,
                               status.MPI_SOURCE, status.MPI_TAG,
                               realComm, MPI_STATUS_IGNORE);
  JASSERT(retval == MPI_SUCCESS);
  RETURN_TO_UPPER_HALF();
  // The wrapper would have incremented local_recv_messages for this
  // real receive; we must do it explicitly when bypassing the wrapper,
  // so the global drain test in drainInFlightP2p() sees a balanced count.
  local_recv_messages++;

  mpi_message_t *message = (mpi_message_t *)JALLOC_HELPER_MALLOC(sizeof(mpi_message_t));
  message->buf        = buf;
  message->count      = count;
  message->datatype   = MPI_BYTE;
  message->comm       = comm;
  message->status     = status;
  message->size       = size * count;

  // queue it
  g_message_queue.push_back(message);

  return count;
}

// Go through each pending MPI_Irecv (and MPI_Isend) and try to complete
// them before checkpointing.
int
completePendingP2pRequests()
{
  int bytesReceived = 0;
  for (MPI_Request request : pendingRequestsInPostingOrder()) {
    mpi_nonblocking_call_t call;
    if (!getPendingCall(request, &call)) {
      continue;  // The application completed it meanwhile.
    }
    int flag = 0;
    MPI_Status status;
    // This is needed if an MPI_Isend was called earlier.  Without this,
    // large messages within the same node will fail under MPICH and some
    // other MPIs.  A previous call to MPI_Irecv caused only the metadata to be
    // exchanged.  So, MPI_Iprobe succeeds and MPI_Irecv will later fail, unless
    // we force the sending of data via MPI_Test.
    MPI_Test_internal(&request, &flag, &status, false);
    if (flag) {
      if (call.type == IRECV_REQUEST) {
        int size = 0;
        MPI_Type_size(call.datatype, &size);
        int worldRank = localRankToGlobalRank(status.MPI_SOURCE,
                                              call.comm);
#ifdef DEBUG_P2P
        g_recvBytesByRank[worldRank] += call.count * size;
#endif
        local_recv_messages++;
        g_drain_stats.irecvs_completed++;
      } else if (call.type == ISEND_REQUEST) {
        g_drain_stats.isends_completed++;
      }
      update_virt_id((mana_mpi_handle){.request = request},(mana_mpi_handle){.request = MPI_REQUEST_NULL});
      clearPendingRequestFromLog(request);
    } else {
      /*  We go on to the next request even if the MPI_Test fails.
       * Otherwise, the message we are waiting for will be sent
       * after the checkpoint. This can result in an infinite loop.
       *
       * NOTE: This function will be called only if the global arrays
       * do not match. This can happen if a second sender has sent
       * a message to us, and we will receive the message only
       * after the checkpoint. The following diagram is an example:
       *
       * RANK 0           RANK 1              RANK 2        TIME
       *                                    Send to Rank 1   |
       *                Recv from Rank 0                     |
       * =====CKPT=======CKPT======CKPT======CKPT========    |
       *                Recv from Rank 2                     |
       * Send to Rank 1                                      V
       */
    }
  }
  return bytesReceived;
}

int
drainRemainingP2pMsgs()
{
  int bytesReceived = 0;
  // Probe every communicator the application may be sending on: the
  // predefined ones in active_comms, and those in the virtual-ID table.
  // (MANA's own g_world_comm carries the Collective Clock's messages,
  // which are not counted as sent or received.)
  std::vector<MPI_Comm> comms(active_comms.begin(), active_comms.end());
  for (MPI_Comm virtComm : live_virt_comms()) {
    if (virtComm != g_world_comm) {
      comms.push_back(virtComm);
    }
  }
  for (MPI_Comm comm : comms) {
    // If the communicator is MPI_COMM_NULL, skip it.
    // MPI_COMM_NULL can be returned from functions like MPI_Comm_split
    // if the color is specified on only one side of the intercommunicator, or
    // specified as MPI_UNDEFINED by the program. In this case, the MPI function
    // still returns MPI_SUCCESS. So the MPI_COMM_NULL can be added to the
    // active communicator set `active_comms'.
    if (comm == MPI_COMM_NULL) {
      continue;
    }
    // Skip a communicator that the application has freed since.
    if (!is_predefined_id((mana_mpi_handle){.comm = comm}) &&
        lookup_virt_id_entry((mana_mpi_handle){.comm = comm}) == NULL) {
      continue;
    }
    g_drain_stats.comms_probed++;
    int flag = 1;
    while (flag) {
      MPI_Status status;
      g_drain_stats.iprobes++;
      int retval = MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag,
                              &status);
      JASSERT(retval == MPI_SUCCESS);
      if (flag) {
        MPI_Request matched_request = MPI_REQUEST_NULL;
        // Check if there are pending MPI_Irecv's that matches the envelope of the
        // probed message.  MPI matches the earliest posted one.
        for (MPI_Request req : pendingRequestsInPostingOrder()) {
          mpi_nonblocking_call_t call;
          if (getPendingCall(req, &call) &&
              call.type == IRECV_REQUEST &&
              call.comm == comm &&
              (call.tag == status.MPI_TAG || call.tag == MPI_ANY_TAG) &&
              (call.remote_node == status.MPI_SOURCE ||
               call.remote_node == MPI_ANY_SOURCE)) {
            matched_request = req;
            break;
          }
        }
        if (matched_request != MPI_REQUEST_NULL) {
          // If there are matched pending MPI_Irecv's, wait
          // on the request to complete the communication.
          // Otherwise, the message will be drained to the MANA internal buffer,
          // and then be received out of order, after restart.
          // As in completePendingP2pRequests(), keep the virtual request:
          // the application still holds it, and its own MPI_Wait/MPI_Test
          // will free it on seeing the real request MPI_REQUEST_NULL.
          // (The MPI_Wait wrapper would free it here, from the checkpoint
          // thread, and leave the application with a stale handle.)
          int done = 0;
          MPI_Status recv_status;
          while (!done) {
            MPI_Test_internal(&matched_request, &done, &recv_status, false);
          }
          local_recv_messages++;
          g_drain_stats.irecvs_completed++;
          update_virt_id((mana_mpi_handle){.request = matched_request},
                         (mana_mpi_handle){.request = MPI_REQUEST_NULL});
          clearPendingRequestFromLog(matched_request);
        } else {
          int bytes = recvMsgIntoInternalBuffer(status, comm);
          bytesReceived += bytes;
          g_drain_stats.drained_msgs++;
          g_drain_stats.drained_bytes += bytes;
        }
      }
    }
  }
  return bytesReceived;
}

// Once every message that was sent has been received, an MPI_Isend that
// the application hasn't waited for yet is complete, but its request is
// still pending, and restart can't replay a send (see
// replayMpiP2pOnRestart()).  Complete such requests the way
// completePendingP2pRequests() does: the application's MPI_Wait/MPI_Test
// then sees the real request MPI_REQUEST_NULL.
static void
completePendingIsends()
{
  for (MPI_Request request : pendingRequestsInPostingOrder()) {
    if (pendingRequestType(request) != ISEND_REQUEST) {
      continue;
    }
    int flag = 0;
    MPI_Status status;
    while (!flag) {
      MPI_Test_internal(&request, &flag, &status, false);
    }
    update_virt_id((mana_mpi_handle){.request = request},
                   (mana_mpi_handle){.request = MPI_REQUEST_NULL});
    clearPendingRequestFromLog(request);
    g_drain_stats.isends_completed++;
  }
}

void
drainInFlightP2p()
{
  uint64_t t0 = drainStatsNow();
  registerLocalSendsAndRecvs();
  while (global_sent_messages > global_recv_messages) {
    g_drain_stats.iterations++;
    // If pending MPI_Irecv or MPI_Isend, use MPI_Test to try to complete it.
    uint64_t t = drainStatsNow();
    completePendingP2pRequests();
    g_drain_stats.t_complete += drainStatsNow() - t;
    // If MPI_Irecv not posted but msg was sent, use MPI_Iprobe to drain msg.
    t = drainStatsNow();
    drainRemainingP2pMsgs();
    g_drain_stats.t_probe += drainStatsNow() - t;
    // Update global recv coutner.
    registerLocalSendsAndRecvs();
  }
  uint64_t t = drainStatsNow();
  completePendingIsends();
  g_drain_stats.t_isends += drainStatsNow() - t;
  g_drain_stats.t_inflight = drainStatsNow() - t0;
}

// FIXME: existsMatchingMsgBuffer and consumeMatchingMsgBuffer both search
// in the g_message_queue with the same condition. Maybe we can
// combine them into one function.
bool
existsMatchingMsgBuffer(int source, int tag, MPI_Comm comm, int *flag,
                        MPI_Status *status)
{
  bool ret = false;
  dmtcp::vector<mpi_message_t*>::iterator req =
    std::find_if(g_message_queue.begin(), g_message_queue.end(),
                 [source, tag, comm](const mpi_message_t *msg)
                 { return ((msg->status.MPI_SOURCE == source) ||
                           (source == MPI_ANY_SOURCE)) &&
                          ((msg->status.MPI_TAG == tag) ||
                           (tag == MPI_ANY_TAG)) &&
                          ((msg->comm == comm)); });
  if (req != std::end(g_message_queue)) {
    *flag = 1;
    *status = (*req)->status;
    ret = true;
  }
  return ret;
}

int
consumeMatchingMsgBuffer(void *buf, int count, MPI_Datatype datatype,
                         int source, int tag, MPI_Comm comm,
                         MPI_Status *mpi_status, int size)
{
  mpi_message_t *foundMsg = NULL;
  dmtcp::vector<mpi_message_t*>::iterator req =
    std::find_if(g_message_queue.begin(), g_message_queue.end(),
                 [source, tag, comm](const mpi_message_t *msg)
                 { return ((msg->status.MPI_SOURCE == source) ||
                           (source == MPI_ANY_SOURCE)) &&
                          ((msg->status.MPI_TAG == tag) ||
                           (tag == MPI_ANY_TAG)) &&
                          ((msg->comm == comm)); });
  // This should never happen (since the caller should always check first using
  // existsMatchingMsgBuffer())!
  JASSERT(req != std::end(g_message_queue))(count)(datatype)
         .Text("Unexpected error: no message in the queue matches the given"
               " attributes.");
  foundMsg = *req;

  // The message was drained as MPI_BYTE, so it is packed.  Unpack it into
  // the receive buffer, whose datatype may have holes or a stride (a raw
  // copy would scramble such data), and only as many whole elements as the
  // message holds.
  int type_size = (count > 0) ? size / count : 0;
  int elements = (type_size > 0) ? foundMsg->size / type_size : 0;
  if (elements > count) {
    elements = count;
  }
  if (elements > 0) {
    int position = 0;
    MPI_Datatype realType =
      get_real_id((mana_mpi_handle){.datatype = datatype}).datatype;
    // Any communicator will do for unpacking; the message's may be freed.
    MPI_Comm realComm =
      get_real_id((mana_mpi_handle){.comm = MPI_COMM_SELF}).comm;
    int retval;
    JUMP_TO_LOWER_HALF(lh_info->fsaddr);
    retval = NEXT_FUNC(Unpack)(foundMsg->buf, foundMsg->size, &position,
                               buf, elements, realType, realComm);
    RETURN_TO_UPPER_HALF();
    JASSERT(retval == MPI_SUCCESS)(retval);
  }
  *mpi_status = foundMsg->status;
  g_message_queue.erase(req);
  JALLOC_HELPER_FREE(foundMsg->buf);
  JALLOC_HELPER_FREE(foundMsg);
  return MPI_SUCCESS;
}

void
drainP2p()
{
  // Name this checkpoint's database by the computation (the same on every
  // rank, and after a restart) and a count of this process's drains (the
  // same on every rank, and restored with the checkpoint image).
  static int drains = 0;
  DmtcpUniqueProcessId id = dmtcp_get_computation_id();
  snprintf(g_drain_db, sizeof(g_drain_db), "/plugin/MANA/p2p-%llx-%llx-%x-%d",
           (unsigned long long)id._hostid, (unsigned long long)id._time,
           (unsigned int)id._pid, ++drains);
  g_drain_round = 0;
  // Baseline P2P design: MPI_Recv never blocks in the lower half (it is
  // MPI_Irecv + MPI_Wait), so there is no pending-Recv exchange.
  drainInFlightP2p();
}

void
resetDrainCounters()
{
#ifdef DEBUG_P2P
  memset(g_sendBytesByRank, 0, g_world_size * sizeof(int));
  memset(g_rsendBytesByRank, 0, g_world_size * sizeof(int));
  memset(g_bytesSentToUsByRank, 0, g_world_size * sizeof(int));
  memset(g_recvBytesByRank, 0, g_world_size * sizeof(int));
#endif
}

int
localRankToGlobalRank(int localRank, MPI_Comm localComm)
{
  int worldRank;
  // FIXME: For interface8, use the new architecture.
  // This only works for interface7
  MPI_Group worldGroup, localGroup;
  MPI_Comm realComm = get_real_id((mana_mpi_handle){.comm = localComm}).comm;
  JUMP_TO_LOWER_HALF(lh_info->fsaddr);
  NEXT_FUNC(Comm_group)(MPI_COMM_WORLD, &worldGroup);
  NEXT_FUNC(Comm_group)(realComm, &localGroup);
  NEXT_FUNC(Group_translate_ranks)(localGroup, 1, &localRank,
                                   worldGroup, &worldRank);
  NEXT_FUNC(Group_free)(&worldGroup);
  NEXT_FUNC(Group_free)(&localGroup);
  RETURN_TO_UPPER_HALF();
  return worldRank;
}

// With MANA_DRAIN_STATS set, rank 0 prints what the drain of this
// checkpoint did: the maximum times and the total counts over all ranks.
// (This adds requests to the coordinator and a barrier; without the
// variable, nothing.)
void
reportDrainStats()
{
  if (getenv("MANA_DRAIN_STATS") == NULL) {
    return;
  }
  static int checkpoint = 0;
  checkpoint++;
  char db[sizeof(g_drain_db) + 16];
  snprintf(db, sizeof(db), "%s-stats", g_drain_db);  // This checkpoint's
  const DrainStats &d = g_drain_stats;
  struct { const char *name; int64_t value; bool is_time; } metrics[] = {
    {"collective", (int64_t)d.t_collective, true},
    {"inflight", (int64_t)d.t_inflight, true},
    {"register", (int64_t)d.t_register, true},
    {"complete", (int64_t)d.t_complete, true},
    {"probe", (int64_t)d.t_probe, true},
    {"isends", (int64_t)d.t_isends, true},
    {"unblock", (int64_t)d.t_unblock, true},
    {"publish", (int64_t)d.t_publish, true},
    {"published", (int64_t)d.t_published, true},
    {"post", (int64_t)d.t_post, true},
    {"posted", (int64_t)d.t_posted, true},
    {"dispatch", (int64_t)d.t_dispatch, true},
    {"dispatched", (int64_t)d.t_dispatched, true},
    {"wait_lower_half", (int64_t)d.t_wait_lower_half, true},
    {"iterations", d.iterations, true},  // The same on every rank
    {"comms_probed", d.comms_probed, false},
    {"iprobes", d.iprobes, false},
    {"drained_msgs", d.drained_msgs, false},
    {"drained_bytes", d.drained_bytes, false},
    {"irecvs_completed", d.irecvs_completed, false},
    {"isends_completed", d.isends_completed, false},
    {"blocked", d.blocked, false},
    {"dummies", d.dummies, false},
    {"kvdb_requests", d.kvdb_requests, false},
    {"barriers", d.barriers, true},      // The same on every rank
  };
  const int n = sizeof(metrics) / sizeof(metrics[0]);
  for (int i = 0; i < n; i++) {
    kvdb::request64(metrics[i].is_time ? KVDBRequest::MAX : KVDBRequest::INCRBY,
                    db, metrics[i].name, metrics[i].value);
  }
  dmtcp_global_barrier("MPI:Drain-Stats");
  if (g_world_rank != 0) {
    return;
  }
  std::map<std::string, long> v;
  for (int i = 0; i < n; i++) {
    int64_t value = 0;
    kvdb::get64(db, metrics[i].name, &value);
    v[metrics[i].name] = (long)value;
  }
  fprintf(stderr,
          "MANA drain stats, checkpoint %d, %d ranks (us: max over ranks):\n"
          "  collective %ld | in-flight %ld (register %ld, complete %ld, "
          "probe %ld, isends %ld) | unblock %ld (publish %ld, barrier %ld, "
          "post %ld, barrier %ld, dispatch %ld, barrier %ld) | "
          "wait-lower-half %ld\n"
          "  iterations %ld, barriers %ld; totals: comms probed %ld, "
          "iprobes %ld, drained %ld msgs %ld bytes, irecvs completed %ld, "
          "isends completed %ld, blocked %ld, dummies %ld, "
          "kvdb requests %ld\n",
          checkpoint, g_world_size, v["collective"], v["inflight"],
          v["register"], v["complete"], v["probe"], v["isends"], v["unblock"],
          v["publish"], v["published"], v["post"], v["posted"],
          v["dispatch"], v["dispatched"], v["wait_lower_half"],
          v["iterations"], v["barriers"], v["comms_probed"], v["iprobes"],
          v["drained_msgs"], v["drained_bytes"], v["irecvs_completed"],
          v["isends_completed"], v["blocked"], v["dummies"],
          v["kvdb_requests"]);
}
