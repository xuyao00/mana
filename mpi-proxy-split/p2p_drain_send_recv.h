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

#ifndef _P2P_SEND_RECV_H
#define _P2P_SEND_RECV_H

#include <unordered_set>
#include "dmtcp.h"
#include "dmtcpalloc.h"
#include "p2p_log_replay.h"

#ifdef DEBUG_P2P
extern int *g_sendBytesByRank; // Number of bytes sent to other ranks
extern int *g_rsendBytesByRank; // Number of bytes sent to other ranks by MPI_Rsend
extern int *g_bytesSentToUsByRank; // Number of bytes other ranks sent to us
extern int *g_recvBytesByRank; // Number of bytes received from other ranks
#endif
extern int64_t global_sent_messages, global_recv_messages;
extern int64_t local_sent_messages, local_recv_messages;
extern std::unordered_set<MPI_Comm> active_comms;
extern dmtcp::vector<mpi_message_t*> g_message_queue;

void initialize_drain_send_recv();
void registerLocalSendsAndRecvs();

// Drain all in-flight point-to-point messages by completing nonblocking
// receives and probing for unexpected messages until global_sent ==
// global_recv.
void drainInFlightP2p();

// Single entry point for draining all P2P communications before
// checkpoint: drains in-flight messages.  (Baseline P2P design: MPI_Send
// and MPI_Recv are MPI_Isend/MPI_Irecv + MPI_Wait, so no thread is blocked
// in the lower half and no pending-Recv exchange is needed.)
void drainP2p();

// What the drain did at one checkpoint on this rank, for measuring it.
// Times are in microseconds.  With MANA_DRAIN_STATS set, rank 0 prints the
// maximum times and the total counts over all ranks at every checkpoint
// (reportDrainStats()).  The fields are those of branch perf, so that the
// two print the same statistics; on this branch the pending-Recv exchange
// doesn't exist, and its times and counts (unblock, blocked, dummies) are 0.
struct DrainStats {
  uint64_t t_collective;     // Collective Clock drain, NBCs, barrier
  uint64_t t_inflight;       // drainInFlightP2p()
  uint64_t t_register;       //   exchanging the send/recv counters
  uint64_t t_complete;       //   completing pending MPI_Isend/MPI_Irecv
  uint64_t t_probe;          //   probing communicators, buffering messages
  uint64_t t_isends;         //   completing the remaining MPI_Isends
  uint64_t t_unblock;        // unblockPendingRecvs()
  uint64_t t_publish;        //   publishing whether blocked in MPI_Recv
  uint64_t t_published;      //   barrier after publishing
  uint64_t t_post;           //   posting the dummy to its sender
  uint64_t t_posted;         //   barrier after posting
  uint64_t t_dispatch;       //   sending the dummies posted to this rank
  uint64_t t_dispatched;     //   barrier after sending
  uint64_t t_wait_lower_half;  // wait_for_threads_to_leave_lower_half()
  int64_t iterations;        // rounds of the in-flight drain
  int64_t comms_probed;
  int64_t iprobes;
  int64_t drained_msgs;      // moved to MANA's buffer
  int64_t drained_bytes;
  int64_t irecvs_completed;  // pending MPI_Irecvs that received a message
  int64_t isends_completed;
  int64_t blocked;           // ranks blocked in MPI_Recv
  int64_t dummies;           // dummy messages sent
  int64_t kvdb_requests;     // requests to the coordinator's database
  int64_t barriers;          // global barriers
};
extern DrainStats g_drain_stats;
uint64_t drainStatsNow();   // microseconds
void resetDrainStats();
void reportDrainStats();

int drainRemainingP2pMsgs(int source);
int recvMsgIntoInternalBuffer(MPI_Status status);
bool existsMatchingMsgBuffer(int source, int tag, MPI_Comm comm, int *flag,
                             MPI_Status *status);
int consumeMatchingMsgBuffer(void *buf, int count, MPI_Datatype datatype,
                             int source, int tag, MPI_Comm comm,
                             MPI_Status *mpi_status, int size);
void removePendingSendRequests();
void resetDrainCounters();
int localRankToGlobalRank(int localRank, MPI_Comm localComm);
#endif
