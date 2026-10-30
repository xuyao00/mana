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
// the config; and what the drain needs (step 3): the operations posted on
// it, and the bytes sent to and received from each peer.
//
// Step 2: the communicators work, but a checkpoint with a live one is
// refused (front_nccl_live(), at MANA_STEP_QUIESCE): NCCL's kernels wait
// for every member, so the GPUs would not quiesce before the drain exists.

#include <nccl.h>
#include <string.h>
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
  bool finalized;
  // For the drain: the operations posted, and point-to-point bytes per peer.
  int64_t seq;
  std::vector<int64_t> sent;
  std::vector<int64_t> received;
};

int live_comms;          // with objs_lock held
int cliques;

Comm *
comm_of(ncclComm_t comm)
{
  if (comm == NULL) {
    return NULL;
  }
  front_real_obj(comm, FRONT_NCCL);  // checks the handle
  return (Comm *)comm;
}

// A new communicator's record, once NCCL made it.  Its rank, size and
// device are the caller's, or -1: then NCCL's, which a non-blocking
// communicator has only once it is made.
ncclComm_t
link_comm(Comm *c, ncclComm_t real, int nranks, int rank, int device)
{
  c->real = real;
  c->ctx = front_object_ctx();
  c->nranks = nranks;
  c->rank = rank;
  c->device = device;
  if (nranks < 0) {
    NCCL_LH(NCCL(ncclCommCount)(real, &c->nranks));
  }
  if (rank < 0) {
    NCCL_LH(NCCL(ncclCommUserRank)(real, &c->rank));
  }
  if (device < 0) {
    NCCL_LH(NCCL(ncclCommCuDevice)(real, &c->device));
  }
  if (c->nranks > 0) {
    c->sent.assign(c->nranks, 0);
    c->received.assign(c->nranks, 0);
  }
  std::lock_guard<std::mutex> guard(objs_lock);
  obj_link(c, FRONT_NCCL);
  live_comms++;
  return (ncclComm_t)c;
}

void
unlink_comm(Comm *c)
{
  std::lock_guard<std::mutex> guard(objs_lock);
  obj_unlink(c);
  live_comms--;
  delete c;
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

// A collective, after it was posted.
void
count_collective(ncclComm_t comm)
{
  Comm *c = comm_of(comm);
  if (c != NULL) {
    __atomic_add_fetch(&c->seq, 1, __ATOMIC_RELAXED);
  }
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

}  // namespace

// MANA_STEP_QUIESCE: the communicators that exist.
int
front_nccl_live()
{
  std::lock_guard<std::mutex> guard(objs_lock);
  return live_comms;
}

extern "C" ncclResult_t
ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId commId,
                       int rank, ncclConfig_t *config)
{
  if (comm == NULL) {
    return ncclInvalidArgument;
  }
  FRONT_ENTER();
  front_before_library();
  ncclComm_t real = NULL;
  ncclResult_t rc;
  if (config != NULL) {
    NCCL_LH(rc = NCCL(ncclCommInitRankConfig)(&real, nranks, commId, rank,
                                              config));
  } else {
    NCCL_LH(rc = NCCL(ncclCommInitRank)(&real, nranks, commId, rank));
  }
  if (rc == ncclSuccess || (rc == ncclInProgress && real != NULL)) {
    Comm *c = new Comm();
    c->made = INIT_RANK;
    c->id = commId;
    keep_config(c, config);
    int device = 0;
    cudaError_t drc;
    FRONT_LH(drc = REAL(cudaGetDevice)(&device));
    *comm = link_comm(c, real, nranks, rank, drc == cudaSuccess ? device : -1);
  }
  FRONT_LEAVE();
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
  FRONT_ENTER();
  front_before_library();
  std::vector<ncclComm_t> real(ndev);
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclCommInitAll)(real.data(), ndev, devlist));
  if (rc == ncclSuccess) {
    int clique;
    {
      std::lock_guard<std::mutex> guard(objs_lock);
      clique = cliques++;
    }
    for (int i = 0; i < ndev; i++) {
      Comm *c = new Comm();
      c->made = INIT_ALL;
      c->clique = clique;
      comms[i] = link_comm(c, real[i], ndev, i,
                           devlist != NULL ? devlist[i] : i);
    }
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" ncclResult_t
ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm,
              ncclConfig_t *config)
{
  if (newcomm == NULL) {
    return ncclInvalidArgument;
  }
  FRONT_ENTER();
  ncclComm_t real = NULL;
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclCommSplit)(front_translate(comm), color, key, &real,
                                   config));
  // NCCL_SPLIT_NOCOLOR gives no communicator.
  if ((rc == ncclSuccess || rc == ncclInProgress) && real != NULL) {
    Comm *c = new Comm();
    c->made = SPLIT;
    c->parent = comm_of(comm);
    c->color = color;
    c->key = key;
    keep_config(c, config);
    *newcomm = link_comm(c, real, -1, -1, -1);
  } else if (rc == ncclSuccess) {
    *newcomm = NULL;
  }
  count_collective(comm);
  FRONT_LEAVE();
  return rc;
}

extern "C" ncclResult_t
ncclCommFinalize(ncclComm_t comm)
{
  FRONT_ENTER();
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclCommFinalize)(front_translate(comm)));
  if (rc == ncclSuccess || rc == ncclInProgress) {
    Comm *c = comm_of(comm);
    if (c != NULL) {
      c->finalized = true;
      count_collective(comm);
    }
  }
  FRONT_LEAVE();
  return rc;
}

// ncclCommDestroy and ncclCommAbort.
static ncclResult_t
end_comm(ncclComm_t comm, bool abort)
{
  FRONT_ENTER();
  ncclResult_t rc;
  if (abort) {
    NCCL_LH(rc = NCCL(ncclCommAbort)(front_translate(comm)));
  } else {
    NCCL_LH(rc = NCCL(ncclCommDestroy)(front_translate(comm)));
  }
  Comm *c = comm_of(comm);
  if (rc == ncclSuccess && c != NULL) {
    unlink_comm(c);
  }
  FRONT_LEAVE();
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

// The collectives: the communicator and the stream translated, then counted.
#define NCCL_COLLECTIVE(name, params, args)                         \
  extern "C" ncclResult_t name params                               \
  {                                                                 \
    FRONT_ENTER();                                                  \
    ncclResult_t rc;                                                \
    ncclComm_t real_comm = front_translate(comm);                   \
    cudaStream_t real_stream = front_real_stream(stream);           \
    NCCL_LH(rc = NCCL(name) args);                                  \
    if (rc == ncclSuccess || rc == ncclInProgress) {                \
      count_collective(comm);                                       \
    }                                                               \
    FRONT_LEAVE();                                                  \
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

// Point-to-point: the bytes, per peer, for the drain.
extern "C" ncclResult_t
ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype, int peer,
         ncclComm_t comm, cudaStream_t stream)
{
  FRONT_ENTER();
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclSend)(sendbuff, count, datatype, peer,
                              front_translate(comm),
                              front_real_stream(stream)));
  Comm *c = comm_of(comm);
  if ((rc == ncclSuccess || rc == ncclInProgress) && c != NULL && peer >= 0 &&
      peer < c->nranks) {
    __atomic_add_fetch(&c->sent[peer], (int64_t)(count * type_size(datatype)),
                       __ATOMIC_RELAXED);
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" ncclResult_t
ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype, int peer,
         ncclComm_t comm, cudaStream_t stream)
{
  FRONT_ENTER();
  ncclResult_t rc;
  NCCL_LH(rc = NCCL(ncclRecv)(recvbuff, count, datatype, peer,
                              front_translate(comm),
                              front_real_stream(stream)));
  Comm *c = comm_of(comm);
  if ((rc == ncclSuccess || rc == ncclInProgress) && c != NULL && peer >= 0 &&
      peer < c->nranks) {
    __atomic_add_fetch(&c->received[peer],
                       (int64_t)(count * type_size(datatype)),
                       __ATOMIC_RELAXED);
  }
  FRONT_LEAVE();
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
