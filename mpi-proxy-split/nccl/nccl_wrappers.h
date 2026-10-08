#ifndef NCCL_WRAPPERS_H
#define NCCL_WRAPPERS_H

#include "inflight.h"
#include "vcomm.h"
#include <nccl.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Job-level identity, latched from the first cross-process communicator the
 * application creates. These are NOT a substitute for a communicator's own
 * rank/size -- a sub-communicator has its own rank space, which lives in
 * vcomm_t. Use these only for job-wide logging and for the rank-0 role in
 * job-wide coordination.
 */
extern int g_my_rank;
extern int g_nranks;
extern volatile int ckpt_pending;

/* Helpers provided by libncclplugin.c */
void add_stream(cudaStream_t stream);
void init_rank_info(vcomm_t *v);

/*
 * Sequence gate, shared between the wrappers (which pass through it) and the
 * checkpoint path (which closes it and waits for the process to go quiet).
 *
 * The gate is conjunctive: a thread is held only when EVERY live communicator
 * has reached its consensus target. A rank that is behind on any comm is let
 * through on any comm, because its program order may require issuing elsewhere
 * first; the resulting overshoot is absorbed by re-running the consensus round.
 */
int nccl_gate_closed_locked(void);  /* call with the seq lock held */
void nccl_seq_lock(void);
void nccl_seq_unlock(void);
int nccl_activity(void); /* threads inside a NCCL call or an open group region */

/*
 * Resolve a real NCCL symbol when dlsym(RTLD_NEXT, ...) cannot see libnccl,
 * i.e. when the application reached it through an RTLD_LOCAL dlopen. Returns
 * NULL if NCCL is not mapped yet; failure is not cached.
 */
void *nccl_real_dlsym(const char *name);

/*
 * Direct access to the real NCCL entry points, for plugin-internal use only.
 * The checkpoint path must never re-enter the interposed names: that would
 * redo wrapper bookkeeping, and in the destroy case would free the registry
 * slot holding the metadata needed to rebuild.
 */
ncclResult_t call_real_ncclSend(const void *sendbuff, size_t count, ncclDataType_t datatype,
                               int peer, ncclComm_t real_comm, cudaStream_t stream);
ncclResult_t call_real_ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype, int peer,
                               ncclComm_t real_comm, cudaStream_t stream);
ncclResult_t call_real_ncclCommDestroy(ncclComm_t real_comm);
ncclResult_t call_real_ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId commId,
                                              int rank, ncclConfig_t *config);
ncclResult_t call_real_ncclCommInitAll(ncclComm_t *comms, int ndev, const int *devlist);
ncclResult_t call_real_ncclCommGetAsyncError(ncclComm_t real_comm, ncclResult_t *asyncError);

/*
 * The dlsym results are dropped across a restart and re-resolved eagerly, on the
 * checkpoint thread, before anything can need them. See the definition of
 * resolve_wrapper_pointers() for why leaving them to lazy resolution is not safe
 * there.
 */
void clear_wrapper_pointers(void);
void resolve_wrapper_pointers(void);

/*
 * Synthesized collectives, built out of the wrapped ncclSend/ncclRecv. Not part
 * of the NCCL API; the caller owns ncclGroupStart/ncclGroupEnd.
 */
ncclResult_t ncclGather_p2p(const void *sendbuff, void *recvbuff, size_t sendcount,
                            ncclDataType_t datatype, int root, ncclComm_t comm,
                            cudaStream_t stream);
ncclResult_t ncclScatter_p2p(const void *sendbuff, void *recvbuff, size_t sendcount,
                             ncclDataType_t datatype, int root, ncclComm_t comm,
                             cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
