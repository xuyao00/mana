// Checkpointing NCCL communicators (nccl_ckpt.c), called from MANA's event
// hook (mpi_plugin.cpp) on the checkpoint thread, on every rank.
#ifndef MANA_NCCL_CKPT_H
#define MANA_NCCL_CKPT_H

#ifdef __cplusplus
extern "C" {
#endif

// DMTCP_EVENT_INIT.
void mana_nccl_init(void);
// DMTCP_EVENT_PRESUSPEND, before MANA's MPI drain: start the cut.
void mana_nccl_ckpt_begin(void);
// A round of the NCCL consensus, in each round of MANA's MPI drain
// (seq_num.cpp, g_drain_round_hook): returns whether this rank's NCCL
// communicators are at their targets.
int mana_nccl_consensus_round(int round);
// MANA's MPI gate (g_other_settled): no NCCL communicator behind its target.
int mana_nccl_settled(void);
// After the drain converged: wait until no thread is in NCCL.
void mana_nccl_ckpt_quiesce(void);
// DMTCP_EVENT_PRESUSPEND, after MANA's MPI drain (the lower half is closed),
// before the CUDA plugin checkpoints the GPU: drain point-to-point messages,
// and destroy every communicator.
void mana_nccl_ckpt_teardown(void);
// DMTCP_EVENT_RESUME.
void mana_nccl_resume(void);
// DMTCP_EVENT_RESTART: forget the lower half's NCCL and the real handles.
void mana_nccl_restart(void);
// DMTCP_EVENT_RUNNING_AFTER, once the GPU and the lower half's MPI are back,
// before the application's threads may enter the lower half: create every
// communicator again.  Does nothing at launch.
void mana_nccl_rebuild(void);

#ifdef __cplusplus
}
#endif

#endif
