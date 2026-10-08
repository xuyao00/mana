#ifndef SEQ_NUM_H
#define SEQ_NUM_H

#include <mpi.h>
#include <pthread.h>
#include <unordered_map>

typedef enum _phase_t {
  IN_CS,
  IS_READY
} phase_t;

// Global communicator for MANA internal use
extern MPI_Comm g_world_comm;

// The main functions of the sequence number algorithm for MPI collectives
void commit_begin(MPI_Comm comm);
void commit_finish(MPI_Comm comm);

void drain_mpi_collective();

// One checkpoint cut of MPI and NCCL collectives (nccl/nccl_ckpt.c).  Each
// round of drain_mpi_collective() first calls g_drain_round_hook, which runs
// a round of the NCCL consensus and returns whether this rank's NCCL
// communicators are at their targets; the round converges only if MPI and
// NCCL do.  An application thread waits at a gate (commit_begin(), the NCCL
// wrappers) only if both are at their targets (g_other_settled()): it may
// owe the other library progress that a peer waits for.  NULL: MPI alone.
extern int (*g_drain_round_hook)(int round);
extern int (*g_other_settled)(void);
// For the NCCL wrappers: whether MPI's cut needs this rank to progress.
extern "C" int mana_mpi_behind_target(void);
void complete_pending_nonblocking_collectives();
int test_pending_nonblocking_collectives();
void share_seq_nums();
int check_seq_nums();
void print_seq_nums();
void seq_num_init();
void seq_num_reset();
void seq_num_destroy();

extern std::unordered_map<unsigned int, unsigned long> seq_num;
extern std::unordered_map<unsigned int, unsigned long> target;
extern std::unordered_map<MPI_Comm, unsigned int> ggid_table;
typedef std::pair<unsigned int, unsigned long> comm_seq_pair_t;

#endif // ifndef SEQ_NUM_H
