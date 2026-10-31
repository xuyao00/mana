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

// What libmana.so, the core of MANA's upper half, gives its layers: MPI
// (libmana_mpi.so) and CUDA (cuda-front/libmana_cuda.so).  The core is the
// DMTCP plugin (mana_core.cpp).  It keeps the split process (the upper half's
// memory, FS bases, signals and fds), and at each checkpoint and restart it
// calls the steps of the layers that have registered.

#ifndef _MANA_CORE_H
#define _MANA_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// The layers, in the order that the core calls them.
enum ManaLayer {
  MANA_LAYER_MPI,
  MANA_LAYER_CUDA,
  MANA_NUM_LAYERS
};

// The steps that the core calls, in this order within each DMTCP event, and
// on the checkpoint thread except for MANA_STEP_THREAD_EXIT and
// MANA_STEP_DRAIN_MEMOPS.  The layers take a step in their order, or in
// reverse order where noted.  A layer ignores the steps it has no use for.
enum ManaStep {
  // DMTCP_EVENT_PRESUSPEND
  MANA_STEP_WAIT_FOR_INIT = 1,  // until the layer can drain (MPI_Init)
  MANA_STEP_AGREE,      // in reverse order, in passes: bring the ranks to a
                        // point where none waits for another (NCCL's drain,
                        // MPI's Collective Clock), with the application's
                        // threads running; see mana_drain_again()
  MANA_STEP_AGREED,     // then, in reverse order: whether the rank has run
                        // past where the layer's agreement left it
  MANA_STEP_DRAIN,      // in reverse order: complete the communication in
                        // flight.  The MPI layer's closes the lower half;
                        // the core closes it after the step, if still open
  MANA_STEP_QUIESCE,    // --cuda, the lower half closed: wait for the GPUs
  MANA_STEP_STASH,      // then, after a global barrier: copy device data
  // DMTCP_EVENT_PRECHECKPOINT, with the threads suspended
  MANA_STEP_PRECHECKPOINT,  // before the core names the checkpoint directory
  MANA_STEP_SAVE,       // then: write files there
  // DMTCP_EVENT_RESUME, in reverse order
  MANA_STEP_RESUME,
  // DMTCP_EVENT_RESTART
  MANA_STEP_RESTART,    // rebuild in the new lower half
  MANA_STEP_RESTART_PEERS,  // --cuda, after a global barrier: what needs
                            // the other ranks' MANA_STEP_RESTART (CUDA IPC)
  MANA_STEP_REPLAY,     // then: replay what was pending (MPI P2P, files)
  // PTHREAD_RETURN or _EXIT, on the exiting thread, in reverse order
  MANA_STEP_THREAD_EXIT,
  // DMTCP_EVENT_EXIT, in reverse order
  MANA_STEP_EXIT,
  // CUDA only: an MPI call waits for the application's pending copies and
  // sets (LOWER_HALF_DISABLE_CKPT(), lower_half_ckpt.h)
  MANA_STEP_DRAIN_MEMOPS,
};

typedef void (*mana_layer_fn_t)(int step);

// Called by a layer's library when it is loaded.
void mana_register_layer(int layer, mana_layer_fn_t fn);
// This process's rank, from the MPI layer: it names the checkpoint
// directory (ckpt_rank_N).
void mana_set_rank(int rank);
// A "name=value" line for the checkpoint's header.mana, which mana_restart
// reads.
void mana_set_header(const char *name, const char *value);
// The checkpoint directory (ckpt_rank_N), from MANA_STEP_SAVE on.
const char *mana_ckpt_dir(void);
// DMTCP's key-value store, which takes one caller at a time from the
// application's threads.
int mana_kv_set(const char *db, const char *key, const char *val);
int mana_kv_get(const char *db, const char *key, char *val, size_t size);
// From the checkpoint thread during a checkpoint: 64-bit values in a
// database of this checkpoint's own, which "what" (at most 24 characters)
// names; the coordinator keeps its databases across checkpoints.  An update
// adds val to the key's value (MANA_KV_ADD) or keeps the larger
// (MANA_KV_MAX); a new key starts at 0, and mana_ckpt_kv_get() gives 0 for
// a key not there.
enum ManaKvOp { MANA_KV_ADD, MANA_KV_MAX };
void mana_ckpt_kv_update(const char *what, const char *key, int op,
                         int64_t val);
int64_t mana_ckpt_kv_get(const char *what, const char *key);
// A barrier of every rank's checkpoint thread.
void mana_global_barrier(const char *id);

// The agreement (MANA_STEP_AGREE).  A layer's gate holds an application
// thread at the layer's next call once this rank owes its peers nothing in
// that layer.  But the program may interleave the layers' calls (NCCL and
// MPI), so a rank that owes one layer must pass every gate: each layer says
// whether this rank owes in it, from the application's threads or the
// checkpoint thread, and its gate lets threads through while another layer
// says so.
void mana_drain_set_behind(int layer, int behind);
int mana_drain_behind_elsewhere(int layer);
// So a later layer's agreement may let a rank run past where an earlier
// layer's left it.  A layer that finds so in its MANA_STEP_AGREED asks for
// another pass, which every rank must do alike.  The core runs passes until
// one in which no layer asks.
void mana_drain_again(void);
// The pass, from 0.
int mana_drain_pass(void);

#ifdef __cplusplus
}

// RUNNING outside a checkpoint or restart.  The MPI layer sets the others.
enum mana_state_t {
  UNKNOWN_STATE,
  RUNNING,
  CKPT_COLLECTIVE,
  CKPT_P2P,
  RESTART_RETORE,
  RESTART_REPLAY
};

extern mana_state_t mana_state;
// Set at DMTCP_EVENT_INIT.
extern bool g_libmana_is_initialized;
#endif

#endif  // ifndef _MANA_CORE_H
