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

// What libmana.so (mpi_plugin.cpp) asks of the CUDA front under --cuda:
// mana_cuda_event(event), on the checkpoint thread, except for
// MANA_CUDA_THREAD_EXIT and MANA_CUDA_DRAIN_MEMOPS.  The front is loaded only
// if the application uses CUDA.

#ifndef _MANA_CUDA_H
#define _MANA_CUDA_H

enum ManaCudaEvent {
  MANA_CUDA_QUIESCE = 1,   // PRESUSPEND, after MANA's drain: wait for the GPUs
  MANA_CUDA_STASH,         // then, after a global barrier: save device data
  MANA_CUDA_RESUME,        // RESUME: free the saved device data
  MANA_CUDA_RESTART,       // RESTART, before the P2P replay: rebuild
  MANA_CUDA_THREAD_EXIT,   // PTHREAD_RETURN or _EXIT, on the exiting thread
  MANA_CUDA_DRAIN_MEMOPS,  // an MPI call: wait for pending copies and sets
  MANA_CUDA_RESTART_IPC,   // after MANA_CUDA_RESTART and a global barrier
};

typedef void (*mana_cuda_event_t)(int event);

#endif  // ifndef _MANA_CUDA_H
