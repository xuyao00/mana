/****************************************************************************
 *   Copyright (C) 2026 by Yao Xu                                           *
 *   xu.yao1@northeastern.edu                                               *
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

#ifndef _MANA_CUDA_H
#define _MANA_CUDA_H

#include <sys/types.h>

// CUDA-aware MPI (mana_launch --cuda; see cuda-forward/README.md): the
// upper half's side of the lower half's forwarding shims.  Each function
// does nothing without --cuda.  All run in the upper half, with its FS.

// Whether the lower half's CUDA libraries are forwarding shims.
bool mana_cuda_enabled();

// The lower half's MPI_Init().  With --cuda, the lower half initializes MPI
// only when the upper half asks: at the application's MPI_Init(), and at
// restart after the CUDA plugin has restored the GPU.
void mana_lower_half_mpi_init();

// Tells the shims the upper half's FS and stack of the calling thread.
void mana_cuda_note_thread(pid_t real_tid, unsigned long fs);
void mana_cuda_forget_thread(pid_t real_tid);

// Around a checkpoint, on the checkpoint thread: see FwdCtl.
void mana_cuda_before_ckpt();
void mana_cuda_after_resume();
void mana_cuda_after_resume_peers();

// Makes the CUDA context that owns device buffer 'buf' current on this
// thread.  The checkpoint thread calls the MPI in the drain and in the
// restart replay with no context, or another device's.
void mana_cuda_use_context_of(const void *buf);

// Whether 'buf' is device memory.
bool mana_cuda_is_device_memory(const void *buf);

// Connects libmana_cuda_vmm.so, if it is loaded (see lower_half_ckpt.h).
void mana_cuda_connect_vmm();
// At restart, before the new lower half's MPI_Init: forget the old lower
// half's FwdCtl.mem_event.
void mana_cuda_restart_vmm();

#endif  // ifndef _MANA_CUDA_H
