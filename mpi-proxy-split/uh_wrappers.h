/****************************************************************************
 *  Copyright (C) 2019-2020 by Twinkle Jain, Rohan garg, and Gene Cooperman *
 *  jain.t@husky.neu.edu, rohgarg@ccs.neu.edu, gene@ccs.neu.edu             *
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

#ifndef UPPER_HALF_WRAPPERS_H
#define UPPER_HALF_WRAPPERS_H

extern int initialized;

extern void initialize_wrappers();
void mana_fwd_note_thread(pid_t real_tid, unsigned long fs);
void mana_lower_half_mpi_init();
void mana_fwd_before_ckpt();
void mana_fwd_after_resume();
void mana_fwd_after_resume_peers();
void mana_fwd_use_context_of(const void *buf);
int mana_fwd_is_device_memory(const void *buf);
void mana_vmm_connect();
extern void reset_wrappers();
extern LowerHalfInfo_t *lh_info;

#endif // ifndef UPPER_HALF_WRAPPERS_H
