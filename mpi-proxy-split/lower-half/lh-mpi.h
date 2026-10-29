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

#ifndef _LH_MPI_H
#define _LH_MPI_H

// The lower half's MPI: lh-mpi.cpp in bin/lower-half, lh-nompi.cpp in
// bin/lower-half-nompi, which has none.

// Starts MPI before the upper half exists; the upper half's MPI_Init then
// finds it started.  Returns this process's rank, which picks its image at
// restart.  Sets *extra_tls if the checkpoint thread needs a lower-half TLS
// of its own (MANA_P2P_WAIT=blocking, with MPI_THREAD_MULTIPLE).
int init_lower_half_mpi(int *argc, char ***argv, bool *extra_tls);

// Fills in lh_info's MPI part: mpi, lh_dlsym and handle_c2f.
void initialize_lh_mpi_info();

// Ends the job after an error: MPI_Abort(), not exit(), with MPI, since srun
// can leave the other ranks waiting.
void abort_lower_half_job();

#endif // ifndef _LH_MPI_H
