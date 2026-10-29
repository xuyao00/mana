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

// bin/lower-half-nompi: a lower half without MPI (see lh-mpi.h), for a
// program that doesn't use MPI, such as one that uses NCCL and is started by
// srun.  It runs where no MPI library is installed.

#include <stdlib.h>

#include "lower-half-api.h"
#include "lh-mpi.h"

// The launcher's rank of this process, as mana_restart and libmana.so read
// it, or 0 if there is none: a single process.
int
init_lower_half_mpi(int *argc, char ***argv, bool *extra_tls)
{
  (void)argc;
  (void)argv;
  *extra_tls = false;
  const char *vars[] = {"PMI_RANK", "PMIX_RANK", "SLURM_PROCID"};
  for (const char *var : vars) {
    const char *value = getenv(var);
    if (value != NULL && value[0] >= '0' && value[0] <= '9') {
      return atoi(value);
    }
  }
  return 0;
}

// lh_info->mpi, lh_dlsym and handle_c2f stay NULL.
void
initialize_lh_mpi_info()
{
}

void
abort_lower_half_job()
{
  exit(1);
}
