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

// The lower half's MPI, in bin/lower-half (see lh-mpi.h).

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

#include "lower-half-mpi.h"
#include "lh-mpi.h"

static LowerHalfMpiInfo_t lh_mpi_info;

int
init_lower_half_mpi(int *argc, char ***argv, bool *extra_tls)
{
  // In blocking mode (MANA_P2P_WAIT=blocking; mana_restart sets it from the
  // checkpoint), the checkpoint thread's P2P drain calls MPI while an
  // application thread waits in the library's MPI_Recv or MPI_Send.  So two
  // threads are in the library at once: this needs MPI_THREAD_MULTIPLE.  In
  // polling mode no thread waits in the library during the drain, so the
  // cheaper MPI_THREAD_SINGLE is enough.
  const char *p2p_wait = getenv("MANA_P2P_WAIT");
  int required = (p2p_wait != NULL && strcmp(p2p_wait, "blocking") == 0)
                   ? MPI_THREAD_MULTIPLE : MPI_THREAD_SINGLE;
  int provided = MPI_THREAD_SINGLE;
  MPI_Init_thread(argc, argv, required, &provided);
  *extra_tls = required == MPI_THREAD_MULTIPLE &&
               provided == MPI_THREAD_MULTIPLE;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

// lh_info->handle_c2f: see lower-half-api.h.
static int
handle_c2f(int kind, int64_t handle)
{
  intptr_t h = (intptr_t)handle;
  switch (kind) {
    case MANA_COMM_KIND:
      return MPI_Comm_c2f((MPI_Comm)h);
    case MANA_GROUP_KIND:
      return MPI_Group_c2f((MPI_Group)h);
    case MANA_DATATYPE_KIND:
      return MPI_Type_c2f((MPI_Datatype)h);
    case MANA_OP_KIND:
      return MPI_Op_c2f((MPI_Op)h);
    case MANA_REQUEST_KIND:
      return MPI_Request_c2f((MPI_Request)h);
    case MANA_FILE_KIND:
      return MPI_File_c2f((MPI_File)h);
    case MANA_ERRHANDLER_KIND:
      return MPI_Errhandler_c2f((MPI_Errhandler)h);
    case MANA_INFO_KIND:
      return MPI_Info_c2f((MPI_Info)h);
    case MANA_WIN_KIND:
      return MPI_Win_c2f((MPI_Win)h);
    case MANA_MESSAGE_KIND:
      return MPI_Message_c2f((MPI_Message)h);
  }
  abort();
}

void
initialize_lh_mpi_info()
{
  LowerHalfMpiInfo_t *mpi = &lh_mpi_info;
  lh_info->mpi = mpi;
  lh_info->lh_dlsym = (void*)&lh_dlsym;
  lh_info->handle_c2f = (void*)&handle_c2f;
  mpi->MANA_GROUP_NULL = MPI_GROUP_NULL;
  mpi->MANA_COMM_NULL = MPI_COMM_NULL;
  mpi->MANA_REQUEST_NULL = MPI_REQUEST_NULL;
  mpi->MANA_MESSAGE_NULL = MPI_MESSAGE_NULL;
  mpi->MANA_OP_NULL = MPI_OP_NULL;
  mpi->MANA_ERRHANDLER_NULL = MPI_ERRHANDLER_NULL;
  mpi->MANA_INFO_NULL = MPI_INFO_NULL;
  mpi->MANA_WIN_NULL = MPI_WIN_NULL;
  mpi->MANA_FILE_NULL = MPI_FILE_NULL;
  mpi->MANA_INFO_ENV = MPI_INFO_ENV;
  mpi->MANA_COMM_WORLD = MPI_COMM_WORLD;
  mpi->MANA_COMM_SELF = MPI_COMM_SELF;
  mpi->MANA_GROUP_EMPTY = MPI_GROUP_EMPTY;
  mpi->MANA_MESSAGE_NO_PROC = MPI_MESSAGE_NO_PROC;
  mpi->MANA_MAX = MPI_MAX;
  mpi->MANA_MIN = MPI_MIN;
  mpi->MANA_SUM = MPI_SUM;
  mpi->MANA_PROD = MPI_PROD;
  mpi->MANA_LAND = MPI_LAND;
  mpi->MANA_BAND = MPI_BAND;
  mpi->MANA_LOR = MPI_LOR;
  mpi->MANA_BOR = MPI_BOR;
  mpi->MANA_LXOR = MPI_LXOR;
  mpi->MANA_BXOR = MPI_BXOR;
  mpi->MANA_MAXLOC = MPI_MAXLOC;
  mpi->MANA_MINLOC = MPI_MINLOC;
  mpi->MANA_REPLACE = MPI_REPLACE;
  mpi->MANA_NO_OP = MPI_NO_OP;
  mpi->MANA_DATATYPE_NULL = MPI_DATATYPE_NULL;
  mpi->MANA_BYTE = MPI_BYTE;
  mpi->MANA_PACKED = MPI_PACKED;
  mpi->MANA_CHAR = MPI_CHAR;
  mpi->MANA_SHORT = MPI_SHORT;
  mpi->MANA_INT = MPI_INT;
  mpi->MANA_LONG = MPI_LONG;
  mpi->MANA_FLOAT = MPI_FLOAT;
  mpi->MANA_DOUBLE = MPI_DOUBLE;
  mpi->MANA_LONG_DOUBLE = MPI_LONG_DOUBLE;
  mpi->MANA_UNSIGNED_CHAR = MPI_UNSIGNED_CHAR;
  mpi->MANA_SIGNED_CHAR = MPI_SIGNED_CHAR;
  mpi->MANA_UNSIGNED_SHORT = MPI_UNSIGNED_SHORT;
  mpi->MANA_UNSIGNED_LONG = MPI_UNSIGNED_LONG;
  mpi->MANA_UNSIGNED = MPI_UNSIGNED;
  mpi->MANA_FLOAT_INT = MPI_FLOAT_INT;
  mpi->MANA_DOUBLE_INT = MPI_DOUBLE_INT;
  mpi->MANA_LONG_DOUBLE_INT = MPI_LONG_DOUBLE_INT;
  mpi->MANA_LONG_INT = MPI_LONG_INT;
  mpi->MANA_SHORT_INT = MPI_SHORT_INT;
  mpi->MANA_2INT = MPI_2INT;
  mpi->MANA_WCHAR = MPI_WCHAR;
  mpi->MANA_LONG_LONG_INT = MPI_LONG_LONG_INT;
  mpi->MANA_LONG_LONG = MPI_LONG_LONG;
  mpi->MANA_UNSIGNED_LONG_LONG = MPI_UNSIGNED_LONG_LONG;
#if 0
  mpi->MANA_2COMPLEX = MPI_2COMPLEX;
  mpi->MANA_CXX_COMPLEX = MPI_CXX_COMPLEX;
  mpi->MANA_2DOUBLE_COMPLEX = MPI_2DOUBLE_COMPLEX;
#endif
  mpi->MANA_CHARACTER = MPI_CHARACTER;
  mpi->MANA_LOGICAL = MPI_LOGICAL;
#if 0
  mpi->MANA_LOGICAL1 = MPI_LOGICAL1;
  mpi->MANA_LOGICAL2 = MPI_LOGICAL2;
  mpi->MANA_LOGICAL4 = MPI_LOGICAL4;
  mpi->MANA_LOGICAL8 = MPI_LOGICAL8;
#endif
  mpi->MANA_INTEGER = MPI_INTEGER;
  mpi->MANA_INTEGER1 = MPI_INTEGER1;
  mpi->MANA_INTEGER2 = MPI_INTEGER2;
  mpi->MANA_INTEGER4 = MPI_INTEGER4;
  mpi->MANA_INTEGER8 = MPI_INTEGER8;
  mpi->MANA_REAL = MPI_REAL;
  mpi->MANA_REAL4 = MPI_REAL4;
  mpi->MANA_REAL8 = MPI_REAL8;
  mpi->MANA_REAL16 = MPI_REAL16;
  mpi->MANA_DOUBLE_PRECISION = MPI_DOUBLE_PRECISION;
  mpi->MANA_COMPLEX = MPI_COMPLEX;
  mpi->MANA_COMPLEX8 = MPI_COMPLEX8;
  mpi->MANA_COMPLEX16 = MPI_COMPLEX16;
  mpi->MANA_COMPLEX32 = MPI_COMPLEX32;
  mpi->MANA_DOUBLE_COMPLEX = MPI_DOUBLE_COMPLEX;
  mpi->MANA_2REAL = MPI_2REAL;
  mpi->MANA_2DOUBLE_PRECISION = MPI_2DOUBLE_PRECISION;
  mpi->MANA_2INTEGER = MPI_2INTEGER;
  mpi->MANA_INT8_T = MPI_INT8_T;
  mpi->MANA_UINT8_T = MPI_UINT8_T;
  mpi->MANA_INT16_T = MPI_INT16_T;
  mpi->MANA_UINT16_T = MPI_UINT16_T;
  mpi->MANA_INT32_T = MPI_INT32_T;
  mpi->MANA_UINT32_T = MPI_UINT32_T;
  mpi->MANA_INT64_T = MPI_INT64_T;
  mpi->MANA_UINT64_T = MPI_UINT64_T;
  mpi->MANA_AINT = MPI_AINT;
  mpi->MANA_OFFSET = MPI_OFFSET;
  mpi->MANA_C_BOOL = MPI_C_BOOL;
  mpi->MANA_C_COMPLEX = MPI_C_COMPLEX;
  mpi->MANA_C_FLOAT_COMPLEX = MPI_C_FLOAT_COMPLEX;
  mpi->MANA_C_DOUBLE_COMPLEX = MPI_C_DOUBLE_COMPLEX;
  mpi->MANA_C_LONG_DOUBLE_COMPLEX = MPI_C_LONG_DOUBLE_COMPLEX;
  mpi->MANA_CXX_BOOL = MPI_CXX_BOOL;
  mpi->MANA_CXX_FLOAT_COMPLEX = MPI_CXX_FLOAT_COMPLEX;
  mpi->MANA_CXX_DOUBLE_COMPLEX = MPI_CXX_DOUBLE_COMPLEX;
  mpi->MANA_CXX_LONG_DOUBLE_COMPLEX = MPI_CXX_LONG_DOUBLE_COMPLEX;
  mpi->MANA_COUNT = MPI_COUNT;
  mpi->MANA_ERRORS_ARE_FATAL = MPI_ERRORS_ARE_FATAL;
  mpi->MANA_ERRORS_RETURN = MPI_ERRORS_RETURN;
}

void
abort_lower_half_job()
{
  MPI_Abort(MPI_COMM_WORLD, 1);
}

int MPI_MANA_Internal(char *dummy) {
  return 0;
}

// MPICH's MPIX_Query_cuda_support(), for the upper half's: whether this MPI
// supports CUDA buffers (GROMACS asks before it uses direct GPU
// communication).  0 if the MPI has no such function.
int MPI_MANA_Query_cuda_support() {
  int (*query)(void) =
    (int (*)(void))dlsym(RTLD_DEFAULT, "MPIX_Query_cuda_support");
  return query != NULL ? query() : 0;
}
