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

#ifndef _LOWER_HALF_MPI_H
#define _LOWER_HALF_MPI_H

#include <mpi.h>

#include "lower-half-api.h"

// What bin/lower-half adds for the MPI layer (libmana_mpi.so): the MPI
// library's constants, which the layer translates the application's to.
// bin/lower-half-nompi has none (lh_info->mpi is NULL).

extern "C" int MPI_MANA_Internal(char *dummy);
extern "C" int MPI_MANA_Query_cuda_support();

typedef struct _LowerHalfMpiInfo
{
  MPI_Group MANA_GROUP_NULL;
  MPI_Comm MANA_COMM_NULL;
  MPI_Request MANA_REQUEST_NULL;
  MPI_Message MANA_MESSAGE_NULL;
  MPI_Op MANA_OP_NULL;
  MPI_Errhandler MANA_ERRHANDLER_NULL;
  MPI_Info MANA_INFO_NULL;
  MPI_Win MANA_WIN_NULL;
  MPI_File MANA_FILE_NULL;
  MPI_Info MANA_INFO_ENV;
  MPI_Comm MANA_COMM_WORLD;
  MPI_Comm MANA_COMM_SELF;
  MPI_Group MANA_GROUP_EMPTY;
  MPI_Message MANA_MESSAGE_NO_PROC;
  MPI_Op MANA_MAX;
  MPI_Op MANA_MIN;
  MPI_Op MANA_SUM;
  MPI_Op MANA_PROD;
  MPI_Op MANA_LAND;
  MPI_Op MANA_BAND;
  MPI_Op MANA_LOR;
  MPI_Op MANA_BOR;
  MPI_Op MANA_LXOR;
  MPI_Op MANA_BXOR;
  MPI_Op MANA_MAXLOC;
  MPI_Op MANA_MINLOC;
  MPI_Op MANA_REPLACE;
  MPI_Op MANA_NO_OP;
  MPI_Datatype MANA_DATATYPE_NULL;
  MPI_Datatype MANA_BYTE;
  MPI_Datatype MANA_PACKED;
  MPI_Datatype MANA_CHAR;
  MPI_Datatype MANA_SHORT;
  MPI_Datatype MANA_INT;
  MPI_Datatype MANA_LONG;
  MPI_Datatype MANA_FLOAT;
  MPI_Datatype MANA_DOUBLE;
  MPI_Datatype MANA_LONG_DOUBLE;
  MPI_Datatype MANA_UNSIGNED_CHAR;
  MPI_Datatype MANA_SIGNED_CHAR;
  MPI_Datatype MANA_UNSIGNED_SHORT;
  MPI_Datatype MANA_UNSIGNED_LONG;
  MPI_Datatype MANA_UNSIGNED;
  MPI_Datatype MANA_FLOAT_INT;
  MPI_Datatype MANA_DOUBLE_INT;
  MPI_Datatype MANA_LONG_DOUBLE_INT;
  MPI_Datatype MANA_LONG_INT;
  MPI_Datatype MANA_SHORT_INT;
  MPI_Datatype MANA_2INT;
  MPI_Datatype MANA_WCHAR;
  MPI_Datatype MANA_LONG_LONG_INT;
  MPI_Datatype MANA_LONG_LONG;
  MPI_Datatype MANA_UNSIGNED_LONG_LONG;
#if 0
  MPI_Datatype MANA_2COMPLEX;
  MPI_Datatype MANA_2DOUBLE_COMPLEX;
  MPI_Datatype MANA_CXX_COMPLEX;
#endif
  MPI_Datatype MANA_CHARACTER;
  MPI_Datatype MANA_LOGICAL;
#if 0
  MPI_Datatype MANA_LOGICAL1;
  MPI_Datatype MANA_LOGICAL2;
  MPI_Datatype MANA_LOGICAL4;
  MPI_Datatype MANA_LOGICAL8;
#endif
  MPI_Datatype MANA_INTEGER;
  MPI_Datatype MANA_INTEGER1;
  MPI_Datatype MANA_INTEGER2;
  MPI_Datatype MANA_INTEGER4;
  MPI_Datatype MANA_INTEGER8;
  MPI_Datatype MANA_REAL;
  MPI_Datatype MANA_REAL4;
  MPI_Datatype MANA_REAL8;
  MPI_Datatype MANA_REAL16;
  MPI_Datatype MANA_DOUBLE_PRECISION;
  MPI_Datatype MANA_COMPLEX;
  MPI_Datatype MANA_COMPLEX8;
  MPI_Datatype MANA_COMPLEX16;
  MPI_Datatype MANA_COMPLEX32;
  MPI_Datatype MANA_DOUBLE_COMPLEX;
  MPI_Datatype MANA_2REAL;
  MPI_Datatype MANA_2DOUBLE_PRECISION;
  MPI_Datatype MANA_2INTEGER;
  MPI_Datatype MANA_INT8_T;
  MPI_Datatype MANA_UINT8_T;
  MPI_Datatype MANA_INT16_T;
  MPI_Datatype MANA_UINT16_T;
  MPI_Datatype MANA_INT32_T;
  MPI_Datatype MANA_UINT32_T;
  MPI_Datatype MANA_INT64_T;
  MPI_Datatype MANA_UINT64_T;
  MPI_Datatype MANA_AINT;
  MPI_Datatype MANA_OFFSET;
  MPI_Datatype MANA_C_BOOL;
  MPI_Datatype MANA_C_COMPLEX;
  MPI_Datatype MANA_C_FLOAT_COMPLEX;
  MPI_Datatype MANA_C_DOUBLE_COMPLEX;
  MPI_Datatype MANA_C_LONG_DOUBLE_COMPLEX;
  MPI_Datatype MANA_CXX_BOOL;
  MPI_Datatype MANA_CXX_FLOAT_COMPLEX;
  MPI_Datatype MANA_CXX_DOUBLE_COMPLEX;
  MPI_Datatype MANA_CXX_LONG_DOUBLE_COMPLEX;
  MPI_Datatype MANA_COUNT;
  MPI_Errhandler MANA_ERRORS_ARE_FATAL;
  MPI_Errhandler MANA_ERRORS_RETURN;
} LowerHalfMpiInfo_t;

// The lower half's MPI constants: LH_MPI->MANA_COMM_WORLD, ...
#define LH_MPI ((LowerHalfMpiInfo_t *)lh_info->mpi)

#endif // ifndef _LOWER_HALF_MPI_H
