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

// NCCL communicators in the forwarded calls of the NCCL front library
// (gen-cuda-front.py --include): the records of front_nccl.cpp.
#ifndef _MANA_CUDA_FRONT_NCCL_H
#define _MANA_CUDA_FRONT_NCCL_H

#include <nccl.h>
#include "front_auto.h"

extern FrontLib front_nccl_lib FRONT_HIDDEN;

FRONT_TRANSLATED(ncclComm_t);

static inline ncclComm_t
front_translate(ncclComm_t comm)
{
  return (ncclComm_t)front_real_obj(comm, FRONT_NCCL);
}

#endif  // ifndef _MANA_CUDA_FRONT_NCCL_H
