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

// cuBLAS and cuBLASLt handles in the forwarded calls of their front
// libraries (gen-cuda-front.py --include): the records of front_blas.cpp.
// cuBLASLt's descriptors are plain data that the front keeps in upper-half
// memory, so they pass as they are.

#ifndef _MANA_CUDA_FRONT_BLAS_H
#define _MANA_CUDA_FRONT_BLAS_H

#include <cublasLt.h>
#include <cublas_v2.h>
#include "front_auto.h"

extern FrontLib front_cublas_lib FRONT_HIDDEN;
extern FrontLib front_cublasLt_lib FRONT_HIDDEN;

FRONT_TRANSLATED(cublasHandle_t);
FRONT_TRANSLATED(cublasLtHandle_t);

static inline cublasHandle_t
front_translate(cublasHandle_t handle)
{
  return (cublasHandle_t)front_real_obj(handle, FRONT_BLAS);
}

// A cuBLAS handle is a cuBLASLt handle too, cast (PyTorch passes one so).
static inline cublasLtHandle_t
front_translate(cublasLtHandle_t handle)
{
  const FrontObj *obj = (const FrontObj *)handle;
  if (obj != NULL && obj->magic == FRONT_MAGIC && obj->kind == FRONT_BLAS) {
    return (cublasLtHandle_t)obj->real;
  }
  return (cublasLtHandle_t)front_real_obj(handle, FRONT_BLASLT);
}

#endif  // ifndef _MANA_CUDA_FRONT_BLAS_H
