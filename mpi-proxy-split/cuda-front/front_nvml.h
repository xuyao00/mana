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

// NVML (libnvidia-ml.so.1) runs in the lower half with the real driver: in
// the upper half, it would call the CUDA front's libcuda for the driver's
// private tables (cuGetExportTable), which only the real driver has.  Every
// function that nvml.h declares is forwarded as it is
// (gen-cuda-front.py --header and --include): also the old names that it
// maps to the new ones, which the library exports too, and which a program
// may call by name (PyTorch calls nvmlInit through ctypes).  NVML's handles
// are its own pointers in the lower half, which the front passes as they
// are; they do not survive a restart, after which a program must call
// nvmlInit and get them again.

#ifndef _MANA_CUDA_FRONT_NVML_H
#define _MANA_CUDA_FRONT_NVML_H

#define NVML_NO_UNVERSIONED_FUNC_DEFS
#include <nvml.h>
#include "front_auto.h"

FRONT_PASSED(nvmlDevice_t);
FRONT_PASSED(nvmlGpuInstance_t);
FRONT_PASSED(nvmlComputeInstance_t);
FRONT_PASSED(nvmlUnit_t);
FRONT_PASSED(nvmlEventSet_t);
FRONT_PASSED(nvmlSystemEventSet_t);
FRONT_PASSED(nvmlGpmSample_t);

#endif  // ifndef _MANA_CUDA_FRONT_NVML_H
