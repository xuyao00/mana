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

// Under --cuda, the lower half adds this LD_AUDIT module for the upper half,
// and puts lib/dmtcp/cuda-front first on LD_LIBRARY_PATH.  The upper half
// then loads MANA's CUDA front libraries in place of libcudart and libcuda,
// also for a program built with an RPATH, or one that dlopen()s them by
// path: a real libcuda in the upper half would be a second CUDA driver in
// the process (cuda/DESIGN-lower-half-cuda.md).
//
// Built without libc, as mpi-wrappers/mpi_stub_audit.c is.

#define _GNU_SOURCE  // For LAV_CURRENT and LA_SER_ORIG
#include <link.h>
#include <stddef.h>
#include <stdint.h>

static const char *
base_name(const char *path)
{
  const char *base = path;
  for (const char *p = path; *p != '\0'; p++) {
    if (*p == '/') {
      base = p + 1;
    }
  }
  return base;
}

static int
starts_with(const char *s, const char *prefix)
{
  for (; *prefix != '\0'; s++, prefix++) {
    if (*s != *prefix) {
      return 0;
    }
  }
  return 1;
}

// libcudart.so, libcudart.so.<N>[...], libcuda.so, libcuda.so.1
static int
is_front_library(const char *base)
{
  const char *rest;
  if (starts_with(base, "libcudart.so")) {
    rest = base + sizeof("libcudart.so") - 1;
  } else if (starts_with(base, "libcuda.so")) {
    rest = base + sizeof("libcuda.so") - 1;
  } else {
    return 0;
  }
  return *rest == '\0' || (rest[0] == '.' && rest[1] >= '0' && rest[1] <= '9');
}

static int
in_front_dir(const char *path)
{
  for (const char *p = path; *p != '\0'; p++) {
    if (starts_with(p, "/cuda-front/")) {
      return 1;
    }
  }
  return 0;
}

unsigned int
la_version(unsigned int version)
{
  (void)version;
  return LAV_CURRENT;
}

char *
la_objsearch(const char *name, uintptr_t *cookie, unsigned int flag)
{
  (void)cookie;
  const char *base = base_name(name);
  if (!is_front_library(base)) {
    return (char *)name;
  }
  // Search by name, so that LD_LIBRARY_PATH finds the front first; refuse
  // every other directory.
  if (flag == LA_SER_ORIG) {
    return (char *)base;
  }
  return in_front_dir(name) ? (char *)name : NULL;
}
