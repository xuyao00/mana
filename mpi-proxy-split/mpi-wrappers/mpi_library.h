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

// is_mpi_library(): whether a library is an MPI library, by its name.  For
// the LD_AUDIT modules, which are built without libc: mpi_stub_audit.c, which
// loads libmpistub.so in place of one, and ../mana_audit.c, which refuses one
// under --no-mpi.

#ifndef _MPI_LIBRARY_H
#define _MPI_LIBRARY_H

#include <stddef.h>

// The names of MPICH-ABI MPI libraries, without ".so...": MPICH, Intel MPI,
// MVAPICH, and Cray MPICH (libmpi_cray, libmpich_gnu_82, ...).  Not a
// profiler like libmpiP, nor libmpistub itself.
static const char *const mpi_libraries[] = {
  "libmpi", "libmpich", "libmpifort", "libmpicxx", "libmpichfort",
  "libmpichcxx", "libfmpich", "libmpichf90", NULL
};
static const char *const mpi_library_prefixes[] = {
  "libmpi_", "libmpich_", "libmpifort_", NULL
};

// Whether s[0..len) is 'word'.
static int
equals(const char *s, size_t len, const char *word)
{
  size_t i;
  for (i = 0; i < len; i++) {
    if (word[i] != s[i]) {
      return 0;
    }
  }
  return word[len] == '\0';
}

// Whether s[0..len) starts with 'prefix'.
static int
starts_with(const char *s, size_t len, const char *prefix)
{
  size_t i;
  for (i = 0; prefix[i] != '\0'; i++) {
    if (i >= len || s[i] != prefix[i]) {
      return 0;
    }
  }
  return 1;
}

static int
is_mpi_library(const char *path)
{
  const char *name = path;
  const char *p;
  size_t len;
  int i;

  for (p = path; *p != '\0'; p++) {
    if (*p == '/') {
      name = p + 1;
    }
  }
  for (len = 0; name[len] != '\0'; len++) {
    if (name[len] == '.' && name[len + 1] == 's' && name[len + 2] == 'o') {
      break;
    }
  }
  if (name[len] == '\0') {
    return 0;
  }
  for (i = 0; mpi_libraries[i] != NULL; i++) {
    if (equals(name, len, mpi_libraries[i])) {
      return 1;
    }
  }
  for (i = 0; mpi_library_prefixes[i] != NULL; i++) {
    if (starts_with(name, len, mpi_library_prefixes[i])) {
      return 1;
    }
  }
  return 0;
}

#endif  // ifndef _MPI_LIBRARY_H
