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

// The core's LD_AUDIT module, which both lower halves give the upper half
// (update_library_path()), after the MPI stub's.  It refuses a library of a layer that the run
// doesn't have, rather than let it run unprotected in the upper half:
// - an MPI library under --no-mpi: nothing would drain or rebuild it, and
//   its constructors (UCX's) make state that DMTCP cannot restore;
// - libcuda or libcudart without --cuda: nothing would save the GPU, and
//   with a CUDA-aware MPI the process would have two CUDA drivers.
// A run has a layer if its library is preloaded (mana_launch): libmana_mpi.so
// or libmana_cuda.so.  ld.so opens the preloaded libraries before it looks
// for the program's.
//
// A refused library is not found: the program fails to start, or its
// dlopen() fails.  Built without libc, as mpi-wrappers/mpi_stub_audit.c is.

#define _GNU_SOURCE  // For LAV_CURRENT, LA_SER_ORIG and Lmid_t
#include <link.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/syscall.h>

#include "mpi-wrappers/mpi_library.h"

static int has_mpi_layer;   // libmana_mpi.so is loaded
static int has_cuda_layer;  // libmana_cuda.so is loaded

static const char *
base_name(const char *path)
{
  const char *base = path;
  const char *p;
  for (p = path; *p != '\0'; p++) {
    if (*p == '/') {
      base = p + 1;
    }
  }
  return base;
}

// Whether base is 'lib' or 'lib' followed by a version (".1", ".13.1").
static int
is_library(const char *base, const char *lib)
{
  size_t i;
  for (i = 0; lib[i] != '\0'; i++) {
    if (base[i] != lib[i]) {
      return 0;
    }
  }
  return base[i] == '\0' || (base[i] == '.' && base[i + 1] >= '0' &&
                             base[i + 1] <= '9');
}

static int
is_cuda_library(const char *base)
{
  return is_library(base, "libcuda.so") || is_library(base, "libcudart.so");
}

static void
write_stderr(const char *s)
{
  // volatile: the compiler would make the loop a call to strlen().
  const volatile char *v = s;
  size_t len = 0;
  while (v[len] != '\0') {
    len++;
  }
#if defined(__x86_64__)
  long ret;
  __asm__ volatile ("syscall"
                    : "=a" (ret)
                    : "a" (SYS_write), "D" (2), "S" (s), "d" (len)
                    : "rcx", "r11", "memory");
  (void)ret;
#elif defined(__aarch64__)
  register long x0 __asm__ ("x0") = 2;
  register long x1 __asm__ ("x1") = (long)s;
  register long x2 __asm__ ("x2") = (long)len;
  register long x8 __asm__ ("x8") = SYS_write;
  __asm__ volatile ("svc 0" : "+r" (x0) : "r" (x1), "r" (x2), "r" (x8)
                    : "memory");
#endif
}

static void
refuse(const char *name, const char *why)
{
  write_stderr("MANA: ");
  write_stderr(name);
  write_stderr(why);
}

unsigned int
la_version(unsigned int version)
{
  (void)version;
  return LAV_CURRENT;
}

unsigned int
la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie)
{
  (void)lmid;
  (void)cookie;
  const char *base = base_name(map->l_name);
  if (is_library(base, "libmana_mpi.so")) {
    has_mpi_layer = 1;
  } else if (is_library(base, "libmana_cuda.so")) {
    has_cuda_layer = 1;
  }
  return 0;
}

char *
la_objsearch(const char *name, uintptr_t *cookie, unsigned int flag)
{
  (void)cookie;
  if (flag != LA_SER_ORIG) {
    return (char *)name;
  }
  if (!has_mpi_layer && is_mpi_library(name)) {
    refuse(name, ": an MPI library, refused under mana_launch --no-mpi\n");
    return NULL;
  }
  if (!has_cuda_layer && is_cuda_library(base_name(name))) {
    refuse(name, ": a CUDA library, refused without mana_launch --cuda\n");
    return NULL;
  }
  return (char *)name;
}
