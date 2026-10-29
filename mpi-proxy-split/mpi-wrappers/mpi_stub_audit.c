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

// The upper half's ld.so loads this module through LD_AUDIT (see the lower
// half's update_library_path()).  It loads MANA's stub library,
// libmpistub.so, in place of every MPI library that the application needs
// or dlopen()s.  The MPI library and what it loads (UCX, libfabric, ...)
// would otherwise run their constructors in the upper half: UCX's creates
// SysV shared memory, which DMTCP cannot restore.  Unlike a symlink on
// LD_LIBRARY_PATH, this also works for a program built with an RPATH.
//
// A program that dlopen()s the MPI library and calls dlsym() on its handle
// would get the stub's function, which aborts.  la_symbind64() gives it
// libmana_mpi.so's function of the same name instead.
//
// Built without libc: an audit module gets its own link namespace, and a
// second libc there would call mmap() without MANA's wrappers.

#define _GNU_SOURCE  // For LAV_CURRENT, LA_SER_ORIG and Lmid_t
#include <link.h>
#include <stddef.h>
#include <stdint.h>

#include "mpi_library.h"

#define STUB_LIBRARY "libmpistub.so"

unsigned int
la_version(unsigned int version)
{
  (void)version;
  return LAV_CURRENT;
}

// ld.so then searches for STUB_LIBRARY as usual; the lower half puts its
// directory on LD_LIBRARY_PATH.
char *
la_objsearch(const char *name, uintptr_t *cookie, unsigned int flag)
{
  (void)cookie;
  if (flag == LA_SER_ORIG && is_mpi_library(name)) {
    return (char *)STUB_LIBRARY;
  }
  return (char *)name;
}

// libmana_mpi.so, which defines MANA's MPI functions; set by la_objopen().
static struct link_map *libmana;

// Whether s and t are the same string.
static int
same(const char *s, const char *t)
{
  for (; *s != '\0' && *s == *t; s++, t++) {
  }
  return *s == *t;
}

// Whether 'path' is 'file' or ends in "/file".
static int
is_file(const char *path, const char *file)
{
  const char *name = path;
  const char *p;

  for (p = path; *p != '\0'; p++) {
    if (*p == '/') {
      name = p + 1;
    }
  }
  return same(name, file);
}

// The address in a dynamic-section entry of 'map'.  ld.so adds the load
// address to these entries, except where the dynamic section is read-only.
static uintptr_t
dynamic_address(const struct link_map *map, ElfW(Addr) addr)
{
  return addr < map->l_addr ? map->l_addr + addr : addr;
}

// Whether 'sym' defines the function or variable 'name'.
static int
defines(const ElfW(Sym) *sym, const char *strtab, const char *name)
{
  int type = ELF64_ST_TYPE(sym->st_info);

  return sym->st_shndx != SHN_UNDEF &&
         (type == STT_FUNC || type == STT_OBJECT) &&
         ELF64_ST_BIND(sym->st_info) != STB_LOCAL &&
         same(strtab + sym->st_name, name);
}

// The address of the function or variable 'name' that 'map' defines, or 0.
// Uses the GNU hash table, which the toolchain gives libmana_mpi.so.
static uintptr_t
lookup(const struct link_map *map, const char *name)
{
  const ElfW(Sym) *symtab = NULL;
  const char *strtab = NULL;
  const uint32_t *table = NULL;
  const ElfW(Dyn) *d;
  const uint32_t *buckets;
  const uint32_t *chain;
  uint32_t hash = 5381;
  uint32_t i;
  const char *p;

  for (d = map->l_ld; d->d_tag != DT_NULL; d++) {
    if (d->d_tag == DT_SYMTAB) {
      symtab = (const ElfW(Sym) *)dynamic_address(map, d->d_un.d_ptr);
    } else if (d->d_tag == DT_STRTAB) {
      strtab = (const char *)dynamic_address(map, d->d_un.d_ptr);
    } else if (d->d_tag == DT_GNU_HASH) {
      table = (const uint32_t *)dynamic_address(map, d->d_un.d_ptr);
    }
  }
  if (symtab == NULL || strtab == NULL || table == NULL) {
    return 0;
  }
  // table: nbuckets, symoffset, bloom_size, bloom_shift, bloom words,
  // buckets, then the chain of hashes for symbols symoffset and up.
  buckets = (const uint32_t *)((const ElfW(Addr) *)&table[4] + table[2]);
  chain = buckets + table[0];
  for (p = name; *p != '\0'; p++) {
    hash = hash * 33 + (unsigned char)*p;
  }
  i = buckets[hash % table[0]];
  if (i < table[1]) {
    return 0;
  }
  for (;; i++) {
    uint32_t chain_hash = chain[i - table[1]];
    if ((chain_hash | 1) == (hash | 1) && defines(&symtab[i], strtab, name)) {
      return map->l_addr + symtab[i].st_value;
    }
    if (chain_hash & 1) {  // The end of the chain
      return 0;
    }
  }
}

// Audits only the bindings to libmpistub.so (see la_symbind64()).
unsigned int
la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie)
{
  (void)lmid;
  (void)cookie;
  if (is_file(map->l_name, "libmana_mpi.so")) {
    libmana = map;
  }
  return is_file(map->l_name, STUB_LIBRARY) ? LA_FLG_BINDTO : 0;
}

// Called for a symbol of libmpistub.so that dlsym() finds.  Returns
// libmana_mpi.so's definition of the same name if there is one.
uintptr_t
la_symbind64(Elf64_Sym *sym, unsigned int ndx, uintptr_t *refcook,
             uintptr_t *defcook, unsigned int *flags, const char *symname)
{
  uintptr_t addr = libmana != NULL ? lookup(libmana, symname) : 0;

  (void)ndx;
  (void)refcook;
  (void)defcook;
  (void)flags;
  return addr != 0 ? addr : sym->st_value;
}
