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

// A static CUDA runtime in the executable (nvcc's default, -cudart=static).
// It would run in the upper half, and its state would come back from the
// image pointing at a driver that is gone.  So each of its entry points that
// the front knows jumps to the front's function of the same name: the
// embedded runtime never starts, and every call goes to the lower half's
// shared cudart (cuda/DESIGN-lower-half-cuda.md, section 2).
//
// This runs before the executable's constructors register its fatbins:
// under --cuda, bin/mana_launch preloads libmana_cuda.so.  The entry points
// are found in the executable's .symtab, local ones too (Phase 0: cudaMalloc
// is local there); a static runtime in an executable without .symtab is
// refused.  The patched text is saved in the image like the rest of the
// upper half.

#include <elf.h>
#include <fcntl.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <map>
#include <string>
#include <vector>
#include "front.h"

extern "C" void *const front_cudart_dispatch[];

namespace {

struct ExeFunc {
  uint64_t value;
  uint64_t size;
  int count;               // definitions of this name
};

struct ExeInfo {
  bool has_symtab;
  bool has_fatbin;         // .nv_fatbin: kernels, which need a runtime
  std::map<std::string, ExeFunc> funcs;
  std::vector<uint64_t> starts;  // of all functions, sorted
};

// The bytes from 'value' to the next function: a small function is padded
// to the alignment of the next one.
uint64_t
room(const ExeInfo &info, const ExeFunc &f)
{
  auto next = std::upper_bound(info.starts.begin(), info.starts.end(),
                               f.value);
  uint64_t gap = next != info.starts.end() ? *next - f.value : f.size;
  return std::max(f.size, gap);
}

int
find_exe(struct dl_phdr_info *info, size_t, void *exe)
{
  *(struct dl_phdr_info *)exe = *info;  // the first object: the executable
  return 1;
}

// The application's file.  Not /proc/self/exe, which is bin/lower-half, the
// loader of the upper half: the file mapped where the executable's program
// headers are.
bool
exe_path(std::string *path, ElfW(Addr) *base)
{
  struct dl_phdr_info exe;
  memset(&exe, 0, sizeof exe);
  dl_iterate_phdr(find_exe, &exe);
  *base = exe.dlpi_addr;
  uintptr_t at = (uintptr_t)exe.dlpi_phdr;
  FILE *maps = fopen("/proc/self/maps", "r");
  if (maps == NULL || at == 0) {
    return false;
  }
  char line[4096];
  bool found = false;
  while (!found && fgets(line, sizeof line, maps) != NULL) {
    unsigned long start, end;
    char file[4096] = "";
    if (sscanf(line, "%lx-%lx %*s %*s %*s %*s %4095s", &start, &end, file) >=
          2 && at >= start && at < end && file[0] == '/') {
      *path = file;
      found = true;
    }
  }
  fclose(maps);
  return found;
}

bool
read_exe(const std::string &path, ExeInfo *info)
{
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    return false;
  }
  struct stat st;
  void *map = MAP_FAILED;
  if (fstat(fd, &st) == 0 && st.st_size > (off_t)sizeof(Elf64_Ehdr)) {
    map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  }
  close(fd);
  if (map == MAP_FAILED) {
    return false;
  }
  const char *file = (const char *)map;
  const Elf64_Ehdr *eh = (const Elf64_Ehdr *)file;
  bool ok = memcmp(eh->e_ident, ELFMAG, SELFMAG) == 0 &&
            eh->e_ident[EI_CLASS] == ELFCLASS64 &&
            eh->e_shoff + (uint64_t)eh->e_shnum * sizeof(Elf64_Shdr) <=
              (uint64_t)st.st_size;
  if (ok) {
    const Elf64_Shdr *sh = (const Elf64_Shdr *)(file + eh->e_shoff);
    const char *names = file + sh[eh->e_shstrndx].sh_offset;
    for (int i = 0; i < eh->e_shnum; i++) {
      if (strcmp(names + sh[i].sh_name, ".nv_fatbin") == 0) {
        info->has_fatbin = true;
      }
      if (sh[i].sh_type != SHT_SYMTAB) {
        continue;
      }
      info->has_symtab = true;
      const Elf64_Sym *syms = (const Elf64_Sym *)(file + sh[i].sh_offset);
      const char *strs = file + sh[sh[i].sh_link].sh_offset;
      size_t n = sh[i].sh_size / sizeof(Elf64_Sym);
      for (size_t k = 0; k < n; k++) {
        if (ELF64_ST_TYPE(syms[k].st_info) != STT_FUNC ||
            syms[k].st_shndx == SHN_UNDEF || syms[k].st_value == 0) {
          continue;
        }
        ExeFunc &f = info->funcs[strs + syms[k].st_name];
        f.value = syms[k].st_value;
        f.size = syms[k].st_size;
        f.count++;
        info->starts.push_back(syms[k].st_value);
      }
    }
  }
  munmap(map, st.st_size);
  std::sort(info->starts.begin(), info->starts.end());
  return ok;
}

int
find_shared_cudart(struct dl_phdr_info *info, size_t, void *found)
{
  if (strstr(info->dlpi_name, "/libcudart.so") != NULL) {
    *(bool *)found = true;
  }
  return 0;
}

// jmp *0(%rip), then the target; or jmp rel32 where 'size' bytes (up to the
// next function) cannot hold that.
void
patch(char *at, uint64_t size, void *target, const char *name)
{
  unsigned char code[14];
  size_t n;
  intptr_t rel = (char *)target - (at + 5);
  if (size >= sizeof code) {
    code[0] = 0xff;
    code[1] = 0x25;
    memset(code + 2, 0, 4);
    memcpy(code + 6, &target, 8);
    n = 14;
  } else if (size >= 5 && rel == (int32_t)rel) {
    code[0] = 0xe9;
    int32_t rel32 = (int32_t)rel;
    memcpy(code + 1, &rel32, 4);
    n = 5;
  } else {
    front_fatal("cannot patch %s of the static CUDA runtime (%lu bytes)\n",
                name, (unsigned long)size);
  }
  long page = sysconf(_SC_PAGESIZE);
  char *start = (char *)((uintptr_t)at & ~(uintptr_t)(page - 1));
  size_t len = (at + n) - start;
  if (mprotect(start, len, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
    front_fatal("cannot patch %s: mprotect failed\n", name);
  }
  memcpy(at, code, n);
  mprotect(start, len, PROT_READ | PROT_EXEC);
}

__attribute__((constructor)) void
patch_static_cudart()
{
  const char *cuda = getenv("MANA_CUDA");
  if (cuda == NULL || cuda[0] == '\0' || strcmp(cuda, "0") == 0) {
    return;
  }
  std::string path;
  ElfW(Addr) base = 0;
  ExeInfo info = { false, false, {} };
  if (!exe_path(&path, &base) || !read_exe(path, &info)) {
    return;
  }
  bool shared = false;
  dl_iterate_phdr(find_shared_cudart, &shared);
  if (!info.has_symtab) {
    if (info.has_fatbin && !shared) {
      front_fatal("the executable has a static CUDA runtime but no symbol "
                  "table, so MANA cannot redirect it: build it with nvcc "
                  "-cudart=shared, or do not strip it\n");
    }
    return;
  }
  auto malloc_fn = info.funcs.find("cudaMalloc");
  if (malloc_fn == info.funcs.end()) {
    return;  // no runtime, or the shared one
  }
  for (auto &entry : info.funcs) {
    if (entry.first.compare(0, 26, "__cudaRegisterLinkedBinary") == 0) {
      front_fatal("relocatable device code (nvcc -rdc) with a static CUDA "
                  "runtime is not supported: build with -cudart=shared\n");
    }
  }
  // Its structures must be the lower half's: the same major version.  This
  // call starts nothing (Phase 0).
  auto version_fn = info.funcs.find("cudaRuntimeGetVersion");
  if (version_fn != info.funcs.end()) {
    int version = 0;
    ((cudaError_t (*)(int *))(base + version_fn->second.value))(&version);
    const char *major = strrchr(front_cudart_lib.soname, '.');
    if (major != NULL && version / 1000 != atoi(major + 1)) {
      front_fatal("the executable's static CUDA runtime is version %d, but "
                  "the lower half's is %s\n", version,
                  front_cudart_lib.soname);
    }
  }
  int patched = 0;
  for (int i = 0; i < front_cudart_lib.n; i++) {
    auto it = info.funcs.find(front_cudart_lib.names[i]);
    if (it == info.funcs.end()) {
      continue;
    }
    if (it->second.count != 1) {
      front_fatal("%s is defined %d times in the executable\n",
                  front_cudart_lib.names[i], it->second.count);
    }
    patch((char *)(base + it->second.value), room(info, it->second),
          front_cudart_dispatch[i], front_cudart_lib.names[i]);
    patched++;
  }
  front_log("redirected %d functions of the static CUDA runtime\n", patched);
}

}  // namespace
