// The lower half's audit module (LD_AUDIT; see README): makes a library that
// has a CUDA runtime linked in statically (NVIDIA's libnccl) use the
// application's runtime instead, as a library linked with libcudart.so does.
//
// That embedded runtime would talk to the lower half's libcuda, which is
// MANA's forwarding shim (../cuda-forward), and its integrity check fails
// against the shim (cudaErrorSoftwareValidityNotEstablished).  So when such a
// library is mapped -- before its relocation and constructors -- each
// function of its embedded runtime (the local cuda* and __cuda* functions of
// its .symtab) that the libcudart.so shim also exports gets a jump to the
// shim's function.  The constructors' kernel registrations then go to the
// shim too, which replays them into the application's runtime.
//
// Built without libc, like ../mpi-wrappers/mpi_stub_audit.c: an audit module
// gets a link namespace of its own, and a second libc there would have a heap
// of its own in the lower half.

#define _GNU_SOURCE  // For LAV_CURRENT and Lmid_t
#include <elf.h>
#include <fcntl.h>
#include <link.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/syscall.h>

#ifndef SEEK_END
# define SEEK_END 2
#endif

// The libraries whose embedded runtime is replaced (basename prefixes).
static const char *const patched_libraries[] = {
  "libnccl.so", "libnccl-net", NULL
};

static long
sys6(long nr, long a, long b, long c, long d, long e, long f)
{
  long ret;
  register long r10 asm("r10") = d;
  register long r8 asm("r8") = e;
  register long r9 asm("r9") = f;
  asm volatile("syscall" : "=a"(ret)
               : "0"(nr), "D"(a), "S"(b), "d"(c), "r"(r10), "r"(r8), "r"(r9)
               : "rcx", "r11", "memory");
  return ret;
}

#define sys3(nr, a, b, c) sys6(nr, (long)(a), (long)(b), (long)(c), 0, 0, 0)

static size_t
length(const char *s)
{
  size_t n = 0;
  while (s[n] != '\0') {
    n++;
  }
  return n;
}

static int
same(const char *s, const char *t)
{
  for (; *s != '\0' && *s == *t; s++, t++) {
  }
  return *s == *t;
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

static const char *
basename_of(const char *path)
{
  const char *name = path;
  for (const char *p = path; *p != '\0'; p++) {
    if (*p == '/') {
      name = p + 1;
    }
  }
  return name;
}

static void
say(const char *a, const char *b, long n, const char *c)
{
  char buf[512];
  size_t k = 0;
  const char *parts[] = { "MANA: ", a, b, NULL };
  for (int i = 0; parts[i] != NULL; i++) {
    for (const char *p = parts[i]; *p != '\0' && k < sizeof buf - 40; p++) {
      buf[k++] = *p;
    }
  }
  if (n >= 0) {
    char digits[24];
    int d = 0;
    do {
      digits[d++] = '0' + n % 10;
      n /= 10;
    } while (n > 0);
    buf[k++] = ' ';
    while (d > 0) {
      buf[k++] = digits[--d];
    }
  }
  for (const char *p = c; *p != '\0' && k < sizeof buf - 2; p++) {
    buf[k++] = *p;
  }
  buf[k++] = '\n';
  sys3(SYS_write, 2, buf, k);
}

// Whether MANA_FWD_VERBOSE is set in /proc/self/environ (no environ here).
static int
verbose(void)
{
  static int v = -1;
  if (v < 0) {
    static char env[65536];
    v = 0;
    long fd = sys3(SYS_openat, AT_FDCWD, "/proc/self/environ", O_RDONLY);
    if (fd >= 0) {
      long n = sys3(SYS_read, fd, env, sizeof env - 1);
      sys3(SYS_close, fd, 0, 0);
      for (long i = 0; i >= 0 && i < n; i += length(env + i) + 1) {
        if (starts_with(env + i, "MANA_FWD_VERBOSE=") &&
            env[i + 17] != '\0' && env[i + 17] != '0') {
          v = 1;
        }
      }
    }
  }
  return v;
}

// --- symbols of a loaded object (GNU hash), as in mpi_stub_audit.c --------

static uintptr_t
dynamic_address(const struct link_map *map, ElfW(Addr) addr)
{
  return addr < map->l_addr ? map->l_addr + addr : addr;
}

static uintptr_t
lookup(const struct link_map *map, const char *name)
{
  const ElfW(Sym) *symtab = NULL;
  const char *strtab = NULL;
  const uint32_t *table = NULL;
  uint32_t hash = 5381;

  for (const ElfW(Dyn) *d = map->l_ld; d->d_tag != DT_NULL; d++) {
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
  const uint32_t *buckets =
    (const uint32_t *)((const ElfW(Addr) *)&table[4] + table[2]);
  const uint32_t *chain = buckets + table[0];
  for (const char *p = name; *p != '\0'; p++) {
    hash = hash * 33 + (unsigned char)*p;
  }
  uint32_t i = buckets[hash % table[0]];
  if (i < table[1]) {
    return 0;
  }
  for (;; i++) {
    uint32_t h = chain[i - table[1]];
    const ElfW(Sym) *s = &symtab[i];
    if ((h | 1) == (hash | 1) && s->st_shndx != SHN_UNDEF &&
        ELF64_ST_TYPE(s->st_info) == STT_FUNC && same(strtab + s->st_name, name)) {
      return map->l_addr + s->st_value;
    }
    if (h & 1) {
      return 0;
    }
  }
}

// --- patching ---------------------------------------------------------------

// The libcudart.so shim of the forwarding shims, once loaded.
static struct link_map *cudart_shim;

// movabs $target, %r11; jmp *%r11
#define ABS_JUMP 13
static void
write_abs_jump(unsigned char *at, uintptr_t target)
{
  at[0] = 0x49;
  at[1] = 0xbb;
  for (int i = 0; i < 8; i++) {
    at[2 + i] = (unsigned char)(target >> (8 * i));
  }
  at[10] = 0x41;
  at[11] = 0xff;
  at[12] = 0xe3;
}

// jmp rel32
static void
write_rel_jump(unsigned char *at, uintptr_t target)
{
  int32_t rel = (int32_t)(target - ((uintptr_t)at + 5));
  at[0] = 0xe9;
  for (int i = 0; i < 4; i++) {
    at[1 + i] = (unsigned char)((uint32_t)rel >> (8 * i));
  }
}

// Pads for functions too short for ABS_JUMP: a page of absolute jumps,
// mapped within reach of a rel32 jump from 'near'.
static unsigned char *pads;
static size_t n_pads;

static unsigned char *
pad_near(uintptr_t near, uintptr_t target)
{
  if (pads == NULL) {
    for (uintptr_t hint = (near & ~0xfffffUL) - (64UL << 20);
         hint > (1UL << 20) && pads == NULL; hint -= (64UL << 20)) {
      long p = sys6(SYS_mmap, hint, 65536, PROT_READ | PROT_WRITE | PROT_EXEC,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
      if (p > 0 && (uintptr_t)p == hint) {
        pads = (unsigned char *)p;
      }
    }
  }
  if (pads == NULL || n_pads >= 65536 / 16) {
    return NULL;
  }
  long distance = (long)((uintptr_t)pads - near);
  if (distance > 0x7fff0000L || distance < -0x7fff0000L) {
    return NULL;
  }
  unsigned char *pad = pads + 16 * n_pads++;
  write_abs_jump(pad, target);
  return pad;
}

// The local cuda* functions of 'map's .symtab that the shim exports.
static void
redirect_embedded_cudart(struct link_map *map)
{
  long fd = sys3(SYS_openat, AT_FDCWD, map->l_name, O_RDONLY);
  if (fd < 0) {
    say("cannot open ", map->l_name, -1, "");
    return;
  }
  long size = sys3(SYS_lseek, fd, 0, SEEK_END);
  long file = sys6(SYS_mmap, 0, size, PROT_READ, MAP_PRIVATE, fd, 0);
  sys3(SYS_close, fd, 0, 0);
  if (size <= 0 || (file < 0 && file > -4096)) {
    say("cannot map ", map->l_name, -1, "");
    return;
  }
  const unsigned char *f = (const unsigned char *)file;
  const Elf64_Ehdr *eh = (const Elf64_Ehdr *)f;
  const Elf64_Shdr *sh = (const Elf64_Shdr *)(f + eh->e_shoff);
  const Elf64_Sym *syms = NULL;
  const char *names = NULL;
  size_t n_syms = 0;
  for (int i = 0; i < eh->e_shnum; i++) {
    if (sh[i].sh_type == SHT_SYMTAB) {
      syms = (const Elf64_Sym *)(f + sh[i].sh_offset);
      n_syms = sh[i].sh_size / sizeof(Elf64_Sym);
      names = (const char *)(f + sh[sh[i].sh_link].sh_offset);
    }
  }
  if (syms == NULL) {
    say(basename_of(map->l_name), ": no symbol table: its CUDA runtime "
        "cannot be redirected", -1, "");
    sys6(SYS_munmap, file, size, 0, 0, 0, 0);
    return;
  }

  // Only a library with an embedded runtime: one that defines cudaMalloc.
  int embedded = 0;
  uintptr_t lo = ~0UL, hi = 0;
  for (size_t i = 0; i < n_syms; i++) {
    const char *name = names + syms[i].st_name;
    if (ELF64_ST_TYPE(syms[i].st_info) != STT_FUNC ||
        syms[i].st_shndx == SHN_UNDEF || syms[i].st_size == 0 ||
        !(starts_with(name, "cuda") || starts_with(name, "__cuda"))) {
      continue;
    }
    embedded |= same(name, "cudaMalloc");
    uintptr_t at = map->l_addr + syms[i].st_value;
    lo = at < lo ? at : lo;
    hi = at + syms[i].st_size > hi ? at + syms[i].st_size : hi;
  }
  if (!embedded) {
    sys6(SYS_munmap, file, size, 0, 0, 0, 0);
    return;
  }

  uintptr_t page_lo = lo & ~4095UL, page_hi = (hi + 4095) & ~4095UL;
  if (sys3(SYS_mprotect, page_lo, page_hi - page_lo,
           PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
    say("cannot write the code of ", map->l_name, -1, "");
    sys6(SYS_munmap, file, size, 0, 0, 0, 0);
    return;
  }
  // The pads of an earlier library are read-only by now.
  if (pads != NULL) {
    sys3(SYS_mprotect, pads, 65536, PROT_READ | PROT_WRITE | PROT_EXEC);
  }
  long patched = 0, skipped = 0;
  for (size_t i = 0; i < n_syms; i++) {
    const char *name = names + syms[i].st_name;
    if (ELF64_ST_TYPE(syms[i].st_info) != STT_FUNC ||
        syms[i].st_shndx == SHN_UNDEF || syms[i].st_size == 0 ||
        !(starts_with(name, "cuda") || starts_with(name, "__cuda"))) {
      continue;
    }
    uintptr_t target = lookup(cudart_shim, name);
    if (target == 0) {
      continue;   // An internal function of the runtime: not called.
    }
    unsigned char *at = (unsigned char *)(map->l_addr + syms[i].st_value);
    unsigned char *pad;
    if (syms[i].st_size >= ABS_JUMP) {
      write_abs_jump(at, target);
      patched++;
    } else if (syms[i].st_size >= 5 &&
               (pad = pad_near((uintptr_t)at, target)) != NULL) {
      write_rel_jump(at, (uintptr_t)pad);
      patched++;
    } else {
      // Too short to patch: in NVIDIA's libnccl, only __cudaGetProcAddress,
      // a "return 0" that libnccl does not call.
      if (verbose()) {
        say(basename_of(map->l_name), ": cannot redirect ", -1, name);
      }
      skipped++;
    }
  }
  sys3(SYS_mprotect, page_lo, page_hi - page_lo, PROT_READ | PROT_EXEC);
  if (pads != NULL) {
    sys3(SYS_mprotect, pads, 65536, PROT_READ | PROT_EXEC);
  }
  sys6(SYS_munmap, file, size, 0, 0, 0, 0);
  if (verbose()) {
    say(basename_of(map->l_name), ": CUDA runtime functions redirected to "
        "the application's:", patched, "");
  }
}

// --- the audit interface ------------------------------------------------------

unsigned int
la_version(unsigned int version)
{
  (void)version;
  return LAV_CURRENT;
}

// Libraries to patch, mapped before the shim (e.g. libnccl as a dependency
// of libmpi, at startup): patched once all the objects being loaded are
// mapped, before their relocation and constructors (la_activity()).
static struct link_map *pending[16];
static int n_pending;

static void
patch_pending(void)
{
  for (int i = 0; i < n_pending; i++) {
    if (cudart_shim == NULL) {
      say(basename_of(pending[i]->l_name), ": the libcudart shim is not "
          "loaded: its CUDA runtime is not redirected", -1, "");
    } else {
      redirect_embedded_cudart(pending[i]);
    }
  }
  n_pending = 0;
}

void
la_activity(uintptr_t *cookie, unsigned int flag)
{
  (void)cookie;
  if (flag == LA_ACT_CONSISTENT) {
    patch_pending();
  }
}

unsigned int
la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie)
{
  (void)cookie;
  if (lmid != LM_ID_BASE || map->l_name == NULL || map->l_name[0] == '\0') {
    return 0;
  }
  if (cudart_shim == NULL && lookup(map, "mana_fwd_cudart_on_upper") != 0) {
    cudart_shim = map;
    return 0;
  }
  const char *name = basename_of(map->l_name);
  for (int i = 0; patched_libraries[i] != NULL; i++) {
    if (starts_with(name, patched_libraries[i])) {
      if (n_pending < 16) {
        pending[n_pending++] = map;
      }
      break;
    }
  }
  return 0;
}
