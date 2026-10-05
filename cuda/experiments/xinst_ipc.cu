// Experiment, run under MANA: can the lower half's CUDA driver instance
// (used by a CUDA-aware MPI) access device memory that the upper half's
// instance (used by the application) allocated?
//   1. cuPointerGetAttribute in the lower instance on the upper pointer
//   2. CUDA IPC: export in upper instance, cuIpcOpenMemHandle in lower one
//   3. cuMemHostRegister-free check: copy through the opened pointer
// Lower-half functions are found in /proc/self/maps + the ELF symbol table,
// and called with the FS register switched to the lower half's TLS.
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <elf.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <asm/prctl.h>
#include <sys/syscall.h>

static long raw_arch_prctl(int code, unsigned long addr) {
  long ret;
  asm volatile("syscall" : "=a"(ret) : "0"(SYS_arch_prctl), "D"(code), "S"(addr)
               : "rcx", "r11", "memory");
  return ret;
}

// Offset of 'sym' in ELF file 'path' (dynamic symbol table).
static unsigned long sym_offset(const char *path, const char *sym) {
  int fd = open(path, O_RDONLY); struct stat st; fstat(fd, &st);
  char *b = (char*)mmap(0, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
  Elf64_Ehdr *eh = (Elf64_Ehdr*)b; Elf64_Shdr *sh = (Elf64_Shdr*)(b + eh->e_shoff);
  for (int i = 0; i < eh->e_shnum; i++) {
    if (sh[i].sh_type != SHT_DYNSYM) continue;
    Elf64_Sym *s = (Elf64_Sym*)(b + sh[i].sh_offset);
    const char *str = b + sh[sh[i].sh_link].sh_offset;
    for (size_t j = 0; j < sh[i].sh_size / sizeof(Elf64_Sym); j++)
      if (strcmp(str + s[j].st_name, sym) == 0 && s[j].st_value) return s[j].st_value;
  }
  return 0;
}

// Base of the libcuda mapped at the highest address (the lower half's).
static char *lh_libcuda(char *path) {
  FILE *f = fopen("/proc/self/maps", "r"); char l[1024]; char *base = 0;
  while (fgets(l, sizeof l, f)) {
    unsigned long a; unsigned long off; char p[512] = "";
    if (sscanf(l, "%lx-%*lx %*s %lx %*s %*s %511s", &a, &off, p) >= 2 &&
        strstr(p, "libcuda.so") && off == 0 && (char*)a > base) {
      base = (char*)a; strcpy(path, p);
    }
  }
  fclose(f); return base;
}

static unsigned long uh_fs, lh_fs;
#define IN_LH(call) ({ raw_arch_prctl(ARCH_SET_FS, lh_fs); \
                       __typeof__(call) _r = (call); \
                       raw_arch_prctl(ARCH_SET_FS, uh_fs); _r; })

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    lh_fs = *(unsigned long*)strtoul(getenv("MANA_LH_INFO_ADDR"), 0, 16);
    raw_arch_prctl(ARCH_GET_FS, (unsigned long)&uh_fs);
    char path[512]; char *base = lh_libcuda(path);
    printf("lower-half libcuda %s at %p, lh fs %lx, uh fs %lx\n", path, base, lh_fs, uh_fs);
#define LHF(_, name) __typeof__(&name) lh_##name = \
      (__typeof__(&name))(base + sym_offset(path, #name))
    LHF(_, cuInit);
    LHF(_, cuDeviceGet);
    LHF(_, cuDevicePrimaryCtxRetain);
    LHF(_, cuCtxSetCurrent);
    LHF(_, cuPointerGetAttribute);
    LHF(_, cuIpcOpenMemHandle_v2);
    LHF(_, cuMemcpyDtoH_v2);

    // Upper instance: allocate and fill.
    int *d; cudaMalloc(&d, 1 << 20);
    int pat[4] = {11, 22, 33, 44};
    cudaMemcpy(d, pat, sizeof pat, cudaMemcpyHostToDevice);
    cudaIpcMemHandle_t h; cudaError_t e = cudaIpcGetMemHandle(&h, d);
    printf("upper: d=%p ipc get %d\n", d, e);

    // Lower instance.
    CUdevice dev; CUcontext ctx;
    printf("lower: cuInit %d\n", IN_LH(lh_cuInit(0)));
    IN_LH(lh_cuDeviceGet(&dev, 0));
    printf("lower: ctx retain %d\n", IN_LH(lh_cuDevicePrimaryCtxRetain(&ctx, dev)));
    IN_LH(lh_cuCtxSetCurrent(ctx));
    unsigned mt = 0;
    CUresult r = IN_LH(lh_cuPointerGetAttribute(&mt, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)d));
    printf("lower: pointer attr of upper ptr: ret %d memtype %u\n", r, mt);
    CUdeviceptr lp = 0; CUipcMemHandle ch; memcpy(&ch, &h, sizeof ch);
    r = IN_LH(lh_cuIpcOpenMemHandle_v2(&lp, ch, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));
    printf("lower: ipc open ret %d lp=%llx\n", r, (unsigned long long)lp);
    if (r == CUDA_SUCCESS) {
      int out[4] = {0};
      r = IN_LH(lh_cuMemcpyDtoH_v2(out, lp, sizeof out));
      printf("lower: read via ipc ret %d: %d %d %d %d\n", r, out[0], out[1], out[2], out[3]);
    }
    fflush(stdout);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
