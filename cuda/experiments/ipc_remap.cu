// VMM IPC across a checkpoint of BOTH processes (as in a MANA checkpoint):
// the importer unmaps and releases its import before the checkpoint (a
// mapped import cannot be restored), keeps a dup of the fd and the VA range,
// and after restore imports again from the kept fd and maps at the same VA.
//   ipc_remap exporter SOCK      ipc_remap importer SOCK
// Both print "ready PID" after the exchange; checkpoint both, then touch
// SOCK.go; the exporter changes the data after restore, and the importer
// must see the change through the re-imported mapping.
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { \
  printf("%s:%d %s -> %d\n", role, __LINE__, #x, (int)r_); fflush(stdout); exit(1); } } while (0)
static const char *role;

static int conn(const char *path, int server) {
  int s = socket(AF_UNIX, SOCK_STREAM, 0);
  struct sockaddr_un a = { AF_UNIX }; strncpy(a.sun_path, path, sizeof a.sun_path - 1);
  if (server) { unlink(path); bind(s, (struct sockaddr *)&a, sizeof a); listen(s, 1); return accept(s, NULL, NULL); }
  while (connect(s, (struct sockaddr *)&a, sizeof a) != 0) usleep(100000);
  return s;
}
static void xfer_fd(int s, int *fd, int send) {
  char b = 0; struct iovec iov = { &b, 1 };
  char ctl[CMSG_SPACE(sizeof(int))]; memset(ctl, 0, sizeof ctl);
  struct msghdr m = { 0 }; m.msg_iov = &iov; m.msg_iovlen = 1;
  m.msg_control = ctl; m.msg_controllen = sizeof ctl;
  if (send) {
    struct cmsghdr *c = CMSG_FIRSTHDR(&m);
    c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS; c->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(c), fd, sizeof *fd); sendmsg(s, &m, 0);
  } else {
    recvmsg(s, &m, 0); memcpy(fd, CMSG_DATA(CMSG_FIRSTHDR(&m)), sizeof *fd);
  }
}
static void sync_byte(int s) { char b = 1; write(s, &b, 1); read(s, &b, 1); }

int main(int argc, char **argv) {
  role = argv[1]; const char *path = argv[2];
  int exporter = !strcmp(role, "exporter");
  CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev, 0));
  CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx, dev)); CK(cuCtxSetCurrent(ctx));
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = 0;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  size_t size; CK(cuMemGetAllocationGranularity(&size, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  CUmemAccessDesc acc = {}; acc.location = prop.location; acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  int s = conn(path, exporter);
  CUmemGenericAllocationHandle h; CUdeviceptr va; int fd;
  CK(cuMemAddressReserve(&va, size, 0, 0, 0));
  if (exporter) {
    CK(cuMemCreate(&h, size, &prop, 0));
    CK(cuMemMap(va, size, 0, h, 0)); CK(cuMemSetAccess(va, size, &acc, 1));
    CK(cuMemsetD32(va, 42, size / 4)); CK(cuCtxSynchronize());
    CK(cuMemExportToShareableHandle(&fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    xfer_fd(s, &fd, 1);
  } else {
    xfer_fd(s, &fd, 0);
    CK(cuMemImportFromShareableHandle(&h, (void *)(long)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    CK(cuMemMap(va, size, 0, h, 0)); CK(cuMemSetAccess(va, size, &acc, 1));
    int v; CK(cuMemcpyDtoH(&v, va, 4));
    printf("importer: before: %d\n", v);
    // Before the checkpoint: drop the import, keep fd and VA.
    CK(cuMemUnmap(va, size)); CK(cuMemRelease(h));
  }
  sync_byte(s);
  printf("ready %d\n", getpid()); fflush(stdout);
  char go[512]; snprintf(go, sizeof go, "%s.go", path);
  while (access(go, F_OK)) usleep(100000);
  if (exporter) {
    int v; CK(cuMemcpyDtoH(&v, va, 4));
    printf("exporter: after restore: %d\n", v);
    CK(cuMemsetD32(va, 4242, size / 4)); CK(cuCtxSynchronize());
    sync_byte(s);
    if (getenv("FRESH")) {        // re-export the restored allocation
      CUmemGenericAllocationHandle h2;
      CK(cuMemRetainAllocationHandle(&h2, (void *)va));
      int fd2; CK(cuMemExportToShareableHandle(&fd2, h2, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
      xfer_fd(s, &fd2, 1);
    }
    sync_byte(s);
  } else {
    sync_byte(s);   // the exporter has written 4242
    if (getenv("FRESH")) { close(fd); xfer_fd(s, &fd, 0); }
    CUresult r = cuMemImportFromShareableHandle(&h, (void *)(long)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    printf("importer: re-import from the kept fd -> %d\n", r);
    if (r == CUDA_SUCCESS) {
      CK(cuMemMap(va, size, 0, h, 0)); CK(cuMemSetAccess(va, size, &acc, 1));
      int v; CK(cuMemcpyDtoH(&v, va, 4));
      printf("importer: after restore, same VA %#llx: %d (%s)\n", (unsigned long long)va, v,
             v == 4242 ? "sees the exporter's restored memory" : "BAD");
    }
    sync_byte(s);
  }
  fflush(stdout);
  return 0;
}
