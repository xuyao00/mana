// Which kinds of CUDA IPC can the CUDA checkpoint API handle?
//   ipc_ckpt2 KIND exporter SOCK [STEP]
//   ipc_ckpt2 KIND importer SOCK [STEP]
// KIND: legacy (cudaIpcGetMemHandle), vmm (cuMemCreate + POSIX fd),
//       pool (cudaMemPool + POSIX fd).
// STEP, applied after the exchange, before the checkpoint:
//   keep     -- exporter keeps the export / importer keeps the mapping
//   release  -- importer: unmap/close; exporter: (legacy) free the buffer,
//               (vmm) release the allocation handle but keep the mapping,
//               (pool) free the pointer, keep the pool.
// Both print "ready PID" and wait for SOCK.go; then they check their data.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define SIZE (2 << 20)
#define CK(x) do { CUresult r_ = (CUresult)(x); if (r_ != 0) { \
  printf("%s:%d %s -> %d\n", role, __LINE__, #x, (int)r_); exit(1); } } while (0)

static const char *role;

static int sock_server(const char *path) {
  int s = socket(AF_UNIX, SOCK_STREAM, 0);
  struct sockaddr_un a = { AF_UNIX }; strncpy(a.sun_path, path, sizeof a.sun_path - 1);
  unlink(path); bind(s, (struct sockaddr *)&a, sizeof a); listen(s, 1);
  return accept(s, NULL, NULL);
}
static int sock_client(const char *path) {
  int s = socket(AF_UNIX, SOCK_STREAM, 0);
  struct sockaddr_un a = { AF_UNIX }; strncpy(a.sun_path, path, sizeof a.sun_path - 1);
  while (connect(s, (struct sockaddr *)&a, sizeof a) != 0) usleep(100000);
  return s;
}
// Sends 'data' and optionally fd 'fd' (SCM_RIGHTS).
static void send_msg(int s, const void *data, int len, int fd) {
  struct iovec iov = { (void *)data, (size_t)len };
  char ctl[CMSG_SPACE(sizeof(int))]; memset(ctl, 0, sizeof ctl);
  struct msghdr m = { 0 }; m.msg_iov = &iov; m.msg_iovlen = 1;
  if (fd >= 0) {
    m.msg_control = ctl; m.msg_controllen = sizeof ctl;
    struct cmsghdr *c = CMSG_FIRSTHDR(&m);
    c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS; c->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(c), &fd, sizeof fd);
  }
  sendmsg(s, &m, 0);
}
static int recv_msg(int s, void *data, int len) {
  struct iovec iov = { data, (size_t)len };
  char ctl[CMSG_SPACE(sizeof(int))];
  struct msghdr m = { 0 }; m.msg_iov = &iov; m.msg_iovlen = 1;
  m.msg_control = ctl; m.msg_controllen = sizeof ctl;
  recvmsg(s, &m, MSG_WAITALL);
  struct cmsghdr *c = CMSG_FIRSTHDR(&m);
  int fd = -1;
  if (c && c->cmsg_type == SCM_RIGHTS) memcpy(&fd, CMSG_DATA(c), sizeof fd);
  return fd;
}

int main(int argc, char **argv) {
  const char *kind = argv[1]; role = argv[2]; const char *path = argv[3];
  const char *step = argc > 4 ? argv[4] : "keep";
  int exporter = strcmp(role, "exporter") == 0;
  CK(cuInit(0));
  CUdevice dev; CK(cuDeviceGet(&dev, 0));
  CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx, dev)); CK(cuCtxSetCurrent(ctx));
  CUdeviceptr own = 0;                 // memory to check after restore
  CK(cuMemAlloc(&own, SIZE)); CK(cuMemsetD32(own, 777, SIZE / 4));
  CUdeviceptr ptr = 0; size_t size = SIZE;
  int s = exporter ? sock_server(path) : sock_client(path);
  CUmemGenericAllocationHandle h = 0;
  cudaMemPool_t pool = NULL;

  if (!strcmp(kind, "legacy")) {
    if (exporter) {
      CK(cuMemAlloc(&ptr, size)); CK(cuMemsetD32(ptr, 42, size / 4));
      CUipcMemHandle ih; CK(cuIpcGetMemHandle(&ih, ptr));
      send_msg(s, &ih, sizeof ih, -1);
    } else {
      CUipcMemHandle ih; recv_msg(s, &ih, sizeof ih);
      CK(cuIpcOpenMemHandle(&ptr, ih, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));
    }
  } else if (!strcmp(kind, "vmm")) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = 0;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    size_t gran; CK(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    size = (SIZE + gran - 1) / gran * gran;
    int fd = -1;
    if (exporter) {
      CK(cuMemCreate(&h, size, &prop, 0));
    } else {
      fd = recv_msg(s, &size, sizeof size);
      CK(cuMemImportFromShareableHandle(&h, (void *)(long)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
      close(fd);
    }
    CK(cuMemAddressReserve(&ptr, size, 0, 0, 0));
    CK(cuMemMap(ptr, size, 0, h, 0));
    CUmemAccessDesc acc = {}; acc.location = prop.location; acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CK(cuMemSetAccess(ptr, size, &acc, 1));
    if (exporter) {
      CK(cuMemsetD32(ptr, 42, size / 4));
      CK(cuCtxSynchronize());
      CK(cuMemExportToShareableHandle(&fd, h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
      send_msg(s, &size, sizeof size, fd);
      close(fd);
    }
  } else if (!strcmp(kind, "pool")) {
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypePosixFileDescriptor;
    props.location.type = cudaMemLocationTypeDevice; props.location.id = 0;
    int fd = -1;
    cudaMemPoolPtrExportData ed;
    if (exporter) {
      CK(cudaMemPoolCreate(&pool, &props));
      CK(cudaMallocFromPoolAsync((void **)&ptr, size, pool, 0));
      CK(cudaStreamSynchronize(0));
      CK(cuMemsetD32(ptr, 42, size / 4));
      CK(cudaMemPoolExportToShareableHandle(&fd, pool, cudaMemHandleTypePosixFileDescriptor, 0));
      CK(cudaMemPoolExportPointer(&ed, (void *)ptr));
      send_msg(s, &ed, sizeof ed, fd);
      close(fd);
    } else {
      fd = recv_msg(s, &ed, sizeof ed);
      CK(cudaMemPoolImportFromShareableHandle(&pool, (void *)(long)fd, cudaMemHandleTypePosixFileDescriptor, 0));
      close(fd);
      CK(cudaMemPoolImportPointer((void **)&ptr, pool, &ed));
    }
  }
  int v = 0;
  CK(cuMemcpyDtoH(&v, ptr + 4096, 4));
  printf("%s %s: shared buffer reads %d\n", kind, role, v);
  char ok = 1;
  send_msg(s, &ok, 1, -1); recv_msg(s, &ok, 1);   // both have used it

  if (!strcmp(step, "release")) {
    if (!strcmp(kind, "legacy")) {
      CK(exporter ? cuMemFree(ptr) : cuIpcCloseMemHandle(ptr));
    } else if (!strcmp(kind, "vmm")) {
      if (exporter) {
        CK(cuMemRelease(h));            // keep the mapping, drop the handle
      } else {
        CK(cuMemUnmap(ptr, size)); CK(cuMemRelease(h)); CK(cuMemAddressFree(ptr, size));
      }
    } else {
      CK(cudaFreeAsync((void *)ptr, 0)); CK(cudaStreamSynchronize(0));
      if (!exporter) CK(cudaMemPoolDestroy(pool));
    }
    ptr = 0;
  }
  printf("ready %d\n", getpid()); fflush(stdout);
  char go[512]; snprintf(go, sizeof go, "%s.go", path);
  while (access(go, F_OK)) usleep(100000);
  int o = 0;
  CUresult r = cuMemcpyDtoH(&o, own + 4096, 4);
  int w = -1;
  if (ptr) cuMemcpyDtoH(&w, ptr + 4096, 4);
  printf("%s %s after: own memory %d (%d)%s\n", kind, role, o, r,
         ptr ? (w == 42 ? ", shared OK" : ", shared BAD") : "");
  fflush(stdout);
  return 0;
}
