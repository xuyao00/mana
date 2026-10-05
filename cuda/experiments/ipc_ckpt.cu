// Does the CUDA checkpoint API support CUDA IPC?
//   ipc_ckpt export FILE          -- allocate, fill, write IPC handle to FILE
//   ipc_ckpt import FILE [close]  -- open the handle, read it ('close': close it)
// Each then prints "ready <pid>" and waits for FILE.go, then checks its data
// again.  Checkpoint each process with ckpt_ext in between.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
  const char *file = argv[2];
  char go[512]; snprintf(go, sizeof go, "%s.go", file);
  int *d = NULL, h[4];
  if (strcmp(argv[1], "export") == 0) {
    cudaMalloc(&d, 1 << 20);
    int v[4] = {1, 2, 3, 4};
    cudaMemcpy(d, v, sizeof v, cudaMemcpyHostToDevice);
    cudaIpcMemHandle_t ih;
    printf("export: get handle %d\n", cudaIpcGetMemHandle(&ih, d));
    FILE *f = fopen(file, "wb"); fwrite(&ih, sizeof ih, 1, f); fclose(f);
  } else {
    cudaIpcMemHandle_t ih;
    while (access(file, F_OK)) usleep(100000);
    usleep(200000);
    FILE *f = fopen(file, "rb"); fread(&ih, sizeof ih, 1, f); fclose(f);
    printf("import: open %d\n", cudaIpcOpenMemHandle((void **)&d, ih, cudaIpcMemLazyEnablePeerAccess));
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    printf("import: read %d %d %d %d\n", h[0], h[1], h[2], h[3]);
    if (argc > 3 && strcmp(argv[3], "close") == 0) {
      printf("import: close %d\n", cudaIpcCloseMemHandle(d));
      cudaMalloc(&d, 1 << 20);   // keep some device state to check
      cudaMemcpy(d, h, sizeof h, cudaMemcpyHostToDevice);
    }
  }
  printf("ready %d\n", getpid()); fflush(stdout);
  while (access(go, F_OK)) usleep(100000);
  cudaError_t e = cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  printf("%s after: %d -> %d %d %d %d\n", argv[1], e, h[0], h[1], h[2], h[3]);
  fflush(stdout);
  sleep(1);
  return 0;
}
