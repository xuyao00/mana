// See nccl_cuda.h.
#define _GNU_SOURCE
#define MANA_NCCL_CUDA_IMPL
#include <dlfcn.h>
#include <stdio.h>

#include "nccl_cuda.h"

static void *
cudart_sym(const char *name)
{
  static void *handle = NULL;
  // The application's libcudart, if it uses a shared one.
  void *fn = dlsym(RTLD_DEFAULT, name);
  if (fn != NULL) {
    return fn;
  }
  if (handle == NULL) {
    const char *libs[] = { "libcudart.so.13", "libcudart.so.12", NULL };
    for (int i = 0; handle == NULL && libs[i] != NULL; i++) {
      handle = dlopen(libs[i], RTLD_NOW | RTLD_GLOBAL);
    }
    if (handle == NULL) {
      fprintf(stderr, "MANA: NCCL support: cannot load libcudart: %s\n",
              dlerror());
      return NULL;
    }
  }
  return dlsym(handle, name);
}

#define CUDART(ret, name, params, args)                                     \
  ret mana_##name params                                                     \
  {                                                                          \
    static ret (*fn) params = NULL;                                          \
    if (fn == NULL) {                                                        \
      *(void **)&fn = cudart_sym(#name);                                     \
      if (fn == NULL) {                                                      \
        return cudaErrorInitializationError;                                 \
      }                                                                      \
    }                                                                        \
    return fn args;                                                          \
  }

CUDART(cudaError_t, cudaSetDevice, (int device), (device))
CUDART(cudaError_t, cudaGetDevice, (int *device), (device))
CUDART(cudaError_t, cudaGetDeviceCount, (int *count), (count))
CUDART(cudaError_t, cudaMalloc, (void **ptr, size_t size), (ptr, size))
CUDART(cudaError_t, cudaFree, (void *ptr), (ptr))
CUDART(cudaError_t, cudaMemset, (void *ptr, int value, size_t count),
       (ptr, value, count))
CUDART(cudaError_t, cudaMemcpy,
       (void *dst, const void *src, size_t count, enum cudaMemcpyKind kind),
       (dst, src, count, kind))
CUDART(cudaError_t, cudaMemcpyAsync,
       (void *dst, const void *src, size_t count, enum cudaMemcpyKind kind,
        cudaStream_t stream),
       (dst, src, count, kind, stream))
CUDART(cudaError_t, cudaStreamSynchronize, (cudaStream_t stream), (stream))

const char *
mana_cudaGetErrorString(cudaError_t error)
{
  static const char *(*fn)(cudaError_t) = NULL;
  if (fn == NULL) {
    *(void **)&fn = cudart_sym("cudaGetErrorString");
    if (fn == NULL) {
      return "(no libcudart)";
    }
  }
  return fn(error);
}
