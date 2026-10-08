// The application's CUDA runtime, for MANA's NCCL code in the upper half
// (the drain and the rebuild).  libmana.so does not link with libcudart: these
// find the application's (or load libcudart.so) on first use.
#ifndef MANA_NCCL_CUDA_H
#define MANA_NCCL_CUDA_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t mana_cudaSetDevice(int device);
cudaError_t mana_cudaGetDevice(int *device);
cudaError_t mana_cudaGetDeviceCount(int *count);
cudaError_t mana_cudaMalloc(void **ptr, size_t size);
cudaError_t mana_cudaFree(void *ptr);
cudaError_t mana_cudaMemset(void *ptr, int value, size_t count);
cudaError_t mana_cudaMemcpy(void *dst, const void *src, size_t count,
                            enum cudaMemcpyKind kind);
cudaError_t mana_cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                 enum cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t mana_cudaStreamSynchronize(cudaStream_t stream);
const char *mana_cudaGetErrorString(cudaError_t error);

#ifdef __cplusplus
}
#endif

#ifndef MANA_NCCL_CUDA_IMPL
# define cudaSetDevice mana_cudaSetDevice
# define cudaGetDevice mana_cudaGetDevice
# define cudaGetDeviceCount mana_cudaGetDeviceCount
# define cudaMalloc mana_cudaMalloc
# define cudaFree mana_cudaFree
# define cudaMemset mana_cudaMemset
# define cudaMemcpy mana_cudaMemcpy
# define cudaMemcpyAsync mana_cudaMemcpyAsync
# define cudaStreamSynchronize mana_cudaStreamSynchronize
# define cudaGetErrorString mana_cudaGetErrorString
#endif

#endif
