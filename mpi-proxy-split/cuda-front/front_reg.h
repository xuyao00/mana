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

// Functions of the front libraries that the CUDA headers do not declare:
// cudart's registration and launch interface, which nvcc's host code calls
// (crt/host_runtime.h), and old driver versions that programs built with
// older headers call.

#ifndef _MANA_CUDA_FRONT_REG_H
#define _MANA_CUDA_FRONT_REG_H

#include <cuda.h>
#include <cuda_runtime_api.h>

extern "C" {
void **__cudaRegisterFatBinary(void *fatCubin);
void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
void __cudaUnregisterFatBinary(void **fatCubinHandle);
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize);
void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                       char *deviceAddress, const char *deviceName, int ext,
                       size_t size, int constant, int global);
void __cudaRegisterManagedVar(void **fatCubinHandle, void **hostVarPtrAddress,
                              char *deviceAddress, const char *deviceName,
                              int ext, size_t size, int constant, int global);
char __cudaInitModule(void **fatCubinHandle);
unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                     size_t sharedMem,
                                     struct CUstream_st *stream);
cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                       size_t *sharedMem, void *stream);
cudaError_t __cudaGetKernel(cudaKernel_t *kernel, const void *hostFun);
cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 gridDim,
                               dim3 blockDim, void **args, size_t sharedMem,
                               cudaStream_t stream);
cudaError_t __cudaLaunchKernel_ptsz(cudaKernel_t kernel, dim3 gridDim,
                                    dim3 blockDim, void **args,
                                    size_t sharedMem, cudaStream_t stream);

// cuda.h before CUDA 13: #define cuCtxCreate cuCtxCreate_v2
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
// The per-thread default stream's versions, which the headers declare only
// with CUDA_API_PER_THREAD_DEFAULT_STREAM (nvcc --default-stream per-thread).
CUresult cuStreamGetCtx_ptsz(CUstream hStream, CUcontext *pctx);
CUresult cuLaunchHostFunc_ptsz(CUstream hStream, CUhostFn fn, void *userData);
CUresult cuStreamAddCallback_ptsz(CUstream hStream, CUstreamCallback callback,
                                  void *userData, unsigned int flags);
CUresult cuMemAllocAsync_ptsz(CUdeviceptr *dptr, size_t bytesize,
                              CUstream hStream);
CUresult cuMemFreeAsync_ptsz(CUdeviceptr dptr, CUstream hStream);
CUresult cuMemAllocFromPoolAsync_ptsz(CUdeviceptr *dptr, size_t bytesize,
                                      CUmemoryPool pool, CUstream hStream);
CUresult cuEventRecord_ptsz(CUevent hEvent, CUstream hStream);
CUresult cuEventRecordWithFlags_ptsz(CUevent hEvent, CUstream hStream,
                                     unsigned int flags);
cudaError_t cudaLaunchHostFunc_ptsz(cudaStream_t stream, cudaHostFn_t fn,
                                    void *userData);
cudaError_t cudaStreamAddCallback_ptsz(cudaStream_t stream,
                                       cudaStreamCallback_t callback,
                                       void *userData, unsigned int flags);
cudaError_t cudaMallocAsync_ptsz(void **devPtr, size_t size,
                                 cudaStream_t hStream);
cudaError_t cudaFreeAsync_ptsz(void *devPtr, cudaStream_t hStream);
cudaError_t cudaMallocFromPoolAsync_ptsz(void **ptr, size_t size,
                                         cudaMemPool_t memPool,
                                         cudaStream_t stream);
cudaError_t cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventRecordWithFlags_ptsz(cudaEvent_t event,
                                          cudaStream_t stream,
                                          unsigned int flags);
cudaError_t cudaStreamBeginCapture_ptsz(cudaStream_t stream,
                                        enum cudaStreamCaptureMode mode);
cudaError_t cudaStreamBeginCaptureToGraph_ptsz(
  cudaStream_t stream, cudaGraph_t graph,
  const cudaGraphNode_t *dependencies,
  const cudaGraphEdgeData *dependencyData, size_t numDependencies,
  enum cudaStreamCaptureMode mode);
cudaError_t cudaStreamEndCapture_ptsz(cudaStream_t stream,
                                      cudaGraph_t *pGraph);
cudaError_t cudaStreamGetCaptureInfo_ptsz(
  cudaStream_t stream, enum cudaStreamCaptureStatus *captureStatus_out,
  unsigned long long *id_out, cudaGraph_t *graph_out,
  const cudaGraphNode_t **dependencies_out,
  const cudaGraphEdgeData **edgeData_out, size_t *numDependencies_out);
cudaError_t cudaStreamUpdateCaptureDependencies_ptsz(
  cudaStream_t stream, cudaGraphNode_t *dependencies,
  const cudaGraphEdgeData *dependencyData, size_t numDependencies,
  unsigned int flags);
cudaError_t cudaGraphInstantiateWithParams_ptsz(
  cudaGraphExec_t *pGraphExec, cudaGraph_t graph,
  cudaGraphInstantiateParams *instantiateParams);
cudaError_t cudaGraphLaunch_ptsz(cudaGraphExec_t graphExec,
                                 cudaStream_t stream);
cudaError_t cudaMemcpy3D_ptds(const struct cudaMemcpy3DParms *p);
cudaError_t cudaMemcpy3DAsync_ptsz(const struct cudaMemcpy3DParms *p,
                                   cudaStream_t stream);
cudaError_t cudaMemcpy3DPeer_ptds(const struct cudaMemcpy3DPeerParms *p);
cudaError_t cudaMemcpy3DPeerAsync_ptsz(const struct cudaMemcpy3DPeerParms *p,
                                       cudaStream_t stream);
CUresult cuMemcpy2D_v2_ptds(const CUDA_MEMCPY2D *pCopy);
CUresult cuMemcpy2DUnaligned_v2_ptds(const CUDA_MEMCPY2D *pCopy);
CUresult cuMemcpy2DAsync_v2_ptsz(const CUDA_MEMCPY2D *pCopy,
                                 CUstream hStream);
CUresult cuMemcpy3D_v2_ptds(const CUDA_MEMCPY3D *pCopy);
CUresult cuMemcpy3DAsync_v2_ptsz(const CUDA_MEMCPY3D *pCopy,
                                 CUstream hStream);
CUresult cuMemcpy3DPeer_ptds(const CUDA_MEMCPY3D_PEER *pCopy);
CUresult cuMemcpy3DPeerAsync_ptsz(const CUDA_MEMCPY3D_PEER *pCopy,
                                  CUstream hStream);
}

#endif  // ifndef _MANA_CUDA_FRONT_REG_H
