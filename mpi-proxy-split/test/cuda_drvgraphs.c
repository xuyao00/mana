// The driver API's graphs (run with --cuda --no-mpi).  A graph of a memset
// node, a kernel node of cuModuleGetFunction, a child graph with another, a
// copy node and a host node is instantiated once; each iteration sets the
// executable graph's kernel node to another element count and launches it,
// and every fifth one also captures a cuLaunchKernel on a stream into a
// graph of its own.  The ranks come from the launcher and never talk.

#include <cuda.h>
#include <stdlib.h>

#define MT_NO_MPI
#include "mana_test.h"
#include "cuda_driver_kernels.h"  // driver_fatbin[] (bin2c)

#define N 4096

#define CU_CHECK(call)                                                       \
  do {                                                                       \
    CUresult r_ = (call);                                                    \
    MT_CHECK(r_ == CUDA_SUCCESS, "%s: error %d", #call, (int)r_);            \
  } while (0)

static int host_calls;  // by the host node: saved and restored with the rest

static void CUDA_CB
on_host(void *data)
{
  (*(int *)data)++;
}

static CUDA_KERNEL_NODE_PARAMS
add_one_params(CUfunction add_one, void **args)
{
  CUDA_KERNEL_NODE_PARAMS p;
  memset(&p, 0, sizeof p);
  p.func = add_one;
  p.gridDimX = (N + 255) / 256;
  p.gridDimY = p.gridDimZ = 1;
  p.blockDimX = 256;
  p.blockDimY = p.blockDimZ = 1;
  p.kernelParams = args;
  return p;
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "cuda_drvgraphs");
  CU_CHECK(cuInit(0));
  int ndev;
  CU_CHECK(cuDeviceGetCount(&ndev));
  CUdevice dev;
  CU_CHECK(cuDeviceGet(&dev, mt_rank % ndev));
  CUcontext ctx;
  CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CU_CHECK(cuCtxSetCurrent(ctx));
  CUmodule module;
  CU_CHECK(cuModuleLoadFatBinary(&module, driver_fatbin));
  CUfunction add_one;
  CU_CHECK(cuModuleGetFunction(&add_one, module, "add_one"));
  CUdeviceptr a, b;
  CU_CHECK(cuMemAlloc(&a, N * sizeof(int)));
  CU_CHECK(cuMemAlloc(&b, N * sizeof(int)));
  int *host = (int *)malloc(N * sizeof(int));
  CUstream stream;
  CU_CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

  // memset(a) -> add_one(a, n) -> child: add_one(a, N) -> copy a to b -> host
  CUgraph graph, child;
  CU_CHECK(cuGraphCreate(&graph, 0));
  CUDA_MEMSET_NODE_PARAMS set;
  memset(&set, 0, sizeof set);
  set.dst = a;
  set.value = 0;
  set.elementSize = sizeof(int);
  set.width = N;
  set.height = 1;
  CUgraphNode set_node, kernel_node, child_node, copy_node, host_node;
  CU_CHECK(cuGraphAddMemsetNode(&set_node, graph, NULL, 0, &set, ctx));
  int full = N;
  void *full_args[] = {&a, &full};
  CUDA_KERNEL_NODE_PARAMS kernel = add_one_params(add_one, full_args);
  CU_CHECK(cuGraphAddKernelNode(&kernel_node, graph, &set_node, 1, &kernel));
  CU_CHECK(cuGraphCreate(&child, 0));
  CUgraphNode child_kernel;
  CU_CHECK(cuGraphAddKernelNode(&child_kernel, child, NULL, 0, &kernel));
  CU_CHECK(cuGraphAddChildGraphNode(&child_node, graph, &kernel_node, 1,
                                    child));
  CUDA_MEMCPY3D copy;
  memset(&copy, 0, sizeof copy);
  copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.srcDevice = a;
  copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.dstDevice = b;
  copy.WidthInBytes = N * sizeof(int);
  copy.Height = 1;
  copy.Depth = 1;
  CU_CHECK(cuGraphAddMemcpyNode(&copy_node, graph, &child_node, 1, &copy,
                                ctx));
  CUDA_HOST_NODE_PARAMS on = {on_host, &host_calls};
  CU_CHECK(cuGraphAddHostNode(&host_node, graph, &copy_node, 1, &on));
  CUgraphExec exec;
  CU_CHECK(cuGraphInstantiate(&exec, graph, 0));

  long it;
  for (it = 0; mt_continue(it); it++) {
    size_t count = 0;
    CU_CHECK(cuGraphGetNodes(graph, NULL, &count));
    MT_CHECK(count == 5, "iteration %ld: %zu nodes", it, count);
    CUgraphNodeType type;
    CU_CHECK(cuGraphNodeGetType(kernel_node, &type));
    MT_CHECK(type == CU_GRAPH_NODE_TYPE_KERNEL, "iteration %ld: type %d", it,
             (int)type);
    CUDA_KERNEL_NODE_PARAMS got;
    CU_CHECK(cuGraphKernelNodeGetParams(kernel_node, &got));
    MT_CHECK(got.func == add_one, "iteration %ld: another function", it);

    // The executable graph's first kernel covers n elements, the child's
    // all: b[i] == (i < n) + 1.
    int n = N - (int)(it % 4) * 512;
    void *args[] = {&a, &n};
    CUDA_KERNEL_NODE_PARAMS some = add_one_params(add_one, args);
    CU_CHECK(cuGraphExecKernelNodeSetParams(exec, kernel_node, &some));
    CU_CHECK(cuGraphLaunch(exec, stream));
    CU_CHECK(cuStreamSynchronize(stream));
    MT_CHECK(host_calls == (int)it + 1, "iteration %ld: %d host calls", it,
             host_calls);
    CU_CHECK(cuMemcpyDtoH(host, b, N * sizeof(int)));
    for (int i = 0; i < N; i++) {
      MT_CHECK(host[i] == (i < n) + 1, "iteration %ld: b[%d] is %d", it, i,
               host[i]);
    }

    if (it % 5 == 0) {
      CUgraph captured;
      CUgraphExec captured_exec;
      void *b_args[] = {&b, &full};
      CU_CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
      CU_CHECK(cuLaunchKernel(add_one, (N + 255) / 256, 1, 1, 256, 1, 1, 0,
                              stream, b_args, NULL));
      CU_CHECK(cuStreamEndCapture(stream, &captured));
      CU_CHECK(cuGraphInstantiate(&captured_exec, captured, 0));
      CU_CHECK(cuGraphLaunch(captured_exec, stream));
      CU_CHECK(cuStreamSynchronize(stream));
      CU_CHECK(cuMemcpyDtoH(host, b, N * sizeof(int)));
      for (int i = 0; i < N; i++) {
        MT_CHECK(host[i] == (i < n) + 2, "iteration %ld: captured b[%d] is "
                 "%d", it, i, host[i]);
      }
      CU_CHECK(cuGraphExecDestroy(captured_exec));
      CU_CHECK(cuGraphDestroy(captured));
    }
  }
  mt_finish(it);
  return 0;
}
