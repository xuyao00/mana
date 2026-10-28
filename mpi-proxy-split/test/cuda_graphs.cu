// CUDA graphs (run with MANA's --cuda).  Every iteration launches:
//  * A: a captured graph whose graph is destroyed after instantiation: a
//    copy from the host, a memset, a fork to a second stream and a join, two
//    kernels, a host function and a copy to the host;
//  * B: a graph built node by node and kept, whose executable graph gets a
//    kernel node update for the next iteration through the graph's node
//    handle (so after a restart it must have the last one), with a node
//    disabled, an event record node, a child graph, copies through a CUDA
//    array, and a host node;
//  * C: an executable graph updated by cudaGraphExecUpdate from a new
//    capture every iteration; it runs before the update, so after a restart
//    it must have the last update's parameters;
//  * D: a captured launch of a kernel of cuModuleLoadData (the driver API),
//    whose module was loaded after other graphs were instantiated;
//  * E: a captured graph kept, instantiated and destroyed every iteration,
//    whose node handle must still read its parameters.
// And a capture that fails, after cudaStreamGetCaptureInfo named its graph:
// the graph is gone, which a checkpoint must not trip on.
// A sum goes to MPI from device memory.

#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>

#include "mana_test.h"

#define N 256

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t e_ = (call);                                                 \
    MT_CHECK(e_ == cudaSuccess, "%s: %s", #call, cudaGetErrorString(e_));    \
  } while (0)

#define DRV_CHECK(call)                                                      \
  do {                                                                       \
    CUresult r_ = (call);                                                    \
    MT_CHECK(r_ == CUDA_SUCCESS, "%s: error %d", #call, (int)r_);            \
  } while (0)

__global__ void
add(int *y, const int *x, int v)
{
  int i = threadIdx.x;
  y[i] += x[i] + v;
}

__global__ void
scale(int *y, int v)
{
  y[threadIdx.x] *= v;
}

__global__ void
accumulate(int *acc, int v)
{
  acc[threadIdx.x] += v;
}

__global__ void
set_to(int *z, int v)
{
  z[threadIdx.x] = v * 3 + threadIdx.x;
}

__global__ void
bump(int *counter)
{
  (*counter)++;
}

// For the driver API: *p += v.  Its module's global must come back at its
// address, though the module was loaded after graphs were instantiated.
static const char *ptx =
  ".version 7.0\n.target sm_52\n.address_size 64\n"
  ".visible .global .u32 loaded_after;\n"
  ".visible .entry bump_by(.param .u64 p, .param .u32 v) {\n"
  "  .reg .u64 %rd<3>; .reg .u32 %r<4>;\n"
  "  ld.param.u64 %rd1, [p]; ld.param.u32 %r1, [v];\n"
  "  cvta.to.global.u64 %rd2, %rd1; ld.global.u32 %r2, [%rd2];\n"
  "  add.u32 %r3, %r2, %r1; st.global.u32 [%rd2], %r3; ret; }\n";

static int host_calls_a, host_calls_b;

static void CUDART_CB
count_a(void *arg)
{
  __atomic_add_fetch((int *)arg, 1, __ATOMIC_SEQ_CST);
}

static void CUDART_CB
count_b(void *arg)
{
  __atomic_add_fetch((int *)arg, 1, __ATOMIC_SEQ_CST);
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "cuda_graphs");
  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  CUDA_CHECK(cudaSetDevice(mt_rank % ndev));

  int *x, *y, *acc, *out, *z, *w, *counter, *counter2, *never, *d_sum;
  CUDA_CHECK(cudaMalloc(&x, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&y, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&acc, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&out, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&z, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&w, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&counter, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&counter2, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&never, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_sum, sizeof(int)));
  CUDA_CHECK(cudaMemset(acc, 0, N * sizeof(int)));
  CUDA_CHECK(cudaMemset(w, 0, N * sizeof(int)));
  CUDA_CHECK(cudaMemset(counter, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(counter2, 0, sizeof(int)));
  cudaArray_t array;
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
  CUDA_CHECK(cudaMallocArray(&array, &desc, N, 1));
  cudaStream_t s1, s2;
  CUDA_CHECK(cudaStreamCreate(&s1));
  CUDA_CHECK(cudaStreamCreate(&s2));
  cudaEvent_t fork, join, marker;
  CUDA_CHECK(cudaEventCreate(&fork));
  CUDA_CHECK(cudaEventCreate(&join));
  CUDA_CHECK(cudaEventCreateWithFlags(&marker, cudaEventDisableTiming));
  static int hx[N], hy[N], h[N];

  // A: captured; the graph is destroyed once instantiated.
  cudaGraphExec_t exec_a;
  {
    CUDA_CHECK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal));
    CUDA_CHECK(cudaMemcpyAsync(x, hx, sizeof hx, cudaMemcpyHostToDevice, s1));
    CUDA_CHECK(cudaMemsetAsync(y, 0, sizeof hy, s1));
    CUDA_CHECK(cudaEventRecord(fork, s1));
    CUDA_CHECK(cudaStreamWaitEvent(s2, fork, 0));
    add<<<1, N, 0, s2>>>(y, x, 1);
    CUDA_CHECK(cudaEventRecord(join, s2));
    CUDA_CHECK(cudaStreamWaitEvent(s1, join, 0));
    scale<<<1, N, 0, s1>>>(y, 2);
    CUDA_CHECK(cudaLaunchHostFunc(s1, count_a, &host_calls_a));
    CUDA_CHECK(cudaMemcpyAsync(hy, y, sizeof hy, cudaMemcpyDeviceToHost, s1));
    cudaGraph_t g;
    CUDA_CHECK(cudaStreamEndCapture(s1, &g));
    CUDA_CHECK(cudaGraphInstantiate(&exec_a, g, 0));
    CUDA_CHECK(cudaGraphDestroy(g));
  }

  // B: built node by node; the graph and its node handles are kept.
  cudaGraph_t graph_b, child;
  cudaGraphNode_t acc_node, never_node;
  cudaGraphExec_t exec_b;
  {
    CUDA_CHECK(cudaGraphCreate(&graph_b, 0));
    int one = 1;
    void *acc_args[] = { &acc, &one };
    cudaKernelNodeParams kp = {};
    kp.func = (void *)accumulate;
    kp.gridDim = dim3(1);
    kp.blockDim = dim3(N);
    kp.kernelParams = acc_args;
    CUDA_CHECK(cudaGraphAddKernelNode(&acc_node, graph_b, NULL, 0, &kp));
    int thousand = 1000;
    void *never_args[] = { &acc, &thousand };
    kp.kernelParams = never_args;
    CUDA_CHECK(cudaGraphAddKernelNode(&never_node, graph_b, &acc_node, 1,
                                      &kp));
    cudaGraphNode_t record;
    CUDA_CHECK(cudaGraphAddEventRecordNode(&record, graph_b, &never_node, 1,
                                           marker));
    CUDA_CHECK(cudaGraphCreate(&child, 0));
    cudaGraphNode_t bump_node;
    void *bump_args[] = { &counter };
    cudaKernelNodeParams bp = {};
    bp.func = (void *)bump;
    bp.gridDim = dim3(1);
    bp.blockDim = dim3(1);
    bp.kernelParams = bump_args;
    CUDA_CHECK(cudaGraphAddKernelNode(&bump_node, child, NULL, 0, &bp));
    cudaGraphNode_t child_node;
    CUDA_CHECK(cudaGraphAddChildGraphNode(&child_node, graph_b, &record, 1,
                                          child));
    cudaMemcpy3DParms to_array = {};
    to_array.srcPtr = make_cudaPitchedPtr(acc, N * sizeof(int), N, 1);
    to_array.dstArray = array;
    to_array.extent = make_cudaExtent(N, 1, 1);
    to_array.kind = cudaMemcpyDeviceToDevice;
    cudaGraphNode_t copy_in, copy_out;
    CUDA_CHECK(cudaGraphAddMemcpyNode(&copy_in, graph_b, &child_node, 1,
                                      &to_array));
    cudaMemcpy3DParms from_array = {};
    from_array.srcArray = array;
    from_array.dstPtr = make_cudaPitchedPtr(out, N * sizeof(int), N, 1);
    from_array.extent = make_cudaExtent(N, 1, 1);
    from_array.kind = cudaMemcpyDeviceToDevice;
    CUDA_CHECK(cudaGraphAddMemcpyNode(&copy_out, graph_b, &copy_in, 1,
                                      &from_array));
    cudaHostNodeParams hp = { count_b, &host_calls_b };
    cudaGraphNode_t host_node;
    CUDA_CHECK(cudaGraphAddHostNode(&host_node, graph_b, &copy_out, 1, &hp));
    CUDA_CHECK(cudaGraphInstantiate(&exec_b, graph_b, 0));
    CUDA_CHECK(cudaGraphNodeSetEnabled(exec_b, never_node, 0));
  }

  // D: a driver-API kernel, captured.
  CUmodule module;
  CUfunction bump_by;
  DRV_CHECK(cuModuleLoadData(&module, ptx));
  DRV_CHECK(cuModuleGetFunction(&bump_by, module, "bump_by"));
  CUdeviceptr loaded_after;
  size_t loaded_after_size;
  DRV_CHECK(cuModuleGetGlobal(&loaded_after, &loaded_after_size, module,
                              "loaded_after"));
  cudaGraphExec_t exec_d;
  {
    CUDA_CHECK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal));
    unsigned by = 1;
    void *args[] = { &counter2, &by };
    DRV_CHECK(cuLaunchKernel(bump_by, 1, 1, 1, 1, 1, 1, 0, (CUstream)s1, args,
                             NULL));
    cudaGraph_t g;
    CUDA_CHECK(cudaStreamEndCapture(s1, &g));
    CUDA_CHECK(cudaGraphInstantiate(&exec_d, g, 0));
    CUDA_CHECK(cudaGraphDestroy(g));
  }

  // E: captured and kept, with its node's handle.
  cudaGraph_t graph_e;
  cudaGraphNode_t node_e;
  {
    CUDA_CHECK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal));
    add<<<1, N, 0, s1>>>(w, never, 0);
    CUDA_CHECK(cudaStreamEndCapture(s1, &graph_e));
    size_t n = 1;
    CUDA_CHECK(cudaGraphGetNodes(graph_e, &node_e, &n));
    MT_CHECK(n == 1, "graph E has %zu nodes", n);
  }
  {
    CUDA_CHECK(cudaStreamBeginCapture(s2, cudaStreamCaptureModeGlobal));
    cudaStreamCaptureStatus status;
    unsigned long long id = 0;
    cudaGraph_t g = NULL;
    CUDA_CHECK(cudaStreamGetCaptureInfo(s2, &status, &id, &g, NULL, NULL,
                                        NULL));
    MT_CHECK(status == cudaStreamCaptureStatusActive && g != NULL,
             "capture status %d", (int)status);
    add<<<1, N, 0, s2>>>(w, never, 0);
    MT_CHECK(cudaStreamSynchronize(s2) != cudaSuccess,
             "synchronizing a capturing stream");
    cudaGraph_t ended = NULL;
    MT_CHECK(cudaStreamEndCapture(s2, &ended) != cudaSuccess && ended == NULL,
             "ending an invalidated capture");
    cudaGetLastError();
  }
  int ones[N];
  for (int i = 0; i < N; i++) {
    ones[i] = 1;
  }
  CUDA_CHECK(cudaMemcpy(never, ones, sizeof ones, cudaMemcpyHostToDevice));

  cudaGraphExec_t exec_c = NULL;
  long it;
  int expect_acc = 0;
  for (it = 0; mt_continue(it); it++) {
    // A.
    for (int i = 0; i < N; i++) {
      hx[i] = i + (int)it;
    }
    CUDA_CHECK(cudaGraphLaunch(exec_a, s1));
    CUDA_CHECK(cudaStreamSynchronize(s1));
    for (int i = 0; i < N; i++) {
      MT_CHECK(hy[i] == 2 * (i + (int)it + 1), "iteration %ld: A: y[%d] = %d",
               it, i, hy[i]);
    }
    MT_CHECK(__atomic_load_n(&host_calls_a, __ATOMIC_SEQ_CST) == it + 1,
             "iteration %ld: A: %d host calls", it, host_calls_a);

    // B: the value that the last iteration set (1 at first), then the next.
    expect_acc += it == 0 ? 1 : (int)((it - 1) % 5) + 2;
    CUDA_CHECK(cudaGraphLaunch(exec_b, s1));
    CUDA_CHECK(cudaStreamSynchronize(s1));
    CUDA_CHECK(cudaMemcpy(h, out, sizeof h, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
      MT_CHECK(h[i] == expect_acc, "iteration %ld: B: acc[%d] = %d, not %d",
               it, i, h[i], expect_acc);
    }
    int count = 0;
    CUDA_CHECK(cudaMemcpy(&count, counter, sizeof count,
                          cudaMemcpyDeviceToHost));
    MT_CHECK(count == it + 1, "iteration %ld: B: child graph ran %d times",
             it, count);
    MT_CHECK(__atomic_load_n(&host_calls_b, __ATOMIC_SEQ_CST) == it + 1,
             "iteration %ld: B: %d host calls", it, host_calls_b);
    CUDA_CHECK(cudaEventQuery(marker));
    unsigned enabled = 1;
    CUDA_CHECK(cudaGraphNodeGetEnabled(exec_b, never_node, &enabled));
    MT_CHECK(enabled == 0, "iteration %ld: B: node enabled again", it);
    {
      int v = (int)(it % 5) + 2;
      void *args[] = { &acc, &v };
      cudaKernelNodeParams kp = {};
      kp.func = (void *)accumulate;
      kp.gridDim = dim3(1);
      kp.blockDim = dim3(N);
      kp.kernelParams = args;
      CUDA_CHECK(cudaGraphExecKernelNodeSetParams(exec_b, acc_node, &kp));
    }

    // C: the last update's values, then an update for the next iteration.
    if (exec_c != NULL) {
      CUDA_CHECK(cudaGraphLaunch(exec_c, s1));
      CUDA_CHECK(cudaStreamSynchronize(s1));
      CUDA_CHECK(cudaMemcpy(h, z, sizeof h, cudaMemcpyDeviceToHost));
      for (int i = 0; i < N; i++) {
        MT_CHECK(h[i] == (int)(it - 1) * 3 + i, "iteration %ld: C: z[%d] = %d",
                 it, i, h[i]);
      }
    }
    {
      CUDA_CHECK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal));
      set_to<<<1, N, 0, s1>>>(z, (int)it);
      cudaGraph_t g;
      CUDA_CHECK(cudaStreamEndCapture(s1, &g));
      if (exec_c == NULL) {
        CUDA_CHECK(cudaGraphInstantiate(&exec_c, g, 0));
      } else {
        cudaGraphExecUpdateResultInfo info;
        CUDA_CHECK(cudaGraphExecUpdate(exec_c, g, &info));
      }
      CUDA_CHECK(cudaGraphDestroy(g));
    }

    // D.
    CUDA_CHECK(cudaGraphLaunch(exec_d, s1));
    CUDA_CHECK(cudaStreamSynchronize(s1));
    CUDA_CHECK(cudaMemcpy(&count, counter2, sizeof count,
                          cudaMemcpyDeviceToHost));
    MT_CHECK(count == it + 1, "iteration %ld: D: counter %d", it, count);

    // E: a new executable graph each time.
    {
      cudaKernelNodeParams kp;
      CUDA_CHECK(cudaGraphKernelNodeGetParams(node_e, &kp));
      MT_CHECK(kp.blockDim.x == N && kp.func == (void *)add,
               "iteration %ld: E: node parameters", it);
      cudaGraphExec_t exec_e;
      CUDA_CHECK(cudaGraphInstantiate(&exec_e, graph_e, 0));
      CUDA_CHECK(cudaGraphLaunch(exec_e, s1));
      CUDA_CHECK(cudaStreamSynchronize(s1));
      CUDA_CHECK(cudaGraphExecDestroy(exec_e));
      CUDA_CHECK(cudaMemcpy(h, w, sizeof h, cudaMemcpyDeviceToHost));
      for (int i = 0; i < N; i++) {
        MT_CHECK(h[i] == it + 1, "iteration %ld: E: w[%d] = %d", it, i, h[i]);
      }
    }

    // A's result of every rank, summed from device memory.
    MT_MPI(MPI_Allreduce(y, d_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
    int sum = 0;
    CUDA_CHECK(cudaMemcpy(&sum, d_sum, sizeof sum, cudaMemcpyDeviceToHost));
    MT_CHECK(sum == mt_size * 2 * ((int)it + 1), "iteration %ld: MPI sum %d",
             it, sum);
  }

  CUDA_CHECK(cudaGraphExecDestroy(exec_a));
  CUDA_CHECK(cudaGraphExecDestroy(exec_b));
  CUDA_CHECK(cudaGraphExecDestroy(exec_c));
  CUDA_CHECK(cudaGraphExecDestroy(exec_d));
  CUDA_CHECK(cudaGraphDestroy(graph_b));
  CUDA_CHECK(cudaGraphDestroy(child));
  CUDA_CHECK(cudaGraphDestroy(graph_e));
  mt_finish(it);
  return 0;
}
