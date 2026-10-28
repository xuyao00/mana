// Kernels of cuda_driver, which loads them as PTX and as a fatbin.

__device__ int counter;

extern "C" __global__ void
add_one(int *a, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += 1;
}

// out[0] = ++counter: state that only the module global holds.
extern "C" __global__ void
bump(int *out)
{
  out[0] = atomicAdd(&counter, 1) + 1;
}

// a[i] = a[i] * 2 - a[i], through dynamic shared memory of n ints (above
// 48 KB, so the launch needs CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES).
extern "C" __global__ void
through_shared(int *a, int n)
{
  extern __shared__ int s[];
  for (int i = threadIdx.x; i < n; i += blockDim.x) s[i] = a[i] * 2;
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) a[i] = s[i] - a[i];
}
