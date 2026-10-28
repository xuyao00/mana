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

// A 2D float array: its surface object advances it, its texture object is
// checked against base + x + w * y.
extern "C" __global__ void
bump_surface(unsigned long long surface, int w)
{
  int x = threadIdx.x, y = blockIdx.x;
  if (x < w) {
    float v = surf2Dread<float>(surface, x * sizeof(float), y);
    surf2Dwrite(v + 1.0f, surface, x * sizeof(float), y);
  }
}

extern "C" __global__ void
check_texture(unsigned long long texture, int w, float base, int *bad)
{
  int x = threadIdx.x, y = blockIdx.x;
  if (x < w && tex2D<float>(texture, x + 0.5f, y + 0.5f) != base + x + w * y) {
    atomicAdd(bad, 1);
  }
}
