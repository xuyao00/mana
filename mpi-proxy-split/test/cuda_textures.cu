// CUDA arrays and texture and surface objects (run with MANA's --cuda).  A
// 2D array is a surface that a kernel advances each iteration and a texture
// that another kernel reads; a 3D array is filled through cudaMemcpy3D; a
// mipmapped array is read at level 1 by level of detail; a texture reads
// linear memory.  Objects made before the live ones are destroyed after them,
// so the live values have gaps, and the kernels read the values from device
// memory: they must come back exactly.

#include <cuda_runtime.h>

#include "mana_test.h"

#define W 64
#define H 32
#define D 8
#define LINEAR 1024

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t e_ = (call);                                                 \
    MT_CHECK(e_ == cudaSuccess, "%s: %s", #call, cudaGetErrorString(e_));    \
  } while (0)

enum { SURF_2D, TEX_2D, TEX_3D, TEX_MIP, TEX_LINEAR, N_OBJECTS };

__host__ __device__ float
init2d(int x, int y)
{
  return (float)(x + W * y);
}

__host__ __device__ float
init3d(int x, int y, int z)
{
  return (float)(100000 + x + W * (y + H * z));
}

__host__ __device__ float
level1(int x, int y)
{
  return (float)(50000 + x + (W / 2) * y);
}

__global__ void
advance(const unsigned long long *objects)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y;
  if (x < W) {
    float v = surf2Dread<float>(objects[SURF_2D], x * sizeof(float), y);
    surf2Dwrite(v + 1.0f, objects[SURF_2D], x * sizeof(float), y);
  }
}

__global__ void
check(const unsigned long long *objects, long it, int *bad)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y;
  if (x >= W) {
    return;
  }
  if (tex2D<float>(objects[TEX_2D], x + 0.5f, y + 0.5f) !=
      init2d(x, y) + it + 1) {
    atomicAdd(&bad[TEX_2D], 1);
  }
  for (int z = 0; z < D; z++) {
    if (tex3D<float>(objects[TEX_3D], x + 0.5f, y + 0.5f, z + 0.5f) !=
        init3d(x, y, z)) {
      atomicAdd(&bad[TEX_3D], 1);
    }
  }
  if (x < W / 2 && y < H / 2) {
    float u = (x + 0.5f) / (W / 2), v = (y + 0.5f) / (H / 2);
    if (tex2DLod<float>(objects[TEX_MIP], u, v, 1.0f) != level1(x, y)) {
      atomicAdd(&bad[TEX_MIP], 1);
    }
  }
  int i = x + W * y;
  if (i < LINEAR && tex1Dfetch<float>(objects[TEX_LINEAR], i) != 2.0f * i) {
    atomicAdd(&bad[TEX_LINEAR], 1);
  }
}

static cudaTextureObject_t
texture(const cudaResourceDesc &res, bool normalized)
{
  cudaTextureDesc t = {};
  t.addressMode[0] = t.addressMode[1] = t.addressMode[2] = cudaAddressModeClamp;
  t.filterMode = cudaFilterModePoint;
  t.mipmapFilterMode = cudaFilterModePoint;
  t.readMode = cudaReadModeElementType;
  t.normalizedCoords = normalized;
  t.maxMipmapLevelClamp = 2;
  cudaTextureObject_t obj = 0;
  CUDA_CHECK(cudaCreateTextureObject(&obj, &res, &t, NULL));
  return obj;
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "cuda_textures");
  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  CUDA_CHECK(cudaSetDevice(mt_rank % ndev));
  cudaChannelFormatDesc f = cudaCreateChannelDesc<float>();

  // Linear memory, and objects that go before the first iteration.
  float *linear;
  CUDA_CHECK(cudaMalloc(&linear, LINEAR * sizeof(float)));
  float host_linear[LINEAR];
  for (int i = 0; i < LINEAR; i++) {
    host_linear[i] = 2.0f * i;
  }
  CUDA_CHECK(cudaMemcpy(linear, host_linear, sizeof host_linear,
                        cudaMemcpyHostToDevice));
  cudaResourceDesc lres = {};
  lres.resType = cudaResourceTypeLinear;
  lres.res.linear.devPtr = linear;
  lres.res.linear.desc = f;
  lres.res.linear.sizeInBytes = LINEAR * sizeof(float);
  cudaTextureObject_t gone[3];
  for (int i = 0; i < 3; i++) {
    gone[i] = texture(lres, false);
  }

  unsigned long long objects[N_OBJECTS];

  // The 2D array: a surface and a texture.
  cudaArray_t a2;
  CUDA_CHECK(cudaMallocArray(&a2, &f, W, H, cudaArraySurfaceLoadStore));
  static float host2d[H][W];
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      host2d[y][x] = init2d(x, y);
    }
  }
  CUDA_CHECK(cudaMemcpy2DToArray(a2, 0, 0, host2d, W * sizeof(float),
                                 W * sizeof(float), H,
                                 cudaMemcpyHostToDevice));
  cudaResourceDesc r2 = {};
  r2.resType = cudaResourceTypeArray;
  r2.res.array.array = a2;
  CUDA_CHECK(cudaCreateSurfaceObject(&objects[SURF_2D], &r2));
  objects[TEX_2D] = texture(r2, false);
  cudaResourceDesc back;
  CUDA_CHECK(cudaGetTextureObjectResourceDesc(&back, objects[TEX_2D]));
  MT_CHECK(back.res.array.array == a2, "%s", "the texture's array");

  // The 3D array, through cudaMemcpy3D.
  cudaArray_t a3;
  CUDA_CHECK(cudaMalloc3DArray(&a3, &f, make_cudaExtent(W, H, D), 0));
  static float host3d[D][H][W];
  for (int z = 0; z < D; z++) {
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        host3d[z][y][x] = init3d(x, y, z);
      }
    }
  }
  cudaMemcpy3DParms p = {};
  p.srcPtr = make_cudaPitchedPtr(host3d, W * sizeof(float), W, H);
  p.dstArray = a3;
  p.extent = make_cudaExtent(W, H, D);
  p.kind = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpy3D(&p));
  cudaResourceDesc r3 = {};
  r3.resType = cudaResourceTypeArray;
  r3.res.array.array = a3;
  objects[TEX_3D] = texture(r3, false);

  // The mipmapped array: level 1, read by level of detail.
  cudaMipmappedArray_t mip;
  CUDA_CHECK(cudaMallocMipmappedArray(&mip, &f, make_cudaExtent(W, H, 0), 3,
                                      0));
  cudaArray_t level;
  CUDA_CHECK(cudaGetMipmappedArrayLevel(&level, mip, 1));
  static float host_level[H / 2][W / 2];
  for (int y = 0; y < H / 2; y++) {
    for (int x = 0; x < W / 2; x++) {
      host_level[y][x] = level1(x, y);
    }
  }
  CUDA_CHECK(cudaMemcpy2DToArray(level, 0, 0, host_level,
                                 (W / 2) * sizeof(float),
                                 (W / 2) * sizeof(float), H / 2,
                                 cudaMemcpyHostToDevice));
  cudaResourceDesc rm = {};
  rm.resType = cudaResourceTypeMipmappedArray;
  rm.res.mipmap.mipmap = mip;
  objects[TEX_MIP] = texture(rm, true);
  objects[TEX_LINEAR] = texture(lres, false);

  CUDA_CHECK(cudaDestroyTextureObject(gone[0]));
  CUDA_CHECK(cudaDestroyTextureObject(gone[2]));

  unsigned long long *d_objects;
  int *d_bad;
  CUDA_CHECK(cudaMalloc(&d_objects, sizeof objects));
  CUDA_CHECK(cudaMalloc(&d_bad, N_OBJECTS * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_objects, objects, sizeof objects,
                        cudaMemcpyHostToDevice));

  dim3 grid((W + 31) / 32, H);
  long it;
  for (it = 0; mt_continue(it); it++) {
    advance<<<grid, 32>>>(d_objects);
    CUDA_CHECK(cudaMemset(d_bad, 0, N_OBJECTS * sizeof(int)));
    check<<<grid, 32>>>(d_objects, it, d_bad);
    int bad[N_OBJECTS];
    CUDA_CHECK(cudaMemcpy(bad, d_bad, sizeof bad, cudaMemcpyDeviceToHost));
    for (int i = TEX_2D; i < N_OBJECTS; i++) {
      MT_CHECK(bad[i] == 0, "iteration %ld: object %d: %d bad values", it, i,
               bad[i]);
    }

    // The arrays from the host.
    float row[W];
    CUDA_CHECK(cudaMemcpy2DFromArray(row, sizeof row, a2, 0, H - 1,
                                     sizeof row, 1, cudaMemcpyDeviceToHost));
    MT_CHECK(row[W - 1] == init2d(W - 1, H - 1) + it + 1,
             "iteration %ld: 2D array %f", it, row[W - 1]);
    float last = 0;
    cudaMemcpy3DParms q = {};
    q.srcArray = a3;
    q.srcPos = make_cudaPos(W - 1, H - 1, D - 1);
    q.dstPtr = make_cudaPitchedPtr(&last, sizeof last, 1, 1);
    q.extent = make_cudaExtent(1, 1, 1);
    q.kind = cudaMemcpyDeviceToHost;
    CUDA_CHECK(cudaMemcpy3D(&q));
    MT_CHECK(last == init3d(W - 1, H - 1, D - 1), "iteration %ld: 3D %f", it,
             last);

    long sum = it, total = 0;
    MT_MPI(MPI_Allreduce(&sum, &total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD));
    MT_CHECK(total == it * mt_size, "iteration %ld: MPI_Allreduce", it);
  }
  CUDA_CHECK(cudaDestroyTextureObject(gone[1]));
  mt_finish(it);
  return 0;
}
