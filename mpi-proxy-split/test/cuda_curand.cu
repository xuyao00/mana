// cuRAND (run with MANA's --cuda).  Each generator has a twin that makes
// the same calls one iteration later.  After a restart, the twin's next
// numbers must be those that its partner made before the checkpoint, so
// each generator must go on as if it had not stopped.  The generators cover
// the ways in which the front brings one back: at an offset (XORWOW
// uniforms, Philox, MRG32k3a uniform doubles, Sobol), by making its numbers
// again (XORWOW normals, MRG32k3a normal doubles, MTGP32, MT19937, a host
// generator), and with setters after generating (Philox and XORWOW seeded
// again, curandGenerateSeeds, an offset set again, an ordering).
//
// The device API: Philox states in device memory that draw from a Poisson
// distribution of curandCreatePoissonDistribution, and Sobol states made
// from direction vectors, with twins too; the pointers to the direction
// vectors and scramble constants must still read the same.  A sum goes to
// MPI from device memory.

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <string.h>

#include "mana_test.h"

#define STATES 256

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t e_ = (call);                                                 \
    MT_CHECK(e_ == cudaSuccess, "%s: %s", #call, cudaGetErrorString(e_));    \
  } while (0)

#define RAND_CHECK(call)                                                     \
  do {                                                                       \
    curandStatus_t s_ = (call);                                              \
    MT_CHECK(s_ == CURAND_STATUS_SUCCESS, "%s: status %d", #call, (int)s_);  \
  } while (0)

enum Fn { U32, UNIFORM, UNIFORM_DOUBLE, NORMAL, NORMAL_DOUBLE, LOG_NORMAL,
          POISSON };

struct Twins {
  const char *name;
  curandRngType_t type;
  int host;
  enum Fn fn;
  size_t n;
  double a, b;
  void (*setup)(curandGenerator_t g);
  void (*before)(curandGenerator_t g, long k);  // before call k
  curandGenerator_t gen[2];  // gen[1] makes call k at iteration k + 1
  void *out;
  unsigned char *last;       // what gen[0] made at the last iteration
  unsigned char *got;
  size_t bytes;
};

static cudaStream_t stream;

static void
seed(curandGenerator_t g, unsigned long long s)
{
  RAND_CHECK(curandSetPseudoRandomGeneratorSeed(g, s + 100 * mt_rank));
}

// A stream made after the generator.
static void
xorwow_uniform(curandGenerator_t g)
{
  seed(g, 11);
  if (stream == NULL) {
    CUDA_CHECK(cudaStreamCreate(&stream));
  }
  RAND_CHECK(curandSetStream(g, stream));
}

static void xorwow_normal(curandGenerator_t g) { seed(g, 12); }
static void philox(curandGenerator_t g) { seed(g, 13); }
static void mtgp32(curandGenerator_t g) { seed(g, 16); }
static void mt19937(curandGenerator_t g) { seed(g, 17); }
static void host_xorwow(curandGenerator_t g) { seed(g, 18); }

static void
mrg_legacy(curandGenerator_t g)
{
  seed(g, 15);
  RAND_CHECK(curandSetGeneratorOrdering(g, CURAND_ORDERING_PSEUDO_LEGACY));
}

static void
sobol64(curandGenerator_t g)
{
  RAND_CHECK(curandSetQuasiRandomGeneratorDimensions(g, 3));
  RAND_CHECK(curandSetGeneratorOffset(g, 7));
}

static void
sobol32(curandGenerator_t g)
{
  RAND_CHECK(curandSetQuasiRandomGeneratorDimensions(g, 2));
}

static void
host_sobol(curandGenerator_t g)
{
  RAND_CHECK(curandSetQuasiRandomGeneratorDimensions(g, 4));
}

static void
offset_again(curandGenerator_t g, long k)
{
  if (k % 50 == 49) {
    RAND_CHECK(curandSetGeneratorOffset(g, k * 7));
  }
}

// Setters on Philox after it has generated since an offset was set: a
// restart in between puts it at an offset, and a setter must still find it
// as the uninterrupted one would be.
static void
philox_setters(curandGenerator_t g, long k)
{
  if (k % 16 == 0) {
    RAND_CHECK(curandSetGeneratorOffset(g, k * 5));
  } else if (k % 16 == 8) {
    RAND_CHECK(curandGenerateSeeds(g));
  } else if (k % 16 == 12) {
    seed(g, 1000 + k);
  }
}

static void
seed_again(curandGenerator_t g, long k)
{
  if (k % 32 == 31) {
    seed(g, 500 + k);
  }
}

static void
ordering_again(curandGenerator_t g, long k)
{
  if (k == 5) {
    RAND_CHECK(curandSetGeneratorOrdering(g, CURAND_ORDERING_PSEUDO_DEFAULT));
  }
}

static struct Twins twins[] = {
  { "XORWOW uniform", CURAND_RNG_PSEUDO_XORWOW, 0, UNIFORM, 1000, 0, 0,
    xorwow_uniform, seed_again },
  { "XORWOW normal", CURAND_RNG_PSEUDO_XORWOW, 0, NORMAL, 1002, 0, 1,
    xorwow_normal, offset_again },
  { "Philox", CURAND_RNG_PSEUDO_PHILOX4_32_10, 0, UNIFORM, 4097, 0, 0,
    philox, philox_setters },
  { "Philox Poisson", CURAND_RNG_PSEUDO_PHILOX4_32_10, 0, POISSON, 1000,
    2000, 0, philox, NULL },
  { "MRG32k3a normal double", CURAND_RNG_PSEUDO_MRG32K3A, 0, NORMAL_DOUBLE,
    2000, 1, 2, mrg_legacy, ordering_again },
  { "MRG32k3a uniform double", CURAND_RNG_PSEUDO_MRG32K3A, 0,
    UNIFORM_DOUBLE, 999, 0, 0, mrg_legacy, NULL },
  { "MTGP32", CURAND_RNG_PSEUDO_MTGP32, 0, UNIFORM, 4096, 0, 0, mtgp32,
    NULL },
  { "MT19937", CURAND_RNG_PSEUDO_MT19937, 0, POISSON, 1002, 3.5, 0, mt19937,
    NULL },
  { "scrambled Sobol64", CURAND_RNG_QUASI_SCRAMBLED_SOBOL64, 0,
    UNIFORM_DOUBLE, 3 * 500, 0, 0, sobol64, NULL },
  { "Sobol32", CURAND_RNG_QUASI_SOBOL32, 0, LOG_NORMAL, 2 * 512, 0, 0.5,
    sobol32, NULL },
  { "host XORWOW", CURAND_RNG_PSEUDO_XORWOW, 1, NORMAL, 1000, 0, 1,
    host_xorwow, NULL },
  { "host Sobol32", CURAND_RNG_QUASI_SOBOL32, 1, U32, 4 * 256, 0, 0,
    host_sobol, NULL },
};
#define NTWINS (int)(sizeof twins / sizeof twins[0])

static size_t
result_size(enum Fn fn)
{
  return fn == UNIFORM_DOUBLE || fn == NORMAL_DOUBLE ? 8 : 4;
}

// Call k of t on g, into t->out, and its numbers into 'into'.
static void
call(struct Twins *t, curandGenerator_t g, long k, unsigned char *into)
{
  if (t->before != NULL) {
    t->before(g, k);
  }
  void *p = t->out;
  switch (t->fn) {
    case U32:
      RAND_CHECK(curandGenerate(g, (unsigned *)p, t->n));
      break;
    case UNIFORM:
      RAND_CHECK(curandGenerateUniform(g, (float *)p, t->n));
      break;
    case UNIFORM_DOUBLE:
      RAND_CHECK(curandGenerateUniformDouble(g, (double *)p, t->n));
      break;
    case NORMAL:
      RAND_CHECK(curandGenerateNormal(g, (float *)p, t->n, t->a, t->b));
      break;
    case NORMAL_DOUBLE:
      RAND_CHECK(curandGenerateNormalDouble(g, (double *)p, t->n, t->a,
                                            t->b));
      break;
    case LOG_NORMAL:
      RAND_CHECK(curandGenerateLogNormal(g, (float *)p, t->n, t->a, t->b));
      break;
    case POISSON:
      RAND_CHECK(curandGeneratePoisson(g, (unsigned *)p, t->n, t->a));
      break;
  }
  if (t->host) {
    memcpy(into, p, t->bytes);
  } else {
    CUDA_CHECK(cudaMemcpy(into, p, t->bytes, cudaMemcpyDeviceToHost));
  }
}

__global__ void
init_states(curandStatePhilox4_32_10_t *philox, curandStateSobol32_t *sobol,
            const curandDirectionVectors32_t *vectors, unsigned long long s)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(s, i, 0, &philox[i]);
  curand_init((unsigned *)vectors[i % 16], i, &sobol[i]);
}

__global__ void
draw(curandStatePhilox4_32_10_t *philox, curandStateSobol32_t *sobol,
     curandDiscreteDistribution_t poisson, unsigned *out)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned k = curand_discrete(&philox[i], poisson);
  out[i] = (k << 16) + (curand(&philox[i]) & 0xffff);
  out[STATES + i] = curand(&sobol[i]);
}

static unsigned long long
checksum(const void *p, size_t size)
{
  const unsigned char *c = (const unsigned char *)p;
  unsigned long long h = 1469598103934665603ULL;
  for (size_t i = 0; i < size; i++) {
    h = (h ^ c[i]) * 1099511628211ULL;
  }
  return h;
}

int
main(int argc, char **argv)
{
  mt_init(&argc, &argv, "cuda_curand");
  int ndev = 0;
  CUDA_CHECK(cudaGetDeviceCount(&ndev));
  CUDA_CHECK(cudaSetDevice(mt_rank % ndev));

  for (int i = 0; i < NTWINS; i++) {
    struct Twins *t = &twins[i];
    t->bytes = t->n * result_size(t->fn);
    for (int j = 0; j < 2; j++) {
      if (t->host) {
        RAND_CHECK(curandCreateGeneratorHost(&t->gen[j], t->type));
      } else {
        RAND_CHECK(curandCreateGenerator(&t->gen[j], t->type));
      }
      t->setup(t->gen[j]);
    }
    if (t->host) {
      t->out = malloc(t->bytes);
    } else {
      CUDA_CHECK(cudaMalloc(&t->out, t->bytes));
    }
    t->last = (unsigned char *)malloc(t->bytes);
    t->got = (unsigned char *)malloc(t->bytes);
  }

  // The device API.
  curandDirectionVectors32_t *vectors;
  unsigned *scramble;
  RAND_CHECK(curandGetDirectionVectors32(&vectors,
                                         CURAND_DIRECTION_VECTORS_32_JOEKUO6));
  RAND_CHECK(curandGetScrambleConstants32(&scramble));
  unsigned long long vectors_sum = checksum(vectors, 1000 * sizeof *vectors);
  unsigned long long scramble_sum =
    checksum(scramble, 20000 * sizeof *scramble);
  curandDirectionVectors32_t *d_vectors;
  CUDA_CHECK(cudaMalloc(&d_vectors, 16 * sizeof *d_vectors));
  CUDA_CHECK(cudaMemcpy(d_vectors, vectors, 16 * sizeof *d_vectors,
                        cudaMemcpyHostToDevice));
  curandDiscreteDistribution_t poisson;
  RAND_CHECK(curandCreatePoissonDistribution(4.0, &poisson));
  curandStatePhilox4_32_10_t *philox[2];
  curandStateSobol32_t *sobol[2];
  unsigned *d_out, *d_sum;
  for (int j = 0; j < 2; j++) {
    CUDA_CHECK(cudaMalloc(&philox[j], STATES * sizeof *philox[j]));
    CUDA_CHECK(cudaMalloc(&sobol[j], STATES * sizeof *sobol[j]));
    init_states<<<STATES / 64, 64>>>(philox[j], sobol[j], d_vectors,
                                     1234 + mt_rank);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaFree(d_vectors));
  CUDA_CHECK(cudaMalloc(&d_out, 2 * STATES * sizeof *d_out));
  CUDA_CHECK(cudaMalloc(&d_sum, STATES * sizeof *d_sum));
  static unsigned last[2 * STATES], got[2 * STATES], sum[STATES];

  long it;
  for (it = 0; mt_continue(it); it++) {
    for (int i = 0; i < NTWINS; i++) {
      struct Twins *t = &twins[i];
      if (it > 0) {
        call(t, t->gen[1], it - 1, t->got);
        MT_CHECK(memcmp(t->got, t->last, t->bytes) == 0,
                 "iteration %ld: %s: call %ld made other numbers the second "
                 "time", it, t->name, it - 1);
      }
      call(t, t->gen[0], it, t->last);
    }

    if (it > 0) {
      draw<<<STATES / 64, 64>>>(philox[1], sobol[1], poisson, d_out);
      CUDA_CHECK(cudaMemcpy(got, d_out, sizeof got, cudaMemcpyDeviceToHost));
      MT_CHECK(memcmp(got, last, sizeof got) == 0,
               "iteration %ld: the device API's draw %ld differs", it,
               it - 1);
    }
    draw<<<STATES / 64, 64>>>(philox[0], sobol[0], poisson, d_out);
    CUDA_CHECK(cudaMemcpy(last, d_out, sizeof last, cudaMemcpyDeviceToHost));
    unsigned mean = 0;
    for (int i = 0; i < STATES; i++) {
      mean += last[i] >> 16;
    }
    MT_CHECK(mean > 2 * STATES && mean < 8 * STATES,
             "iteration %ld: Poisson draws of mean %f, not 4", it,
             (double)mean / STATES);

    // The draws of all ranks, summed from device memory.
    MT_MPI(MPI_Allreduce(d_out, d_sum, STATES, MPI_UNSIGNED, MPI_SUM,
                         MPI_COMM_WORLD));
    MT_MPI(MPI_Allreduce(MPI_IN_PLACE, last, STATES, MPI_UNSIGNED, MPI_SUM,
                         MPI_COMM_WORLD));
    CUDA_CHECK(cudaMemcpy(sum, d_sum, sizeof sum, cudaMemcpyDeviceToHost));
    MT_CHECK(memcmp(sum, last, sizeof sum) == 0,
             "iteration %ld: MPI sum from device memory", it);
    CUDA_CHECK(cudaMemcpy(last, d_out, sizeof last, cudaMemcpyDeviceToHost));

    MT_CHECK(checksum(vectors, 1000 * sizeof *vectors) == vectors_sum,
             "iteration %ld: the direction vectors changed", it);
    MT_CHECK(checksum(scramble, 20000 * sizeof *scramble) == scramble_sum,
             "iteration %ld: the scramble constants changed", it);
  }

  RAND_CHECK(curandDestroyDistribution(poisson));
  for (int i = 0; i < NTWINS; i++) {
    for (int j = 0; j < 2; j++) {
      RAND_CHECK(curandDestroyGenerator(twins[i].gen[j]));
    }
  }
  mt_finish(it);
  return 0;
}
