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

// cuRAND (cuda/DESIGN-lower-half-cuda.md, Phase 3), in the lower half with
// its own static cudart.  Every function with a generator is the front's: a
// generator's state is the library's, in the lower half, so the record must
// say where in its sequence the generator is.
//
// Probes on cuRAND 10.4 (CUDA 13.1), device and host generators alike:
//  * after some generations a generator is exactly where
//    curandSetGeneratorOffset would put it, one offset per result (per
//    point for Sobol): Sobol always; Philox, which is counter-based, for
//    32-bit results; XORWOW and MRG32k3a, whose threads each walk their own
//    subsequence, for one 32-bit value per result (MRG32k3a: doubles too).
//    Not under the dynamic ordering, and never MTGP32 and MT19937, which
//    have no offsets;
//  * a seed, ordering or dimensions set after such generations keep the
//    position, as after an offset; except for Philox, which then goes to
//    about a quarter of it, by the history of its calls (a setter after an
//    offset keeps the offset);
//  * curandSetGeneratorOffset puts any generator at the offset.
// So the record is the settings and offset that a new generator is given,
// plus the generations since that are at an offset, plus a log of the
// calls since the first one that is not (Philox: since the generations
// before a setter).  A restart makes the generator again at its place in
// the replay, with the settings and offset, and once all records exist
// repeats the log into scratch memory.  A Philox generator that came back
// at an offset is made again by its generations (remake()) before a setter
// or a generation that is not at an offset.  A generator that only
// generates at offsets costs nothing; one that is replayed costs GPU time
// for the numbers it made (XORWOW makes 46 G normals/s on an RTX 3070
// Laptop).  The same calls do not always give the same numbers natively
// either: some MT19937 sequences later in a process, Philox under the
// dynamic ordering, and MTGP32 generators that share state once one's
// ordering went from dynamic to another after generating.
//
// The front also copies what the library returns in its own memory, which a
// restart would move: a Poisson distribution for the device API goes into
// device memory of the application's (the VMM range), and the direction
// vectors and scramble constants of Sobol into upper-half memory.

#include <curand.h>
#include <curand_discrete.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include "front_objs.h"
#include "gen/curand-index.h"

extern FrontLib front_curand_lib FRONT_HIDDEN;

#define RAND(f) ((decltype(&::f))front_curand_lib.real[FRONT_IDX_curand_##f])

#define RAND_LH(stmt)                     \
  do {                                    \
    front_check_bound(&front_curand_lib); \
    FRONT_LH(stmt);                       \
  } while (0)

namespace {

enum Fn {
  U32,
  U64,
  UNIFORM,
  UNIFORM_DOUBLE,
  NORMAL,
  NORMAL_DOUBLE,
  LOG_NORMAL,
  LOG_NORMAL_DOUBLE,
  POISSON,
  POISSON_METHOD
};

size_t
result_size(Fn fn)
{
  return fn == U64 || fn == UNIFORM_DOUBLE || fn == NORMAL_DOUBLE ||
         fn == LOG_NORMAL_DOUBLE ? 8 : 4;
}

// A call of the log.
struct Op {
  enum Kind { SEED, ORDERING, DIMENSIONS, SEEDS, GENERATE } kind;
  unsigned long long value;   // SEED, ORDERING, DIMENSIONS
  Fn fn;                      // GENERATE
  size_t n;
  double a, b;                // mean and stddev, or lambda
  int method;                 // POISSON_METHOD
  size_t times;               // the same generation, repeated
};

struct Rand : FrontObj {
  Ctx *ctx;              // NULL: a host generator
  bool gone;             // its context was destroyed
  curandRngType_t type;  // not a DEFAULT one
  // What a new generator is set to: the settings, and the offset plus
  // 'advance'.
  bool has_seed;
  unsigned long long seed;
  bool has_ordering;
  curandOrdering_t ordering;
  bool has_dimensions;
  unsigned dimensions;
  bool has_offset;
  unsigned long long offset;
  unsigned long long advance;  // generated at offsets since then
  std::vector<Op> since;       // Philox: those generations
  std::vector<Op> log;         // then, in order
  // Philox, after a restart: at the offset of 'since', but not made by it
  // (remake()).
  bool at_offset_only;
  bool has_stream;
  cudaStream_t stream;         // virtual
};

bool
quasi(curandRngType_t type)
{
  return type >= CURAND_RNG_QUASI_DEFAULT;
}

// Whether a generation leaves the generator at an offset (see the top).
bool
at_offset(const Rand *r, Fn fn)
{
  if (r->has_ordering && r->ordering == CURAND_ORDERING_PSEUDO_DYNAMIC) {
    return false;
  }
  switch (r->type) {
    case CURAND_RNG_QUASI_SOBOL32:
    case CURAND_RNG_QUASI_SCRAMBLED_SOBOL32:
    case CURAND_RNG_QUASI_SOBOL64:
    case CURAND_RNG_QUASI_SCRAMBLED_SOBOL64:
      return true;
    case CURAND_RNG_PSEUDO_PHILOX4_32_10:
      return fn == U32 || fn == UNIFORM || fn == NORMAL || fn == LOG_NORMAL ||
             fn == POISSON;
    case CURAND_RNG_PSEUDO_XORWOW:
      return fn == U32 || fn == UNIFORM;
    case CURAND_RNG_PSEUDO_MRG32K3A:
      return fn == U32 || fn == UNIFORM || fn == UNIFORM_DOUBLE;
    default:
      return false;
  }
}

void
append(std::vector<Op> *ops, Op op)
{
  if (op.kind == Op::GENERATE && !ops->empty()) {
    Op &last = ops->back();
    if (last.kind == Op::GENERATE && last.fn == op.fn && last.n == op.n &&
        last.a == op.a && last.b == op.b && last.method == op.method) {
      last.times++;
      return;
    }
  }
  op.times = 1;
  ops->push_back(op);
}

void
apply_setting(Rand *r, const Op &op)
{
  switch (op.kind) {
    case Op::SEED:
      r->has_seed = true;
      r->seed = op.value;
      break;
    case Op::ORDERING:
      r->has_ordering = true;
      r->ordering = (curandOrdering_t)op.value;
      break;
    case Op::DIMENSIONS:
      r->has_dimensions = true;
      r->dimensions = (unsigned)op.value;
      break;
    default:
      break;
  }
}

// The note_*() functions record a call that succeeded; with objs_lock held.
void
note_setter(Rand *r, Op op)
{
  bool philox = r->type == CURAND_RNG_PSEUDO_PHILOX4_32_10;
  if (r->log.empty() && (r->advance == 0 || !philox)) {
    apply_setting(r, op);
    return;
  }
  if (r->log.empty()) {
    // Philox: its generations come back as they were made.
    r->log.swap(r->since);
    r->advance = 0;
  }
  append(&r->log, op);
}

void
note_offset(Rand *r, unsigned long long offset)
{
  for (const Op &op : r->log) {
    apply_setting(r, op);
  }
  r->log.clear();
  r->since.clear();
  r->advance = 0;
  r->has_offset = true;
  r->offset = offset;
  r->at_offset_only = false;
}

void
note_generate(Rand *r, Op op)
{
  bool philox = r->type == CURAND_RNG_PSEUDO_PHILOX4_32_10;
  if (!r->log.empty()) {
    append(&r->log, op);
    return;
  }
  if (at_offset(r, op.fn)) {
    unsigned dimensions =
      quasi(r->type) && r->has_dimensions ? r->dimensions : 1;
    r->advance += op.n / dimensions;
    if (philox) {
      append(&r->since, op);
    }
    return;
  }
  if (philox) {
    r->log.swap(r->since);
    r->advance = 0;
  } else if (r->advance > 0) {
    r->has_offset = true;
    r->offset += r->advance;
    r->advance = 0;
  }
  append(&r->log, op);
}

Rand *
rand_of(curandGenerator_t generator)
{
  front_real_obj(generator, FRONT_RAND);
  return (Rand *)generator;
}

curandGenerator_t
real_of(curandGenerator_t generator)
{
  return (curandGenerator_t)front_real_obj(generator, FRONT_RAND);
}

// The call that 'op' records, on 'real'; inside RAND_LH().
curandStatus_t
call_op(curandGenerator_t real, const Op &op, void *out)
{
  switch (op.kind) {
    case Op::SEED:
      return RAND(curandSetPseudoRandomGeneratorSeed)(real, op.value);
    case Op::ORDERING:
      return RAND(curandSetGeneratorOrdering)(real,
                                              (curandOrdering_t)op.value);
    case Op::DIMENSIONS:
      return RAND(curandSetQuasiRandomGeneratorDimensions)(
        real, (unsigned)op.value);
    case Op::SEEDS:
      return RAND(curandGenerateSeeds)(real);
    case Op::GENERATE:
      break;
  }
  switch (op.fn) {
    case U32:
      return RAND(curandGenerate)(real, (unsigned *)out, op.n);
    case U64:
      return RAND(curandGenerateLongLong)(real, (unsigned long long *)out,
                                          op.n);
    case UNIFORM:
      return RAND(curandGenerateUniform)(real, (float *)out, op.n);
    case UNIFORM_DOUBLE:
      return RAND(curandGenerateUniformDouble)(real, (double *)out, op.n);
    case NORMAL:
      return RAND(curandGenerateNormal)(real, (float *)out, op.n,
                                        (float)op.a, (float)op.b);
    case NORMAL_DOUBLE:
      return RAND(curandGenerateNormalDouble)(real, (double *)out, op.n,
                                              op.a, op.b);
    case LOG_NORMAL:
      return RAND(curandGenerateLogNormal)(real, (float *)out, op.n,
                                           (float)op.a, (float)op.b);
    case LOG_NORMAL_DOUBLE:
      return RAND(curandGenerateLogNormalDouble)(real, (double *)out, op.n,
                                                 op.a, op.b);
    case POISSON:
      return RAND(curandGeneratePoisson)(real, (unsigned *)out, op.n, op.a);
    case POISSON_METHOD:
      break;
  }
  return RAND(curandGeneratePoissonMethod)(real, (unsigned *)out, op.n, op.a,
                                           (curandMethod_t)op.method);
}

// A new generator for 'r' in the current context, with its settings and
// offset, plus 'advance' if 'advanced'.
curandStatus_t
make_generator(Rand *r, bool advanced, curandGenerator_t *out)
{
  curandGenerator_t real = NULL;
  curandStatus_t rc;
  if (r->ctx != NULL) {
    RAND_LH(rc = RAND(curandCreateGenerator)(&real, r->type));
  } else {
    RAND_LH(rc = RAND(curandCreateGeneratorHost)(&real, r->type));
  }
  if (rc == CURAND_STATUS_SUCCESS && r->has_seed) {
    RAND_LH(rc = RAND(curandSetPseudoRandomGeneratorSeed)(real, r->seed));
  }
  if (rc == CURAND_STATUS_SUCCESS && r->has_dimensions) {
    RAND_LH(rc = RAND(curandSetQuasiRandomGeneratorDimensions)(
              real, r->dimensions));
  }
  if (rc == CURAND_STATUS_SUCCESS && r->has_ordering) {
    RAND_LH(rc = RAND(curandSetGeneratorOrdering)(real, r->ordering));
  }
  unsigned long long offset = r->offset + (advanced ? r->advance : 0);
  if (rc == CURAND_STATUS_SUCCESS && (r->has_offset || offset > 0)) {
    RAND_LH(rc = RAND(curandSetGeneratorOffset)(real, offset));
  }
  if (rc != CURAND_STATUS_SUCCESS && real != NULL) {
    curandStatus_t destroyed;
    RAND_LH(destroyed = RAND(curandDestroyGenerator)(real));
    (void)destroyed;
    real = NULL;
  }
  *out = real;
  return rc;
}

// 'ops' on 'real', whose numbers go into scratch memory of the lower half's.
void
run_ops(const Rand *r, curandGenerator_t real, const std::vector<Op> &ops)
{
  size_t bytes = 0, numbers = 0;
  for (const Op &op : ops) {
    if (op.kind == Op::GENERATE) {
      bytes = std::max(bytes, op.n * result_size(op.fn));
      numbers += op.n * op.times;
    }
  }
  void *scratch = NULL;
  CUresult err = CUDA_SUCCESS;
  if (r->ctx == NULL) {
    scratch = malloc(std::max(bytes, (size_t)1));
    err = scratch != NULL ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
  } else if (bytes > 0) {
    FRONT_LH(err = DRV(cuMemAlloc_v2)((CUdeviceptr *)&scratch, bytes));
  }
  if (err != CUDA_SUCCESS) {
    front_fatal("no %zu bytes of scratch memory to make a cuRAND "
                "generator's numbers again: error %d\n", bytes, err);
  }
  front_log("making %zu numbers of a cuRAND generator again\n", numbers);
  for (const Op &op : ops) {
    for (size_t i = 0; i < op.times; i++) {
      curandStatus_t rc;
      RAND_LH(rc = call_op(real, op, scratch));
      if (rc != CURAND_STATUS_SUCCESS) {
        front_fatal("making a cuRAND generator's numbers again: status "
                    "%d\n", rc);
      }
    }
  }
  if (r->ctx == NULL) {
    free(scratch);
  } else if (scratch != NULL) {
    FRONT_LH(err = DRV(cuCtxSynchronize)());
    FRONT_LH(err = DRV(cuMemFree_v2)((CUdeviceptr)scratch));
  }
}

void
rand_restore(FrontObj *obj)
{
  Rand *r = (Rand *)obj;
  r->real = NULL;
  if (r->gone) {
    return;
  }
  if (r->ctx != NULL) {
    front_ctx_use(r->ctx);
  }
  curandGenerator_t real;
  curandStatus_t rc = make_generator(r, true, &real);
  if (rc != CURAND_STATUS_SUCCESS) {
    front_fatal("making a cuRAND generator again: status %d\n", rc);
  }
  r->real = real;
  r->at_offset_only = !r->since.empty();
}

// A Philox generator that a restart put at the offset of its generations
// (at_offset_only), made again by them: a setter, or a generation that is
// not at an offset, would find it otherwise than the uninterrupted one
// (see the top).  In the current context.
curandStatus_t
remake(Rand *r)
{
  curandGenerator_t real;
  curandStatus_t rc = make_generator(r, false, &real);
  if (rc != CURAND_STATUS_SUCCESS) {
    return rc;
  }
  front_log("a Philox generator, made again by its numbers:\n");
  run_ops(r, real, r->since);
  if (r->has_stream && front_stream_live(r->stream)) {
    RAND_LH(rc = RAND(curandSetStream)(real, front_real_stream(r->stream)));
  }
  curandStatus_t destroyed;
  RAND_LH(destroyed = RAND(curandDestroyGenerator)(
            (curandGenerator_t)r->real));
  (void)destroyed;
  r->real = real;
  r->at_offset_only = false;
  return rc;
}

void
rand_settle()
{
  for (FrontObj *o = obj_heads[FRONT_RAND]; o != NULL; o = o->next) {
    Rand *r = (Rand *)o;
    if (r->gone) {
      continue;
    }
    if (r->ctx != NULL) {
      front_ctx_use(r->ctx);
    }
    if (!r->log.empty()) {
      run_ops(r, (curandGenerator_t)r->real, r->log);
    }
    curandStatus_t rc = CURAND_STATUS_SUCCESS;
    if (r->has_stream && front_stream_live(r->stream)) {
      RAND_LH(rc = RAND(curandSetStream)((curandGenerator_t)r->real,
                                         front_real_stream(r->stream)));
    }
    if (rc != CURAND_STATUS_SUCCESS) {
      front_fatal("setting the stream of a cuRAND generator again: status "
                  "%d\n", rc);
    }
  }
}

void
rand_forget(Ctx *ctx)
{
  for (FrontObj *o = obj_heads[FRONT_RAND]; o != NULL; o = o->next) {
    if (((Rand *)o)->ctx == ctx && ctx != NULL) {
      ((Rand *)o)->gone = true;
    }
  }
}

const FrontLibRecords rand_records = { rand_restore, rand_settle,
                                       rand_forget };

__attribute__((constructor)) void
register_records()
{
  front_lib_records(FRONT_RAND, &rand_records);
}

curandStatus_t
create(curandGenerator_t *generator, curandRngType_t type, bool host)
{
  FRONT_ENTER();
  if (!host) {
    front_before_library();
  }
  curandGenerator_t real = NULL;
  curandStatus_t rc;
  if (host) {
    RAND_LH(rc = RAND(curandCreateGeneratorHost)(
              generator != NULL ? &real : NULL, type));
  } else {
    RAND_LH(rc = RAND(curandCreateGenerator)(
              generator != NULL ? &real : NULL, type));
  }
  if (rc == CURAND_STATUS_SUCCESS && generator != NULL) {
    Rand *r = new Rand();
    r->real = real;
    r->type = type == CURAND_RNG_PSEUDO_DEFAULT ? CURAND_RNG_PSEUDO_XORWOW
              : type == CURAND_RNG_QUASI_DEFAULT ? CURAND_RNG_QUASI_SOBOL32
                                                  : type;
    r->ctx = host ? NULL : front_object_ctx();
    std::lock_guard<std::mutex> guard(objs_lock);
    obj_link(r, FRONT_RAND);
    *generator = (curandGenerator_t)r;
  }
  FRONT_LEAVE();
  return rc;
}

// A call with the generator translated, and noted if it succeeded.
// 'exact': the call needs the generator as the uninterrupted one would be,
// not only at its offset (remake()).
template <typename Call, typename Note>
curandStatus_t
with_generator(curandGenerator_t generator, bool exact, Call call, Note note)
{
  FRONT_ENTER();
  curandStatus_t rc = CURAND_STATUS_SUCCESS;
  if (exact && generator != NULL && rand_of(generator)->at_offset_only) {
    rc = remake(rand_of(generator));
  }
  if (rc == CURAND_STATUS_SUCCESS) {
    curandGenerator_t real = real_of(generator);
    RAND_LH(rc = call(real));
  }
  if (rc == CURAND_STATUS_SUCCESS && generator != NULL) {
    std::lock_guard<std::mutex> guard(objs_lock);
    note(rand_of(generator));
  }
  FRONT_LEAVE();
  return rc;
}

curandStatus_t
setter(curandGenerator_t generator, Op op)
{
  return with_generator(
    generator, true,
    [&](curandGenerator_t real) { return call_op(real, op, NULL); },
    [&](Rand *r) { note_setter(r, op); });
}

Op
setting(Op::Kind kind, unsigned long long value)
{
  Op op = Op();
  op.kind = kind;
  op.value = value;
  return op;
}

curandStatus_t
generation(curandGenerator_t generator, Fn fn, void *out, size_t n,
           double a = 0, double b = 0, int method = 0)
{
  Op op = Op();
  op.kind = Op::GENERATE;
  op.fn = fn;
  op.n = n;
  op.a = a;
  op.b = b;
  op.method = method;
  bool exact = generator != NULL && !at_offset(rand_of(generator), fn);
  return with_generator(
    generator, exact,
    [&](curandGenerator_t real) { return call_op(real, op, out); },
    [&](Rand *r) { note_generate(r, op); });
}

// ---- Memory of the library's ---------------------------------------------

// Whether [addr, addr + size) is mapped and readable.
bool
readable(const void *addr, size_t size)
{
  FILE *maps = fopen("/proc/self/maps", "r");
  if (maps == NULL) {
    return false;
  }
  uintptr_t at = (uintptr_t)addr, end = at + size;
  char line[4096];
  while (at < end && fgets(line, sizeof line, maps) != NULL) {
    unsigned long start, stop;
    char perms[8] = "";
    if (sscanf(line, "%lx-%lx %7s", &start, &stop, perms) == 3 &&
        at >= start && at < stop && perms[0] == 'r') {
      at = stop;
    }
  }
  fclose(maps);
  return at >= end;
}

// The application may keep these pointers across a restart, which loads the
// library elsewhere: copies in upper-half memory, made once.  Sobol has up
// to 20000 dimensions (curandSetQuasiRandomGeneratorDimensions).
const size_t SOBOL_DIMENSIONS = 20000;
std::map<int, void *> tables;

template <typename T, typename Get>
curandStatus_t
library_table(int key, T **out, Get get)
{
  if (out != NULL) {
    std::lock_guard<std::mutex> guard(objs_lock);
    auto it = tables.find(key);
    if (it != tables.end()) {
      *out = (T *)it->second;
      return CURAND_STATUS_SUCCESS;
    }
  }
  FRONT_ENTER();
  T *table = NULL;
  curandStatus_t rc;
  RAND_LH(rc = get(out != NULL ? &table : NULL));
  if (rc == CURAND_STATUS_SUCCESS && out != NULL) {
    size_t size = SOBOL_DIMENSIONS * sizeof(T);
    if (!readable(table, size)) {
      front_fatal("cuRAND's table %d at %p is not %zu bytes\n", key,
                  (void *)table, size);
    }
    void *copy = malloc(size);
    if (copy == NULL) {
      rc = CURAND_STATUS_ALLOCATION_FAILED;
    } else {
      memcpy(copy, table, size);
      std::lock_guard<std::mutex> guard(objs_lock);
      auto added = tables.emplace(key, copy);
      if (!added.second) {
        free(copy);
      }
      *out = (T *)added.first->second;
    }
  }
  FRONT_LEAVE();
  return rc;
}

// Poisson distributions for the device API: copies in the application's
// device memory, which comes back at its address.  Kernels read the
// structures of curand_discrete.h through their device pointers; the host
// pointers in them are the library's, and are cleared.
std::set<void *> distributions;

cudaError_t
from_device(void *dst, const void *src, size_t size)
{
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaMemcpy)(dst, src, size, cudaMemcpyDeviceToHost));
  return rc;
}

size_t
aligned(size_t size)
{
  return (size + 15) & ~(size_t)15;
}

curandStatus_t
copy_distribution(curandDiscreteDistribution_t real,
                  curandDiscreteDistribution_t *out)
{
  curandDiscreteDistribution_st d;
  curandDistributionM2Shift_st m2 = curandDistributionM2Shift_st();
  curandHistogramM2_st h = curandHistogramM2_st();
  cudaError_t err = from_device(&d, real, sizeof d);
  if (err == cudaSuccess && d.M2 != NULL) {
    err = from_device(&m2, d.M2, sizeof m2);
  }
  if (err == cudaSuccess && m2.histogram != NULL) {
    err = from_device(&h, m2.histogram, sizeof h);
  }
  size_t length = m2.histogram != NULL ? m2.length : 0;
  size_t at_m2 = aligned(sizeof d), at_h = at_m2 + aligned(sizeof m2),
         at_v = at_h + aligned(sizeof h),
         at_k = at_v + aligned(length * sizeof(double)),
         size = at_k + length * sizeof(unsigned);
  std::vector<char> host(size);
  if (err == cudaSuccess && length > 0) {
    err = from_device(&host[at_v], h.V, length * sizeof(double));
  }
  if (err == cudaSuccess && length > 0) {
    err = from_device(&host[at_k], h.K, length * sizeof(unsigned));
  }
  if (err != cudaSuccess) {
    return CURAND_STATUS_INTERNAL_ERROR;
  }
  char *block = NULL;
  if (front_vmm_alloc((void **)&block, size) != cudaSuccess) {
    return CURAND_STATUS_ALLOCATION_FAILED;
  }
  d.self_host_ptr = NULL;
  d.host_M2 = NULL;
  if (d.M2 != NULL) {
    d.M2 = (curandDistributionM2Shift_t)(block + at_m2);
  }
  m2.host_histogram = NULL;
  if (m2.histogram != NULL) {
    m2.histogram = (curandHistogramM2_t)(block + at_h);
    h.V = (curandHistogramM2V_t)(block + at_v);
    h.K = (curandHistogramM2K_t)(block + at_k);
  }
  h.host_V = NULL;
  h.host_K = NULL;
  memcpy(&host[0], &d, sizeof d);
  memcpy(&host[at_m2], &m2, sizeof m2);
  memcpy(&host[at_h], &h, sizeof h);
  FRONT_LH(err = REAL(cudaMemcpy)(block, host.data(), size,
                                  cudaMemcpyHostToDevice));
  if (err != cudaSuccess) {
    front_vmm_free(block, &err);
    return CURAND_STATUS_INTERNAL_ERROR;
  }
  std::lock_guard<std::mutex> guard(objs_lock);
  distributions.insert(block);
  *out = (curandDiscreteDistribution_t)block;
  return CURAND_STATUS_SUCCESS;
}

}  // namespace

// ---- Generators ----------------------------------------------------------

extern "C" curandStatus_t
curandCreateGenerator(curandGenerator_t *generator, curandRngType_t rng_type)
{
  return create(generator, rng_type, false);
}

extern "C" curandStatus_t
curandCreateGeneratorHost(curandGenerator_t *generator,
                          curandRngType_t rng_type)
{
  return create(generator, rng_type, true);
}

extern "C" curandStatus_t
curandDestroyGenerator(curandGenerator_t generator)
{
  FRONT_ENTER();
  curandStatus_t rc;
  RAND_LH(rc = RAND(curandDestroyGenerator)(real_of(generator)));
  if (rc == CURAND_STATUS_SUCCESS && generator != NULL) {
    Rand *r = rand_of(generator);
    std::lock_guard<std::mutex> guard(objs_lock);
    obj_unlink(r);  // clears the magic
    delete r;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" curandStatus_t
curandSetStream(curandGenerator_t generator, cudaStream_t stream)
{
  return with_generator(
    generator, false,
    [&](curandGenerator_t real) {
      return RAND(curandSetStream)(real, front_real_stream(stream));
    },
    [&](Rand *r) {
      r->has_stream = true;
      r->stream = stream;
    });
}

extern "C" curandStatus_t
curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator,
                                   unsigned long long seed)
{
  return setter(generator, setting(Op::SEED, seed));
}

extern "C" curandStatus_t
curandSetGeneratorOrdering(curandGenerator_t generator,
                           curandOrdering_t order)
{
  return setter(generator, setting(Op::ORDERING, order));
}

extern "C" curandStatus_t
curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator,
                                        unsigned int num_dimensions)
{
  return setter(generator, setting(Op::DIMENSIONS, num_dimensions));
}

extern "C" curandStatus_t
curandGenerateSeeds(curandGenerator_t generator)
{
  return setter(generator, setting(Op::SEEDS, 0));
}

extern "C" curandStatus_t
curandSetGeneratorOffset(curandGenerator_t generator,
                         unsigned long long offset)
{
  return with_generator(
    generator, false,
    [&](curandGenerator_t real) {
      return RAND(curandSetGeneratorOffset)(real, offset);
    },
    [&](Rand *r) { note_offset(r, offset); });
}

extern "C" curandStatus_t
curandGenerate(curandGenerator_t generator, unsigned int *outputPtr,
               size_t num)
{
  return generation(generator, U32, outputPtr, num);
}

extern "C" curandStatus_t
curandGenerateLongLong(curandGenerator_t generator,
                       unsigned long long *outputPtr, size_t num)
{
  return generation(generator, U64, outputPtr, num);
}

extern "C" curandStatus_t
curandGenerateUniform(curandGenerator_t generator, float *outputPtr,
                      size_t num)
{
  return generation(generator, UNIFORM, outputPtr, num);
}

extern "C" curandStatus_t
curandGenerateUniformDouble(curandGenerator_t generator, double *outputPtr,
                            size_t num)
{
  return generation(generator, UNIFORM_DOUBLE, outputPtr, num);
}

extern "C" curandStatus_t
curandGenerateNormal(curandGenerator_t generator, float *outputPtr, size_t n,
                     float mean, float stddev)
{
  return generation(generator, NORMAL, outputPtr, n, mean, stddev);
}

extern "C" curandStatus_t
curandGenerateNormalDouble(curandGenerator_t generator, double *outputPtr,
                           size_t n, double mean, double stddev)
{
  return generation(generator, NORMAL_DOUBLE, outputPtr, n, mean, stddev);
}

extern "C" curandStatus_t
curandGenerateLogNormal(curandGenerator_t generator, float *outputPtr,
                        size_t n, float mean, float stddev)
{
  return generation(generator, LOG_NORMAL, outputPtr, n, mean, stddev);
}

extern "C" curandStatus_t
curandGenerateLogNormalDouble(curandGenerator_t generator, double *outputPtr,
                              size_t n, double mean, double stddev)
{
  return generation(generator, LOG_NORMAL_DOUBLE, outputPtr, n, mean,
                    stddev);
}

extern "C" curandStatus_t
curandGeneratePoisson(curandGenerator_t generator, unsigned int *outputPtr,
                      size_t n, double lambda)
{
  return generation(generator, POISSON, outputPtr, n, lambda);
}

extern "C" curandStatus_t
curandGeneratePoissonMethod(curandGenerator_t generator,
                            unsigned int *outputPtr, size_t n, double lambda,
                            curandMethod_t method)
{
  return generation(generator, POISSON_METHOD, outputPtr, n, lambda, 0,
                    method);
}

// ---- Memory of the library's ---------------------------------------------

extern "C" curandStatus_t
curandCreatePoissonDistribution(double lambda,
                                curandDiscreteDistribution_t
                                  *discrete_distribution)
{
  FRONT_ENTER();
  front_before_library();
  curandDiscreteDistribution_t real = NULL;
  curandStatus_t rc;
  RAND_LH(rc = RAND(curandCreatePoissonDistribution)(
            lambda, discrete_distribution != NULL ? &real : NULL));
  if (rc == CURAND_STATUS_SUCCESS && discrete_distribution != NULL) {
    rc = copy_distribution(real, discrete_distribution);
    curandStatus_t destroyed;
    RAND_LH(destroyed = RAND(curandDestroyDistribution)(real));
    (void)destroyed;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" curandStatus_t
curandDestroyDistribution(curandDiscreteDistribution_t discrete_distribution)
{
  FRONT_ENTER();
  bool ours;
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    ours = distributions.erase(discrete_distribution) > 0;
  }
  curandStatus_t rc = CURAND_STATUS_SUCCESS;
  if (ours) {
    cudaError_t err;
    front_vmm_free(discrete_distribution, &err);
  } else {
    RAND_LH(rc = RAND(curandDestroyDistribution)(discrete_distribution));
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" curandStatus_t
curandGetDirectionVectors32(curandDirectionVectors32_t *vectors[],
                            curandDirectionVectorSet_t set)
{
  return library_table((int)set, vectors,
                       [&](curandDirectionVectors32_t **out) {
                         return RAND(curandGetDirectionVectors32)(out, set);
                       });
}

extern "C" curandStatus_t
curandGetDirectionVectors64(curandDirectionVectors64_t *vectors[],
                            curandDirectionVectorSet_t set)
{
  return library_table((int)set, vectors,
                       [&](curandDirectionVectors64_t **out) {
                         return RAND(curandGetDirectionVectors64)(out, set);
                       });
}

extern "C" curandStatus_t
curandGetScrambleConstants32(unsigned int **constants)
{
  return library_table(32, constants, [&](unsigned **out) {
    return RAND(curandGetScrambleConstants32)(out);
  });
}

extern "C" curandStatus_t
curandGetScrambleConstants64(unsigned long long **constants)
{
  return library_table(64, constants, [&](unsigned long long **out) {
    return RAND(curandGetScrambleConstants64)(out);
  });
}
