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

// cuFFT (cuda/DESIGN-lower-half-cuda.md, Phase 3), in the lower half with
// its own static cudart.  A plan is an int, which no type tells from other
// ints, so every function with a plan is the front's: the application's
// cufftHandle is a number of the front's, and names a record of the plan.
//
// The record keeps the call that made the plan (cufftPlan*, cufftMakePlan*,
// cufftXtMakePlanMany) with its arguments, the properties and the
// auto-allocation set before it, and the work area and stream set after.  A
// restart makes the plan again with the same calls, at the place of the
// making in the replay (it loads kernels and allocates device memory), then
// sets the work area and stream once all records exist.  The plan's own
// device memory (an auto-allocated work area, twiddles) is the library's, in
// the lower half, and is made again: it is scratch between executions.
//
// Not supported: several GPUs (cufftXtSetGPUs and the cudaLibXtDesc
// functions), callbacks, and the device API of cufftDevice*.

#include <cufftXt.h>
#include <algorithm>
#include <map>
#include <utility>
#include <vector>
#include "front_objs.h"
#include "gen/cufft-index.h"

extern FrontLib front_cufft_lib FRONT_HIDDEN;

#define FFT(f) ((decltype(&::f))front_cufft_lib.real[FRONT_IDX_cufft_##f])

#define FFT_LH(stmt)                     \
  do {                                   \
    front_check_bound(&front_cufft_lib); \
    FRONT_LH(stmt);                      \
  } while (0)

namespace {

enum Make {
  NOT_MADE,
  PLAN_1D,               // cufftPlan1d: made with the plan
  PLAN_2D,
  PLAN_3D,
  PLAN_MANY,
  MAKE_1D,               // cufftMakePlan1d: on a plan of cufftCreate
  MAKE_2D,
  MAKE_3D,
  MAKE_MANY,
  MAKE_MANY64,
  XT_MAKE_MANY
};

// cufftSetPlanPropertyInt64, or cufftSetAutoAllocation (property -1).
struct PlanSetting {
  int property;
  long long value;
  bool after_make;
};

struct Plan : FrontObj {
  int handle;            // the application's
  int real_plan;         // the lower half's, of generation front_gen
  Ctx *ctx;
  bool gone;             // its context was destroyed
  Make how;
  // The arguments of the making.
  int nx, ny, nz;
  int rank;
  std::vector<long long> n, inembed, onembed;  // empty: NULL
  long long istride, idist, ostride, odist, batch;
  cufftType type;
  cudaDataType itype, otype, etype;
  bool knows_work_size;
  size_t work_size;      // what the making reported
  std::vector<PlanSetting> settings;  // in the order of their last calls
  bool has_work_area;
  void *work_area;
  bool has_stream;
  cudaStream_t stream;   // virtual
};

int next_handle;
std::map<int, Plan *> plans;     // by the application's handle

// Under the gate.
Plan *
plan_of(cufftHandle handle)
{
  std::lock_guard<std::mutex> guard(objs_lock);
  auto it = plans.find(handle);
  return it != plans.end() ? it->second : NULL;
}

std::vector<int>
to_int(const std::vector<long long> &v)
{
  return std::vector<int>(v.begin(), v.end());
}

// The sizes of a rank, copied; cuFFT's ranks are 1 to 3, and it refuses
// others before it reads them.
template <typename T>
std::vector<long long>
from(int rank, const T *v)
{
  rank = std::max(0, std::min(rank, 3));
  return v != NULL ? std::vector<long long>(v, v + rank)
                   : std::vector<long long>();
}

template <typename T>
T *
data_or_null(std::vector<T> &v)
{
  return v.empty() ? NULL : v.data();
}

// The call that made 'p', again: on p->real_plan, or, for cufftPlan*,
// making it.
cufftResult
make(Plan *p, size_t *work)
{
  cufftResult rc = CUFFT_SUCCESS;
  int real = p->real_plan;
  std::vector<int> n = to_int(p->n), in = to_int(p->inembed),
                   on = to_int(p->onembed);
  int *n_ = data_or_null(n), *in_ = data_or_null(in),
      *on_ = data_or_null(on);
  long long *n64 = data_or_null(p->n), *in64 = data_or_null(p->inembed),
            *on64 = data_or_null(p->onembed);
  int batch = (int)p->batch;
  switch (p->how) {
    case NOT_MADE:
      break;
    case PLAN_1D:
      FFT_LH(rc = FFT(cufftPlan1d)(&p->real_plan, p->nx, p->type, batch));
      break;
    case PLAN_2D:
      FFT_LH(rc = FFT(cufftPlan2d)(&p->real_plan, p->nx, p->ny, p->type));
      break;
    case PLAN_3D:
      FFT_LH(rc = FFT(cufftPlan3d)(&p->real_plan, p->nx, p->ny, p->nz,
                                   p->type));
      break;
    case PLAN_MANY:
      FFT_LH(rc = FFT(cufftPlanMany)(&p->real_plan, p->rank, n_, in_,
                                     (int)p->istride, (int)p->idist, on_,
                                     (int)p->ostride, (int)p->odist, p->type,
                                     batch));
      break;
    case MAKE_1D:
      FFT_LH(rc = FFT(cufftMakePlan1d)(real, p->nx, p->type, batch, work));
      break;
    case MAKE_2D:
      FFT_LH(rc = FFT(cufftMakePlan2d)(real, p->nx, p->ny, p->type, work));
      break;
    case MAKE_3D:
      FFT_LH(rc = FFT(cufftMakePlan3d)(real, p->nx, p->ny, p->nz, p->type,
                                       work));
      break;
    case MAKE_MANY:
      FFT_LH(rc = FFT(cufftMakePlanMany)(real, p->rank, n_, in_,
                                         (int)p->istride, (int)p->idist, on_,
                                         (int)p->ostride, (int)p->odist,
                                         p->type, batch, work));
      break;
    case MAKE_MANY64:
      FFT_LH(rc = FFT(cufftMakePlanMany64)(real, p->rank, n64, in64,
                                           p->istride, p->idist, on64,
                                           p->ostride, p->odist, p->type,
                                           p->batch, work));
      break;
    case XT_MAKE_MANY:
      FFT_LH(rc = FFT(cufftXtMakePlanMany)(real, p->rank, n64, in64,
                                           p->istride, p->idist, p->itype,
                                           on64, p->ostride, p->odist,
                                           p->otype, p->batch, work,
                                           p->etype));
      break;
  }
  return rc;
}

bool
made_with_plan(Make how)
{
  return how >= PLAN_1D && how <= PLAN_MANY;
}

void
apply_settings(Plan *p, bool after_make)
{
  for (const PlanSetting &s : p->settings) {
    if (s.after_make != after_make) {
      continue;
    }
    cufftResult rc;
    if (s.property < 0) {
      FFT_LH(rc = FFT(cufftSetAutoAllocation)(p->real_plan, (int)s.value));
    } else {
      FFT_LH(rc = FFT(cufftSetPlanPropertyInt64)(
               p->real_plan, (cufftProperty)s.property, s.value));
    }
    if (rc != CUFFT_SUCCESS) {
      front_fatal("setting %s of a cuFFT plan again: result %d\n",
                  s.property < 0 ? "auto-allocation" : "a property", rc);
    }
  }
}

void
note_setting(Plan *p, int property, long long value)
{
  std::lock_guard<std::mutex> guard(objs_lock);
  for (auto it = p->settings.begin(); it != p->settings.end(); ++it) {
    if (it->property == property) {
      p->settings.erase(it);
      break;
    }
  }
  p->settings.push_back({ property, value, p->how != NOT_MADE });
}

void
fft_restore(FrontObj *obj)
{
  Plan *p = (Plan *)obj;
  if (p->gone) {
    return;
  }
  front_ctx_use(p->ctx);
  cufftResult rc;
  size_t work[64] = { 0 };  // one per GPU
  if (made_with_plan(p->how)) {
    rc = make(p, NULL);
  } else {
    FFT_LH(rc = FFT(cufftCreate)(&p->real_plan));
    if (rc == CUFFT_SUCCESS) {
      apply_settings(p, false);
      rc = make(p, work);
    }
  }
  if (rc != CUFFT_SUCCESS) {
    front_fatal("making cuFFT plan %d again: result %d\n", p->handle, rc);
  }
  apply_settings(p, true);
  if (p->has_work_area && p->knows_work_size && work[0] > p->work_size) {
    front_fatal("cuFFT plan %d needs a work area of %zu bytes now, not %zu\n",
                p->handle, work[0], p->work_size);
  }
}

void
fft_settle()
{
  for (FrontObj *o = obj_heads[FRONT_FFT]; o != NULL; o = o->next) {
    Plan *p = (Plan *)o;
    if (p->gone) {
      continue;
    }
    cufftResult rc = CUFFT_SUCCESS;
    front_ctx_use(p->ctx);
    if (p->has_work_area) {
      FFT_LH(rc = FFT(cufftSetWorkArea)(p->real_plan, p->work_area));
    }
    if (rc == CUFFT_SUCCESS && p->has_stream &&
        front_stream_live(p->stream)) {
      FFT_LH(rc = FFT(cufftSetStream)(p->real_plan,
                                      front_real_stream(p->stream)));
    }
    if (rc != CUFFT_SUCCESS) {
      front_fatal("setting the work area or stream of cuFFT plan %d again: "
                  "result %d\n", p->handle, rc);
    }
  }
}

void
fft_forget(Ctx *ctx)
{
  for (FrontObj *o = obj_heads[FRONT_FFT]; o != NULL; o = o->next) {
    if (((Plan *)o)->ctx == ctx) {
      ((Plan *)o)->gone = true;
    }
  }
}

const FrontLibRecords fft_records = { fft_restore, fft_settle, fft_forget };

__attribute__((constructor)) void
register_records()
{
  front_lib_records(FRONT_FFT, &fft_records);
}

// A new plan's record, with its handle.
void
link_plan(Plan *p, cufftHandle *handle)
{
  p->ctx = front_object_ctx();
  std::lock_guard<std::mutex> guard(objs_lock);
  p->handle = ++next_handle;
  plans[p->handle] = p;
  obj_link(p, FRONT_FFT);
  *handle = p->handle;
}

// cufftPlan*: a plan made at once.
cufftResult
plan_now(cufftHandle *handle, Plan *p)
{
  if (handle == NULL) {
    delete p;
    return CUFFT_INVALID_VALUE;
  }
  FRONT_ENTER();
  front_before_library();
  cufftResult rc = make(p, NULL);
  if (rc == CUFFT_SUCCESS) {
    link_plan(p, handle);
  } else {
    delete p;
  }
  FRONT_LEAVE();
  return rc;
}

// cufftMakePlan*, cufftXtMakePlanMany: 'args' has the arguments.  The
// plan's place in the replay is now.
cufftResult
make_now(cufftHandle handle, const Plan &args, size_t *work)
{
  FRONT_ENTER();
  cufftResult rc = CUFFT_INVALID_PLAN;
  Plan *p = plan_of(handle);
  if (p != NULL) {
    Plan q = args;
    q.real_plan = p->real_plan;
    rc = make(&q, work);
    if (rc == CUFFT_SUCCESS) {
      std::lock_guard<std::mutex> guard(objs_lock);
      p->how = q.how;
      p->nx = q.nx;
      p->ny = q.ny;
      p->nz = q.nz;
      p->rank = q.rank;
      p->n = q.n;
      p->inembed = q.inembed;
      p->onembed = q.onembed;
      p->istride = q.istride;
      p->idist = q.idist;
      p->ostride = q.ostride;
      p->odist = q.odist;
      p->batch = q.batch;
      p->type = q.type;
      p->itype = q.itype;
      p->otype = q.otype;
      p->etype = q.etype;
      p->knows_work_size = work != NULL;
      p->work_size = work != NULL ? work[0] : 0;
      obj_unlink(p);
      obj_link(p, FRONT_FFT);
    }
  }
  FRONT_LEAVE();
  return rc;
}

Plan *
args(Make how)
{
  Plan *p = new Plan();
  p->how = how;
  return p;
}

}  // namespace

extern "C" cufftResult
cufftCreate(cufftHandle *handle)
{
  if (handle == NULL) {
    return CUFFT_INVALID_VALUE;
  }
  FRONT_ENTER();
  front_before_library();
  Plan *p = new Plan();
  cufftResult rc;
  FFT_LH(rc = FFT(cufftCreate)(&p->real_plan));
  if (rc == CUFFT_SUCCESS) {
    link_plan(p, handle);
  } else {
    delete p;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cufftResult
cufftDestroy(cufftHandle handle)
{
  FRONT_ENTER();
  cufftResult rc = CUFFT_INVALID_PLAN;
  Plan *p = plan_of(handle);
  if (p != NULL) {
    FFT_LH(rc = FFT(cufftDestroy)(p->real_plan));
  }
  if (rc == CUFFT_SUCCESS) {
    std::lock_guard<std::mutex> guard(objs_lock);
    plans.erase(handle);
    obj_unlink(p);
    delete p;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cufftResult
cufftPlan1d(cufftHandle *handle, int nx, cufftType type, int batch)
{
  Plan *p = args(PLAN_1D);
  p->nx = nx;
  p->type = type;
  p->batch = batch;
  return plan_now(handle, p);
}

extern "C" cufftResult
cufftPlan2d(cufftHandle *handle, int nx, int ny, cufftType type)
{
  Plan *p = args(PLAN_2D);
  p->nx = nx;
  p->ny = ny;
  p->type = type;
  return plan_now(handle, p);
}

extern "C" cufftResult
cufftPlan3d(cufftHandle *handle, int nx, int ny, int nz, cufftType type)
{
  Plan *p = args(PLAN_3D);
  p->nx = nx;
  p->ny = ny;
  p->nz = nz;
  p->type = type;
  return plan_now(handle, p);
}

extern "C" cufftResult
cufftPlanMany(cufftHandle *handle, int rank, int *n, int *inembed,
              int istride, int idist, int *onembed, int ostride, int odist,
              cufftType type, int batch)
{
  Plan *p = args(PLAN_MANY);
  p->rank = rank;
  p->n = from(rank, n);
  p->inembed = from(rank, inembed);
  p->onembed = from(rank, onembed);
  p->istride = istride;
  p->idist = idist;
  p->ostride = ostride;
  p->odist = odist;
  p->type = type;
  p->batch = batch;
  return plan_now(handle, p);
}

extern "C" cufftResult
cufftMakePlan1d(cufftHandle handle, int nx, cufftType type, int batch,
                size_t *work)
{
  Plan a = Plan();
  a.how = MAKE_1D;
  a.nx = nx;
  a.type = type;
  a.batch = batch;
  return make_now(handle, a, work);
}

extern "C" cufftResult
cufftMakePlan2d(cufftHandle handle, int nx, int ny, cufftType type,
                size_t *work)
{
  Plan a = Plan();
  a.how = MAKE_2D;
  a.nx = nx;
  a.ny = ny;
  a.type = type;
  return make_now(handle, a, work);
}

extern "C" cufftResult
cufftMakePlan3d(cufftHandle handle, int nx, int ny, int nz, cufftType type,
                size_t *work)
{
  Plan a = Plan();
  a.how = MAKE_3D;
  a.nx = nx;
  a.ny = ny;
  a.nz = nz;
  a.type = type;
  return make_now(handle, a, work);
}

extern "C" cufftResult
cufftMakePlanMany(cufftHandle handle, int rank, int *n, int *inembed,
                  int istride, int idist, int *onembed, int ostride,
                  int odist, cufftType type, int batch, size_t *work)
{
  Plan a = Plan();
  a.how = MAKE_MANY;
  a.rank = rank;
  a.n = from(rank, n);
  a.inembed = from(rank, inembed);
  a.onembed = from(rank, onembed);
  a.istride = istride;
  a.idist = idist;
  a.ostride = ostride;
  a.odist = odist;
  a.type = type;
  a.batch = batch;
  return make_now(handle, a, work);
}

extern "C" cufftResult
cufftMakePlanMany64(cufftHandle handle, int rank, long long *n,
                    long long *inembed, long long istride, long long idist,
                    long long *onembed, long long ostride, long long odist,
                    cufftType type, long long batch, size_t *work)
{
  Plan a = Plan();
  a.how = MAKE_MANY64;
  a.rank = rank;
  a.n = from(rank, n);
  a.inembed = from(rank, inembed);
  a.onembed = from(rank, onembed);
  a.istride = istride;
  a.idist = idist;
  a.ostride = ostride;
  a.odist = odist;
  a.type = type;
  a.batch = batch;
  return make_now(handle, a, work);
}

extern "C" cufftResult
cufftXtMakePlanMany(cufftHandle handle, int rank, long long *n,
                    long long *inembed, long long istride, long long idist,
                    cudaDataType itype, long long *onembed, long long ostride,
                    long long odist, cudaDataType otype, long long batch,
                    size_t *work, cudaDataType etype)
{
  Plan a = Plan();
  a.how = XT_MAKE_MANY;
  a.rank = rank;
  a.n = from(rank, n);
  a.inembed = from(rank, inembed);
  a.onembed = from(rank, onembed);
  a.istride = istride;
  a.idist = idist;
  a.itype = itype;
  a.ostride = ostride;
  a.odist = odist;
  a.otype = otype;
  a.batch = batch;
  a.etype = etype;
  return make_now(handle, a, work);
}

extern "C" cufftResult
cufftSetAutoAllocation(cufftHandle handle, int auto_allocate)
{
  FRONT_ENTER();
  cufftResult rc = CUFFT_INVALID_PLAN;
  Plan *p = plan_of(handle);
  if (p != NULL) {
    FFT_LH(rc = FFT(cufftSetAutoAllocation)(p->real_plan, auto_allocate));
  }
  if (rc == CUFFT_SUCCESS) {
    note_setting(p, -1, auto_allocate);
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cufftResult
cufftSetPlanPropertyInt64(cufftHandle handle, cufftProperty property,
                          const long long value)
{
  FRONT_ENTER();
  cufftResult rc = CUFFT_INVALID_PLAN;
  Plan *p = plan_of(handle);
  if (p != NULL) {
    FFT_LH(rc = FFT(cufftSetPlanPropertyInt64)(p->real_plan, property,
                                               value));
  }
  if (rc == CUFFT_SUCCESS) {
    note_setting(p, (int)property, value);
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cufftResult
cufftResetPlanProperty(cufftHandle handle, cufftProperty property)
{
  FRONT_ENTER();
  cufftResult rc = CUFFT_INVALID_PLAN;
  Plan *p = plan_of(handle);
  if (p != NULL) {
    FFT_LH(rc = FFT(cufftResetPlanProperty)(p->real_plan, property));
  }
  if (rc == CUFFT_SUCCESS) {
    std::lock_guard<std::mutex> guard(objs_lock);
    for (auto it = p->settings.begin(); it != p->settings.end(); ++it) {
      if (it->property == (int)property) {
        p->settings.erase(it);
        break;
      }
    }
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cufftResult
cufftSetWorkArea(cufftHandle handle, void *work_area)
{
  FRONT_ENTER();
  cufftResult rc = CUFFT_INVALID_PLAN;
  Plan *p = plan_of(handle);
  if (p != NULL) {
    FFT_LH(rc = FFT(cufftSetWorkArea)(p->real_plan, work_area));
  }
  if (rc == CUFFT_SUCCESS) {
    std::lock_guard<std::mutex> guard(objs_lock);
    p->has_work_area = true;
    p->work_area = work_area;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cufftResult
cufftSetStream(cufftHandle handle, cudaStream_t stream)
{
  FRONT_ENTER();
  cufftResult rc = CUFFT_INVALID_PLAN;
  Plan *p = plan_of(handle);
  if (p != NULL) {
    FFT_LH(rc = FFT(cufftSetStream)(p->real_plan,
                                    front_real_stream(stream)));
  }
  if (rc == CUFFT_SUCCESS) {
    std::lock_guard<std::mutex> guard(objs_lock);
    p->has_stream = true;
    p->stream = stream;
  }
  FRONT_LEAVE();
  return rc;
}

// The plan translated; the other arguments as they are.
#define FFT_FORWARD(name, params, ...)                    \
  extern "C" cufftResult name params                      \
  {                                                       \
    FRONT_ENTER();                                        \
    cufftResult rc = CUFFT_INVALID_PLAN;                  \
    Plan *p = plan_of(handle);                            \
    if (p != NULL) {                                      \
      FFT_LH(rc = FFT(name)(p->real_plan, __VA_ARGS__));  \
    }                                                     \
    FRONT_LEAVE();                                        \
    return rc;                                            \
  }

FFT_FORWARD(cufftExecC2C,
            (cufftHandle handle, cufftComplex *in, cufftComplex *out,
             int direction),
            in, out, direction)
FFT_FORWARD(cufftExecR2C,
            (cufftHandle handle, cufftReal *in, cufftComplex *out), in, out)
FFT_FORWARD(cufftExecC2R,
            (cufftHandle handle, cufftComplex *in, cufftReal *out), in, out)
FFT_FORWARD(cufftExecZ2Z,
            (cufftHandle handle, cufftDoubleComplex *in,
             cufftDoubleComplex *out, int direction),
            in, out, direction)
FFT_FORWARD(cufftExecD2Z,
            (cufftHandle handle, cufftDoubleReal *in,
             cufftDoubleComplex *out),
            in, out)
FFT_FORWARD(cufftExecZ2D,
            (cufftHandle handle, cufftDoubleComplex *in,
             cufftDoubleReal *out),
            in, out)
FFT_FORWARD(cufftXtExec,
            (cufftHandle handle, void *in, void *out, int direction), in, out,
            direction)
FFT_FORWARD(cufftGetSize, (cufftHandle handle, size_t *work), work)
FFT_FORWARD(cufftGetSize1d,
            (cufftHandle handle, int nx, cufftType type, int batch,
             size_t *work),
            nx, type, batch, work)
FFT_FORWARD(cufftGetSize2d,
            (cufftHandle handle, int nx, int ny, cufftType type,
             size_t *work),
            nx, ny, type, work)
FFT_FORWARD(cufftGetSize3d,
            (cufftHandle handle, int nx, int ny, int nz, cufftType type,
             size_t *work),
            nx, ny, nz, type, work)
FFT_FORWARD(cufftGetSizeMany,
            (cufftHandle handle, int rank, int *n, int *inembed, int istride,
             int idist, int *onembed, int ostride, int odist, cufftType type,
             int batch, size_t *work),
            rank, n, inembed, istride, idist, onembed, ostride, odist, type,
            batch, work)
FFT_FORWARD(cufftGetSizeMany64,
            (cufftHandle handle, int rank, long long *n, long long *inembed,
             long long istride, long long idist, long long *onembed,
             long long ostride, long long odist, cufftType type,
             long long batch, size_t *work),
            rank, n, inembed, istride, idist, onembed, ostride, odist, type,
            batch, work)
FFT_FORWARD(cufftXtGetSizeMany,
            (cufftHandle handle, int rank, long long *n, long long *inembed,
             long long istride, long long idist, cudaDataType itype,
             long long *onembed, long long ostride, long long odist,
             cudaDataType otype, long long batch, size_t *work,
             cudaDataType etype),
            rank, n, inembed, istride, idist, itype, onembed, ostride, odist,
            otype, batch, work, etype)
FFT_FORWARD(cufftGetPlanPropertyInt64,
            (cufftHandle handle, cufftProperty property, long long *value),
            property, value)
