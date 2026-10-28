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

// cuBLAS and cuBLASLt (cuda/DESIGN-lower-half-cuda.md, Phase 3).  Both run
// in the lower half, each with its own static cudart; the front forwards
// their calls with the handles translated (front_blas.h), and owns:
//
//  * handles: records of the context that they were made in, and of the
//    settings of a cuBLAS handle (stream, workspace, pointer mode, ...).  A
//    restart makes each again at its place in the replay, then sets the
//    settings, once the streams exist.  A handle's own device memory (the
//    default workspace, ...) is the library's, in the lower half: only
//    scratch between calls, so it is made again, not saved;
//  * cuBLASLt's descriptors, made in upper-half memory with the library's
//    ...Init functions: they are plain data, and survive in the image;
//  * status strings, copied into upper-half memory.
//
// Not supported: the loggers (callbacks and FILEs would be the lower
// half's), cublasXt, and the legacy API of cublas.h (its implicit handle).

#include <stdlib.h>
#include <string.h>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "front_blas.h"
#include "front_objs.h"
#include "gen/cublas-index.h"
#include "gen/cublasLt-index.h"

#define BLAS(f) ((decltype(&::f))front_cublas_lib.real[FRONT_IDX_cublas_##f])
#define BLASLT(f) \
  ((decltype(&::f))front_cublasLt_lib.real[FRONT_IDX_cublasLt_##f])

// A call into the lower half's cuBLAS (BLAS(f)) or cuBLASLt (BLASLT(f)).
#define BLAS_LH(lib, stmt)     \
  do {                         \
    front_check_bound(&(lib)); \
    FRONT_LH(stmt);            \
  } while (0)

namespace {

// Restores one setting of a cuBLAS handle.
typedef cublasStatus_t (*BlasApply)(cublasHandle_t real, intptr_t value);

struct Blas : FrontObj {
  Ctx *ctx;
  bool gone;             // its context was destroyed
  cudaStream_t stream;   // virtual: cublasSetStream's
  bool has_stream;
  // cublasSetWorkspace, since the last cublasSetStream (which goes back to
  // the default workspace).
  bool has_workspace;
  void *workspace;
  size_t workspace_size;
  // The other settings, in the order of their last calls.
  std::vector<std::pair<BlasApply, intptr_t> > settings;
};

struct BlasLt : FrontObj {
  Ctx *ctx;
  bool gone;
};

std::map<std::pair<int, int>, std::string> status_strings;

Blas *
blas_of(cublasHandle_t handle)
{
  return (Blas *)handle;
}

// A status string of the lower half's library, copied: the application may
// keep it across a restart.  'which': 0 or 1 (cuBLAS name or string), 2 or 3
// (cuBLASLt's).
template <typename F>
const char *
status_string(int which, cublasStatus_t status, F lookup)
{
  std::pair<int, int> key(which, (int)status);
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    auto it = status_strings.find(key);
    if (it != status_strings.end()) {
      return it->second.c_str();
    }
  }
  FRONT_ENTER();
  const char *s = lookup();
  FRONT_LEAVE();
  std::lock_guard<std::mutex> guard(objs_lock);
  return status_strings.emplace(key, s != NULL ? s : "").first->second.c_str();
}

void
note_setting(cublasHandle_t handle, BlasApply apply, intptr_t value)
{
  Blas *b = blas_of(handle);
  std::lock_guard<std::mutex> guard(objs_lock);
  for (auto it = b->settings.begin(); it != b->settings.end(); ++it) {
    if (it->first == apply) {
      b->settings.erase(it);
      break;
    }
  }
  b->settings.emplace_back(apply, value);
}

void
blas_restore(FrontObj *obj)
{
  cublasStatus_t rc = CUBLAS_STATUS_SUCCESS;
  if (obj->kind == FRONT_BLAS) {
    Blas *b = (Blas *)obj;
    b->real = NULL;
    if (!b->gone) {
      front_ctx_use(b->ctx);
      BLAS_LH(front_cublas_lib,
              rc = BLAS(cublasCreate_v2)((cublasHandle_t *)&b->real));
    }
  } else {
    BlasLt *lt = (BlasLt *)obj;
    lt->real = NULL;
    if (!lt->gone) {
      front_ctx_use(lt->ctx);
      BLAS_LH(front_cublasLt_lib,
              rc = BLASLT(cublasLtCreate)((cublasLtHandle_t *)&lt->real));
    }
  }
  if (rc != CUBLAS_STATUS_SUCCESS) {
    front_fatal("creating a %s handle again: status %d\n",
                obj->kind == FRONT_BLAS ? "cuBLAS" : "cuBLASLt", rc);
  }
}

// The stream first: it resets the workspace.
void
blas_settle()
{
  for (FrontObj *o = obj_heads[FRONT_BLAS]; o != NULL; o = o->next) {
    Blas *b = (Blas *)o;
    if (b->gone) {
      continue;
    }
    cublasHandle_t real = (cublasHandle_t)b->real;
    cublasStatus_t rc = CUBLAS_STATUS_SUCCESS;
    front_ctx_use(b->ctx);
    if (b->has_stream && front_stream_live(b->stream)) {
      BLAS_LH(front_cublas_lib,
              rc = BLAS(cublasSetStream_v2)(real,
                                            front_real_stream(b->stream)));
    }
    if (rc == CUBLAS_STATUS_SUCCESS && b->has_workspace) {
      BLAS_LH(front_cublas_lib,
              rc = BLAS(cublasSetWorkspace_v2)(real, b->workspace,
                                               b->workspace_size));
    }
    for (auto &setting : b->settings) {
      if (rc == CUBLAS_STATUS_SUCCESS) {
        rc = setting.first(real, setting.second);
      }
    }
    if (rc != CUBLAS_STATUS_SUCCESS) {
      front_fatal("restoring the settings of a cuBLAS handle: status %d\n",
                  rc);
    }
  }
}

void
blas_forget(Ctx *ctx)
{
  for (FrontObj *o = obj_heads[FRONT_BLAS]; o != NULL; o = o->next) {
    if (((Blas *)o)->ctx == ctx) {
      ((Blas *)o)->gone = true;
    }
  }
  for (FrontObj *o = obj_heads[FRONT_BLASLT]; o != NULL; o = o->next) {
    if (((BlasLt *)o)->ctx == ctx) {
      ((BlasLt *)o)->gone = true;
    }
  }
}

void
no_settle()
{
}

void
no_forget(Ctx *)
{
}

const FrontLibRecords blas_records = { blas_restore, blas_settle,
                                       blas_forget };
const FrontLibRecords blaslt_records = { blas_restore, no_settle,
                                         no_forget };

__attribute__((constructor)) void
register_records()
{
  front_lib_records(FRONT_BLAS, &blas_records);
  front_lib_records(FRONT_BLASLT, &blaslt_records);
}

// A cuBLASLt descriptor in upper-half memory, made by 'init' (BLASLT_INIT).
template <typename Opaque, typename Init>
cublasStatus_t
make_descriptor(Opaque **desc, Init init)
{
  if (desc == NULL) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  Opaque *d = (Opaque *)calloc(1, sizeof(Opaque));
  if (d == NULL) {
    return CUBLAS_STATUS_ALLOC_FAILED;
  }
  FRONT_ENTER();
  cublasStatus_t rc = init(d);
  FRONT_LEAVE();
  if (rc == CUBLAS_STATUS_SUCCESS) {
    *desc = d;
  } else {
    free(d);
  }
  return rc;
}

#define BLASLT_INIT(init, ...)                                       \
  [=](auto *d) {                                                     \
    cublasStatus_t rc;                                               \
    BLAS_LH(front_cublasLt_lib,                                      \
            rc = BLASLT(init)(d, sizeof *d, ##__VA_ARGS__));         \
    return rc;                                                       \
  }

}  // namespace

// ---- cuBLAS --------------------------------------------------------------

extern "C" cublasStatus_t
cublasCreate_v2(cublasHandle_t *handle)
{
  if (handle == NULL) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  FRONT_ENTER();
  front_before_library();
  Blas *b = new Blas();
  cublasStatus_t rc;
  BLAS_LH(front_cublas_lib,
          rc = BLAS(cublasCreate_v2)((cublasHandle_t *)&b->real));
  if (rc == CUBLAS_STATUS_SUCCESS) {
    b->ctx = front_object_ctx();
    std::lock_guard<std::mutex> guard(objs_lock);
    obj_link(b, FRONT_BLAS);
    *handle = (cublasHandle_t)b;
  } else {
    delete b;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cublasStatus_t
cublasDestroy_v2(cublasHandle_t handle)
{
  FRONT_ENTER();
  cublasStatus_t rc;
  BLAS_LH(front_cublas_lib,
          rc = BLAS(cublasDestroy_v2)(front_translate(handle)));
  if (rc == CUBLAS_STATUS_SUCCESS && handle != NULL) {
    std::lock_guard<std::mutex> guard(objs_lock);
    obj_unlink(blas_of(handle));
    delete blas_of(handle);
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cublasStatus_t
cublasSetStream_v2(cublasHandle_t handle, cudaStream_t stream)
{
  FRONT_ENTER();
  cublasStatus_t rc;
  BLAS_LH(front_cublas_lib,
          rc = BLAS(cublasSetStream_v2)(front_translate(handle),
                                        front_real_stream(stream)));
  if (rc == CUBLAS_STATUS_SUCCESS) {
    Blas *b = blas_of(handle);
    std::lock_guard<std::mutex> guard(objs_lock);
    b->stream = stream;
    b->has_stream = true;
    b->has_workspace = false;
  }
  FRONT_LEAVE();
  return rc;
}

// The application's stream, not the lower half's.
extern "C" cublasStatus_t
cublasGetStream_v2(cublasHandle_t handle, cudaStream_t *stream)
{
  FRONT_ENTER();
  cudaStream_t real = NULL;
  cublasStatus_t rc;
  BLAS_LH(front_cublas_lib,
          rc = BLAS(cublasGetStream_v2)(front_translate(handle),
                                        stream != NULL ? &real : NULL));
  if (rc == CUBLAS_STATUS_SUCCESS && stream != NULL) {
    Blas *b = blas_of(handle);
    std::lock_guard<std::mutex> guard(objs_lock);
    *stream = b->has_stream ? b->stream : real;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cublasStatus_t
cublasSetWorkspace_v2(cublasHandle_t handle, void *workspace, size_t size)
{
  FRONT_ENTER();
  cublasStatus_t rc;
  BLAS_LH(front_cublas_lib,
          rc = BLAS(cublasSetWorkspace_v2)(front_translate(handle),
                                           workspace, size));
  if (rc == CUBLAS_STATUS_SUCCESS) {
    Blas *b = blas_of(handle);
    std::lock_guard<std::mutex> guard(objs_lock);
    b->has_workspace = true;
    b->workspace = workspace;
    b->workspace_size = size;
  }
  FRONT_LEAVE();
  return rc;
}

// A setting of a handle, which a restart sets again: the front's function,
// and apply_<setter>() for the restart.
// (The setter's type is spelled out: C++ overloads some, such as
// cublasSetEmulationStrategy.)
#define BLAS_SETTER(setter, type)                     \
  ((cublasStatus_t (*)(cublasHandle_t, type))         \
     front_cublas_lib.real[FRONT_IDX_cublas_##setter])
#define BLAS_SETTING(setter, type)                                        \
  static cublasStatus_t apply_##setter(cublasHandle_t real, intptr_t value) \
  {                                                                       \
    cublasStatus_t rc;                                                    \
    BLAS_LH(front_cublas_lib,                                             \
            rc = BLAS_SETTER(setter, type)(real, (type)value));           \
    return rc;                                                            \
  }                                                                       \
  extern "C" cublasStatus_t setter(cublasHandle_t handle, type value)     \
  {                                                                       \
    FRONT_ENTER();                                                        \
    cublasStatus_t rc;                                                    \
    BLAS_LH(front_cublas_lib,                                             \
            rc = BLAS_SETTER(setter, type)(front_translate(handle),       \
                                           value));                       \
    if (rc == CUBLAS_STATUS_SUCCESS) {                                    \
      note_setting(handle, apply_##setter, (intptr_t)value);              \
    }                                                                     \
    FRONT_LEAVE();                                                        \
    return rc;                                                            \
  }

BLAS_SETTING(cublasSetPointerMode_v2, cublasPointerMode_t)
BLAS_SETTING(cublasSetAtomicsMode, cublasAtomicsMode_t)
BLAS_SETTING(cublasSetMathMode, cublasMath_t)
BLAS_SETTING(cublasSetSmCountTarget, int)
// Floating-point emulation (newer cuBLAS).
#ifdef FRONT_IDX_cublas_cublasSetEmulationStrategy
BLAS_SETTING(cublasSetEmulationStrategy, cublasEmulationStrategy_t)
#endif
#ifdef FRONT_IDX_cublas_cublasSetEmulationSpecialValuesSupport
BLAS_SETTING(cublasSetEmulationSpecialValuesSupport,
             cudaEmulationSpecialValuesSupport)
#endif
#ifdef FRONT_IDX_cublas_cublasSetFixedPointEmulationMantissaControl
BLAS_SETTING(cublasSetFixedPointEmulationMantissaControl,
             cudaEmulationMantissaControl)
#endif
#ifdef FRONT_IDX_cublas_cublasSetFixedPointEmulationMaxMantissaBitCount
BLAS_SETTING(cublasSetFixedPointEmulationMaxMantissaBitCount, int)
#endif
#ifdef FRONT_IDX_cublas_cublasSetFixedPointEmulationMantissaBitOffset
BLAS_SETTING(cublasSetFixedPointEmulationMantissaBitOffset, int)
#endif
#ifdef FRONT_IDX_cublas_cublasSetFixedPointEmulationMantissaBitCountPointer
BLAS_SETTING(cublasSetFixedPointEmulationMantissaBitCountPointer, int *)
#endif

extern "C" const char *
cublasGetStatusName(cublasStatus_t status)
{
  return status_string(0, status, [status] {
    const char *s;
    BLAS_LH(front_cublas_lib, s = BLAS(cublasGetStatusName)(status));
    return s;
  });
}

extern "C" const char *
cublasGetStatusString(cublasStatus_t status)
{
  return status_string(1, status, [status] {
    const char *s;
    BLAS_LH(front_cublas_lib, s = BLAS(cublasGetStatusString)(status));
    return s;
  });
}

// ---- cuBLASLt ------------------------------------------------------------

extern "C" cublasStatus_t
cublasLtCreate(cublasLtHandle_t *handle)
{
  if (handle == NULL) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  FRONT_ENTER();
  front_before_library();
  BlasLt *lt = new BlasLt();
  cublasStatus_t rc;
  BLAS_LH(front_cublasLt_lib,
          rc = BLASLT(cublasLtCreate)((cublasLtHandle_t *)&lt->real));
  if (rc == CUBLAS_STATUS_SUCCESS) {
    lt->ctx = front_object_ctx();
    std::lock_guard<std::mutex> guard(objs_lock);
    obj_link(lt, FRONT_BLASLT);
    *handle = (cublasLtHandle_t)lt;
  } else {
    delete lt;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cublasStatus_t
cublasLtDestroy(cublasLtHandle_t handle)
{
  FRONT_ENTER();
  cublasStatus_t rc;
  BLAS_LH(front_cublasLt_lib,
          rc = BLASLT(cublasLtDestroy)(front_translate(handle)));
  if (rc == CUBLAS_STATUS_SUCCESS && handle != NULL) {
    std::lock_guard<std::mutex> guard(objs_lock);
    obj_unlink((BlasLt *)handle);
    delete (BlasLt *)handle;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" cublasStatus_t
cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *desc,
                         cublasComputeType_t compute_type,
                         cudaDataType_t scale_type)
{
  return make_descriptor(desc, BLASLT_INIT(cublasLtMatmulDescInit_internal,
                                           compute_type, scale_type));
}

extern "C" cublasStatus_t
cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *layout, cudaDataType type,
                           uint64_t rows, uint64_t cols, int64_t ld)
{
  return make_descriptor(layout,
                         BLASLT_INIT(cublasLtMatrixLayoutInit_internal, type,
                                     rows, cols, ld));
}

extern "C" cublasStatus_t
cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref)
{
  return make_descriptor(pref,
                         BLASLT_INIT(cublasLtMatmulPreferenceInit_internal));
}

extern "C" cublasStatus_t
cublasLtMatrixTransformDescCreate(cublasLtMatrixTransformDesc_t *desc,
                                  cudaDataType scale_type)
{
  return make_descriptor(
    desc, BLASLT_INIT(cublasLtMatrixTransformDescInit_internal, scale_type));
}

#ifdef FRONT_IDX_cublasLt_cublasLtEmulationDescCreate
extern "C" cublasStatus_t
cublasLtEmulationDescCreate(cublasLtEmulationDesc_t *desc)
{
  return make_descriptor(desc,
                         BLASLT_INIT(cublasLtEmulationDescInit_internal));
}
#endif

#ifdef FRONT_IDX_cublasLt_cublasLtGroupedMatrixLayoutCreate
extern "C" cublasStatus_t
cublasLtGroupedMatrixLayoutCreate(cublasLtMatrixLayout_t *layout,
                                  cudaDataType type, int group_count,
                                  const void *rows, const void *cols,
                                  const void *ld)
{
  return make_descriptor(
    layout, BLASLT_INIT(cublasLtGroupedMatrixLayoutInit_internal, type,
                        group_count, rows, cols, ld));
}
#endif

// The descriptors are the front's memory.
#define BLASLT_DESTROY(name, type) \
  extern "C" cublasStatus_t name(type desc) \
  {                                         \
    free(desc);                             \
    return CUBLAS_STATUS_SUCCESS;           \
  }

BLASLT_DESTROY(cublasLtMatmulDescDestroy, cublasLtMatmulDesc_t)
BLASLT_DESTROY(cublasLtMatrixLayoutDestroy, cublasLtMatrixLayout_t)
BLASLT_DESTROY(cublasLtMatmulPreferenceDestroy, cublasLtMatmulPreference_t)
BLASLT_DESTROY(cublasLtMatrixTransformDescDestroy,
               cublasLtMatrixTransformDesc_t)
#ifdef FRONT_IDX_cublasLt_cublasLtEmulationDescDestroy
BLASLT_DESTROY(cublasLtEmulationDescDestroy, cublasLtEmulationDesc_t)
#endif

extern "C" const char *
cublasLtGetStatusName(cublasStatus_t status)
{
  return status_string(2, status, [status] {
    const char *s;
    BLAS_LH(front_cublasLt_lib, s = BLASLT(cublasLtGetStatusName)(status));
    return s;
  });
}

extern "C" const char *
cublasLtGetStatusString(cublasStatus_t status)
{
  return status_string(3, status, [status] {
    const char *s;
    BLAS_LH(front_cublasLt_lib, s = BLASLT(cublasLtGetStatusString)(status));
    return s;
  });
}
