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

// CUDA arrays, mipmapped arrays, and texture and surface objects, in both
// APIs (cuda/DESIGN-lower-half-cuda.md, section 6).
//
// An array is a record, like a stream; its contents are saved at the
// checkpoint and copied back after the restart's replay, which makes the
// arrays again in creation order.  The copies whose structures hold arrays
// (cudaMemcpy3D, cuMemcpy2D, ...) translate them here.
//
// A texture or surface object is a 64-bit value that kernels get by value
// and may keep in device memory, so it must come back exactly.  The driver
// hands out the lowest free value of the context, from 1, textures and
// surfaces alike.  So at restart, in a new context, the objects are made
// in ascending order of their values, each with its own descriptors, at the
// point of the last one's creation in the restart's replay; an object that
// gets a lower value than its own is kept as a filler, and the fillers are
// destroyed once all are made.

#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include "front_objs.h"
#include "front_reg.h"

namespace {

enum ArrayApi {
  ARRAY_RUNTIME_2D,      // cudaMallocArray
  ARRAY_RUNTIME_3D,      // cudaMalloc3DArray
  ARRAY_DRIVER_2D,       // cuArrayCreate
  ARRAY_DRIVER_3D,       // cuArray3DCreate
  ARRAY_LEVEL,           // a level of a mipmapped array
};

struct Mipmap;

struct Array : FrontObj {
  Ctx *ctx;
  int api;
  cudaChannelFormatDesc desc;            // the runtime's
  cudaExtent extent;
  unsigned flags;
  CUDA_ARRAY_DESCRIPTOR desc2d;          // the driver's
  CUDA_ARRAY3D_DESCRIPTOR desc3d;
  Mipmap *mipmap;                        // ARRAY_LEVEL
  unsigned level;
  void *shadow;                          // contents during a checkpoint
};

struct Mipmap : FrontObj {
  Ctx *ctx;
  bool drv;
  cudaChannelFormatDesc desc;
  cudaExtent extent;
  unsigned levels;
  unsigned flags;
  CUDA_ARRAY3D_DESCRIPTOR desc3d;
  std::vector<Array *> level_records;    // that the application has
  std::vector<void *> shadows;           // per level, during a checkpoint
};

struct TexObj : FrontObj {
  Ctx *ctx;
  bool surface;
  bool drv;
  unsigned long long value;
  // With the application's (virtual) array handles.
  cudaResourceDesc res;
  cudaTextureDesc tex;
  cudaResourceViewDesc view;
  CUDA_RESOURCE_DESC dres;
  CUDA_TEXTURE_DESC dtex;
  CUDA_RESOURCE_VIEW_DESC dview;
  bool has_view;
};

unsigned texobjs_restored_gen;

Ctx *
current_ctx()
{
  Ctx *ctx = front_ctx_current();
  if (ctx == NULL) {
    int device = 0;
    FRONT_LH(REAL(cudaGetDevice)(&device));
    ctx = front_ctx_primary(device);
  }
  return ctx;
}

void *
real_array(const void *array)
{
  return front_real_obj(array, FRONT_ARRAY);
}

void *
real_mipmap(const void *mipmap)
{
  return front_real_obj(mipmap, FRONT_MIPMAP);
}

// The bytes of one element of the formats that a copy through the host can
// save; 0 for the others (block-compressed, NV12, packed, ...).
size_t
element_bytes(const CUDA_ARRAY3D_DESCRIPTOR &d)
{
  size_t channel;
  switch (d.Format) {
    case CU_AD_FORMAT_UNSIGNED_INT8:
    case CU_AD_FORMAT_SIGNED_INT8:
      channel = 1;
      break;
    case CU_AD_FORMAT_UNSIGNED_INT16:
    case CU_AD_FORMAT_SIGNED_INT16:
    case CU_AD_FORMAT_HALF:
      channel = 2;
      break;
    case CU_AD_FORMAT_UNSIGNED_INT32:
    case CU_AD_FORMAT_SIGNED_INT32:
    case CU_AD_FORMAT_FLOAT:
      channel = 4;
      break;
    default:
      return 0;
  }
  return channel * d.NumChannels;
}

CUresult
describe(CUarray real, CUDA_ARRAY3D_DESCRIPTOR *d)
{
  CUresult rc;
  FRONT_LH(rc = DRV(cuArray3DGetDescriptor_v2)(d, real));
  return rc;
}

size_t
array_bytes(const CUDA_ARRAY3D_DESCRIPTOR &d)
{
  return element_bytes(d) * d.Width * std::max<size_t>(d.Height, 1) *
         std::max<size_t>(d.Depth, 1);
}

// The whole array, to or from 'host' (array_bytes()).
void
copy_array(CUarray real, void *host, bool to_host)
{
  CUDA_ARRAY3D_DESCRIPTOR d;
  if (describe(real, &d) != CUDA_SUCCESS) {
    front_fatal("describing a CUDA array\n");
  }
  size_t row = element_bytes(d) * d.Width;
  CUDA_MEMCPY3D c;
  memset(&c, 0, sizeof c);
  if (to_host) {
    c.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    c.srcArray = real;
    c.dstMemoryType = CU_MEMORYTYPE_HOST;
    c.dstHost = host;
    c.dstPitch = row;
    c.dstHeight = std::max<size_t>(d.Height, 1);
  } else {
    c.srcMemoryType = CU_MEMORYTYPE_HOST;
    c.srcHost = host;
    c.srcPitch = row;
    c.srcHeight = std::max<size_t>(d.Height, 1);
    c.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    c.dstArray = real;
  }
  c.WidthInBytes = row;
  c.Height = std::max<size_t>(d.Height, 1);
  c.Depth = std::max<size_t>(d.Depth, 1);
  CUresult rc;
  FRONT_LH(rc = DRV(cuMemcpy3D_v2)(&c));
  if (rc != CUDA_SUCCESS) {
    front_fatal("copying a CUDA array %s the host: error %d\n",
                to_host ? "to" : "from", rc);
  }
}

// After a create: a format whose contents the checkpoint can save, and no
// sparse or deferred mapping.
bool
savable(CUarray real, unsigned flags)
{
  CUDA_ARRAY3D_DESCRIPTOR d;
  return (flags & (CUDA_ARRAY3D_SPARSE | CUDA_ARRAY3D_DEFERRED_MAPPING)) == 0 &&
         describe(real, &d) == CUDA_SUCCESS && element_bytes(d) != 0;
}

Array *
new_array(int api)
{
  Array *a = new Array();
  a->api = api;
  return a;
}

void
link_array(Array *a, void *real)
{
  a->real = real;
  std::lock_guard<std::mutex> guard(objs_lock);
  obj_link(a, FRONT_ARRAY);
}

// Arrays are not deleted when freed: a texture may still name one.
void
unlink_array(Array *a)
{
  std::lock_guard<std::mutex> guard(objs_lock);
  obj_unlink(a);
}

int
make_array(Array *a)
{
  int rc;
  void *real = NULL;
  switch (a->api) {
    case ARRAY_RUNTIME_2D:
      FRONT_LH(rc = REAL(cudaMallocArray)((cudaArray_t *)&real, &a->desc,
                                          a->extent.width, a->extent.height,
                                          a->flags));
      break;
    case ARRAY_RUNTIME_3D:
      FRONT_LH(rc = REAL(cudaMalloc3DArray)((cudaArray_t *)&real, &a->desc,
                                            a->extent, a->flags));
      break;
    case ARRAY_DRIVER_2D:
      FRONT_LH(rc = DRV(cuArrayCreate_v2)((CUarray *)&real, &a->desc2d));
      break;
    case ARRAY_DRIVER_3D:
      FRONT_LH(rc = DRV(cuArray3DCreate_v2)((CUarray *)&real, &a->desc3d));
      break;
    default:
      FRONT_LH(rc = DRV(cuMipmappedArrayGetLevel)(
                 (CUarray *)&real, (CUmipmappedArray)a->mipmap->real,
                 a->level));
  }
  if (rc == 0) {
    a->real = real;
  }
  return rc;
}

// cudaMallocArray, cudaMalloc3DArray, cuArrayCreate, cuArray3DCreate.
int
create_array(void **array, Array *a, bool drv)
{
  if (array == NULL) {
    delete a;
    return cudaErrorInvalidValue;
  }
  FRONT_ENTER();
  int rc = make_array(a);
  if (rc == 0) {
    unsigned flags = a->api == ARRAY_DRIVER_3D ? a->desc3d.Flags : a->flags;
    if (!savable((CUarray)a->real, flags)) {
      if (drv) {
        FRONT_LH(DRV(cuArrayDestroy)((CUarray)a->real));
      } else {
        FRONT_LH(REAL(cudaFreeArray)((cudaArray_t)a->real));
      }
      front_unsupported("CUDA arrays of this format, sparse or with "
                        "deferred mapping");
    }
    a->ctx = current_ctx();
    if (!drv) {
      front_note_cudart(a->ctx->device);
    }
    link_array(a, a->real);
    *array = a;
  } else {
    delete a;
  }
  FRONT_LEAVE();
  return rc;
}

int
free_array(void *array, bool drv)
{
  if (array == NULL) {
    return 0;
  }
  Array *a = (Array *)array;
  void *real = real_array(array);
  if (a->api == ARRAY_LEVEL) {
    return cudaErrorInvalidValue;  // freed with its mipmapped array
  }
  FRONT_ENTER();
  int rc;
  if (drv) {
    FRONT_LH(rc = DRV(cuArrayDestroy)((CUarray)real));
  } else {
    FRONT_LH(rc = REAL(cudaFreeArray)((cudaArray_t)real));
  }
  if (rc == 0) {
    unlink_array(a);
  }
  FRONT_LEAVE();
  return rc;
}

int
create_mipmap(void **mipmap, Mipmap *m)
{
  if (mipmap == NULL) {
    delete m;
    return cudaErrorInvalidValue;
  }
  if (m->drv ? (m->desc3d.Flags & (CUDA_ARRAY3D_SPARSE |
                                   CUDA_ARRAY3D_DEFERRED_MAPPING))
             : (m->flags & (cudaArraySparse | cudaArrayDeferredMapping))) {
    delete m;
    front_unsupported("sparse mipmapped arrays, or with deferred mapping");
  }
  FRONT_ENTER();
  int rc;
  void *real = NULL;
  if (m->drv) {
    FRONT_LH(rc = DRV(cuMipmappedArrayCreate)((CUmipmappedArray *)&real,
                                              &m->desc3d, m->levels));
  } else {
    FRONT_LH(rc = REAL(cudaMallocMipmappedArray)(
               (cudaMipmappedArray_t *)&real, &m->desc, m->extent, m->levels,
               m->flags));
  }
  if (rc == 0) {
    CUarray level0 = NULL;
    FRONT_LH(DRV(cuMipmappedArrayGetLevel)(&level0, (CUmipmappedArray)real,
                                           0));
    if (!savable(level0, 0)) {
      FRONT_LH(DRV(cuMipmappedArrayDestroy)((CUmipmappedArray)real));
      front_unsupported("mipmapped arrays of this format");
    }
    m->real = real;
    m->ctx = current_ctx();
    if (!m->drv) {
      front_note_cudart(m->ctx->device);
    }
    m->level_records.assign(m->levels, NULL);
    {
      std::lock_guard<std::mutex> guard(objs_lock);
      obj_link(m, FRONT_MIPMAP);
    }
    *mipmap = m;
  } else {
    delete m;
  }
  FRONT_LEAVE();
  return rc;
}

int
free_mipmap(void *mipmap, bool drv)
{
  if (mipmap == NULL) {
    return 0;
  }
  Mipmap *m = (Mipmap *)mipmap;
  void *real = real_mipmap(mipmap);
  FRONT_ENTER();
  int rc;
  if (drv) {
    FRONT_LH(rc = DRV(cuMipmappedArrayDestroy)((CUmipmappedArray)real));
  } else {
    FRONT_LH(rc = REAL(cudaFreeMipmappedArray)((cudaMipmappedArray_t)real));
  }
  if (rc == 0) {
    std::lock_guard<std::mutex> guard(objs_lock);
    for (Array *level : m->level_records) {
      if (level != NULL) {
        obj_unlink(level);
      }
    }
    obj_unlink(m);
  }
  FRONT_LEAVE();
  return rc;
}

// cudaGetMipmappedArrayLevel, cuMipmappedArrayGetLevel: a record per level,
// the same one each time.
int
get_level(void **array, const void *mipmap, unsigned level, bool drv)
{
  if (array == NULL) {
    return cudaErrorInvalidValue;
  }
  Mipmap *m = (Mipmap *)mipmap;
  void *real_m = real_mipmap(mipmap);
  FRONT_ENTER();
  int rc = 0;
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    if (level < m->level_records.size() && m->level_records[level] != NULL) {
      *array = m->level_records[level];
      FRONT_LEAVE();
      return 0;
    }
  }
  void *real = NULL;
  if (drv) {
    FRONT_LH(rc = DRV(cuMipmappedArrayGetLevel)(
               (CUarray *)&real, (CUmipmappedArray)real_m, level));
  } else {
    FRONT_LH(rc = REAL(cudaGetMipmappedArrayLevel)(
               (cudaArray_t *)&real, (cudaMipmappedArray_const_t)real_m,
               level));
  }
  if (rc == 0) {
    Array *a = new_array(ARRAY_LEVEL);
    a->ctx = m->ctx;
    a->mipmap = m;
    a->level = level;
    link_array(a, real);
    std::lock_guard<std::mutex> guard(objs_lock);
    m->level_records[level] = a;
    *array = a;
  }
  FRONT_LEAVE();
  return rc;
}

// The application's resource descriptions name its records.
void
real_resource(cudaResourceDesc *r)
{
  if (r->resType == cudaResourceTypeArray) {
    r->res.array.array = (cudaArray_t)real_array(r->res.array.array);
  } else if (r->resType == cudaResourceTypeMipmappedArray) {
    r->res.mipmap.mipmap =
      (cudaMipmappedArray_t)real_mipmap(r->res.mipmap.mipmap);
  }
}

void
real_resource(CUDA_RESOURCE_DESC *r)
{
  if (r->resType == CU_RESOURCE_TYPE_ARRAY) {
    r->res.array.hArray = (CUarray)real_array(r->res.array.hArray);
  } else if (r->resType == CU_RESOURCE_TYPE_MIPMAPPED_ARRAY) {
    r->res.mipmap.hMipmappedArray =
      (CUmipmappedArray)real_mipmap(r->res.mipmap.hMipmappedArray);
  }
}

// Makes the object of 't' in the current context; its value in *value.
int
make_texobj(TexObj *t, unsigned long long *value)
{
  int rc;
  if (t->drv) {
    CUDA_RESOURCE_DESC res = t->dres;
    real_resource(&res);
    if (t->surface) {
      FRONT_LH(rc = DRV(cuSurfObjectCreate)((CUsurfObject *)value, &res));
    } else {
      FRONT_LH(rc = DRV(cuTexObjectCreate)((CUtexObject *)value, &res,
                                           &t->dtex,
                                           t->has_view ? &t->dview : NULL));
    }
  } else {
    cudaResourceDesc res = t->res;
    real_resource(&res);
    if (t->surface) {
      FRONT_LH(rc = REAL(cudaCreateSurfaceObject)(
                 (cudaSurfaceObject_t *)value, &res));
    } else {
      FRONT_LH(rc = REAL(cudaCreateTextureObject)(
                 (cudaTextureObject_t *)value, &res, &t->tex,
                 t->has_view ? &t->view : NULL));
    }
  }
  return rc;
}

int
destroy_texobj(bool surface, bool drv, unsigned long long value)
{
  int rc;
  if (drv) {
    if (surface) {
      FRONT_LH(rc = DRV(cuSurfObjectDestroy)(value));
    } else {
      FRONT_LH(rc = DRV(cuTexObjectDestroy)(value));
    }
  } else if (surface) {
    FRONT_LH(rc = REAL(cudaDestroySurfaceObject)(value));
  } else {
    FRONT_LH(rc = REAL(cudaDestroyTextureObject)(value));
  }
  return rc;
}

int
create_texobj(unsigned long long *value, TexObj *t)
{
  if (value == NULL) {
    delete t;
    return cudaErrorInvalidValue;
  }
  FRONT_ENTER();
  int rc = make_texobj(t, &t->value);
  if (rc == 0) {
    t->ctx = current_ctx();
    t->real = (void *)(uintptr_t)t->value;
    std::lock_guard<std::mutex> guard(objs_lock);
    obj_link(t, FRONT_TEXOBJ);
    *value = t->value;
  } else {
    delete t;
  }
  FRONT_LEAVE();
  return rc;
}

// Locked.  The live object of this value in the current context.
TexObj *
find_texobj(unsigned long long value, Ctx *ctx)
{
  for (FrontObj *o = obj_heads[FRONT_TEXOBJ]; o != NULL; o = o->next) {
    TexObj *t = (TexObj *)o;
    if (t->value == value && t->ctx == ctx) {
      return t;
    }
  }
  return NULL;
}

int
destroy_texobj_of_app(unsigned long long value, bool surface, bool drv)
{
  FRONT_ENTER();
  Ctx *ctx = current_ctx();
  int rc = destroy_texobj(surface, drv, value);
  if (rc == 0) {
    std::lock_guard<std::mutex> guard(objs_lock);
    TexObj *t = find_texobj(value, ctx);
    if (t != NULL) {
      obj_unlink(t);
      delete t;
    }
  }
  FRONT_LEAVE();
  return rc;
}

// cudaGetTextureObjectResourceDesc ...: the description with the
// application's handles, as it gave it.
template <typename Desc>
int
resource_desc(Desc *out, unsigned long long value, bool drv)
{
  if (out == NULL) {
    return cudaErrorInvalidValue;
  }
  FRONT_ENTER();
  Ctx *ctx = current_ctx();
  int rc = cudaErrorInvalidValue;
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    TexObj *t = find_texobj(value, ctx);
    if (t != NULL && t->drv == drv) {
      memcpy(out, drv ? (const void *)&t->dres : (const void *)&t->res,
             sizeof *out);
      rc = 0;
    }
  }
  FRONT_LEAVE();
  return rc;
}

// All the texture and surface objects, with their values (see the top).
void
restore_texobjs()
{
  std::map<Ctx *, std::vector<TexObj *> > by_ctx;
  for (FrontObj *o = obj_heads[FRONT_TEXOBJ]; o != NULL; o = o->next) {
    by_ctx[((TexObj *)o)->ctx].push_back((TexObj *)o);
  }
  for (auto &entry : by_ctx) {
    std::vector<TexObj *> &objs = entry.second;
    std::sort(objs.begin(), objs.end(),
              [](TexObj *a, TexObj *b) { return a->value < b->value; });
    front_ctx_use(entry.first);
    std::vector<TexObj> fillers;
    for (TexObj *t : objs) {
      for (;;) {
        unsigned long long value = 0;
        int rc = make_texobj(t, &value);
        if (rc != 0) {
          front_fatal("making a %s object again: error %d\n",
                      t->surface ? "surface" : "texture", rc);
        }
        if (value == t->value) {
          break;
        }
        if (value > t->value) {
          front_fatal("the %s object %llu came back as %llu\n",
                      t->surface ? "surface" : "texture", t->value, value);
        }
        TexObj filler = *t;
        filler.value = value;
        fillers.push_back(filler);
      }
    }
    for (TexObj &f : fillers) {
      destroy_texobj(f.surface, f.drv, f.value);
    }
    front_log("restored %zu texture and surface objects, with %zu fillers\n",
              objs.size(), fillers.size());
  }
}

// A runtime or driver copy structure, with real arrays.
cudaMemcpy3DParms
real_copy(const cudaMemcpy3DParms *p)
{
  cudaMemcpy3DParms c = *p;
  c.srcArray = (cudaArray_t)real_array(c.srcArray);
  c.dstArray = (cudaArray_t)real_array(c.dstArray);
  return c;
}

cudaMemcpy3DPeerParms
real_copy(const cudaMemcpy3DPeerParms *p)
{
  cudaMemcpy3DPeerParms c = *p;
  c.srcArray = (cudaArray_t)real_array(c.srcArray);
  c.dstArray = (cudaArray_t)real_array(c.dstArray);
  return c;
}

CUDA_MEMCPY2D
real_copy(const CUDA_MEMCPY2D *p)
{
  CUDA_MEMCPY2D c = *p;
  if (c.srcMemoryType == CU_MEMORYTYPE_ARRAY) {
    c.srcArray = (CUarray)real_array(c.srcArray);
  }
  if (c.dstMemoryType == CU_MEMORYTYPE_ARRAY) {
    c.dstArray = (CUarray)real_array(c.dstArray);
  }
  return c;
}

CUDA_MEMCPY3D
real_copy(const CUDA_MEMCPY3D *p)
{
  CUDA_MEMCPY3D c = *p;
  if (c.srcMemoryType == CU_MEMORYTYPE_ARRAY) {
    c.srcArray = (CUarray)real_array(c.srcArray);
  }
  if (c.dstMemoryType == CU_MEMORYTYPE_ARRAY) {
    c.dstArray = (CUarray)real_array(c.dstArray);
  }
  return c;
}

CUDA_MEMCPY3D_PEER
real_copy(const CUDA_MEMCPY3D_PEER *p)
{
  CUDA_MEMCPY3D_PEER c = *p;
  if (c.srcMemoryType == CU_MEMORYTYPE_ARRAY) {
    c.srcArray = (CUarray)real_array(c.srcArray);
  }
  if (c.dstMemoryType == CU_MEMORYTYPE_ARRAY) {
    c.dstArray = (CUarray)real_array(c.dstArray);
  }
  c.srcContext = (CUcontext)front_real_obj(c.srcContext, FRONT_CTX);
  c.dstContext = (CUcontext)front_real_obj(c.dstContext, FRONT_CTX);
  return c;
}

// Whether a copy writes device memory, which the MPI may read.
bool
to_device(const cudaMemcpy3DParms *p)
{
  return p->dstArray == NULL && p->kind != cudaMemcpyDeviceToHost &&
         p->kind != cudaMemcpyHostToHost;
}

bool
to_device(const cudaMemcpy3DPeerParms *p)
{
  return p->dstArray == NULL;
}

template <typename Parms>
bool
to_device(const Parms *p)
{
  return p->dstMemoryType == CU_MEMORYTYPE_DEVICE ||
         p->dstMemoryType == CU_MEMORYTYPE_UNIFIED;
}

// A copy with a structure: 'call' gets the structure with real handles.  A
// synchronous one into device memory is ordered with the next MPI call
// (front_memops.cpp).
template <typename Parms, typename Call>
int
copy(const Parms *p, bool sync, bool ptds, Call call)
{
  if (p == NULL) {
    return cudaErrorInvalidValue;
  }
  FRONT_ENTER();
  Parms real = real_copy(p);
  int rc = call(&real);
  if (rc == 0 && sync && to_device(p)) {
    front_order_memop(ptds);
  }
  FRONT_LEAVE();
  return rc;
}

}  // namespace

// ---- Checkpoint and restart (front_objs.cpp) -----------------------------

// cuCtxDestroy, with objs_lock held: the context's arrays and objects are
// gone with it.
void
front_arrays_forget(Ctx *ctx)
{
  for (int kind : { FRONT_TEXOBJ, FRONT_ARRAY, FRONT_MIPMAP }) {
    for (FrontObj *o = obj_heads[kind], *next; o != NULL; o = next) {
      next = o->next;
      Ctx *of = kind == FRONT_ARRAY ? ((Array *)o)->ctx
                : kind == FRONT_MIPMAP ? ((Mipmap *)o)->ctx
                : ((TexObj *)o)->ctx;
      if (of == ctx) {
        obj_unlink(o);
      }
    }
  }
}

void
front_arrays_stash()
{
  for (FrontObj *o = obj_heads[FRONT_ARRAY]; o != NULL; o = o->next) {
    Array *a = (Array *)o;
    if (a->api == ARRAY_LEVEL) {
      continue;  // with its mipmapped array
    }
    CUDA_ARRAY3D_DESCRIPTOR d;
    front_ctx_use(a->ctx);
    describe((CUarray)a->real, &d);
    a->shadow = malloc(array_bytes(d));
    copy_array((CUarray)a->real, a->shadow, true);
  }
  for (FrontObj *o = obj_heads[FRONT_MIPMAP]; o != NULL; o = o->next) {
    Mipmap *m = (Mipmap *)o;
    front_ctx_use(m->ctx);
    m->shadows.assign(m->levels, NULL);
    for (unsigned i = 0; i < m->levels; i++) {
      CUarray level = NULL;
      CUDA_ARRAY3D_DESCRIPTOR d;
      FRONT_LH(DRV(cuMipmappedArrayGetLevel)(&level,
                                             (CUmipmappedArray)m->real, i));
      describe(level, &d);
      m->shadows[i] = malloc(array_bytes(d));
      copy_array(level, m->shadows[i], true);
    }
  }
}

void
front_arrays_drop_shadows()
{
  for (FrontObj *o = obj_heads[FRONT_ARRAY]; o != NULL; o = o->next) {
    free(((Array *)o)->shadow);
    ((Array *)o)->shadow = NULL;
  }
  for (FrontObj *o = obj_heads[FRONT_MIPMAP]; o != NULL; o = o->next) {
    for (void *shadow : ((Mipmap *)o)->shadows) {
      free(shadow);
    }
    ((Mipmap *)o)->shadows.clear();
  }
}

// In the restart's replay, in creation order.
void
front_arrays_restore(FrontObj *obj)
{
  int rc = 0;
  switch (obj->kind) {
    case FRONT_ARRAY: {
      Array *a = (Array *)obj;
      front_ctx_use(a->ctx);
      rc = make_array(a);
      break;
    }
    case FRONT_MIPMAP: {
      Mipmap *m = (Mipmap *)obj;
      front_ctx_use(m->ctx);
      void *real = NULL;
      if (m->drv) {
        FRONT_LH(rc = DRV(cuMipmappedArrayCreate)((CUmipmappedArray *)&real,
                                                  &m->desc3d, m->levels));
      } else {
        FRONT_LH(rc = REAL(cudaMallocMipmappedArray)(
                   (cudaMipmappedArray_t *)&real, &m->desc, m->extent,
                   m->levels, m->flags));
      }
      m->real = real;
      break;
    }
    case FRONT_TEXOBJ: {
      // All of them, where the last one was made: then every array that one
      // of them names exists again.
      uint64_t last = 0;
      for (FrontObj *o = obj_heads[FRONT_TEXOBJ]; o != NULL; o = o->next) {
        last = std::max(last, o->seq);
      }
      if (obj->seq == last && texobjs_restored_gen != front_gen) {
        restore_texobjs();
        texobjs_restored_gen = front_gen;
      }
      break;
    }
  }
  if (rc != 0) {
    front_fatal("making a CUDA array again: error %d\n", rc);
  }
}

// After the replay: the contents.
void
front_arrays_fill()
{
  for (FrontObj *o = obj_heads[FRONT_ARRAY]; o != NULL; o = o->next) {
    Array *a = (Array *)o;
    if (a->shadow != NULL) {
      front_ctx_use(a->ctx);
      copy_array((CUarray)a->real, a->shadow, false);
    }
  }
  for (FrontObj *o = obj_heads[FRONT_MIPMAP]; o != NULL; o = o->next) {
    Mipmap *m = (Mipmap *)o;
    front_ctx_use(m->ctx);
    for (unsigned i = 0; i < m->shadows.size(); i++) {
      CUarray level = NULL;
      FRONT_LH(DRV(cuMipmappedArrayGetLevel)(&level,
                                             (CUmipmappedArray)m->real, i));
      copy_array(level, m->shadows[i], false);
    }
  }
}

// ---- Arrays --------------------------------------------------------------

extern "C" cudaError_t
cudaMallocArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc,
                size_t width, size_t height, unsigned int flags)
{
  if (desc == NULL) {
    return cudaErrorInvalidValue;
  }
  Array *a = new_array(ARRAY_RUNTIME_2D);
  a->desc = *desc;
  a->extent = { width, height, 0 };
  a->flags = flags;
  return (cudaError_t)create_array((void **)array, a, false);
}

extern "C" cudaError_t
cudaMalloc3DArray(cudaArray_t *array, const struct cudaChannelFormatDesc *desc,
                  struct cudaExtent extent, unsigned int flags)
{
  if (desc == NULL) {
    return cudaErrorInvalidValue;
  }
  Array *a = new_array(ARRAY_RUNTIME_3D);
  a->desc = *desc;
  a->extent = extent;
  a->flags = flags;
  return (cudaError_t)create_array((void **)array, a, false);
}

extern "C" cudaError_t
cudaFreeArray(cudaArray_t array)
{
  return (cudaError_t)free_array(array, false);
}

extern "C" CUresult
cuArrayCreate_v2(CUarray *pHandle,
                 const CUDA_ARRAY_DESCRIPTOR *pAllocateArray)
{
  if (pAllocateArray == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  Array *a = new_array(ARRAY_DRIVER_2D);
  a->desc2d = *pAllocateArray;
  return (CUresult)create_array((void **)pHandle, a, true);
}

extern "C" CUresult
cuArray3DCreate_v2(CUarray *pHandle,
                   const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray)
{
  if (pAllocateArray == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  Array *a = new_array(ARRAY_DRIVER_3D);
  a->desc3d = *pAllocateArray;
  return (CUresult)create_array((void **)pHandle, a, true);
}

extern "C" CUresult
cuArrayDestroy(CUarray hArray)
{
  return (CUresult)free_array(hArray, true);
}

extern "C" cudaError_t
cudaMallocMipmappedArray(cudaMipmappedArray_t *mipmappedArray,
                         const struct cudaChannelFormatDesc *desc,
                         struct cudaExtent extent, unsigned int numLevels,
                         unsigned int flags)
{
  if (desc == NULL) {
    return cudaErrorInvalidValue;
  }
  Mipmap *m = new Mipmap();
  m->drv = false;
  m->desc = *desc;
  m->extent = extent;
  m->levels = numLevels;
  m->flags = flags;
  return (cudaError_t)create_mipmap((void **)mipmappedArray, m);
}

extern "C" cudaError_t
cudaFreeMipmappedArray(cudaMipmappedArray_t mipmappedArray)
{
  return (cudaError_t)free_mipmap(mipmappedArray, false);
}

extern "C" cudaError_t
cudaGetMipmappedArrayLevel(cudaArray_t *levelArray,
                           cudaMipmappedArray_const_t mipmappedArray,
                           unsigned int level)
{
  return (cudaError_t)get_level((void **)levelArray, mipmappedArray, level,
                                false);
}

extern "C" CUresult
cuMipmappedArrayCreate(CUmipmappedArray *pHandle,
                       const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                       unsigned int numMipmapLevels)
{
  if (pMipmappedArrayDesc == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  Mipmap *m = new Mipmap();
  m->drv = true;
  m->desc3d = *pMipmappedArrayDesc;
  m->levels = numMipmapLevels;
  return (CUresult)create_mipmap((void **)pHandle, m);
}

extern "C" CUresult
cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray)
{
  return (CUresult)free_mipmap(hMipmappedArray, true);
}

extern "C" CUresult
cuMipmappedArrayGetLevel(CUarray *pLevelArray,
                         CUmipmappedArray hMipmappedArray, unsigned int level)
{
  return (CUresult)get_level((void **)pLevelArray, hMipmappedArray, level,
                             true);
}

// ---- Copies whose structures hold arrays ---------------------------------

extern "C" cudaError_t
cudaMemcpy3D(const struct cudaMemcpy3DParms *p)
{
  return (cudaError_t)copy(p, true, false, [](cudaMemcpy3DParms *c) {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaMemcpy3D)(c));
    return (int)rc;
  });
}

extern "C" cudaError_t
cudaMemcpy3D_ptds(const struct cudaMemcpy3DParms *p)
{
  return (cudaError_t)copy(p, true, true, [](cudaMemcpy3DParms *c) {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaMemcpy3D_ptds)(c));
    return (int)rc;
  });
}

extern "C" cudaError_t
cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream)
{
  cudaStream_t real = front_real_stream(stream);
  return (cudaError_t)copy(p, false, false, [real](cudaMemcpy3DParms *c) {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaMemcpy3DAsync)(c, real));
    return (int)rc;
  });
}

extern "C" cudaError_t
cudaMemcpy3DAsync_ptsz(const struct cudaMemcpy3DParms *p, cudaStream_t stream)
{
  cudaStream_t real = front_real_stream(stream);
  return (cudaError_t)copy(p, false, false, [real](cudaMemcpy3DParms *c) {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaMemcpy3DAsync_ptsz)(c, real));
    return (int)rc;
  });
}

extern "C" cudaError_t
cudaMemcpy3DPeer(const struct cudaMemcpy3DPeerParms *p)
{
  return (cudaError_t)copy(p, true, false, [](cudaMemcpy3DPeerParms *c) {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaMemcpy3DPeer)(c));
    return (int)rc;
  });
}

extern "C" cudaError_t
cudaMemcpy3DPeer_ptds(const struct cudaMemcpy3DPeerParms *p)
{
  return (cudaError_t)copy(p, true, true, [](cudaMemcpy3DPeerParms *c) {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaMemcpy3DPeer_ptds)(c));
    return (int)rc;
  });
}

extern "C" cudaError_t
cudaMemcpy3DPeerAsync(const struct cudaMemcpy3DPeerParms *p,
                      cudaStream_t stream)
{
  cudaStream_t real = front_real_stream(stream);
  return (cudaError_t)copy(p, false, false, [real](cudaMemcpy3DPeerParms *c) {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaMemcpy3DPeerAsync)(c, real));
    return (int)rc;
  });
}

extern "C" cudaError_t
cudaMemcpy3DPeerAsync_ptsz(const struct cudaMemcpy3DPeerParms *p,
                           cudaStream_t stream)
{
  cudaStream_t real = front_real_stream(stream);
  return (cudaError_t)copy(p, false, false, [real](cudaMemcpy3DPeerParms *c) {
    cudaError_t rc;
    FRONT_LH(rc = REAL(cudaMemcpy3DPeerAsync_ptsz)(c, real));
    return (int)rc;
  });
}

// The driver's, with _ptds/_ptsz forms.
#define DRIVER_COPY(name, Parms, sync, ptds)                               \
  extern "C" CUresult name(const Parms *pCopy)                             \
  {                                                                        \
    return (CUresult)copy(pCopy, sync, ptds, [](Parms *c) {                \
      CUresult rc;                                                         \
      FRONT_LH(rc = DRV(name)(c));                                         \
      return (int)rc;                                                      \
    });                                                                    \
  }
#define DRIVER_COPY_ASYNC(name, Parms)                                     \
  extern "C" CUresult name(const Parms *pCopy, CUstream hStream)           \
  {                                                                        \
    CUstream real = front_real_stream(hStream);                            \
    return (CUresult)copy(pCopy, false, false, [real](Parms *c) {          \
      CUresult rc;                                                         \
      FRONT_LH(rc = DRV(name)(c, real));                                   \
      return (int)rc;                                                      \
    });                                                                    \
  }

DRIVER_COPY(cuMemcpy2D_v2, CUDA_MEMCPY2D, true, false)
DRIVER_COPY(cuMemcpy2D_v2_ptds, CUDA_MEMCPY2D, true, true)
DRIVER_COPY(cuMemcpy2DUnaligned_v2, CUDA_MEMCPY2D, true, false)
DRIVER_COPY(cuMemcpy2DUnaligned_v2_ptds, CUDA_MEMCPY2D, true, true)
DRIVER_COPY(cuMemcpy3D_v2, CUDA_MEMCPY3D, true, false)
DRIVER_COPY(cuMemcpy3D_v2_ptds, CUDA_MEMCPY3D, true, true)
DRIVER_COPY(cuMemcpy3DPeer, CUDA_MEMCPY3D_PEER, true, false)
DRIVER_COPY(cuMemcpy3DPeer_ptds, CUDA_MEMCPY3D_PEER, true, true)
DRIVER_COPY_ASYNC(cuMemcpy2DAsync_v2, CUDA_MEMCPY2D)
DRIVER_COPY_ASYNC(cuMemcpy2DAsync_v2_ptsz, CUDA_MEMCPY2D)
DRIVER_COPY_ASYNC(cuMemcpy3DAsync_v2, CUDA_MEMCPY3D)
DRIVER_COPY_ASYNC(cuMemcpy3DAsync_v2_ptsz, CUDA_MEMCPY3D)
DRIVER_COPY_ASYNC(cuMemcpy3DPeerAsync, CUDA_MEMCPY3D_PEER)
DRIVER_COPY_ASYNC(cuMemcpy3DPeerAsync_ptsz, CUDA_MEMCPY3D_PEER)

// ---- Texture and surface objects -----------------------------------------

extern "C" cudaError_t
cudaCreateTextureObject(cudaTextureObject_t *pTexObject,
                        const struct cudaResourceDesc *pResDesc,
                        const struct cudaTextureDesc *pTexDesc,
                        const struct cudaResourceViewDesc *pResViewDesc)
{
  if (pResDesc == NULL || pTexDesc == NULL) {
    return cudaErrorInvalidValue;
  }
  TexObj *t = new TexObj();
  t->res = *pResDesc;
  t->tex = *pTexDesc;
  t->has_view = pResViewDesc != NULL;
  if (t->has_view) {
    t->view = *pResViewDesc;
  }
  return (cudaError_t)create_texobj(pTexObject, t);
}

extern "C" cudaError_t
cudaDestroyTextureObject(cudaTextureObject_t texObject)
{
  return (cudaError_t)destroy_texobj_of_app(texObject, false, false);
}

extern "C" cudaError_t
cudaCreateSurfaceObject(cudaSurfaceObject_t *pSurfObject,
                        const struct cudaResourceDesc *pResDesc)
{
  if (pResDesc == NULL) {
    return cudaErrorInvalidValue;
  }
  TexObj *t = new TexObj();
  t->surface = true;
  t->res = *pResDesc;
  return (cudaError_t)create_texobj(pSurfObject, t);
}

extern "C" cudaError_t
cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject)
{
  return (cudaError_t)destroy_texobj_of_app(surfObject, true, false);
}

extern "C" cudaError_t
cudaGetTextureObjectResourceDesc(struct cudaResourceDesc *pResDesc,
                                 cudaTextureObject_t texObject)
{
  return (cudaError_t)resource_desc(pResDesc, texObject, false);
}

extern "C" cudaError_t
cudaGetSurfaceObjectResourceDesc(struct cudaResourceDesc *pResDesc,
                                 cudaSurfaceObject_t surfObject)
{
  return cudaGetTextureObjectResourceDesc(pResDesc, surfObject);
}

extern "C" CUresult
cuTexObjectCreate(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc,
                  const CUDA_TEXTURE_DESC *pTexDesc,
                  const CUDA_RESOURCE_VIEW_DESC *pResViewDesc)
{
  if (pResDesc == NULL || pTexDesc == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  TexObj *t = new TexObj();
  t->drv = true;
  t->dres = *pResDesc;
  t->dtex = *pTexDesc;
  t->has_view = pResViewDesc != NULL;
  if (t->has_view) {
    t->dview = *pResViewDesc;
  }
  return (CUresult)create_texobj(pTexObject, t);
}

extern "C" CUresult
cuTexObjectDestroy(CUtexObject texObject)
{
  return (CUresult)destroy_texobj_of_app(texObject, false, true);
}

extern "C" CUresult
cuSurfObjectCreate(CUsurfObject *pSurfObject,
                   const CUDA_RESOURCE_DESC *pResDesc)
{
  if (pResDesc == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  TexObj *t = new TexObj();
  t->surface = true;
  t->drv = true;
  t->dres = *pResDesc;
  return (CUresult)create_texobj(pSurfObject, t);
}

extern "C" CUresult
cuSurfObjectDestroy(CUsurfObject surfObject)
{
  return (CUresult)destroy_texobj_of_app(surfObject, true, true);
}

extern "C" CUresult
cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                           CUtexObject texObject)
{
  return (CUresult)resource_desc(pResDesc, texObject, true);
}

extern "C" CUresult
cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                            CUsurfObject surfObject)
{
  return cuTexObjectGetResourceDesc(pResDesc, surfObject);
}
