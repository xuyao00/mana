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

// The driver API's own functions ("own" in cuda-own.txt): contexts, which
// both APIs share, driver modules and functions, memory, and the entry-point
// lookups.
//
// A context is a record (Ctx).  A device's primary context has one record,
// which the front holds a reference of; cudart makes it current by itself,
// so a real context that the front has no record of is looked up as a
// primary one.  Each thread's context stack is copied in its FrontThread, so
// that its first call after a restart makes the stack again.
//
// A module keeps a copy of its image and of the JIT options that have no
// pointers.  Its globals that the application looked up are saved at the
// checkpoint, and must come back at the same addresses.

#include <elf.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include "front_objs.h"
#include "front_reg.h"

extern "C" void *const front_cuda_dispatch[];

namespace {

struct Global {
  std::string name;
  CUdeviceptr dptr;
  size_t bytes;
  void *shadow;
};

struct Module : FrontObj {
  Ctx *ctx;
  std::string image;
  std::vector<CUjit_option> options;     // those without pointers
  std::vector<void *> values;
  std::vector<Global> globals;           // that the application looked up
};

struct Function : FrontObj {
  Module *module;
  std::string name;
  std::map<int, int> attrs;              // cuFuncSetAttribute
  int cache_config;                      // -1: never set
};

Ctx *primaries[64];
std::map<void *, Ctx *> ctx_by_real;     // of this generation
std::map<int, std::string> error_strings;
std::map<void *, void *> entry_points;   // real function -> the front's
unsigned entry_points_gen;

#define DRV_CALL(rc, call) FRONT_LH(rc = (call))


Ctx *
new_ctx(int device, bool primary, unsigned flags)
{
  Ctx *ctx = new Ctx();
  ctx->device = device;
  ctx->primary = primary;
  ctx->destroyed = false;
  ctx->flags = flags;
  ctx->retains = 0;
  ctx->cache_config = -1;
  return ctx;
}

Ctx *
ctx_of(CUcontext ctx)
{
  return (Ctx *)ctx;
}

// The record of a real context of this generation; primary contexts that
// cudart made current are found by device.  NULL if there is none and not
// 'must'.
Ctx *
ctx_of_real(CUcontext real, bool must = true)
{
  if (real == NULL) {
    return NULL;
  }
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    auto it = ctx_by_real.find(real);
    if (it != ctx_by_real.end()) {
      return it->second;
    }
  }
  CUdevice device = 0;
  CUresult rc;
  DRV_CALL(rc, DRV(cuCtxGetDevice_v2)(&device, real));
  if (rc == CUDA_SUCCESS) {
    Ctx *ctx = front_ctx_primary(device);
    if (ctx->real == real) {
      return ctx;
    }
  }
  if (must) {
    front_fatal("context %p is not one that the application made\n",
                (void *)real);
  }
  return NULL;
}

// Objects of a destroyed context are gone with it.
void
drop_objects_of(Ctx *ctx)
{
  for (int kind : { FRONT_STREAM, FRONT_EVENT }) {
    for (FrontObj *o = obj_heads[kind], *next; o != NULL; o = next) {
      next = o->next;
      Ctx *of = kind == FRONT_STREAM ? ((Stream *)o)->ctx : ((Event *)o)->ctx;
      if (of == ctx) {
        obj_unlink(o);
      }
    }
  }
  for (FrontObj *o = obj_heads[FRONT_FUNCTION], *next; o != NULL; o = next) {
    next = o->next;
    if (((Function *)o)->module->ctx == ctx) {
      obj_unlink(o);
    }
  }
  for (FrontObj *o = obj_heads[FRONT_MODULE], *next; o != NULL; o = next) {
    next = o->next;
    if (((Module *)o)->ctx == ctx) {
      obj_unlink(o);
    }
  }
  for (auto it = pinned.begin(); it != pinned.end();) {
    it = it->second.ctx == ctx ? pinned.erase(it) : std::next(it);
  }
  front_arrays_forget(ctx);
  front_libs_forget(ctx);
}

// The size of a module image: a cubin (ELF), a fatbin, or PTX text.
size_t
image_size(const void *image)
{
  const unsigned char *p = (const unsigned char *)image;
  if (memcmp(p, ELFMAG, SELFMAG) == 0) {
    const Elf64_Ehdr *eh = (const Elf64_Ehdr *)p;
    size_t end = std::max(eh->e_shoff + (size_t)eh->e_shnum * eh->e_shentsize,
                          eh->e_phoff + (size_t)eh->e_phnum * eh->e_phentsize);
    const Elf64_Shdr *sh = (const Elf64_Shdr *)(p + eh->e_shoff);
    for (int i = 0; i < eh->e_shnum; i++) {
      if (sh[i].sh_type != SHT_NOBITS) {
        end = std::max(end, (size_t)(sh[i].sh_offset + sh[i].sh_size));
      }
    }
    return end;
  }
  uint32_t magic;
  memcpy(&magic, p, sizeof magic);
  if (magic == 0xba55ed50) {  // fatbin: magic, version, header size, size
    uint16_t header_size;
    uint64_t fat_size;
    memcpy(&header_size, p + 6, sizeof header_size);
    memcpy(&fat_size, p + 8, sizeof fat_size);
    return header_size + fat_size;
  }
  return strlen((const char *)p) + 1;
}

// nvcc's __fatBinC_Wrapper_t, or the image itself.
const void *
unwrap_fatbin(const void *image)
{
  struct Wrapper {
    int magic;
    int version;
    const unsigned long long *data;
    void *filename_or_fatbins;
  };
  const Wrapper *w = (const Wrapper *)image;
  return w->magic == 0x466243b1 ? (const void *)w->data : image;
}

// JIT options whose values are not pointers into the application, nor
// results: these are given again at restart.
bool
replayable_option(CUjit_option option)
{
  switch (option) {
    case CU_JIT_MAX_REGISTERS:
    case CU_JIT_THREADS_PER_BLOCK:
    case CU_JIT_OPTIMIZATION_LEVEL:
    case CU_JIT_TARGET_FROM_CUCONTEXT:
    case CU_JIT_TARGET:
    case CU_JIT_FALLBACK_STRATEGY:
    case CU_JIT_GENERATE_DEBUG_INFO:
    case CU_JIT_LOG_VERBOSE:
    case CU_JIT_GENERATE_LINE_INFO:
    case CU_JIT_CACHE_MODE:
    case CU_JIT_POSITION_INDEPENDENT_CODE:
    case CU_JIT_MIN_CTA_PER_SM:
    case CU_JIT_MAX_THREADS_PER_BLOCK:
    case CU_JIT_OVERRIDE_DIRECTIVE_VALUES:
      return true;
    default:
      return false;
  }
}

CUresult
load_module(CUmodule *module, const void *image, size_t size,
            unsigned n_options, CUjit_option *options, void **values)
{
  if (module == NULL || image == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  FRONT_ENTER();
  Module *m = new Module();
  CUresult rc;
  DRV_CALL(rc, DRV(cuModuleLoadDataEx)((CUmodule *)&m->real, image,
                                       n_options, options, values));
  if (rc == CUDA_SUCCESS) {
    m->image.assign((const char *)image, size);
    for (unsigned i = 0; i < n_options; i++) {
      if (replayable_option(options[i])) {
        m->options.push_back(options[i]);
        m->values.push_back(values[i]);
      }
    }
    m->ctx = front_ctx_current();
    std::lock_guard<std::mutex> guard(objs_lock);
    obj_link(m, FRONT_MODULE);
    *module = (CUmodule)m;
  } else {
    delete m;
  }
  FRONT_LEAVE();
  return rc;
}

Module *
module_of(CUmodule module)
{
  front_real_obj(module, FRONT_MODULE);  // checks it
  return (Module *)module;
}

Function *
function_of(CUfunction function)
{
  front_real_obj(function, FRONT_FUNCTION);
  return (Function *)function;
}

const char *
error_string(CUresult error, bool name)
{
  int key = 2 * (int)error + (name ? 1 : 0);
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    auto it = error_strings.find(key);
    if (it != error_strings.end()) {
      return it->second.c_str();
    }
  }
  const char *s = NULL;
  CUresult rc;
  if (name) {
    DRV_CALL(rc, DRV(cuGetErrorName)(error, &s));
  } else {
    DRV_CALL(rc, DRV(cuGetErrorString)(error, &s));
  }
  if (rc != CUDA_SUCCESS || s == NULL) {
    return NULL;
  }
  std::lock_guard<std::mutex> guard(objs_lock);
  return error_strings.emplace(key, s).first->second.c_str();
}

// The front's function for a real driver function, or NULL.
void *
front_entry_point(void *real)
{
  front_check_bound(&front_cuda_lib);
  std::lock_guard<std::mutex> guard(objs_lock);
  if (entry_points_gen != front_gen) {
    entry_points.clear();
    for (int i = 0; i < front_cuda_lib.n; i++) {
      if (front_cuda_lib.real[i] != NULL) {
        entry_points.emplace(front_cuda_lib.real[i], front_cuda_dispatch[i]);
      }
    }
    entry_points_gen = front_gen;
  }
  auto it = entry_points.find(real);
  return it != entry_points.end() ? it->second : NULL;
}

}  // namespace

Ctx *
front_ctx_of_real(CUcontext real)
{
  return ctx_of_real(real, false);
}

// ---- Contexts, shared with the runtime -----------------------------------

Ctx *
front_ctx_primary(int device)
{
  if (device < 0 || device >= 64) {
    front_fatal("device %d is out of range\n", device);
  }
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    Ctx *ctx = primaries[device];
    if (ctx != NULL && ctx->real != NULL) {
      return ctx;
    }
  }
  // The front's own reference: the context lives as long as the record.
  CUcontext real = NULL;
  CUresult rc;
  DRV_CALL(rc, DRV(cuDevicePrimaryCtxRetain)(&real, device));
  if (rc != CUDA_SUCCESS) {
    front_fatal("retaining the primary context of device %d: error %d\n",
                device, rc);
  }
  bool extra = false;
  Ctx *ctx;
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    ctx = primaries[device];
    if (ctx == NULL) {
      ctx = primaries[device] = new_ctx(device, true, 0);
      obj_link(ctx, FRONT_CTX);
    }
    if (ctx->real == NULL) {
      ctx->real = real;
      ctx_by_real[real] = ctx;
    } else {
      extra = true;  // another thread was first
    }
  }
  if (extra) {
    DRV_CALL(rc, DRV(cuDevicePrimaryCtxRelease_v2)(device));
  }
  return ctx;
}

Ctx *
front_ctx_current()
{
  CUcontext real = NULL;
  CUresult rc;
  DRV_CALL(rc, DRV(cuCtxGetCurrent)(&real));
  return rc == CUDA_SUCCESS ? ctx_of_real(real) : NULL;
}

// Through the driver only: cudart starts on a device where its record says
// (front_note_cudart()), not here.
void
front_ctx_use(Ctx *ctx)
{
  CUresult rc;
  DRV_CALL(rc, DRV(cuCtxSetCurrent)((CUcontext)ctx->real));
  if (rc != CUDA_SUCCESS) {
    front_fatal("using a context of device %d: error %d\n", ctx->device, rc);
  }
}

void
front_stack_set_top(Ctx *ctx)
{
  FrontThread *t = &front_thread;
  if (t->n_ctx == 0) {
    t->ctx_stack[t->n_ctx++] = ctx;
  } else {
    t->ctx_stack[t->n_ctx - 1] = ctx;
  }
}

void
front_stack_push(Ctx *ctx)
{
  FrontThread *t = &front_thread;
  if (t->n_ctx == FRONT_MAX_CTX_STACK) {
    front_fatal("more than %d contexts pushed\n", FRONT_MAX_CTX_STACK);
  }
  t->ctx_stack[t->n_ctx++] = ctx;
}

void
front_stack_pop()
{
  FrontThread *t = &front_thread;
  if (t->n_ctx > 0) {
    t->n_ctx--;
  }
}

// After a restart, on t's new lower-half TLS (t->fs): t's context stack, and
// cudart's current device.
void
front_stack_restore(FrontThread *t)
{
  CUresult rc = CUDA_SUCCESS;
  int n = 0;
  Ctx *top = NULL;
  for (int i = 0; i < t->n_ctx && rc == CUDA_SUCCESS; i++) {
    Ctx *ctx = t->ctx_stack[i];
    if (ctx->destroyed) {
      continue;
    }
    CUcontext real = (CUcontext)ctx->real;
    JUMP_TO_LOWER_HALF(t->fs);
    rc = n++ == 0 ? DRV(cuCtxSetCurrent)(real) : DRV(cuCtxPushCurrent_v2)(real);
    RETURN_TO_UPPER_HALF();
    top = ctx;
  }
  cudaError_t rrc = cudaSuccess;
  if (rc == CUDA_SUCCESS && top != NULL && top->primary) {
    JUMP_TO_LOWER_HALF(t->fs);
    rrc = REAL(cudaSetDevice)(top->device);
    RETURN_TO_UPPER_HALF();
  } else if (rc == CUDA_SUCCESS && n == 0 && t->device >= 0) {
    // cudart remembers the device with no context current.
    JUMP_TO_LOWER_HALF(t->fs);
    rrc = REAL(cudaSetDevice)(t->device);
    if (rrc == cudaSuccess) {
      rc = DRV(cuCtxSetCurrent)(NULL);
    }
    RETURN_TO_UPPER_HALF();
  }
  if (rc != CUDA_SUCCESS || rrc != cudaSuccess) {
    front_fatal("restoring a thread's contexts: error %d/%d\n", rc, rrc);
  }
}

void
front_stack_sync(FrontThread *t)
{
  CUcontext real = NULL;
  CUresult rc;
  JUMP_TO_LOWER_HALF(t->fs);
  rc = DRV(cuCtxGetCurrent)(&real);
  RETURN_TO_UPPER_HALF();
  Ctx *ctx = rc == CUDA_SUCCESS ? ctx_of_real(real, false) : NULL;
  Ctx *top = t->n_ctx > 0 ? t->ctx_stack[t->n_ctx - 1] : NULL;
  if (rc != CUDA_SUCCESS || ctx == top || (real != NULL && ctx == NULL)) {
    return;  // the same, or a context that is not the application's
  }
  if (ctx == NULL) {
    t->n_ctx--;
  } else if (t->n_ctx == 0) {
    t->ctx_stack[t->n_ctx++] = ctx;
  } else {
    t->ctx_stack[t->n_ctx - 1] = ctx;
  }
}

// At the checkpoint: the work of every context, also of those that cudart
// does not know.
void
front_ctx_quiesce()
{
  for (FrontObj *o = obj_heads[FRONT_CTX]; o != NULL; o = o->next) {
    Ctx *ctx = (Ctx *)o;
    if (ctx->destroyed || ctx->real == NULL) {
      continue;
    }
    CUresult rc;
    DRV_CALL(rc, DRV(cuCtxSynchronize_v2)((CUcontext)ctx->real));
    if (rc != CUDA_SUCCESS) {
      front_fatal("synchronizing a context before a checkpoint: error %d\n",
                  rc);
    }
  }
}

// At restart, before the records are made again: no context is real yet.
void
front_ctx_forget()
{
  ctx_by_real.clear();
  for (FrontObj *o = obj_heads[FRONT_CTX]; o != NULL; o = o->next) {
    o->real = NULL;
  }
}

static void
restore_ctx(Ctx *ctx)
{
  if (ctx->destroyed) {
    return;
  }
  CUresult rc;
  CUcontext real = NULL;
  if (ctx->primary) {
    // The front's reference, then the application's.
    for (int i = 0; i <= ctx->retains; i++) {
      DRV_CALL(rc, DRV(cuDevicePrimaryCtxRetain)(&real, ctx->device));
      if (rc != CUDA_SUCCESS) {
        front_fatal("retaining the primary context of device %d: error %d\n",
                    ctx->device, rc);
      }
    }
  } else {
    DRV_CALL(rc, DRV(cuCtxCreate_v4)(&real, NULL, ctx->flags, ctx->device));
    CUcontext popped;
    if (rc == CUDA_SUCCESS) {
      DRV_CALL(rc, DRV(cuCtxPopCurrent_v2)(&popped));
    }
    if (rc != CUDA_SUCCESS) {
      front_fatal("creating a context on device %d again: error %d\n",
                  ctx->device, rc);
    }
  }
  ctx->real = real;
  ctx_by_real[real] = ctx;
  if (ctx->limits.empty() && ctx->cache_config < 0) {
    return;
  }
  DRV_CALL(rc, DRV(cuCtxPushCurrent_v2)(real));
  for (auto &limit : ctx->limits) {
    DRV_CALL(rc, DRV(cuCtxSetLimit)((CUlimit)limit.first, limit.second));
    if (rc != CUDA_SUCCESS) {
      front_fatal("setting limit %d again: error %d\n", limit.first, rc);
    }
  }
  if (ctx->cache_config >= 0) {
    DRV_CALL(rc, DRV(cuCtxSetCacheConfig)((CUfunc_cache)ctx->cache_config));
  }
  CUcontext popped;
  DRV_CALL(rc, DRV(cuCtxPopCurrent_v2)(&popped));
}

// ---- Driver modules at checkpoint ----------------------------------------

void
front_modules_stash()
{
  for (FrontObj *o = obj_heads[FRONT_MODULE]; o != NULL; o = o->next) {
    Module *m = (Module *)o;
    if (m->globals.empty()) {
      continue;
    }
    front_ctx_use(m->ctx);
    for (Global &g : m->globals) {
      g.shadow = malloc(g.bytes);
      CUresult rc;
      DRV_CALL(rc, DRV(cuMemcpyDtoH_v2)(g.shadow, g.dptr, g.bytes));
      if (rc != CUDA_SUCCESS) {
        front_fatal("saving the module global %s: error %d\n",
                    g.name.c_str(), rc);
      }
    }
  }
}

void
front_modules_drop_shadows()
{
  for (FrontObj *o = obj_heads[FRONT_MODULE]; o != NULL; o = o->next) {
    for (Global &g : ((Module *)o)->globals) {
      free(g.shadow);
      g.shadow = NULL;
    }
  }
}

static void
restore_module(Module *m)
{
  CUresult rc;
  front_ctx_use(m->ctx);
  DRV_CALL(rc, DRV(cuModuleLoadDataEx)((CUmodule *)&m->real, m->image.data(),
                                       m->options.size(), m->options.data(),
                                       m->values.data()));
  if (rc != CUDA_SUCCESS) {
    front_fatal("loading a module again: error %d\n", rc);
  }
  for (Global &g : m->globals) {
    CUdeviceptr dptr = 0;
    size_t bytes = 0;
    DRV_CALL(rc, DRV(cuModuleGetGlobal_v2)(&dptr, &bytes, (CUmodule)m->real,
                                           g.name.c_str()));
    if (rc != CUDA_SUCCESS) {
      front_fatal("finding the module global %s again: error %d\n",
                  g.name.c_str(), rc);
    }
    if (dptr != g.dptr) {
      front_fatal("the module global %s moved from %#llx to %#llx, and the "
                  "application has its address\n", g.name.c_str(),
                  (unsigned long long)g.dptr, (unsigned long long)dptr);
    }
    if (g.shadow != NULL) {
      DRV_CALL(rc, DRV(cuMemcpyHtoD_v2)(dptr, g.shadow, g.bytes));
      if (rc != CUDA_SUCCESS) {
        front_fatal("restoring the module global %s: error %d\n",
                    g.name.c_str(), rc);
      }
    }
  }
}

static void
restore_function(Function *f)
{
  CUresult rc;
  front_ctx_use(f->module->ctx);
  DRV_CALL(rc, DRV(cuModuleGetFunction)((CUfunction *)&f->real,
                                        (CUmodule)f->module->real,
                                        f->name.c_str()));
  for (auto it = f->attrs.begin(); rc == CUDA_SUCCESS && it != f->attrs.end();
       ++it) {
    DRV_CALL(rc, DRV(cuFuncSetAttribute)((CUfunction)f->real,
                                         (CUfunction_attribute)it->first,
                                         it->second));
  }
  if (rc == CUDA_SUCCESS && f->cache_config >= 0) {
    DRV_CALL(rc, DRV(cuFuncSetCacheConfig)((CUfunction)f->real,
                                           (CUfunc_cache)f->cache_config));
  }
  if (rc != CUDA_SUCCESS) {
    front_fatal("restoring the function %s: error %d\n", f->name.c_str(),
                rc);
  }
}

void
front_drv_restore(FrontObj *obj)
{
  switch (obj->kind) {
    case FRONT_CTX:
      restore_ctx((Ctx *)obj);
      break;
    case FRONT_MODULE:
      restore_module((Module *)obj);
      break;
    case FRONT_FUNCTION:
      restore_function((Function *)obj);
      break;
    default:
      front_fatal("not a driver record: kind %u\n", obj->kind);
  }
}

// ---- Contexts ------------------------------------------------------------

static CUresult
create_ctx(CUcontext *pctx, unsigned flags, CUdevice dev, bool v4)
{
  if (pctx == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  FRONT_ENTER();
  CUcontext real = NULL;
  CUresult rc;
  if (v4) {
    DRV_CALL(rc, DRV(cuCtxCreate_v4)(&real, NULL, flags, dev));
  } else {
    DRV_CALL(rc, DRV(cuCtxCreate_v2)(&real, flags, dev));
  }
  if (rc == CUDA_SUCCESS) {
    Ctx *ctx = new_ctx(dev, false, flags);
    ctx->real = real;
    {
      std::lock_guard<std::mutex> guard(objs_lock);
      obj_link(ctx, FRONT_CTX);
      ctx_by_real[real] = ctx;
    }
    front_stack_push(ctx);
    *pctx = (CUcontext)ctx;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" CUresult
cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
  return create_ctx(pctx, flags, dev, false);
}

extern "C" CUresult
cuCtxCreate_v4(CUcontext *pctx, CUctxCreateParams *ctxCreateParams,
               unsigned int flags, CUdevice dev)
{
  if (ctxCreateParams != NULL) {
    front_unsupported("cuCtxCreate with execution affinity or CIG");
  }
  return create_ctx(pctx, flags, dev, true);
}

extern "C" CUresult
cuCtxDestroy_v2(CUcontext ctx)
{
  Ctx *c = ctx_of(ctx);
  if (c == NULL || c->primary) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  front_real_obj(ctx, FRONT_CTX);
  FRONT_ENTER();
  CUresult rc;
  DRV_CALL(rc, DRV(cuCtxDestroy_v2)((CUcontext)c->real));
  if (rc == CUDA_SUCCESS) {
    front_memops_forget((CUcontext)c->real);
    FrontThread *t = &front_thread;
    if (t->n_ctx > 0 && t->ctx_stack[t->n_ctx - 1] == c) {
      front_stack_pop();
    }
    std::lock_guard<std::mutex> guard(objs_lock);
    ctx_by_real.erase(c->real);
    c->destroyed = true;
    drop_objects_of(c);
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" CUresult
cuCtxPushCurrent_v2(CUcontext ctx)
{
  FRONT_ENTER();
  CUresult rc;
  DRV_CALL(rc, DRV(cuCtxPushCurrent_v2)(
             (CUcontext)front_real_obj(ctx, FRONT_CTX)));
  if (rc == CUDA_SUCCESS) {
    front_stack_push(ctx_of(ctx));
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" CUresult
cuCtxPopCurrent_v2(CUcontext *pctx)
{
  FRONT_ENTER();
  CUcontext real = NULL;
  CUresult rc;
  DRV_CALL(rc, DRV(cuCtxPopCurrent_v2)(&real));
  if (rc == CUDA_SUCCESS) {
    front_stack_pop();
    if (pctx != NULL) {
      *pctx = (CUcontext)ctx_of_real(real);
    }
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" CUresult
cuCtxSetCurrent(CUcontext ctx)
{
  FRONT_ENTER();
  CUresult rc;
  DRV_CALL(rc, DRV(cuCtxSetCurrent)(
             (CUcontext)front_real_obj(ctx, FRONT_CTX)));
  if (rc == CUDA_SUCCESS) {
    if (ctx != NULL) {
      front_stack_set_top(ctx_of(ctx));
    } else {
      front_stack_pop();
    }
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" CUresult
cuCtxGetCurrent(CUcontext *pctx)
{
  if (pctx == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  FRONT_ENTER();
  CUcontext real = NULL;
  CUresult rc;
  DRV_CALL(rc, DRV(cuCtxGetCurrent)(&real));
  if (rc == CUDA_SUCCESS) {
    *pctx = (CUcontext)ctx_of_real(real);
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" CUresult
cuCtxSetLimit(CUlimit limit, size_t value)
{
  FRONT_ENTER();
  CUresult rc;
  DRV_CALL(rc, DRV(cuCtxSetLimit)(limit, value));
  if (rc == CUDA_SUCCESS) {
    Ctx *ctx = front_ctx_current();
    if (ctx != NULL) {
      std::lock_guard<std::mutex> guard(objs_lock);
      ctx->limits[(int)limit] = value;
    }
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" CUresult
cuCtxSetCacheConfig(CUfunc_cache config)
{
  FRONT_ENTER();
  CUresult rc;
  DRV_CALL(rc, DRV(cuCtxSetCacheConfig)(config));
  if (rc == CUDA_SUCCESS) {
    Ctx *ctx = front_ctx_current();
    if (ctx != NULL) {
      std::lock_guard<std::mutex> guard(objs_lock);
      ctx->cache_config = (int)config;
    }
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" CUresult
cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev)
{
  if (pctx == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  FRONT_ENTER();
  CUcontext real = NULL;
  CUresult rc;
  DRV_CALL(rc, DRV(cuDevicePrimaryCtxRetain)(&real, dev));
  if (rc == CUDA_SUCCESS) {
    Ctx *ctx = front_ctx_primary(dev);
    std::lock_guard<std::mutex> guard(objs_lock);
    ctx->retains++;
    *pctx = (CUcontext)ctx;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" CUresult
cuDevicePrimaryCtxRelease_v2(CUdevice dev)
{
  FRONT_ENTER();
  CUresult rc;
  DRV_CALL(rc, DRV(cuDevicePrimaryCtxRelease_v2)(dev));
  if (rc == CUDA_SUCCESS && dev >= 0 && dev < 64) {
    std::lock_guard<std::mutex> guard(objs_lock);
    if (primaries[dev] != NULL && primaries[dev]->retains > 0) {
      primaries[dev]->retains--;
    }
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" CUresult
cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags)
{
  FRONT_ENTER();
  CUresult rc;
  DRV_CALL(rc, DRV(cuDevicePrimaryCtxSetFlags_v2)(dev, flags));
  if (rc == CUDA_SUCCESS) {
    front_set_device_flags(dev, flags);
  }
  FRONT_LEAVE();
  return rc;
}

// Peer access from the current context's device to the peer context's.
static CUresult
ctx_peer_access(CUcontext peerContext, unsigned flags, bool on)
{
  Ctx *peer = ctx_of(peerContext);
  front_real_obj(peerContext, FRONT_CTX);
  FRONT_ENTER();
  CUresult rc;
  CUdevice device = 0;
  DRV_CALL(rc, DRV(cuCtxGetDevice)(&device));
  if (rc == CUDA_SUCCESS) {
    if (on) {
      DRV_CALL(rc, DRV(cuCtxEnablePeerAccess)((CUcontext)peer->real, flags));
    } else {
      DRV_CALL(rc, DRV(cuCtxDisablePeerAccess)((CUcontext)peer->real));
    }
  }
  if (rc == CUDA_SUCCESS) {
    front_peer_access(device, peer->device, flags, on);
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" CUresult
cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags)
{
  return ctx_peer_access(peerContext, Flags, true);
}

extern "C" CUresult
cuCtxDisablePeerAccess(CUcontext peerContext)
{
  return ctx_peer_access(peerContext, 0, false);
}

extern "C" CUresult
cuStreamGetCtx(CUstream hStream, CUcontext *pctx)
{
  if (pctx == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if ((uintptr_t)hStream > 2) {
    *pctx = (CUcontext)((Stream *)hStream)->ctx;
    return CUDA_SUCCESS;
  }
  return cuCtxGetCurrent(pctx);
}

extern "C" CUresult
cuStreamGetCtx_ptsz(CUstream hStream, CUcontext *pctx)
{
  return cuStreamGetCtx(hStream, pctx);
}

// ---- Memory --------------------------------------------------------------

// The allocator's errors (invalid value, out of memory, invalid device) have
// the same values as CUresults.
extern "C" CUresult
cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
{
  FRONT_ENTER();
  CUresult rc = (CUresult)front_vmm_alloc((void **)dptr, bytesize);
  FRONT_LEAVE();
  return rc;
}

extern "C" CUresult
cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes,
                   size_t Height, unsigned int ElementSizeBytes)
{
  if (pPitch == NULL || (ElementSizeBytes != 4 && ElementSizeBytes != 8 &&
                         ElementSizeBytes != 16)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  FRONT_ENTER();
  CUresult rc = (CUresult)front_vmm_alloc_pitch((void **)dptr, pPitch,
                                                WidthInBytes, Height);
  FRONT_LEAVE();
  return rc;
}

extern "C" CUresult
cuMemFree_v2(CUdeviceptr dptr)
{
  FRONT_ENTER();
  cudaError_t result = cudaSuccess;
  CUresult rc;
  if (front_vmm_free((void *)dptr, &result)) {
    rc = result == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
  } else {
    DRV_CALL(rc, DRV(cuMemFree_v2)(dptr));
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" CUresult
cuMemAllocHost_v2(void **pp, size_t bytesize)
{
  return (CUresult)front_host_alloc(pp, bytesize, 0, true);
}

extern "C" CUresult
cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags)
{
  // CU_MEMHOSTALLOC_WRITECOMBINED cannot be reproduced on registered memory.
  unsigned reg = 0;
  if (Flags & CU_MEMHOSTALLOC_PORTABLE) {
    reg |= CU_MEMHOSTREGISTER_PORTABLE;
  }
  if (Flags & CU_MEMHOSTALLOC_DEVICEMAP) {
    reg |= CU_MEMHOSTREGISTER_DEVICEMAP;
  }
  return (CUresult)front_host_alloc(pp, bytesize, reg, true);
}

extern "C" CUresult
cuMemFreeHost(void *p)
{
  return (CUresult)front_host_free(p, true);
}

extern "C" CUresult
cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int Flags)
{
  return (CUresult)front_host_register(p, bytesize, Flags, true);
}

extern "C" CUresult
cuMemHostUnregister(void *p)
{
  return (CUresult)front_host_unregister(p, true);
}

static void
translate_pointer_attribute(CUpointer_attribute attribute, void *data)
{
  if (attribute == CU_POINTER_ATTRIBUTE_CONTEXT && data != NULL) {
    *(CUcontext *)data = (CUcontext)ctx_of_real(*(CUcontext *)data);
  }
}

extern "C" CUresult
cuPointerGetAttribute(void *data, CUpointer_attribute attribute,
                      CUdeviceptr ptr)
{
  FRONT_ENTER();
  CUresult rc;
  DRV_CALL(rc, DRV(cuPointerGetAttribute)(data, attribute, ptr));
  if (rc == CUDA_SUCCESS) {
    translate_pointer_attribute(attribute, data);
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" CUresult
cuPointerGetAttributes(unsigned int numAttributes,
                       CUpointer_attribute *attributes, void **data,
                       CUdeviceptr ptr)
{
  FRONT_ENTER();
  CUresult rc;
  DRV_CALL(rc, DRV(cuPointerGetAttributes)(numAttributes, attributes, data,
                                           ptr));
  for (unsigned i = 0; rc == CUDA_SUCCESS && i < numAttributes; i++) {
    translate_pointer_attribute(attributes[i], data[i]);
  }
  FRONT_LEAVE();
  return rc;
}

// ---- Modules and functions -----------------------------------------------

extern "C" CUresult
cuModuleLoadDataEx(CUmodule *module, const void *image,
                   unsigned int numOptions, CUjit_option *options,
                   void **optionValues)
{
  if (image == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  image = unwrap_fatbin(image);
  return load_module(module, image, image_size(image), numOptions, options,
                     optionValues);
}

extern "C" CUresult
cuModuleLoadData(CUmodule *module, const void *image)
{
  return cuModuleLoadDataEx(module, image, 0, NULL, NULL);
}

extern "C" CUresult
cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin)
{
  return cuModuleLoadDataEx(module, fatCubin, 0, NULL, NULL);
}

// The file may be gone at restart: the module keeps its contents.
extern "C" CUresult
cuModuleLoad(CUmodule *module, const char *fname)
{
  if (fname == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  int fd = open(fname, O_RDONLY);
  if (fd < 0) {
    return CUDA_ERROR_FILE_NOT_FOUND;
  }
  std::string image;
  char buf[65536];
  ssize_t n;
  while ((n = read(fd, buf, sizeof buf)) > 0) {
    image.append(buf, n);
  }
  close(fd);
  if (n < 0 || image.empty()) {
    return CUDA_ERROR_FILE_NOT_FOUND;
  }
  image.push_back('\0');  // for PTX
  return load_module(module, image.data(), image.size(), 0, NULL, NULL);
}

extern "C" CUresult
cuModuleUnload(CUmodule hmod)
{
  Module *m = module_of(hmod);
  FRONT_ENTER();
  CUresult rc;
  DRV_CALL(rc, DRV(cuModuleUnload)((CUmodule)m->real));
  if (rc == CUDA_SUCCESS) {
    std::lock_guard<std::mutex> guard(objs_lock);
    for (FrontObj *o = obj_heads[FRONT_FUNCTION], *next; o != NULL;
         o = next) {
      next = o->next;
      if (((Function *)o)->module == m) {
        obj_unlink(o);
        delete (Function *)o;
      }
    }
    obj_unlink(m);
    delete m;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" CUresult
cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)
{
  if (hfunc == NULL || name == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  Module *m = module_of(hmod);
  FRONT_ENTER();
  CUresult rc = CUDA_SUCCESS;
  Function *f = NULL;
  {
    std::lock_guard<std::mutex> guard(objs_lock);
    for (FrontObj *o = obj_heads[FRONT_FUNCTION]; o != NULL; o = o->next) {
      if (((Function *)o)->module == m && ((Function *)o)->name == name) {
        f = (Function *)o;
        break;
      }
    }
  }
  if (f == NULL) {
    CUfunction real = NULL;
    DRV_CALL(rc, DRV(cuModuleGetFunction)(&real, (CUmodule)m->real, name));
    if (rc == CUDA_SUCCESS) {
      f = new Function();
      f->real = real;
      f->module = m;
      f->name = name;
      f->cache_config = -1;
      std::lock_guard<std::mutex> guard(objs_lock);
      obj_link(f, FRONT_FUNCTION);
    }
  }
  if (rc == CUDA_SUCCESS) {
    *hfunc = (CUfunction)f;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" CUresult
cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                     const char *name)
{
  if (name == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  Module *m = module_of(hmod);
  FRONT_ENTER();
  CUdeviceptr ptr = 0;
  size_t size = 0;
  CUresult rc;
  DRV_CALL(rc, DRV(cuModuleGetGlobal_v2)(&ptr, &size, (CUmodule)m->real,
                                         name));
  if (rc == CUDA_SUCCESS) {
    {
      std::lock_guard<std::mutex> guard(objs_lock);
      bool known = false;
      for (Global &g : m->globals) {
        known = known || g.name == name;
      }
      if (!known) {
        m->globals.push_back({ name, ptr, size, NULL });
      }
    }
    if (dptr != NULL) {
      *dptr = ptr;
    }
    if (bytes != NULL) {
      *bytes = size;
    }
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" CUresult
cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int value)
{
  Function *f = function_of(hfunc);
  FRONT_ENTER();
  CUresult rc;
  DRV_CALL(rc, DRV(cuFuncSetAttribute)((CUfunction)f->real, attrib, value));
  if (rc == CUDA_SUCCESS) {
    std::lock_guard<std::mutex> guard(objs_lock);
    f->attrs[(int)attrib] = value;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" CUresult
cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config)
{
  Function *f = function_of(hfunc);
  FRONT_ENTER();
  CUresult rc;
  DRV_CALL(rc, DRV(cuFuncSetCacheConfig)((CUfunction)f->real, config));
  if (rc == CUDA_SUCCESS) {
    std::lock_guard<std::mutex> guard(objs_lock);
    f->cache_config = (int)config;
  }
  FRONT_LEAVE();
  return rc;
}

extern "C" CUresult
cuFuncGetModule(CUmodule *hmod, CUfunction hfunc)
{
  if (hmod == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *hmod = (CUmodule)function_of(hfunc)->module;
  return CUDA_SUCCESS;
}

// ---- Streams, events, errors ---------------------------------------------

extern "C" CUresult
cuStreamCreate(CUstream *phStream, unsigned int Flags)
{
  return (CUresult)front_create_stream(phStream, Flags, 0, true);
}

extern "C" CUresult
cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags,
                           int priority)
{
  return (CUresult)front_create_stream(phStream, flags, priority, true);
}

extern "C" CUresult
cuStreamDestroy_v2(CUstream hStream)
{
  return (CUresult)front_destroy_stream(hStream, true);
}

extern "C" CUresult
cuEventCreate(CUevent *phEvent, unsigned int Flags)
{
  return (CUresult)front_create_event(phEvent, Flags, true);
}

extern "C" CUresult
cuEventDestroy_v2(CUevent hEvent)
{
  return (CUresult)front_destroy_event(hEvent, true);
}

extern "C" CUresult
cuGetErrorString(CUresult error, const char **pStr)
{
  if (pStr == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *pStr = error_string(error, false);
  return *pStr != NULL ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

extern "C" CUresult
cuGetErrorName(CUresult error, const char **pStr)
{
  if (pStr == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *pStr = error_string(error, true);
  return *pStr != NULL ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

// ---- Entry points --------------------------------------------------------

// The lower half's driver picks the function for the version and flags; the
// application gets the front's function of the same name.
extern "C" CUresult
cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                    cuuint64_t flags,
                    CUdriverProcAddressQueryResult *symbolStatus)
{
  if (pfn == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  FRONT_ENTER();
  void *real = NULL;
  CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SUCCESS;
  CUresult rc;
  DRV_CALL(rc, DRV(cuGetProcAddress_v2)(symbol, &real, cudaVersion, flags,
                                        &status));
  *pfn = NULL;
  if (rc == CUDA_SUCCESS && real != NULL) {
    *pfn = front_entry_point(real);
    if (*pfn == NULL) {
      front_log("cuGetProcAddress(%s): not a function of libcuda.so.1\n",
                symbol);
      rc = CUDA_ERROR_NOT_FOUND;
      status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    }
  }
  if (symbolStatus != NULL) {
    *symbolStatus = status;
  }
  FRONT_LEAVE();
  return rc;
}

static cudaError_t
runtime_entry_point(cudaError_t rc, void *real, void **funcPtr,
                    cudaDriverEntryPointQueryResult *driverStatus,
                    const char *symbol)
{
  if (funcPtr != NULL) {
    *funcPtr = NULL;
  }
  if (rc == cudaSuccess && real != NULL && funcPtr != NULL) {
    *funcPtr = front_entry_point(real);
    if (*funcPtr == NULL) {
      front_log("cudaGetDriverEntryPoint(%s): not a function of "
                "libcuda.so.1\n", symbol);
      if (driverStatus != NULL) {
        *driverStatus = cudaDriverEntryPointSymbolNotFound;
      }
      return cudaErrorSymbolNotFound;
    }
  }
  return rc;
}

// Deprecated in CUDA 13, but still exported.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
extern "C" cudaError_t
cudaGetDriverEntryPoint(const char *symbol, void **funcPtr,
                        unsigned long long flags,
                        cudaDriverEntryPointQueryResult *driverStatus)
{
  FRONT_ENTER();
  void *real = NULL;
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGetDriverEntryPoint)(symbol, &real, flags,
                                              driverStatus));
  rc = runtime_entry_point(rc, real, funcPtr, driverStatus, symbol);
  FRONT_LEAVE();
  return rc;
}
#pragma GCC diagnostic pop

extern "C" cudaError_t
cudaGetDriverEntryPointByVersion(const char *symbol, void **funcPtr,
                                 unsigned int cudaVersion,
                                 unsigned long long flags,
                                 cudaDriverEntryPointQueryResult *driverStatus)
{
  FRONT_ENTER();
  void *real = NULL;
  cudaError_t rc;
  FRONT_LH(rc = REAL(cudaGetDriverEntryPointByVersion)(
             symbol, &real, cudaVersion, flags, driverStatus));
  rc = runtime_entry_point(rc, real, funcPtr, driverStatus, symbol);
  FRONT_LEAVE();
  return rc;
}
