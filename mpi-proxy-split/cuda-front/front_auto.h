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

// The dispatch tables of the front libraries (gen-cuda-front.py): each
// exported function jumps through its library's table to an own function, to
// FrontAuto<decltype(f)>::call, or to front_unsupported_at().

#ifndef _MANA_CUDA_FRONT_AUTO_H
#define _MANA_CUDA_FRONT_AUTO_H

#include <type_traits>
#include "front.h"

template <typename T, typename = void>
struct front_complete : std::false_type {};
template <typename T>
struct front_complete<T, decltype(void(sizeof(T)))> : std::true_type {};

// An opaque handle type: a pointer to an incomplete struct (CUstream_st,
// CUgraph_st, cudaArray, ...).
template <typename T>
constexpr bool
front_is_handle()
{
  if constexpr (std::is_pointer_v<T>) {
    using P = std::remove_cv_t<std::remove_pointer_t<T>>;
    return std::is_class_v<P> && !front_complete<P>::value;
  }
  return false;
}

// Whether the front can forward a parameter of type T as it is (after
// translating streams, events, contexts, modules and functions).  Not: other
// handles, pointers to handles (results), and callbacks, which run on the
// lower half's threads.
template <typename T>
constexpr bool
front_arg_ok()
{
  using U = std::remove_cv_t<T>;
  if constexpr (std::is_same_v<U, cudaStream_t> ||
                std::is_same_v<U, cudaEvent_t> ||
                std::is_same_v<U, CUcontext> || std::is_same_v<U, CUmodule> ||
                std::is_same_v<U, CUfunction>) {
    return true;
  } else if constexpr (front_is_handle<U>()) {
    return false;
  } else if constexpr (std::is_pointer_v<U>) {
    using P = std::remove_cv_t<std::remove_pointer_t<U>>;
    return !std::is_function_v<P> && !front_is_handle<P>();
  }
  return true;
}

template <typename T>
static inline T
front_translate(T value)
{
  return value;
}

static inline cudaStream_t
front_translate(cudaStream_t stream)
{
  return front_real_stream(stream);
}

static inline cudaEvent_t
front_translate(cudaEvent_t event)
{
  return front_real_event(event);
}

static inline CUcontext
front_translate(CUcontext ctx)
{
  return (CUcontext)front_real_obj(ctx, FRONT_CTX);
}

static inline CUmodule
front_translate(CUmodule module)
{
  return (CUmodule)front_real_obj(module, FRONT_MODULE);
}

static inline CUfunction
front_translate(CUfunction function)
{
  return (CUfunction)front_real_obj(function, FRONT_FUNCTION);
}

template <typename T>
struct front_identity {
  using type = T;
};

// Calls 'fn' in the lower half with arguments that are already translated.
template <typename R, typename... B>
static inline R
front_call_lh(R (*fn)(B...), typename front_identity<B>::type... args)
{
  void *fs = front_lh_fs();
  if constexpr (std::is_void_v<R>) {
    JUMP_TO_LOWER_HALF(fs);
    fn(args...);
    RETURN_TO_UPPER_HALF();
  } else {
    R ret;
    JUMP_TO_LOWER_HALF(fs);
    ret = fn(args...);
    RETURN_TO_UPPER_HALF();
    return ret;
  }
}

template <typename F, FrontLib *L, int I>
struct FrontAuto;

template <typename R, typename... A, FrontLib *L, int I>
struct FrontAuto<R(A...), L, I> {
  static R call(A... args)
  {
    if constexpr (!(front_arg_ok<A>() && ...)) {
      front_unsupported(L->names[I]);
    } else {
      FRONT_ENTER();
      front_check_bound(L);
      using Fn = R (*)(A...);
      Fn fn = (Fn)L->real[I];
      if constexpr (std::is_void_v<R>) {
        front_call_lh(fn, front_translate(args)...);
        FRONT_LEAVE();
      } else {
        R ret = front_call_lh(fn, front_translate(args)...);
        FRONT_LEAVE();
        return ret;
      }
    }
  }
};

// A synchronous copy or set (--memops): see front_memops.cpp.
template <typename F, FrontLib *L, int I, bool PTDS>
struct FrontAutoMemop;

template <typename R, typename... A, FrontLib *L, int I, bool PTDS>
struct FrontAutoMemop<R(A...), L, I, PTDS> {
  static_assert((front_arg_ok<A>() && ...), "a memop the front can forward");

  static R call(A... args)
  {
    FRONT_ENTER();
    front_check_bound(L);
    using Fn = R (*)(A...);
    R ret = front_call_lh((Fn)L->real[I], front_translate(args)...);
    if (ret == 0) {
      front_order_memop(PTDS);
    }
    FRONT_LEAVE();
    return ret;
  }
};

template <FrontLib *L, int I>
[[noreturn]] static void
front_unsupported_at()
{
  front_unsupported(L->names[I]);
}

// An own function is defined in libmana_cuda.so under its own name; the
// version script keeps it local there.
#define FRONT_OWN(name) (void *)&::name
#define FRONT_AUTO(prefix, base, index) \
  (void *)&FrontAuto<decltype(::base), &front_##prefix##_lib, index>::call
#define FRONT_AUTO_MEMOP(prefix, base, index, ptds)                      \
  (void *)&FrontAutoMemop<decltype(::base), &front_##prefix##_lib, index, \
                          ptds>::call
#define FRONT_UNSUPPORTED(prefix, index) \
  (void *)&front_unsupported_at<&front_##prefix##_lib, index>

#endif  // ifndef _MANA_CUDA_FRONT_AUTO_H
