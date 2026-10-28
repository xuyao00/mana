# The CUDA front of the upper half under --cuda
# (cuda/DESIGN-lower-half-cuda.md): MANA's libcudart.so.<N> and libcuda.so.1,
# generated from the real libraries that bin/lower-half uses
# (gen-cuda-front.py), the same for cuBLAS, cuBLASLt, cuFFT and cuRAND if
# the toolkit has them, and libmana_cuda.so, which they all jump into.  All are
# installed in lib/dmtcp/cuda-front/.

include ../Makefile_config

ifndef MANA_ROOT
  MANA_ROOT=../..
endif
ifndef DMTCP_ROOT
  DMTCP_ROOT=${MANA_ROOT}/dmtcp
endif

NVCC_PATH := $(shell which nvcc 2>/dev/null)
ifdef NVCC_PATH
  CUDA_HOME ?= $(realpath $(dir $(NVCC_PATH))/..)
endif
CUDA_HOME ?= /usr/local/cuda

# The CUDA libraries of the lower half's MPI: the front stands for them.
LOWER_HALF = ../lower-half/lower-half
REAL_LIBCUDART ?= $(shell ldd ${LOWER_HALF} 2>/dev/null | \
                    awk '/libcudart\.so/ {print $$3; exit}')
REAL_LIBCUDA ?= $(shell ldd ${LOWER_HALF} 2>/dev/null | \
                  awk '/libcuda\.so/ {print $$3; exit}')
soname = $(shell readelf -d $(1) 2>/dev/null | \
           sed -n 's/.*soname: \[\(.*\)\]/\1/p')
CUDART_SONAME := $(call soname,${REAL_LIBCUDART})

# The application's libraries that run in the lower half too (Phase 3):
# next to the lower half's libcudart, or in CUDA_MATH_LIBDIR (at NERSC, the
# HPC SDK's math_libs).  cuBLAS needs cuBLASLt.
CUDA_MATH_LIBDIR ?= $(dir $(realpath ${REAL_LIBCUDART}))
CUDA_MATH_INCLUDE ?= $(realpath ${CUDA_MATH_LIBDIR}/../include)
real_lib = $(firstword $(wildcard ${CUDA_MATH_LIBDIR}/$(1).so.[0-9]*))
REAL_LIBCUBLAS ?= $(call real_lib,libcublas)
REAL_LIBCUBLASLT ?= $(call real_lib,libcublasLt)
REAL_LIBCUFFT ?= $(call real_lib,libcufft)
REAL_LIBCURAND ?= $(call real_lib,libcurand)
ifneq (${REAL_LIBCUBLAS},)
ifneq (${REAL_LIBCUBLASLT},)
  CUBLAS_SONAME := $(call soname,${REAL_LIBCUBLAS})
  CUBLASLT_SONAME := $(call soname,${REAL_LIBCUBLASLT})
endif
endif
ifneq (${REAL_LIBCUFFT},)
  CUFFT_SONAME := $(call soname,${REAL_LIBCUFFT})
endif
ifneq (${REAL_LIBCURAND},)
  CURAND_SONAME := $(call soname,${REAL_LIBCURAND})
endif

INSTALL_DIR = ${MANA_ROOT}/lib/dmtcp/cuda-front

# -fno-stack-protector: the front switches the FS register (see front.h).
FRONT_CXXFLAGS = -std=c++17 -fPIC -fno-stack-protector -g3 -O2 -Wall -I. \
                 -I.. -I../lower-half -I${DMTCP_ROOT}/include \
                 -I${DMTCP_ROOT}/jalib -I${CUDA_HOME}/include \
                 $(addprefix -I,${CUDA_MATH_INCLUDE})
GEN_CFLAGS = -I${CUDA_HOME}/include $(addprefix -I,${CUDA_MATH_INCLUDE})

FRONT_OBJS = front_core.o front_vmm.o front_objs.o front_drv.o \
             front_async.o front_memops.o front_ipc.o front_static.o \
             front_arrays.o \
             gen/cudart-gen.o gen/cuda-gen.o \
             $(if ${CUBLAS_SONAME},front_blas.o gen/cublas-gen.o \
               gen/cublasLt-gen.o) \
             $(if ${CUFFT_SONAME},front_fft.o gen/cufft-gen.o) \
             $(if ${CURAND_SONAME},front_rand.o gen/curand-gen.o)
FRONT_LIBS = ${CUDART_SONAME} libcuda.so.1 ${CUBLAS_SONAME} \
             ${CUBLASLT_SONAME} ${CUFFT_SONAME} ${CURAND_SONAME}
FRONT_HEADERS = front.h front_auto.h front_objs.h front_reg.h mana_cuda.h \
                gen/cudart-index.h gen/cuda-index.h \
                ../lower-half/lower-half-api.h

default: check-real libmana_cuda.so ${FRONT_LIBS} libmana_cuda_audit.so

check-real:
	@test -n "${REAL_LIBCUDART}" -a -n "${REAL_LIBCUDA}" || \
	  { echo "cuda-front: ${LOWER_HALF} links no libcudart/libcuda;" \
	         "set REAL_LIBCUDART and REAL_LIBCUDA"; exit 1; }

gen/cudart-gen.cpp gen/cudart-tramp.S gen/cudart.map gen/cudart-index.h: \
    gen-cuda-front.py cudart-own.txt cudart-unsupported.txt \
    cudart-memops.txt ${REAL_LIBCUDART}
	python3 gen-cuda-front.py --real=${REAL_LIBCUDART} --prefix=cudart \
	  --out-dir=gen --header=cuda_runtime_api.h --cxx=g++ \
	  --cflags="${GEN_CFLAGS}" --own=cudart-own.txt \
	  --unsupported=cudart-unsupported.txt --memops=cudart-memops.txt

# The driver forwards only what cuda-pass.txt lists.
gen/cuda-gen.cpp gen/cuda-tramp.S gen/cuda.map gen/cuda-index.h: \
    gen-cuda-front.py cuda-own.txt cuda-pass.txt cuda-memops.txt \
    ${REAL_LIBCUDA}
	python3 gen-cuda-front.py --real=${REAL_LIBCUDA} --prefix=cuda \
	  --out-dir=gen --header=cuda.h --cxx=g++ --cflags="${GEN_CFLAGS}" \
	  --own=cuda-own.txt --pass=cuda-pass.txt --memops=cuda-memops.txt

# cuBLAS and cuBLASLt (front_blas.h translates their handles); cuFFT and
# cuRAND, whose plans and generators are all own: they forward only their
# -pass.txt.
gen/cublas-gen.cpp gen/cublas-tramp.S gen/cublas.map gen/cublas-index.h: \
    gen-cuda-front.py cublas-own.txt cublas-unsupported.txt \
    cublas-memops.txt ${REAL_LIBCUBLAS}
	python3 gen-cuda-front.py --real=${REAL_LIBCUBLAS} --prefix=cublas \
	  --out-dir=gen --header=cublas_v2.h --cxx=g++ \
	  --cflags="${GEN_CFLAGS}" --own=cublas-own.txt \
	  --unsupported=cublas-unsupported.txt --memops=cublas-memops.txt \
	  --include=front_blas.h

gen/cublasLt-gen.cpp gen/cublasLt-tramp.S gen/cublasLt.map \
gen/cublasLt-index.h: \
    gen-cuda-front.py cublasLt-own.txt cublasLt-unsupported.txt \
    ${REAL_LIBCUBLASLT}
	python3 gen-cuda-front.py --real=${REAL_LIBCUBLASLT} --prefix=cublasLt \
	  --out-dir=gen --header=cublasLt.h --cxx=g++ \
	  --cflags="${GEN_CFLAGS}" --own=cublasLt-own.txt \
	  --unsupported=cublasLt-unsupported.txt --include=front_blas.h

gen/cufft-gen.cpp gen/cufft-tramp.S gen/cufft.map gen/cufft-index.h: \
    gen-cuda-front.py cufft-own.txt cufft-pass.txt ${REAL_LIBCUFFT}
	python3 gen-cuda-front.py --real=${REAL_LIBCUFFT} --prefix=cufft \
	  --out-dir=gen --header=cufftXt.h --cxx=g++ --cflags="${GEN_CFLAGS}" \
	  --own=cufft-own.txt --pass=cufft-pass.txt

gen/curand-gen.cpp gen/curand-tramp.S gen/curand.map gen/curand-index.h: \
    gen-cuda-front.py curand-own.txt curand-pass.txt ${REAL_LIBCURAND}
	python3 gen-cuda-front.py --real=${REAL_LIBCURAND} --prefix=curand \
	  --out-dir=gen --header=curand.h --cxx=g++ --cflags="${GEN_CFLAGS}" \
	  --own=curand-own.txt --pass=curand-pass.txt

front_blas.o gen/cublas-gen.o gen/cublasLt-gen.o: front_blas.h \
    gen/cublas-index.h gen/cublasLt-index.h
front_fft.o: gen/cufft-index.h
front_rand.o: gen/curand-index.h

%.o: %.cpp ${FRONT_HEADERS}
	${MPICXX} ${FRONT_CXXFLAGS} -c -o $@ $<

gen/%.o: gen/%.cpp ${FRONT_HEADERS}
	${MPICXX} ${FRONT_CXXFLAGS} -Wno-deprecated-declarations -c -o $@ $<

gen/%.o: gen/%.S
	${MPICXX} -c -o $@ $<

# Everything but the exported functions: those are the front libraries'.
libmana_cuda.so: ${FRONT_OBJS} libmana_cuda.map
	${MPICXX} -shared -o $@ ${FRONT_OBJS} -Wl,-soname,libmana_cuda.so \
	  -Wl,--version-script=libmana_cuda.map -Wl,-z,now

${CUDART_SONAME}: gen/cudart-tramp.o gen/cudart.map libmana_cuda.so
	${MPICXX} -shared -o $@ gen/cudart-tramp.o libmana_cuda.so \
	  -Wl,-soname,${CUDART_SONAME} -Wl,--version-script=gen/cudart.map \
	  -Wl,-rpath,'$$ORIGIN' -Wl,-z,now

libcuda.so.1: gen/cuda-tramp.o gen/cuda.map libmana_cuda.so
	${MPICXX} -shared -o $@ gen/cuda-tramp.o libmana_cuda.so \
	  -Wl,-soname,libcuda.so.1 -Wl,--version-script=gen/cuda.map \
	  -Wl,-rpath,'$$ORIGIN' -Wl,-z,now

${CUBLAS_SONAME}: gen/cublas-tramp.o gen/cublas.map libmana_cuda.so
	${MPICXX} -shared -o $@ gen/cublas-tramp.o libmana_cuda.so \
	  -Wl,-soname,${CUBLAS_SONAME} -Wl,--version-script=gen/cublas.map \
	  -Wl,-rpath,'$$ORIGIN' -Wl,-z,now

${CUBLASLT_SONAME}: gen/cublasLt-tramp.o gen/cublasLt.map libmana_cuda.so
	${MPICXX} -shared -o $@ gen/cublasLt-tramp.o libmana_cuda.so \
	  -Wl,-soname,${CUBLASLT_SONAME} -Wl,--version-script=gen/cublasLt.map \
	  -Wl,-rpath,'$$ORIGIN' -Wl,-z,now

${CUFFT_SONAME}: gen/cufft-tramp.o gen/cufft.map libmana_cuda.so
	${MPICXX} -shared -o $@ gen/cufft-tramp.o libmana_cuda.so \
	  -Wl,-soname,${CUFFT_SONAME} -Wl,--version-script=gen/cufft.map \
	  -Wl,-rpath,'$$ORIGIN' -Wl,-z,now

${CURAND_SONAME}: gen/curand-tramp.o gen/curand.map libmana_cuda.so
	${MPICXX} -shared -o $@ gen/curand-tramp.o libmana_cuda.so \
	  -Wl,-soname,${CURAND_SONAME} -Wl,--version-script=gen/curand.map \
	  -Wl,-rpath,'$$ORIGIN' -Wl,-z,now

# An LD_AUDIT module for the upper half; it must not need libc.
libmana_cuda_audit.so: cuda_audit.c
	${CC} -O2 -fPIC -shared -nostdlib -fno-stack-protector \
	  -Wl,--no-undefined -o $@ $<

install: default
	mkdir -p ${INSTALL_DIR}
	cp -f libmana_cuda.so ${FRONT_LIBS} ${INSTALL_DIR}/
	ln -sf ${CUDART_SONAME} ${INSTALL_DIR}/libcudart.so
	ln -sf libcuda.so.1 ${INSTALL_DIR}/libcuda.so
	$(if ${CUBLAS_SONAME},ln -sf ${CUBLAS_SONAME} ${INSTALL_DIR}/libcublas.so)
	$(if ${CUBLASLT_SONAME},ln -sf ${CUBLASLT_SONAME} \
	  ${INSTALL_DIR}/libcublasLt.so)
	$(if ${CUFFT_SONAME},ln -sf ${CUFFT_SONAME} ${INSTALL_DIR}/libcufft.so)
	$(if ${CURAND_SONAME},ln -sf ${CURAND_SONAME} \
	  ${INSTALL_DIR}/libcurand.so)
	cp -f libmana_cuda_audit.so ${MANA_ROOT}/lib/dmtcp/

clean:
	rm -rf gen *.o libmana_cuda.so libcudart.so.* libcuda.so.1 \
	  libcublas.so.* libcublasLt.so.* libcufft.so.* libcurand.so.* \
	  libmana_cuda_audit.so

.PHONY: default check-real install clean
