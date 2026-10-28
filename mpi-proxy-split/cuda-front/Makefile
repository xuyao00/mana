# The CUDA front libraries of the upper half under --cuda
# (cuda/DESIGN-lower-half-cuda.md): MANA's libcudart.so.<N> and libcuda.so.1,
# generated from the real libraries that bin/lower-half uses
# (gen-cuda-front.py), and installed in lib/dmtcp/cuda-front/.

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
CUDART_SONAME := $(shell readelf -d ${REAL_LIBCUDART} 2>/dev/null | \
                   sed -n 's/.*soname: \[\(.*\)\]/\1/p')

INSTALL_DIR = ${MANA_ROOT}/lib/dmtcp/cuda-front

# -fno-stack-protector: the front switches the FS register (see front.h).
FRONT_CXXFLAGS = -std=c++17 -fPIC -fno-stack-protector -g3 -O2 -Wall -I. \
                 -I.. -I../lower-half -I${DMTCP_ROOT}/include \
                 -I${DMTCP_ROOT}/jalib -I${CUDA_HOME}/include \
                 -DFRONT_CUDART_SONAME=\"${CUDART_SONAME}\"
GEN_CFLAGS = -I${CUDA_HOME}/include

CUDART_OBJS = front_core.o front_vmm.o front_objs.o gen/cudart-gen.o \
              gen/cudart-tramp.o
CUDA_OBJS = front_unsup.o gen/cuda-gen.o gen/cuda-tramp.o

default: check-real ${CUDART_SONAME} libcuda.so.1 libmana_cuda_audit.so

check-real:
	@test -n "${REAL_LIBCUDART}" -a -n "${REAL_LIBCUDA}" || \
	  { echo "cuda-front: ${LOWER_HALF} links no libcudart/libcuda;" \
	         "set REAL_LIBCUDART and REAL_LIBCUDA"; exit 1; }

gen/cudart-gen.cpp gen/cudart-tramp.S gen/cudart.map: gen-cuda-front.py \
    cudart-own.txt cudart-unsupported.txt ${REAL_LIBCUDART}
	python3 gen-cuda-front.py --real=${REAL_LIBCUDART} --prefix=cudart \
	  --out-dir=gen --header=cuda_runtime_api.h --cxx=g++ \
	  --cflags="${GEN_CFLAGS}" --own=cudart-own.txt \
	  --unsupported=cudart-unsupported.txt

# Every driver function aborts for now: the application must use cudart.
gen/cuda-gen.cpp gen/cuda-tramp.S gen/cuda.map: gen-cuda-front.py \
    ${REAL_LIBCUDA}
	python3 gen-cuda-front.py --real=${REAL_LIBCUDA} --prefix=cuda \
	  --out-dir=gen

%.o: %.cpp front.h front_auto.h ../lower-half/lower-half-api.h
	${MPICXX} ${FRONT_CXXFLAGS} -c -o $@ $<

gen/%.o: gen/%.cpp front.h front_auto.h ../lower-half/lower-half-api.h
	${MPICXX} ${FRONT_CXXFLAGS} -Wno-deprecated-declarations -c -o $@ $<

gen/%.o: gen/%.S
	${MPICXX} -c -o $@ $<

${CUDART_SONAME}: ${CUDART_OBJS} gen/cudart.map
	${MPICXX} -shared -o $@ ${CUDART_OBJS} -Wl,-soname,${CUDART_SONAME} \
	  -Wl,--version-script=gen/cudart.map -Wl,-z,now

libcuda.so.1: ${CUDA_OBJS} gen/cuda.map
	${MPICXX} -shared -o $@ ${CUDA_OBJS} -Wl,-soname,libcuda.so.1 \
	  -Wl,--version-script=gen/cuda.map -Wl,-z,now

# An LD_AUDIT module for the upper half; it must not need libc.
libmana_cuda_audit.so: cuda_audit.c
	${CC} -O2 -fPIC -shared -nostdlib -fno-stack-protector \
	  -Wl,--no-undefined -o $@ $<

install: default
	mkdir -p ${INSTALL_DIR}
	cp -f ${CUDART_SONAME} libcuda.so.1 ${INSTALL_DIR}/
	ln -sf ${CUDART_SONAME} ${INSTALL_DIR}/libcudart.so
	ln -sf libcuda.so.1 ${INSTALL_DIR}/libcuda.so
	cp -f libmana_cuda_audit.so ${MANA_ROOT}/lib/dmtcp/

clean:
	rm -rf gen *.o libcudart.so.* libcuda.so.1 libmana_cuda_audit.so

.PHONY: default check-real install clean
