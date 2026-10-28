# The CUDA front of the upper half under --cuda
# (cuda/DESIGN-lower-half-cuda.md): MANA's libcudart.so.<N> and libcuda.so.1,
# generated from the real libraries that bin/lower-half uses
# (gen-cuda-front.py), and libmana_cuda.so, which they both jump into.  All
# are installed in lib/dmtcp/cuda-front/.

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
                 -I${DMTCP_ROOT}/jalib -I${CUDA_HOME}/include
GEN_CFLAGS = -I${CUDA_HOME}/include

FRONT_OBJS = front_core.o front_vmm.o front_objs.o front_drv.o \
             front_async.o front_memops.o front_ipc.o front_static.o \
             gen/cudart-gen.o gen/cuda-gen.o
FRONT_HEADERS = front.h front_auto.h front_objs.h front_reg.h mana_cuda.h \
                gen/cudart-index.h gen/cuda-index.h \
                ../lower-half/lower-half-api.h

default: check-real libmana_cuda.so ${CUDART_SONAME} libcuda.so.1 \
         libmana_cuda_audit.so

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

# An LD_AUDIT module for the upper half; it must not need libc.
libmana_cuda_audit.so: cuda_audit.c
	${CC} -O2 -fPIC -shared -nostdlib -fno-stack-protector \
	  -Wl,--no-undefined -o $@ $<

install: default
	mkdir -p ${INSTALL_DIR}
	cp -f libmana_cuda.so ${CUDART_SONAME} libcuda.so.1 ${INSTALL_DIR}/
	ln -sf ${CUDART_SONAME} ${INSTALL_DIR}/libcudart.so
	ln -sf libcuda.so.1 ${INSTALL_DIR}/libcuda.so
	cp -f libmana_cuda_audit.so ${MANA_ROOT}/lib/dmtcp/

clean:
	rm -rf gen *.o libmana_cuda.so libcudart.so.* libcuda.so.1 \
	  libmana_cuda_audit.so

.PHONY: default check-real install clean
