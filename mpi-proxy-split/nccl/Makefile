# NCCL in the lower half (see README): libmananccl.a, linked into libmana.so,
# and libmana_ncclstub.so, the upper half's libnccl.
PLATFORM=${shell echo $$HOST}
include ../Makefile_config

ifndef MANA_ROOT
  MANA_ROOT=../..
endif
ifndef DMTCP_ROOT
  DMTCP_ROOT=${MANA_ROOT}/dmtcp
endif
CUDA_HOME ?= $(shell d=$$(dirname $$(command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc))/..; echo $$d)
NCCL_H ?= $(firstword $(wildcard $(NCCL_HOME)/include/nccl.h /usr/include/nccl.h /usr/local/include/nccl.h))
NCCL_INCLUDE = $(dir $(NCCL_H))
# The real library, for the names of the stub's functions.
NCCL_LIB ?= $(firstword $(wildcard $(NCCL_HOME)/lib/libnccl.so.2) \
              $(shell /sbin/ldconfig -p 2>/dev/null | awk '/libnccl.so.2 / {print $$NF; exit}'))

# The lower half's FS is switched in, so no stack protector (see ../Makefile).
FLAGS = -fPIC -g3 -O0 -fno-stack-protector -I. -I.. -I../lower-half \
        -I${DMTCP_ROOT}/include -I${DMTCP_ROOT}/jalib -I${DMTCP_ROOT}/src \
        -I${CUDA_HOME}/include -I${NCCL_INCLUDE} \
        -Dckpt_pending=mana_nccl_ckpt_pending -Dg_my_rank=mana_nccl_my_rank \
        -Dg_nranks=mana_nccl_nranks
# libmana.so is preloaded into every MANA process: only the NCCL API is
# visible from the code of nccl_wrappers.c and nccl_forward.c.
HIDDEN = -fvisibility=hidden

HIDDEN_OBJS = vcomm.o inflight.o nccl_ckpt.o nccl_cuda.o dmtcp_kvdb_c.o \
              nccl_lh.o nccl_lh_thunks.o
API_OBJS = nccl_wrappers.o nccl_forward.o
OBJS = ${HIDDEN_OBJS} ${API_OBJS}

default: libmananccl.a libmana_ncclstub.so

libmananccl.a: ${OBJS}
	rm -f $@
	ar cr $@ $^

nccl_lh_thunks.cpp: gen-nccl-wrappers.py ${NCCL_H}
	python3 gen-nccl-wrappers.py thunks ${NCCL_H} > $@
nccl_forward.c: gen-nccl-wrappers.py ${NCCL_H} nccl_wrappers.c
	python3 gen-nccl-wrappers.py forward ${NCCL_H} nccl_wrappers.c > $@
nccl_stub_names.txt: ${NCCL_LIB}
	nm -D --defined-only $< | awk '$$2 == "T" {print $$3}' > $@
nccl_stub.c: gen-nccl-wrappers.py nccl_stub_names.txt
	python3 gen-nccl-wrappers.py stub nccl_stub_names.txt > $@

# The upper half loads it in place of libnccl (../mpi-wrappers/mpi_stub_audit.c).
libmana_ncclstub.so: nccl_stub.c
	gcc -shared -fPIC -O2 -Wl,-soname,libnccl.so.2 -o $@ $<

${HIDDEN_OBJS:.o=.o}: nccl_lh.h nccl_wrappers.h vcomm.h inflight.h
nccl_wrappers.o nccl_forward.o nccl_ckpt.o: nccl_cuda.h nccl_lh.h

vcomm.o inflight.o nccl_ckpt.o nccl_cuda.o: %.o: %.c
	gcc ${FLAGS} ${HIDDEN} -std=gnu11 -c -o $@ $<
nccl_wrappers.o nccl_forward.o: %.o: %.c
	gcc ${FLAGS} -std=gnu11 -c -o $@ $<
dmtcp_kvdb_c.o: dmtcp_kvdb_c.cpp
	g++ ${FLAGS} ${HIDDEN} -c -o $@ $<
nccl_lh.o nccl_lh_thunks.o: %.o: %.cpp
	${MPICXX} ${FLAGS} ${HIDDEN} -c -o $@ $<

install: default
	cp -f libmana_ncclstub.so ${MANA_ROOT}/lib/dmtcp/

clean:
	rm -f *.o libmananccl.a libmana_ncclstub.so nccl_lh_thunks.cpp \
	      nccl_forward.c nccl_stub.c nccl_stub_names.txt

.PHONY: default install clean
