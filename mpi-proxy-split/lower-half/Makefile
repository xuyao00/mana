# MPI compiler according to the platform
UNAME_RELEASE=${shell uname -r}
include ../Makefile_config

# Modify if your MANA_ROOT is located elsewhere.
ifndef MANA_ROOT
  MANA_ROOT=../..
endif
# Modify if your DMTCP_ROOT is located elsewhere.
ifndef DMTCP_ROOT
  DMTCP_ROOT=${MANA_ROOT}/dmtcp
endif

ifndef PLUGIN_ROOT
  PLUGIN_ROOT=..
endif

DMTCP_INCLUDE=${DMTCP_ROOT}/include
JALIB_INCLUDE=${DMTCP_ROOT}/jalib
MANA_INCLUDE=${MANA_ROOT}/mpi-proxy-split
PLUGIN_INCLUDE=${PLUGIN_ROOT}

DMTCP_DEPS = ${DMTCP_ROOT}/src/libdmtcprestart.a \
	     ${DMTCP_ROOT}/src/libdmtcpinternal.a \
	     ${DMTCP_ROOT}/src/libjalib.a \
	     ${DMTCP_ROOT}/src/libnohijack.a

override CFLAGS += -fPIC -I${DMTCP_INCLUDE} -I${MANA_INCLUDE} -g3 -O0 \
                   -I${DMTCP_ROOT}/src 
override CXXFLAGS += -fPIC -I${DMTCP_INCLUDE} -I${JALIB_INCLUDE} -I${MANA_INCLUDE} -I${DMTCP_ROOT}/src/mtcp \
                     -I${DMTCP_ROOT}/src

TEXTSEG_ADDR_FLAG=-Wl,-Ttext-segment=0x10000000

# bin/lower-half with the MPI layer (configure's MANA_MPI), and
# bin/lower-half-nompi always.
LOWER_HALVES = $(if $(filter yes,${MANA_MPI}),lower-half) lower-half-nompi

default: ${LOWER_HALVES}

.cpp.o:
	${MPICXX} ${CXXFLAGS} -c -o $@ $<

.c.o:
	${MPICXX} ${CFLAGS} -c -o $@ $<

# With the CUDA front (configure's MANA_CUDA): virtual legacy CUDA IPC for
# the MPI (lh-ipc.cpp), whose functions bin/lower-half exports.
ifeq (${MANA_CUDA},yes)
  LH_IPC_OBJS = lh-ipc.o
  LH_IPC_LDFLAGS = -Wl,--dynamic-list=lh-ipc.list
endif

lh-ipc.o: lh-ipc.cpp
	${MPICXX} ${CXXFLAGS} -I${CUDA_HOME}/include -c -o $@ $<

# Normally, gcc (or clang) will automatically link with libatomic.so.
# However, for come MPICXX it's hard to know if they cover it.
# So this is good as a safety measure.
lower-half: get-symbol-offset.o build-id-debug.o copy-stack.o patch-trampoline.o lh-func-ptr.o mem-wrapper.o switch-context.o lower-half.o mmap-fixed-noreplace.o lh-mpi.o ${LH_IPC_OBJS} ${DMTCP_DEPS}
	${MPICXX} -g3 ${TEXTSEG_ADDR_FLAG} ${CXXFLAGS} ${LH_IPC_LDFLAGS} -o $@ $^ -lpthread -ldl -latomic

# bin/lower-half-nompi: the same lower half without MPI, with lh-nompi.cpp
# in place of lh-mpi.cpp and lh-func-ptr.c.  Built with the plain compiler,
# so that it runs where no MPI library is installed.
NOMPI_OBJS = $(addprefix nompi/,get-symbol-offset.o build-id-debug.o \
               copy-stack.o patch-trampoline.o mem-wrapper.o switch-context.o \
               lower-half.o mmap-fixed-noreplace.o lh-nompi.o)

lower-half-nompi: ${NOMPI_OBJS} ${DMTCP_DEPS}
	${CXX} -g3 ${TEXTSEG_ADDR_FLAG} ${CXXFLAGS} -o $@ $^ -lpthread -ldl -latomic

# With the C++ standard that MPICXX sets.
nompi/%.o: %.cpp
	@mkdir -p nompi
	${CXX} -std=c++20 ${CXXFLAGS} -c -o $@ $<

nompi/%.o: %.c
	@mkdir -p nompi
	${CXX} -std=c++20 ${CFLAGS} -c -o $@ $<

build-id-debug.o lower-half.o nompi/build-id-debug.o nompi/lower-half.o: \
  build-id-debug.h
lower-half.o lh-mpi.o nompi/lower-half.o nompi/lh-nompi.o: lh-mpi.h
lh-mpi.o: lower-half-mpi.h

install: ${LOWER_HALVES}
	cp -f $^ ${MANA_ROOT}/bin/

tidy:
	rm -f *~ .*.swp dmtcp_restart_script*.sh ckpt_*.dmtcp
	rm -rf ckpt_rank_*
	rm -rf dmtcp_coordinator_db-*.json

clean: tidy
	rm -f lower-half lower-half-nompi
	rm -f ${MANA_ROOT}/bin/lower-half ${MANA_ROOT}/bin/lower-half-nompi
	rm -f *.o
	rm -rf nompi

distclean: clean

dist: distclean
	dir=`basename $$PWD` && cd .. && tar czvf $$dir.tgz ./$$dir
	dir=`basename $$PWD` && ls -l ../$$dir.tgz

.PHONY: default clean dist distclean install
