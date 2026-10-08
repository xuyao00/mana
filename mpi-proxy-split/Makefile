# MPI compiler according to the platform
PLATFORM=${shell echo $$HOST}
include Makefile_config

# These are needed for libmana.so, since the wrapper functions there will
# call the lower half.  That requires temporarily changing the FS register,
# and apparently, glibc detects this as an attempt to do stack smashing.
override CFLAGS += -fno-stack-protector
override CXXFLAGS += -fno-stack-protector

ifeq (${MPICXX},)
  MPICXX = PLEASE_DEFINE_MPICXX
endif

ifeq (${MPICC},)
  MPICC = PLEASE_DEFINE_MPICC
endif

# Targets to install:
#   libmana.so (using mpi-wrappers/libmpiwrappers.a), libmpistub.so,
#   libmpistub_audit.so,
#   mana_coordinator.o (needed to compile dmtcp_coordinator)

# Subdirectories needed to install
# CUDA-aware MPI support is built if there are cuda.h and libcuda.so.1.
CUDA_HOME ?= $(shell d=$$(dirname $$(command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc))/..; echo $$d)
HAVE_CUDA = $(and $(wildcard $(CUDA_HOME)/include/cuda.h),$(shell /sbin/ldconfig -p 2>/dev/null | grep -q 'libcuda.so.1 ' && echo yes))
# NCCL in the lower half (nccl/): with CUDA, if there are nccl.h and libnccl.
NCCL_H = $(firstword $(wildcard $(NCCL_HOME)/include/nccl.h /usr/include/nccl.h /usr/local/include/nccl.h))
HAVE_NCCL = $(and $(HAVE_CUDA),$(NCCL_H),$(or $(wildcard $(NCCL_HOME)/lib/libnccl.so.2),$(shell /sbin/ldconfig -p 2>/dev/null | grep -q 'libnccl.so.2 ' && echo yes)))
ifneq ($(HAVE_NCCL),)
  NCCL_ARCHIVE = nccl/libmananccl.a
  override CXXFLAGS += -DMANA_NCCL
endif

LOWER_HALF_SRCDIR=lower-half
WRAPPERS_SRCDIR=mpi-wrappers
LH_SRCDIR=lower-half

# Test subdirectory:
#   tests

# As you add new files to your plugin library, add the object file names here.

LIBOBJS = uh_wrappers.o mpi_plugin.o p2p_drain_send_recv.o p2p_log_replay.o \
          seq_num.o virtual_id.o lower_half_ckpt.o \
          lower-half/switch-context.o \

#MANA_COORD_OBJS = mana_coordinator.o

LIBPROXY = libproxy.a

# Modify if your MANA_ROOT is located elsewhere.
ifndef MANA_ROOT
  MANA_ROOT=..
endif
# Modify if your DMTCP_ROOT is located elsewhere.
ifndef DMTCP_ROOT
  DMTCP_ROOT=${MANA_ROOT}/dmtcp
endif
DMTCP_INCLUDE=${DMTCP_ROOT}/include
JALIB_INCLUDE=${DMTCP_ROOT}/jalib

override CFLAGS += -fPIC -I${DMTCP_INCLUDE} -I${LOWER_HALF_SRCDIR}
override CXXFLAGS += -fPIC -I${DMTCP_INCLUDE} -I${JALIB_INCLUDE} \
                     -I${WRAPPERS_SRCDIR} -I. \
                     -I${DMTCP_ROOT}/src -I${LOWER_HALF_SRCDIR}

# ${WRAPPERS_SRCDIR}/libmpiwrappers.a ia a prerequisite for libmana.so
# Always do 'make default' or 'make install' -- not 'make libmana.so'
default: ${MANA_COORD_OBJS}
	make -C ${LH_SRCDIR} lower-half
	make -C ${WRAPPERS_SRCDIR} libmpiwrappers.a
	make libmana.so
	make -C ${WRAPPERS_SRCDIR} libmpistub.so libmpistub_audit.so

# ${WRAPPERS_SRCDIR}/libmpiwrappers.a made in install/default before libmana.so
libmana.so: ${LIBOBJS} ${WRAPPERS_SRCDIR}/libmpiwrappers.a ${NCCL_ARCHIVE}
	${CXX} -shared -fPIC -g3 -O0 -o $@ ${LIBOBJS} -Wl,--whole-archive ${WRAPPERS_SRCDIR}/libmpiwrappers.a ${NCCL_ARCHIVE} -Wl,--no-whole-archive -ldl

nccl/libmananccl.a: FORCE
	+ make -C nccl libmananccl.a

FORCE:

mpi_unimplemented_wrappers.cpp: generate-mpi-unimplemented-wrappers.py \
                                mpi_unimplemented_wrappers.txt
	python $^ > $@

.c.o:
	${MPICC} ${CFLAGS} -c -o $@ $<

.cpp.o:
	${MPICXX} ${CXXFLAGS} -c -o $@ $<

# Make ignores prerequisites on suffix rules, so list header dependencies
# here: the headers that define inline code.
$(filter-out lower-half/%,${LIBOBJS}): virtual_id.h lower_half_ckpt.h

${LOWER_HALF_SRCDIR}/procmapsutils.o: ${LOWER_HALF_SRCDIR}/procmapsutils.c
	@make -C ${LOWER_HALF_SRCDIR} procmapsutils.o

vi vim: ${FILE}
	vim $<

touch: ${FILE}
	$@ $<

tests:
	+ @make -C test/ default
	+ @make -C examples/

check: libmana.so
	@make -C test/ e2e
	@test/autotest.py

${MANA_ROOT}/lib/dmtcp/libmana.so: libmana.so
	cp -f $< $@
${MANA_ROOT}/lib/dmtcp/libmpistub.so: ${WRAPPERS_SRCDIR}/mpi_stub_wrappers.txt \
                                      ${WRAPPERS_SRCDIR}/mpi_stub_data.c
	+ make -C ${WRAPPERS_SRCDIR} install
${MANA_ROOT}/lib/dmtcp/libmpistub_audit.so: ${WRAPPERS_SRCDIR}/mpi_stub_audit.c
	+ make -C ${WRAPPERS_SRCDIR} install

# ${WRAPPERS_SRCDIR}/libmpiwrappers.a ia a prerequisite for libmana.so,
#   which is a prerequisite for ${MANA_ROOT}/lib/dmtcp/libmana.so
# Always do 'make default' or 'make install' -- not 'make libmana.so'
# MANA_COORD_OBJS needed for 'make mana' at top level.
install: ${MANA_COORD_OBJS}
	+ make -C ${WRAPPERS_SRCDIR} libmpiwrappers.a
	+ make ${MANA_ROOT}/lib/dmtcp/libmana.so
	+ make ${MANA_ROOT}/lib/dmtcp/libmpistub.so
	+ make ${MANA_ROOT}/lib/dmtcp/libmpistub_audit.so
	+ make -C ${LH_SRCDIR} install
	@ # CUDA-aware MPI (mana_launch --cuda): with a CUDA toolkit and driver.
	+ if [ -n "$(HAVE_CUDA)" ]; then \
	    make -C cuda-forward install && \
	    make -C ${DMTCP_ROOT}/plugin/cuda libdmtcp_cuda.so && \
	    cp -f ${DMTCP_ROOT}/plugin/cuda/libdmtcp_cuda.so ${MANA_ROOT}/lib/dmtcp/; \
	  fi
	+ if [ -n "$(HAVE_NCCL)" ]; then make -C nccl install; fi

tidy:
	rm -f *~ .*.swp dmtcp_restart_script*.sh ckpt_*.dmtcp
	rm -rf ckpt_rank_*
	rm -rf dmtcp_coordinator_db-*.json

clean: tidy
	- make -C cuda-forward clean
	- make -C nccl clean
	rm -f ${LIBOBJS} ${MANA_COORD_OBJS}
	rm -f libmana.so
	rm -f ${MANA_ROOT}/lib/dmtcp/libmana.so
	cd ${LOWER_HALF_SRCDIR} && make clean
	cd ${WRAPPERS_SRCDIR} && make clean

distclean: clean
	#cd test && make clean

dist: distclean
	dir=`basename $$PWD` && cd .. && tar czvf $$dir.tgz ./$$dir
	dir=`basename $$PWD` && ls -l ../$$dir.tgz

.PHONY: default clean dist distclean vi vim touch gdb tidy check \
        tests install
