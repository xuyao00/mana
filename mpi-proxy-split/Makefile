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
#   libmana.so, the core: the DMTCP plugin, which keeps the split process and
#     calls the layers' checkpoint and restart steps (mana_core.h),
#   libmana_mpi.so, the MPI layer (using mpi-wrappers/libmpiwrappers.a),
#   libmpistub.so, libmpistub_audit.so,
#   mana_coordinator.o (needed to compile dmtcp_coordinator)

# Subdirectories needed to install
LOWER_HALF_SRCDIR=lower-half
WRAPPERS_SRCDIR=mpi-wrappers
LH_SRCDIR=lower-half

# Test subdirectory:
#   tests

# As you add new files to your plugin library, add the object file names here.

CORE_OBJS = mana_core.o uh_wrappers.o lower_half_ckpt.o \
            lower-half/switch-context.o
MPI_OBJS = mpi_plugin.o p2p_drain_send_recv.o p2p_log_replay.o seq_num.o \
           virtual_id.o
LIBOBJS = ${CORE_OBJS} ${MPI_OBJS}

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
# The upper half has no MPI library, so leave out the C++ bindings that
# mpi.h declares in C++ (Open MPI 4.1: MPI::Datatype::Free(), ...), whose
# code is in libmpi_cxx.so.
override CXXFLAGS += -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX

# ${WRAPPERS_SRCDIR}/libmpiwrappers.a ia a prerequisite for libmana.so
# Always do 'make default' or 'make install' -- not 'make libmana.so'
default: ${MANA_COORD_OBJS}
	make -C ${LH_SRCDIR} lower-half lower-half-nompi
	make -C ${WRAPPERS_SRCDIR} libmpiwrappers.a libmpistub.so \
	  libmpistub_audit.so
	make libmana.so libmana_mpi.so

libmana.so: ${CORE_OBJS}
	${CXX} -shared -fPIC -g3 -O0 -o $@ ${CORE_OBJS} -Wl,-soname,libmana.so

# ${WRAPPERS_SRCDIR}/libmpiwrappers.a made in install/default before
# libmana_mpi.so.  libmana_mpi.so needs libmana.so, and libmpistub.so (in the
# same directory once installed): with Open MPI, it refers to the MPI
# library's objects (see generate-mpi-stub-objects.py), and a program that
# loads the MPI library with dlopen() has not loaded libmpistub.so yet.
libmana_mpi.so: ${MPI_OBJS} ${WRAPPERS_SRCDIR}/libmpiwrappers.a libmana.so
	${CXX} -shared -fPIC -g3 -O0 -o $@ ${MPI_OBJS} -Wl,--whole-archive ${WRAPPERS_SRCDIR}/libmpiwrappers.a -Wl,--no-whole-archive \
	  -L${WRAPPERS_SRCDIR} -lmpistub -L. -lmana -Wl,-rpath,'$$ORIGIN'

mpi_unimplemented_wrappers.cpp: generate-mpi-unimplemented-wrappers.py \
                                mpi_unimplemented_wrappers.txt
	python $^ > $@

.c.o:
	${MPICC} ${CFLAGS} -c -o $@ $<

.cpp.o:
	${MPICXX} ${CXXFLAGS} -c -o $@ $<

# The core needs no MPI: the plain compiler, with the C++ standard that
# MPICXX sets.
$(filter-out lower-half/%,${CORE_OBJS}): %.o: %.cpp
	${CXX} -std=c++20 ${CXXFLAGS} -c -o $@ $<

# Make ignores prerequisites on suffix rules, so list header dependencies
# here: the headers that define inline code.
$(filter-out lower-half/%,${LIBOBJS}): virtual_id.h lower_half_ckpt.h mana_core.h

${LOWER_HALF_SRCDIR}/procmapsutils.o: ${LOWER_HALF_SRCDIR}/procmapsutils.c
	@make -C ${LOWER_HALF_SRCDIR} procmapsutils.o

vi vim: ${FILE}
	vim $<

touch: ${FILE}
	$@ $<

tests:
	+ @make -C test/ default
	+ @make -C examples/

check: libmana.so libmana_mpi.so
	@make -C test/ e2e
	@test/autotest.py

${MANA_ROOT}/lib/dmtcp/libmana.so: libmana.so
	cp -f $< $@
${MANA_ROOT}/lib/dmtcp/libmana_mpi.so: libmana_mpi.so
	cp -f $< $@
${MANA_ROOT}/lib/dmtcp/libmpistub.so: ${WRAPPERS_SRCDIR}/mpi_stub_wrappers.txt \
                                      ${WRAPPERS_SRCDIR}/mpi_stub_data.c
	+ make -C ${WRAPPERS_SRCDIR} install
${MANA_ROOT}/lib/dmtcp/libmpistub_audit.so: ${WRAPPERS_SRCDIR}/mpi_stub_audit.c
	+ make -C ${WRAPPERS_SRCDIR} install

# ${WRAPPERS_SRCDIR}/libmpiwrappers.a ia a prerequisite for libmana_mpi.so,
#   which is a prerequisite for ${MANA_ROOT}/lib/dmtcp/libmana_mpi.so
# Always do 'make default' or 'make install' -- not 'make libmana.so'
# MANA_COORD_OBJS needed for 'make mana' at top level.
install: ${MANA_COORD_OBJS}
	+ make -C ${WRAPPERS_SRCDIR} libmpiwrappers.a libmpistub.so
	+ make ${MANA_ROOT}/lib/dmtcp/libmana.so
	+ make ${MANA_ROOT}/lib/dmtcp/libmana_mpi.so
	+ make ${MANA_ROOT}/lib/dmtcp/libmpistub.so
	+ make ${MANA_ROOT}/lib/dmtcp/libmpistub_audit.so
	+ make -C ${LH_SRCDIR} install
	@# --cuda: the CUDA front, if the lower half's MPI uses CUDA.
	+ if ldd ${LH_SRCDIR}/lower-half 2>/dev/null | grep -q libcudart; then \
	    make -C cuda-front install; fi

tidy:
	rm -f *~ .*.swp dmtcp_restart_script*.sh ckpt_*.dmtcp
	rm -rf ckpt_rank_*
	rm -rf dmtcp_coordinator_db-*.json

clean: tidy
	+ make -C cuda-front clean
	rm -f ${LIBOBJS} ${MANA_COORD_OBJS}
	rm -f libmana.so libmana_mpi.so
	rm -f ${MANA_ROOT}/lib/dmtcp/libmana.so ${MANA_ROOT}/lib/dmtcp/libmana_mpi.so
	cd ${LOWER_HALF_SRCDIR} && make clean
	cd ${WRAPPERS_SRCDIR} && make clean

distclean: clean
	#cd test && make clean

dist: distclean
	dir=`basename $$PWD` && cd .. && tar czvf $$dir.tgz ./$$dir
	dir=`basename $$PWD` && ls -l ../$$dir.tgz

.PHONY: default clean dist distclean vi vim touch gdb tidy check \
        tests install
