PLATFORM=${shell echo $$HOST}
include ../Makefile_config

###############################################################
# NOTE:  More Fortran/C examples (under LGPL license) at:
#        https://people.math.sc.edu/Burkardt/f_src/mpi/mpi.html
###############################################################

ifndef DMTCP_ROOT
  DMTCP_ROOT=../../dmtcp
endif
DMTCP_BIN=${DMTCP_ROOT}/bin
DMTCP_LIB=${DMTCP_ROOT}/lib/dmtcp
DMTCP_INCLUDE=${DMTCP_ROOT}/include
# LIBNAME=libdmtcp_mpi-proxy
LIBNAME=libmana

# Benchmarks of MANA's overhead; built with -O2 and not run by 'make check'.
BENCH_FILES = virtid_bench synthetic_app

default: default_prereqs
	+ $(MAKE) ${BENCH_FILES} e2e

# End-to-end tests, run by autotest.py natively and under MANA.  Each test is
# one program, named after the test and linked as any MPI program.
E2E_TESTS = p2p_blocking p2p_nonblocking p2p_ring p2p_any_source p2p_probe \
            p2p_large p2p_proc_null collectives nonblocking_collectives \
            communicators cartesian datatypes memory init_thread \
            exit_after_finalize attributes misc_calls \
            finalize_unsync abort
E2E_FORTRAN = fortran_p2p fortran_nonblocking

# CUDA-aware MPI tests (run under mana_launch --cuda); built if nvcc is there.
NVCC ?= $(shell command -v nvcc 2>/dev/null)
E2E_CUDA = $(if $(NVCC),cuda_aware cuda_ordering)

# The suite also runs the example of ../examples (ring_example).
e2e: ${E2E_TESTS} ${E2E_FORTRAN} dlopen_mpi ${E2E_CUDA}
	+ $(MAKE) -C ../examples

# The application's CUDA runtime must be a shared library: the lower half's
# MPI uses it too (see ../cuda-forward).
cuda_aware cuda_ordering: %: %.cu mana_test.h
	$(NVCC) -g -O0 -ccbin $(firstword $(MPICXX)) -cudart=shared -o $@ $< \
	  $(CUDA_LIBS)
cuda_ordering: CUDA_LIBS = -lcuda

${E2E_TESTS}: %: %.c mana_test.h
	$(MPICC) -g3 -O0 -o $@ $< $(MPI_CFLAGS) ${MPI_LDFLAGS}

# Opens the MPI library with dlopen(), so it is not linked with it.
dlopen_mpi: dlopen_mpi.c
	$(MPICC) -g3 -O0 -c -o $@.o $< $(MPI_CFLAGS)
	$(CC) -g3 -o $@ $@.o -ldl

${E2E_FORTRAN}: %: %.f90
	$(MPIFORTRAN) ${FFLAGS} -g3 -O0 -o $@ $<

${BENCH_FILES}: %: %.c
	$(MPICC) -g -O2 -o $@ $< $(MPI_CFLAGS) ${MPI_LDFLAGS}

check: e2e
	./autotest.py

default_prereqs:
	@ # The darshan module is used at NERSC.
	@ if command -v module >/dev/null && module list 2>&1| grep -q darshan;\
	  then \
	    echo '** DMTCP is incompatible with the darshan module.'; \
	    echo '** Please do "module unload darshan" before building DMTCP.';\
	    exit 1; \
	fi
	@ # The altd module is used at NERSC.
	@ if command -v module >/dev/null && module list 2>&1| grep -q altd;\
	  then \
	    echo '** DMTCP is incompatible with the altd module.'; \
	    echo '** Please do "module unload altd" before building DMTCP.';\
	    exit 1; \
	fi

tidy:
	rm -f ckpt_*.dmtcp dmtcp_restart_script* \
	dmtcp-shared-memory.* dmtcp-test-typescript.tmp core*
	rm -rf ckpt_*
	rm -f dmtcp_coordinator_db-*.json

clean: tidy
	rm -f *.pyc *.so
	rm -f ${E2E_TESTS} ${E2E_FORTRAN} ${BENCH_FILES} cuda_aware cuda_ordering
