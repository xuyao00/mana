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
	+ $(MAKE) $(if $(filter yes,${MANA_MPI}),${BENCH_FILES}) e2e

# End-to-end tests, run by autotest.py natively and under MANA.  Each test is
# one program, named after the test and linked as any MPI program.
E2E_TESTS = p2p_blocking p2p_nonblocking p2p_ring p2p_any_source p2p_probe \
            p2p_large p2p_proc_null collectives nonblocking_collectives \
            communicators cartesian datatypes memory init_thread \
            exit_after_finalize attributes misc_calls \
            finalize_unsync abort p2p_ssend p2p_ssend_sync \
            p2p_ssend_collective p2p_ssend_probe p2p_ssend_wildcard \
            p2p_ssend_order p2p_ssend_comm p2p_ssend_blocked \
            p2p_issend_envelope p2p_issend_twice p2p_issend_nbx \
            shared_memory mmap_threads comm_free keyvals
E2E_FORTRAN = fortran_p2p fortran_nonblocking fortran_handles fortran_attributes
# Without MPI (mana_launch --no-mpi), built with the plain compilers.
E2E_NOMPI = nompi
# CUDA tests (mana_launch --cuda); built with the CUDA front (configure's
# MANA_CUDA).  All but cuda_nompi use MPI.
NVCC = $(if $(filter yes,${MANA_CUDA}),${CUDA_HOME}/bin/nvcc)
E2E_CUDA_MPI = cuda_aware cuda_threads cuda_driver cuda_async cuda_ipc \
               cuda_textures cuda_static cuda_static_stripped cuda_cublas \
               cuda_cufft cuda_curand cuda_graphs
E2E_CUDA = $(if $(NVCC),cuda_nompi cuda_quiesce cuda_pools cuda_managed \
             cuda_drvgraphs \
             $(if $(filter yes,${MANA_MPI}),${E2E_CUDA_MPI}))

# What MANA builds (configure's MANA_MPI and MANA_CUDA): the MPI tests with
# the MPI layer, and the suite also runs the example of ../examples
# (ring_example).
e2e: $(if $(filter yes,${MANA_MPI}),${E2E_TESTS} ${E2E_FORTRAN} dlopen_mpi) \
     ${E2E_NOMPI} ${E2E_CUDA}
ifeq (${MANA_MPI},yes)
	+ $(MAKE) -C ../examples
endif

${E2E_TESTS}: %: %.c mana_test.h
	$(MPICC) -g3 -O0 -o $@ $< $(MPI_CFLAGS) ${MPI_LDFLAGS}

mmap_threads: MPI_LDFLAGS += -pthread

# With the shared CUDA runtime: the front stands for it under --cuda.
cuda_aware cuda_async cuda_textures: %: %.cu mana_test.h
	$(NVCC) -g -O0 -ccbin $(firstword $(MPICXX)) -cudart=shared -o $@ $<

# The libraries: their front libraries stand for them too.
cuda_cublas: %: %.cu mana_test.h
	$(NVCC) -g -O0 -ccbin $(firstword $(MPICXX)) -cudart=shared -o $@ $< \
	  -lcublas -lcublasLt

cuda_cufft: %: %.cu mana_test.h
	$(NVCC) -g -O0 -ccbin $(firstword $(MPICXX)) -cudart=shared -o $@ $< \
	  -lcufft

cuda_curand: %: %.cu mana_test.h
	$(NVCC) -g -O0 -ccbin $(firstword $(MPICXX)) -cudart=shared -o $@ $< \
	  -lcurand

# cuda_aware with nvcc's default, a static CUDA runtime; and stripped.
cuda_static: cuda_aware.cu mana_test.h
	$(NVCC) -g -O0 -ccbin $(firstword $(MPICXX)) -DTEST_NAME='"cuda_static"' \
	  -o $@ $<

cuda_static_stripped: cuda_static
	strip -o $@ $<

cuda_ipc cuda_graphs: %: %.cu mana_test.h
	$(NVCC) -g -O0 -ccbin $(firstword $(MPICXX)) -cudart=shared -o $@ $< \
	  -lcuda

# Also the _ptsz/_ptds functions of the per-thread default stream.
cuda_threads: %: %.cu mana_test.h
	$(NVCC) -g -O0 -ccbin $(firstword $(MPICXX)) -cudart=shared \
	  --default-stream per-thread -o $@ $< -lpthread

# The driver API only, with its kernels as PTX and as a fatbin.
cuda_driver: cuda_driver.c cuda_driver_kernels.h mana_test.h
	$(MPICC) -g -O0 -I$(CUDA_HOME)/include -o $@ $< -lcuda

cuda_driver_kernels.h: cuda_driver_kernels.cu
	$(NVCC) -ptx -o cuda_driver_kernels.ptx $<
	$(NVCC) -fatbin -o cuda_driver_kernels.fatbin $<
	$(CUDA_HOME)/bin/bin2c -c -n driver_ptx cuda_driver_kernels.ptx > $@
	$(CUDA_HOME)/bin/bin2c -c -n driver_fatbin cuda_driver_kernels.fatbin \
	  >> $@

nompi: nompi.c mana_test.h
	$(CC) -g3 -O0 -o $@ $< -pthread

cuda_nompi: cuda_nompi.cu mana_test.h
	$(NVCC) -g -O0 -cudart=shared -o $@ $<

# The driver API only, without MPI.
cuda_drvgraphs: cuda_drvgraphs.c cuda_driver_kernels.h mana_test.h
	$(CC) -g -O0 -I$(CUDA_HOME)/include -o $@ $< -L$(CUDA_HOME)/lib64 -lcuda

cuda_managed: cuda_managed.cu mana_test.h
	$(NVCC) -g -O0 -cudart=shared -o $@ $<

cuda_pools: cuda_pools.cu mana_test.h
	$(NVCC) -g -O0 -cudart=shared -o $@ $<

cuda_quiesce: cuda_quiesce.cu mana_test.h
	$(NVCC) -g -O0 -cudart=shared -o $@ $< -lpthread

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
	rm -f ${E2E_TESTS} ${E2E_FORTRAN} ${BENCH_FILES} ${E2E_CUDA}
	rm -f cuda_driver_kernels.h cuda_driver_kernels.ptx \
	  cuda_driver_kernels.fatbin
