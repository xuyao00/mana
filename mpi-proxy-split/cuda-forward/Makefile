# MANA's support of CUDA-aware MPI (mana_launch --cuda); see README.md.
#   - lh-shims/: the lower half's forwarding shims of libcuda.so.1,
#     libcudart.so.<N> and libnvidia-ml.so.1, and their runtime libmana_fwd.so;
#   - libmana_cuda_vmm.so: the upper half's device-memory allocator.
# 'make install' copies them to $(MANA_ROOT)/lib/dmtcp/.

ifndef MANA_ROOT
  MANA_ROOT=../..
endif
CUDA_HOME ?= /usr/local/cuda

# The shims are generated from the real libraries.  The libcudart shim must
# have the soname of the CUDA runtime that the lower half's MPI uses: that
# of bin/lower-half's libraries, or else the toolkit's.
REAL_LIBCUDA ?= $(shell /sbin/ldconfig -p | awk '/libcuda.so.1 / {print $$NF; exit}')
REAL_LIBNVML ?= $(shell /sbin/ldconfig -p | awk '/libnvidia-ml.so.1 / {print $$NF; exit}')
REAL_LIBCUDART ?= $(or \
  $(shell ldd $(MANA_ROOT)/bin/lower-half 2>/dev/null | \
          awk '/libcudart\.so\.[0-9]+ / && $$3 !~ /cuda-forward/ {print $$3; exit}'), \
  $(lastword $(sort $(wildcard $(CUDA_HOME)/lib64/libcudart.so.[0-9]*))))
CUDART_SONAME := $(shell readelf -d $(REAL_LIBCUDART) 2>/dev/null | \
                   sed -n 's/.*(SONAME).*\[\(.*\)\]/\1/p')

CC = gcc
CXX = g++
# The shims switch the FS register: no stack protector, as in all of MANA.
CFLAGS = -g -O2 -fPIC -fno-stack-protector -Wall -I. -I$(CUDA_HOME)/include
OUT = lh-shims
SHIMS = $(OUT)/libmana_fwd.so $(OUT)/libcuda.so.1 $(OUT)/$(CUDART_SONAME) \
        $(OUT)/libcudart.so $(OUT)/libnvidia-ml.so.1

default: check-cudart $(SHIMS) libmana_cuda_vmm.so

check-cudart:
	@if [ -z "$(CUDART_SONAME)" ]; then \
	  echo "cuda-forward: no CUDA runtime found; set REAL_LIBCUDART"; exit 1; \
	fi

libmana_cuda_vmm.so: vmm-alloc.c
	$(CC) $(CFLAGS) -shared -o $@ $< -L$(dir $(REAL_LIBCUDA)) -l:libcuda.so.1 \
	  -ldl -lpthread

$(OUT) gen:
	mkdir -p $@

$(OUT)/libmana_fwd.so: fwd-runtime.c fwd-dyn.S fwd-tramp.inc fwd-runtime.h | $(OUT)
	$(CC) $(CFLAGS) -shared -o $@ fwd-runtime.c fwd-dyn.S -Wl,-soname,libmana_fwd.so

# gen-forward-shim.py writes PREFIX-tramp.S, PREFIX-desc.c and PREFIX.map;
# the last arguments are the functions written by hand (*-special.c).
gen/cuda.stamp: gen-forward-shim.py $(REAL_LIBCUDA) | gen
	./gen-forward-shim.py --header=cuda.h --cxx="$(CXX) -I$(CUDA_HOME)/include" \
	  $(REAL_LIBCUDA) libcuda.so.1 cuda gen \
	  cuGetProcAddress cuGetProcAddress_v2 cuGetExportTable \
	  cuMemHostRegister cuMemHostRegister_v2 cuMemHostUnregister \
	  cuMemRetainAllocationHandle cuIpcGetMemHandle cuIpcOpenMemHandle \
	  cuIpcOpenMemHandle_v2 cuIpcCloseMemHandle cuPointerGetAttribute \
	  cuPointerGetAttributes
	touch $@

gen/cudart.stamp: gen-forward-shim.py $(REAL_LIBCUDART) | gen
	./gen-forward-shim.py $(REAL_LIBCUDART) $(CUDART_SONAME) cudart gen \
	  cudaGetDriverEntryPoint cudaGetDriverEntryPointByVersion \
	  __cudaRegisterFatBinary __cudaRegisterFatBinaryEnd \
	  __cudaUnregisterFatBinary __cudaRegisterFunction __cudaRegisterVar \
	  __cudaRegisterManagedVar __cudaInitModule \
	  cudaHostRegister cudaHostUnregister
	touch $@

gen/nvml.stamp: gen-forward-shim.py $(REAL_LIBNVML) | gen
	./gen-forward-shim.py $(REAL_LIBNVML) libnvidia-ml.so.1 nvml gen \
	  nvmlInternalGetExportTable
	touch $@

$(OUT)/libcuda.so.1: gen/cuda.stamp cuda-special.c fwd-runtime.h $(OUT)/libmana_fwd.so
	$(CC) $(CFLAGS) -shared -o $@ gen/cuda-tramp.S gen/cuda-desc.c \
	  cuda-special.c -Wl,-soname,libcuda.so.1 -Wl,--version-script,gen/cuda.map \
	  -L$(OUT) -lmana_fwd -Wl,-rpath,'$$ORIGIN'

$(OUT)/$(CUDART_SONAME): gen/cudart.stamp cudart-special.c fwd-runtime.h $(OUT)/libmana_fwd.so
	$(CC) $(CFLAGS) -DCUDART_SONAME='"$(CUDART_SONAME)"' -shared -o $@ \
	  gen/cudart-tramp.S gen/cudart-desc.c cudart-special.c \
	  -Wl,-soname,$(CUDART_SONAME) -Wl,--version-script,gen/cudart.map \
	  -L$(OUT) -lmana_fwd -Wl,-rpath,'$$ORIGIN'

# libfabric (Slingshot's CXI provider) dlopen()s "libcudart.so", without a
# version: it must get the shim too, not a second CUDA runtime.
$(OUT)/libcudart.so: $(OUT)/$(CUDART_SONAME)
	ln -sf $(CUDART_SONAME) $@

$(OUT)/libnvidia-ml.so.1: gen/nvml.stamp nvml-special.c fwd-runtime.h $(OUT)/libmana_fwd.so
	$(CC) $(CFLAGS) -shared -o $@ gen/nvml-tramp.S gen/nvml-desc.c \
	  nvml-special.c -Wl,-soname,libnvidia-ml.so.1 \
	  -Wl,--version-script,gen/nvml.map -L$(OUT) -lmana_fwd -Wl,-rpath,'$$ORIGIN'

# A test of the libcuda shim without MANA (needs a GPU).
test-fwd: test-fwd.c fwd-runtime.h $(OUT)/libcuda.so.1
	$(CC) $(CFLAGS) -o $@ $< -L$(OUT) -l:libcuda.so.1 -lmana_fwd \
	  -Wl,-rpath,'$$ORIGIN/$(OUT)' -ldl -lpthread

check: test-fwd
	REAL_LIBCUDA=$(REAL_LIBCUDA) ./test-fwd

install: default
	mkdir -p $(MANA_ROOT)/lib/dmtcp/cuda-forward
	rm -f $(MANA_ROOT)/lib/dmtcp/cuda-forward/libcudart.so*
	cp -df $(OUT)/* $(MANA_ROOT)/lib/dmtcp/cuda-forward/
	cp -f libmana_cuda_vmm.so $(MANA_ROOT)/lib/dmtcp/

clean:
	rm -rf $(OUT) gen libmana_cuda_vmm.so test-fwd

.PHONY: default check-cudart check install clean
