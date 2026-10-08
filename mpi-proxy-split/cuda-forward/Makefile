# Forwarding shims of libcuda.so.1 and libcudart.so.13 for MANA's lower half
# (see fwd-runtime.c).  'make' builds them in ./lh-shims; 'make install'
# copies that directory to $(MANA_ROOT)/lib/dmtcp/cuda-forward.

ifndef MANA_ROOT
  MANA_ROOT=../..
endif
CUDA_HOME ?= /usr/local/cuda
REAL_LIBCUDA ?= $(shell /sbin/ldconfig -p | awk '/libcuda.so.1 / {print $$NF; exit}')
REAL_LIBCUDART ?= $(CUDA_HOME)/lib64/libcudart.so.13
REAL_LIBNVML ?= $(shell /sbin/ldconfig -p | awk '/libnvidia-ml.so.1 / {print $$NF; exit}')

CC = gcc
CFLAGS = -g -O2 -fPIC -Wall -I. -I$(CUDA_HOME)/include
OUT = lh-shims

default: $(OUT)/libmana_fwd.so $(OUT)/libcuda.so.1 $(OUT)/libcudart.so.13 \
  $(OUT)/libnvidia-ml.so.1 libmana_cuda_vmm.so

# The upper half's device allocator (VMM memory, for checkpointable CUDA
# IPC); not in $(OUT), which is the lower half's library directory.
libmana_cuda_vmm.so: vmm-alloc.c
	$(CC) $(CFLAGS) -shared -o $@ $< -L$(dir $(REAL_LIBCUDA)) -l:libcuda.so.1 -ldl -lpthread

$(OUT):
	mkdir -p $@

$(OUT)/libmana_fwd.so: fwd-runtime.c fwd-dyn.S fwd-tramp.inc fwd-runtime.h | $(OUT)
	$(CC) $(CFLAGS) -shared -o $@ fwd-runtime.c fwd-dyn.S -Wl,-soname,libmana_fwd.so

gen/cuda-tramp.S gen/cuda-desc.c: gen-forward-shim.py
	mkdir -p gen
	./gen-forward-shim.py $(REAL_LIBCUDA) libcuda.so.1 cuda gen \
	  cuGetProcAddress cuGetProcAddress_v2 cuGetExportTable \
	  cuMemHostRegister cuMemHostRegister_v2 cuMemHostUnregister \
	  cuMemRetainAllocationHandle cuIpcGetMemHandle cuIpcOpenMemHandle \
	  cuIpcOpenMemHandle_v2 cuIpcCloseMemHandle cuPointerGetAttribute cuPointerGetAttributes

gen/cudart-tramp.S gen/cudart-desc.c: gen-forward-shim.py
	mkdir -p gen
	./gen-forward-shim.py $(REAL_LIBCUDART) libcudart.so.13 cudart gen \
	  cudaGetDriverEntryPoint cudaGetDriverEntryPointByVersion \
	  __cudaRegisterFatBinary __cudaRegisterFatBinaryEnd __cudaUnregisterFatBinary \
	  __cudaRegisterFunction __cudaRegisterVar __cudaRegisterManagedVar __cudaInitModule \
	  cudaHostRegister cudaHostUnregister

$(OUT)/libcuda.so.1: gen/cuda-tramp.S gen/cuda-desc.c cuda-special.c $(OUT)/libmana_fwd.so
	$(CC) $(CFLAGS) -shared -o $@ gen/cuda-tramp.S gen/cuda-desc.c cuda-special.c \
	  -Wl,-soname,libcuda.so.1 -L$(OUT) -lmana_fwd -Wl,-rpath,'$$ORIGIN'

$(OUT)/libcudart.so.13: gen/cudart-tramp.S gen/cudart-desc.c cudart-special.c $(OUT)/libmana_fwd.so
	$(CC) $(CFLAGS) -shared -o $@ gen/cudart-tramp.S gen/cudart-desc.c cudart-special.c \
	  -Wl,-soname,libcudart.so.13 -Wl,--version-script,gen/cudart.map -L$(OUT) -lmana_fwd -Wl,-rpath,'$$ORIGIN'

gen/nvml-tramp.S gen/nvml-desc.c: gen-forward-shim.py
	mkdir -p gen
	./gen-forward-shim.py $(REAL_LIBNVML) libnvidia-ml.so.1 nvml gen \
	  nvmlInternalGetExportTable

$(OUT)/libnvidia-ml.so.1: gen/nvml-tramp.S gen/nvml-desc.c nvml-special.c $(OUT)/libmana_fwd.so
	$(CC) $(CFLAGS) -shared -o $@ gen/nvml-tramp.S gen/nvml-desc.c nvml-special.c \
	  -Wl,-soname,libnvidia-ml.so.1 -L$(OUT) -lmana_fwd -Wl,-rpath,'$$ORIGIN'

install: default
	mkdir -p $(MANA_ROOT)/lib/dmtcp/cuda-forward
	cp -f $(OUT)/* $(MANA_ROOT)/lib/dmtcp/cuda-forward/
	cp -f libmana_cuda_vmm.so $(MANA_ROOT)/lib/dmtcp/

clean:
	rm -rf $(OUT) gen libmana_cuda_vmm.so

.PHONY: default install clean
