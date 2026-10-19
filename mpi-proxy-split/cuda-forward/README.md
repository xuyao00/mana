# CUDA-aware MPI in MANA

`mana_launch --cuda` (or `MANA_CUDA=1`) checkpoints and restarts MPI programs
that pass device memory to MPI, as with a CUDA-aware MPI such as MPICH or
Open MPI on UCX, or Cray MPICH. The GPU itself is checkpointed by the DMTCP
CUDA plugin (`dmtcp/plugin/cuda`), with NVIDIA's CUDA checkpoint API.
`mana_restart --cuda` restarts such a program.

```sh
mpirun -np 2 bin/mana_launch --cuda ./app
bin/dmtcp_command -bc
mpirun -np 2 bin/mana_restart --cuda
```

The support is built when `make` finds `cuda.h` and `libcuda.so.1`. The tests
are `cuda_aware` and `cuda_ordering` (`test/autotest.py`), and
`make -C mpi-proxy-split/cuda-forward check` tests the shims without MANA.

## The problem

The CUDA checkpoint API supports one CUDA driver per process. MANA's lower
half, with the MPI library, is not checkpointed: a new one is made at
restart, so CUDA state that belongs to a lower-half driver could not be
restored. And a CUDA-aware MPI uses CUDA itself: it copies device buffers,
registers them with the NIC, and shares them between processes with CUDA
IPC.

## The design

**One CUDA driver, the application's.** The lower half's `libcuda.so.1`,
`libcudart.so.<N>` and `libnvidia-ml.so.1` are forwarding shims
(`lh-shims/`, installed in `lib/dmtcp/cuda-forward/`). They are generated
from the real libraries (`gen-forward-shim.py`): each function is a
trampoline (`fwd-tramp.inc`) that switches the FS register to the calling
thread's upper-half TLS and jumps to the same function of the upper half's
library (`fwd-runtime.c`). So the MPI runs on the application's driver, in
the application's contexts, as natively. MANA registers each upper-half
thread with its stack, and a forwarded call finds its thread by its stack
pointer, without a system call. A function whose arguments and result are
all in registers (the generator classifies them by their types in `cuda.h`)
gets a fast trampoline, which calls the target with its own `call`
instruction and saves no vector registers: 10 ns per forwarded call instead
of 23 ns. UCX makes about 100 such calls per message besides its polling.

Some functions are written by hand (`cuda-special.c`, `cudart-special.c`,
`nvml-special.c`):

* Function pointers handed to the lower half (`cuGetProcAddress`,
  `cudaGetDriverEntryPoint`, the private tables of `cuGetExportTable`) get
  trampolines too.
* Kernel registrations of lower-half libraries (yaksa in MPICH) are recorded
  at load and replayed to the upper half's runtime once it is there.
* The lower half's host-memory registrations are released before a
  checkpoint and registered again on resume.
* Buffer ids (`CU_POINTER_ATTRIBUTE_BUFFER_ID`) change on each resume, since
  the GPU's checkpoint may move memory: the MPI's caches (e.g. libfabric's
  NIC registrations) must not match old entries.

**Virtual CUDA IPC handles.** A legacy IPC export (`cuIpcGetMemHandle` of
`cudaMalloc` memory) makes a process impossible to checkpoint. VMM memory
exported as a POSIX fd can be checkpointed, if no importer has it mapped.
So the shim's `cuIpcGetMemHandle` exports VMM memory as an fd, and returns a
handle that names it (pid, fd number, buffer id); `cuIpcOpenMemHandle` gets
the fd with `pidfd_getfd()` and maps it in a VA range that it keeps. Before a
checkpoint, importers unmap and exporters close their exports; on resume,
they export and map again, at the same fd numbers and addresses, so the MPI's
IPC cache stays valid. The shim reports only such VMM memory as legacy-IPC
capable; the MPI copies other memory.

**A VMM allocator** (`libmana_cuda_vmm.so`, `vmm-alloc.c`), loaded in the
upper half, makes the device allocations of the application (`cudaMalloc`,
also with a static CUDA runtime) VMM memory that can be exported as a POSIX
fd. It redirects the driver's internal pointers behind `cuMemAlloc_v2`,
`cuMemAllocPitch_v2` and `cuMemFree_v2`. It keeps two orderings of
`cudaMalloc` memory that VMM memory lacks: `cudaFree` waits for the device's
work, and a synchronous copy or set (`cudaMemcpy` between device buffers,
`cudaMemset`) is complete before the MPI accesses its memory. For the
latter, an application's copy records an event, and the next MPI call waits
for it (`lower_half_disable_ckpt()`); a copy that the MPI issues waits at
once.

**Checkpoint and restart** (`mpi_plugin.cpp`, `mana_cuda.cpp`):

* At launch, the lower half initializes MPI only at the application's
  `MPI_Init()`, when the application's CUDA is there.
* Checkpoint: after MANA's drain, pending receives into device memory are
  cancelled (the GPU's checkpoint may move the memory that the NIC would
  write), and the shims take down the lower half's CUDA state (host
  registrations, IPC mappings). Then the CUDA plugin checkpoints the GPU.
* Resume: after the CUDA plugin restored the GPU, the shims set up their
  state again (exports, a global barrier, then imports), the cancelled
  receives are posted again, and only then may the application's threads
  enter the lower half.
* Restart: the CUDA plugin restores the GPU in `RUNNING_AFTER`, with the
  threads running. MANA then initializes the new lower half's MPI and
  restores the MPI state, in its own `RUNNING_AFTER`. The rank, which picks
  the checkpoint image, comes from the launcher (`PMI_RANK`, ...).
* The upper half's mmap arena keeps the CUDA driver's preferred address
  range (from 0x200000000) free, honors `MAP_FIXED_NOREPLACE` and free
  address hints, and never maps over memory that the lower half mapped:
  the CUDA restore needs the driver's original addresses.

## Versions

* CUDA 12.4 or later (the CUDA checkpoint API), with a driver that supports
  it. The allocator's redirection was checked on drivers 550 to 615.
* The libcudart shim gets the soname of the CUDA runtime that the lower
  half's MPI uses (found from `bin/lower-half`'s libraries, or set
  `REAL_LIBCUDART`). The application may use another CUDA runtime, or a
  static one: the shim then loads the MPI's runtime in the upper half, and
  both run on the application's driver.
* The shims are generated from the installed driver: rebuild MANA
  (`make -C mpi-proxy-split install`) after a driver upgrade.
* If the allocator does not recognize the driver, it says so, and device
  memory stays `cudaMalloc` memory: the MPI then copies instead of using CUDA
  IPC (set `MANA_CUDA_VMM_ALLOC=0` to choose that).

## Environment variables

| Variable | Effect |
|---|---|
| `MANA_CUDA=1` | Same as `--cuda`. |
| `MANA_CUDA_VMM_ALLOC=0` | No VMM allocator, and no CUDA IPC (`UCX_TLS=^cuda_ipc`, `MPICH_GPU_IPC_ENABLED=0`). |
| `MANA_FWD_VERBOSE=1` | Logs what the shims and the allocator do. |
| `MANA_FWD_PROFILE=1` | At exit, prints the calls and TSC cycles of each forwarded function (fast trampolines then take the general path). |
| `MANA_FWD_VIRTUAL_IPC=0` | Real legacy CUDA IPC: no checkpoint then. |

`--cuda` also sets `FI_HMEM_CUDA_USE_GDRCOPY=0` (GDRCopy cannot pin memory
that the CUDA checkpoint API restored) and `DMTCP_DISABLE_PRGNAME_PREFIX=1`.

## Limitations

* Managed memory: the CUDA checkpoint API fails on it, also without MANA.
* CUDA calls from threads that the lower half creates itself have no upper
  half, and fail with "no device". UCX and the MPIs tested make none.
* Small allocations (under 1 MB) share 2 MB VMM chunks, so a CUDA IPC export
  shares the neighbors of the buffer in the same process.
* The CUDA driver must place its memory where it did before the checkpoint;
  this holds on x86-64 Linux.
