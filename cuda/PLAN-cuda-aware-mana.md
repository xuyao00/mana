# CUDA-aware MPI in MANA: design, prototype status, Perlmutter plan

Prototype machine: lab-gpu, 1x RTX 3070 Laptop, driver 595.91.07, CUDA 13.1,
MPICH 5.0.1 `ch4:ucx --with-cuda`, UCX 1.24. CUDA C/R: the DMTCP CUDA plugin
(`dmtcp/plugin/cuda`, unmodified). Everything is uncommitted in the working
tree (branch `cuda-aware`). A report with diagrams: `cuda/report/index.html`.

## Requirements (agreed)

1. The lower half's MPI (UCX; Cray GTL on Perlmutter) works on the
   application's device memory -- no staging in MANA.
2. CUDA C/R stays in DMTCP (the CUDA plugin).
3. Prototype here, then test on Perlmutter (target: Cray MPICH).
4. The memory-placement fixes go in the same PR.
5. No NCCL (separate project). Lazy `MPI_Init` is OK.

## Constraints (verified)

* The CUDA checkpoint API handles **one CUDA driver per process**.
* MANA **replaces the lower half at restart**; CUDA state owned by a
  lower-half driver cannot be restored.
* **No CUDA IPC state survives a checkpoint**: a legacy export
  (`cuIpcGetMemHandle`) or open import makes `cuCheckpointProcessCheckpoint`
  fail (304) and the process die; VMM/mempool POSIX-fd exports can be
  checkpointed, but importers must hold no mapping, and a handle exported
  before the checkpoint maps the old memory after it.
* `cuCheckpointProcessRestore` needs the threads running (it hangs in
  `DMTCP_EVENT_RESTART`), and needs the driver's original address layout.

## Design

1. **One CUDA driver, the application's, in the upper half. The lower half's
   `libcuda.so.1`, `libcudart.so.13`, `libnvidia-ml.so.1` are forwarding shims**
   (`mpi-proxy-split/cuda-forward/`): generated trampolines switch the FS
   register to the calling thread's upper-half TLS and jump to the upper
   half's function. Hand-written parts: function pointers handed out
   (`cuGetProcAddress*`, `cudaGetDriverEntryPoint*`), private tables
   (`cuGetExportTable`, `nvmlInternalGetExportTable`), kernel registration
   recorded at load and replayed into the application's cudart, host-memory
   registrations (released before a checkpoint, restored on resume).
2. **Virtual CUDA IPC handles** in the shim: VMM POSIX-fd memory is reported
   legacy-IPC capable; `cuIpcGetMemHandle` returns `{magic, exporter pid, fd
   number, size}` backed by one POSIX-fd export; `cuIpcOpenMemHandle` does
   `pidfd_getfd` + import + map in a reserved VA, which UCX caches. Before a
   checkpoint importers unmap (VA kept) and exporters park their fd numbers;
   on resume exporters re-export onto the same fd numbers, then (barrier)
   importers re-map at the same VA. At restart UCX's cache is new anyway.
   `cudaMalloc` memory is reported not IPC capable (UCX uses its copy path).
3. **VMM device allocator** in the upper half (`libmana_cuda_vmm.so`):
   `cuMemAlloc_v2`, `cuMemAllocPitch_v2`, `cuMemFree_v2` of the real driver
   become POSIX-fd shareable VMM allocations, so `cudaMalloc` buffers can be
   shared with (virtual) IPC. The driver's exported functions are stubs
   (`call *slot(%rip)`); the library redirects the slot, so it catches a
   static cudart and `cuGetProcAddress` pointers too. Allocations < 1 MB are
   carved out of 2 MB VMM chunks.
4. MANA:
   * lazy lower-half `MPI_Init` (at the application's `MPI_Init`);
   * restart: rank from the launcher's environment, memory restored without
     `MPI_Init`, GPU restored by the CUDA plugin in `RUNNING_AFTER`, then MPI
     init + replay in MANA's `RUNNING_AFTER` (runs after the plugin's); the
     lower-half gate keeps application threads out until then;
   * resume: the gate stays closed until the shims restored host
     registrations and IPC mappings (global barrier between re-export and
     re-import);
   * every upper-half thread is registered with the shims with its stack
     range (thread lookup without a system call: 20 ns per forwarded call);
   * **drain fix**: the drain round's barrier keeps MPI progressing
     (a rendezvous that needs the sender's progress -- device-to-host staged
     through a device fragment -- deadlocked the blocking barrier);
   * memory placement: the upper-half mmap arena no longer clobbers foreign
     mappings, honors `MAP_FIXED_NOREPLACE` and free hints, keeps CUDA's UVA
     range `0x200000000` free (else `cuCheckpointProcessRestore` fails);
   * `mana_launch --cuda` / `mana_restart --cuda` (or `MANA_CUDA=1`).

## Status on this machine

| Test | Result |
|---|---|
| MANA e2e suite, default | 23/24 (known `cartesian` bug) |
| MANA e2e suite, `MANA_CUDA=1` (all tests through lazy init, deferred restart) | 23/24 (same) |
| `test/cuda_aware` (Sendrecv, pending device Irecv across ckpt, in-place Allreduce), 3 C/R cycles | 10/10 runs after the drain fix (3/5 failed before) |
| plain `cudaMalloc` app, virtual IPC + UCX IPC cache, 3 generations ckpt/resume/ckpt/restart | passes, data checked every iteration |
| VMM allocator, shared and static cudart (`vmm_alloc_test.cu`) | passes |
| Checkpoint/restart with 64 MB-2 GB device memory per rank | passes, data verified |
| Managed memory | checkpoint API: 801 (driver/GPU limit, also without MANA) |

Ping-pong on device buffers, 2 ranks, same GPU (median of 3 runs):

| bytes | native (cuda_ipc) | native, no IPC | native + VMM allocator | MANA, no IPC | **MANA, virtual IPC** |
|---|---|---|---|---|---|
| 8 | 89.8 us | 108.1 us | 90.3 us | 140.1 us | **114.4 us** |
| 4 KB | 87.8 us | 92.1 us | 88.3 us | 102.9 us | **95.3 us** |
| 256 KB | 3.0 GB/s | 3.0 GB/s | 3.0 GB/s | 2.7 GB/s | **2.7 GB/s** |
| 1 MB | 11.7 GB/s | 4.1 GB/s | 11.6 GB/s | 3.9 GB/s | **10.8 GB/s** |
| 4 MB | 37.2 GB/s | 5.0 GB/s | 37.0 GB/s | 4.9 GB/s | **34.8 GB/s** |
| 16 MB | 91.1 GB/s | 5.3 GB/s | 90.9 GB/s | 5.2 GB/s | **87.5 GB/s** |

* MANA + virtual IPC: 92-96% of native bandwidth from 1 MB, 17x MANA
  without IPC at 16 MB; small messages +7 us (+8%).
* The VMM allocator alone costs nothing measurable; `cudaMalloc`+`cudaFree`
  takes about as long as native (77-83 us vs 94 us small, 447 vs 545 us at
  16 MB, ~equal at 256 MB).
* Forwarding: UCX makes ~370 CUDA calls per small message (polling); a
  forwarded call costs 20 ns (4.5 ns direct) since threads are found by their
  stack, 190 ns before (two `gettid` calls).
* MANA itself adds nothing on host-buffer messages (0.2-0.5 us at 8 B).

Checkpoint/restart cost (2 ranks, image per rank):

| device MB/rank | allocator | checkpoint | image | restart (to first progress) |
|---|---|---|---|---|
| 64 | cudaMalloc | 0.67 s | 514 MB | 2.3 s |
| 64 | VMM | 0.66 s | 514 MB | 2.3 s |
| 512 | cudaMalloc | 1.28 s | 962 MB | 2.9 s |
| 512 | VMM | 1.07 s | 962 MB | 2.9 s |
| 2048 | cudaMalloc | 3.39 s | 2498 MB | 4.3 s |
| 2048 | VMM | 3.18 s | 2498 MB | 4.7 s |

## Open items

1. CUDA calls from threads the lower half creates (no upper-half TLS)
   return "no device" (none seen with UCX). Fix: an upper-half proxy thread.
2. UCX's memory hooks (UCM) warn about `cuMemFree` of a suballocated pointer
   natively (not under MANA, where UCM hooks are off): give UCM the chunk, or
   use exact-size VMM allocations when UCM is active.
3. Small allocations share a chunk, so an IPC export exposes neighbors of the
   same process; memory overhead: up to 2 MB per device and size class.
4. Orphaned CUDA objects of an old lower half are restored at each restart
   and leak (small; bounded by restarts).
5. A checkpoint request while MANA's deferred replay runs in `RUNNING_AFTER`
   is not tested.
6. `lower-half-api.h`/`fwd-runtime.h` changes need a clean rebuild (no
   header dependencies in the Makefiles).
7. Peer access between different GPUs of a node (`cuMemSetAccess` is set
   for the importer's device): to test on Perlmutter.
8. The `0x200000000` reservation and the hint policy: confirm on other
   drivers (Perlmutter A100).
9. UCX picks IPC rendezvous from 32 KB on; below 256 KB its copy path is
   faster (natively too): a UCX threshold tuning question.

## Perlmutter plan (Cray MPICH)

1. Build with `cudatoolkit`; check `cuCheckpointProcess*` with plain DMTCP
   (`make -C dmtcp/plugin/cuda check`).
2. Regenerate shims for `libcudart.so.12`; check what Cray MPICH/GTL loads
   with CUDA (`ldd libmpi_gtl_cuda.so`, `LD_DEBUG=libs`).
3. GTL uses legacy IPC (`cuIpcGetMemHandle`): the virtual IPC handles and the
   VMM allocator should apply unchanged -- verify; fall back to
   `MPICH_GPU_IPC_ENABLED=0`.
4. Tests: `cuda_aware` 2 ranks/1 GPU; 4 ranks/node with `--gpus-per-task=1`
   (GPU binding at restart); 2 nodes.
5. Main risk: inter-node GPU RDMA (libfabric/cxi registers device memory
   with the NIC, cached): does the checkpoint API accept it? If not, release
   registrations after the drain, like host registrations.
6. MANA suite with `MANA_CUDA=1`, a real application, C/R timing.

## Files

* `mpi-proxy-split/cuda-forward/`: shims, runtime, virtual IPC
  (`cuda-special.c`), VMM allocator (`vmm-alloc.c`).
* MANA: `lower-half/{lower-half.cpp,mem-wrapper.cpp,lower-half-api.h}`,
  `mpi_plugin.cpp`, `uh_wrappers.{cpp,h}`, `mpi-wrappers/mpi_wrappers.cpp`,
  `p2p_drain_send_recv.cpp`, `bin/mana_launch`, `bin/mana_restart`,
  `mpi-proxy-split/Makefile`, `test/{cuda_aware.cu,autotest.py,Makefile}`.
* Experiment and benchmark programs: `cuda/experiments/`; report:
  `cuda/report/index.html`.
