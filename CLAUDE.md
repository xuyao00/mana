# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

MANA (MPI-Agnostic, Network-Agnostic) does transparent checkpoint-restart of MPI
programs. It is a DMTCP plugin; DMTCP lives in the `dmtcp/` git submodule. All
MANA code is in `mpi-proxy-split/`. The user manual is the GitHub wiki
(https://github.com/mpickpt/mana/wiki). `mpi-proxy-split/NOTES` and `doc-old/`
date from 2020 and are mostly out of date.

## Build

```sh
git submodule update --init
./configure            # or ./configure-mana (MPI_INSTALL_DIR=..., --enable-debug)
make -j8 mana          # dmtcp, then mpi-proxy-split install + tests
```

- Plain `make` also runs prereq checks and symlinks `util/hooks/{pre-commit,post-rewrite}` into `.git/hooks`.
- Outputs go to `bin/` (`lower-half`, the `mana_*` scripts, DMTCP binaries) and `lib/dmtcp/` (`libmana.so`, `libmpistub.so`, `libmpistub_audit.so`, symlinks named after MPI libraries, and `cuda-forward/` when CUDA is present).
- After you edit MANA sources, `make -C mpi-proxy-split install` rebuilds and installs the lower half, the wrappers and `libmana.so`.
- **Header dependencies are mostly untracked.** The Makefiles only list `virtual_id.h` and `lower_half_ckpt.h`. After you change any other header, run `make -C mpi-proxy-split clean install`. This matters most for `lower-half/lower-half-api.h`: `LowerHalfInfo_t` is shared by the `bin/lower-half` binary and `libmana.so`, and its layout must match in both.
- MPI compilers come from `mpi-proxy-split/Makefile_config` (generated from `Makefile_config.in`). When `$HOST` contains `nid0` or `login` (NERSC Perlmutter), the build uses `cc`/`CC`/`ftn` and `srun`; otherwise it uses `mpicc`/`mpic++`/`mpifort`.
- Everything is built with `-fno-stack-protector`, because the wrappers switch the FS register. Don't remove it.
- At NERSC, unload the `darshan` and `altd` modules before you build, and build on a login node.
- On Debian/Ubuntu, `libc6-dbg` is required.
- In containers, run with `--security-opt seccomp=unconfined`.
- `DMTCP_ROOT=...` points the build at a different DMTCP tree. Changes to DMTCP itself belong upstream in the `dmtcp` repo, followed by a submodule bump.
- The top-level `.gitignore` ignores every file named `Makefile`, because autotools generates some of them. Hand-written Makefiles must be added with `git add -f`.

## Tests

`mpi-proxy-split/test/autotest.py` runs end-to-end tests. Each test launches under MANA, then checkpoints, kills and restarts it `--cycles` times (default 2).

```sh
make -C mpi-proxy-split/test e2e                 # build test programs
mpi-proxy-split/test/autotest.py                 # all tests (or: make -C mpi-proxy-split check)
mpi-proxy-split/test/autotest.py p2p_ring        # one or more named tests
mpi-proxy-split/test/autotest.py --list
mpi-proxy-split/test/autotest.py --native        # run without MANA, to check the test itself
mpi-proxy-split/test/autotest.py --keep --workdir /tmp/at   # keep run dirs/logs
mpi-proxy-split/test/autotest.py --launcher 'srun -n {n}' --slow 3
```

- Test programs use `test/mana_test.h`. They loop until autotest writes a stop iteration to `$MT_CONTROL/stop`, report `"<name>: rank R: iteration I"`, and call `MPI_Abort` on any wrong value (`MT_CHECK`, `MT_MPI`).
- To add a test, register it in **both** the `TESTS` list in `autotest.py` (ranks, kind `loop`/`run`, args, `cuda=`, `known_bug=`) and `E2E_TESTS`/`E2E_FORTRAN` in `test/Makefile`.
- Each run gets a private `HOME` and coordinator port, so tests don't touch `~/.mana.rc`.
- CI (`.github/workflows/mana-tests.yml`) runs on PRs to `main`: Ubuntu 26.04 with MPICH (ch4:ucx), `./configure && make mana`, then `autotest.py --retry-once`.
- `virtid_bench` and `synthetic_app` are overhead benchmarks built with `-O2`. They are not part of the suite (see `test/README-synthetic_app.md`).

To run by hand:

1. Start the coordinator with `bin/mana_coordinator`. It writes `~/.mana.rc`, or `~/.mana-slurm-$SLURM_JOB_ID.rc` under Slurm, which `mana_launch` reads.
2. Launch with `mpirun -np N bin/mana_launch ./prog`.
3. Checkpoint with `bin/dmtcp_command -bc`, then kill the job with `-k`.
4. Restart with `mpirun -np N bin/mana_restart` in the directory that holds `ckpt_rank_*/`.

`mana_launch` refuses to start if old `ckpt_rank_*` directories are in the checkpoint directory.

## Style

`.clang-format` is Google-based (2-space indent). The pre-commit hook runs `git diff --check` and `util/dmtcp-style.py` on staged files; the linter also checks license headers.

## Architecture: the split process

Each MPI rank is one process with two halves.

**Lower half** (`mpi-proxy-split/lower-half/`, built as `bin/lower-half`)
- A standalone program linked against the real MPI library, with its text at `0x10000000`. It calls `MPI_Init` and fills `LowerHalfInfo_t lh_info` (`lower-half-api.h`) with its FS base, the MPI constants (`MANA_COMM_WORLD`, ...), `lh_dlsym`, and mmap wrappers.
- It then works as a kernel loader (`lower-half.cpp`): it maps the application's `ld.so`, builds the upper-half stack and auxv, patches `ld.so`'s `mmap`/`munmap` with trampolines to `mem-wrapper.cpp`, and jumps into `ld.so`.
- The upper half finds `lh_info` through `MANA_LH_INFO_ADDR`.

**Launch chain**
- `bin/mana_launch` (Python) execs `dmtcp_launch --kernel-loader --with-plugin <MANA_PRELOAD libs>:libmana.so[:user/CUDA plugins] bin/lower-half <app> <args>`.

**Upper half**
- Holds the application, DMTCP, and `libmana.so` (DMTCP plugin, `mpi_plugin.cpp`; `mpi_plugin_event_hook` handles DMTCP events).
- The application's MPI library is never loaded there. The LD_AUDIT module `mpi_stub_audit.c` swaps in `libmpistub.so`, whose functions all `assert(0)`. So a binary built against any MPICH-ABI MPI runs unchanged.
- The `PMPI_*` wrappers in `libmana.so` take every MPI call (`#pragma weak MPI_X = PMPI_X`).

**Wrapper pattern** (`mpi-wrappers/mpi_*_wrappers.cpp`):
```c
LOWER_HALF_DISABLE_CKPT();                         // lower_half_ckpt.h: checkpoint gate
MPI_Comm real = get_real_id((mana_mpi_handle){.comm = comm}).comm;
JUMP_TO_LOWER_HALF(lh_info->fsaddr);               // switch FS to the lower half's TLS
retval = NEXT_FUNC(Comm_size)(real, size);         // via lh_dlsym(MPI_Fnc_Comm_size)
RETURN_TO_UPPER_HALF();
/* new handles: *newcomm = new_virt_comm(*newcomm); */
LOWER_HALF_ENABLE_CKPT();
```
- Nothing may leave the block between `JUMP_TO_LOWER_HALF` and `RETURN_TO_UPPER_HALF` (no `return`, `goto`, `break`, or exception), or FS is never restored.
- Calls that create communicators are bracketed by `commit_begin`/`commit_finish` (`seq_num.cpp`).

**Adding or implementing an MPI function**
1. Add it to `FOREACH_FNC` in `lower-half-api.h`. That list generates `enum MPI_Fncs` and the lower half's function-pointer table (`lh-func-ptr.c`).
2. Write the `PMPI_` wrapper.
3. Remove it from `mpi_unimplemented_wrappers.txt`.
4. Make sure it is in `mpi_stub_wrappers.txt`.
5. If Fortran needs it, add it to `mpi_fortran_wrappers.txt`.

The `.txt` files are inputs to `generate-mpi-*.py`. Edit them, not the generated `mpi_stub_wrappers.c`, `mpi_unimplemented_wrappers.cpp` or `mpi_fortran_wrappers.cpp`.

**Virtual IDs** (`virtual_id.{h,cpp}`)
- Every handle the application sees is virtual: 32 bits made of kind, generation and slot, stored in a chunked table.
- Each entry holds the real lower-half handle plus a descriptor that records how to rebuild it at restart: group ranks, Cartesian topology, datatype constructor and arguments, user op.

**Checkpoint** (the lower half is never saved)
- `dmtcp_skip_memory_region_ckpting()` in `mpi_plugin.cpp` excludes lower-half memory.
- `DMTCP_EVENT_PRESUSPEND`:
  1. Waits for `MPI_Init` to finish.
  2. Drains collectives with the Collective Clock (`seq_num.cpp`): per-group sequence numbers and targets, so every rank stops at a consistent point (Cluster'24 paper).
  3. Drains point-to-point messages (`p2p_drain_send_recv.cpp`): in-flight messages are counted and received into buffers.
  4. Completes pending non-blocking collectives.
  5. Closes the lower half: no thread may be inside it when DMTCP suspends threads.
- `MANA_P2P_WAIT=polling|blocking` controls how `MPI_Send`/`MPI_Recv` wait. Blocking mode makes the lower half run with `MPI_THREAD_MULTIPLE` and gives the checkpoint thread an extra lower-half TLS. Application `MPI_THREAD_MULTIPLE` is otherwise unsupported: there is a single pending-recv slot.
- `PRECHECKPOINT` saves the MANA header and MPI-file state, and names the checkpoint directory `ckpt_rank_<N>`.

**Restart**
1. `mana_restart` starts a fresh `bin/lower-half --restore`. It calls `MPI_Init`, picks its image by rank, and restores upper-half memory itself (`restoreMemoryArea`, not DMTCP's `mtcp_restart`).
2. It jumps to DMTCP's `postRestart`.
3. `DMTCP_EVENT_RESTART` in the plugin rebinds wrappers to the new lower half (`uh_wrappers.cpp`), rebuilds virtual IDs, replays pending `Isend`/`Irecv` (`p2p_log_replay.cpp`), serves drained messages, and reopens MPI files.
- Upper-half fd numbers are kept free in the new lower half by `MANA_RESERVED_FDS` / `reserve_restart_fds()`.

**CUDA-aware MPI** (`--cuda` or `MANA_CUDA=1`; design in `cuda/PLAN-cuda-aware-mana.md`)
- Only one CUDA driver may exist per process: the application's. The lower half's `libcuda`/`libcudart`/`libnvidia-ml` are generated forwarding shims (`mpi-proxy-split/cuda-forward/`).
- `libmana_cuda_vmm.so` turns device allocations into VMM memory, so CUDA IPC handles can be virtualized.
- The DMTCP CUDA plugin (`dmtcp/plugin/cuda`) checkpoints the GPU.
- In this mode the lower half's `MPI_Init` is lazy. At restart, MPI init and replay are deferred to `DMTCP_EVENT_RUNNING_AFTER`, after the GPU is restored.
- Built only if `cuda.h` and `libcuda.so.1` are found. `cuda_aware` is the test.
- `cuda-forward/Makefile` generates the shims from the real libraries with `gen-forward-shim.py`. The shim's soname must match the application's CUDA runtime (`libcudart.so.13`).

**Perlmutter (Cray MPICH, Slingshot)**
- Build with `module load cudatoolkit/13.0 craype-accel-nvidia80` and `MPICH_GPU_SUPPORT_ENABLED=1`. Cray's GTL (`libmpi_gtl_cuda`) needs `libcudart.so.13`, and the accel module makes `CC` link GTL into `bin/lower-half`.
- Under `--cuda`, `mana_launch`/`mana_restart` set `FI_HMEM_CUDA_USE_GDRCOPY=0`: GDRCopy can't pin memory restored by the CUDA checkpoint API. With `MANA_CUDA_VMM_ALLOC=0` they also set `MPICH_GPU_IPC_ENABLED=0`.
- Compute nodes can't see the login node's `/tmp`; put scripts and run directories under `$SCRATCH`. Under Slurm, `mana_coordinator` writes `~/.mana-slurm-$SLURM_JOB_ID.rc`.
- To run tests across nodes: `autotest.py --launcher 'srun -N 2 -n {n} --gpu-bind=none'`. `--gpu-bind=none` keeps all 4 GPUs visible to each rank.

## Runtime environment variables

| Variable | Effect |
|---|---|
| `MANA_P2P_WAIT` | `polling` or `blocking`; see Checkpoint above. |
| `MANA_PRELOAD` | Colon-separated libraries, e.g. a PMPI tool, loaded ahead of `libmana.so`. |
| `MANA_CUDA` | Same as `--cuda`. |
| `MANA_CUDA_VMM_ALLOC` | Set to `0` to turn off the VMM allocator (this also turns off CUDA IPC). |
| `MANA_DRAIN_STATS` | Prints drain timings. |
| `MANA_TIMING` | Same as `--timing`. |
| `MANA_QUIET` | Same as `-q`. |
