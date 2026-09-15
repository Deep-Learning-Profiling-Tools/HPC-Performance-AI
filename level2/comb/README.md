# Comb

Comb is LLNL's configurable structured-mesh communication proxy application.
This Level 2 case repeatedly packs, exchanges, unpacks, and verifies two-cell
periodic halos for a three-dimensional mesh. It represents the ECP
communication motif: native GPU packing kernels coupled to GPU-aware MPI. The
selected execution and memory policies are Comb's direct CUDA and HIP paths;
RAJA is disabled.

## Provenance

Upstream repository: https://github.com/LLNL/Comb
CUDA source: `develop`, commit `e0a74ee38f6e69ba4e9915c57dc9ad5249a8b687`
HIP source: same LLNL `develop` commit, using its native HIP implementation
Bundled BLT commit: `296bf64e64edfcfcce6a53e3b396d6529e76b986`
License: MIT -- upstream `LICENSE` and `NOTICE` retained under `upstream/`;
`LICENSE` also copied to this directory's root

The HIP implementation is authoritative because it is maintained in LLNL's
Comb repository alongside CUDA. The upstream interface provides `hip`,
`hip_device`, and `-hip_aware_mpi` modes. No HIP translation or portability
backend was created here.

Copied into this directory:

- `upstream/` -- pinned Comb source and required BLT submodule, without git
  metadata, generated files, or the unused RAJA submodule.
- `LICENSE` -- root-level copy of the upstream license.
- `build.sh`, `run.sh`, `validate.sh` -- Level 2 wrappers.

## Changes from upstream

1. `upstream/include/exec_utils_cuda.hpp`: made NVTX optional and adapted the
   integer device ordinal to CUDA 13's `cudaMemLocation` API. Runtime semantics
   are unchanged; upstreamable.
2. `upstream/include/memory.hpp`: uses the CUDA compatibility helper above.
   Allocation and memory-advice behavior are unchanged; upstreamable.
3. `upstream/include/exec_utils_hip.hpp`: includes rocTX only when the
   corresponding upstream option is enabled. Kernels are unchanged;
   upstreamable.
4. `upstream/include/profiling.hpp`: guards NVTX and rocTX calls with the
   correct upstream options. Only profiling annotations are affected;
   upstreamable.
5. The Level 2 wrappers provide separate build directories, architecture
   selection, the common MPI launcher, a short workload, and an automatic
   correctness gate. No algorithm was changed.

## Dependencies

- CMake and a C++14 compiler.
- MPI C/C++ wrappers.
- CUDA variant: `nvcc`, CUDA runtime, and CUDA-aware MPI.
- HIP variant: ROCm `hipcc` and HIP-aware MPI.
- No Kokkos, RAJA, OpenMP target, SYCL, or OpenACC backend is used.

CUDA architecture is detected automatically. Override it with
`HPCPERF_CUDA_ARCH=90`. HIP accepts a site-selected target such as
`HPCPERF_HIP_ARCH=gfx942`; no AMD architecture is hard-coded.

## Backends

CUDA: native Comb CUDA execution/device-memory policies, locally validated.
HIP: native LLNL HIP execution/device-memory policies, integrated but not
locally compiled or executed.

## Build

```bash
source hpcperf_env.sh
level2/comb/build.sh             # CUDA (default)
level2/comb/build.sh HIP         # requires ROCm
```

Outputs are isolated under `build/level2/comb/cuda/` and
`build/level2/comb/hip/`. The executable is `bin/comb` in the selected
backend directory.

## Run

```bash
HPCPERF_GPUS=1 level2/comb/run.sh
HPCPERF_GPUS=2 level2/comb/run.sh
HPCPERF_GPUS=1 level2/comb/run.sh HIP
HPCPERF_GPUS=2 level2/comb/run.sh HIP
```

The common launcher starts one MPI rank per GPU and maps MPI local rank to a
scheduler-visible device. `hpcperf_topology.py` selects a three-dimensional
process grid whose product is exactly `HPCPERF_GPUS`. Every rank owns one
subdomain and exchanges halos with neighboring ranks, so multi-GPU execution
is one distributed problem rather than independent jobs.

The default is 128 cubed cells per rank, three variables, a two-cell halo, and
100 cycles. The global mesh weak-scales with the process grid. Controls:
`HPCPERF_COMB_LOCAL_SIZE`, `HPCPERF_COMB_CYCLES`,
`HPCPERF_COMB_VARIABLES`, and `HPCPERF_GPUS`.

Observed on one NVIDIA B200: approximately 13 seconds end to end.

## Validate

```bash
HPCPERF_GPUS=1 level2/comb/validate.sh
HPCPERF_GPUS=2 level2/comb/validate.sh
```

## Validation

Comb initializes mesh and halo values from global coordinates. Its native
`test-comm` phase asserts every received value after MPI communication.
`NDEBUG` is deliberately not defined, so an incorrect halo aborts and
`validate.sh` returns nonzero. Validation also requires the native
`Comm mpi Mesh cuda` marker.

CUDA clean build, one-GPU execution, correctness, meaningful GPU work, and
rank-to-device mapping passed. The machine exposed one NVIDIA GPU, so the
two-GPU runtime test is `NOT-TESTED-HARDWARE-UNAVAILABLE`.

Future AMD validation:

```bash
HPCPERF_HIP_ARCH=gfx942 level2/comb/build.sh HIP
HPCPERF_GPUS=1 level2/comb/validate.sh HIP
HPCPERF_GPUS=2 level2/comb/validate.sh HIP
```

Both runs must exit zero, print `Comm mpi Mesh hip`, contain no assertion
failure, and show the expected rank/device map.

## Warnings

- The standard workload requires working GPU-aware MPI.
- Multi-node transport remains site-specific.
- HIP compilation and runtime were unavailable on the CUDA-only development
  system.

## LOC

Comb application source: 16,603 lines of C/C++ in `upstream/include/` and
`upstream/src/` (BLT and build files excluded).

## Verified Environment

GCC/G++ 13.3.0 (conda) | C++14 | CMake 3.28.4 | Ninja 1.13.2
CUDA Toolkit 13.2 (nvcc 13.2.78, `/usr/local/cuda`) | NVIDIA B200 (sm_100)
Open MPI 5.0.10 (conda) | Slurm allocation with one visible GPU
HIP/ROCm: authoritative source and build/run configuration present, unverified

Reproduce the common toolchain from the repository root with
`./setup_env.sh`, then `source hpcperf_env.sh`.

## Status

CUDA Build: PASS
CUDA Single GPU: PASS
CUDA Multi-GPU: NOT-TESTED-HARDWARE-UNAVAILABLE
CUDA Correctness: PASS
HIP Integration: INTEGRATED-NOT-LOCALLY-VALIDATED
HIP Build: NOT-TESTED-HARDWARE-UNAVAILABLE
HIP Runtime: NOT-TESTED-HARDWARE-UNAVAILABLE

HIP runtime was not validated locally because AMD GPU hardware is unavailable.
