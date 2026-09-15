# SW4lite

SW4lite is the ECP proxy application for performance-critical kernels from the
SW4 seismic-wave application. The selected point-source case advances the
elastic wave equations over an MPI spatial decomposition, runs native GPU
finite-difference kernels, and compares the final field with an analytical
solution. Main motif: structured seismic stencils with MPI boundary exchange.

## Provenance

ECP catalog: https://proxyapps.exascaleproject.org/app/sw4lite/
CUDA repository: https://github.com/geodynamics/sw4lite
CUDA source: `master`, commit `06b888cd991c61e4b0168ec31b55e9af4135843a`
HIP source: former AMD/ECP `https://github.com/rwvo/sw4lite/tree/hip`,
commit `41e9a74d0eedcc0903645c51b75f34450a9c1b0f`
HIP archive: Software Heritage directory
`swh:1:dir:fb884670df41e16b0e364f20cbf40d87f9da5d47`, snapshot
`swh:1:snp:30fceac252ff23f2e0510bc1b4877cd858233410`
License: GNU GPL with LLNL notice -- copied to `LICENSE`

The HIP source is the native AMD/ECP port used by the ecosystem's SW4lite HIP
work. It was authored by AMD engineer Rene van Oostrum and contains native HIP
kernels plus its own `Makefile.hip`/hipify build procedure. The original fork
is no longer online, so the exact pinned revision was recovered from Software
Heritage. It was not recreated from CUDA for this integration.

Copied into this directory:

- `source/cuda/` -- minimal application source from geodynamics/sw4lite.
- `source/hip/` -- minimal source from the archived AMD/ECP HIP revision.
- `inputs/pointsource.in` -- upstream analytical point-source workload.
- `LICENSE`, `build.sh`, `run.sh`, and `validate.sh`.

Generated binaries, build trees, git metadata, tests not required by this
workload, and CI files were not copied.

## Changes from upstream

1. `source/cuda/CMakeLists.txt`: uses the retained production-kernel include
   directory `src/double` instead of the omitted standalone
   `tests/testil` directory. Numerical kernels are unchanged; upstreamable as
   conditional include discovery.
2. `source/hip/Makefile.hip`: applies `HIP_ARCH_FLAGS` during compilation
   and linking. This exposes architecture selection without changing the
   algorithm; upstreamable.
3. `build.sh` isolates backend build trees, discovers MPI/LAPACK, performs
   the authoritative HIP-generation step only under `build/`, and supplies
   selectable GPU architecture flags.
4. `run.sh` and `validate.sh` add the common launcher and turn the upstream
   analytical comparison into an exit-status correctness gate.

## Dependencies

- CMake or GNU make.
- C++ and Fortran compilers.
- MPI C++ headers and libraries.
- LAPACK; OpenBLAS is sufficient.
- CUDA variant: `nvcc` and the CUDA runtime.
- HIP variant: ROCm `hipcc`, `hipify-perl`, ROCm-compatible MPI, and LAPACK.
- No OpenMP target, OpenACC, Kokkos, SYCL, or other replacement backend.

CUDA architecture is detected automatically; override it with
`HPCPERF_CUDA_ARCH=90`. HIP accepts a target such as
`HPCPERF_HIP_ARCH=gfx942`, applied during compilation and linking.

## Backends

CUDA: native geodynamics SW4lite CUDA implementation, locally validated.
HIP: authoritative AMD/ECP HIP implementation, integrated but not locally
compiled or executed.

## Build

```bash
source hpcperf_env.sh
level2/sw4lite/build.sh             # CUDA (default)
level2/sw4lite/build.sh HIP         # requires ROCm
```

CUDA produces `build/level2/sw4lite/cuda/sw4lite`. HIP copies the pinned
source into the ignored build tree, performs the upstream hipify step there,
and produces `build/level2/sw4lite/hip/app/sw4lite`. Backend builds never
overwrite each other.

## Run

```bash
HPCPERF_GPUS=1 level2/sw4lite/run.sh
HPCPERF_GPUS=2 level2/sw4lite/run.sh
HPCPERF_GPUS=1 level2/sw4lite/run.sh HIP
HPCPERF_GPUS=2 level2/sw4lite/run.sh HIP
```

The common launcher starts one MPI rank per device. SW4lite creates its native
Cartesian decomposition and exchanges boundary planes between spatial
subdomains; all ranks advance one point-source wavefield. Device selection is
derived from MPI local rank.

The default is the upstream `pointsource.in` case with final time 0.6.
Select another compatible deck with `HPCPERF_SW4LITE_INPUT`.

Observed on one NVIDIA B200: approximately 13 seconds end to end.

## Validate

```bash
HPCPERF_GPUS=1 level2/sw4lite/validate.sh
HPCPERF_GPUS=2 level2/sw4lite/validate.sh
```

## Validation

`validate.sh` parses the point-source analytical norms and checks tolerances
around `Linf=0.569416`, `L2=0.0245361`, and solution norm `3.7439`.
Missing or out-of-tolerance results return nonzero. It also requires a nonzero
native GPU-device count.

CUDA clean build, one-GPU execution, analytical correctness, native device use,
and rank-to-device mapping passed. The recorded validations used one GPU (the contributor's allocation, and the 2026-09-15 clean-clone
re-validation on dgx003, a node with four B200); a 2- or 4-GPU CUDA correctness run has not been performed
yet, so the multi-GPU status is `not yet (1-GPU validation only)`.

Future AMD validation:

```bash
HPCPERF_HIP_ARCH=gfx942 level2/sw4lite/build.sh HIP
HPCPERF_GPUS=1 level2/sw4lite/validate.sh HIP
HPCPERF_GPUS=2 level2/sw4lite/validate.sh HIP
```

Both runs must reproduce all three analytical norms within the same
tolerances, report the requested MPI task count, and show the expected HIP
rank/device map.

## Warnings

- The archived HIP branch predates current ROCm releases and requires
  compile/runtime confirmation on the intended AMD platform.
- MPI transport and multi-node behavior remain site-specific.
- HIP compilation and runtime were unavailable locally.

## LOC

CUDA snapshot: 46,040 lines of C/C++/Fortran under `source/cuda/src/`.
HIP snapshot: 46,294 lines of C/C++/Fortran under `source/hip/src/`.
Build files, scripts, and inputs are excluded.

## Verified Environment

GCC/G++ 13.3.0 (conda) | GNU Fortran 9.3 | CMake 3.28.4
CUDA Toolkit 13.2 (nvcc 13.2.78, `/usr/local/cuda`) | NVIDIA B200 (sm_100)
Open MPI 5.0.10 (conda) | OpenBLAS/LAPACK | one visible GPU
HIP/ROCm: authoritative source and build/run configuration present, unverified

Reproduce the common toolchain from the repository root with
`./setup_env.sh`, then `source hpcperf_env.sh`.

## Status

CUDA Build: PASS
CUDA Single GPU: PASS
CUDA Multi-GPU: not yet (1-GPU validation only; dgx003 has four B200, the 2+/4-GPU CUDA correctness run has not been performed)
CUDA Correctness: PASS
HIP Integration: INTEGRATED-NOT-LOCALLY-VALIDATED
HIP Build: UNTESTED (no ROCm on the validation node)
HIP Runtime: UNTESTED (no AMD GPU on the validation node)

HIP runtime was not validated locally because AMD GPU hardware is unavailable.
