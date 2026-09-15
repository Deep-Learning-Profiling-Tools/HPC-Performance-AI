# Quicksilver

Quicksilver is LLNL's ECP proxy application for the Mercury Monte Carlo
particle-transport workload. It reproduces irregular table lookups, branch
divergence, particle movement, and MPI traffic from multigroup transport. This
integration uses LLNL's native CUDA source and official `AMD-HIP` branch.
Main motif: divergent particle tracking with data-dependent memory access and
cross-domain particle migration.

## Provenance

ECP catalog: https://proxyapps.exascaleproject.org/app/quicksilver/
Upstream repository: https://github.com/LLNL/Quicksilver
CUDA source: `master`, commit `eb68bb8d6fc53de1f65011d4e79ff2ed0dd60f3b`
HIP source: official LLNL `AMD-HIP` branch, commit
`c1e29f2a34f6505969720e2ea72b9fa25ef00743`
License: LLNL BSD-style license -- copied to `LICENSE`

The HIP source is authoritative because the native HIP kernels, device-memory
management, and MPI particle exchange are maintained on the named branch in
LLNL's own repository. This integration does not run hipify and does not
substitute a portability framework.

Copied into this directory:

- `source/cuda/` -- application files from the pinned CUDA revision.
- `source/hip/` -- application files from the pinned LLNL HIP revision.
- `inputs/coral2_p1_profile.inp` -- short CORAL-2 P1 physics deck.
- `LICENSE`, `build.sh`, `run.sh`, and `validate.sh`.

Git metadata, examples not needed by the selected workload, generated files,
and upstream CI files were not copied.

## Changes from upstream

1. `source/{cuda,hip}/EnergySpectrum.hh` and
   `source/{cuda,hip}/Parameters.hh` include `<cstdint>` explicitly. This
   fixes modern compiler failures for fixed-width integer types. Numerical
   behavior is unchanged; upstreamable.
2. `build.sh` supplies the official Makefile variables, MPI flags, pinned
   revision strings, and selectable CUDA/HIP architecture.
3. `run.sh` derives the global mesh, particle count, and exact MPI topology
   from the requested GPU count.
4. `validate.sh` converts the upstream CORAL diagnostics into an exit-status
   correctness gate. No transport algorithm or GPU kernel was changed.

## Dependencies

- GNU make and a C++14-capable compiler.
- MPI C++ headers and libraries.
- CUDA variant: `nvcc` and the CUDA runtime.
- HIP variant: ROCm `hipcc`.
- GPU-aware MPI is not required: particles are packed into host MPI buffers
  after native GPU tracking.
- No framework-library dependency.

CUDA architecture is detected automatically; override it with
`HPCPERF_CUDA_ARCH=90`. HIP accepts a target such as
`HPCPERF_HIP_ARCH=gfx942`, passed as `--offload-arch`.

## Backends

CUDA: native LLNL CUDA cycle-tracking implementation, locally validated.
HIP: native LLNL `AMD-HIP` implementation, integrated but not locally
compiled or executed.

## Build

```bash
source hpcperf_env.sh
level2/quicksilver/build.sh             # CUDA (default)
level2/quicksilver/build.sh HIP         # requires ROCm
```

Outputs are `build/level2/quicksilver/cuda/qs` and
`build/level2/quicksilver/hip/qs`. Missing MPI, `nvcc`, `hipcc`, or
architecture configuration produces an explanatory failure.

## Run

```bash
HPCPERF_GPUS=1 level2/quicksilver/run.sh
HPCPERF_GPUS=2 level2/quicksilver/run.sh
HPCPERF_GPUS=1 level2/quicksilver/run.sh HIP
HPCPERF_GPUS=2 level2/quicksilver/run.sh HIP
```

Each rank owns one domain of a factorable three-dimensional Cartesian
decomposition, runs native GPU tracking kernels, and exchanges particles that
cross domain boundaries. Global cells and particles increase with the process
grid, making multi-GPU execution one coupled weak-scaling problem.

The default is the CORAL-2 P1 deck with 8 cubed cells and 100,000 source
particles per rank for 20 transport steps. Controls:
`HPCPERF_QUICKSILVER_CELLS_PER_RANK`,
`HPCPERF_QUICKSILVER_PARTICLES_PER_RANK`,
`HPCPERF_QUICKSILVER_STEPS`, `HPCPERF_QUICKSILVER_INPUT`, and
`HPCPERF_GPUS`.

Observed on one NVIDIA B200: approximately 14 seconds end to end.

## Validate

```bash
HPCPERF_GPUS=1 level2/quicksilver/validate.sh
HPCPERF_GPUS=2 level2/quicksilver/validate.sh
```

## Validation

`validate.sh` requires all four native CORAL checks to print `PASS::`, no
`FAIL::` line, a native `cycleTracking_Kernel` timing entry, and a zero
application exit. The checks cover reaction ratios, collision/facet-crossing
balance, particle conservation, and homogeneous fluence.

CUDA clean build, one-GPU execution, correctness, native GPU work, and
rank-to-device mapping passed. The recorded validations used one GPU (the contributor's allocation, and the 2026-09-15 clean-clone
re-validation on dgx003, a node with four B200); a 2- or 4-GPU CUDA correctness run has not been performed
yet, so the multi-GPU status is `not yet (1-GPU validation only)`.

Future AMD validation:

```bash
HPCPERF_HIP_ARCH=gfx942 level2/quicksilver/build.sh HIP
HPCPERF_GPUS=1 level2/quicksilver/validate.sh HIP
HPCPERF_GPUS=2 level2/quicksilver/validate.sh HIP
```

Both runs must satisfy the same four CORAL checks, show native HIP kernel
timings, and report the expected rank/device mapping.

## Warnings

- Monte Carlo timing varies with accelerator and system load.
- HIP compilation requires a compatible ROCm/MPI toolchain.
- HIP compilation and runtime were unavailable locally.

## LOC

CUDA snapshot: 9,618 lines of C/C++.
HIP snapshot: 10,241 lines of C/C++.
Makefiles, scripts, and documentation are excluded.

## Verified Environment

GCC/G++ 13.3.0 (conda) | C++14 | GNU Make 4.4.1
CUDA Toolkit 13.2 (nvcc 13.2.78, `/usr/local/cuda`) | NVIDIA B200 (sm_100)
Open MPI 5.0.10 (conda) | Slurm allocation with one visible GPU
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
