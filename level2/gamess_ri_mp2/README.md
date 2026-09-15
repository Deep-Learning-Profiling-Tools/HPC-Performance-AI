# GAMESS RI-MP2 MiniApp

The GAMESS RI-MP2 MiniApp is an Argonne/ECP proxy for the
resolution-of-identity MP2 correlation-energy kernel from GAMESS. Its dominant
work forms blocks of three-index integrals and evaluates dense matrix
products. CUDA uses the upstream cuBLAS implementation; HIP uses the upstream
hipBLAS/hipfort implementation. Main motif: distributed quantum-chemistry
dense linear algebra with an MPI energy reduction.

## Provenance

Upstream repository: https://github.com/jkwack/GAMESS_RI-MP2_MiniApp
CUDA source: `ECP-proxy`, commit
`70f287ebff3e3c48db31996828ce115e21054ab3`
HIP source: same official `ECP-proxy` revision and commit
License: University of Illinois/NCSA-style Argonne license -- copied to
`LICENSE`

The HIP path is authoritative because the upstream developer's ECP branch
documents and implements its Crusher AMD route with hipfort and hipBLAS in the
same Fortran source as the cuBLAS path. The compile-time `HIPBLAS` and
`CUBLAS` paths call vendor libraries directly. No new HIP translation or
portability abstraction was added.

Copied into this directory:

- `source/cublasf.f90` and `source/rimp2_energy_whole_KERN.f90` -- minimal
  source required by both native GPU paths.
- `inputs/benz.kern` -- upstream physical benzene kernel input.
- `LICENSE`, `build.sh`, `run.sh`, and `validate.sh`.

Upstream job scripts, machine-specific environment files, figures, git
metadata, and generated build products were not copied.

## Changes from upstream

1. `source/rimp2_energy_whole_KERN.f90`: the CUDA-only
   `HPCPERF_MINIMAL_MPIF` path includes Open MPI handle declarations without
   its generated high-rank generic interfaces. This works around an
   NVFORTRAN/Open MPI 5 parser incompatibility; MPI handles and calls are
   unchanged. The HIP path retains upstream `mpif.h`. Upstreamable.
2. `build.sh` supplies reproducible cuBLAS and hipBLAS/hipfort links,
   extracts MPI wrapper flags, isolates backend outputs, and exposes both GPU
   architecture controls.
3. `run.sh` applies the common rank-to-device mapping and rejects MPI ranks
   beyond the available active-orbital work partitions.
4. `validate.sh` converts the upstream energy test and backend marker into a
   nonzero-on-failure gate. No numerical algorithm was changed.

## Dependencies

- Fortran compiler, MPI Fortran wrapper/libraries, and OpenMP support.
- CUDA variant: NVIDIA HPC SDK `nvfortran`, `nvcc`, CUDA runtime, and
  cuBLAS.
- HIP variant: ROCm `hipcc`, hipfort `hipfc`, hipBLAS, and an
  MPI-compatible Fortran compiler.
- The authoritative Crusher/CCE route uses `-homp`. Sites may set
  `HPCPERF_HIP_OPENMP_FLAG` for an equivalent compiler-specific spelling.
- Cray systems must load the matching `craype-accel-amd-<arch>` target or
  provide `HPCPERF_HIP_ARCH_FLAG`.

CUDA architecture is detected automatically; override it with
`HPCPERF_CUDA_ARCH=90`. HIP accepts a target such as
`HPCPERF_HIP_ARCH=gfx942`; no architecture is hard-coded.

## Backends

CUDA: upstream NVFORTRAN OpenMP GPU regions and native cuBLAS, locally
validated.
HIP: upstream hipfort and native hipBLAS, integrated but not locally compiled
or executed.

## Build

```bash
source hpcperf_env.sh
level2/gamess_ri_mp2/build.sh             # CUDA (default)
level2/gamess_ri_mp2/build.sh HIP         # requires ROCm/hipfort
```

Outputs are `build/level2/gamess_ri_mp2/cuda/rimp2-cublas` and
`build/level2/gamess_ri_mp2/hip/rimp2-hipblas`. Missing Fortran/MPI,
cuBLAS, hipfort, hipBLAS, ROCm, or architecture prerequisites produce an
explanatory failure.

## Run

```bash
HPCPERF_GPUS=1 level2/gamess_ri_mp2/run.sh
HPCPERF_GPUS=2 level2/gamess_ri_mp2/run.sh
HPCPERF_GPUS=1 level2/gamess_ri_mp2/run.sh HIP
HPCPERF_GPUS=2 level2/gamess_ri_mp2/run.sh HIP
```

Every rank receives a disjoint trapezoidal portion of the active-orbital pair
domain, runs vendor BLAS work on its local-rank GPU, and participates in the
final energy reduction. This is one sharded molecular problem, not replicated
jobs. Rank counts larger than the input's active-orbital partitions are
rejected so every requested GPU performs useful work.

The default `w30.rand` mode deterministically generates the dimensions of the
upstream 30-water workload with `NQVV=30`. It has 120 active orbitals.
Controls: `HPCPERF_GAMESS_INPUT`, `HPCPERF_GAMESS_NQVV`, and
`HPCPERF_GPUS`. Set `HPCPERF_GAMESS_INPUT=benz.kern` to select the retained
physical input.

Observed on one NVIDIA B200: approximately 8 seconds end to end; maximum
reported compute time was approximately 0.74 seconds.

## Validate

```bash
HPCPERF_GPUS=1 level2/gamess_ri_mp2/validate.sh
HPCPERF_GPUS=2 level2/gamess_ri_mp2/validate.sh
```

## Validation

For generated inputs, upstream constructs a closed-form reference MP2 energy.
`validate.sh` requires `Passed :-)`, a relative error no larger than
`1e-6`, the selected `cublas on GPU` or `HIPBLAS on GPU` backend marker,
and a zero application exit.

CUDA clean build, one-GPU execution, cuBLAS work, correctness, and
rank-to-device mapping passed. The observed relative error was approximately
`4.4e-14`. The recorded validations used one GPU (the contributor's allocation, and the 2026-09-15 clean-clone
re-validation on dgx003, a node with four B200); a 2- or 4-GPU CUDA correctness run has not been performed
yet, so the multi-GPU status is `not yet (1-GPU validation only)`.

Future AMD validation:

```bash
HPCPERF_HIP_ARCH=gfx942 level2/gamess_ri_mp2/build.sh HIP
HPCPERF_GPUS=1 level2/gamess_ri_mp2/validate.sh HIP
HPCPERF_GPUS=2 level2/gamess_ri_mp2/validate.sh HIP
```

Both runs must print `HIPBLAS on GPU`, satisfy the relative-energy tolerance,
show the requested MPI rank count and local-rank device map, and exit zero.

## Warnings

- The HIP compiler route is compiler-family sensitive, as documented
  upstream, and needs confirmation with the target site's hipfort stack.
- HIP compilation and runtime were unavailable locally.
- `inputs/benz.kern` is a 23 MB upstream benchmark input, not a generated
  binary or build product.

## LOC

Application source: 806 lines of Fortran in `source/`.
Build scripts, inputs, and documentation are excluded.

## Verified Environment

Contributor validation: NVIDIA HPC SDK 25.9 (`nvfortran`) | CUDA/cuBLAS 13.2

dgx003 re-validation from a clean clone of main 4604377 (2026-09-15): the node has no
`nvfortran` on PATH and lmod is unusable there, so the site install
`/opt/sw/other/apps/nvidia/hpc_sdk/Linux_x86_64/25.7` was used directly. Its shipped `localrc`
points at a GCC 8 that the RHEL 10 image no longer has (`Error in path
/usr/lib/gcc/x86_64-redhat-linux/8/...`), so a per-user configuration for the installed GCC 14 is
generated first:

```bash
SDK=/opt/sw/other/apps/nvidia/hpc_sdk/Linux_x86_64/25.7
mkdir -p "$HOME/.nvlocalrc" && "$SDK/compilers/bin/makelocalrc" -gcc /usr/bin/gcc -gpp /usr/bin/g++ -g77 /usr/bin/gfortran -x -d "$HOME/.nvlocalrc"
NVLOCALRC="$HOME/.nvlocalrc/localrc" NVFORTRAN="$SDK/compilers/bin/nvfortran" ./build.sh CUDA   # arch from nvidia-smi (cc100 here); HPCPERF_CUDA_ARCH overrides
HPCPERF_GPUS=1 ./validate.sh CUDA
```

Result: build 7 s (two deprecation warnings about `USE_DEVICE_PTR`); the binary links the project CUDA 13.2
cuBLAS/cudart and the conda Open MPI, `libnvf` from the SDK (RPATH embedded); validation PASS, relative
error of the MP2 correlation energy 4.4e-14, launcher binding audit 1 verified / 0 mismatch. The build was
repeated with `HPCPERF_CUDA_ARCH` and `CUDA_ARCH` unset: `build.sh` detected cc100 through nvidia-smi and
the binary carries sm_100 device code; validation PASS again. `nvfortran` 25.7 accepted `-gpu=cc100`; the
SDK's bundled CUDA 12.9 is not used at run time.
CUDA Toolkit 13.2 (nvcc 13.2.78, `/usr/local/cuda`) | NVIDIA B200 (sm_100)
Open MPI 5.0.10 | Slurm allocation with one visible GPU
HIP/ROCm/hipfort: authoritative source and build/run configuration present,
unverified

Reproduce the common repository environment with `./setup_env.sh`, then
`source hpcperf_env.sh`; NVHPC remains a benchmark-native system dependency.

## Status

CUDA Build: PASS
CUDA Single GPU: PASS
CUDA Multi-GPU: not yet (1-GPU validation only; dgx003 has four B200, the 2+/4-GPU CUDA correctness run has not been performed)
CUDA Correctness: PASS
HIP Integration: INTEGRATED-NOT-LOCALLY-VALIDATED
HIP Build: UNTESTED (no ROCm on the validation node)
HIP Runtime: UNTESTED (no AMD GPU on the validation node)

HIP runtime was not validated locally because AMD GPU hardware is unavailable.
