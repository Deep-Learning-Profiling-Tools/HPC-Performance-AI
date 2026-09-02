# TeaLeaf

TeaLeaf (UoB-HPC C++ version) solves the linear heat-conduction equation on a
2D structured grid with an implicit time step. Every time step requires the
solution of a sparse, symmetric positive-definite linear system (5-point
stencil), which is done with a matrix-free iterative solver: Conjugate
Gradient (default), Chebyshev, PPCG or Jacobi. The main motif is therefore
memory-bandwidth-bound stencil SpMV-like sweeps plus global dot-product
reductions every iteration, with halo exchange between MPI ranks. The whole
field lives on the device; the host drives the iteration loop.

## Provenance

Upstream repository: https://github.com/UoB-HPC/TeaLeaf
Upstream commit: e70261c0be40537da75b258108ed2898f84f3c58 (master, "Merge pull request #10 from illuhad/patch-1")
Cloned: 2026-09-01 (reference clone in `_upstream/level2/TeaLeaf`)
License: **none shipped upstream** -- the repository has no LICENSE file and
GitHub shows no license for it. The README states it "replicates the
functionality of the reference version of TeaLeaf
(https://github.com/UK-MAC/TeaLeaf_ref)", which is LGPL-3.0, but this C++
port itself carries no license text, so nothing was copied as `LICENSE`.
This needs a maintainer decision (ask the upstream authors / treat as
all-rights-reserved until clarified).

Copied from upstream (byte-identical):

- `CMakeLists.txt`, `cmake/register_models.cmake` -- upstream CMake build
- `driver/` -- shared host driver (main, config parser, chunk/comms, solver drivers, MPI shim), 32 files
- `src/cuda/` -- CUDA model (kernels + `model.cmake`), 11 files
- `src/hip/` -- HIP model (kernels + `model.cmake`), 11 files
- `tea.in`, `tea.problems` -- default deck and the reference-values table used by the built-in check
- `Benchmarks/` -- upstream benchmark decks `tea_bm_*.in` and the reference outputs `tea_bm_{1..6}.out` of the Fortran reference code

Left out: the other programming models (`src/{serial,omp,kokkos,std-indices,sycl-acc,sycl-usm}`),
upstream `build.sh` (only `module load cmake`), `test.sh` (developer test matrix with
hardcoded local paths), `.clang-format`, `.gitignore`, upstream `README.md`.

Added here: `build.sh`, `run.sh`, `validate.sh`, this `README.md`.

## Changes from upstream

None to the source or CMake files -- everything under `driver/`, `src/`,
`cmake/`, `CMakeLists.txt`, the decks and `tea.problems` is byte-identical to
upstream. The only environment adaptation lives in `build.sh`:

- `-DCMAKE_CUDA_ARCHITECTURES=OFF` is passed at configure time. Upstream adds
  the GPU architecture itself via `-arch=${CUDA_ARCH}` (and sets policy
  CMP0104 to OLD). Our environment exports `CUDAARCHS=native`, which makes
  CMake 3.28 append a second `-arch=native`, so nvcc would see two
  conflicting `-arch` options. Turning CMake's own architecture handling off
  leaves exactly upstream's flag on the command line.
- Upstream's `build.sh` (`module load cmake`) is replaced by ours (modules are
  not used in this repo).

Upstream behaviours worth knowing (deliberately *not* changed):

- Upstream marks every source file as `LANGUAGE CUDA` and gates its
  `-O3 -Wall -march=native` release flags on `COMPILE_LANGUAGE:CXX`, and also
  strips CMake's default `-O3` from `CMAKE_CUDA_FLAGS_RELEASE`. As a result
  the effective nvcc line is `-std=c++17 -forward-unknown-to-host-compiler
  -arch=sm_100 -use_fast_math -restrict -keep -DNDEBUG` (device code at
  nvcc's default optimisation, host code at gcc's default, no `-Wall`). A test
  build with `-DCUDA_EXTRA_FLAGS="-O3"` changed the 4000x4000 solve time by
  <2 % (18.0 s vs 18.2 s), so it does not matter here; pass it through
  `./build.sh CUDA -DCUDA_EXTRA_FLAGS=-O3` if wanted.
- Upstream's `-keep` makes nvcc leave its intermediates (`.ii`, `.ptx`,
  `.cubin`, `.fatbin`, ...) in the build directory (~170 MB). Harmless; the
  build tree is gitignored.
- `cmake_policy(SET CMP0104 OLD)` in `src/cuda/model.cmake` prints one CMake
  deprecation warning at configure time.

## Dependencies

- CUDA toolkit (nvcc) -- system CUDA 13.2 via `hpcperf_env.sh`
- Host C++17 compiler -- conda GCC 13.3.0 (`$CC`/`$CXX` from `hpcperf_env.sh`)
- MPI (optional, **enabled by default here**) -- conda OpenMPI 5.0.10 from
  `$CONDA_PREFIX`, found by CMake's `find_package(MPI)` without extra hints.
  It configured cleanly, so `build.sh` uses `-DENABLE_MPI=ON`; set
  `HPCPERF_TEALEAF_MPI=OFF` for the MPI-free build (uses upstream's
  `driver/mpi_shim.cpp`; also verified to build and validate).
  Note: with this OpenMPI `MPIX_Query_cuda_support()` returns 0 at runtime, so
  TeaLeaf's `--staging-buffer auto` selects a host staging buffer for halo
  exchange (only relevant with more than one rank).
- HIP variant: `hipcc` (ROCm) -- not available on this machine.

No framework libraries (Kokkos, RAJA, ...) are needed for the CUDA/HIP models.

## Build / Run / Validate

```bash
source hpcperf_env.sh                     # optional -- the scripts source it themselves
./level2/tealeaf/build.sh                 # CUDA (default); ./build.sh HIP for the HIP model
./level2/tealeaf/run.sh                   # Benchmarks/tea_bm_5.in, 4000x4000 @ 10 steps
./level2/tealeaf/validate.sh              # tea.in, 512x512 @ 20 steps, PASS/FAIL + exit code
```

`build.sh` detects the GPU (`nvidia-smi --query-gpu=compute_cap`) and can be
overridden with `HPCPERF_CUDA_ARCH=100`; extra `-D...` options are forwarded to
CMake. `run.sh` accepts a different deck via `HPCPERF_TEALEAF_DECK` and a rank
count via `HPCPERF_NP` (uses `mpirun --oversubscribe` for >1); extra arguments
go to the `cuda-tealeaf` binary (e.g. `--solver ppcg`). The log file `tea.out`
is written to `build/level2/tealeaf/<model>/run/`.

Equivalent raw commands (CUDA, MPI on):

```bash
source hpcperf_env.sh
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
cmake -S level2/tealeaf -B build/level2/tealeaf/cuda -DMODEL=cuda -DENABLE_MPI=ON \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX \
      -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCUDA_ARCH=sm_$ARCH -DCMAKE_CUDA_ARCHITECTURES=OFF
cmake --build build/level2/tealeaf/cuda -j4
cd build/level2/tealeaf/cuda
mpirun -np 1 ./cuda-tealeaf --file ../../../../level2/tealeaf/Benchmarks/tea_bm_5.in \
                            --problems ../../../../level2/tealeaf/tea.problems
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level2/tealeaf -B build/level2/tealeaf/hip -DMODEL=hip -DENABLE_MPI=ON \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=hipcc \
      -DCXX_EXTRA_FLAGS=--offload-arch=gfx90a        # or HPCPERF_HIP_ARCH=gfx90a ./build.sh HIP
cmake --build build/level2/tealeaf/hip -j4
```

Deck options (`--file`): `tea.in` (512^2, 20 steps, validation),
`Benchmarks/tea_bm_4.in` (1000^2), `tea_bm_5e_2.in` (2000^2),
`tea_bm_5.in` (4000^2, standard benchmark), `tea_bm_6.in` (8000^2), and the
`tea_bm_5e_{1,2,4,8}_{2,4}.in` variants with 2 or 4 steps. `tea_bm_1..3`
(10^2 .. 500^2) have no entry in `tea.problems` and report FAILED by design.

## Validation

TeaLeaf has a built-in reference check (`driver/field_summary_driver.cpp`,
`check_result` on by default): after the last step it sums the temperature
field over the whole grid, looks up the `(x_cells, y_cells, end_step)` row in
`tea.problems` and requires the relative difference to be below 0.001 %. The
expected values were produced upstream by the single-core reference code
(Intel compiler, IEEE mode, eps 1e-15). The program prints
` This run PASSED/FAILED`, `Outcome: PASSED/FAILED` and exits 0/1;
`validate.sh` forwards that as a final `PASS`/`FAIL` line.

Observed here (CUDA, 1 rank, B200):

| deck | grid | expected | actual | diff |
|---|---|---|---|---|
| `tea.in` (validate.sh) | 512x512 @ 20 | 1.034697091898282e+02 | 1.034697092107607e+02 | 2e-8 % PASSED |
| `Benchmarks/tea_bm_5.in` (run.sh) | 4000x4000 @ 10 | 9.546235158221428e+01 | 9.546235158231137e+01 | <1e-8 % PASSED |

`validate.sh` takes ~10 s wall (1.0 s in the solver, the rest is MPI/CUDA
start-up); `run.sh` takes ~25 s wall (18.5 s solver, 4873 -> 3678 CG
iterations per step). A 2-rank run (`HPCPERF_NP=2`, both ranks on the one
GPU) also PASSED.

## Warnings

Compiler: none (nvcc 13.2 + GCC 13.3, 32 translation units). Note that
upstream's `-Wall` never reaches nvcc (see "Changes from upstream"); forcing
`-Xcompiler -Wall` in a test build produced exactly 1 warning:
`src/cuda/kernel_initialise.cpp:30: ignoring '#pragma omp parallel for'
[-Wunknown-pragmas]` (host-side init loop, harmless).

CMake: 1 deprecation warning at configure time (`CMP0104 OLD` in
`src/cuda/model.cmake`).

## LOC

cloc v2.06, code lines only (blank/comment excluded); `model.cmake`, decks,
`tea.problems`, scripts and this README excluded.

CUDA variant: **2863** (42 files) = `driver/` 1906 (23 .cpp + 9 .h) + `src/cuda/` 957 (8 .cpp + 2 .h)
HIP variant:  **2870** (42 files) = `driver/` 1906 + `src/hip/` 964 (8 .cpp + 2 .h)

```bash
cloc --quiet level2/tealeaf/driver level2/tealeaf/src/cuda --exclude-ext=cmake
cloc --quiet level2/tealeaf/driver level2/tealeaf/src/hip  --exclude-ext=cmake
```

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++17 | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
OpenMPI 5.0.10 (conda, CUDA-aware build; `MPIX_Query_cuda_support()` reports 0 at runtime here)
HIP/ROCm: source + build config present where noted, unverified (no AMD GPU available)
OS: RHEL 10 (Linux 6.12)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Status

CUDA: **Working** -- configure + build (0 compiler warnings) + run
(`tea_bm_5.in`, PASSED) + validate (`tea.in`, PASSED) all succeeded on this
machine, from a clean shell and a foreign cwd, with MPI on (1 and 2 ranks) and
with MPI off.
HIP: extracted, untested (no ROCm/hipcc on the development machine;
`./build.sh HIP` fails at configure with "hipcc ... was not found in the PATH").
Open item: upstream ships no license (see Provenance).
