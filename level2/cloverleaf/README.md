# CloverLeaf

Compressible-Euler hydrodynamics mini-app (2D, structured grid). CloverLeaf
solves the compressible Euler equations with an explicit second-order finite
volume method on a staggered Cartesian mesh: a Lagrangian predictor-corrector
step (ideal gas, viscosity, PdV work, acceleration, flux calculation) followed by
a second-order van Leer advective remap of density, energy and momentum back to
the fixed mesh. Main motifs: memory-bandwidth-bound structured-grid stencils over
~15 field arrays, global reductions (time-step control, field summary) and halo
exchange (tile halos on device, MPI halos between ranks). The UoB-HPC C++ port
used here keeps the shared host driver in `driver/` and one kernel
implementation per programming model in `src/<model>/`; the `cuda` and `hip`
models are included.

## Provenance

Upstream repository: https://github.com/UoB-HPC/CloverLeaf (UoB-HPC C++ port,
clone kept at `_upstream/level2/CloverLeaf_UoB`)
Upstream commit: 3e10ff9268d3704a744e5a2a4ea3c62831078082 (2024-08-12)
Cloned: 2026-09-01
License: GNU GPL v3 (upstream `LICENCE.txt`, copied here as `LICENSE`).
Crown Copyright 2012 AWE; Copyright (c) 2019-24 Wei-Chen Lin, Tom Deakin,
Simon McIntosh-Smith.

Copied from upstream (byte-identical, verified with `diff -r`):

- `CMakeLists.txt`, `cmake/register_models.cmake` -- build system
- `driver/` -- shared host driver (input parsing, time stepping, MPI comms,
  report/validation), 14 `.cpp` + 34 headers
- `src/cuda/` -- CUDA model (20 `.cpp` kernels + `context.h` + `model.cmake`)
- `src/hip/` -- HIP model (same layout)
- `InputDecks/` -- all upstream input decks (`clover_bm*.in`, `clover.in`,
  `clover_qa.in`, `clover_sod*.in`)
- `LICENCE.txt` -> `LICENSE`

Left out: the other programming-model directories (`src/omp`, `omp-target`,
`serial`, `std-indices`, `tbb`, `kokkos`, `acc`, `sycl-acc`, `sycl-usm`),
`test.sh`, `load_openmpi.sh`, `.clang-format`, upstream `README.md`.

Reference implementations (not built): the original UK-MAC ports listed in the
Level 2 plan, `_upstream/level2/CloverLeaf_CUDA`
(https://github.com/UK-MAC/CloverLeaf_CUDA, commit
03c780320cfab3a47379a1c5952ab24011647f2e) and `_upstream/level2/CloverLeaf_HIP`
(https://github.com/UK-MAC/CloverLeaf_HIP, commit
5f91a4bc53baac69c436e7ef1df0343928f3f682), are Fortran drivers with CUDA/HIP
kernels and require `gfortran`, which is not part of this suite's toolchain
(conda GCC 13.3 C/C++ + system CUDA). The UoB-HPC version is a pure C++ port of
the same CloverLeaf 1.3 algorithm: same kernels and kernel order, same
`clover_bm*.in` input decks (physics parameters identical; the UoB decks only
replace the UK-MAC `use_cuda_kernels` switch by `profiler_on`), the CloverLeaf
1.3 output format, and the same reference kinetic energies for the
`test_problem` checks (the values for test problems 2, 4 and 5 in
`driver/report.cpp` match `field_summary.f90` in both UK-MAC clones), so it is
used as the buildable stand-in for both the CUDA and HIP variants.

## Changes from upstream

None to the copied source, decks or CMake files (all byte-identical). Wrapper-level:

- `build.sh` passes `-DCMAKE_CUDA_ARCHITECTURES=OFF`. Upstream adds
  `-arch=${CUDA_ARCH}` to `CMAKE_CUDA_FLAGS` itself (`src/cuda/model.cmake`) and
  sets policy CMP0104 to OLD; with our environment's `CUDAARCHS=native`, CMake
  3.28 would also add `-arch=native`, and nvcc warns "incompatible redefinition
  for option 'gpu-architecture'". `OFF` stops CMake from adding its own flag so
  upstream's `-arch=sm_XX` (detected by `build.sh`) is the only one.
- Upstream `LICENCE.txt` is stored as `LICENSE` (suite convention).
- The upstream `cmake_policy(SET CMP0104 OLD)` in `src/cuda/model.cmake:34` is
  left as is; it produces one CMake deprecation notice at configure time only
  (see Warnings).

## Dependencies

- CUDA toolkit (`nvcc`, system CUDA 13.2 via `hpcperf_env.sh`) and a C++17
  host compiler (conda GCC 13.3.0, `$CXX` / `CUDAHOSTCXX`).
- MPI: built with `-DENABLE_MPI=ON` using the conda OpenMPI 5.0.10 found by
  `find_package(MPI)` (links `MPI::MPI_C`; `mpirun` from `$CONDA_PREFIX`). A
  non-MPI build is possible with `-DENABLE_MPI=OFF` (no other change needed).
  Note: OpenMPI's header advertises CUDA-awareness but
  `MPIX_Query_cuda_support()` returns false at run time here, so CloverLeaf's
  default `--staging-buffer auto` uses host staging buffers for halo exchange
  (irrelevant for the single-rank default run; do not pass
  `--staging-buffer false` with this MPI).
- HIP variant: `hipcc` / ROCm (not available on this machine).
- No Kokkos/RAJA or other framework libraries.

## Backends

CUDA: working (configure + build + run + validation verified on B200)
HIP: extracted, untested (no AMD GPU / ROCm on the development machine)

## Build

```bash
source hpcperf_env.sh
./level2/cloverleaf/build.sh          # CUDA (default) -> build/level2/cloverleaf/cuda/cuda-cloverleaf
./level2/cloverleaf/build.sh HIP      # HIP  -> build/level2/cloverleaf/hip/hip-cloverleaf (needs hipcc)
```

GPU architecture is detected with `nvidia-smi --query-gpu=compute_cap` (override:
`HPCPERF_CUDA_ARCH=90`). Equivalent raw commands (CUDA, sm_100):

```bash
cmake -S level2/cloverleaf -B build/level2/cloverleaf/cuda \
  -DMODEL=cuda -DENABLE_MPI=ON \
  -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CUDA_COMPILER=$(which nvcc) \
  -DCUDA_ARCH=sm_100 -DCMAKE_CUDA_ARCHITECTURES=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/level2/cloverleaf/cuda -j4
```

HIP (build configuration present, unverified without ROCm; set
`HPCPERF_HIP_ARCH=gfx90a` to add `--offload-arch`):

```bash
cmake -S level2/cloverleaf -B build/level2/cloverleaf/hip \
  -DMODEL=hip -DENABLE_MPI=ON -DCMAKE_CXX_COMPILER=$(which hipcc) \
  -DCMAKE_BUILD_TYPE=Release [-DCXX_EXTRA_FLAGS=--offload-arch=gfx90a]
cmake --build build/level2/cloverleaf/hip -j4
```

## Run

```bash
./level2/cloverleaf/run.sh [CUDA|HIP] [extra cloverleaf args]
```

Runs `mpirun -np 1 cuda-cloverleaf --file InputDecks/clover_bm16.in --out
build/level2/cloverleaf/cuda/clover.out`. `clover_bm16.in` is the standard
UK-MAC "bm16" benchmark deck: 3840 x 3840 cells, 2955 time steps (end_time
15.6), `profiler_on`, `test_problem 5`. On the B200 this is 21.6 s of hydro wall
clock, about 35 s end to end (about 5 s of that is `MPI_Init` of the conda
OpenMPI on this machine, measured independently). The upstream driver prints a
YAML header (device, MPI, model), the CloverLeaf 1.3-style step log, the
per-kernel profiler table and a final `Result:` block; the detailed log goes to
`clover.out` in the build tree.

Options: `HPCPERF_CLOVERLEAF_DECK=<deck>` or append `--file <deck>` (last
option wins) to run another deck; `HPCPERF_NP=<n>` runs n ranks (adds
`--oversubscribe`). Other useful decks in `InputDecks/`:

- `clover_bm_short.in` (960^2, 87 steps, test_problem 2, 0.10 s hydro),
  `clover_bm16_short.in` (3840^2, 87 steps, test_problem 4, 0.65 s hydro),
  `clover_bm16_very_short.in` (8 steps, test_problem 168) -- quick checks
- `clover_bm.in` (960^2, 2955 steps, test_problem 3), `clover_bm16_300.in`
- larger: `clover_bm32/64/128/.../8192*.in` (7680^2 and up; no test_problem
  reference)

Raw equivalent:

```bash
mpirun -np 1 build/level2/cloverleaf/cuda/cuda-cloverleaf \
  --file level2/cloverleaf/InputDecks/clover_bm16.in --out build/level2/cloverleaf/cuda/clover.out
```

## Validate

```bash
./level2/cloverleaf/validate.sh [CUDA|HIP]
```

Runs two decks with upstream's built-in reference check
(`driver/report.cpp`), requires exit code 0 and the line
`This test is considered PASSED` for each, then prints
`CloverLeaf CUDA validation: PASS|FAIL` (exit 0/1).

## Validation

Upstream's own mechanism: each `test_problem N` deck has a hard-coded reference
final total kinetic energy in `driver/report.cpp`; after the last step the
driver prints `Test problem N is within X% of the expected solution` and
`This test is considered PASSED` iff `|100*(KE/KE_ref) - 100| < 0.001` (%),
and `main` returns non-zero on failure. Decks checked:

| deck | grid | steps | test_problem | KE reference | observed (CUDA, B200) |
|---|---|---|---|---|---|
| `InputDecks/clover_bm_short.in` | 960 x 960 | 87 | 2 | 1.19316898756307 | within 1.16813e-11 %, PASSED |
| `InputDecks/clover_bm16_short.in` | 3840 x 3840 | 87 | 4 | 0.307475452287895 | within 4.58016e-11 %, PASSED |

The `run.sh` deck `clover_bm16.in` (test_problem 5, KE ref 4.85350315783719)
also passes: within 2.984e-10 %, PASSED. A negative check (deck with a wrong
`test_problem` id) gives `NOT PASSED`, `Outcome: FAILED`, exit 1, so failures
propagate to `validate.sh`. A 2-rank run (`HPCPERF_NP=2`, one GPU, host staging
buffers) of `clover_bm_short.in` also passes.

## Warnings

- Compiler: none (0 warnings with upstream's `-Wall` at `-O3`, nvcc 13.2.78 +
  GCC 13.3.0 host, CUDA build).
- CMake: 1 deprecation notice at configure time, `CMake Deprecation Warning at
  src/cuda/model.cmake:34 (cmake_policy): The OLD behavior for policy CMP0104
  will be removed from a future version of CMake` (upstream, left unmodified).

## LOC

cloc 2.06, code lines only (blank/comment excluded; CMake, decks, scripts,
README excluded):

CUDA: 5220 (69 files: `driver/` 2693 [14 C++ 2054 + 34 headers 639] +
`src/cuda/` 2527 [20 C++ 2356 + `context.h` 171])
HIP: 5223 (69 files: `driver/` 2693 + `src/hip/` 2530 [20 C++ 2359 +
`context.h` 171])

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++17 | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
OpenMPI 5.0.10 (conda, `mpicc`/`mpicxx`/`mpirun`), RHEL 10.0
HIP/ROCm: source + build config present where noted, unverified (no AMD GPU available)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Status

CUDA: Working -- configure + build (`./build.sh`), run (`./run.sh`,
clover_bm16.in, test_problem 5 PASSED) and validate (`./validate.sh`, PASS) all
succeeded on this machine from a fresh shell.
HIP: untested -- `./build.sh HIP` stops with "hipcc not found" here; the CMake
logic for `-DMODEL=hip` was dry-configured successfully with g++ as a stand-in
compiler.
