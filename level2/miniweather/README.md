# miniWeather

Mini-app for the 2-D dry compressible, stratified, non-hydrostatic Euler
equations of the atmosphere (upstream C++ YAKL `parallel_for` variant,
`cpp/miniWeather_mpi_parallelfor.cpp`). A 20 km x 10 km x-z slice is
discretised with a 4th-order finite-volume method and hyper-viscosity,
advanced in time with a 3-stage Runge-Kutta scheme and dimensional splitting
(x and z sweeps alternate each step). The main motif is stencil (structured
grid) work: every RK stage computes per-cell fluxes from a 4-cell-wide
stencil and applies tendencies, with MPI halo exchanges in x between ranks.
Problem parameters (grid, simulated time, output frequency, initial
condition) are compile-time constants read by `cpp/const.h`.

## Provenance

Upstream repository: https://github.com/mrnorman/miniWeather
Upstream commit: b001069e1f7654914744641013e2bd408a9206fb (2025-08-11)
Bundled dependency: YAKL (git submodule `cpp/YAKL`), https://github.com/mrnorman/YAKL,
commit 71a059c4701d22f3d60157b01a922776261993c0 (2024-03-22)
Cloned: 2026-09-01 into `_upstream/level2/miniWeather`
License: BSD 2-Clause (ORNL / NVIDIA), copied as `LICENSE`; YAKL is BSD 2-Clause
(Matt Norman), its license is kept as `cpp/YAKL/LICENSE`.

Copied from upstream (byte-identical unless listed under "Changes from upstream"):

| here | upstream |
|---|---|
| `LICENSE` | `LICENSE` |
| `cpp/miniWeather_mpi_parallelfor.cpp` | `cpp/miniWeather_mpi_parallelfor.cpp` (the YAKL `parallel_for` variant) |
| `cpp/const.h` | `cpp/const.h` |
| `cpp/CMakeLists.txt` | `cpp/CMakeLists.txt` (trimmed, see below) |
| `cpp/build/check_output.sh` | `cpp/build/check_output.sh` (upstream correctness check) |
| `cpp/YAKL/{CMakeLists.txt, yakl_utils.cmake, LICENSE, README.md, src/, external/}` | `cpp/YAKL/` submodule, same paths |

Not copied: the serial, plain-MPI, `parallel_for_simd_x`, OpenACC, OpenMP
offload, Fortran, Julia, and Python variants, `cpp/build/cmake_*.sh` site
scripts, documentation/images, CI files, and YAKL's `unit/` tests,
`doxygen.config`, `.git*`.

## Changes from upstream

- `cpp/CMakeLists.txt`: removed the three target blocks for variants whose
  sources are not copied (`serial`/`serial_test`, `mpi`/`mpi_test`,
  `parallelfor_simd_x`/`parallelfor_simd_x_test`), the matching
  `yakl_process_target(...)` calls, and their entries in the two
  `set_target_properties(... LINKER_LANGUAGE CXX / CUDA_ARCHITECTURES OFF)`
  lists. Everything else (parameter defaults, `EXE_DEFS`/`TEST_DEFS`,
  `add_subdirectory(YAKL)`, the `parallelfor` and `parallelfor_test` targets,
  the `YAKL_TEST` ctest entry) is unchanged. Reason: CMake fails on missing
  source files.
- `cpp/YAKL/`: `unit/` (tests), `doxygen.config` and git metadata omitted;
  no file content changed.
- No source (`.cpp`/`.h`) changes. The two compiler warnings listed below are
  left as in upstream.
- Build configuration (in `build.sh`, not in upstream files): the MPI include
  directory is passed to nvcc with `-I` (from `mpicxx --showme:incdirs`)
  instead of upstream's `-ccbin mpic++`, because CMake already sets nvcc's
  host compiler from `$CUDAHOSTCXX`; `-DNO_INFORM` (an upstream flag,
  suppresses per-step "Elapsed Time" prints) and `OUT_FREQ=-1` (no NetCDF
  output during the benchmark run) are used as in upstream's GPU site
  scripts. PnetCDF stays linked as upstream requires.

## Dependencies

- MPI: OpenMPI 5.0.10 (conda, `$CONDA_PREFIX`), `mpicxx`/`mpicc` are the
  CMake compilers (they wrap conda GCC 13.3.0). `-DHAVE_MPI` is set.
- PnetCDF 1.15.0 (conda, `$CONDA_PREFIX`; `PNETCDF_PATH` override), needed
  for the NetCDF output path even when `OUT_FREQ=-1`.
- CUDA 13.2 (`/usr/local/cuda`): nvcc, CUB (from `cccl/`), `cufft`, NVTX3 --
  all pulled in by YAKL.
- YAKL (bundled, header-only, `cpp/YAKL`). Its `CMakeLists.txt` declares
  `LANGUAGES C CXX Fortran`, so a Fortran compiler must be found at configure
  time; the conda environment has none (`mpif90` is not functional), and
  CMake picks up the system `/usr/bin/gfortran` 14.2.1. Only the two
  `parallelfor*` targets are built, so YAKL's Fortran interface library is
  never compiled.
- HIP variant: same single source through `YAKL_ARCH=HIP`; `build.sh HIP`
  is written after upstream `cpp/build/cmake_crusher_amd_gpu.sh` but cannot
  run here (no ROCm).

## Build / Run / Validate

```bash
source hpcperf_env.sh 2>/dev/null
level2/miniweather/build.sh            # CUDA, -> build/level2/miniweather/cuda/{parallelfor,parallelfor_test}
level2/miniweather/run.sh              # mpirun -np 1 ./parallelfor   (benchmark problem, ~40-60 s wall)
level2/miniweather/validate.sh         # mpirun -np 1 ./parallelfor_test + check_output.sh, PASS/FAIL
level2/miniweather/build.sh HIP        # form only; exits 1 here: "hipcc not found on PATH"
```

`run.sh` accepts a backend and mpirun options that replace the default
`-np 1`, e.g. `run.sh CUDA --oversubscribe -np 2` (both ranks then share the
one GPU). The benchmark size is fixed at configure time; `build.sh` reads
`MINIWEATHER_NX/NZ/SIM_TIME/OUT_FREQ/DATA_SPEC` (defaults 2048 / 1024 / 1000
/ -1 / DATA_SPEC_THERMAL), `HPCPERF_CUDA_ARCH` (default: detected from
`nvidia-smi`), `HPCPERF_HIP_ARCH` (default gfx90a), `PNETCDF_PATH`,
`MAKE_JOBS` (default 4).

Equivalent raw commands (CUDA, sm_100):

```bash
cmake -S level2/miniweather/cpp -B build/level2/miniweather/cuda \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc \
  -DNX=2048 -DNZ=1024 -DSIM_TIME=1000 -DOUT_FREQ=-1 -DDATA_SPEC=DATA_SPEC_THERMAL \
  -DYAKL_ARCH=CUDA \
  -DYAKL_CUDA_FLAGS="-DHAVE_MPI -DNO_INFORM -O3 --use_fast_math -arch sm_100 -I$CONDA_PREFIX/include" \
  -DLDFLAGS="-L$CONDA_PREFIX/lib -lpnetcdf"
cmake --build build/level2/miniweather/cuda -j4 --target parallelfor parallelfor_test
cp level2/miniweather/cpp/build/check_output.sh build/level2/miniweather/cuda/
cd build/level2/miniweather/cuda && mpirun -np 1 ./parallelfor                        # run
TEST_MPI_COMMAND="mpirun -np 1" ./check_output.sh ./parallelfor_test 1e-9 4.5e-5      # validate
TEST_MPI_COMMAND="mpirun -np 1" ctest --test-dir . --output-on-failure                 # same, via upstream's YAKL_TEST
```

HIP (form after upstream `cmake_crusher_amd_gpu.sh`, untested):

```bash
OMPI_CXX=hipcc cmake -S level2/miniweather/cpp -B build/level2/miniweather/hip \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc \
  -DNX=2048 -DNZ=1024 -DSIM_TIME=1000 -DOUT_FREQ=-1 -DDATA_SPEC=DATA_SPEC_THERMAL \
  -DYAKL_ARCH=HIP \
  -DYAKL_HIP_FLAGS="-DHAVE_MPI -DNO_INFORM -O3 -ffast-math --rocm-path=$ROCM_PATH --offload-arch=gfx90a -x hip -Wno-unused-result -I$CONDA_PREFIX/include" \
  -DLDFLAGS="-L$CONDA_PREFIX/lib -lpnetcdf --rocm-path=$ROCM_PATH -L$ROCM_PATH/lib -lamdhip64"
cmake --build build/level2/miniweather/hip -j4 --target parallelfor parallelfor_test
```

### Benchmark problem (run.sh)

Rising thermal (`DATA_SPEC_THERMAL`) on upstream's GPU grid of 2048 x 1024
cells (from upstream's `cmake_thatchroof_gnu_gpu.sh` / `cmake_crusher_amd_gpu.sh`,
which also set `OUT_FREQ=-1`) for upstream's default `SIM_TIME=1000` s,
dx = dz = 9.77 m, dt = 0.0326 s (about 30,700 steps), double precision,
one MPI rank on one GPU. Observed on the B200 (GPU shared with other jobs,
so timings vary):

```
Using memory pool. Initial size: 22.293GB ;  Grow size: 22.293GB.
NVIDIA B200
nx_glob, nz_glob: 2048 1024
dx,dz: 9.765625 9.765625
dt: 0.032552
CPU Time: 32.2223 sec           (36.28 s and 51.48 s in two other runs)
d_mass: 0.000000e+00
d_te:   1.255246e-04
Pool Memory High Water Mark:       269330048
```

wall time 39-58 s including MPI/YAKL start-up. "CPU Time" is upstream's
name for the wall time of the time-step loop. The positive `d_te` here is
expected: upstream's energy criterion is stated only for the 400 s
validation problem below, not for other grids/times. Size sweep used to pick this: 1024x512, T=1000 ->
4.3 s; 2048x1024, T=100 -> 3.3 s; 4096x2048, T=200 -> 73 s.

## Validation

miniWeather has no reference solution; upstream's own check
(`README.md` "Checking for Correctness", implemented by
`cpp/build/check_output.sh` and run by `make test` / ctest `YAKL_TEST`) is
conservation on the fixed `parallelfor_test` problem: NX=100, NZ=50,
SIM_TIME=400 s, OUT_FREQ=400 (writes 2 frames of `output.nc` via PnetCDF),
DATA_SPEC_THERMAL, 1 rank. `validate.sh` runs exactly that
(`TEST_MPI_COMMAND="mpirun -np 1"`, tolerances 1e-9 and 4.5e-5 as in the
upstream CMake test) and compares the two summary lines the program prints:

- `d_mass` (relative change in domain-integrated mass): not NaN and
  |d_mass| < 1e-9 (mass is conserved to round-off by the FV scheme).
- `d_te` (relative change in domain-integrated total energy): not NaN,
  negative (hyper-viscosity can only remove energy) and |d_te| < 4.5e-5.

Observed here (CUDA, B200):

```
nx_glob, nz_glob: 100 50
dx,dz: 200.000000 200.000000
dt: 0.666667
CPU Time: 0.0594418 sec
d_mass: 0.000000e+00
d_te:   -4.141519e-05
PASS: miniweather CUDA (d_mass=0.000000e+00, |d_mass|<1e-9; d_te=-4.141519e-05, negative and |d_te|<4.5e-5; upstream check_output.sh criteria)
```

`validate.sh` takes ~15 s (two runs: one logged to
`build/level2/miniweather/cuda/validate.log`, one inside `check_output.sh`).
Cross-checks: `ctest` `YAKL_TEST` passes with the same numbers; lowering the
energy tolerance to 1e-6 makes `check_output.sh` fail as expected; a 2-rank
run (`mpirun --oversubscribe -np 2 ./parallelfor_test`, exercises the MPI
halo exchange) gives d_mass = 1.95e-16, d_te = -4.141519e-05.

## Warnings

After the build (CUDA, nvcc 13.2, two compilations of the same source):

- 2x nvcc `warning #550-D: variable "ierr" was set but never used` at
  `cpp/miniWeather_mpi_parallelfor.cpp` lines 379 and 521 (upstream code,
  unmodified).
- 1x `CMake Deprecation Warning at YAKL/CMakeLists.txt:1
  (cmake_minimum_required): Compatibility with CMake < 3.5 will be removed`
  (bundled YAKL, unmodified).

## LOC

miniWeather source (cloc 2.06, code lines; the same single source serves
CUDA and HIP through YAKL):

- CUDA: 659 (2 files: `cpp/miniWeather_mpi_parallelfor.cpp` 596,
  `cpp/const.h` 63)
- HIP: 659 (same files)

Bundled dependency YAKL, counted separately (`cpp/YAKL/src` +
`cpp/YAKL/external`): 13,563 lines of C/C++ headers in 76 files (of which
`external/` = ArrayIR.h + pocketfft_hdronly.h, 3,247 lines). CMake files,
`check_output.sh`, LICENSE and README files excluded.

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++17 (YAKL) | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
OpenMPI 5.0.10 (conda) | PnetCDF 1.15.0 (conda) | gfortran 14.2.1 (system, configure-time only)
HIP/ROCm: source + build config present, unverified (no AMD GPU available)
Red Hat Enterprise Linux 10.0

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Status

CUDA: Working (configure + build + run + validate verified on the B200 from a
clean shell: `build.sh` ~35 s, `run.sh` 39-58 s wall, `validate.sh` PASS).
HIP: build configuration present, untested (no ROCm on this machine;
`build.sh HIP` exits with "hipcc not found on PATH").
