# ExaMPM

ECP CoPA proxy application for the Material Point Method (MPM): a weakly
compressible fluid is represented by Lagrangian particles that carry mass,
volume, velocity, an APIC affine-velocity matrix and the deformation
determinant J, and is advanced on a background Cartesian grid. Each step does
a particle-to-grid transfer (P2G: mass, momentum), a grid field solve
(pressure from a Tait-like equation of state, gravity, boundary conditions),
a grid-to-particle transfer (G2P) and a particle position correction, with a
CFL time-step controller (dt only ever shrinks). Particles live in a Cabana
AoSoA, grid fields in Cabana::Grid arrays with MPI halo exchange, all
kernels are Kokkos `parallel_for`s. Two example drivers: `DamBreak` (water
column collapsing in a closed 1 m^3 box with free-slip walls -- the
benchmark) and `FreeFall` (a sphere falling through a fully periodic box --
used for validation). Output is HDF5 + XDMF particle dumps.

## Provenance

- Upstream: https://github.com/ECP-copa/ExaMPM
- Commit: `1cdc9869356e9dbbbeaee0c49b57d6ffe7f62d8e` (2025-02-19, "Merge pull
  request #61 from ECP-copa/ci_base_image"), cloned 2026-09-01 into
  `_upstream/level2/ExaMPM`
- License: BSD-3-Clause (`LICENSE`, Copyright 2018-2020 the ExaMPM authors)
- Copied: `src/` (all 12 files), `examples/` (`CMakeLists.txt`,
  `dam_break.cpp`, `free_fall.cpp`), `CMakeLists.txt`, `cmake/FindCLANG_FORMAT.cmake`,
  `LICENSE`. Not copied: `.clang-format`, `.github/`, `.gitignore`, upstream
  `README.md` (three links to the wiki). Upstream has no unit tests.
- Added: `build.sh`, `run.sh`, `validate.sh`, this README.

## Changes from upstream

One source change, needed to compile against Cabana 0.8.0 (upstream targets
Cabana >= 0.6.1, where `createLocalMesh` still accepted an execution space or
device type; Cabana 0.8's `LocalMesh` has
`static_assert( Kokkos::is_memory_space<MemorySpace>() )`):

`src/ExaMPM_TimeIntegrator.hpp`, three occurrences (p2g, g2p,
correctParticlePositions):

```diff
-    auto local_mesh = Cabana::Grid::createLocalMesh<ExecutionSpace>(
+    auto local_mesh = Cabana::Grid::createLocalMesh<typename ExecutionSpace::memory_space>(
         *( pm.mesh()->localGrid() ) );
```

Everything else (`examples/`, `CMakeLists.txt`, `cmake/`, `LICENSE`) is
byte-identical to upstream (`diff -ru _upstream/level2/ExaMPM/{src,examples,cmake} level2/exampm/`).
`find_package(Cabana 0.6.1 REQUIRED COMPONENTS Cabana::Grid Cabana::Core)`
is satisfied by Cabana 0.8.0 and was left alone. `find_package(CLANG_FORMAT 14)`
is optional and not found here (only defines a `format` target).

Runtime, not source: the scripts export `OMPI_MCA_opal_cuda_support=true`
(see Dependencies).

## Dependencies

All from the project toolchain (`hpcperf_env.sh`); nothing was rebuilt.

- Kokkos 5.2.1, `$R/.deps/install/kokkos` (CUDA + OpenMP + Serial, C++20,
  `Kokkos_ARCH_BLACKWELL100`, `Kokkos_ENABLE_IMPL_VIEW_LEGACY`), compiled
  through its `bin/nvcc_wrapper` (host compiler = conda g++ 13.3.0).
- Cabana 0.8.0, `$R/.deps/install/cabana` (Core + Grid, MPI, parallel HDF5
  2.2.0, heFFTe 2.4.1; no Silo -- ExaMPM prefers HDF5 and only falls back to
  Silo when HDF5 is off).
- OpenMPI 5.0.10 (conda). Cabana's grid halo exchange and particle
  migration hand CudaSpace buffers straight to `MPI_Isend`/`MPI_Irecv`, so a
  CUDA-aware MPI is needed whenever a rank has a neighbour -- and with
  periodic boundaries a single rank is its own neighbour (FreeFall is
  periodic in all three directions). The conda OpenMPI is built with CUDA
  support (`mpi_built_with_cuda_support = true`, components
  `accelerator/cuda`, `btl/smcuda`), but its
  `.conda_env/etc/openmpi-mca-params.conf` sets `opal_cuda_support = 0`,
  which makes `accelerator_cuda_init()` return NULL before touching CUDA;
  `MPIX_Query_cuda_support()` then reports 0 and any device-buffer send
  segfaults in `mca_pml_ob1_send_request_start_rdma -> memcpy`. Setting the
  MCA parameter back per run (`OMPI_MCA_opal_cuda_support=true`, done in
  `run.sh`) is sufficient: `MPIX_Query_cuda_support()` = 1 and FreeFall
  runs. The non-periodic DamBreak deck on one rank has no MPI traffic and
  runs either way. (Verified with a 40-line test program doing a
  device-buffer self `Isend/Irecv`; the same fix is used by `level2/examinimd`.)
- HDF5 tooling for validation: `h5dump` (conda HDF5 2.2.0) + numpy; h5py is
  not in the environment.

## Build / Run / Validate

```bash
cd $R && source hpcperf_env.sh
level2/exampm/build.sh CUDA        # -> build/level2/exampm/cuda/examples/{DamBreak,FreeFall}, ~64 s at -j4
level2/exampm/run.sh CUDA          # DamBreak 0.01 2 0 0.001 1.0 1000 cuda, ~14 s on one B200
level2/exampm/validate.sh CUDA     # FreeFall vs analytic free fall (+ DamBreak conservation), ~30 s, PASS/FAIL
level2/exampm/build.sh HIP         # form only: needs .deps/install/{kokkos-hip,cabana-hip} + hipcc (untested)
```

`build.sh [CUDA|HIP] [extra -D...]` detects the GPU with
`nvidia-smi --query-gpu=compute_cap` (override `HPCPERF_CUDA_ARCH`), but the
architecture is fixed by the Kokkos install (it only checks that the two
agree and warns otherwise). `HPCPERF_BUILD_JOBS` (default 4),
`HPCPERF_KOKKOS_ROOT`, `HPCPERF_CABANA_ROOT`, `HIPCXX`.

Raw CMake equivalent of `build.sh CUDA` (upstream requires CMake >= 3.12, so
`<Pkg>_ROOT` is honoured):

```bash
cmake -S level2/exampm -B build/level2/exampm/cuda -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=$R/.deps/install/kokkos/bin/nvcc_wrapper \
      -DKokkos_ROOT=$R/.deps/install/kokkos -DCabana_ROOT=$R/.deps/install/cabana
cmake --build build/level2/exampm/cuda -j4
```

`run.sh [CUDA|HIP] [cell_size ppc halo_cells dt t_final write_freq]` runs
upstream's executable with its documented seven positional arguments
(`Usage: ./DamBreak cell_size parts_per_cell_size halo_cells dt t_end write_freq exec_space`,
exec_space = serial|openmp|cuda|hip, filled in from the backend):

```
mpirun -np 1 build/level2/exampm/cuda/examples/DamBreak 0.01 2 0 0.001 1.0 1000 cuda
```

i.e. upstream's CI deck `DamBreak 0.05 2 0 0.001 0.25 100` (15,360
particles, 2.8 s here, mostly start-up) refined from a 20^3 to a 100^3 cell
grid and run to t = 1.0 s: 1,920,000 particles, initial dt 1e-3 s that the
CFL controller (0.5 h / (|u|+|v|+|w| + c sqrt(3)), c = sqrt(B/rho) = 10 m/s)
shrinks to ~2e-4 s, ~4,600 steps, an HDF5/XDMF dump every 1000 steps
(5 x 107 MB). Measured 13.6-14.0 s wall on one B200 (three runs). The
output is upstream's normal output:

```
# ExaMPM CUDA: mpirun -np 1 .../DamBreak 0.01 2 0 0.001 1.0 1000 cuda  (cwd .../build/level2/exampm/cuda/run)
Time 0.000000 / 1.000000
Time 0.251085 / 1.000000
Time 0.460846 / 1.000000
Time 0.667591 / 1.000000
Time 0.874335 / 1.000000
```

ExaMPM writes `particles_<step>.h5/.xmf` into its cwd, so `run.sh` runs in
`build/level2/exampm/<cuda|hip>/run` (override `HPCPERF_EXAMPM_RUN_DIR`),
removing the previous dumps first. `HPCPERF_EXAMPM_EXAMPLE=FreeFall`
selects the other driver, `HPCPERF_NP` the rank count (one GPU per rank).
Other sizes measured (all `2 0 0.001`): `0.02 ... 0.25 1000` 4.2 s (240k
particles); `0.01 ... 0.25 500` 4.7 s; `0.005 ... 0.25 2000` 26.4 s
(15.4 M particles, 860 MB per dump).

Kokkos' OpenMP host backend is initialised too; the scripts set
`OMP_NUM_THREADS=1 OMP_PROC_BIND=spread OMP_PLACES=threads` (all work is on
the GPU; this also silences Kokkos' OMP_PROC_BIND warning).

## Validation

ExaMPM has no built-in check and upstream ships no reference output, so
`validate.sh` compares the simulation with an analytic solution.

**Primary -- free fall (always run).** `FreeFall 0.05 2 0 0.001 0.25 50 cuda`:
a sphere of fluid (r = 0.25 m, rho = 1000 kg/m^3, 4,224 particles on a 20^3
grid) at rest at the origin of a fully periodic 1 m^3 box, gravity 9.81 m/s^2
in -z, no walls (400 steps, 9 dumps, ~6 s). Only gravity acts on the fluid,
so its centre of mass is in free fall: v_cm,z = -g t, z_cm = z0 - g t^2/2,
v_cm,x = v_cm,y = 0. APIC P2G/G2P conserve linear momentum exactly and
gravity enters the grid velocity as g dt, so the mean particle velocity must
equal -g t to round-off; positions use symplectic Euler
(x_{n+1} = x_n + v_{n+1} dt_n), whose discrete centre of mass lags the
continuous one by (g/2) sum dt_n^2 <= (g/2) t dt_0. For every dump
(`position` Nx3, `velocity` Nx3, attribute `Time`, read with
`h5dump -b LE` + numpy):

| check | tolerance |
|---|---|
| particle count N == initial N | exact |
| \|mean v_z + g t\| | <= 1e-9 max(1, g t) |
| \|mean v_x\|, \|mean v_y\| | <= 1e-9 |
| \|z_cm - (z0 - g t^2/2)\| (z unwrapped through the periodic boundary per particle) | <= g t dt_0 + 1e-12 (2x the symplectic-Euler bound) |
| last dump at t >= 0.9 t_final | run completed |

The sphere falls 0.31 m by t = 0.25 s and crosses the periodic z boundary, so
the grid halo exchange and particle migration over MPI (device buffers) are
exercised, not only the kernels. Result on the B200 build (from
`build/level2/exampm/cuda/validate/validate.log`):

```
  file                       t       N      mean vz         -g t     |dvz|  z_cm-z_exact       tol
  particles_0.h5      0.000000    4224  +0.00000000  -0.00000000  0.00e+00    +0.000e+00  1.00e-12
  particles_50.h5     0.032147    4224  -0.31535816  -0.31535816  1.67e-16    -1.012e-04  3.15e-04
  particles_200.h5    0.127065    4224  -1.24650666  -1.24650666  3.55e-15    -3.957e-04  1.25e-03
  particles_400.h5    0.250285    4224  -2.45529827  -2.45529827  4.84e-14    -7.681e-04  2.46e-03
  FreeFall centre-of-mass check: ok
```

(|dvz| is at round-off, 1e-16..5e-14; the centre-of-mass error is the
expected O(dt) symplectic-Euler lag at ~1/3 of the tolerance.) A negative
test with g perturbed by 1e-6 relative in the checker fails every dump, so
the check is sensitive.

**Secondary -- dam break (only if `run.sh` output exists).** Across all
dumps of the benchmark deck: particle count constant and positions finite
(exact); all positions within [-0.02, 1.02]^3 (particles may overshoot the
free-slip wall by a fraction of a cell before the boundary velocity and
position correction push them back; observed -0.003..1.003); total volume
sum(J_p)/N within 1 % of 1 (near-incompressible fluid; observed
0.9980-1.0000). 

```
  particles_4000.h5   0.874335  1920000   -0.00308   +1.00264   0.998993
  DamBreak conservation check: ok
ExaMPM CUDA validation (FreeFall 0.05 2 0 0.001 0.25 50 vs analytic free fall): PASS
```

`validate.sh` exits 0 on PASS, 1 on FAIL, and logs to
`build/level2/exampm/<cuda|hip>/validate/validate.log`. It runs FreeFall in
its own directory so the benchmark dumps survive. `HPCPERF_EXAMPM_FF_ARGS`
overrides the six FreeFall arguments.

## Warnings

Build (clean, `build.sh CUDA`, log checked with grep; 0 errors):

- CMake: `The installed Kokkos configuration does not support CXX extensions`
  (Kokkos forces `CMAKE_CXX_EXTENSIONS=Off`; informational) and
  `Could NOT find CLANG_FORMAT ... required is at least "14"` (optional
  `format` target only).
- 5x `nvcc_wrapper - *warning* you have set multiple optimization flags (-O*), only the last is used`:
  conda's `CXXFLAGS` carry `-O2` and CMake Release adds `-O3`; `-O3` wins.
  Toolchain-wide, not ExaMPM's.
- 4x nvcc `Warning #20014-D: calling a __host__ function from a __host__ __device__ function is not allowed`,
  all inside `Kokkos_ViewLegacy.hpp:1033` reached from
  `Cabana_Migrate_Mpi.hpp:120` (`Cabana::migrate`, instantiated for
  HostSpace and CudaSpace in each of `dam_break.cpp` and `free_fall.cpp`).
  Kokkos/Cabana header code, not ExaMPM's.

Runtime: none after the OpenMP binding variables are set (without them
Kokkos prints its `OMP_PROC_BIND environment variable not set` notice).
Without `OMPI_MCA_opal_cuda_support=true`, FreeFall (or any multi-rank /
periodic run) segfaults in `MPI_Isend` -- see Dependencies.

## LOC

cloc v2.06, code lines only (blank/comment excluded); `CMakeLists.txt`,
scripts and this README excluded. ExaMPM is a Kokkos/Cabana application:
the same source is the CUDA and the HIP variant, so both counts are the
same.

CUDA variant = HIP variant: **1501** (13 files) = `src/` 10 headers + 1 .cpp + `examples/` 2 .cpp
(C/C++ header 1276 + C++ 225)

```bash
cloc --quiet level2/exampm/src level2/exampm/examples --exclude-ext=txt
```

(cloc's SUM line reads 1534 / 15 files because it still recognises the two
`CMakeLists.txt` by name -- 33 CMake lines -- which are excluded above.)

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++20 | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
OpenMPI 5.0.10 (conda, CUDA-aware build; CUDA support disabled by its
`openmpi-mca-params.conf` and re-enabled per run with `OMPI_MCA_opal_cuda_support=true`)
Kokkos 5.2.1 + Cabana 0.8.0 + HDF5 2.2.0 + heFFTe 2.4.1 from `$R/.deps/install`
HIP/ROCm: source + build config present where noted, unverified (no AMD GPU available)
OS: RHEL 10 (Linux 6.12)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Status

- CUDA: builds, runs (DamBreak benchmark 13.6-14.0 s on one B200), validates
  (PASS). All three scripts verified from a foreign cwd with an empty
  environment (`env -i ... bash -c level2/exampm/<script>.sh`).
- HIP: `build.sh HIP` is correct in form (`.deps/install/{kokkos-hip,cabana-hip}`,
  `hipcc`); on this machine it exits 1 with
  `HIP requested but hipcc was not found (no ROCm on this machine)`. Untested.
- Not done: multi-rank runs (one GPU per rank; the machine has several GPUs
  but the benchmark is a single-GPU one), Silo output (Cabana built without
  Silo), a Serial-vs-CUDA cross-check (the same binary contains the serial,
  openmp and cuda paths; `run.sh` always passes the backend, but e.g.
  `mpirun -np 1 build/level2/exampm/cuda/examples/FreeFall 0.05 2 0 0.001 0.25 50 serial`
  can be run by hand and checked with the same analytic criteria).
