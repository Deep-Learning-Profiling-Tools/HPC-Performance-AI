# P3-miniapps vlp4d

4D Vlasov-Poisson solver (2D position x 2D velocity) from the P3-miniapps
(Performance, Portability, Productivity) suite, CUDA/HIP programming model
(upstream `PROGRAMMING_MODEL=CUDA|HIP`, source directory `miniapps/vlp4d/thrust`).
The electron distribution function `f(x, y, vx, vy)` on a periodic 4D grid is
advanced with a semi-Lagrangian scheme: Strang-split 1D advections along
`x`, `y` (half steps), `vx`, `vy` (full steps) using 5th-order Lagrange
interpolation (`LAG_ORDER 5`), the charge density is obtained by reducing over
velocity space, and the self-consistent electric field comes from a 2D FFT Poisson
solve (cuFFT / rocFFT). Diagnostics (`nrj.out`: time, `log ||E||_2`, and a third
column that is `sum(phi)`) are written every step. Main motifs: 4D stencil-like
interpolation sweeps over `mdspan` views backed by `thrust::device_vector`,
reductions, batched 2D FFTs; no MPI.

## Provenance

Upstream repository: https://github.com/yasahi-hpc/P3-miniapps
Upstream commit: c1b946b91b608929ac7306e95e01222f726cdb5d (merge of PR #27 "more-docs", 2022-08-04)
Cloned: 2026-09-01 into `_upstream/level2/P3-miniapps` (recursive, submodules resolved)
License: MIT (copied as `LICENSE`); the bundled `ext_lib/mdspan` is under the Kokkos
3-clause BSD license (`ext_lib/mdspan/LICENSE`).

Copied from upstream (byte-identical unless listed under "Changes from upstream"):

- `CMakeLists.txt` (trimmed, see below), `cmake/FindFFTW.cmake` (referenced by the
  upstream module path; unused by the CUDA/HIP build)
- `lib/CMakeLists.txt`, `lib/Iteration.hpp`, `lib/Layout.hpp`, `lib/ComplexType.hpp`,
  `lib/Cuda_FFT.hpp`, `lib/Cuda_Helper.hpp`, `lib/HIP_FFT.hpp`, `lib/HIP_Helper.hpp`,
  `lib/thrust/` (all 12 files: the `Impl::for_each` / `Impl::transform_reduce` /
  `View` / `FFT` layer used by the CUDA and HIP models; `OpenMP_Parallel_For.hpp`,
  `Thrust_Parallel_For.hpp`, `Transpose.hpp`, `Utils.hpp` are shipped for
  completeness of the directory but are not compiled by vlp4d)
- `ext_lib/mdspan/{CMakeLists.txt,LICENSE,cmake/,include/}` -- the upstream mdspan
  submodule (kokkos/mdspan 0.4.0, commit 3aad018ddeb1a5ea089282ce73c92beda07071eb),
  header-only reference implementation of `std::experimental::mdspan`
- `miniapps/CMakeLists.txt`, `miniapps/vlp4d/{CMakeLists.txt,timer.hpp}`,
  `miniapps/vlp4d/thrust/` (15 files)
- `wk/SLD10.dat`, `wk/SLD10_large.dat` -- the two upstream input decks for test
  case 10 (2D Landau damping)
- `LICENSE`

Left out: the other programming models (`miniapps/vlp4d/{kokkos,openacc,openmp,stdpar}`,
`lib/{kokkos,openacc,openmp,stdpar}`, `lib/OpenMP_*.hpp`, `lib/*_Transpose.hpp`),
`vlp4d_mpi`, `heat3d*`, `ext_lib/{kokkos,googletest}`, `tests/`, `docs/`, the other
`wk/*.dat` decks (TSI20, MPI decks), CI files.

## Changes from upstream

- `lib/thrust/View.hpp` (`View::swap`, lines 231/233): `thrust::swap(a, b)` on
  `thrust::host_vector` / `thrust::device_vector` replaced by the member call
  `a.swap(b)`. Reason: CUDA 13.2 ships CCCL 3.x, which removed the free-function
  `thrust::swap` overloads for these containers (hard error "no matching function
  for call to swap(host_vector_type&, host_vector_type&)"). The member function
  has identical O(1) semantics and also exists in rocThrust. Marked with
  `[HPC-Performance-AI]` comments.
- `CMakeLists.txt` (top level) trimmed: removed the `PROGRAMMING_MODEL STREQUAL "KOKKOS"`
  block (`add_subdirectory(ext_lib/kokkos)`) and the `BUILD_TESTING` block
  (`ext_lib/googletest`, `tests/`) because those directories are not shipped;
  `include(CTest)` is kept so `-DBUILD_TESTING=OFF` remains a recognised option;
  the `APPLICATION` cache default is `vlp4d` instead of `AUTO` (with `AUTO`
  `miniapps/CMakeLists.txt` would try to add the four non-shipped app directories).
  All other CMake files (`lib/`, `lib/thrust/`, `miniapps/`, `miniapps/vlp4d/`,
  `miniapps/vlp4d/thrust/`) are byte-identical.
- Not changed, but noted: `lib/thrust/CMakeLists.txt` requests the CUDAToolkit
  component `culbas` (typo for `cublas`). CMake 3.28's `FindCUDAToolkit` ignores
  unknown component names and still defines `CUDA::cublas`, so configure and link
  succeed; the file is left byte-identical.
- Kept C++17 (`cuda_std_17` as upstream); no C++20 needed with GCC 13 / nvcc 13.2.
- No kernel, algorithm or default-parameter changes.

## Dependencies

- CUDA Toolkit 13.2 (`nvcc`, Thrust/CCCL, cuFFT; cuBLAS is linked by the shared
  `math_lib` target but not called) -- system install `/usr/local/cuda` via
  `hpcperf_env.sh`.
- GCC 13.3.0 as host compiler -- conda env via `hpcperf_env.sh`.
- mdspan -- bundled in `ext_lib/mdspan` (used automatically when no installed
  `mdspan` CMake package is found).
- HIP variant: ROCm with `hipcc`, rocThrust, rocFFT, rocBLAS (`find_package(HIP)`,
  `find_package(rocthrust|rocfft|rocblas CONFIG)`) -- not available on this machine.
- No MPI, no Kokkos; `validate.sh` uses only `awk`.

## Build / Run / Validate

```bash
cd /projects/kzhou6/bcui2/research/HPC-Performance-AI && source hpcperf_env.sh
level2/p3_vlp4d/build.sh             # CUDA (default); HPCPERF_CUDA_ARCH=100 overrides detection
level2/p3_vlp4d/run.sh               # SLD10_large.dat: 128^4, dt 0.125, 128 steps
level2/p3_vlp4d/run.sh CUDA SLD10.dat   # the small upstream deck (dt 0.01, 40 steps)
level2/p3_vlp4d/validate.sh          # PASS/FAIL, exit 0/1
level2/p3_vlp4d/build.sh HIP         # HIP variant (needs hipcc; untested here)
```

The scripts source `hpcperf_env.sh` themselves when needed and can be called from
any directory. Build tree: `build/level2/p3_vlp4d/{cuda,hip}`; executable
`build/level2/p3_vlp4d/cuda/miniapps/vlp4d/thrust/vlp4d`. Because vlp4d writes
`nrj.out` into the current directory (and `data/vlp4d/fxvx_*.csv` if the deck sets
"Diagnostics of fxvx" to a non-zero value), `run.sh` and `validate.sh` execute in
`build/level2/p3_vlp4d/<backend>/{run,validate}/` and create `data/vlp4d/` there.

Equivalent raw commands (CUDA):

```bash
cmake -S level2/p3_vlp4d -B build/level2/p3_vlp4d/cuda -DAPPLICATION=vlp4d \
      -DPROGRAMMING_MODEL=CUDA -DBACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=100 \
      -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CUDA_HOST_COMPILER=$CXX \
      -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/level2/p3_vlp4d/cuda -j4
mkdir -p build/level2/p3_vlp4d/cuda/run && cd build/level2/p3_vlp4d/cuda/run
../miniapps/vlp4d/thrust/vlp4d ../../../../../level2/p3_vlp4d/wk/SLD10_large.dat
```

HIP (form as upstream documents; unverified here):

```bash
cmake -S level2/p3_vlp4d -B build/level2/p3_vlp4d/hip -DAPPLICATION=vlp4d \
      -DPROGRAMMING_MODEL=HIP -DBACKEND=HIP -DCMAKE_CXX_COMPILER=hipcc \
      -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release   # optionally -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build build/level2/p3_vlp4d/hip -j4
```

Standard run (upstream `docs/vlp4d.md` example, `SLD10_large.dat`) observed on the B200:

```
total 2.96834 [s], 1 calls
MainLoop 2.96389 [s], 128 calls
advec1D_x 0.463202 [s], 256 calls
advec1D_y 0.955187 [s], 256 calls
advec1D_vx 0.423655 [s], 128 calls
advec1D_vy 0.519404 [s], 128 calls
field 0.498417 [s], 128 calls
Fourier 0.058212 [s], 128 calls
diag 0.0457508 [s], 128 calls
```

(about 10 s wall including host-side initialisation of the 128^4 array; upstream
reports `total 4.78 s` on an A100. With other jobs sharing the GPU the main loop
was seen at up to 10.9 s.)

## Validation

vlp4d has no built-in reference check and upstream documents no expected numbers;
its only output is `nrj.out` with `t`, `log(||E||_2)` and `sum(phi)`. The third
column is not usable: `Diags::compute` sums the array `rho` after the Poisson solve
has overwritten it with the potential, so it is ~1e-15 for every step by
construction (the upstream `docs/vlp4d.md` labels it "mass" but it does not measure
mass conservation).

The check used instead is the physics of the test case. `SLD10_large.dat` is the
2D Landau damping problem of Crouseilles, Latu and Sonnendruecker, JCP 228 (2009),
sect. 5.3.1: `f0 = (2 pi)^-1 exp(-(vx^2 + vy^2)/2) (1 + 0.05 cos(0.5 x) cos(0.5 y))`
on `[0, 4 pi]^2 x [-9, 9]^2`, so the perturbed modes are `k = (+-0.5, +-0.5)`,
`|k| = 0.5 sqrt(2)`. For a unit-temperature Maxwellian the linear dispersion relation
`1 - Z'(omega / (sqrt(2) |k|)) / (2 |k|^2) = 0` (`Z` = plasma dispersion function,
solved here with scipy's Faddeeva function; the same solver reproduces the textbook
`k = 0.5` root `1.4156 - 0.1533 i`) gives `omega = 1.68289 - 0.40208 i`. Hence the
peaks of `log ||E||_2` must lie on a line of slope `gamma = -0.4021` and be spaced
`pi / omega_r = 1.867` apart. `validate.sh`:

1. runs `SLD10_large.dat` and requires exit code 0 and 129 finite lines in `nrj.out`;
2. finds the local maxima of `log ||E||_2` (needs at least 5);
3. least-squares fits a line through them: the slope must match `-0.4021` within 5 %;
4. checks `pi (n_max - 1) / (t_last - t_first)` against `1.6829` within 5 %.

Observed here (8 maxima, t = 2.25 .. 15.375):

```
damping rate  gamma = -0.4018   linear theory -0.4021   rel. diff 0.001   (tol 0.05)
frequency   omega_r = 1.6755   linear theory 1.6829   rel. diff 0.004   (tol 0.05)
log||E||_2: t=0 -> -0.8113, t=16 -> -7.8006
PASS: p3_vlp4d (CUDA) Landau damping rate and frequency of ||E||_2 match linear theory (gamma=-0.40208, omega=1.68289, rel tol 0.05) for SLD10_large.dat
```

The 0.4 % frequency offset is the quantisation of the peak positions to the
`dt = 0.125` output cadence. A broken advection, interpolation or Poisson solve
changes the damping rate by O(1) or makes the field grow, so the 5 % tolerance is
robust while still tolerating GPU-to-GPU FFT/reduction round-off (~1e-12). A
deliberately wrong reference rate makes the script print `FAIL` and exit 1 (checked).
Validation takes about 10 s.

## Warnings

1 (upstream code, left as is): `miniapps/vlp4d/thrust/diags.cpp(23): warning #177-D:
variable "zeros" was declared but never referenced`.

## LOC

cloc 2.06, code lines only; CMake files, data decks, scripts and READMEs excluded.

- CUDA variant: **1965** (26 files) = `miniapps/vlp4d/timer.hpp` +
  `miniapps/vlp4d/thrust/*` (together 15 files, 1117 lines) + the lib pieces it
  compiles: `lib/{Iteration,Layout,ComplexType,Cuda_FFT,Cuda_Helper}.hpp`,
  `lib/thrust/{View,Parallel_For,Cuda_Parallel_For,Parallel_Reduce,Thrust_Parallel_Reduce,FFT}.hpp`
  (848 lines).
- HIP variant: **2002** -- same with `lib/thrust/HIP_Parallel_For.hpp` (180),
  `lib/HIP_FFT.hpp` (178), `lib/HIP_Helper.hpp` (16) instead of the CUDA files
  (183, 139, 15).
- For reference: all of `lib/` as shipped (18 headers incl. unused OpenMP/Thrust/Transpose
  wrappers) is 1515 lines; the bundled third-party `ext_lib/mdspan/include` is 4051
  lines (19 `.hpp` files; the two extensionless umbrella headers `experimental/mdspan` and `experimental/mdarray` are not recognised by cloc) and is not counted above.

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++17 | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
HIP/ROCm: source + build config present where noted, unverified (no AMD GPU available)
RHEL 10 (kernel 6.12); no MPI used.

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Status

CUDA: **Working** -- configure + build + run + validate passed on this machine
(build 30 s, run ~10 s, validate ~10 s).
HIP: extracted, untested (no hipcc/ROCm on this machine; `build.sh HIP` stops with a
clear error).
