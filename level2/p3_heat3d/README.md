# P3-miniapps heat3d

3D heat-equation solver from the P3-miniapps (Performance, Portability, Productivity)
suite, CUDA/HIP programming model (upstream `PROGRAMMING_MODEL=CUDA|HIP`, source
directory `miniapps/heat3d/thrust`). The unknown `u(x,y,z)` on a periodic unit cube
(`L = 1`, `kappa = 1`) is advanced with the explicit FTCS 7-point stencil
(`dt = 0.1 dx^2 / kappa`), double buffered, for `nbiter` steps. Initial condition
`u0 = cos(2 pi (x + y + z))` has the analytical solution
`u = u0 exp(-3 kappa (2 pi)^2 t)`, and the program prints the L2 norm of the error
against it at the end. Main motif: memory-bound 3D stencil on `mdspan` views over
`thrust::device_vector` storage, launched through a small hand-written
`Impl::for_each` layer (native CUDA/HIP grid-stride kernels) plus one
`thrust::transform_reduce` for the error norm; no MPI.

## Provenance

Upstream repository: https://github.com/yasahi-hpc/P3-miniapps
Upstream commit: c1b946b91b608929ac7306e95e01222f726cdb5d (merge of PR #27 "more-docs", 2022-08-04)
Cloned: 2026-09-01 into `_upstream/level2/P3-miniapps` (recursive, submodules resolved)
License: MIT (copied as `LICENSE`); the bundled `ext_lib/mdspan` is under the Kokkos
3-clause BSD license (`ext_lib/mdspan/LICENSE`).

Copied from upstream (byte-identical unless listed under "Changes from upstream"):

- `CMakeLists.txt` (trimmed, see below), `cmake/FindFFTW.cmake` (referenced by the
  upstream module path; unused by the CUDA/HIP build)
- `lib/CMakeLists.txt`, `lib/Iteration.hpp`, `lib/thrust/` (all 12 files: the
  `Impl::for_each` / `Impl::transform_reduce` / `View` layer used by the CUDA and
  HIP models; `OpenMP_Parallel_For.hpp`, `Thrust_Parallel_For.hpp`, `Transpose.hpp`,
  `Utils.hpp`, `FFT.hpp` are shipped for completeness of the directory but are not
  compiled by heat3d)
- `ext_lib/mdspan/{CMakeLists.txt,LICENSE,cmake/,include/}` -- the upstream mdspan
  submodule (kokkos/mdspan 0.4.0, commit 3aad018ddeb1a5ea089282ce73c92beda07071eb),
  header-only reference implementation of `std::experimental::mdspan`
- `miniapps/CMakeLists.txt`, `miniapps/heat3d/{CMakeLists.txt,Helper.hpp,Parser.hpp,Timer.hpp}`,
  `miniapps/heat3d/thrust/` (9 files)
- `LICENSE`

Left out: the other programming models (`miniapps/heat3d/{kokkos,openacc,openmp,stdpar}`,
`lib/{kokkos,openacc,openmp,stdpar}`), `heat3d_mpi`, `vlp4d*`, `ext_lib/{kokkos,googletest}`,
`tests/`, `docs/`, `wk/`, CI files.

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
  the `APPLICATION` cache default is `heat3d` instead of `AUTO` (with `AUTO`
  `miniapps/CMakeLists.txt` would try to add the four non-shipped app directories).
  All other CMake files (`lib/`, `lib/thrust/`, `miniapps/`, `miniapps/heat3d/`,
  `miniapps/heat3d/thrust/`) are byte-identical.
- Not changed, but noted: `lib/thrust/CMakeLists.txt` requests the CUDAToolkit
  component `culbas` (typo for `cublas`). CMake 3.28's `FindCUDAToolkit` ignores
  unknown component names and still defines `CUDA::cublas`, so configure and link
  succeed; the file is left byte-identical.
- Kept C++17 (`cuda_std_17` as upstream); no C++20 needed with GCC 13 / nvcc 13.2.
- No kernel, algorithm or default-parameter changes.

## Dependencies

- CUDA Toolkit 13.2 (`nvcc`, Thrust/CCCL, cuFFT and cuBLAS are linked by the shared
  `math_lib` target although heat3d itself uses neither) -- system install
  `/usr/local/cuda` via `hpcperf_env.sh`.
- GCC 13.3.0 as host compiler -- conda env via `hpcperf_env.sh`.
- mdspan -- bundled in `ext_lib/mdspan` (used automatically when no installed
  `mdspan` CMake package is found).
- HIP variant: ROCm with `hipcc`, rocThrust, rocFFT, rocBLAS (`find_package(HIP)`,
  `find_package(rocthrust|rocfft|rocblas CONFIG)`) -- not available on this machine.
- No MPI, no Kokkos, no Python.

## Build / Run / Validate

```bash
cd /projects/kzhou6/bcui2/research/HPC-Performance-AI && source hpcperf_env.sh
level2/p3_heat3d/build.sh            # CUDA (default); HPCPERF_CUDA_ARCH=100 overrides detection
level2/p3_heat3d/run.sh              # 512^3, 1000 steps, prints L2_norm + timers
level2/p3_heat3d/validate.sh         # PASS/FAIL, exit 0/1
level2/p3_heat3d/build.sh HIP        # HIP variant (needs hipcc; untested here)
```

The scripts source `hpcperf_env.sh` themselves when needed and can be called from
any directory. Build tree: `build/level2/p3_heat3d/{cuda,hip}`; executable
`build/level2/p3_heat3d/cuda/miniapps/heat3d/thrust/heat3d`. `run.sh` and
`validate.sh` execute in `build/level2/p3_heat3d/<backend>/{run,validate}/` (and
create `data/heat3d/` there, where heat3d writes CSV slices if `--freq_diag N > 0`
is passed as an extra argument).

Equivalent raw commands (CUDA):

```bash
cmake -S level2/p3_heat3d -B build/level2/p3_heat3d/cuda -DAPPLICATION=heat3d \
      -DPROGRAMMING_MODEL=CUDA -DBACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=100 \
      -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_CUDA_HOST_COMPILER=$CXX \
      -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/level2/p3_heat3d/cuda -j4
build/level2/p3_heat3d/cuda/miniapps/heat3d/thrust/heat3d --nx 512 --ny 512 --nz 512 --nbiter 1000 --freq_diag 0
```

HIP (form as upstream documents; unverified here):

```bash
cmake -S level2/p3_heat3d -B build/level2/p3_heat3d/hip -DAPPLICATION=heat3d \
      -DPROGRAMMING_MODEL=HIP -DBACKEND=HIP -DCMAKE_CXX_COMPILER=hipcc \
      -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release   # optionally -DCMAKE_HIP_ARCHITECTURES=gfx90a
cmake --build build/level2/p3_heat3d/hip -j4
```

Standard run (upstream `docs/heat3d.md` example) observed on the B200:

```
(nx, ny, nz) = 512, 512, 512
L2_norm: 0.00355178
Backend: CUDA
Elapsed time: 1.64845 [s]
Bandwidth: 1302.73 [GB/s]
Flops: 732.787 [GFlops]
MainLoop 1.64843 [s], 1000 calls
```

(about 10 s wall including host-side initialisation of the 512^3 arrays; the
upstream reference for the same command on a V100 is `L2_norm: 0.00355178`).

## Validation

heat3d has no built-in pass/fail, but it prints the L2 norm of
`u_numerical - u_analytical` after the last step, and the upstream documentation
gives the reference value for the standard command (`L2_norm: 0.00355178`). Because
the initial condition is an eigenmode of the discrete Laplacian, the FTCS result is
known in closed form, `u_n = A^n u0` with `A = 1 + 0.1 (6 cos(2 pi dx) - 6)`, so the
printed norm equals `|A^n - exp(-3 kappa (2 pi)^2 n dt)| sqrt(N^3 / 2)` for any
`N`, `n`. `validate.sh` runs two cases and compares the printed `L2_norm` with these
values (relative tolerance 1e-3; the value has 6 printed digits and GPU reduction
order only affects the ~1e-12 level):

| case | expected | observed (B200) |
|---|---|---|
| 128^3, 1000 steps | 0.0577255 (closed form) | 0.0577255 |
| 512^3, 1000 steps | 0.00355178 (closed form = upstream reference) | 0.00355178 |

Result here: `PASS: p3_heat3d (CUDA) L2_norm vs. analytical solution matches the
closed-form/upstream reference (rel tol 1e-3) for 128^3 and 512^3, 1000 steps`
(about 10 s). A deliberately wrong expected value makes the script print `FAIL`
and exit 1 (checked).

## Warnings

None (CUDA build with nvcc 13.2 / GCC 13.3, `-O3`).

## LOC

cloc 2.06, code lines only; CMake files, data, scripts and READMEs excluded.

- CUDA variant: **1160** (17 files) = `miniapps/heat3d/{Helper,Parser,Timer}.hpp` +
  `miniapps/heat3d/thrust/*` (together 11 files, 511 lines) + the lib pieces it compiles:
  `lib/Iteration.hpp`, `lib/thrust/{View,Parallel_For,Cuda_Parallel_For,Parallel_Reduce,Thrust_Parallel_Reduce}.hpp`
  (649 lines).
- HIP variant: **1157** -- same with `lib/thrust/HIP_Parallel_For.hpp` (180) instead
  of `Cuda_Parallel_For.hpp` (183).
- For reference: all of `lib/` as shipped (12 headers incl. unused OpenMP/Thrust/FFT/Transpose
  wrappers) is 1132 lines; the bundled third-party `ext_lib/mdspan/include` is 4051
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
(build 21 s, run ~10 s, validate ~10 s).
HIP: extracted, untested (no hipcc/ROCm on this machine; `build.sh HIP` stops with a
clear error).
