# Level 3 build strategy

Companion to `APPLICATION_AUDIT.md`. For each application the four build
routes were assessed against what upstream documents and what dgx003 provides
(RHEL 10, 4x B200, CUDA 13.2.78, conda GCC 13.3.0 / Open MPI 5.0.10
CUDA-aware / CMake 3.28.4, system gfortran 14.2.1, no ROCm, no
Apptainer/Singularity, lmod broken, Spack 1.0.0.dev0 checkout of 2025-05).
Vocabulary: `BUILD_RECOMMENDATION = NATIVE | SPACK | NATIVE+SPACK_DEPS |
APPTAINER | SPACK+APPTAINER | SITE_NATIVE | DEFER`.

## Decision matrix

| Application | Native (upstream build system) | Spack | Apptainer/Singularity | Site-native modules | BUILD_RECOMMENDATION |
|---|---|---|---|---|---|
| LAMMPS | feasible, documented (CMake presets, bundled Kokkos 4.6.2 with BLACKWELL100 + CUDA 13 back-ports); deps = MPI + CUDA; **built in 220 s** | package exists but local checkout has no `cuda_arch=100`; upstream recipe would force EXTERNAL_KOKKOS 4.7.1 = configuration upstream calls "untested"; upstream does not recommend Spack | no runtime; no official image (NGC image sm_90-max, 2023) | lmod broken | **NATIVE** |
| SPARTA | feasible, documented (`cmake -C presets/kokkos_cuda.cmake -DKokkos_ARCH_BLACKWELL100=ON`); deps = MPI + CUDA; **built in 579 s** | **no package** (Spack's `sparta` is a bioinformatics tool) | no runtime; no recipe | lmod broken | **NATIVE** |
| WarpX | feasible, documented superbuild; small graph (AMReX local checkout, PICSAR/openPMD off); CUDA vs HIP one switch; upstream itself builds with CUDA 13.2 | possible only with a fresh spack-packages (26.09 missing; arch via `^amrex cuda_arch=100` legacy path); graph balloons with `+openpmd +python` | no runtime; only Perlmutter Containerfiles (sm_80) | lmod broken | **NATIVE** |
| SPECFEM3D Cartesian | feasible, only documented route (autotools; bundled SCOTCH); needs two devel back-ports for CUDA 13 + make-time `GENCODE` for sm_100; Fortran via system gfortran + `OMPI_FC` | no package | none; tiny dependency graph, nothing to gain | lmod broken | **NATIVE** |
| nekRS | feasible, only documented route (CMake; all TPLs vendored); CUDA/HIP separable by OCCA options; two build variants (see below) | recipe stale (23.0, wrong option names); deps vendored -> nothing for Spack to provide | none official; JIT couples to host toolchain | lmod broken | **NATIVE** |
| CP2K | feasible (CMake + `install_cp2k_toolchain.sh`); ~15 packages for a GPU-DFT build; needs the DBCSR B200 sed upstream master applies; 3-6 h | officially recommended (`make_cp2k.sh`, `spack install cp2k+cuda`) **but `cp2k` and `dbcsr` recipes hard-reject `cuda_arch=100`**; 60-100 packages; local checkout too old | official `cp2k/cp2k` images stop at H100; multi-node needs host MPI; no runtime here | lmod broken | **NATIVE+SPACK_DEPS** |
| Nyx | feasible, only documented path (CMake superbuild or GNU make); can consume the WarpX AMReX 26.09 checkout (`AMREX_MINIMUM_VERSION 20.11`); SUNDIALS CUDA superbuild for HEATCOOL | no package | none | lmod broken | **NATIVE** |
| QMCPACK | feasible today only for `QMC_GPU=cuda` (partial GPU); the recommended `openmp;cuda` needs Clang with NVPTX offload (absent) + Boost (absent) + tested HDF5 | package's `+cuda` is inert for 4.x (no `QMC_GPU`), no offload variant -> use Spack only for `llvm+cuda`, `boost`, `hdf5@1.14` | CI dependency images only (CPU) | lmod broken | **NATIVE+SPACK_DEPS** |
| GEOS | feasible, documented (thirdPartyLibs superbuild + host-config); ~20 TPLs, 4-5 h; `ENABLE_TRILINOS=OFF` with hypre on device; develop can share Level 2's RAJA/CHAI/Umpire 2026.07.0 | no upstream package; uberenv recipe is LC-only and hits the `raja ^cuda@13:` conflict upstream calls blocking | CI TPL images (CUDA <= 12.9, in-container MPI) | lmod broken | **NATIVE** |
| DFT-FE | feasible, only documented route (per-machine `install_DFTFE` shell scripts + CMake); ~12 autotools/CMake deps, 3-4 h; CUDA vs HIP = rebuild of DFT-FE + ELPA only | `dftfe` recipe unusable (0.6); a dealii/elpa/libxc hybrid possible but local Spack too old and ELPA `cuda_arch=100` unverified | CPU Docker recipe only | lmod broken | **NATIVE** |

No candidate gets `DEFER`: every one has an officially supported native CUDA
path on this toolchain. `APPTAINER`/`SPACK+APPTAINER`/`SITE_NATIVE` are
unavailable on this node regardless of application.

nekRS has two verified variants (both NATIVE; select with
`HPCPERF_NEKRS_HYPRE_GPU`/`HPCPERF_NEKRS_VARIANT`; isolated src/build/install/JIT
cache; see `level3/nekrs/COMPATIBILITY.md`):
- `hypregpu` (`ENABLE_HYPRE_GPU=ON`, 3 patches) -- required for a GPU (DEVICE)
  HYPRE coarse solve; verified with cimode 3 (GPU coarse) at 1/4 GPU.
- `cpucoarse` (`ENABLE_HYPRE_GPU=OFF`, 0 patches, 113 s build) -- covers the
  current Ethier CPU-coarse workload; a DEVICE-coarse request is rejected by
  nekRS itself, not silently down-graded. Recommended default only once the
  workload's coarse-solver placement is fixed by review; the CPU-coarse option's
  scalability at 40/80 GPUs is UNVERIFIED.

## Spack policy (Level 3)

- Spack is used **only** where upstream documents it as a supported route and
  the recipe can express the target (`cuda_arch=100`, CUDA 13.2 external,
  conda Open MPI external). In this round that is nowhere; the two
  `NATIVE+SPACK_DEPS` candidates (CP2K, QMCPACK) will use it for CPU-side
  dependencies when they are brought up.
- When used, each application/backend gets its own environment
  `level3/envs/<app>/{cuda,rocm}/spack.yaml` with a committed `spack.lock`;
  the lock's SHA-256 is recorded in the application fingerprint
  (`spack_lock_sha256=` line, currently `none`). CUDA and ROCm environments are
  never merged (they cannot concretize together here anyway: no ROCm).
- The personal Spack checkout (`/projects/kzhou6/bcui2/env_software/spack`,
  1.0.0.dev0, 2025-05-06) is too old for every 2026 release and lacks
  Blackwell; a Level 3 Spack environment will need a fresh Spack >= 1.2 with
  `spack-packages` >= 2026-07 and the conda Open MPI/CUDA declared as externals.
  This is recorded as a prerequisite, not done in this round.

## Apptainer policy (Level 3)

- Not applicable on dgx003 (no runtime). If a future site provides Apptainer:
  only the `.def`, a README, the build script and the image SHA-256 are
  committed (never a `.sif`); the definition must pin the base image digest and
  the application commit; containers do not solve the host driver, the MPI
  transport or the interconnect, and multi-node runs still require the host
  MPI/PMIx -- so a container build is treated as one more `Build Strategy`
  variant with its own fingerprint, never as the default.

## Per-application dependency isolation

Every Level 3 application owns `.deps/level3/<app>/{src,build,install,logs}`
(private copy of patched sources where a build must be in-tree or patched,
private dependency builds, install prefix, logs) plus the fingerprint
`.deps/level3/<app>/install/.hpcperf-l3-fingerprint` written by
`level3/tools/l3_common.sh` (schema `l3-1`: upstream commit, backend/arch,
dependency versions, compiler, Fortran compiler, CUDA/ROCm, MPI, CMake options,
GPU-aware MPI setting, site profile, Spack lock SHA-256, container SHA-256,
patch list, build time). A fingerprint mismatch makes `build.sh` fail fast with
the differing lines. Nothing under `.deps/install/` (Level 2) is modified or
reused: the Level 2 Kokkos 5.2.1 / RAJA suite / hypre / AMReX pins are not the
versions these applications validate against (LAMMPS pins Kokkos 4.6.2, SPARTA
5.0.2, WarpX AMReX 26.09, nekRS hypre 2.32.0), and Level 2 must keep building.

## Modification classes used in the first batch

| Application | Class | What |
|---|---|---|
| LAMMPS | A | none; derived `in.lj` deck (`run ${steps}`, weak `processors`) written into the build tree |
| SPARTA | A | none |
| WarpX | A | none; derived inputs file (sizes, `warpx.numprocs`, reduced diagnostics) written into the build tree |
| SPECFEM3D | B + C + D | make-time `GENCODE` override (sm_100, devel's `cuda13` value) and bundled SCOTCH built without gzip support (generated `Makefile.inc`, no `zlib.h` in the conda sysroot); `OMPI_FC=/usr/bin/gfortran`, `MPI_INC`; two devel back-ports (CUDA 13 `deviceOverlap` guard 10 lines, Blackwell device block 8 lines) in `level3/specfem3d/patches/` |
| nekRS | B + C + D | B: `cmake/hypre.cmake` `HYPRE_CUDA_SM=80 90` -> `80 90 100` (1 line), CMake generator pinned to upstream's Unix Makefiles (the conda `CMAKE_GENERATOR=Ninja` yields an invalid rule for the vendored HYPRE install step); C: `OMPI_FC=/usr/bin/gfortran`, `LDFLAGS+=-fno-lto` (mixed GCC 13 / gfortran 14 LTO bytecode in CMake's Fortran/C detection), `FFLAGS+=-fPIC` (PIE default of the conda GCC), `unset AR` (HYPRE's configure takes `$AR` as the full archive command), run time `unset CMAKE_GENERATOR` (UDF build), `OMPI_MCA_osc=^ucx` (one-sided ops otherwise go through UCX and abort with 4 ranks), `ulimit -s unlimited` (as upstream's job scripts); D: vendored HYPRE 2.32.0 vs the Thrust 3.2 shipped with CUDA 13 -- `thrust::pair` result type -> `auto` (2 lines), explicit `<thrust/iterator/reverse_iterator.h>`/`<thrust/pair.h>` includes in `device_utils.h` and in the pre-generated concatenated `_hypre_utilities.hpp`, `thrust::not1` -> `thrust::not_fn` (16 lines); all in `level3/nekrs/patches/` |

No class E change anywhere; no numerics, physics or algorithm touched. The
nekRS list shows the general pattern for Fortran + CMake applications on this
node (CP2K and DFT-FE will meet the same OMPI_FC / LTO / PIE issues) and that
nekRS' vendored HYPRE 2.32.0 is not CUDA 13-ready as shipped.
