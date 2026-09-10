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
  `NATIVE+SPACK_DEPS` candidates (CP2K, QMCPACK) were **brought up without
  Spack in the second batch** (CP2K: upstream's own toolchain script is the
  documented dependency route and already pins every version; QMCPACK: the
  private LLVM/HDF5/Boost builds are 3 tarballs with recorded SHA-256s) -- so
  every `spack_lock_sha256=` stays `none` and no `spack.yaml` exists. Their
  matrix rows below are therefore realised as `NATIVE`.
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

## Second batch (2026-09-05/06): what was actually built and how

Per-profile isolation was added for this batch: `.deps/level3/<app>/<profile>/{src,build,install,logs,cache}`
and `build/level3/<app>/<profile>/` (`l3_paths_profile`), the profile naming the
compiler/Toolkit/backend/key-dependency combination (`cuda132-gcc142-ompi5010`,
`clang231-cuda132-offload`, `cuda132-gcc133-adiabatic|heatcool`, ...). Two
differently configured builds never share a source, build or install tree, and
the fingerprint (schema `l3-2`; the profile and every dependency version/SHA are
part of the `dependencies=` line) refuses a mismatching reuse. Source and
build trees that suffer on NFS (LLVM 23, the CP2K toolchain -- 150k+ small files)
live on local `/tmp/hpcperf-l3-b2-scratch/`; installs, logs and fingerprints stay
under `.deps/`.

| Application | Route actually used | Class | What |
|---|---|---|---|
| Nyx 26.09 | NATIVE; external **AMReX 26.09 built per profile** (the pinned submodule `6e875b7c` cannot emit sm_100), SUNDIALS 7.2.1 CUDA per profile for HEATCOOL; CPU-backend profile for cross-references | A | none in Nyx/AMReX/SUNDIALS; derived decks (step counts, fixed `amr.max_grid_size`/`refine_grid_layout=0`, checkpoint output, synthetic `RandomPerCell` sizes) written into the build tree, upstream decks otherwise verbatim |
| CP2K v2026.2 | NATIVE + upstream `install_cp2k_toolchain.sh` (OpenBLAS/ScaLAPACK/FFTW/libint/libxc/LIBXSMM/spglib/DBCSR); **no Spack** (recipes cap `cuda_arch` at 90) | B + C | B: back-port of upstream commit `378b2fab` (B200 in the toolchain: `--gpu-ver=B200 -> ARCH_NUM 100`, DBCSR `GPU_ARCH_NUMBER_B200 100` + `parameters_H100.json -> parameters_B200.json`) -- `level3/cp2k/patches/0001`; C: toolchain copy on local scratch (DBCSR's `GetGitRevisionDescription.cmake` aborts inside a git worktree), `PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe` for DBCSR's own `mpiexec -n 4` ctest launches, `unset -f grep` (the login shell exports a `grep` function), conda build variables cleared, BLAS/LAPACK given explicitly as the toolchain `libopenblas.a` (`CP2K_BLAS_VENDOR=CUSTOM`) with the toolchain library directories first in the RPATH of libcp2k.so/cp2k.psmp (`CMAKE_INSTALL_RPATH` + `-Wl,-rpath` linker flags; attempt 1 had inherited conda's `LDFLAGS` -- `-Wl,--disable-new-dtags -rpath <conda lib>` -- so `libopenblas.so.0` resolved to the conda pthreads OpenBLAS at run time, attempt 2 with `CMAKE_INSTALL_RPATH` alone still lost to the MPI wrapper's rpath; run.sh refuses to run unless `ldd` resolves BLAS under the toolchain and records `blas_resolved=`), `setup` still sourced at run time |
| QMCPACK v4.4.0 | NATIVE + **private toolchain** (LLVM 23.1.0 from source with the NVPTX offload runtime, HDF5 1.14.5 parallel, Boost 1.90 headers, OpenBLAS 0.3.30); no Spack (recipe inert for 4.x GPU options) | A (+ C) | none in QMCPACK; C: `-DCMAKE_IGNORE_PATH=/usr/lib64/cmake/ZLIB;/lib64/cmake/ZLIB` for HDF5 (the node's zlib-ng CMake package references an absent `libz.a`); derived decks change only the walker-population parameter (strong/weak) |
| DFT-FE 1.2.0 | NATIVE; recipe transcribed from `install_DFTFE` (`frontierDevelop`) with **system GCC 14.2.1 for C/C++/Fortran**; all 9 dependencies per profile; **deal.II 9.6.2** (attempt 1 with 9.7.1 -- the version the current recipe pairs with dftfe *develop* -- fails: 9.7 removed `Utilities::MPI::create_group`, `Triangulation::load(name, autopartition)`, `VtkFlags::ZlibCompressionLevel` that release 1.2.0 uses; 9.6.2 keeps them deprecated, so no API back-port) | C + D | D: `level3/dftfe/patches/0001-std-isnan.patch` -- two unqualified `isnan(` calls in a template of `src/atom/AtomicCenteredNonLocalOperator.cc` qualified as `std::isnan(` (GCC 14 rejects the unqualified spelling; same function, no numerical change); C: dftfe's `p4est-setup.sh` given `CC=mpicc CXX=mpicxx FC=mpifort F77=mpifort LIBS=-lm` (Cray wrappers hardcoded, implicit libm) and its zlib check pointed at p4est 2.8.7's `config/p4est_config.h`; conda build variables cleared; `TARGET=SAPPHIRERAPIDS` for OpenBLAS; ELPA configured with `-march=native` (its AVX-512 probe needs the SIMD flags in CFLAGS) and the ScaLAPACK/OpenBLAS paths in `LDFLAGS` (its cublas link check drops `SCALAPACK_LDFLAGS`); no deal.II/ELPA/p4est source change |
| GEOS develop `b7a0f133` + TPL `9b55672` | NATIVE: upstream `thirdPartyLibs` superbuild (`config-build.py -n`) + private host-config; TPLs per profile (not Level 2's RAJA/CHAI/Umpire installs: different compiler/CUDA flags); GEOS 1718 s at `-j32` | B + C + D | B (thirdPartyLibs, `level3/geos/patches/`): `0001` 65-character `SUPERLU_URL_HASH` typo, `0002` `RAJA_ENABLE_VECTORIZATION` overridable (built OFF: nvcc 13.2 + GCC 14/x86-64-v3 cannot compile RAJA's AVX2 tensor layer; upstream's own ROCm choice), `0003` hdf5 step gets `CMAKE_GENERATOR ${TPL_GENERATOR}` and `${TPL_BUILD_COMMAND}` like every other step (hardcoded `make` breaks under Ninja); D (BLT submodule, `geos-blt-0001-cuda13-memoryClockRate.patch`): back-port of LLNL/blt `38b46203` -- BLT's CUDA runtime smoke-test program reads `cudaDeviceProp::memoryClockRate`, removed in CUDA 13, now `cudaDeviceGetAttribute` for `CUDART_VERSION >= 13000`; test/benchmark switches left at upstream's default ON; C: configure-once guard (`config-build.py` deletes an existing build tree), stale-stamp cleanup; host-config `ENABLE_HYPREDRV=OFF` (GEOS-documented option; the TPL hypredrive step lacks Umpire's include path, and the beam workflow does not use hypredrive) and `-I<TPL metis/parmetis include>` ahead of the `-isystem` conda MPI include dir (which carries an unrelated 32-bit `metis.h` that broke GEOS' 64-bit ParMETIS assertion); no GEOS/LvArray/hypre source change |

Still no class E change; no numerics, physics, precision, solver placement or
input tolerance was touched. Every patch file carries its source, rationale,
conditions, impact and verification in its header and its hash is in the
profile fingerprint.

## Source input rules (scheme 3: external source artifacts, 2026-09-10)

Every Level 3 `build.sh` builds from the materialized frozen source artifact and nothing else
(see [EXTERNAL_ARTIFACT_DESIGN.md](EXTERNAL_ARTIFACT_DESIGN.md)):

- application source = `$HERE/src`, benchmark-specific source dependencies = `$HERE/deps`; both are
  placed by `tools/prepare_benchmark.sh level3 <app>` after size/sha256/tree-hash/safety verification of
  `<app>[-<variant>]-<source_version>.tar.zst`. A build never calls prepare (a DIRTY tree is never
  overwritten), never fetches, clones, downloads or patches source (patches are pre-applied at freeze
  time; their series and hashes are still recorded in `provenance/patch_series*.txt` / `source.lock*.yaml`
  and go into the install fingerprint). When `src/` is absent the build fails plainly with
  `source not materialized ... run tools/prepare_benchmark.sh level3 <app>` (`l3_require_materialized`).
- forbidden as source inputs: `_upstream/` (freeze-time checkout only, used by `fetch.sh` and the freeze),
  `.deps/<...>/src` of another benchmark, another repository or worktree, a maintainer's `/tmp` or home
  directory, a floating upstream branch. `tools/check_workspace.py` (checks 7 and 15) greps the scripts
  for such references.
- allowed environment-provided inputs (declared in `source.lock*.yaml: dependencies.environment_provided`):
  CUDA/ROCm toolkit and driver, compilers (conda GCC, system GCC/gfortran, the private LLVM offload
  toolchain for QMCPACK), MPI, Slurm, site UCX/libfabric, system runtime libraries, `.conda_env`, and --
  for a workspace created with `--link-prebuilt-deps` -- a stage-complete dependency install prefix of the
  canonical tree (never the application itself, which is always rebuilt in the workspace).
- builds that write into their source tree (SPECFEM3D autotools, nekRS, DFT-FE `git_info.h`, GEOS LvArray
  docs, the CP2K toolchain) copy `src/` to a build-side tree under `.deps/level3/<app>/` first; the
  materialized `src/` stays byte-identical to the artifact (its hash is re-checked by `check_workspace.py`).
- dependency tarballs that upstream build systems would download (GEOS TPL superbuild, CP2K toolchain,
  ExaCA's nlohmann_json) are pre-seeded from `$HERE/deps` into the build tree; the upstream sha256 check
  then passes without a download.
- an agent workspace (`tools/create_agent_workspace.sh`) is a real copy of `level3/<app>` under its own root
  with its own `build/` and `.deps/level3/<app>/`; `build.sh` there compiles the workspace's current
  `src/` (Ninja incremental: only changed files are recompiled; the install fingerprint carries the
  configuration, not the source hash, so an edited file is always rebuilt, never served from a stale
  binary of another run).
