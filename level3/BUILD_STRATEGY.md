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

Every Level 3 application owns, per backend/profile,
`.deps/level3/<app>/<profile>/{src,build,install,logs,cache}` (build-side copy of
the frozen sources where a build must be in-tree, private dependency builds,
install prefix, logs, JIT caches) plus the fingerprint
`.deps/level3/<app>/<profile>/install/.hpcperf-l3-fingerprint` written by
`level3/tools/l3_common.sh` (schema `l3-2`: upstream commit, backend/arch,
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
  docs, the CP2K toolchain) copy `src/` to a build-side tree under `.deps/level3/<app>/<profile>/src` first; the
  materialized `src/` stays byte-identical to the artifact (its hash is re-checked by `check_workspace.py`).
- dependency tarballs that upstream build systems would download (GEOS TPL superbuild, CP2K toolchain,
  ExaCA's nlohmann_json) are pre-seeded from `$HERE/deps` into the build tree; the upstream sha256 check
  then passes without a download.
- an agent workspace (`tools/create_agent_workspace.sh`) is a real copy of `level3/<app>` under its own root
  with its own `build/` and `.deps/level3/<app>/`; `build.sh` there compiles the workspace's current
  `src/` (Ninja incremental: only changed files are recompiled; the install fingerprint carries the
  configuration, not the source hash, so an edited file is always rebuilt, never served from a stale
  binary of another run).

## Level 3 policy, correctness policy, isolation, runtime and per-application layout

Moved here from `level3/README.md` on 2026-09-11, when that README was rewritten for first-time users. The
content is unchanged; it is contributor/maintainer material (policy, conventions and the record of how the
correctness criteria were chosen), not the user entry point. The user-facing description of source artifacts,
GPU selection, workspaces and evidence levels now lives in [README.md](README.md) and
[EXTERNAL_ARTIFACT_DESIGN.md](EXTERNAL_ARTIFACT_DESIGN.md).

### Hard requirements (summary of the Level 3 policy)

1. Full application workflow (mesher/solver/IO stages included where upstream
   has them); no hotspot-only or single-kernel runs.
2. `HPCPERF_GPUS=N|all` selects the GPU count; requested == launched. A rank
   count the application's decomposition cannot support is an error -- never a
   silent change of N, never silent GPU sharing, never a fallback to 1 GPU,
   never a failure reported as PASS.
3. Default policy is one MPI rank per GPU; if upstream officially recommends
   another model (threads per GPU, MPI+OpenMP, several GPUs per rank) the
   application follows upstream and its README says so. All five first-batch
   applications document one rank per GPU.
4. Every application defines smoke / strong / weak inputs (global size,
   per-rank size, memory estimate, process topology, expected runtime,
   validation quantity), the rank->GPU mapping and multi-node requirements.
5. 40/80-GPU shapes are `DRY-RUN / UNVALIDATED` until a real allocation
   exists; multi-node is BLOCKED/UNVERIFIED on this site; HIP is `untested`
   without an AMD GPU.
6. Toolchain follows the application's officially supported versions, not
   Level 1's pins; compatibility modifications are classified (A none,
   B build-system-only, C environment, D source-level compatibility) -- E
   algorithm/performance modifications are forbidden in bring-up.
7. The validated Level 2 dependency tree (`.deps/install`) is never modified.

### Correctness policy as applied

Exit code is never sufficient. Each `validate.sh` uses the application's own
mechanism and prints a single `... validation (N GPU, ...): PASS|FAIL` line:
LAMMPS thermo vs the shipped reference log (bit-identical here); SPARTA
statistical stats vs the shipped reference log with justified tolerances
(particle count exact, temperature 2 %, collision attempts 15 %); WarpX
upstream's analytic Langmuir-wave regression test (5e-2) and charge
conservation (1e-11) read from the plotfile, plus exact particle conservation;
SPECFEM3D reference seismograms through upstream's comparison script
(correlation, misfit, time shift); nekRS upstream's `--cimode` CI checks on the
analytic Ethier solution. For these five, no tolerance was loosened to obtain a
PASS and no precision or physics setting was changed. Second batch: CP2K
regtest tolerances + MD energy consistency, QMCPACK `check_scalars.py`, DFT-FE
upstream GPU reference, GEOS geos-ats metrics/restart baseline, Nyx official
`fcompare` tolerances -- with one recorded exception: the Nyx heat/cool
tolerance 5e-5 (upstream's value for that deck) was adopted after a first run at
the adiabatic 2e-10 had FAILED, and the `I_R` field of that deck is not accepted
by any tolerance (PENDING), see `nyx/README.md`.

### Dependency and backend/profile isolation

One frozen source tree per benchmark; ALL generated state belongs to exactly one backend/profile (no shared
Level 3 install root; nothing writable is shared between CUDA and HIP):

```
level3/<app>/{src,deps}                                       materialized frozen source artifact (the ONLY application
                                                              source input; backend-independent, never duplicated per backend)
.deps/level3/<app>/<profile>/{src,build,install,logs,cache}   build-side source copy (in-tree-writing builds only), dependency
                                                              builds, install prefix + fingerprint, logs, JIT caches
build/level3/<app>/<profile>/                                 application build tree (+ run directories of run.sh/validate.sh)
_upstream/level3/<Name>                                       freeze-time checkout (fetch.sh; input of the freeze only)
.artifacts/sha256/<archive_sha256>.tar.zst                    content-addressed local artifact cache (prepare_benchmark.sh)
workspaces/<run-id>/                                          per-run agent workspace (real copy; own build/ and .deps/)
```

A profile uniquely identifies a configuration whose binaries and installs are not interchangeable, and it
always names its backend: `cuda` / `hip` where only the accelerator backend differs (LAMMPS, SPARTA, WarpX,
SPECFEM3D, ExaCA); `<variant>.<backend>` for nekRS (`hypregpu.cuda`, `cpucoarse.cuda`; `cpucoarse.hip` is
defined but untested; `hypregpu.hip` does not exist and is refused); toolchain identities for the second
batch (`cuda132-gcc142-ompi5010`, `clang231-cuda132-offload`, `cuda132-gcc133-adiabatic|heatcool`).
`HPCPERF_<APP>_PROFILE` overrides the name but must still name the backend; a profile that names another
backend than the one requested is refused before any directory is created
(`l3_paths_profile <app> <profile> <backend>`). build.sh, run.sh and validate.sh derive the profile through
the same helper (`l3_backend_profile`), run.sh accepts only the profile's own fingerprint with the matching
backend (`l3_fingerprint_expect_backend`), and a pre-migration shared install (`.deps/level3/<app>/install`)
is reported and never read. Build scratch that must live outside the worktree (the CP2K toolchain copy, the
QMCPACK LLVM build tree; both break on a git worktree's `.git` file over NFS) follows the same rule through
`l3_local_scratch_dir`: `${TMPDIR:-/tmp}/hpcperf-l3-scratch/<component>/<hash of the workspace root>/<source
identity prefix>/<profile>` -- private to the worktree/workspace, the frozen source and the profile (explicit
overrides `HPCPERF_CP2K_TOOLCHAIN_SCRATCH`, `HPCPERF_LLVM_SCRATCH`); the earlier `/tmp/hpcperf-l3-b2-scratch/`
locations shared by name are legacy local state. **Backend separation applies to generated state, not to
source duplication.**
Migration record and per-application matrix: [PROFILE_ISOLATION.md](PROFILE_ISOLATION.md) (2026-09-15).

Installs carry `.hpcperf-l3-fingerprint` (schema `l3-2`: application, upstream
commit, dependency versions, compiler, Fortran compiler, CUDA/ROCm, GPU arch,
MPI, CMake/configure options, GPU-aware-MPI setting, patch list, site profile,
Spack lock hash, container image hash, build time). A recorded fingerprint
that differs from the requested configuration fails fast
(`level3/tools/l3_common.sh`).

Spack, when chosen, uses one environment per application and backend
(`level3/envs/<app>/{cuda,rocm}/spack.yaml` + `spack.lock`); containers, when
chosen, commit the `.def`, build script, image SHA256 and README -- never the
`.sif`. Neither is used by the first batch (see BUILD_STRATEGY.md for why).

### Runtime

Launches go through the common launcher (`HPCPERF_GPUS`, `HPCPERF_NODES`,
`HPCPERF_GPUS_PER_NODE`, `HPCPERF_CPUS_PER_RANK`, `HPCPERF_SCALE_MODE`,
`HPCPERF_SITE_PROFILE`, `HPCPERF_DRY_RUN=1`) with the per-rank GPU wrapper
(each rank sees one GPU; expected vs observed GPU audited). Level 3 refers to
it through `HPCPERF_RUNTIME_DIR` (default `level2/tools`); the plan to move
the shared tools to `tools/runtime/` without breaking Level 2 is in
[../tools/runtime/README.md](../tools/runtime/README.md).

Site/transport observations recorded in the READMEs (single node, `pml ob1 /
btl self,sm,smcuda`): GPU-aware MPI makes WarpX's 4-GPU step 3x slower
(0.081 vs 0.026 s/step) but LAMMPS 2.5x faster (2.38 vs 5.87 s); SPARTA is
indifferent. Defaults stay upstream's; this is a performance topic for a later
round, not a bring-up change. Open MPI's one-sided layer still selects
`osc ucx` on this node and aborts inside `uct_ib` with 4 ranks (nekRS uses
`MPI_Win_lock`); nekRS' `run.sh` sets `OMPI_MCA_osc=^ucx`, which is proposed
for the gmu-hopper site profile in the runtime commonization PR.

### Per-application layout

```
level3/<app>/
├── README.md        provenance, version/commit, license, LOC, build strategy, changes (A-D), execution model,
│                    inputs (smoke/strong/weak), validation, 1/2/4-GPU results, dry-runs, limitations
├── fetch.sh         shallow clone at the recorded tag/commit (no source trees committed)
├── build.sh         native build into .deps/level3/<app>/<profile>/ + build/level3/<app>/<profile>/, fingerprinted per profile; HIP branch present, untested
├── run.sh           HPCPERF_GPUS + HPCPERF_SCALE_MODE aware, launched via the common launcher
├── validate.sh      upstream correctness mechanism, PASS/FAIL line, exit code
└── patches/         compatibility patches (classified, documented; SPECFEM3D, nekRS)
```

Inputs are upstream's own decks referenced from the read-only checkout;
derived decks (size, steps, topology, diagnostics) are written into the build
tree at run time and documented per application, so no upstream input file is
modified and nothing large is committed.
