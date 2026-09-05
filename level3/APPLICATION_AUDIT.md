# Level 3 application audit

Audit date 2026-09-04, node dgx003 (GMU Hopper: RHEL 10, 4x NVIDIA B200 in
one Slurm allocation, CUDA 13.2.78, driver 595.58.03, conda GCC 13.3.0 + Open
MPI 5.0.10 (CUDA-aware) + CMake 3.28.4; system gfortran 14.2.1 only; no ROCm,
no Apptainer/Singularity, lmod broken; personal Spack checkout 1.0.0.dev0 of
2025-05-06). Every statement below is either "upstream documents ..." (from the
official repository/docs at the recorded commit) or explicitly marked as
verified on this node. Nothing here claims multi-node, 8/40/80-GPU or HIP
validation. Upstream sources were cloned read-only under `_upstream/level3/`
(gitignored); LOC = cloc 2.06 code lines.

Status legend: **FIRST_BATCH** = brought up in this round (LAMMPS, SPARTA,
WarpX, SPECFEM3D Cartesian, nekRS, in the order requested); **SECOND_BATCH** =
feasible natively on this node but excluded from this round by the
dependency-time rule (> 2 h of dependency builds), a toolchain gap, or a
dataset blocker; **DEFER** / **REPLACE_CANDIDATE**: none needed -- all ten
candidates have an officially supported native CUDA path.

| Application | Version audited | GPU model | Deps complexity | B200+CUDA 13.2 risk | HIP/MI355X risk | Spack | Container | Priority |
|---|---|---|---|---|---|---|---|---|
| LAMMPS | stable_22Jul2025_update6 | Kokkos 4.6.2 (bundled) | low | low-medium | high (no gfx950 in bundled Kokkos) | pkg exists, local checkout too old, external Kokkos "untested" | none official | FIRST_BATCH |
| SPARTA | 27Aug2026 | Kokkos 5.0.2 (bundled) | low | low-medium | medium-high | **no package** (name collision) | none | FIRST_BATCH |
| WarpX | 26.09 (+AMReX 26.09) | AMReX | medium | low-medium | medium | pkg to 26.08 | site recipe only | FIRST_BATCH |
| SPECFEM3D Cartesian | v4.1.1 | native CUDA/HIP | low | high as tagged (2 backports needed) | high | none | none | FIRST_BATCH |
| nekRS | v26.0 | OCCA (JIT) + HYPRE | medium (all vendored) | medium | medium | pkg stale (23.0) | none | FIRST_BATCH |
| CP2K | v2026.2 | native CUDA/HIP + DBCSR | very high (~40 pkgs) | medium-high | high | official but `cuda_arch=100` rejected | official, no B200 image | SECOND_BATCH |
| Nyx | 26.09 | AMReX | low-medium | medium (SUNDIALS) | medium-high | none | none | SECOND_BATCH |
| QMCPACK | v4.4.0 | OpenMP offload + cuBLAS | medium (needs Clang offload) | medium-high | medium | pkg `+cuda` broken for 4.x | CI only | SECOND_BATCH |
| GEOS | 1.2.0 (develop differs) | RAJA/CHAI/Umpire + hypre | very high (~20 TPLs) | high (tag) / medium (develop) | high | uberenv only (LC systems) | CI images | SECOND_BATCH |
| DFT-FE | 1.2.0 | native CUDA/HIP/SYCL + deal.II (CPU) | high | medium | medium-high | pkg stale (0.6) | CPU only | SECOND_BATCH |

---

## LAMMPS

- official_repository: https://github.com/lammps/lammps (not migrated)
- official_documentation: https://docs.lammps.org/stable/ (Kokkos: `Speed_kokkos.html`; build: `Build_extras.html#kokkos`; run switches: `Run_options.html`); `doc/src/` in the clone is authoritative for this tag
- latest_stable_release: `stable_22Jul2025_update6` (2026-09-03). Policy (`doc/src/Manual_version.rst`): feature releases `patch_<date>` every 4-8 weeks (latest `patch_2Sep2026`, bundles Kokkos 5.0.2, needs CUDA >= 12.2); one stable per year plus `_updateN` bug-fix updates (back-ports only). Update 6 notes list KOKKOS fixes.
- selected_commit_sha: `9c5ab448c78a14fd534619622162ba418d6a1fb1`
- license: GPL-2.0
- application_owned_loc: `src/` 852,527 (3,865 files; `src/KOKKOS` 108,766 in 401 files). Bundled TPLs counted separately: `lib/kokkos` 217,774; other `lib/*` 276,282.
- main_languages: C++ 79 %, headers 20 %, shell/Cython/CMake
- build_system: CMake >= 3.16 (`cmake/CMakeLists.txt`), presets in `cmake/presets/`; GNU make also supported but Kokkos+CUDA is documented via CMake
- cxx_standard: C++11 core, C++17 forced with `PKG_KOKKOS`
- supported_compilers: GCC/Clang/Intel/NVHPC; bundled Kokkos 4.6.2 enforces GCC >= 8.2, nvcc >= 11.0, hipcc >= 5.2 (`lib/kokkos/cmake/kokkos_compiler_id.cmake`)
- cuda_support: yes (Kokkos CUDA backend, `nvcc_wrapper`, `FFT_KOKKOS=CUFFT`). Documented CUDA >= 11.0. Blackwell: `Kokkos_ARCH_BLACKWELL100/120` present in the bundled Kokkos. CUDA 13: the bundled 4.6.2 carries LAMMPS-applied back-ports (`#if CUDART_VERSION >= 13000` in `Kokkos_Cuda_Instance.hpp`) absent from upstream 4.6.02; no removed `cudaDeviceProp` fields used.
- hip_rocm_support: yes (`kokkos-hip.cmake`, ROCm >= 5.2, `FFT_KOKKOS=HIPFFT`); arch table ends at `AMD_GFX942`; **no gfx950 in bundled Kokkos 4.6.2**
- mpi_support: yes (spatial decomposition; `BUILD_MPI`)
- official_gpu_programming_model: Kokkos (bundled 4.6.2; external must be `>= 4.6.02`, upstream calls external Kokkos "untested"; source-incompatible with Kokkos 5 before 10Dec2025)
- multi_gpu_support: one MPI rank per GPU documented ("-np ... equal to the number of physical GPUs on the node"); several ranks per GPU need MPS; one rank never drives several GPUs
- multi_node_support: yes (`mpirun -np 32 -ppn 2 ... -k on g 2`); needs a launcher exposing a local-rank variable and GPU-aware MPI for device buffers
- gpu_aware_mpi_requirement: optional, default on; auto-detected for Open MPI via `MPIX_Query_cuda_support`, otherwise warned; `-pk kokkos gpu/aware off` falls back to host staging
- rank_to_gpu_binding: self-binding (`-k on g Ng`: device = local rank % Ng from `OMPI_COMM_WORLD_LOCAL_RANK`, `SLURM_LOCALID`, ...); with the Level 3 launcher wrapper each rank sees one GPU (`g 1`)
- topology_decomposition_controls: `processors Px Py Pz` (optional), `comm_style`, `balance`; bench decks size via `-var x y z` (32,000 atoms x x*y*z)
- major_dependencies: MPI, CUDA + cuFFT, bundled Kokkos; optional FFTW/MKL, JPEG/PNG (headers absent here -> disabled)
- dependency_complexity: low
- official_inputs_datasets: `bench/in.{lj,eam,chain,chute,rhodo}` + `.scaled` variants, data files < 7 MB, reference logs `bench/log.15Jul25.*.g++.{1,4}`; `examples/` (~800 inputs), `unittest/`, `tools/regression-tests/` (config_kokkos.yaml: `-k on g 2 -sf kk -pk kokkos newton on neigh half`, tol abs 1e-4 / rel 1e-6)
- correctness_mechanism: thermo-output comparison with shipped reference logs within tolerances (regression tooling), force-style YAML unit tests
- strong_scaling_input_availability: yes (`bench/README`: fixed-size problems; 32k atoms far too small for B200 -> replicate with `-var x y z`)
- weak_scaling_input_availability: yes (`-var x Px -var y Py -var z Pz`)
- expected_build_time: **verified 220 s** at -j32 on dgx003 (audit estimate was 30-60 min)
- expected_disk_usage: source 580 MB; build ~3 GB; install ~0.2 GB (verified order of magnitude)
- expected_input_data_size: ~37 MB (`bench/`), no downloads
- b200_cuda132_risk: low-medium (arch flag and CUDA 13 back-ports present; not upstream-validated for CUDA 13.x) -- **verified working on this node**
- mi355x_hip_risk: high for this tag (no gfx950; needs feature release with Kokkos >= 5.1 or unsupported external Kokkos)
- container_availability: no official application image (`tools/singularity/*.def` are build-environment recipes; NGC image is 2023, sm_90 max)
- spack_availability: `lammps` package exists (upstream `20250722.4`, would force EXTERNAL_KOKKOS 4.7.1 for cuda@13); local checkout lacks `cuda_arch=100`; upstream does not recommend Spack
- recommended_integration_priority: FIRST_BATCH
- blocker: none. Watch: long nvcc compiles of templated pair styles (not observed: 220 s); bundled Kokkos 4.6.2 + CUDA 13.2 not upstream-validated
- build_strategy_notes: see BUILD_STRATEGY.md -- **NATIVE**

## SPARTA

- official_repository: https://github.com/sparta/sparta (docs https://sparta.github.io/doc/Manual.html; Kokkos `Section_accelerate.html`)
- official_documentation: as above + `BUILD_CMAKE.md`, `bench/README`, https://sparta.github.io/bench.html
- latest_stable_release: `27Aug2026` (2026-08-28); single dated-tag stream. Notes: Kokkos 5.0.2, KOKKOS builds CMake-only and C++20
- selected_commit_sha: `95b9abaa8bd548991cc3c3f1c58b34722f7ade74`
- license: GPL-2.0
- application_owned_loc: `src/` 131,181 (`src/KOKKOS` 36,909 / 194 files); bundled `lib/kokkos` 223,495 separately
- main_languages: C++ 80 %, headers 19 %
- build_system: CMake >= 3.16 (`sparta/cmake`, presets `cmake/presets/kokkos_{common,cuda,hip}.cmake`)
- cxx_standard: C++11 core, C++20 with KOKKOS
- supported_compilers: GNU default, Intel documented; Kokkos 5.0.2 enforces GCC >= 10.4, nvcc >= 12.2, ROCm >= 6.2
- cuda_support: yes; docs explicitly list "GB200 (Blackwell) -> -DKokkos_ARCH_BLACKWELL100=ON" (override the preset's HOPPER90). Kokkos 5.0.2 has native CUDA 13 support (4.7.01 fix, 5.0.2 CUDA 13.1 mdspan fix)
- hip_rocm_support: yes (`kokkos_hip.cmake`, `elcapitan_kokkos.cmake` gfx942); **no gfx950** in bundled Kokkos 5.0.2
- mpi_support: yes (grid-cell/particle decomposition)
- official_gpu_programming_model: Kokkos (bundled 5.0.2; `USE_EXTERNAL_KOKKOS` without version pin)
- multi_gpu_support: one rank per GPU documented; several ranks per GPU with MPS "recommended"
- multi_node_support: yes (`mpirun -np 32 -ppn 2 spa_kokkos_cuda -k on g 2`)
- gpu_aware_mpi_requirement: optional, default `gpu/aware yes`; **no runtime auto-detection** (`kokkos.cpp` sets the flag unconditionally) -> must be set `no` with a non-CUDA-aware MPI
- rank_to_gpu_binding: self-binding as LAMMPS (`-k on g Ng`, local rank env vars)
- topology_decomposition_controls: `create_grid ... block Px Py Pz|clump|stride|random`, `balance_grid rcb part|cell` (any rank count), `fix balance`; bench size `-var x y z` (particles = 10 x cells)
- major_dependencies: MPI, CUDA (+cuFFT only with PKG_FFT), bundled Kokkos
- dependency_complexity: low
- official_inputs_datasets: `bench/in.{free,collide,sphere}` (+ `ar.species`, `ar.vss`, `data.sphere`), reference logs `bench/log.7Jul14.*.icc.{10K,100K,1M,10M}.{1,8}` (2014 CPU), `examples/` (44 problems with 2023-26 logs)
- correctness_mechanism: statistical log comparison (`tools/testing/regression.py`, tolerances; `examples/README`: "statistically similar answers ... not identical"); invariants: particle count conserved, temperature ~273 K
- strong_scaling_input_availability: yes (fixed `-var x y z`, e.g. 100^3 cells = 10M particles)
- weak_scaling_input_availability: yes (website uses 1M and 16M particles/node)
- expected_build_time: **verified 579 s** at -j32 (estimate 10-20 min)
- expected_disk_usage: source 100 MB; build ~1.5 GB; executable 302 MB (static Kokkos)
- expected_input_data_size: 0.2 MB (`bench/`)
- b200_cuda132_risk: low-medium -- **verified working**
- mi355x_hip_risk: medium-high (no gfx950 in bundled Kokkos; external Kokkos >= 5.1 possible since no pin)
- container_availability: none
- spack_availability: **none** -- the Spack `sparta` package is the unrelated sPARTA bioinformatics tool
- recommended_integration_priority: FIRST_BATCH
- blocker: none. Watch: gpu/aware default with no detection; 2014 reference logs (1 and 8 ranks only)
- build_strategy_notes: **NATIVE**

## WarpX

- official_repository: **https://github.com/BLAST-WarpX/warpx** (ECP-WarpX/WarpX redirects, HTTP 301 verified)
- official_documentation: https://warpx.readthedocs.io/ (install/cmake, install/hpc + 21 machine pages, usage/parameters, usage/workflows/domain_decomposition, developers/how_to_test)
- latest_stable_release: 26.09 (2026-09-03), monthly YY.MM tags
- selected_commit_sha: `0c62c75e53a9ad08241535444bd7e53fd1deba88`; pinned AMReX 26.09 `a52ca73324ac2c7b65ec04f131e6df99eec9c576` (`dependencies.json`)
- license: BSD-3-Clause-LBNL
- application_owned_loc: `Source/` 112,459 (C++ 69,266 / 247 files; headers 37,146); no bundled TPL source (AMReX, pyAMReX, PICSAR-QED, openPMD-api, pybind11 fetched at configure time); AMReX `Src/` 273,313
- main_languages: C++ (~all of `Source/`), Python (PICMI/tests), CMake
- build_system: CMake >= 3.25 (superbuild; `-DWarpX_amrex_src=<path>` for a local AMReX)
- cxx_standard: C++20
- supported_compilers: GCC 12+, Clang 14+, NVCC 12.4+ (docs); Perlmutter profile: gcc-native/13.2 with **NVCC 13.2.78**; Containerfile `nvidia/cuda:13.2.1-devel`
- cuda_support: yes (`WarpX_COMPUTE=CUDA`, `CMAKE_CUDA_ARCHITECTURES=100`); AMReX >= 25.10 has the CUDA 13 fix; AMReX docs translate legacy "Blackwell" to 100 and 120; GPU CI is H100 (sm_90) only
- hip_rocm_support: yes (`WarpX_COMPUTE=HIP`, `AMReX_AMD_ARCH`, ROCm 6.0+; Frontier/LUMI gfx90a, Tuolumne gfx942); no gfx950
- mpi_support: yes (default ON, MPI 3.0+, `WarpX_MPI_THREAD_MULTIPLE`)
- official_gpu_programming_model: AMReX (`ParallelFor`) via ablastr
- multi_gpu_support: one rank per GPU (AMReX: "MPI ranks == Number of GPUs")
- multi_node_support: yes; docs recommend GPU-aware MPI
- gpu_aware_mpi_requirement: optional; AMReX auto-detects (`MPIX_Query_cuda_support`), `amrex.use_gpu_aware_mpi=0/1`
- rank_to_gpu_binding: self-binding (rank-in-node when ranks/node == visible GPUs; otherwise `rank % nGPU` with a warning); Perlmutter script pins via `CUDA_VISIBLE_DEVICES`
- topology_decomposition_controls: `warpx.numprocs nx ny nz` (product == ranks, one box per rank), or `amr.n_cell`/`amr.max_grid_size`/`amr.blocking_factor` (n_cell and max_grid_size divisible by blocking_factor), `algo.load_balance_*`
- major_dependencies: AMReX 26.09, PICSAR-QED 26.05 (default `WarpX_QED=ON`), openPMD-api 0.17.1 (default ON, needs HDF5/ADIOS2 for useful backends), pybind11 (Python only); cuFFT from toolkit
- dependency_complexity: medium (minimal CUDA build is self-contained)
- official_inputs_datasets: 407 `inputs*` under `Examples/` (12 Physics_applications incl. `uniform_plasma`, `laser_acceleration`); `Regression/Checksum/benchmarks_json/` (381 files)
- correctness_mechanism: per-test checksums (sum |Q| per field/particle attribute, rtol 1e-9) + Python analysis scripts via ctest; upstream warns checksums are **architecture-dependent** ("may differ on your computer architecture"; repo CLAUDE.md: "ignore checksum failures, since they can be platform-dependent")
- strong_scaling_input_availability: none labelled; `uniform_plasma` is "commonly used to study performance" but shipped at 64x32x32 x 10 steps
- weak_scaling_input_availability: none shipped; periodic uniform plasma scales trivially with `amr.n_cell` + `warpx.numprocs`
- expected_build_time: 25-45 min at -j32 (estimate; measured value recorded in `level3/warpx/README.md`)
- expected_disk_usage: source 24 MB + AMReX 35 MB; build 2-4 GB
- expected_input_data_size: KB-scale inputs
- b200_cuda132_risk: low-medium (upstream already on CUDA 13.2 at Perlmutter; sm_100 untested upstream)
- mi355x_hip_risk: medium
- container_availability: only Perlmutter-specific Containerfiles (sm_80); no published images
- spack_availability: `warpx` package to 26.08 (not a CudaPackage: arch via `^amrex cuda_arch=100`); local checkout too old
- recommended_integration_priority: FIRST_BATCH
- blocker: none hard. Watch: configure-time GitHub fetches (avoided with `WarpX_amrex_src`, `WarpX_QED=OFF`, `WarpX_OPENPMD=OFF`)
- build_strategy_notes: **NATIVE**

## SPECFEM3D Cartesian

- official_repository: https://github.com/SPECFEM/specfem3d (moved from geodynamics/specfem3d). Default branch `devel` (HEAD `cc2e9ffa7e7cb5338e05f5a7df81cfbe60e00683`, 2026-07-24) is ~2.5 years ahead of the last release
- official_documentation: https://specfem3d.readthedocs.io/ ; `doc/USER_MANUAL/manual_SPECFEM3D_Cartesian.pdf`; wiki
- latest_stable_release: v4.1.1 (2024-03-15, bug-fix release)
- selected_commit_sha: `c67d3ae7d4bfc5ac75cb9e5601d93afa262d3d8d`
- license: GPL-3.0
- application_owned_loc: `src/` 142,398 (Fortran 90 119,057; CUDA 12,210; `src/gpu` 15,058 in 63 `.cu`); `utils/` 163,847 and `external_libs/` 137,463 (SCOTCH 5.1.12b, PaToH, METIS) separate
- main_languages: Fortran 90 (~84 %), CUDA C (~9 %), C
- build_system: GNU autotools (pre-generated `configure`; regenerating needs `autoreconf` + the empty `m4/` submodule -- neither available here); in-tree `make all`
- cxx_standard: n/a; Fortran `-std=f2008 -pedantic-errors -ffpe-trap=...` under gfortran (`flags.guess`)
- supported_compilers: gfortran (default), Intel, NVHPC, IBM; CI is CPU-only (gfortran/ifort)
- cuda_support: yes, native CUDA. `--with-cuda=cudaN` selects the *architecture generation*: v4.1.1 ends at `cuda12` = sm_90 + `GPU_DEVICE_Hopper`; devel added `cuda13` = sm_100 + `GPU_DEVICE_Blackwell` (2026-02-14). **v4.1.1 does not compile against CUDA 13**: `src/gpu/initialize_gpu.cu` reads `cudaDeviceProp.deviceOverlap`, removed in CUDA 13 (devel guards it). Legacy `cudaThread*` calls are behind `CUDA_VERSION < 4000`; texture references only under `USE_TEXTURES_*` (commented out). nvcc host compiler = first `gcc` on PATH
- hip_rocm_support: yes (`--with-hip=MI8..MI250` -> gfx803..gfx90a in v4.1.1; devel adds MI300/MI350 = gfx942/gfx950)
- mpi_support: yes (`use mpi`; needs a Fortran MPI module built by a compatible gfortran + MPI-IO)
- official_gpu_programming_model: native CUDA kernels (`src/gpu/kernels/*.cu`), same source built as HIP
- multi_gpu_support: one MPI process per GPU, device = `myrank % device_count` (global rank), or compile-time `-DGPU_DEVICE_ID`
- multi_node_support: yes (databases must be reachable by all ranks)
- gpu_aware_mpi_requirement: not used in v4.1.1 (halo exchange staged through host buffers); devel adds optional `--enable-cuda-aware-mpi`
- rank_to_gpu_binding: self-binding by global-rank modulo; no wrapper needed for ranks == GPUs on one node
- topology_decomposition_controls: `Par_file` `NPROC` (= mesh slices, fixed at mesh time), `PARTITIONING_TYPE` (SCOTCH/METIS/PaToH/rows), `GPU_MODE`, `NSTEP`, `DT`; in-house mesher `Mesh_Par_file`: `NEX_XI/NEX_ETA` multiples of `NPROC_XI/NPROC_ETA` (`NPROC_XI*NPROC_ETA = NPROC`), regions/layers
- major_dependencies: Fortran + C compilers, MPI, CUDA or ROCm; bundled SCOTCH (needs flex/bison, present); optional ADIOS2/HDF5/ASDF
- dependency_complexity: low
- official_inputs_datasets: `EXAMPLES/` 496 MB: `applications/homogeneous_halfspace` (36x36x16 = 20,736 HEX8, CUBIT mesh 3.2 MB + `meshfem3D_files/`, NPROC 4, NSTEP 5000), layered_halfspace, `meshfem3D_examples/*`, Mount_StHelens, CPML, fault (tpv5/tpv102), ...; 35 `REF_SEIS/` reference-seismogram sets; `benchmarks/` (analytic elastic solution, attenuation); `tests/` unit/compile checks
- correctness_mechanism: reference seismograms compared with `utils/scripts/compare_seismogram_correlations.py` (correlation >= 0.8, normalised L2 misfit <= 1 %, time shift <= 0.01 s); analytic benchmark; GPU is single precision -> tolerance-based by design
- strong_scaling_input_availability: not labelled; any fixed mesh with varying NPROC (re-decompose per rank count); shipped meshes are ~20k elements -> larger fixed meshes via `xmeshfem3D`
- weak_scaling_input_availability: yes via `xmeshfem3D` (`NEX_XI/NEX_ETA` with `NPROC_XI x NPROC_ETA`)
- expected_build_time: 5-10 min at -j32 (estimate; measured value in `level3/specfem3d/README.md`)
- expected_disk_usage: source 997 MB (EXAMPLES 496 MB, `.git` 289 MB); build 0.2-0.4 GB; databases tens of MB (20k elements) to GBs
- expected_input_data_size: shipped, < 0.5 GB total; per case < 40 MB
- b200_cuda132_risk: **high as tagged**, medium with two devel back-ports (CUDA 13 `deviceOverlap` guard; Blackwell block) + make-time `GENCODE` override to sm_100; mixed toolchain (conda GCC 13.3 for C/nvcc, system gfortran 14.2.1 for Fortran, conda Open MPI with `OMPI_FC`) -- the Fortran/MPI mix was verified with a 2-rank MPI Fortran program on this node
- mi355x_hip_risk: high for v4.1.1 (no gfx942/gfx950), medium on devel
- container_availability: none
- spack_availability: none (`specfem3d-globe` exists, not Cartesian)
- recommended_integration_priority: FIRST_BATCH (requested order); the audit alone would have said SECOND_BATCH because the tag needs source back-ports
- blocker: (1) v4.1.1 + CUDA 13 compile error -> class D back-port (10 lines, upstream devel provenance); (2) conda `mpif90` unusable without `OMPI_FC=/usr/bin/gfortran` (class C); (3) `NPROC` fixed per decomposition (cheap re-run per rank count); (4) no `autoreconf`
- build_strategy_notes: **NATIVE**

## nekRS

- official_repository: https://github.com/Nek5000/nekRS (`master` = latest stable release; HPC scripts in Nek5000/nekRS_HPCsupport)
- official_documentation: https://nekrs.readthedocs.io/ ; in-repo `doc/envHelp.txt`, `doc/parHelp.txt`, `RELEASE.md`, `examples/README.md`
- latest_stable_release: v26.0 (2026-01-27); previous v23.0 (2023-05)
- selected_commit_sha: `96b3cf9e5bacede16568826c04a21bc0fe50dc7d`
- license: BSD-3-Clause
- application_owned_loc: `src/` 53,131 (C++ 35,784; headers 12,322; C 3,782; Fortran 77 1,104); `examples/` 4,983; vendored `3rd_party/` ~2.27 M (lapack 829,795; adios 618,827; hypre 445,492; cvode 169,316; occa 89,888; nek5000 87,143; gslib 13,038; parRSB 6,450)
- main_languages: C++ (~67 %), C, OKL (OCCA kernel language, JIT), Fortran 77 (Nek5000 interface, case `.usr`)
- build_system: CMake >= 3.21 (`build.sh` wraps it with interactive prompts and `-j8`; call cmake directly)
- cxx_standard: C++17, C99
- supported_compilers: GNU >= 9.1 (fatal below), IntelLLVM, Clang, NVHPC; Fortran GNU/IntelLLVM/NVHPC/Flang for the Nek5000 part; CI: ubuntu, MPICH, gfortran, serial backend
- cuda_support: yes -- OCCA CUDA backend (`OCCA_ENABLE_CUDA`, auto-detected toolkit) + HYPRE on GPU (`ENABLE_HYPRE_GPU`, `find_package(CUDAToolkit 12.0)`). OKL kernels are JIT-compiled at run time with `-arch=sm_<device>` (sm_100 on B200 automatically). `cmake/hypre.cmake` has an explicit CUDA >= 13 branch but hard-codes `HYPRE_CUDA_SM=80 90` (SASS only, no PTX) -> **hypre device kernels would have no Blackwell code without a 1-line change**. Bundled hypre 2.32.0 carries `CUDA_VERSION >= 13000` shims. All 47 CUDA driver-API symbols used by OCCA exist in CUDA 13.2 (checked)
- hip_rocm_support: yes (OCCA HIP, hypre HIP; `--offload-arch` derived from the device at run time); MI250X documented; no gfx950 statement
- mpi_support: yes, required (MPI-3.1; parRSB partitioning, gslib/oogs gather-scatter)
- official_gpu_programming_model: OCCA (vendored development snapshot, `OCCA_VERSION_STR 2.0.0`) + HYPRE
- multi_gpu_support: one rank per GPU; default `--device-id LOCAL-RANK` (node-local rank via `MPI_Comm_split_type`)
- multi_node_support: yes; JIT cache handling (`NEKRS_CACHE_DIR`, `NEKRS_CACHE_LOCAL/BCAST`)
- gpu_aware_mpi_requirement: optional; `NEKRS_GPU_MPI` default OFF (RELEASE.md: enabling "may cause a performance regression"); env `NEKRS_GPU_MPI=1`
- rank_to_gpu_binding: self-binding (`device_id = local rank`); with the Level 3 wrapper (one visible GPU per rank) `--device-id 0` must be passed
- topology_decomposition_controls: none for the process grid (graph partitioning); size = elements in `.re2` x `polynomialOrder`; `ethierRefine.par` `hrefine = N` (uniform h-refinement); `numSteps`, `dt`
- major_dependencies: all vendored (OCCA, HYPRE 2.32.0 built twice host/device, gslib, Nek5000 + parRSB, ADIOS2 2.10.1, CVODE 6.5 off, reference LAPACK in Fortran); external: MPI with Fortran bindings, CMake, OpenMP, CUDA >= 12; **run time needs g++ + nvcc + gfortran** (JIT and `.usr` compilation)
- dependency_complexity: medium (nothing to fetch, but heavy vendored builds and a hard Fortran requirement)
- official_inputs_datasets: `examples/` (59 MB, 20 cases; elements: ethier 32, channel 64, periodicHill 864, gabls1/hit/kershaw 8000, turbPipe 7920, tcf 17,280, tgv 46,656, pb146 pebble bed); ctest harness `examples/CMakeLists.txt` (`--cimode`); CI runs ethier (13 modes), ethierRefine (5), lowMach, mv_cyl, conj_ht, channel, ... with 2 CPU ranks
- correctness_mechanism: ethier = Ethier-Steinman exact Navier-Stokes solution; `ethier.usr` computes L2 errors of velocity/pressure/scalars vs exact; `ci.inc` asserts them (reference values, EPS 0.3) plus iteration counts per `--cimode`; prints "CI test <...> passed|failed", exit code
- strong_scaling_input_availability: not labelled; kershaw (8000 el.), turbPipe, tcf, tgv, pb146 usable; upstream perf numbers at E/GPU = 8000
- weak_scaling_input_availability: partial: kershaw needs `genbox` (not shipped); `ethierRefine.par` `hrefine` (x8 elements per level) is built in
- expected_build_time: 25-45 min at -j32 (estimate; measured in `level3/nekrs/README.md`); first run of each case adds minutes of JIT
- expected_disk_usage: source 279 MB; build 3-5 GB; install 0.5-1 GB; JIT cache 10s-100s MB
- expected_input_data_size: < 60 MB
- b200_cuda132_risk: medium (hypre SM list; unpinned OCCA snapshot with no CUDA 13 statement; heavy OKL kernels JIT-compiled at `-O3 --use_fast_math` for sm_100)
- mi355x_hip_risk: medium (JIT arch automatic; hypre HIP arch list unaudited)
- container_availability: none official; JIT couples to host toolchain anyway
- spack_availability: `nekrs` package stale (23.0, 21.0; option names do not match v26.0)
- recommended_integration_priority: FIRST_BATCH
- blocker: (1) conda `mpif90` needs `OMPI_FC=/usr/bin/gfortran` (class C, verified); (2) hypre SM list (class B, 1 line); (3) `build.sh` interactive -> cmake called directly; (4) `genbox` missing for kershaw weak scaling. **Found during bring-up (not visible in the audit):** (5) the vendored HYPRE 2.32.0 does not compile against the Thrust 3.2 shipped with CUDA 13 (`thrust::pair` result types, non-transitive `reverse_iterator`/`pair` headers, removed `thrust::not1`) -- ~20 mechanical class-D lines; (6) with conda GCC 13 for C/C++ and system gfortran 14, CMake's FortranCInterface detection fails on LTO bytecode versions (`-fno-lto` at link) and on PIE (`-fPIC` for Fortran); (7) HYPRE's configure takes the conda `AR` variable as the full archive command (`unset AR`); (8) the conda `CMAKE_GENERATOR=Ninja` produces an invalid rule for the HYPRE ExternalProject and breaks the run-time UDF build (upstream's Makefiles generator pinned / env unset at run time); (9) Open MPI's `osc ucx` is selected for nekRS' `MPI_Win_lock` calls and aborts in `uct_ib` with 4 ranks (`OMPI_MCA_osc=^ucx`); (10) the default 8 MB stack limit segfaults the h-refined cases in `useric` (`ulimit -s unlimited`, as upstream's job scripts). All resolved; nekRS validated at 1/2/4 GPUs
- build_strategy_notes: **NATIVE**

## CP2K

- official_repository: https://github.com/cp2k/cp2k (DBCSR now an external dependency: https://github.com/cp2k/dbcsr; containers https://github.com/cp2k/cp2k-containers)
- official_documentation: https://manual.cp2k.org/ (technologies/accelerators/cuda.html, hip.html; getting-started/build-from-source.html, build-with-spack.html); compiler matrix wiki; https://www.cp2k.org/performance
- latest_stable_release: v2026.2 (2026-07-15). 2026.1 removed the GNU Makefile (CMake-only); 2026.2 ships `make_cp2k.sh` (Spack-based). (2025.2 is two releases behind.)
- selected_commit_sha: `67b5da876dd6a76b8b021d5a04d1c81ba79a4c50`
- license: GPL-2.0-or-later
- application_owned_loc: `src/` 1,085,842 (Fortran 1,015,193 in 1,325 files; C 58,310; CUDA 1,572 in 5 `.cu`; C++ 1,341; OpenCL 311); no bundled third-party source
- main_languages: Fortran 2008 (93 %), C, CUDA/C++
- build_system: CMake >= 3.24 + Ninja; dependency bootstraps: `tools/toolchain/install_cp2k_toolchain.sh` (shell, ~40 pinned deps) or `make_cp2k.sh` (private Spack)
- cxx_standard: C11 + C++17; Fortran 2008
- supported_compilers: GCC 9-16 recommended (toolchain default 14.3.0), Intel oneAPI 2024.2.1 with limitations, `%clang` conflicts; `-allow-unsupported-compiler` forced for nvcc
- cuda_support: yes (`CP2K_USE_ACCEL=CUDA`; `CMAKE_CUDA_ARCHITECTURES=<sm>` or `CP2K_WITH_GPU=<name>`). **2026.2's name list ends at H100/GB10 (no B200); upstream master adds `B200 -> 100`.** `CMAKE_CUDA_ARCHITECTURES=100` is accepted by 2026.2's own CMake; the gap is **DBCSR 2.10.0** whose `WITH_GPU` list stops at H100 and derives the arch from it (upstream master's toolchain patches it with a sed + copies `parameters_H100.json -> parameters_B200.json`, i.e. untuned SMM parameters). CI/Spack pin CUDA 12.9.1; no CUDA 13 mention. GPU components toggle individually (DBCSR, DBM, GRID, PW, libGint HFX, SPLA, ELPA, cuSOLVERMp)
- hip_rocm_support: yes (`CP2K_USE_ACCEL=HIP`, Mi50..Mi300 -> gfx906..gfx942); **no gfx950 in CP2K** (DBCSR 2.10 has Mi350 = gfx950)
- mpi_support: yes (MPI-3 required; hybrid MPI+OpenMP `psmp` is production)
- official_gpu_programming_model: native CUDA/HIP (offload layer) + DBCSR JIT kernels (NVRTC), cuBLAS/cuFFT
- multi_gpu_support: device = `MOD(rank, device_count)` (global rank); several ranks per GPU normal (CI runs 2-4 ranks on 1 GPU); no explicit ranks-per-GPU recommendation found
- multi_node_support: yes
- gpu_aware_mpi_requirement: not required (DBM/DBCSR communicate through pinned host buffers; DBCSR `+g2g` optional)
- rank_to_gpu_binding: self-binding by global-rank modulo -> correct when ranks/node is a multiple of visible GPUs; otherwise external per-rank `CUDA_VISIBLE_DEVICES`
- topology_decomposition_controls: none on the CLI; input `&GLOBAL`/`&DBCSR`/`&QS` options, `PREFERRED_DIAG_LIBRARY`, `OMP_NUM_THREADS`; regtest driver `do_regtest.py --mpiranks --ompthreads --num_gpus`
- major_dependencies (toolchain pins): DBCSR 2.10.0, libxsmm 2.0.0, libint 2.13.1, libxc 7.0.0, FFTW 3.3.11, OpenBLAS 0.3.33, ScaLAPACK 2.2.3, ELPA 2026.02.002, COSMA 2.8.4, SpLA 1.6.1, SpFFT 1.1.1, SIRIUS 7.11.1, spglib, HDF5, plumed, dftd4, tblite, libvori, GauXC, libtorch 2.7.1, ...; minimal GPU-DFT set: DBCSR, BLAS/LAPACK/ScaLAPACK, FFTW, libxsmm, libint, libxc
- dependency_complexity: very high
- official_inputs_datasets: all in-repo: `benchmarks/QS/H2O-{32..8192}.inp`, `QS_DM_LS` (`NREP` weak scaling), `QS_ot_ls`, `QS_single_node/*`, `QS_LiH_HFX`, `QS_mp2_rpa`, QMMM, ...; `tests/` 5,255 inputs with reference values (`TEST_FILES.toml`); `data/` 76 MB basis sets
- correctness_mechanism: regtests (matcher values vs `ref=` with `tol=`), `benchmarks/QS_reference/`, `check-release-comparison.py` (energy invariance 1e-10 across MPIxOMP layouts); GPU CI runs the full regtest with `--num_gpus`
- strong_scaling_input_availability: yes (H2O-64/128/256, `H2O-dft-ls.NREP4`, LiH-HFX)
- weak_scaling_input_availability: yes (`QS_DM_LS` `NREP`; `QS/H2O-N` doubling series)
- expected_build_time: dependencies 3-6 h at -j32 (libint lmax >= 5 ~1 h; ELPA, SIRIUS, COSMA, libxc, DBCSR); CP2K 40-90 min; regtests 1-2 h
- expected_disk_usage: source 452 MB; toolchain 15-25 GB (minimal ~5 GB); build 4-6 GB; Spack path 20-40 GB
- expected_input_data_size: 152 MB benchmarks + 76 MB data + 52 MB tests; no downloads
- b200_cuda132_risk: medium-high (no B200 name in 2026.2; DBCSR 2.10.0 arch/parameter patch; CUDA 13.2 untested upstream; Spack `cp2k` and `dbcsr` recipes hard-reject `cuda_arch=100`)
- mi355x_hip_risk: high
- container_availability: official Docker Hub `cp2k/cp2k` tags `{version}_{mpich|openmpi}_{generic|native}_{cuda_P100|A100|H100}_psmp` -- no B200 image; multi-node "requires the MPI of the host system"; no Apptainer here
- spack_availability: yes (`cp2k` 2026.2 upstream; officially recommended via `make_cp2k.sh`), but `cuda_arch` limited to 35-90 in both `cp2k` and `dbcsr` recipes; local checkout too old
- recommended_integration_priority: SECOND_BATCH (mature GPU regtests and inputs; excluded from this round by the > 2 h dependency rule and the DBCSR Blackwell patch)
- blocker: DBCSR sm_100 entry/parameters; 3-6 h dependency stack; Spack `conflicts()`; ELPA-GPU/SIRIUS/COSMA with CUDA 13.2 unverified (keep off first)
- build_strategy_notes: **NATIVE+SPACK_DEPS** (upstream toolchain/Spack for the CPU-side stack, DBCSR + CP2K natively with `CMAKE_CUDA_ARCHITECTURES=100`)

## Nyx

- official_repository: https://github.com/AMReX-Astro/Nyx (default branch `development`)
- official_documentation: https://amrex-astro.github.io/Nyx/docs_html/ (`getting_started/BuildingCMake.html`, `NyxSundials.html`, `RunningTheCode.html`, `ICs.html`, `LoadBalancing.html`, `NightlyTests.html`)
- latest_stable_release: 26.09 (tag 2026-08-26, release 2026-09-01); irregular tagging (26.07 before, then nothing since 21.10)
- selected_commit_sha: `e06eabc1b9dbcad5612db9529aced682402daede`
- license: BSD-3-Clause-LBNL
- application_owned_loc: `Source/` 21,644 (C++ 15,696 / 51 files); `Exec/` 10,120; `Util/` 3,398; submodules `subprojects/amrex` @ `6e875b7c` (development, ancestor of 26.09) and `subprojects/sundials` @ v7.2.1
- main_languages: C++ (98 %), CMake/GNU make; residual Fortran in `Exec/GravityTests`
- build_system: CMake >= 3.14 stated, effectively 3.25 (AMReX submodule); or GNU Make per `Exec/*` directory
- cxx_standard: inherits C++20 from AMReX 26.09 (stale README says C++11/CUDA 9; CI passes `-DCMAKE_CXX_STANDARD=17`)
- supported_compilers: effective constraints from AMReX 26.09 (GCC >= 11, CUDA >= 12.2, ROCm >= 6); CI: GCC, Clang, NVCC 12.6 (job still named "cuda11"), HIP gfx908
- cuda_support: yes (`Nyx_GPU_BACKEND=CUDA`, `CMAKE_CUDA_ARCHITECTURES=100`; `Nyx_OMP` forced off); `Nyx_HEATCOOL=YES` needs SUNDIALS built with `ENABLE_CUDA` + fused kernels. No sm_100/CUDA 13 mention
- hip_rocm_support: yes, documented "under development" (Spock gfx908 script, CI gfx908); no gfx942/gfx950
- mpi_support: yes (`Nyx_MPI` default ON)
- official_gpu_programming_model: AMReX
- multi_gpu_support: one rank per GPU via AMReX (Summit `jsrun -a 1 -g 1`, Spock `--gpus-per-task=1`)
- multi_node_support: yes
- gpu_aware_mpi_requirement: optional (AMReX auto-detect)
- rank_to_gpu_binding: self-binding by AMReX (rank-in-node)
- topology_decomposition_controls: `amr.n_cell`, `amr.max_grid_size` (GPU: 128 or 256 recommended), `amr.blocking_factor`, `amr.max_level`, `DistributionMapping.strategy`, `nyx.load_balance_*`; boxes >= ranks
- major_dependencies: AMReX (any `>= 20.11` external, or submodule), SUNDIALS >= 6.0 (HEATCOOL only), MPI, CUDA; optional Reeber/Gimlet/Ascent. A single AMReX 26.09 install (3D, PARTICLES, LINEAR_SOLVERS, EB, FFT, MPI, CUDA, SUNDIALS) satisfies WarpX's pin and Nyx's minimum
- dependency_complexity: low-medium
- official_inputs_datasets: `Exec/LyA/inputs` (64^3, IC `64sssss_20mpc.nyx` 14.7 MB), `inputs.rt` (32^3), `Exec/AMR-density/inputs.cuda`, `Exec/AMR-zoom`, `Exec/MiniSB/inputs.32` (Santa Barbara), `Exec/Scaling/inputs` (64^3 `RandomPerCell`, no IC file) + `inputs.256.noreduceverb`, HydroTests (Sedov/Sod/shock tubes), GravityTests, ...; **larger ICs (256^3, 1024^3) exist only at OLCF paths, no public URL**; `nyx.particle_init_type = Cosmological` in code but undocumented
- correctness_mechanism: AMReX nightly regression suite (`fcompare`/`particle_compare` vs LBNL-hosted benchmark plotfiles -- not in repo); analytic hydro tests; MiniSB comparison
- strong_scaling_input_availability: `Exec/Scaling/` (64^3, 256^3); Spock script references 768^3-2048^3 inputs not shipped
- weak_scaling_input_availability: informal (`RandomPerCell` init scales `amr.n_cell` freely; `inputs.cuda` documents `prob_hi` for 512^3/1024^3/6144^3)
- expected_build_time: 20-35 min at -j32 (AMReX 3D CUDA 10-15, SUNDIALS CUDA 5-10, Nyx few)
- expected_disk_usage: source 104 MB (65 MB ICs) + submodules ~100 MB; build 1-3 GB
- expected_input_data_size: ~65 MB shipped
- b200_cuda132_risk: medium (AMReX part as WarpX; SUNDIALS 7.2.1 predates CUDA 13; stale CI; only 64^3 ICs)
- mi355x_hip_risk: medium-high
- container_availability: none
- spack_availability: no `nyx` package (only `amrex +sundials`, `sundials`)
- recommended_integration_priority: SECOND_BATCH (cheap and AMReX-native, but stale docs/CI, SUNDIALS-CUDA for the flagship problem, no downloadable large ICs, no shipped baselines)
- blocker: submodules to initialise (or external AMReX with the component set above); SUNDIALS 7.2.1 + CUDA 13.2 untested; large ICs unavailable
- build_strategy_notes: **NATIVE** (against the WarpX AMReX 26.09 checkout; start with MiniSB / adiabatic LyA, then `Nyx_HEATCOOL=YES`)

## QMCPACK

- official_repository: https://github.com/QMCPACK/qmcpack (`develop`; `main` = release)
- official_documentation: https://qmcpack.readthedocs.io/en/develop/ (installation, running, performance_portable); Nexus docs
- latest_stable_release: v4.4.0 (2026-08-31); 4.3.0 raised CUDA minimum to 12.3; legacy drivers slated for removal
- selected_commit_sha: `2601d62e353934f1526cab1f67f30b6672b7c76f`
- license: University of Illinois/NCSA Open Source License
- application_owned_loc: `src/` 337,813 (C++ 168,240; headers 146,034; CUDA 8,066); `external_codes/` 294,038 and `nexus/` 140,384 separate
- main_languages: C++ (~93 %), CUDA, Python
- build_system: CMake >= 3.21
- cxx_standard: C++17 (C++20 auto-selected if the compiler defaults to it)
- supported_compilers: GCC >= 9 (but "OpenMP offload is not ready for GCC"), Clang >= 7, oneAPI >= 2021, NVHPC, XL; docs: **"For NVIDIA GPUs, LLVM clang"**; nightly: Clang 22.1.1, CUDA 12.9, ROCm 7.0.1, Open MPI 5.0.10, HDF5 1.14.5, Boost 1.90/1.84
- cuda_support: yes: `-DQMC_GPU="openmp;cuda" -DQMC_GPU_ARCHS=sm_100` (arbitrary `sm_XX` passthrough); `QMC_GPU=cuda` alone = cuBLAS/cuSOLVER batched LA with the rest on CPU (CMake-valid, not the recommended GPU build); `find_package(CUDAToolkit 12.3)`; no CUDA-13-removed APIs found; no sm_100/CUDA 13 mention
- hip_rocm_support: yes (`QMC_GPU="openmp;hip"`, rocBLAS/hipBLAS, amdclang; Frontier gfx90a; ROCm 7.0.1 tested); gfx950 passthrough undocumented
- mpi_support: yes (walkers across ranks; ensembles)
- official_gpu_programming_model: OpenMP target offload + vendor BLAS/solver libraries, small native CUDA/HIP/SYCL kernels
- multi_gpu_support: docs: "1 MPI task should be used per GPU per node" for medium/large runs; device from node-local rank (`DeviceManager`), warns if `local_size % num_devices != 0`
- multi_node_support: yes (`shared_ranks` spline sharing in 4.4.0)
- gpu_aware_mpi_requirement: not used
- rank_to_gpu_binding: self-binding (node-local rank -> device; respects `CUDA_VISIBLE_DEVICES`)
- topology_decomposition_controls: none (walker-parallel): `walkers_per_rank`/`total_walkers`, `blocks/steps/timestep`, `OMP_NUM_THREADS`, `--dryrun`
- major_dependencies: MPI, BLAS/LAPACK, HDF5 >= 1.10 (1.14.5 tested; conda has 2.2.0 -- untested upstream), FFTW3, **Boost >= 1.70 headers (absent on node)**, libxml2, Python 3 + numpy (h5py), CUDA >= 12.3, **Clang with NVPTX offload (absent on node)**; bundled boost_multi, Catch2, mpi3
- dependency_complexity: medium (standard libs mostly in conda; official GPU path needs an LLVM/Clang offload toolchain)
- official_inputs_datasets: `tests/` 178 MB (solids, molecules, heg, afqmc, ...; ctest labels `unit`, `deterministic`, `short`, `performance`); `tests/performance/NiO` S1-S256 spline files 43 MB - 8.8 GB each from an external Box link ("direct links ... may be fragile"), `-DQMC_DATA=<dir>`; `examples/`
- correctness_mechanism: deterministic ctests (fixed seeds, exact scalar checks, 142 entries), statistical energy-within-error tests (`qmc-ref`), Catch2 unit tests
- strong_scaling_input_availability: yes (NiO S-series at fixed walker count; needs downloads)
- weak_scaling_input_availability: yes (fixed `walkers_per_rank`; or S8 -> S16 -> S32 electron counts)
- expected_build_time: 15-30 min (`QMC_GPU=cuda`, GCC) / 30-60 min (`openmp;cuda`, Clang) at -j32, + 1-2 h if LLVM must be built
- expected_disk_usage: source 575 MB; build 3-6 GB; LLVM +5-10 GB
- expected_input_data_size: 178 MB shipped; NiO 0.3-8.8 GB per size (S1-S32 ~3 GB practical)
- b200_cuda132_risk: medium-high (sm_100 passthrough trivial; but Clang-offload + CUDA 13.2 unverified upstream (LLVM 22.1.1 tested with CUDA 12.9); GCC-only fallback leaves most kernels on CPU; conda HDF5 2.2.0 untested; Boost missing)
- mi355x_hip_risk: medium
- container_availability: CI dependency images only (CPU, no CUDA); no production/GPU image
- spack_availability: `qmcpack` package to 4.3.0 upstream (local 4.1.0); its `+cuda` passes `QMC_CUDA=1`, which 4.x CMake ignores -> CPU-only binary (inferred from CMake); no offload/rocm variants
- recommended_integration_priority: SECOND_BATCH (excellent benchmark suite; needs an LLVM/Clang offload toolchain, Boost, possibly HDF5 1.14, and multi-GB datasets)
- blocker: Clang/LLVM with NVPTX offload; Boost headers; HDF5 version; NiO datasets external
- build_strategy_notes: **NATIVE+SPACK_DEPS** (Spack only for `llvm+cuda`, `boost`, `hdf5@1.14`; QMCPACK itself natively with `QMC_GPU`/`QMC_GPU_ARCHS`)

## GEOS

- official_repository: https://github.com/GEOS-DEV/GEOS (formerly GEOSX; TPLs https://github.com/GEOS-DEV/thirdPartyLibs; submodules LvArray, BLT, PVTPackage, hdf5_interface, uberenv)
- official_documentation: https://geosx-geosx.readthedocs-hosted.com/en/latest/ (QuickStart, buildGuide/{Prerequisites,Dependencies,BuildProcess,SpackUberenv,ContinuousIntegration}, advancedExamples/performanceBenchmarks)
- latest_stable_release: 1.2.0 (2024-10-02, "Latest"); `develop` (`b7a0f133...`, 2026) is ~2 years ahead and is what docs/CI describe (TPL tag 361-1070, CUDA 12.9.1 images, RAJA 2026.07.0)
- selected_commit_sha: `920e17a00b2ab86a5c0c98088fc69b904b1af55b`
- license: LGPL-2.1-only
- application_owned_loc: `src/` ~240 k (C++ 123,485 + headers 105,242 + CMake 3,866 + Python 7,738; 511 .cpp / 714 .hpp); submodules (LvArray, PVTPackage, hdf5_interface, BLT) not counted; `inputFiles/` 108 MB (582 XML)
- main_languages: C++17 with RAJA/CHAI device lambdas (no `.cu`; nvcc compiles `.cpp` under BLT), Python (ATS/pygeosx)
- build_system: CMake >= 3.24 + BLT; host-config files; TPLs via `thirdPartyLibs/scripts/config-build.py` superbuild or uberenv/Spack
- cxx_standard: C++17
- supported_compilers: docs "gcc 12+ or clang 13.0+"; 1.2.0 CUDA CI rows: clang 10/gcc 9.4 + CUDA 11.8.89, clang 17 + CUDA 12.5.1, gcc 8.5 + CUDA 12.5.1; develop TPL images gcc13/clang19 + CUDA 12.9.1; **CUDA 13.2.1 rows commented out: "CUDA 13 is blocked by the pinned RAJA package: raja '^cuda@13:' conflicts with '+cuda'"**
- cuda_support: yes (`ENABLE_CUDA`, `CMAKE_CUDA_ARCHITECTURES`, `ENABLE_HYPRE_DEVICE=CUDA` with hypre `--with-cuda --enable-unified-memory --with-umpire`); documented CUDA 11.5-12.5 (1.2.0), 12.9.1 (develop); no sm_100 mention (CI `cuda_arch=86,120`)
- hip_rocm_support: yes (`ENABLE_HIP`, `CMAKE_HIP_ARCHITECTURES` gfx90a/gfx942/gfx1100, Frontier/Tioga host-configs, ROCm 5.4-6.4); no gfx950
- mpi_support: yes (mesh decomposition, hypre/Trilinos/PETSc, parallel HDF5/Silo/VTK)
- official_gpu_programming_model: RAJA + CHAI + Umpire + camp via LvArray; hypre on device
- multi_gpu_support: one MPI rank per GPU (Frontier launch scripts `--ntasks-per-gpu=1 --gpu-bind=closest`; lassen jsrun)
- multi_node_support: yes (27 B-element weak-scaling study on Frontier)
- gpu_aware_mpi_requirement: optional (GEOS uses pinned host buffers, `-s/--suppress-pinned`; hypre `ENABLE_HYPRE_GPU_AWARE_MPI` off by default)
- rank_to_gpu_binding: **external** (no `cudaSetDevice`/local-rank logic in `src/`; relies on `--gpu-bind`, jsrun or per-rank `CUDA_VISIBLE_DEVICES`)
- topology_decomposition_controls: CLI `-x/-y/-z` partitions (InternalMesh), `-s`, `-b`; external meshes partitioned by ParMETIS/Scotch (any rank count); `<Benchmarks>` XML blocks (`scaling="strong" scaleList=...`) + `benchmarks/runBenchmarks.py`
- major_dependencies: pinned by thirdPartyLibs `0e2ed33e` (tag 284-535) for 1.2.0: RAJA/CHAI(+Umpire, camp) **v2024.07.0**, hypre v2.31.0-12, conduit 0.9.2, HDF5 1.12.1, silo 4.11, VTK 9.3.1, Trilinos 15.1.1 (optional), PETSc 3.19.4 (optional), superlu_dist, ParMETIS 4.0.3, Scotch 7.0.3, SuiteSparse 5.10.1, Caliper 2.11.0, Adiak, pugixml, fmt 11.0.1, mathpresso. thirdPartyLibs HEAD (develop): RAJA/CHAI/Umpire **v2026.07.0** (= Level 2's pins exactly), hypre master `f1374fb6`, VTK 9.7.0, Trilinos 16.1.0, ... -> Level 2's `.deps/install/{raja,umpire,chai}` match GEOS *develop*, not 1.2.0
- dependency_complexity: very high
- official_inputs_datasets: 582 XML (`*_smoke.xml`, `*_benchmark.xml`; solidMechanics, singlePhaseFlow, compositionalMultiphaseFlow incl. SPE10 tables in-repo, poromechanics, hydraulicFracturing, wavePropagation, `wellboreECP/*/level01-06`); some inputs reference the separate LFS GEOSXDATA repo; integrated tests need `geos-ats` + a baseline tarball from a GCP bucket
- correctness_mechanism: gtest unit tests; integrated restart-check vs baselines with tolerances (`BASELINE_NOTES.md`); analytical examples in docs (Mandel, Terzaghi, Sneddon, KGD)
- strong_scaling_input_availability: yes (`<Benchmarks>` blocks, `runBenchmarks.py`; `-x -y -z` overrides)
- weak_scaling_input_availability: yes (`wellboreECP` level01-06, 826 k -> 27 B elements)
- expected_build_time: TPL superbuild 2-3 h at -j32 (VTK/Trilinos/hypre-CUDA; Trilinos can be disabled with hypre on device) + GEOS 1.5-2.5 h -> ~4-5 h
- expected_disk_usage: source 381 MB (+ ~150 MB submodules); TPLs 15-25 GB; GEOS build 10-20 GB
- expected_input_data_size: 108 MB in-repo; optional LFS data and baseline tarball (sizes unpublished)
- b200_cuda132_risk: high for tag 1.2.0 (RAJA suite 2024.07 + hypre 2.31, CUDA <= 12.5 heritage, no sm_100), medium for develop (RAJA 2026.07.0 built with CUDA 13.2 in Level 2; upstream's own Spack path calls CUDA 13 blocked; hypre master + cuSPARSE/cuSOLVER on CUDA 13.2 unverified)
- mi355x_hip_risk: high
- container_availability: CI TPL images on Docker Hub (`geosx/<os>-<compiler>-cuda<ver>:<tag>`, CUDA 11.8/12.5/12.9, in-container Open MPI) -- CI/devcontainer use only
- spack_availability: **no upstream package** (Spack's `geos` is libgeos); GEOS ships its own `geosx` recipe consumed only through uberenv with LC-system `spack.yaml` files (host-config generation only, "must never be used without a spack.yaml")
- recommended_integration_priority: SECOND_BATCH (strong benchmark story; 4-5 h build, no Blackwell/CUDA 13 upstream coverage, tag-vs-develop decision, cloud-hosted baselines)
- blocker: choose 1.2.0 (independent 2024.07 RAJA suite) vs develop (shares Level 2's RAJA 2026.07.0); multi-hour TPL build; submodules; hypre GPU on CUDA 13.2 unverified; baseline download + `geos-ats`; `cuda_arch=100` never used upstream
- build_strategy_notes: **NATIVE** (thirdPartyLibs superbuild + custom host-config, `ENABLE_TRILINOS=OFF`, `ENABLE_HYPRE_DEVICE=CUDA`, `CMAKE_CUDA_ARCHITECTURES=100`, conda Open MPI; prefer develop so RAJA/CHAI/Umpire 2026.07.0 can be shared with Level 2)

## DFT-FE

- official_repository: https://github.com/dftfeDevelopers/dftfe (release branch `release1.2`); install scripts https://github.com/dftfeDevelopers/install_DFTFE (per-machine branches); benchmarks https://github.com/dftfeDevelopers/dftfe-benchmarks
- official_documentation: https://sites.google.com/umich.edu/dftfe ; manual PDF (`manual` branch); Doxygen https://dftfedevelopers.github.io/dftfe/
- latest_stable_release: 1.2.0 (2025-08-17): "seamless support for NVIDIA, AMD and Intel GPUs", meta-GGA, DFT+U, mixed precision
- selected_commit_sha: `7147faa51f7c9f3075fffaa5e48ba989bcd329c1`
- license: LGPL-2.1-or-later
- application_owned_loc: `src/ include/ utils/` 130,659 (C++ 109,351 in 221 files; headers 21,187; device code is `.cc` compiled as CUDA/HIP/SYCL); no bundled TPLs
- main_languages: C++17 (device kernels in CUDA/HIP/SYCL-flavoured C++)
- build_system: CMake >= 3.17; two builds per install (`WITH_COMPLEX=OFF/ON`); helper `setupUser.sh`
- cxx_standard: C++17
- supported_compilers: not formally documented; scripts use Cray CC (Perlmutter gcc-native 12.3, Frontier cpe/25.09), icpx (NSM A100), gcc-10 (Ubuntu Docker)
- cuda_support: yes (`WITH_GPU=ON GPU_LANG=cuda GPU_VENDOR=nvidia CMAKE_CUDA_ARCHITECTURES=<sm> CMAKE_CUDA_FLAGS="-arch=sm_XX"`, cuBLAS; optional NCCL `WITH_DCCL`, `WITH_GPU_AWARE_MPI`); scripts hard-code sm_70/sm_80; **no sm_90/sm_100 anywhere**; documented CUDA 11.7 / 12.9.1; small device API surface (cuBLAS, `cudaSetDevice`), no removed APIs; deal.II built CPU-only (Kokkos Serial)
- hip_rocm_support: yes (`GPU_LANG=hip`, hipBLAS, gfx90a Frontier; RCCL optional); no gfx942/gfx950
- mpi_support: yes (FE domain decomposition x band groups `NPBAND` x k-point pools `NPKPT`; ELPA/ScaLAPACK)
- official_gpu_programming_model: native CUDA/HIP/SYCL via DFT-FE's device abstraction + cuBLAS/hipBLAS/oneMKL
- multi_gpu_support: one rank per GPU documented (Frontier `--ntasks-per-gpu 1 --gpu-bind closest`; Perlmutter `--gpus-per-task=1`); oversubscription seen in test scripts (18 ranks / 6 GPUs); threads = 1
- multi_node_support: yes (to ~40,000 GPUs; Summit benchmarks)
- gpu_aware_mpi_requirement: optional (`WITH_GPU_AWARE_MPI`, "use with care"; NCCL alternative)
- rank_to_gpu_binding: self-binding (`device_id = mpi_rank % n_devices`, global rank)
- topology_decomposition_controls: `.prm` `subsection Parallelization { NPKPT, NPBAND, BAND PARAL OPT }`, `subsection GPU { USE GPU, AUTO GPU BLOCK SIZES, USE GPUDIRECT MPI ALL REDUCE, USE ELPA GPU KERNEL }`, `USE ELPA`; ranks divisible by `NPKPT*NPBAND`
- major_dependencies (install_DFTFE pins): deal.II >= 9.5.1 (`+P4EST +64BIT_INDICES +MPI +LAPACK`; scripts 9.6.2/9.7.1), p4est 2.8.6/7, Kokkos 4.3/4.6 (CPU-only, for deal.II), Boost 1.86, ScaLAPACK 2.2.x, BLIS/libflame or OpenBLAS, ELPA 2025.01/2025.06 (`--enable-nvidia-gpu-kernels --with-NVIDIA-GPU-compute-capability=sm_80`), ALGLIB, libxc 6.2.2/7.0.0, spglib, libxml2, numdiff (tests); optional PETSc/SLEPc, NCCL, dftd3/4, libtorch
- dependency_complexity: high
- official_inputs_datasets: `demo/ex1-3` (with reference `.output`), `testsGPU/pseudopotential/{real,complex}` (56 GPU `.prm` cases, `accuracyBenchmarks/`, `diffScript`, Slurm/PBS scripts), `tests/dft/*` (112 `.prm.in`, 133 `*.mpirun=N.output` references), 57 ONCV pseudopotentials (11 MB), `data/` 73 MB; external `dftfe-benchmarks` (Mo 431-8,191 atoms; Al nanoparticles)
- correctness_mechanism: ctest via deal.II harness with `numdiff` against `.output`; GPU: `REPRODUCIBLE OUTPUT = true` + `diffScript` vs `accuracyBenchmarks/`
- strong_scaling_input_availability: indirect (any fixed system on 1..4 GPUs; Summit references per size in the benchmarks repo)
- weak_scaling_input_availability: `dftfe-benchmarks` Mo N x N x N supercell series (24 -> 3,600 GPUs); in-repo inputs do not scale automatically
- expected_build_time: deal.II 1-1.5 h; ELPA-GPU 15-30 min; small libs 20 min; DFT-FE real + complex 25-40 min each -> ~3-4 h
- expected_disk_usage: source 126 MB; deps 8-12 GB; DFT-FE 3-4 GB
- expected_input_data_size: ~85 MB in-repo; benchmarks repo external
- b200_cuda132_risk: medium (arch is user-supplied and kernels generic; never run above sm_80 upstream; ELPA `sm_100` acceptance and ELPA 2025.x with CUDA 13.2 unverified -> ELPA CPU/ScaLAPACK fallback; NCCL for CUDA 13 separate; scripts assume Cray/icpx)
- mi355x_hip_risk: medium-high
- container_availability: CPU Docker recipe only (`install_DFTFE` `generalUbuntuCPU`); no GPU image
- spack_availability: `dftfe` package stale (0.5-0.6, 2019, no GPU variants); upstream route is the per-machine `install_DFTFE` scripts (not Spack)
- recommended_integration_priority: SECOND_BATCH
- blocker: deal.II/p4est/Kokkos/Boost multi-hour stack; ELPA GPU kernels for sm_100; NCCL; gcc/Open MPI adaptation of the Cray/icpx scripts; numdiff
- build_strategy_notes: **NATIVE** (node-specific `install_dftfe.sh` derived from the `nsm_A100` branch: conda GCC 13.3 + system gfortran + Open MPI 5.0.10, OpenBLAS/ScaLAPACK, Kokkos Serial, deal.II 9.7.1, ELPA (GPU kernels attempted at sm_100, CPU fallback), DFT-FE real+complex `CMAKE_CUDA_ARCHITECTURES=100`, `WITH_DCCL=OFF`, `WITH_TESTING=ON`)

---

## Cross-cutting findings

1. **Spack is not a shortcut to Blackwell for any candidate on this node.** The
   personal checkout (2025-05-06) predates every 2026 release and lacks
   `cuda_arch=100`/`gfx950`; upstream recipes are missing (SPARTA, Nyx, GEOS,
   SPECFEM3D), stale (nekRS 23.0, DFT-FE 0.6, QMCPACK `+cuda` inert for 4.x),
   or reject `cuda_arch=100` outright (`cp2k`, `dbcsr`; `raja ^cuda@13:`
   conflict blocks GEOS' path). Where Spack helps it is for CPU-side
   dependencies only (CP2K, QMCPACK) -> `NATIVE+SPACK_DEPS`.
2. **Containers do not apply here**: no Apptainer/Singularity on dgx003, no
   official B200 image for any candidate, and containers would not provide the
   host driver, the CUDA-aware Open MPI transport or the network.
3. **Blackwell/CUDA 13 upstream coverage** is thin everywhere: only WarpX
   (Perlmutter profile, CUDA 13.2.78) and the two Kokkos codes (arch flag +
   CUDA 13 fixes in the bundled Kokkos) have first-class support; SPECFEM3D
   v4.1.1 needs two back-ports; CP2K 2026.2/DBCSR 2.10.0, GEOS, DFT-FE never
   mention sm_100; GEOS explicitly lists CUDA 13 as blocked in its Spack path.
4. **Fortran**: the conda environment has no gfortran; the system gfortran
   14.2.1 works with the conda Open MPI Fortran bindings (`OMPI_FC=/usr/bin/gfortran`,
   verified with a 2-rank MPI Fortran program). Needed by SPECFEM3D, nekRS,
   CP2K, DFT-FE (ScaLAPACK/ELPA/p4est).
5. **Rank -> GPU mapping**: LAMMPS/SPARTA/nekRS bind by node-local rank,
   WarpX/Nyx by AMReX's rank-in-node, SPECFEM3D/CP2K/DFT-FE by *global* rank
   modulo device count, GEOS not at all. The Level 3 launcher's per-rank
   wrapper (one visible GPU per rank, audited) makes all of them correct on a
   node; nekRS additionally needs `--device-id 0` under the wrapper.
6. **Multi-node MPI is BLOCKED/UNVERIFIED on this site**; every application
   above is multi-node capable per upstream, none is verified beyond one node.
