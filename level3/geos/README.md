# GEOS -- Level 3 second batch

Multiphysics reservoir/geomechanics simulator (LLNL); here the solid-mechanics
`beamBending` workflow (linear-elastic cantilever, analytic solution shipped
upstream) with the hypre linear solver on the GPU (RAJA/CHAI/Umpire device
kernels via LvArray).

## Provenance / versions

| Item | Value |
|---|---|
| GEOS | `develop` snapshot `b7a0f13305277c3d825ee93f34946ac4e8c94fee` (2026-09-04, fixed SHA; the last tag 1.2.0 is from 2024-10 and pins RAJA 2024.07/hypre 2.31 with CUDA <= 12.5 heritage -- see `APPLICATION_AUDIT.md` for the stable-vs-develop decision) with submodules (LvArray, BLT `9ff77344` = 0.6.2-era, PVTPackage, hdf5_interface, ...), LGPL-2.1 |
| thirdPartyLibs | `9b55672f6f8a73d02fd632396eb0410e58c9b120` = the `GEOS_TPL_TAG 361-1070` that this GEOS commit requires (the repo has no git tags); RAJA/CHAI/Umpire/camp **2026.07.0**, hypre `f1374fb6` (`--with-cuda --enable-cusparse --enable-cusolver --enable-unified-memory --with-umpire --with-gpu-arch=100`), HDF5 1.12.1, conduit 0.9.5, silo 4.11.1, VTK 9.7.0, pugixml, fmt, SuiteSparse 5.10.1, ParMETIS/METIS, SuperLU_DIST `0f6efc3`, Scotch; Trilinos/PETSc/Caliper/MathPresso **off** |
| Compilers | system GCC 14.2.1 (C/C++/Fortran), nvcc 13.2.78 with g++ 14.2.1 host (`-allow-unsupported-compiler` not needed), conda Open MPI 5.0.10 (`OMPI_CC/CXX/FC` -> system GCC) |
| GPU target | `CMAKE_CUDA_ARCHITECTURES 100` / `CUDA_ARCH sm_100` in the host-config; hypre device backend (`ENABLE_HYPRE_DEVICE=CUDA`), `ENABLE_HYPRE_GPU_AWARE_MPI=OFF` (GEOS' default pinned-host-buffer path) |
| BLAS/LAPACK | private OpenBLAS 0.3.30 (`TARGET=SAPPHIRERAPIDS`, single-threaded) |
| Build strategy | NATIVE: upstream `thirdPartyLibs` superbuild (`scripts/config-build.py -hc <host-config>`) + GEOS with the same host-config; no Spack (`geosx` recipe is uberenv-only), no container |
| Host-config | `host-configs/gmu-hopper-gcc14-ompi5010-cuda132-sm100.cmake` (paths injected as `-D HPCPERF_*` by build.sh) |
| Profile | `cuda132-gcc142-ompi5010` |

## Build-system patches to thirdPartyLibs (class B, `patches/`)

- `0001-tpl-superlu_dist-url-hash-typo.patch`: upstream's `SUPERLU_URL_HASH` is a
  65-character string (also on master, checked 2026-09-05), so the superbuild's
  download step can never verify it; the tarball's real sha256 is that string
  minus the trailing character. No code impact.
- `0002-tpl-raja-vectorization-overridable.patch`: upstream forces
  `RAJA_ENABLE_VECTORIZATION=ON` for every non-HIP build (OFF for ROCm because the
  layer "does not compile with the ROCm toolchain"). With nvcc 13.2 + EL10's GCC
  14 (default `-march=x86-64-v3` -> `__AVX2__` in every nvcc host pass) RAJA's
  AVX2 tensor-register headers fail to compile in every CUDA translation unit.
  The patch only makes the option overridable; build.sh passes
  `-D RAJA_ENABLE_VECTORIZATION=OFF` (upstream's own ROCm choice). GEOS/LvArray do
  not use the experimental `RAJA::expt` tensor layer.
- `0003-tpl-hdf5-step-generator-and-build-command.patch`: the hdf5 step is the
  only CMake-based TPL step without `CMAKE_GENERATOR ${TPL_GENERATOR}` and with
  a hardcoded `make`; under a Ninja-generated superbuild (`config-build.py -n`)
  its sub-build inherits Ninja and `make` finds no Makefile. Same generator and
  build/install commands as every other step; no effect on the built HDF5.
- Not built (disabled, no patch): **hypredrive** (optional hypre driver library;
  GEOS' `ENABLE_HYPREDRV` defaults to OFF and the beam workflow does not use it).
  Upstream's TPL step compiles it against the Umpire-enabled hypre without
  Umpire's include path (`umpire/config.hpp: No such file`) -- disabled with
  `-D ENABLE_HYPREDRV=OFF` rather than patched.
- Node adaptations (no patch): `config-build.py` deletes an existing build tree
  before configuring, so build.sh configures once and re-asserts cache values
  (`RAJA_ENABLE_VECTORIZATION`, `ENABLE_HYPREDRV`, `NUM_PROC`) with plain `cmake`
  on re-runs; stale ExternalProject configure stamps left by an interrupted run
  are removed when the sub-build has no Makefile.

## GEOS itself: one BLT back-port (class D) and one include-order adaptation (class C)

- `patches/geos-blt-0001-cuda13-memoryClockRate.patch` (applied to the BLT
  submodule `src/cmake/blt`, `9ff77344`): BLT's CUDA runtime smoke test
  (`cmake/thirdparty/BLTCudaRuntimeSmokeTest.cpp` — the tiny program built with
  `ENABLE_TESTS=ON`, upstream's default) reads `cudaDeviceProp::memoryClockRate`,
  which CUDA 13 removed from the struct. The patch is the back-port of upstream
  LLNL/blt commit `38b46203` ("use cudaDeviceGetAttribute for memoryClockRate on
  CUDA >= 13"): `#if CUDART_VERSION >= 13000 cudaDeviceGetAttribute(&rate,
  cudaDevAttrMemoryClockRate, i) #else prop.memoryClockRate #endif`. Only the
  smoke-test program changes; no GEOS code, no kernel. The alternative --
  switching the tests off -- cascaded through `ENABLE_BENCHMARKS`, the GEOS
  `GEOS_ENABLE_TESTS`/LvArray `ENABLE_EXAMPLES` and HPCReact test switches and
  was abandoned so that the tree stays at upstream's defaults (the 100+ GEOS/LvArray
  unit-test executables are built but **not run** here -- see the status table).
- The conda Open MPI include directory (`$CONDA_PREFIX/include`, added by GEOS as
  `-isystem` from the MPI imported target) carries an unrelated 32-bit `metis.h`
  (2013) that shadowed the TPL's 64-bit one and broke ParMETIS' `static_assert(
  sizeof(idx_t) == 8)` in `VTKMeshGenerator`; the host-config puts the TPL
  METIS/ParMETIS include dirs first with `-I` in `CMAKE_CXX_FLAGS`/`CMAKE_CUDA_FLAGS`
  (`-I` is searched before every `-isystem`). No source change.
- Build time (`BUILD_INFO.txt`): GEOS 1718 s at `-j32` (the heaviest translation
  unit, `CompositionalMultiphaseBase.cpp`, takes ~26 min alone in cicc/ptxas);
  `cuobjdump` on `geosx`, `libHYPRE.a`, `libRAJA.a`: sm_100 only.

## Cases, launch, validation

`run.sh` (see its header): `beam` = `inputFiles/solidMechanics/beamBending_*.xml`
family; smoke `80x8x4` elements with the hypre AMG (GPU) or direct solver, strong
`160x16x8` (upstream's `beamBending_benchmark.xml` size, fixed over rank counts),
weak = mesh refined with the rank count (elements per GPU constant). Partitions
through GEOS' `-x -y -z` (product = ranks; the launcher gives one GPU per rank;
GEOS itself has no device-selection logic, so the per-rank `CUDA_VISIBLE_DEVICES`
wrapper is what binds it), output prefix `-o`.

`validate.sh` re-implements geos-ats's checks for this test (`beamBending.ats`:
`CurveCheckParameters(tolerance=[0.0002], script beamBending_curve.py)` and
`RestartcheckParameters(atol=1e-3, rtol=1e-7)`) in `geos_beam_check.py`, with the
metrics of geosPythonPackages `curve_check.py` / `restart_check.py`, and was
checked against upstream's published baseline **before** any GEOS run here:
[1] completion (exit 0, `displacement_history.hdf5` with the 10 output times,
finite, launcher audit N verified / 0 mismatch); [2] the geos-ats "script" curve
check, `||u - u_analytic||_2 / N <= 0.0002` over the whole (10 x 81 x 3) trace vs
`beamBending_curve.py` (Euler-Bernoulli, upstream's 1.043 factor) -- upstream's
own 80x8x4 baseline scores 1.376e-4, i.e. this tolerance sits just above the
mesh's discretisation error, whose per-time relative L-inf is 9.1e-4 (printed as
information; my first reading of "2e-4 relative per time" would have failed
upstream's own baseline and was replaced before the runs); [3] rank-count
independence of the AMG runs: N-GPU vs 1-GPU history (trace nodes matched by
reference position), relative L-inf <= 1e-4 (100x krylovTol; whether 1e-5 is met
is printed -- upstream's direct-solver 1- and 8-rank baselines agree to 1.5e-12)
and the geos-ats baseline metric <= 0.0002; [4] the shipped serial-direct-solver
smoke deck on 1 rank vs the public baseline `beamBending_smoke_01`
(`baseline_integratedTests-pr3994-17525-4ae3593.tar.gz`, the `.integrated_tests.yaml`
baseline of this commit): restart file at cycle 10, every dataset and attribute
with upstream's tolerances and default exclusions (`commandLine`, `schema`,
`globalToLocalMap`, `timeHistoryOutput*/restart`), plus its history vs the
baseline history (metric <= 0.0002, rel L-inf <= 1e-4).

The first restart comparison read the LvArray `__values__` datasets raw and reported
23 disagreements: a device build stores 2-D node/element fields (displacements,
stresses, `nodeList`, ...) in a different memory permutation than the host build that
produced the baseline (`__permutation__` differs for 12 of the 93 LvArrays), so the raw
datasets differ while the logical arrays are equal. geos-ats handles this with
`permute_array.py`; `geos_beam_check.py` now does the same (unit-checked against
upstream's own permutation test cases) and the validation was re-run (run ids in the
manifests; the first pass is kept in `validate.*.stdout` history only as the FAIL it
was). Three `LinearSolverParameters` values differ by construction --
`krylovMaxRestart` 100 vs 200, `amgCoarseningType` PMIS vs HMIS, `amgSmootherType`
l1jacobi vs l1sgs are GEOS's compile-time defaults for `GEOS_USE_HYPRE_DEVICE` builds
(`src/coreComponents/linearAlgebra/utilities/LinearSolverParameters.hpp`) -- they are
reported by the checker and not gating (the direct solver of the compared deck does not
use them; nothing else may differ).

### Results (2026-09-06, profile `cuda132-gcc142-ompi5010`, one rank per GPU)

| Run | GPUs | Mesh (C3D8) / partitions | Solver | geos-ats curve metric (tol 2e-4) | vs 1 GPU: rel L-inf / metric | GEOS run time | Audit |
|---|---|---|---|---|---|---|---|
| validate | 1 | 80x8x4 / 1x1x1 | GMRES + hypre AMG (device) | 1.3761e-4 | -- | 7.1 s | 1 verified |
| validate | 2 | 80x8x4 / 2x1x1 | same | 1.3761e-4 | 6.2e-9 / 6.6e-10 | -- | 2 verified, 0 mismatch |
| validate | 4 | 80x8x4 / 2x2x1 | same | 1.3761e-4 | 1.1e-8 / 8.4e-10 | -- | 4 verified, 0 mismatch |
| validate, shipped smoke deck | 1 | 80x8x4 / 1x1x1 | serial direct | 1.3761e-4; restart vs upstream baseline: 1191 arrays/attributes, 0 disagree, worst rel 3.6e-12; history vs baseline rel 5.1e-12 | -- | 6.5 s | 1 verified |
| strong (`beamBending_benchmark.xml` verbatim) | 1 / 2 / 4 | 160x16x8 / 1x1x1, 2x1x1, 2x2x1 | GMRES + AMG (device) | -- | -- | 6.7 / 10.9 / 13.3 s | all verified |
| weak (80x8x4 per GPU) | 1 / 2 / 4 | 80x8x4, 160x8x4, 160x16x4 | GMRES + AMG (device) | -- | -- | 9.5 / 14.2 / 8.7 s | all verified |

VALIDATED_PASS at 1/2/4 GPUs. Upstream's own baseline scores the same 1.3761e-4 on the
curve metric (the mesh's discretisation error; the tolerance 2e-4 sits just above it),
and upstream's 1- and 8-rank baselines agree to 1.5e-12 -- the same order as our
direct-solver run vs their baseline. The scaling numbers are completeness records
only: the official beam decks are far too small for a B200 (20 480 elements at most;
hypre setup, silo/HDF5 output and MPI latency dominate, more ranks cost more). Larger
meshes are one variable away (`HPCPERF_GEOS_WEAK_NX`, or an upstream
`performanceBenchmarks` case) but were not run so that the workflow stays the
official small one. Dry-runs 8/40/80 (strong + weak): planned as HYPOTHETICAL
(partitions 2x2x2 / 5x4x2 / 5x4x4; weak meshes up to 400x32x16); 3 ranks are refused
(80x8x4 cannot be partitioned). Raw material: `build/level3/geos/<profile>/run/`
(`stdout.log`, `displacement_history.hdf5`, restart files, `analytic_check.txt`,
`cross_rank_check.txt`, `restart_baseline_check.txt`, `history_baseline_check.txt`,
`run_manifest.txt`), `.deps/level3/geos/<profile>/install/BUILD_INFO.txt`.

### Unit tests (dependency probe, not a validation criterion of the beam workflow)

`ctest -j4` in `build/level3/geos/<profile>` on GPU 0 (upstream default `ENABLE_TESTS=ON`,
`PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe` for the tests' own `mpirun -np 1`
launches): **254 / 261 passed** in 351 s, log
`.deps/level3/geos/<profile>/logs/geos-ctest.log`. Failed: `testMath` (LvArray
`TestComplexMath/8.asinh`, `float` on the device: CUDA's `asinhf(5)` differs from the host
value by >= 1 float ulp, the test allows `epsilon`; CUDA documents up to 3 ulp), `testErrorHandling`
(`testYamlFileAssertOutput` aborts on purpose; prterun turns the abort into a non-zero test
exit), and five fluid-flow/well physics tests -- `testCompMultiphaseFlow`
(phase-mobility derivative check, analytical vs finite-difference error norm ~1.0),
`testCompMultiphaseFlowHybrid` (flux Jacobian, error norms 0.008-0.15),
`testThermalEstimatorProdWell`, `testThermalEstimatorInjWell`,
`testReservoirThermalSinglePhaseMSWells_RateInj` (`ExternalError` in the well solvers).
The last five are gross, not precision-level, and were not investigated: **the
compositional multiphase flow and well modules of this build are UNVERIFIED**; the
validated solid-mechanics workflow is not affected (its restart state matches upstream's
baseline to 3.6e-12).

<!-- hpcperf:source-section:begin -->
## Source distribution (frozen source artifact, scheme 3, 2026-09-10)

**RETIRED_FROM_DEFAULT_SUITE**: this application is no longer part of the Level 3 default suite (dependency redistribution/licensing constraints: ParMETIS 4.0.3 in the third-party dependency set, and the project decision to replace the application). No source artifact is staged or published for it; the code, provenance and historical results below stay in git as a record. The local materialized tree, if present, is research-only.

The application source is not in git and not read from `_upstream/`: `tools/prepare_benchmark.sh level3 geos` materializes the frozen source artifact (`<app>[-<variant>]-<source_version>.tar.zst`, found in the local content-addressed cache `.artifacts/sha256/` or downloaded from the immutable URL recorded in `provenance/source.lock*.yaml` once published; `--artifact FILE` for a local copy) into `src/` (+ `deps/`), the only source `build.sh`/`run.sh`/`validate.sh` use. Archive size + sha256 and `source_tree_sha256` are verified before anything is placed. Identity, patch series, licenses, redistribution status and the equivalence proof against the tree the results above were validated from are under `provenance/` (`source.lock*.yaml`, `patch_series*.txt`, `original_vs_baseline*.diff`, `LICENSES*.md`, `equivalence*.md`, `LOC*.md`); `benchmark.yaml` is the machine-readable contract (entries, inputs, references, identity). The benchmark does not prescribe which part of the source an optimization agent may modify; the integrity layer only protects the harness and the validation assets. Remote status: see `level3/SOURCE_ARTIFACTS.md`.

| variant | artifact | source version | compressed / uncompressed | entries | source_tree_sha256 | archive sha256 | upstream | patches (pre-applied) | redistribution | equivalence | remote | LOC app-owned / bundled deps / benchmark deps / tests / total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| - | `geos-hpcperf-l3-v1.tar.zst` | hpcperf-l3-v1 | 362.6 MB / 751.1 MB | 7026 | `208e8f98027e5a5f674ad3676272f9a4762d8aad59168da87821e8494563d741` | `cc5a6bfaf27504a200f46ca2f610dbca86a3a3e8a7753faa7bf7ca5dc8b309d1` | develop @ 2026-09-04 `b7a0f1330527` | geos-blt-0001-cuda13-memoryClockRate.patch, 0001-tpl-superlu_dist-url-hash-typo.patch, 0002-tpl-raja-vectorization-overridable.patch, 0003-tpl-hdf5-step-generator-and-build-command.patch | blocked | src: EQUIVALENT, deps/thirdPartyLibs: EQUIVALENT | REMOTE_ARTIFACT_UNPUBLISHED | 406925 / 330939 / 14602640 / 46778 / 15395169 |

LOC = cloc 2.06 code lines of the materialized tree (no blank/comment lines, documentation and data excluded); source-ownership categories from `provenance/source.lock*.yaml` (`source_scope`, descriptive metadata written at freeze time). Dependencies are counted per benchmark, so totals overlap across benchmarks that ship the same dependency. The validated results recorded above were produced from trees proven content-equivalent to this artifact (`provenance/equivalence*.md`); they are not re-run by the migration.
<!-- hpcperf:source-section:end -->
