# Level 3 second batch -- status (Nyx, CP2K, QMCPACK, DFT-FE, GEOS)

Branch `level3/second-batch-bringup` (worktree `HPC-Performance-AI-b2`, from the
first-batch checkpoint `366b72f`). Node dgx003 (4x B200, CUDA 13.2.78, conda
Open MPI 5.0.10, system GCC 14.2.1 / conda GCC 13.3.0), Slurm job 9552083.
Everything below was produced on that allocation between 2026-09-05 and
2026-09-06; nothing was pushed, no PR, no merge. Last update: **2026-09-06 05:25 UTC --
batch complete: all five applications VALIDATED_PASS at 1/2/4 GPUs; open items are listed
per application (QMCPACK walker-memory anomaly, GEOS flow/well unit tests, size of the
official GEOS/Nyx decks).**

States: `PLANNED` (not started) / `NOT_RUN` (deliberately not run, reason given) /
`BUILD_PASS` / `COMPLETED` (ran to completion, timings or plans recorded, no
correctness claim) / `VALIDATED_PASS` (validate.sh PASS) / `FAILED` / `BLOCKED` /
`UNVERIFIED`.

## Status table

| Application | Selected version / SHA | Build strategy | Compiler / Toolkit | Dependency probe | CUDA build | 1-GPU smoke | 2-GPU correctness | 4-GPU correctness | Strong | Weak / size sweep | 40/80 dry-run | Multi-node | HIP | Source changes | Blocker |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Nyx | 26.09 `e06eabc1` + external AMReX 26.09 `a52ca733` (+ SUNDIALS 7.2.1 for heat/cool) | NATIVE, per-profile AMReX/SUNDIALS | conda GCC 13.3.0 + nvcc 13.2.78, sm_100 | COMPLETED (AMReX archs = sm_100 verified; SUNDIALS CUDA examples 5/6, see README) | BUILD_PASS (2 profiles + CPU reference profiles) | VALIDATED_PASS (MiniSB, LyA-adiabatic, LyA heat/cool) | VALIDATED_PASS | VALIDATED_PASS | COMPLETED (LyA 64^3 adiabatic; synthetic 256^3; heat/cool LyA 64^3: 2.85/2.62/2.65 s on 1/2/4 GPUs -- too small to scale, fixed-cost dominated) | COMPLETED (synthetic 64^3/rank, labelled synthetic) | COMPLETED (8/40/80 planned or refused as designed) | BLOCKED (site) | NOT_RUN (no ROCm) | none (class A) | large ICs not public |
| CP2K | v2026.2 `67b5da87`; DBCSR 2.10.0; toolchain OpenBLAS 0.3.33 | NATIVE + upstream toolchain (B200 back-port patch) | system GCC 14.2.1 + nvcc 13.2.78, sm_100 | COMPLETED (DBCSR ctest 19/19 on 4 GPUs) | BUILD_PASS (attempt 3: toolchain OpenBLAS via rpath order; attempts 1-2 resolved BLAS to the conda pthreads OpenBLAS -- archived under `install/ATTEMPT-*`) | VALIDATED_PASS (attempt-3 binary: regtests within upstream tol, H2O-64 MD 10/10 SCF converged) | VALIDATED_PASS (max FORCE_EVAL diff 1.1e-11 Ha vs 1 GPU) | VALIDATED_PASS (8.6e-12 Ha) | COMPLETED (H2O-128, 10 MD steps: 119.8 / 90.8 / 60.0 s on 1/2/4 GPUs = 1.32x / 2.0x; 8 OpenMP threads per rank) | COMPLETED (size sweep, 32 waters per GPU: H2O-32 on 1 GPU 17.3 s, H2O-64 on 2 GPUs 32.9 s, H2O-128 on 4 GPUs 62.8 s -- the GPW cost per molecule grows with system size, so this is a size sweep, not an iso-efficiency series) | COMPLETED (8/40/80 strong planned; weak 40/80 REFUSED: no upstream H2O-1280/2560 deck) | BLOCKED (site) | NOT_RUN | toolchain patch (class B) + node adaptations (C) | none |
| QMCPACK | v4.4.0 `2601d62e`; LLVM 23.1.0, HDF5 1.14.5, Boost 1.90 | NATIVE + private LLVM offload toolchain | clang 23.1.0 (host) + nvcc 13.2.78, `QMC_GPU=openmp;cuda` sm_100 | COMPLETED (offload probe PASS 1/2/4 ranks; unit ctests 64/64; deterministic diamond ctests 526/526) | BUILD_PASS (982 s) | VALIDATED_PASS (DMC E = -21.8492 +- 0.0152 vs ref -21.844975 +- 0.02, -0.32 sigma) | VALIDATED_PASS (-21.8566 +- 0.0126; -0.58 sigma vs ref, 0.37 sigma vs 1-GPU) | VALIDATED_PASS (-21.8313 +- 0.0105; +0.68 sigma vs ref, 0.97 sigma vs 1-GPU) | FAILED at 4096 walkers (cuSOLVER INTERNAL_ERROR: device memory ~320 MB/walker exhausted, see README) -> COMPLETED with the verbatim 256-walker deck over 1/2/4 GPUs: 534 / 332 / 211 s (1.61x, 2.53x; the validation runs of the same deck: 316/254 s); DMC energies -21.8365 +- 0.0088 / -21.8511 +- 0.0100 Ha, upstream check_scalars pass | FAILED at 1024 walkers/GPU (same cause) -> COMPLETED with 256 walkers/GPU: 534 / 632 / 656 s on 1/2/4 GPUs (84 % / 81 % weak efficiency; 512 and 1024 walkers in total), DMC -21.8466 +- 0.0077 / -21.8433 +- 0.0034 Ha (error bars shrink as 1/sqrt(walkers) as they should), check_scalars pass | COMPLETED (8/40/80 planned, HYPOTHETICAL) | BLOCKED (site) | NOT_RUN | none (class A); HDF5 zlib discovery flag (C) | NiO datasets external |
| DFT-FE | 1.2.0 `7147faa5`; deal.II **9.6.2** (9.7.1 attempt failed: API removals), ELPA 2026.02.001 (sm_100 kernels), p4est 2.8.7 | NATIVE (install_DFTFE recipe transcribed) | system GCC 14.2.1 + nvcc 13.2.78, sm_100 | COMPLETED (ELPA GPU probe PASS: ELPA's analytic 1-stage and 2-stage GPU eigensolver tests on 1/2/4 GPUs, max eigenvalue error <= 7.3e-15 (tol 5e-14), eigenvector error <= 1.0e-11 (tol 6e-10), GPU timers present, audit 0 mismatch; two earlier probe versions mis-parsed -- recorded) | BUILD_PASS (deal.II 9.6.2 rebuild + 2-line `std::isnan` patch) | VALIDATED_PASS (al_md 32-atom BOMD, 4 steps, vs upstream's GPU reference: e0, MD energies, temperatures, forces identical at printed precision, d = 0.0) | VALIDATED_PASS (identical to the reference and to the 1-GPU run) | VALIDATED_PASS (identical) | COMPLETED (LLZO 192 atoms / 720 states: 295 / 163 / 96 s on 1/2/4 GPUs = 1.81x / 3.07x; E = -3579.26588980 Ha identical on 1/2/4 GPUs) | COMPLETED (derived Al supercells, 32 atoms per GPU: 32 / 64 / 128 atoms on 1/2/4 GPUs: 213 / 284 / 378 s -- KS-DFT cost grows superlinearly with atoms, SYNTHETIC series labelled as such) | COMPLETED (8/40/80 planned, HYPOTHETICAL) | BLOCKED (site) | NOT_RUN | 2-line `std::isnan` patch (D); p4est-setup.sh + ELPA configure adaptations (C) | ELPA CPU cross-check programs not built with the GPU configuration (SKIPPED, recorded) |
| GEOS | develop `b7a0f13305` (2026-09-04) + thirdPartyLibs `9b55672` (TPL 361-1070) | NATIVE (thirdPartyLibs superbuild + host-config) | system GCC 14.2.1 + nvcc 13.2.78, sm_100; hypre device (`ENABLE_HYPRE_DEVICE=CUDA`) | COMPLETED with findings (TPL superbuild: libHYPRE.a / libRAJA.a embed sm_100 only; GEOS/LvArray ctest on GPU 0: **254 / 261 passed** -- failed: `testMath` (device `asinhf` vs host within 1 float ulp), `testErrorHandling` (an abort-on-purpose test under `mpirun -np 1`/prterun), and **5 fluid-flow/well physics tests** -- `testCompMultiphaseFlow` (analytical vs numerical phase-mobility derivatives, error norm ~1.0), `testCompMultiphaseFlowHybrid` (flux Jacobian, 0.008-0.15), `testThermalEstimator{Prod,Inj}Well`, `testReservoirThermalSinglePhaseMSWells_RateInj` (ExternalError in the well solvers) -- cause not determined; these modules are UNVERIFIED on this stack, see section) | BUILD_PASS (1718 s at -j32; attempts: conda 32-bit metis.h shadowing the TPL one -> host-config include order; BLT CUDA runtime smoke test vs CUDA 13 -> back-port of LLNL/blt 38b46203) | VALIDATED_PASS (beam 80x8x4, hypre GMRES+AMG on device: geos-ats curve metric 1.376e-4 <= 2e-4 = upstream's own baseline value; shipped direct-solver deck vs the public restart baseline: 1191 arrays/attributes, 0 disagree, worst rel 3.6e-12; history vs baseline history rel 5e-12) | VALIDATED_PASS (curve metric 1.376e-4; vs 1 GPU rel L-inf 6.2e-9, metric 6.6e-10) | VALIDATED_PASS (curve metric 1.376e-4; vs 1 GPU rel L-inf 1.1e-8, metric 8.4e-10) | COMPLETED (beamBending_benchmark 160x16x8 = 20 480 C3D8, 10 quasi-static steps: GEOS run time 6.7 / 10.9 / 13.3 s on 1/2/4 GPUs -- the official beam decks are far too small for a B200; fixed hypre-setup/I/O cost dominates and more ranks cost more) | COMPLETED (refinement weak, 80x8x4 elements per GPU: 80x8x4 / 160x8x4 / 160x16x4 -> 9.5 / 14.2 / 8.7 s; same caveat) | COMPLETED (8/40/80 strong + weak planned, HYPOTHETICAL; partitions 2x2x2 / 5x4x2 / 5x4x4; 3 ranks refused: cannot partition 80x8x4) | BLOCKED (site) | NOT_RUN | 3 thirdPartyLibs build-system patches (B) + 1 BLT smoke-test back-port (D); host-config include order, ENABLE_HYPREDRV=OFF (C); no GEOS/LvArray/hypre source change | beam decks too small for GPU scaling (bigger official meshes = `HPCPERF_GEOS_WEAK_NX`); 5 compositional-flow / well unit tests FAIL on this GPU build (not the validated workflow; unresolved) |

## Per application

### Nyx -- `level3/nyx/` (commits 769482f, 1f7e14f)

Official decks (MiniSB `inputs.32 nyx.ppm_type=0`, LyA `inputs.rt.garuda`, LyA
heat/cool `inputs.rt`, 10 steps as in upstream's nightly GPU suite) on 1/2/4 GPUs;
criteria: completeness/finiteness, `fcompare -n 0 --rel_tol` at upstream's
tolerances (2e-10; 5e-5 for heat/cool with the integrator-rate diagnostic `I_R`
excluded and reported), DM particle identity tracking across rank counts
(`nyx_particle_compare.py` -- AMReX's own tool cannot compare different rank
counts), CPU-backend cross-reference (1e-8 pre-fixed), baryon mass conservation
1e-9. Strong/weak/dry-run numbers and the full derivation are in
`level3/nyx/README.md` (heat/cool LyA 64^3 strong: 2.85 / 2.62 / 2.65 s on 1/2/4
GPUs -- too small to scale; dry-runs 8 planned, 40 IMBALANCED, 80 refused as designed).
Nothing remaining.

### CP2K -- `level3/cp2k/` (commit b04db93)

Toolchain: `install_cp2k_toolchain.sh --gpu-ver=B200` with the back-port of
upstream's B200 commit `378b2fab` (DBCSR gets `GPU_ARCH_NUMBER_B200 100`,
`parameters_H100.json` reused as `parameters_B200.json`); DBCSR 2.10.0 test build
and its ctest suite (19/19, 4 ranks x 4 threads on 4 GPUs) before CP2K;
`CMAKE_CUDA_ARCHITECTURES=100`. Validation at 1/2/4 GPUs x 8 threads: adapted
subset of upstream's regression tests (Ar, H2O-geoopt, pyridine, H2-big-1,
H2-big-5) all within upstream tolerances (max |diff| 6.0e-14); H2O-64 GPW MD 10
steps: all 10 MD-step SCF cycles converged, GRID GPU tasks / DBCSR ACC device /
pw_gpu timers observed, per-step FORCE_EVAL energies agree with the 1-GPU run to
<= 8.6e-12 Ha (pre-fixed 1e-8; upstream's CPU 1e-10 also met). Documented
peculiarity: the upstream deck's initial ATOMIC-guess SCF does not converge
within MAX_SCF=50 and says so (`IGNORE_CONVERGENCE_FAILURE`); reported, not
hidden, negative-tested. GPU stages confirmed: DBCSR sparse multiply (ACC), GRID
collocate/integrate, PW FFT (`pw_gpu_*`); ScaLAPACK diagonalisation/Cholesky
remain CPU (as configured upstream without ELPA/cuSOLVERMp). Strong scaling
(H2O-128, 10 MD steps, toolchain OpenBLAS binary): 119.8 / 90.8 / 60.0 s on 1/2/4 GPUs
(1.32x, 2.0x -- the 64/128-water GPW MD is communication/latency heavy at 8 threads
per rank); size sweep H2O-32/64/128 on 1/2/4 GPUs: 17.3 / 32.9 / 62.8 s. All runs
record `blas_resolved=.../openblas-0.3.33/lib/libopenblas.so.0`.

### QMCPACK -- `level3/qmcpack/`

Toolchain: LLVM 23.1.0 built from source on local scratch (clang/lld + openmp/offload
runtimes with the `nvptx64-nvidia-cuda` GPU runtimes target; the official binary
release has no device runtime), offload probe (`toolchain/probe_offload.sh`,
`OMP_TARGET_OFFLOAD=MANDATORY`, target region not on the initial device, numerics
vs host, MPI 2/4 ranks through the launcher) PASS before any QMCPACK build. HDF5
1.14.5 parallel, Boost 1.90 headers, OpenBLAS 0.3.30 private. QMCPACK 4.4.0
`QMC_GPU="openmp;cuda" QMC_GPU_ARCHS=sm_100 QMC_MPI=ON BUILD_UNIT_TESTS=ON`, cuobjdump
archs sm_100. Validation design (`validate.sh`): upstream unit ctests (64/64 PASS)
and the deterministic diamond-case ctests (526/526 PASS), then
`tests/solids/diamondC_2x1x1_pp/qmc_short_vmcbatch_dmcbatch.in.xml` verbatim on N
GPUs with upstream's own `check_scalars.py` criterion (DMC total energy
-21.844975 +- 0.02 Ha, 3 sigma, 2 equilibration blocks) plus completeness, device
code-path evidence ("Running OpenMP offload code path", device memory allocated
via the offload runtime) and, for N > 1, statistical consistency with the 1-GPU
run (3 sqrt(sigma_1^2 + sigma_N^2)). Results (256 walkers, 8 crowds/rank):
1 GPU -21.8492 +- 0.0152 Ha (534 s), 2 GPUs -21.8566 +- 0.0126 (316 s, 128 walkers/rank),
4 GPUs -21.8313 +- 0.0105 (254 s, 64 walkers/rank); all within upstream's 3-sigma window
and mutually consistent (<= 0.97 sigma); launcher audit N verified / 0 mismatch at every
N. VALIDATED_PASS 1/2/4. Strong = fixed `total_walkers`, weak = `walkers_per_rank`:
the first series (4096 total / 1024 per rank) FAILED in cuSOLVER on every rank --
device memory exhausted at ~320 MB per walker although QMCPACK's allocators report
~27 MiB (open issue, documented in the README, not worked around; run.sh now refuses
> 300 walkers/GPU unless forced). Re-defined series with the verbatim population:
strong (256 walkers total) 534 / 332 / 211 s on 1/2/4 GPUs (1.61x, 2.53x -- 64
walkers per GPU at 4 GPUs is far below what a B200 needs), weak (256 walkers per GPU)
534 / 632 / 656 s (84 % / 81 %); every run's DMC energy passes upstream's
`check_scalars.py` window and the launcher audit (`qmc_summary.txt` in each run
dir). NiO performance decks not used (data only via anl.box.com).

### DFT-FE -- `level3/dftfe/`

Dependency chain: OpenBLAS, ScaLAPACK 2.2.2, libxc 7.0.0, spglib, ALGLIB, p4est 2.8.7
(three adaptations of dftfe's Cray-oriented `p4est-setup.sh`: MPI wrappers,
`LIBS=-lm`, 2.8.7 header location), Kokkos 4.6.00 Serial, **deal.II 9.6.2** (attempt 1
with 9.7.1 -- the version the current install_DFTFE recipe pairs with dftfe
*develop* -- fails: 9.7 removed three APIs release 1.2.0 uses; 9.6.2 keeps them
deprecated), ELPA 2026.02.001 with sm_100 kernels (`-march=native` for its AVX-512
probe, ScaLAPACK/OpenBLAS paths in `LDFLAGS` for its cublas check), DFT-FE real with
a 2-line `std::isnan` patch (unqualified `isnan` in a template, GCC 14). Validation
(`validate.sh`, `dftfe_check.py`): upstream GPU regression deck `Input_MD_0.prm`
(32-atom Al BOMD, 4 steps, USE GPU) on 1/2/4 GPUs vs upstream's own GPU reference
`accuracyBenchmarks/output_MD_0` under pre-fixed tolerances (1e-5 Ha ground state,
2e-5 Ha per MD step, 0.1 K, 2e-5 Ha/Bohr forces): **every quantity identical at the
printed precision** (d = 0.0) at 1, 2 and 4 GPUs, and 2/4-GPU runs identical to the
1-GPU run; launcher audit N verified / 0 mismatch. Criterion [0], the independent
ELPA GPU-kernel probe (`elpa_probe.sh`: ELPA's own `validate_*_gpu_analytic` programs,
1-stage and 2-stage solvers, na=2000 nev=1000, 1/2/4 GPUs): PASS on all six runs,
max eigenvalue error 2.0e-15..7.3e-15 (ELPA's limit 5e-14), max eigenvector error
1.5e-12..1.0e-11 (limit 6e-10), exit 0, ELPA's GPU timers in the output, audit 0
mismatch. Two probe versions before it were wrong and are recorded (wrapper-script
names; then the residual/orthogonality lines of ELPA's *random-matrix* programs, which
the analytic programs do not print -- the empty grep aborted the script under
`pipefail`, so the validations of 05:01-05:03 UTC FAILED on [0] only and were re-run at
05:04-05:12 UTC with the corrected probe: VALIDATED_PASS 1/2/4). Strong (LLZO, 192
atoms, 720 states, ELPA): 295 / 163 / 96 s on 1/2/4 GPUs (1.81x / 3.07x), ground-state
energy -3579.26588980 Ha identical at every N. Weak (derived Al supercells, 32 atoms per
GPU; synthetic, same construction as upstream's dftfe-benchmarks Mo series): 32 / 64 /
128 atoms in 213 / 284 / 378 s -- the KS-DFT cost per atom grows with system size
(more states x more DOFs), so this is a size sweep at constant atoms/GPU, not an
iso-efficiency series. Dry-runs 8/40/80 planned (HYPOTHETICAL). ELPA's CPU test
programs are not built with the GPU configuration, so the CPU cross-check is SKIPPED
(recorded in the probe file).

### GEOS -- `level3/geos/`

TPL superbuild COMPLETED after three build-system fixes to thirdPartyLibs
(superlu_dist hash typo; RAJA vectorization layer not compilable by nvcc 13.2 +
GCC 14/x86-64-v3 -> `RAJA_ENABLE_VECTORIZATION=OFF` as upstream does for ROCm;
hdf5 step generator) and node adaptations (`config-build.py` deletes existing
build trees; hypredrive step lacks Umpire's include path -> `ENABLE_HYPREDRV=OFF`,
a GEOS-documented option, the beam workflow does not use it). GEOS build: the conda
MPI include dir (added by GEOS as `-isystem` ahead of the TPLs) carries an unrelated
32-bit `metis.h` that broke the 64-bit ParMETIS static assertion -> TPL
METIS/ParMETIS include dirs first (`-I`) in the host-config; BLT 0.6.2's CUDA
runtime smoke test uses `cudaDeviceProp::memoryClockRate` (removed in CUDA 13) ->
back-port of upstream BLT commit `38b46203` (`patches/geos-blt-0001`), tests kept
at upstream's default ON. "CUDA 13 blocked" upstream is a Spack/uberenv `raja ^cuda@13:`
conflict only; the native superbuild has no such check. GEOS built in 1718 s at
-j32 (BUILD_PASS; `geosx`, `libHYPRE.a`, `libRAJA.a` embed sm_100 only).

Validation (`validate.sh`, criteria = geos-ats's own checks for `beamBending`,
re-implemented with the metrics of geosPythonPackages `curve_check.py` /
`restart_check.py` / `permute_array.py` and verified against upstream's published
baseline before the GEOS runs): the derived `beamBending_benchmark` deck at the smoke
mesh (80x8x4, GMRES + hypre AMG **on the device**) on 1/2/4 GPUs -- geos-ats curve
metric `||u - u_analytic||_2 / N` = 1.3761e-4 at every N (tolerance 2e-4; upstream's own
baseline scores the same 1.3761e-4, i.e. the metric measures the mesh's discretisation
error, which the partitioning does not change); 2/4-GPU histories vs the 1-GPU history:
rel L-inf 6.2e-9 / 1.1e-8 (gate 1e-4, the 1e-5 information threshold met), geos-ats
metric 6.6e-10 / 8.4e-10; the shipped serial-direct-solver smoke deck on 1 rank vs the
public integrated-test baseline `beamBending_smoke_01` (tarball
`pr3994-17525-4ae3593`, sha256 `de62cec7...`): restart file at cycle 10, 1191
arrays/datasets/attributes under upstream's atol 1e-3 / rtol 1e-7 and default
exclusions, **0 disagree, worst relative difference 3.6e-12** (`maxForce`); its
history vs the baseline history rel 5.1e-12. Launcher audit N verified / 0 mismatch at
every N. VALIDATED_PASS 1/2/4. Two checker corrections were needed before this result
and are recorded in the README: (1) the per-time relative L-inf reading of the 2e-4
tolerance would have failed upstream's own baseline (9.1e-4) -- replaced by the geos-ats
metric before the first run; (2) the first restart comparison read the LvArray
`__values__` datasets raw: a device build stores 2-D node/element fields in another
memory permutation than the CPU build that produced the baseline (12 of 93 LvArrays), so
the raw datasets differ while the arrays are equal -- fixed with geos-ats's permutation
handling and re-validated (run ids in the run manifests). Three `LinearSolverParameters`
values (`krylovMaxRestart` 100 vs 200, `amgCoarseningType` PMIS vs HMIS,
`amgSmootherType` l1jacobi vs l1sgs) are GEOS's compile-time GPU-build defaults
(`LinearSolverParameters.hpp`) and are reported, not gating (unused by the direct
solver). Scaling: strong (`beamBending_benchmark.xml` verbatim, 160x16x8) 6.7 / 10.9 /
13.3 s GEOS run time on 1/2/4 GPUs; refinement-weak (80x8x4 per GPU) 9.5 / 14.2 / 8.7 s
-- the official beam decks are far too small for a B200 (hypre setup and silo/HDF5
output dominate), so these are completeness records, not scaling results; larger
meshes are one variable away (`HPCPERF_GEOS_WEAK_NX`) but were not run to keep the
"official small workflow" definition. Dry-runs 8/40/80 strong + weak planned
(partitions 2x2x2, 5x4x2, 5x4x4); 3 ranks refused (cannot partition 80x8x4).

Unit tests (upstream default `ENABLE_TESTS=ON`, `ctest -j4` on GPU 0, 351 s, log
`.deps/level3/geos/<profile>/logs/geos-ctest.log`): **254 of 261 passed**. The seven
failures, characterised from the log: (1) `testMath` -- LvArray's
`TestComplexMath/8.asinh` for `float` on the device: CUDA's `asinhf(5)` differs from the
host value by at least one float ulp while the test allows `epsilon` (CUDA documents up
to 3 ulp for `asinhf`); a libm-precision test, no GEOS code path; (2) `testErrorHandling`
-- `testYamlFileAssertOutput` deliberately triggers `abort()`; under `mpirun -np 1`
prterun reports the abort and the test exits non-zero (test-harness/launcher
interaction, the preceding sub-tests pass); (3)-(7) **five fluid-flow and well physics
tests fail on this GPU build**: `testCompMultiphaseFlow`
(`derivativeNumericalCheck_phaseMobility`: analytical vs finite-difference derivatives
disagree grossly, e.g. 653.7 vs -3.67, error norm ~1.0 against a 0.05 limit),
`testCompMultiphaseFlowHybrid` (`jacobianNumericalCheck_flux`, error norms 0.008-0.15
against 0.005), `testThermalEstimatorProdWell`, `testThermalEstimatorInjWell`,
`testReservoirThermalSinglePhaseMSWells_RateInj` (GEOS `ExternalError` raised in the
well solvers after warnings in `WellControls.cpp:1229` / `SinglePhaseWell.cpp:205`).
These are not precision-level differences and were not investigated further (candidates:
nvcc 13.2 + GCC 14 host code generation, RAJA 2026.07.0 on sm_100, or tests that upstream
does not run on GPU); **the compositional multiphase flow and well modules of this GEOS
build must be treated as UNVERIFIED**. The validated beam workflow (solid mechanics, hypre
AMG on the device) is unaffected: its restart state agrees with upstream's public baseline
to 3.6e-12.

## Cross-cutting

- Launcher: `level2/tools/hpcperf_mpi_launch.sh` unchanged (`HPCPERF_RUNTIME_DIR`
  semantics kept); one rank per GPU, per-rank `CUDA_VISIBLE_DEVICES` wrapper,
  nvidia-smi audit (verified / unverified / mismatch) in every run log; 8/40/80 GPUs
  only as HYPOTHETICAL dry-runs (`HPCPERF_DRY_RUN=1 HPCPERF_NODES=N/4`), multi-node
  BLOCKED/UNVERIFIED on this site.
- Shared infra changes (committed separately): `l3_paths_profile`, `l3_version_mm`,
  `l3_clean_conda_build_env`, real-grep guard, cuobjdump-based backend check for
  statically linked cudart (9327c26, 0b9daaf); validator negative tests
  `level3/tools/tests/test_l3_validators.sh` (352cbb3).
- Private Toolkit/compiler exceptions: none for the CUDA Toolkit (13.2.78
  everywhere); private compilers: LLVM 23.1.0 for QMCPACK (required by upstream's
  GPU path), system GCC 14.2.1 (CP2K, DFT-FE, GEOS: one compiler for C/C++/Fortran)
  vs conda GCC 13.3.0 (Nyx, first batch). No `/usr/local/cuda` link or driver touched.

## Raw material

- Builds/logs/fingerprints: `.deps/level3/<app>/<profile>/{logs,install/BUILD_INFO.txt,install/.hpcperf-l3-fingerprint}`
  (LLVM/CP2K toolchain sources on `/tmp/hpcperf-l3-b2-scratch/`, logs copied under `.deps`).
- Runs: `build/level3/<app>/<profile>/run/<case>.<mode>.np<N>[.t<T>]/` with
  `run_manifest.txt` (run_id, exit code, binary/deck/input hashes, fingerprint
  hash, GPU evidence), full stdout/stderr, validator summaries
  (`validate.*.stdout`, `md_summary.txt`, `qmc_summary.txt`, `check_vs_*.txt`);
  dry-runs under `.dryrun/`.
- Patches: `level3/<app>/patches/*.patch` (header: source, rationale, conditions, impact, verification).

## Local commits on `level3/second-batch-bringup` (no push)

366b72f first-batch checkpoint; tools 9327c26 (per-profile paths, static-cudart
backend check), 0b9daaf (conda build variables cleared, real grep), 32e25e9 (foreign
include/library search paths cleared), 352cbb3 (validator negative tests, 17/17);
Nyx 769482f, 1f7e14f, c7b90f9; CP2K b04db93, 7408614 (toolchain OpenBLAS relink +
BLAS guard); QMCPACK 166657e, 1e5d00d (population guard, 256-walker series); DFT-FE
62f48e5, e91119c (ELPA probe fixed, validation 1/2/4 PASS); GEOS e354870 (build +
beam validation), fbaa962 (unit-test probe findings); documentation commit (this
file, `APPLICATION_AUDIT.md`, `BUILD_STRATEGY.md`) last. Nothing pushed, no PR, no
merge; `main` untouched.
