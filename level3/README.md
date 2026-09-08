# Level 3: Production / End-to-End HPC Applications

Level 3 integrates **full production applications** -- complete workflows, not
extracted kernels, not proxies, and never N independent replicas presented as
one distributed run. Its primary execution mode is **multi-GPU with a
user-selected GPU count**; single-GPU runs exist only for build smoke tests,
environment compatibility and basic correctness bring-up.

Branch `level3/second-batch-bringup` (contains the first-batch branch
`level3/full-apps-bringup`, from `main`): STEP 1-2 audit and build strategy for
all ten candidates ([APPLICATION_AUDIT.md](APPLICATION_AUDIT.md),
[BUILD_STRATEGY.md](BUILD_STRATEGY.md)), STEP 3-6 first-batch bring-up on
dgx003 (4x B200, CUDA 13.2.78, Slurm job 9552083, 2026-09-04), second batch
2026-09-05/06 ([SECOND_BATCH_STATUS.md](SECOND_BATCH_STATUS.md)), validator
rework and joint-HEAD regression 2026-09-07/08. Nothing is claimed validated
beyond what the per-application README records for runs that actually happened
on this node.

Correctness / reproducibility hardening (post-review, 2026-09-05):
[CORRECTNESS_FIXES.md](CORRECTNESS_FIXES.md) -- validators now capture the real
exit code and fail on timeout/missing/non-finite output, reject NaN/Inf, require
the expected steps/fields/traces, and write a per-run manifest; dry-runs can no
longer overwrite real results; fingerprints record ordered patch-content hashes;
Level 3 builds are isolated from Level 2 prefixes. CPU-only negative tests:
`level3/tools/tests/run_all.sh` (four groups: infra helpers, second-batch
checkers, the Nyx strict comparator + `validate.sh` chain, verdict classes; 88
checks on 2026-09-08). nekRS CUDA/dependency decision:
[nekrs/COMPATIBILITY.md](nekrs/COMPATIBILITY.md).

Verdict classes (`level3/tools/l3_verdict.py`): a `validate.sh` exit code is
0 PASS, 1 FAIL, 3 PENDING (Nyx heat/cool: `STATE_AND_PARTICLES_PASS;
I_R_CHECK_PENDING`), 4 UNSUPPORTED_LAYOUT (Nyx: legal but different box
layouts, not compared). Only 0 is a pass; 3 and 4 are their own classes in every
summary, never counted as PASS, never fed into a performance summary, and never
a reason for a queue to stop (`l3_run_recorded` in `l3_common.sh` records the
code and continues).

**Joint-HEAD regression (2026-09-07, code state `fc4d2a1`)**: after the shared
helper change (`HPCPERF_L3_RUN_SUBDIR`, run trees `run.regress-<sha>` that never
overwrite `run/`) and the Nyx validator rework, 27 `validate.sh` calls were made
on dgx003: LAMMPS, SPARTA, WarpX, SPECFEM3D smoke at 1/2/4 GPUs (12 PASS), nekRS
ethier hypregpu cimode 2 and 3 and cpucoarse cimode 2 at 1/2/4 GPUs (9 PASS), Nyx
MiniSB + LyA-adiabatic at 1/2/4 GPUs (3 PASS), Nyx LyA heat/cool at 1/2/4 GPUs
(3 PENDING). **24 PASS, 3 PENDING, 0 FAIL** -- not "all PASS". Launcher audits: 0
mismatch; four logs carry unverified ranks (SPARTA np1/np2, WarpX np1/np2 second
run: sub-0.3 s runs missed by the nvidia-smi sampling -- a binding-evidence gap,
not a correctness signal). CP2K, QMCPACK, DFT-FE and GEOS were **not** re-run on
the GPU in that round (their change is the one-line run-directory variable; the
status below is their 2026-09-05/06 result, their checkers are covered offline by
`test_l3_validators.sh`); a GPU regression of these four under the new directory
logic is a follow-up. Logs: `build/level3/regress-firstbatch-fc4d2a1/` and
`build/level3/nyx/regress-fc4d2a1/` (not committed).

## Status

| Application | Version | Build Strategy | CUDA Build | 1 GPU | 2 GPU | 4 GPU | HIP | Strong | Weak | Multi-node | Source Mod | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [LAMMPS](lammps/README.md) | stable_22Jul2025_update6 | NATIVE (bundled Kokkos 4.6.2) | OK, 220 s | PASS | PASS | PASS | untested | 16.4M atoms, 1/4 GPU run | 2.05M atoms/rank, 4 GPU run | BLOCKED/UNVERIFIED | A (derived deck) | FIRST_BATCH done |
| [SPARTA](sparta/README.md) | 27Aug2026 | NATIVE (bundled Kokkos 5.0.2) | OK, 579 s | PASS | PASS | PASS | untested | 10M particles, 1/4 GPU run | 1.25M particles/rank, 4 GPU run | BLOCKED/UNVERIFIED | A | FIRST_BATCH done |
| [WarpX](warpx/README.md) | 26.09 (+AMReX 26.09) | NATIVE (local AMReX source) | OK, 1219 s | PASS | PASS | PASS | untested | 33.6M particles, 1/4 GPU run | 4.2M particles/rank, 4 GPU run | BLOCKED/UNVERIFIED | A (derived inputs) | FIRST_BATCH done |
| [SPECFEM3D Cartesian](specfem3d/README.md) | v4.1.1 (+2 devel back-ports) | NATIVE (autotools, bundled SCOTCH) | OK, 21 s | PASS | PASS | PASS | untested | 165,888 elements, 1/4 GPU run | 165,888 elements/rank, 4 GPU run | BLOCKED/UNVERIFIED | B+C+D (18 lines, upstream devel) | FIRST_BATCH done |
| [nekRS](nekrs/README.md) | v26.0 | NATIVE (vendored OCCA/HYPRE) | OK (hypregpu ~30 min; cpucoarse 113 s) | PASS | PASS | PASS | untested | 32,000 elements N=7, 1/4 GPU run | 8,000 elements/rank, 4 GPU run | BLOCKED/UNVERIFIED | B+C+D, hypregpu variant (39 lines; vendored HYPRE 2.32.0 vs CUDA 13); cpucoarse variant 0 patches | FIRST_BATCH done; GPU-coarse verified (cimode 3), CPU-coarse candidate verified -- see [COMPATIBILITY.md](nekrs/COMPATIBILITY.md) |
| [Nyx](nyx/README.md) | 26.09 (+AMReX 26.09, SUNDIALS 7.2.1) | NATIVE (private AMReX per profile) | OK, 121 s (adiabatic) / 310 s (heatcool) | PASS (MiniSB, LyA-adiabatic); heat/cool PENDING | PASS; heat/cool PENDING | PASS; heat/cool PENDING | untested | LyA 64^3 adiabatic + synthetic 256^3, 1/2/4 GPU run (too small to scale) | synthetic 64^3/rank, 4 GPU run | BLOCKED/UNVERIFIED | A | SECOND_BATCH: adiabatic decks VALIDATED_PASS (re-run 2026-09-07); heat/cool STATE_AND_PARTICLES_PASS; **I_R_CHECK_PENDING** (not a pass) |
| [CP2K](cp2k/README.md) | v2026.2 (+DBCSR 2.10.0, upstream toolchain) | NATIVE + upstream toolchain (B200 back-port) | OK (toolchain hours; CP2K attempt 3) | PASS | PASS | PASS | untested | H2O-128 MD, 1/2/4 GPU run | H2O-32/64/128 size sweep, 1/2/4 GPU run | BLOCKED/UNVERIFIED | B (toolchain patch) + C | SECOND_BATCH done 2026-09-06 (historical result; not re-run on GPU 2026-09-07) |
| [QMCPACK](qmcpack/README.md) | v4.4.0 (+LLVM 23.1.0 offload) | NATIVE + private LLVM toolchain | OK, 982 s | PASS | PASS | PASS | untested | 256 walkers total, 1/2/4 GPU run (4096-walker series FAILED: cuSOLVER/device memory) | 256 walkers/GPU, 1/2/4 GPU run (1024/GPU FAILED, same cause) | BLOCKED/UNVERIFIED | A (+ C zlib flag) | SECOND_BATCH done 2026-09-06; walker-memory anomaly open, <= 300 walkers/GPU enforced (historical; not re-run 2026-09-07) |
| [DFT-FE](dftfe/README.md) | 1.2.0 (+deal.II 9.6.2, ELPA 2026.02.001) | NATIVE (install_DFTFE recipe) | OK | PASS | PASS | PASS | untested | LLZO 192 atoms, 1/2/4 GPU run | derived Al 32 atoms/GPU, synthetic size sweep | BLOCKED/UNVERIFIED | D (2-line isnan) + C | SECOND_BATCH done 2026-09-06 (historical; not re-run 2026-09-07); ELPA CPU cross-check SKIPPED |
| [GEOS](geos/README.md) | develop `b7a0f13305` + TPL `9b55672` | NATIVE (thirdPartyLibs superbuild) | OK, 1718 s | PASS (beam workflow) | PASS | PASS | untested | beamBending_benchmark 160x16x8, 1/2/4 GPU run (too small to scale) | refinement weak 80x8x4/GPU (same caveat) | BLOCKED/UNVERIFIED | B (3 TPL) + D (BLT back-port) + C | SECOND_BATCH done 2026-09-06 (historical; not re-run 2026-09-07); **5 compositional-flow/well unit tests FAIL on this build -- those modules UNVERIFIED** |

8/40/80-GPU shapes exist for every application as launcher dry-runs only
(`HPCPERF_DRY_RUN=1`): **DRY-RUN / UNVALIDATED** (no 40/80-GPU run has ever
happened). Multi-node MPI is BLOCKED/UNVERIFIED on this site. HIP recipes exist
in every `build.sh` and exit with a clear message here (no ROCm): **untested**.
Strong/weak entries above are runs that completed at 1/2/4 GPUs with the stated
decks; where the deck is too small for a B200 (Nyx, GEOS, CP2K/DFT-FE size
sweeps) they are completeness records, not scaling results; numerical
acceptance of the strong/weak runs beyond the smoke criteria is not claimed.

## Hard requirements (summary of the Level 3 policy)

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

## Correctness policy as applied

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

## Dependency isolation

Every application owns a private tree -- no shared Level 3 install root:

```
.deps/level3/<app>/{src,build,install,logs}     patched source copy (where needed), deps, install, logs
_upstream/level3/<Name>                        shallow upstream checkout at the selected tag (read-only)
build/level3/<app>/<cuda|hip>                  application build tree (+ run/ directories of run.sh)
```

Installs carry `.hpcperf-l3-fingerprint` (schema `l3-1`: application, upstream
commit, dependency versions, compiler, Fortran compiler, CUDA/ROCm, GPU arch,
MPI, CMake/configure options, GPU-aware-MPI setting, patch list, site profile,
Spack lock hash, container image hash, build time). A recorded fingerprint
that differs from the requested configuration fails fast
(`level3/tools/l3_common.sh`).

Spack, when chosen, uses one environment per application and backend
(`level3/envs/<app>/{cuda,rocm}/spack.yaml` + `spack.lock`); containers, when
chosen, commit the `.def`, build script, image SHA256 and README -- never the
`.sif`. Neither is used by the first batch (see BUILD_STRATEGY.md for why).

## Runtime

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

## Per-application layout

```
level3/<app>/
├── README.md        provenance, version/commit, license, LOC, build strategy, changes (A-D), execution model,
│                    inputs (smoke/strong/weak), validation, 1/2/4-GPU results, dry-runs, limitations
├── fetch.sh         shallow clone at the recorded tag/commit (no source trees committed)
├── build.sh         native build into .deps/level3/<app>, fingerprinted; HIP branch present, untested
├── run.sh           HPCPERF_GPUS + HPCPERF_SCALE_MODE aware, launched via the common launcher
├── validate.sh      upstream correctness mechanism, PASS/FAIL line, exit code
└── patches/         compatibility patches (classified, documented; SPECFEM3D, nekRS)
```

Inputs are upstream's own decks referenced from the read-only checkout;
derived decks (size, steps, topology, diagnostics) are written into the build
tree at run time and documented per application, so no upstream input file is
modified and nothing large is committed.
