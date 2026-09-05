# Level 3: Production / End-to-End HPC Applications

Level 3 integrates **full production applications** -- complete workflows, not
extracted kernels, not proxies, and never N independent replicas presented as
one distributed run. Its primary execution mode is **multi-GPU with a
user-selected GPU count**; single-GPU runs exist only for build smoke tests,
environment compatibility and basic correctness bring-up.

Branch `level3/full-apps-bringup` (from `main`): STEP 1-2 audit and build
strategy for all ten candidates ([APPLICATION_AUDIT.md](APPLICATION_AUDIT.md),
[BUILD_STRATEGY.md](BUILD_STRATEGY.md)), STEP 3-6 first-batch bring-up on
dgx003 (4x B200, CUDA 13.2.78, Slurm job 9552083, 2026-09-04). Nothing is
claimed validated beyond what the per-application README records for runs
that actually happened on this node.

## Status

| Application | Version | Build Strategy | CUDA Build | 1 GPU | 2 GPU | 4 GPU | HIP | Strong | Weak | Multi-node | Source Mod | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [LAMMPS](lammps/README.md) | stable_22Jul2025_update6 | NATIVE (bundled Kokkos 4.6.2) | OK, 220 s | PASS | PASS | PASS | untested | 16.4M atoms, 1/4 GPU run | 2.05M atoms/rank, 4 GPU run | BLOCKED/UNVERIFIED | A (derived deck) | FIRST_BATCH done |
| [SPARTA](sparta/README.md) | 27Aug2026 | NATIVE (bundled Kokkos 5.0.2) | OK, 579 s | PASS | PASS | PASS | untested | 10M particles, 1/4 GPU run | 1.25M particles/rank, 4 GPU run | BLOCKED/UNVERIFIED | A | FIRST_BATCH done |
| [WarpX](warpx/README.md) | 26.09 (+AMReX 26.09) | NATIVE (local AMReX source) | OK, 1219 s | PASS | PASS | PASS | untested | 33.6M particles, 1/4 GPU run | 4.2M particles/rank, 4 GPU run | BLOCKED/UNVERIFIED | A (derived inputs) | FIRST_BATCH done |
| [SPECFEM3D Cartesian](specfem3d/README.md) | v4.1.1 (+2 devel back-ports) | NATIVE (autotools, bundled SCOTCH) | OK, 21 s | PASS | PASS | PASS | untested | 165,888 elements, 1/4 GPU run | 165,888 elements/rank, 4 GPU run | BLOCKED/UNVERIFIED | B+C+D (18 lines, upstream devel) | FIRST_BATCH done |
| [nekRS](nekrs/README.md) | v26.0 | NATIVE (vendored OCCA/HYPRE) | OK, ~30 min | PASS | PASS | PASS | untested | 32,000 elements N=7, 1/4 GPU run | 8,000 elements/rank, 4 GPU run | BLOCKED/UNVERIFIED | B+C+D (39 lines; vendored HYPRE 2.32.0 vs CUDA 13) | FIRST_BATCH done |
| CP2K | v2026.2 | NATIVE+SPACK_DEPS | not started | -- | -- | -- | -- | H2O-N series (upstream) | QS_DM_LS NREP (upstream) | -- | -- | SECOND_BATCH (deps 3-6 h; DBCSR B200 patch) |
| Nyx | 26.09 | NATIVE (shared AMReX 26.09) | not started | -- | -- | -- | -- | Exec/Scaling (upstream) | RandomPerCell init | -- | -- | SECOND_BATCH |
| QMCPACK | v4.4.0 | NATIVE+SPACK_DEPS | not started | -- | -- | -- | -- | NiO S-series (download) | walkers_per_rank | -- | -- | SECOND_BATCH (needs Clang offload, Boost) |
| GEOS | 1.2.0 / develop | NATIVE (thirdPartyLibs superbuild) | not started | -- | -- | -- | -- | `<Benchmarks>` XML (upstream) | wellboreECP level01-06 | -- | -- | SECOND_BATCH (TPLs 4-5 h) |
| DFT-FE | 1.2.0 | NATIVE (install_DFTFE model) | not started | -- | -- | -- | -- | testsGPU systems | dftfe-benchmarks Mo series | -- | -- | SECOND_BATCH (deal.II stack 3-4 h) |

8/40/80-GPU shapes exist for every first-batch application as launcher
dry-runs only (`HPCPERF_DRY_RUN=1`): **DRY-RUN / UNVALIDATED**. Multi-node MPI
is BLOCKED/UNVERIFIED on this site. HIP recipes exist in every `build.sh` and
exit with a clear message here (no ROCm): **untested**.

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
analytic Ethier solution. No tolerance was loosened to obtain a PASS; no
precision or physics setting was changed.

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
