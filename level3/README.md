# Level 3: Full HPC Applications

Level 3 runs **complete scientific applications** on their own inputs -- not extracted kernels, not proxy
apps. Each benchmark is the upstream application at a pinned commit, its real build system, an official (or
officially derived) input case, and a correctness check taken from the application's own validation mechanism
wherever one exists.

The benchmarks are **multi-GPU by construction**: you choose the number of GPUs explicitly and the harness
uses exactly that many, one MPI rank per GPU unless an application documents otherwise. A single GPU is used
for deployment checks and small correctness cases. Multi-node is the design goal, **not** a validated state on
the machine these results come from.

Application source is **not stored in this repository**. Each benchmark carries the metadata of a frozen
source artifact (upstream commit, patch series, archive SHA-256, source-tree hash, licenses) and
`tools/prepare_benchmark.sh` restores the complete `src/` and `deps/` trees before any build or LLM
optimization run begins.

## Current availability (2026-09-15)

| | state |
|---|---|
| Source artifacts | **Published** on 2026-09-12 as the prerelease [`level3-source-hpcperf-l3-v1-rc1`](https://github.com/Deep-Learning-Profiling-Tools/HPC-Performance-AI/releases/tag/level3-source-hpcperf-l3-v1-rc1): 11 archives, `SHA256SUMS`, one `SOURCE_MANIFEST` per artifact and the release plan. Every archive was downloaded anonymously, with no credentials, and verified against the hashes the locks already carried. `prepare_benchmark.sh` now downloads without `--artifact`. |
| Code | In `main` (PR #5, source freeze, merged 2026-09-14; PR #8, backend/profile isolation of generated state, merged 2026-09-15). A normal clone of `main` contains the harness, the locks and the tools; the application source trees are **external** GitHub Release assets that `tools/prepare_benchmark.sh level3 <app>` downloads, verifies against the lock and unpacks into `level3/<app>/{src,deps}`. The release tag `level3-source-hpcperf-l3-v1-rc1` is the artifacts' identity (the commit they were cut from, whose locks still say unpublished), **not** the recommended code checkout -- use `main`. Generated build/install/run state is isolated per backend/profile ([PROFILE_ISOLATION.md](PROFILE_ISOLATION.md)). |
| Validated hardware | One node, 4 × NVIDIA B200 (CUDA 13.2.78, driver 595.58.03), conda GCC 13.3.0, Open MPI 5.0.10. **CUDA backend only.** |
| Scale | Single node, 1/2/4 GPUs. 8/40/80 GPUs exist as dry-run plans only. Multi-node is **unverified/blocked** on this site. |
| HIP / ROCm | Build path present in the scripts, **never executed** (no ROCm here). |
| Workspace coverage | The agent-workspace build/edit loop has been exercised end to end for **LAMMPS only**; other applications carry materialization and historical build/validation evidence. See [WORKSPACE_EVIDENCE.md](WORKSPACE_EVIDENCE.md). |
| Known limitations | [Section below](#known-limitations) and [WORKSPACE_EVIDENCE.md](WORKSPACE_EVIDENCE.md); Nyx `I_R` is PENDING, GEOS is retired. |
| ExaCA release acceptance | Put **ON_HOLD** on 2026-09-11 when 22 of 52 upstream unit tests failed, and **restored** the same day after every one of those 22 failures was attributed with before/after evidence and upstream's own small full-application cases passed at 1/2/4 real GPUs. The attribution matrix contains **no application-class failure**, which is a statement about these 22 failures in this environment and **not** a claim that ExaCA is defect-free here or anywhere else. Unmodified upstream tests still fail under CTest's own launcher; the fixture-only test patches that make them pass are in git (`exaca/patches/upstream-tests/`) and are deliberately **not** part of the frozen source artifact. [exaca/README.md](exaca/README.md) |

Development history (how these applications were brought up, what failed and why) lives in
[APPLICATION_AUDIT.md](APPLICATION_AUDIT.md), [BUILD_STRATEGY.md](BUILD_STRATEGY.md),
[CORRECTNESS_FIXES.md](CORRECTNESS_FIXES.md) and [SECOND_BATCH_STATUS.md](SECOND_BATCH_STATUS.md).
The project policy for a Level 3 benchmark (what counts as an application, how correctness criteria are
chosen, per-application dependency isolation, the launcher contract and the directory conventions) is in
[BUILD_STRATEGY.md](BUILD_STRATEGY.md#level-3-policy-correctness-policy-isolation-runtime-and-per-application-layout).

## Application catalog

Ten applications, eleven source artifacts (nekRS ships two variants). The workload column names the **case that
is actually run and validated**, which is a small part of what each application can do.

| Application | Frozen version | Selected workload / motif | App-owned code LOC | Details |
|---|---|---|---|---|
| [LAMMPS](lammps/) | `stable_22Jul2025_update6` | Lennard-Jones molecular dynamics (`bench/in.lj`), KOKKOS/CUDA package. No long-range electrostatics (PPPM) is exercised. | 852,527 | [README](lammps/README.md) |
| [SPARTA](sparta/) | `27Aug2026` | Direct simulation Monte Carlo: collisional flow (`bench/in.collide`), particle move/collide/sort. | 131,181 | [README](sparta/README.md) |
| [WarpX](warpx/) | `26.09` (+ AMReX 26.09) | Electromagnetic particle-in-cell, Yee FDTD solver: `uniform_plasma` + the analytic `langmuir_multi` case. The FFT/PSATD solver path is not exercised. | 112,459 | [README](warpx/README.md) |
| [SPECFEM3D Cartesian](specfem3d/) | `v4.1.1` | Spectral-element seismic wave propagation, `homogeneous_halfspace` mesher → database → solver workflow. No full-waveform inversion. | 142,516 | [README](specfem3d/README.md) |
| [nekRS](nekrs/) | `v26.0` (2 variants) | Spectral-element CFD, `ethier` case with upstream `--cimode` checks. Variants: `hypregpu` (GPU coarse solver, 3 HYPRE CUDA-13 patches) and `cpucoarse` (exact upstream). | 53,131 | [README](nekrs/README.md) |
| [Nyx](nyx/) | `26.09` (+ AMReX) | Cosmological N-body + hydrodynamics: `MiniSB` and `LyA-adiabatic` decks. The `LyA` heat/cool deck is **PENDING**, not a pass. | 32,330 | [README](nyx/README.md) |
| [CP2K](cp2k/) | `v2026.2` | Gaussian-plane-wave DFT: `H2O-64` SCF + Born-Oppenheimer MD, plus a subset of upstream regtests. | 1,085,842 | [README](cp2k/README.md) |
| [QMCPACK](qmcpack/) | `v4.4.0` | Quantum Monte Carlo: `diamondC_2x1x1_pp` VMC + DMC batched drivers (walker count limited, see the app README). | 337,840 | [README](qmcpack/README.md) |
| [DFT-FE](dftfe/) | `1.2.0` | Finite-element DFT: 32-atom Al Born-Oppenheimer MD regression deck vs upstream's GPU reference. | 107,718 | [README](dftfe/README.md) |
| [ExaCA](exaca/) | `2.1.0` | Cellular-automaton solidification / grain growth: directional-solidification case. Finch coupling is **not** exercised. | 6,512 | [README](exaca/README.md) |

LOC = `cloc` **code** lines of the frozen application-owned source (no blank or comment lines; documentation,
examples and data not counted), from each benchmark's `provenance/LOC.json`. The full breakdown (bundled
dependencies, benchmark-specific dependencies, tests, total materialized) is in
[APPLICATION_AUDIT.md](APPLICATION_AUDIT.md) and the per-application `provenance/LOC.md`. Total materialized
counts include dependencies and therefore overlap across benchmarks that ship the same one (AMReX in WarpX and
Nyx). Neither figure is a diff size: the line count of the integration PR is harness, not benchmark LOC, and
none of these numbers says what an optimization agent may modify.

**Retired:** [GEOS](geos/) was removed from the default suite (a third-party dependency, ParMETIS 4.0.3, may
not be redistributed, and the application was replaced by ExaCA). Its code, provenance and historical results
remain in the tree; no artifact is staged or published for it and it is not part of any download or count.

## Prerequisites

**To download and unpack source artifacts** (`tools/prepare_benchmark.sh`): Python 3 with PyYAML, `zstd` on
`PATH`, and network access (or a local artifact file). Nothing else -- no compiler and no GPU.

**To build**: a C/C++/Fortran toolchain, CUDA (or ROCm) matching your GPUs, an MPI implementation, CMake or
GNU Make. Versions differ per application: several need a specific host-compiler/CUDA combination, CP2K builds
its own dependency toolchain, and QMCPACK needs a private LLVM offload compiler. The per-application README
and `benchmark.yaml` (`environment_profile`, `dependency_installs`) state what each one expects.

**To run**: a real GPU allocation (Slurm or equivalent), a working NVIDIA driver, enough CPU cores and host
memory for the rank count, and a site MPI transport. Obtain an allocation first -- these are full applications
and must not be started on a login node.

The repository's `./setup_env.sh` bootstraps a project-local conda environment (compilers, MPI, CMake, Python,
`cloc`) into `.conda_env`/`.tools` and `source hpcperf_env.sh` loads it in each shell. It does **not** install
the NVIDIA driver, the CUDA Toolkit, ROCm, or any Level 3 application-specific toolchain (LLVM for QMCPACK,
the CP2K dependency toolchain, deal.II for DFT-FE): those are built by the application's own `build.sh` or
expected from the system, as documented per application.

Site profiles (`HPCPERF_SITE_PROFILE`) encode machine-specific launch settings. The profile used for these
results is single-node and its transport settings are not a cross-node configuration.

## Quick start

All commands are run **from the repository root**, after `./setup_env.sh` (once) and
`source hpcperf_env.sh` (each shell), inside a GPU allocation.

**A. From `main`** (the code and the locks are in `main`; the source artifacts are release assets). Clone,
and `prepare` downloads the artifact from the release and verifies it against the lock. This is the path
that was exercised anonymously, from a clean clone with an empty cache. Do **not** check out the release tag
`level3-source-hpcperf-l3-v1-rc1` for this: it marks the commit the artifacts were built from, before the
locks received their download URLs, so a plain `prepare` there reports the artifact as unpublished.

```bash
git clone https://github.com/Deep-Learning-Profiling-Tools/HPC-Performance-AI.git
cd HPC-Performance-AI
source hpcperf_env.sh

tools/prepare_benchmark.sh level3 lammps          # downloads, verifies and unpacks the source artifact
level3/lammps/build.sh CUDA
HPCPERF_GPUS=2 HPCPERF_SCALE_MODE=smoke level3/lammps/validate.sh CUDA
```

**B. Offline or from a local copy**: supply the artifact as a file you obtained separately. `prepare` verifies
it exactly as it verifies a download, so an air-gapped machine needs no network at all.

```bash
git clone https://github.com/Deep-Learning-Profiling-Tools/HPC-Performance-AI.git
cd HPC-Performance-AI
source hpcperf_env.sh

tools/prepare_benchmark.sh level3 lammps --artifact /path/to/your/lammps-hpcperf-l3-v1.tar.zst
level3/lammps/build.sh CUDA
HPCPERF_GPUS=2 HPCPERF_SCALE_MODE=smoke level3/lammps/validate.sh CUDA
```

`validate.sh` **runs the case itself** (it invokes `run.sh` with the validated settings) and prints one
verdict line; you do not run `run.sh` separately for validation. Use `run.sh` directly only for performance
or scaling runs. There is no `prepare-all` command: prepare each benchmark you intend to use.

For nekRS, choose the variant once and use it for every step:

```bash
tools/prepare_benchmark.sh level3 nekrs --variant hypregpu      # add --artifact FILE to use a local copy
level3/nekrs/build.sh CUDA
HPCPERF_GPUS=2 level3/nekrs/validate.sh CUDA
```

## Layout: one source tree, generated state per backend/profile

Each benchmark has exactly one frozen source tree, `level3/<app>/src/` (+ `deps/`), materialized from its
artifact and independent of the GPU backend -- it is never copied per backend. Everything a build or run
generates belongs to one backend/profile and is never shared between profiles:

```
level3/lammps/src/                                 the ONE frozen source tree (backend-independent)
build/level3/lammps/cuda/                          CUDA application build tree + run*/ result directories
.deps/level3/lammps/cuda/{install,logs,build,src,cache}   CUDA install (+ .hpcperf-l3-fingerprint), logs, dependency
                                                   builds, build-side source copy (in-tree builds), caches
build/level3/lammps/hip/, .deps/level3/lammps/hip/ the same for a HIP build (path present; HIP is untested here)
.deps/level3/cp2k/cuda132-gcc142-ompi5010/         a toolchain-identity profile (second batch); nekRS uses
                                                   <variant>.<backend>: .deps/level3/nekrs/hypregpu.cuda/
```

The profile always names its backend; `HPCPERF_<APP>_PROFILE` may rename it but a name that contradicts the
requested backend is refused. `run.sh` uses only the binary and fingerprint of the profile it derives
itself (the same derivation as `build.sh`); there is no fallback to another install. Backend separation
applies to generated state, not to source duplication -- the same principle Level 2 follows with
`build/level2/<app>/<cuda|hip>`. Details and the per-application matrix: [PROFILE_ISOLATION.md](PROFILE_ISOLATION.md).

## GPU selection and workload sizes

| Variable | Meaning |
|---|---|
| `HPCPERF_GPUS=N\|all` | Number of GPUs to use, one MPI rank per GPU by default. `N` means exactly `N`, even when the allocation holds more. `all` means every GPU of **your allocation**, not of the machine. |
| `HPCPERF_NODES`, `HPCPERF_GPUS_PER_NODE` | Select a sub-shape of the allocation. They cannot create resources you were not allocated; larger values are accepted only in a dry run, as a hypothetical plan. |
| `HPCPERF_CPUS_PER_RANK` | Host threads per rank (applications that use OpenMP on the host). |
| `HPCPERF_SCALE_MODE` | `smoke` (deployment and correctness), `strong` (one fixed global problem split over the ranks), `weak` (fixed work per GPU). |
| `HPCPERF_SITE_PROFILE` | Launch profile for the machine. |
| `HPCPERF_DRY_RUN=1` | Print the plan (ranks, decomposition, command) without executing; writes into a throwaway directory and never touches real results. |

A rank count that cannot decompose the case, or that exceeds the allocation, is **refused** with an error --
the harness never silently changes `N`. Not every application accepts every `N` (some require a factorizable
grid). `strong` and `weak` mean what they say; for electronic-structure applications a size sweep is not
automatically a weak-scaling series, and those runs are recorded as completed runs, not as scaling results.

```bash
# real runs
HPCPERF_GPUS=2 HPCPERF_SCALE_MODE=smoke  level3/lammps/validate.sh CUDA
HPCPERF_GPUS=4 HPCPERF_SCALE_MODE=strong level3/lammps/run.sh CUDA

# hypothetical plans only (never executed here)
HPCPERF_DRY_RUN=1 HPCPERF_NODES=10 HPCPERF_GPUS=40 HPCPERF_SCALE_MODE=strong level3/exaca/run.sh CUDA
HPCPERF_DRY_RUN=1 HPCPERF_NODES=20 HPCPERF_GPUS=80 HPCPERF_SCALE_MODE=strong level3/exaca/run.sh CUDA
```

## Source artifacts, hashes, cache and offline use

```
Git metadata (source.lock.yaml)  ->  local cache or Release asset  ->  archive SHA-256  ->  source-tree hash  ->  src/ + deps/
```

* **In Git**: upstream repository/tag/commit, the patch series and its hashes, the artifact filename, byte
  size, archive SHA-256, `source_tree_sha256`, license and redistribution status, and -- once published --
  the immutable asset URL. **Not in Git**: the source archive itself, materialized `src/`/`deps/`, build or
  run output.
* **Cache**: `$HPC_PERFORMANCE_AI_ROOT/.artifacts/sha256/<archive sha256>.tar.zst`, overridable with
  `HPCPERF_ARTIFACT_CACHE` or `--cache-dir`. Entries are content-addressed and read-only; a download is
  verified before it becomes a cache entry.
* **Resolution order**: `--artifact FILE` → cache → the lock's `primary` URL → mirrors. `--offline` forbids
  network access and fails on a cache miss. `--artifact` is verified exactly like a download (size, zstd
  magic, SHA-256, tree hash, safety scans): a local file is never trusted because it exists.
* **After prepare**, no build step needs the artifact, the cache or any upstream repository. Where a benchmark
  needs pinned dependency source (for example CP2K's toolchain tarballs, GEOS' third-party sources, ExaCA's
  JSON library), that source is inside the artifact under `deps/` and the build consumes it from there.
* **Local modifications are never overwritten**: prepare reports `DIRTY` and stops. `--force-rematerialize`
  *discards* your changes and restores the frozen baseline -- there is no undo.
* **Four different identities**: `source_version` (which frozen source), archive SHA-256 (the file),
  `source_tree_sha256` (the extracted tree), and the Git commit of this harness. Publishing metadata (a URL
  becoming known) changes none of the source hashes and requires no re-freeze.
* Hash verification proves you have the intended bytes. It does not prove that they build or produce correct
  results on your machine.

Status and per-artifact numbers: [SOURCE_ARTIFACTS.md](SOURCE_ARTIFACTS.md). Design and the full lifecycle:
[EXTERNAL_ARTIFACT_DESIGN.md](EXTERNAL_ARTIFACT_DESIGN.md). Publication plan:
[RELEASE_PLAN.md](RELEASE_PLAN.md).

## LLM optimization workspace

HPC-Performance-AI packages complete frozen source workspaces. The benchmark itself does not prescribe which
subset of source an optimization agent may modify. Application-only, dependency-aware, hotspot-only, or
whole-stack optimization policies belong to the downstream evaluation protocol and can be applied to the same
frozen benchmark. What the harness does guarantee is integrity: the judge (validator, references, inputs,
build/run entry points, provenance) cannot be changed by the run being judged.

The canonical `level3/<app>` tree is never handed to an agent. Each optimization run gets its own copy:

```bash
tools/create_agent_workspace.sh level3 lammps lammps-demo-001
python3 tools/check_workspace.py workspaces/lammps-demo-001/level3/lammps
HPCPERF_GPUS=2 HPCPERF_SCALE_MODE=smoke tools/validate_workspace.sh level3 lammps \
    workspaces/lammps-demo-001/level3/lammps --iteration 1 -- CUDA
```

`validate_workspace.sh` is the **trusted** entry point: it is executed from this repository, not from inside
the workspace, it builds the workspace's current source for the same backend and variant it then validates,
and it runs the workspace's (unmodified) `validate.sh`. Pass `--skip-build` only when you know a trusted
build record already covers the current source; the record then states the build provenance explicitly.

```
workspaces/<run-id>/                     workspace root -- hand over this whole directory, not just the app folder
├── hpcperf_env.sh, check_env.sh         copies of the environment loader
├── level2/tools/, level3/tools/         copies of the launcher and Level 3 helpers (read-only)
├── .conda_env, .tools, .deps/install    symlinks to the environment (must stay reachable)
├── workspace.yaml, workspace_baseline.json
├── build/level3/<app>/<profile>/, .deps/level3/<app>/<profile>/   created by this run's build; private to the run and profile
├── reports/                             iteration records written by the trusted harness
└── level3/<app>/                        <- the agent's working directory
    ├── src/, deps/                      materialized source tree -- the mutable part
    ├── build.sh, run.sh, validate.sh, benchmark.yaml      protected (benchmark harness and contract)
    └── inputs/, references/, provenance/                  protected (validation assets, identity)
```

The **trusted baseline** used to detect tampering lives outside the workspace, in
`.hpcperf/workspace_baselines/<run-id>.json` of this repository; the copy inside the workspace is
informational. Rules:

* iteration 0 must match the frozen source hash; in later iterations the **source tree** (`src/**`, `deps/**`)
  may be modified, extended or pruned, and every change is recorded (file lists, diff, source hash);
* everything else is **protected by default**: `build.sh`, `run.sh`, `validate.sh`, `benchmark.yaml`,
  `inputs/`, `references/`, `provenance/`, the workspace metadata, the files `benchmark.yaml` declares as
  inputs or references even when they live under `src/`, and the harness copies. They are compared with the
  trusted baseline at every iteration; a change there is refused before anything is built or run;
* the benchmark integrity layer protects the benchmark harness and validation assets; **optimization policy over
  the source tree is intentionally left to the evaluation protocol.** Workspace integrity is not an
  optimization policy, and it is not an OS sandbox;
* a build compiles the workspace's current source; nothing re-materializes the baseline behind the agent's back;
* file permissions and hashes are integrity checks, **not** an operating-system sandbox;
* the environment symlinks (and `--link-prebuilt-deps`, if used) mean the workspace is self-contained with
  respect to *application source*, not with respect to the machine's toolchain: those paths must remain
  reachable, so a workspace is not automatically portable to another machine;
* giving a model an API key does not give it file access: reading, editing, building and running happen
  through your agent harness's tools.

For nekRS, prepare, create, build and validate must all use the same `--variant`; do not switch variants by
patching inside a workspace.

The closed loop (inject a compile error → build fails → restore → build succeeds → validate) has been
executed for **LAMMPS only** ([lammps/provenance/agent_workspace_verification.yaml](lammps/provenance/agent_workspace_verification.yaml)).
It is not evidence for the other applications.

## Validation and evidence status

| Evidence | Where it stands |
|---|---|
| Source equivalence + materialization | All 11 artifacts: verified (extracted tree hash equals the frozen hash; 15/15 contract checks, re-run 2026-09-13 after the scope-file removal). |
| Current workspace build | LAMMPS: verified. ExaCA: canonical build from the artifact verified. All others: historical builds only. |
| Agent edit takes effect | LAMMPS: verified (changed binary hash, rebuilt and revalidated). All others: not run. |
| Scientific validation | See the per-application README for the criterion and the date; several results are historical runs on trees proven content-equivalent to the artifacts. |
| Workspace multi-GPU | LAMMPS: 1 and 2 GPUs inside a workspace. Others: not run in a workspace. |
| GPU binding | Recorded per run from the launcher audit (`N verified, 0 mismatch`); short runs can end before the sampler observes them and are reported as `unverified`, which is an observation gap, not a mismatch. |
| Remote fetch by an ordinary user | **VERIFIED** on 2026-09-12 for all 11 artifacts, twice: an anonymous download from the release URL checked against the locks' own hashes, then a clean clone of the lock-update commit with an empty cache and an ordinary `prepare_benchmark.sh`. No credential was present in either check. |

Full matrix: [WORKSPACE_EVIDENCE.md](WORKSPACE_EVIDENCE.md).

This repository's own regression suite (`level3/tools/tests/run_all.sh`, CPU only, no GPU) currently reports
**247/247 checks passing** in seven groups: harness infrastructure 30, second-batch numerical checkers 21, Nyx
comparator 33, verdict classes 18, ExaCA validator 19, source-distribution tools 81, release publication and
anonymous-fetch API mock 45. These are **this project's** harness, validator and publisher tests. They are a
different thing from an application's own upstream test suite: ExaCA's upstream unit-test result (30/52 as
upstream runs them) is **not** part of that number and never counted as a pass here.

A numerical PASS recorded with `build_provenance: UNVERIFIED` (a `--skip-build` iteration) stays a scientific
result for the binary that ran, but it does **not** demonstrate that the current source modification was
compiled, and such iterations must not be used as evidence of an optimization gain.

Criteria differ per application and are **not** all "upstream official validation":

* upstream reference output: DFT-FE (GPU reference), SPECFEM3D (reference seismograms), LAMMPS (shipped CPU
  reference log), nekRS (upstream `--cimode` checks), CP2K (upstream regtest tolerances);
* analytic solution: WarpX (`langmuir_multi`);
* adapted subset of upstream's own regression comparison: LAMMPS thermo columns, SPARTA statistics;
* cross-rank consistency (same build, different rank counts): all applications, as an additional check;
* CPU cross-check: where an application provides one;
* project-defined statistical protocol: ExaCA's `dirsolid` case only -- upstream provides no reference for it;
  calibration and holdout are separated (`exaca/references/validation_protocol.md`);
* upstream full-application reference: ExaCA's two **small official** cases inside its own test suite
  (`VolFractionNucleated` 0.1882 ± 0.0100, `TimeStepOfOutput` 4820 ± 1) -- verified here at 1/2/4 GPUs, and not
  transferable to the 128³ `dirsolid` case;
* upstream kernel-level unit tests: run for ExaCA. Three separate results, never merged into one sentence:
  the **first observation** (2026-09-11) was 30 of 52 passing as upstream defines the run; the **attribution**
  of those 22 failures classifies every one of them as test-side or configuration
  (`exaca/references/upstream_unit_test_matrix.json`), with zero in the APPLICATION class -- a statement about
  those 22 failures in this environment, not a general claim about the application; the
  **patched-upstream-tests** result is 23/23 of the applicable CUDA tests passing with the fixture-only patches
  in `exaca/patches/upstream-tests/` plus per-rank GPU binding, and 23/23 in a Serial-only build. No unit-test
  result is counted as a pass in this repository's own totals.

For ExaCA these are three separate statements and must stay separate: (a) the project's statistical `dirsolid`
smoke protocol passes (calibration + 9/9 holdout); (b) upstream's applicable tests fail as upstream runs them in
this environment, and pass only with the test-side fixture patches (in git, not in the artifact) and per-rank
GPU binding; (c) release acceptance
was on hold and is now restored on the basis of (a), the attribution of (b), and the official small full cases at
1/2/4 GPUs.

Exit codes, by program:

| Program | Codes |
|---|---|
| `level3/<app>/validate.sh` | `0` PASS, `1` FAIL, `3` PENDING (Nyx `I_R_CHECK_PENDING`), `4` UNSUPPORTED_LAYOUT |
| `tools/validate_workspace.sh` | the above, plus `6` REFUSED (workspace integrity: tampering or untrusted baseline; nothing was built or run) and `7` BUILD_FAIL; `2` for usage errors |
| `tools/prepare_benchmark.sh` | `0` ok, `1` invalid, `2` usage, `3` DIRTY (local modifications), `4` artifact unavailable/unpublished/offline miss, `5` hash, size or content mismatch |

A completed run is not a correctness result. PENDING, UNSUPPORTED_LAYOUT and REFUSED are never counted as
passes and never enter a performance summary.

## Known limitations

* **Nyx**: the `LyA` heat/cool deck ends as `STATE_AND_PARTICLES_PASS; I_R_CHECK_PENDING` (exit 3) at 1/2/4
  GPUs -- state and particle checks pass, the `I_R` field is not accepted by any tolerance and the cause is
  **not** established. `MiniSB` and `LyA-adiabatic` pass separately. [nyx/README.md](nyx/README.md).
* **ExaCA**: acceptance covers the `dirsolid` case at 128³, seed 0, 1/2/4 GPUs, under a project-defined
  empirical protocol (calibration and holdout separated; 8 + 9 runs; an empirical range rule, **not** a 3σ
  guarantee, and not a general oracle for spatial fields or other sizes). Upstream's kernel-level unit tests
  gave **30/52 passing as upstream runs them** when first observed here, and unmodified upstream tests under
  CTest's own launcher still fail in this environment today. All 22 failures are attributed (7 test-fixture, 13
  host-space variants inside a CUDA build, 2 CTest GPU-binding, 0 application, 0 unresolved); `0 application`
  describes this attribution matrix only and is not a claim that the application has no defects in other
  environments or in untested code paths
  ([exaca/references/upstream_unit_test_matrix.json](exaca/references/upstream_unit_test_matrix.json)).
  Separately, with the fixture-only test patches applied (they live in git under
  [exaca/patches/upstream-tests/](exaca/patches/upstream-tests/) and are **not** inside the frozen source
  artifact) the applicable CUDA tests pass 23/23, as they do in a Serial-only build. Upstream's two small
  full-application cases pass at 1/2/4 GPUs within upstream tolerances. Release acceptance was ON_HOLD and is
  restored on that basis; the Finch coupling, `FromFile` problems and other sizes remain unexercised.
* **QMCPACK**: walker count and cuSOLVER behaviour constrain the validated case; this is a property of that
  case, not a universal limit. [qmcpack/README.md](qmcpack/README.md).
* **Not rebuilt from a materialized workspace**: eight applications keep historical build evidence only.
  DFT-FE's install fingerprint changed with the freeze, so its next build refuses to reuse the existing
  install until that install is removed.
* **Strong/weak runs** for several applications are recorded as COMPLETED (they ran to completion) and are not
  scaling conclusions.
* **HIP untested; multi-node unverified/blocked on this site; 8/40/80 GPUs are dry-run plans only.**
* The environment (driver, MPI transport, site scheduler) is a prerequisite, not something this repository
  provides: nothing here is guaranteed to work out of the box on an arbitrary machine.
* **GEOS is retired** (see the catalog); its results remain for reference only.

## Troubleshooting

| Symptom | Meaning |
|---|---|
| `REMOTE_ARTIFACT_UNPUBLISHED` / exit 4 | No published URL yet: pass `--artifact <file>`. |
| `--offline` + cache miss | The artifact is not in the cache; supply it or allow the download. |
| Archive or tree hash mismatch (exit 5) | The file is not the frozen artifact. Do not force it; re-obtain the artifact. |
| `source not materialized ... run tools/prepare_benchmark.sh` | The benchmark has no `src/` yet. |
| `DIRTY` (exit 3) | `src/`/`deps/` differ from the frozen baseline. `--force-rematerialize` **discards** those changes. |
| Variant mismatch | Marker and requested variant disagree (nekRS): prepare the variant you intend to use. |
| Fingerprint mismatch at build time | The recorded install was built with a different toolchain/configuration; remove that install or restore the configuration. |
| Rank count refused | The case cannot be decomposed that way, or the allocation is smaller than requested. |
| Multi-node hang or launch failure | The site transport is unverified for multi-node here. |
| `REFUSED` (exit 6) | Workspace integrity: a protected file (harness, validator, inputs, references, provenance, metadata) changed or the baseline is untrusted. Restore it; do not weaken the check. |
| `PENDING` (exit 3) | A scientific result that is explicitly incomplete (Nyx `I_R`). It is not a pass and must not be converted into one. |

Forcing a rematerialization, loosening a tolerance or lowering precision are **not** general remedies; they
change what is being measured.

Where to look: each run writes `run_manifest.txt` (run id, binary and input hashes, exit code) and its logs
next to the results under `build/level3/<app>/<profile>/run*/`; each workspace iteration writes
`reports/iter-N.{check.json,diff,verdict.yaml,validate.log}` in the workspace root. Older bring-up logs from
the original development runs are not packaged for download; the facts extracted from them are in the
documents linked above.

Provenance and further reading: [WORKSPACE_EVIDENCE.md](WORKSPACE_EVIDENCE.md),
[SOURCE_ARTIFACTS.md](SOURCE_ARTIFACTS.md), [EXTERNAL_ARTIFACT_DESIGN.md](EXTERNAL_ARTIFACT_DESIGN.md),
[RELEASE_PLAN.md](RELEASE_PLAN.md), [BUILD_STRATEGY.md](BUILD_STRATEGY.md),
[APPLICATION_AUDIT.md](APPLICATION_AUDIT.md), [CORRECTNESS_FIXES.md](CORRECTNESS_FIXES.md),
per-application `README.md` and `provenance/` (licenses, patch series, manifests). Maintainer/history:
[SECOND_BATCH_STATUS.md](SECOND_BATCH_STATUS.md), [LFS_TO_ARTIFACT_MIGRATION.md](LFS_TO_ARTIFACT_MIGRATION.md).

Every application here is upstream software under its own license, redistributed unmodified except for the
recorded patch series; this repository contributes the benchmark harness, the input selection, the validation
criteria and the provenance.
