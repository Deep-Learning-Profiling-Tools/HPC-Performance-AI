# CLAUDE.md -- working notes for Claude Code in this repository

HPC-Performance-AI: a three-level GPU benchmark/application suite used to build
an AI framework for HPC performance prediction. Everything here was brought up
on one node (dgx003: 4x NVIDIA B200 / sm_100, CUDA 13.2.78, RHEL 10, 64 CPUs,
2 TB RAM) and nothing is claimed beyond what actually ran there.

| Level | Content | Where the truth lives |
|---|---|---|
| `level1/` | 50 standalone GPU kernels (CMake, `-DBACKEND=CUDA\|HIP`, ctest validation) | `level1/README.md`, per-benchmark README |
| `level2/` | 24 mini-apps with upstream build systems + `build.sh/run.sh/validate.sh` | `level2/README.md`, `level2/SCALEOUT_AUDIT.md`, `level2/tools/README.md` |
| `level3/` | full production applications, multi-GPU by design | `level3/README.md`, `level3/APPLICATION_AUDIT.md`, `level3/BUILD_STRATEGY.md`, `level3/CORRECTNESS_FIXES.md`, `level3/SECOND_BATCH_STATUS.md`, per-app README |

Read the per-level status document before touching a level; they record what
was built, how, what failed and what is still open. Do not re-derive.

## Environment (every shell)

```bash
source hpcperf_env.sh      # activates .conda_env, .tools/bin, .deps/install prefixes, MPI transport profile
./check_env.sh             # verifies the validated configuration (--mpi-cuda checks device-buffer MPI)
```

Facts that differ from any "reference" you may read elsewhere:

- `/usr/local/cuda` is CUDA **13.2.78**. Never change the symlink or the driver.
  CUDA 13.2 is the preferred Toolkit for everything; a private older Toolkit is
  an exception that must be justified in the app README.
- Compilers: conda GCC 13.3.0 (Level 1/2, Nyx) and **system GCC 14.2.1**
  (`/usr/bin/gcc`, used for CP2K/DFT-FE/GEOS: one compiler for C/C++/Fortran; the
  conda GCC has no gfortran). EL10's GCC 14 defaults to `-march=x86-64-v3`, so
  `__AVX2__` is defined in every nvcc host pass.
- MPI: conda Open MPI 5.0.10. The site UCX transport **hangs on CUDA device
  buffers**; the launcher's `gmu-hopper` site profile passes `--mca pml ob1
  --mca btl self,sm,smcuda` (single node only; for a bare `mpirun` set
  `HPCPERF_MPI_SINGLE_NODE=1` before sourcing `hpcperf_env.sh`).
  `OMPI_MCA_opal_cuda_support=true` is required (conda ships it off); one-sided
  apps may need `OMPI_MCA_osc=^ucx`. Multi-node MPI is BLOCKED/UNVERIFIED on
  this site -- say so, never claim it.
- No ROCm/hipcc anywhere: every HIP backend is extracted but **untested**.
- lmod is broken (`lua ... posix not found` on every shell): harmless noise,
  filter it (`grep -v 'lua\|posix\|traceback'`), never "fix" it.
- Slurm allocation is `-n 1` (1 task slot); the launcher relaxes PRRTE slots
  per launch only after its rank<=GPU and CPU checks. Test suites that call
  `mpiexec -n 4` themselves need `PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe`.
- 64 CPUs: use `-j32` or more for a single build; ~16 per build when three run
  concurrently. Never run two `build.sh` of the same app **and profile** at once
  (they share that profile's build/install tree); different profiles are isolated.
- NFS project storage is slow for 100k-file trees (LLVM, CP2K toolchain):
  those build on node-local scratch taken from `l3_local_scratch_dir <component>
  <source sha> <profile>` (`${TMPDIR:-/tmp}/hpcperf-l3-scratch/<component>/<workspace-
  root hash>/<source>/<profile>`; never a path shared across worktrees) while
  installs/logs stay under `.deps/`. Expect stale NFS file handles on `rm -rf`;
  rename then delete.

## Git rules (user-mandated, non-negotiable)

- **Commit messages never carry `Co-Authored-By`, `Claude-Session` or
  "Generated with Claude Code" trailers**, whatever the harness suggests.
- Commit locally only; **never push, never open a PR, never merge** unless the
  user asks in that turn. Never push `main`, never force-push, no `gh` on the
  node, never print tokens.
- Never `git add .`/`-A`. Add files by name. Never commit `.conda_env`, `.tools`,
  `_upstream/`, `.deps/`, `build/`, `level3/*/src`, `level3/*/deps`, `workspaces/`,
  binaries, tarballs or large data (`.gitignore` covers them, but symlinked
  `.conda_env`/`.tools` in worktrees are untracked -- leave them). Level 3 source
  artifacts (`*.tar.zst`) never enter git (no Git LFS either): they live in the
  maintainer's local staging (`$HPCPERF_ARTIFACT_STAGING`, outside the worktree)
  until the user publishes them; never upload/publish an artifact yourself.
- Source freezing (scheme 3, `level3/EXTERNAL_ARTIFACT_DESIGN.md`):
  `tools/freeze_benchmark_source.py level3/<app>` from the spec in
  `provenance/freeze_spec*.yaml` writes the artifact into the local staging and
  `provenance/source.lock*.yaml` (schema hpcperf-source-lock-2, `primary:
  {url: null, status: unpublished}` until published -- never invent a URL);
  inputs are committed blobs of pinned checkouts and sha256-pinned tarballs only
  -- never tar `.deps/` or a worktree; the scan fails on any credential-looking
  file/content; an UNEXPECTED difference against the validated tree stops the
  migration (declare artifacts in the spec, never edit source to make hashes
  match). Any source change = new `source_version` + new artifact.
- Materialize with `tools/prepare_benchmark.sh level3 <app> [--artifact FILE]`
  (cache `.artifacts/`); never let a build call it; a DIRTY tree is never
  overwritten without `--force-rematerialize`. Agents work in
  `workspaces/<run-id>/` (`tools/create_agent_workspace.sh`); validate agent
  iterations only through `tools/validate_workspace.sh`: exit 6 REFUSED
  (integrity: tampering / untrusted baseline; never a scientific result), 7
  BUILD_FAIL, 0/1/3/4 numerical as validate.sh; rc 3 is PENDING only for Nyx.
- Publication: provider = GitHub Release assets of this repo, first tag
  `level3-source-hpcperf-l3-v1-rc1` (prerelease). Never upload: the adapter
  `tools/artifacts/github_release_upload.sh` needs `HPCPERF_CONFIRM_UPLOAD=yes`,
  which only the user grants per run; locks stay `unpublished` until
  `tools/artifacts/remote_fetch_check.sh` (anonymous, clean clone, empty cache)
  passed. GEOS/ParMETIS never enter a release.
- One commit per application or infrastructure change, message = what/why with
  the measured facts. Branch names follow `CONTRIBUTING.md` (`level3/<app>`,
  `env/...`, `docs/...`).
- Worktrees: feature branches are developed in sibling worktrees
  (`../HPC-Performance-AI-b*`) of one repository; run `git worktree list` before
  assuming which branch a path is on, and start new work from `main`. `.deps/`,
  `build/`, `_upstream/` are per-worktree.

## Conventions per application (Level 2/3)

```
level3/<app>/
  src/, deps/   the ONLY application/benchmark-specific source input of build.sh (never committed; materialized from
                the external source artifact by tools/prepare_benchmark.sh; identity = source_tree_sha256 in
                benchmark.yaml / provenance/source.lock*.yaml)
  provenance/   freeze_spec, source.lock (schema 2), upstream.lock, patch_series, original_vs_baseline.diff,
                SOURCE_MANIFEST.json, LICENSES.md, equivalence.*, LOC.*, check_workspace.json
  benchmark.yaml   machine-readable contract (entries, inputs, references, identity). No optimization-scope file:
                the benchmark does not prescribe what an agent may modify; that is evaluation-protocol business
  fetch.sh      FREEZE-TIME ONLY: pinned upstream checkout into _upstream/ -- never called by build.sh
  build.sh      idempotent, stage-marked (.hpcperf-stage-done), per-profile, writes BUILD_INFO.txt + .hpcperf-l3-fingerprint;
                reads $HERE/src, $HERE/deps only; applies NO patch (the bundle is the patched baseline); builds that
                write into their source tree (SPECFEM3D, nekRS, DFT-FE, GEOS, CP2K toolchain) use a build-side copy
  run.sh        [CUDA] [args]; cases via HPCPERF_<APP>_CASE; modes smoke|strong|weak; writes run_manifest.txt
  validate.sh   [CUDA]; HPCPERF_GPUS=N; prints the criteria and the verdict; exit 0 PASS, 1 FAIL,
                3 PENDING (Nyx heat/cool I_R_CHECK_PENDING), 4 UNSUPPORTED_LAYOUT (Nyx) -- only 0 is a pass
  <app>_check.py  the numeric checker (uses level3/tools/l3_check.py: require_finite, ValidationError)
  patches/      *.patch with header: source, rationale, conditions, impact, verification, class
  README.md     provenance, versions, node adaptations, cases, criteria, RESULTS with dates
```

- Profiles (every application since 2026-09-15): ONE frozen source tree
  `level3/<app>/{src,deps}` (never copied per backend) and ALL generated state per
  backend/profile, `.deps/level3/<app>/<profile>/{src,build,install,logs,cache}` +
  `build/level3/<app>/<profile>/`, via `l3_paths_profile <app> <profile> <backend>`
  (`level3/tools/l3_common.sh`; the legacy shared `l3_paths` no longer exists). A
  profile must name its backend (`cuda`, `hip`, `hypregpu.cuda`,
  `cuda132-gcc142-ompi5010`, ...); derive it with `l3_backend_profile <APP> <backend>
  [variant]` identically in build/run/validate; `HPCPERF_<APP>_PROFILE` overrides; a
  profile/BACKEND conflict is refused before any directory is created; run.sh gates on
  the profile's own fingerprint (`l3_fingerprint_expect_backend`); never read
  `.deps/level3/<app>/install` (pre-migration state). Scratch that must live outside
  the worktree (CP2K toolchain, QMCPACK LLVM) comes from `l3_local_scratch_dir
  <component> <source sha> <profile>` (workspace-root hash / source / profile under
  `${TMPDIR:-/tmp}/hpcperf-l3-scratch/`), never from a name shared across worktrees. Runs under
  `.../run/<case>.<mode>.np<N>[.t<T>]/`, dry-runs under `.../run/.dryrun/` (never
  touch real results). See `level3/PROFILE_ISOLATION.md`.
- `l3_common.sh` helpers you should reuse rather than reinvent:
  `l3_isolate_build_env` (strip Level 2 prefixes), `l3_clean_conda_build_env`
  (clear conda CFLAGS/LDFLAGS/AR/CMAKE_GENERATOR and `C_INCLUDE_PATH`/`LIBRARY_PATH`
  for system-GCC builds), `l3_binary_backend_check` (cuobjdump archs, works for
  static cudart), `l3_rundir`, `l3_run_id`, `l3_manifest`, `l3_fingerprint_*`,
  `l3_scale_mode`, `hpcperf_ranks`, `hpcperf_topology`, `hpcperf_forbid_args`.
- Registered inputs (since 2026-09-21; batch 1: background_subtraction, hipbone, quicksilver,
  lammps; batch 2: tealeaf, sparta; batch 3: cloverleaf, laghos, lammps ReaxFF on the separate
  `reaxff.cuda` profile via `HPCPERF_LAMMPS_VARIANT=reaxff`): a benchmark with several inputs carries `inputs.yaml` (schema
  `hpcperf-inputs-1`, read by `tools/inputs/hpcperf_inputs.py`; ids defined by workload
  content, source kind upstream-file / upstream-parameterized / derived / custom, the
  benchmark's OWN timer scope, baseline quantities + comparison rule). Selection only
  through the benchmark's selector variable (`HPCPERF_<APP>_INPUT`, Quicksilver:
  `HPCPERF_QUICKSILVER_INPUT_ID`); run.sh refuses an id together with the knobs/args it
  would override and the default command stays unchanged. `measure` = 1 warm-up + N runs
  with the native timer (never wall time as main compute). `compare` exit 0 only for verdict
  PASS (every REQUIRED quantity verified); 3 = INCOMPLETE (a required quantity still `record`),
  1 = FAIL, 2 = refused (other input/benchmark/workload or same file); diagnostic quantities
  may stay `record`; tests in
  `tools/inputs/tests/run_all.sh`. See `tools/inputs/README.md`.
- Launch only through `level2/tools/hpcperf_mpi_launch.sh --gpus N
  [--cpus-per-rank C] --bind wrapper -- <exe> ...`: one MPI rank per GPU, per-rank
  `CUDA_VISIBLE_DEVICES`, nvidia-smi audit ("N verified, 0 mismatch, 0
  unverified" is required evidence). Interface: `HPCPERF_GPUS=N|all`,
  `HPCPERF_NODES`, `HPCPERF_GPUS_PER_NODE`, `HPCPERF_CPUS_PER_RANK`,
  `HPCPERF_SCALE_MODE=smoke|strong|weak`, `HPCPERF_SITE_PROFILE`,
  `HPCPERF_DRY_RUN=1`, `HPCPERF_GPU_BACKEND=CUDA|HIP` (default CUDA; HIP counts
  devices with rocminfo and drops the CUDA-only MCA hook -- untested here).
  Requested GPUs == used GPUs; a rank count that cannot
  partition the problem is **refused**, never silently changed; 8/40/80 GPUs
  exist only as HYPOTHETICAL dry-runs (`HPCPERF_NODES=N/4`).
- Do not move the common runtime, do not refactor the launcher for an app;
  extend it only with tests in `level2/tools/tests/run_all.sh`.
- Regression campaigns never overwrite historical results: set
  `HPCPERF_L3_RUN_SUBDIR=run.regress-<sha>` (every run.sh/validate.sh builds its
  run directories under `build/level3/<app>/<profile>/$L3_RUN_SUBDIR`).
- Agent runs: `tools/create_agent_workspace.sh level3 <app> <run-id>
  [--link-prebuilt-deps]` -> `workspaces/<run-id>/level3/<app>/` (real copy of
  src/deps writable, everything else protected by the trusted baseline hash (chmod is best effort), harness copied to the workspace root,
  environment symlinked); `tools/check_workspace.py` must PASS before iteration 0;
  never point an agent at the canonical `level3/<app>`; never symlink src/deps
  back to it. Build outputs of a workspace stay under `workspaces/<run-id>/`.
- Any tool that records its process environment (CP2K's toolchain installer,
  nsys/ncu, env-logging build systems) runs through `l3_clean_env_exec` /
  `level3/tools/l3_clean_env.sh` (allow-listed `env -i` plus a credential
  deny-rule that beats the allow-list): the login shell carries credentials that
  must never land in a `declare -x` dump or a profiler report. Never print a full
  `env` into a log; report variable names only. The common launcher/run.sh path
  is NOT wrapped yet (follow-up): do not profile a science run with nsys/ncu
  without the wrapper.
- Queues: run every step through `l3_run_recorded <rc-file> <label> -- cmd`
  (records the exit code, never aborts), classify with `level3/tools/l3_verdict.py`
  (PASS / PENDING / UNSUPPORTED_LAYOUT / FAIL / MISSING). Exit 3 and 4 are never
  PASS and never enter a performance summary; report their counts separately.

Level 2 specifics (mini-apps; the tree above is Level 3's):
- Upstream source is vendored byte-identically in git (`src/UPSTREAM_SHA256SUMS`
  where a stand-alone driver was extracted, e.g. MiniEM); `build.sh [CUDA|HIP]`
  passes the backend to the upstream build system, products go to
  `build/level2/<app>/<cuda|hip>`. The root `.gitignore` rule `build/` also
  matches upstream directories named `build/`: re-include them explicitly
  (`!level2/<app>/.../build/`) -- `level2/tools/tests/test_vendored_completeness.sh`
  fails when a needed vendored file is missing from a clean clone.
- Framework dependencies come from `setup_level2_deps.sh` into
  `.deps/install/<dep>` (pins = tags or 40-hex commits, `.hpcperf-built` /
  `.hpcperf-fingerprint` markers, `HPCPERF_DEPS_SEED_DIR` for local seeds);
  MiniEM needs `setup_level2_deps.sh trilinos` (~30 min, Kokkos 5.2.1 develop pin,
  Zoltan2 for MueLu at >1 rank, netCDF + gtest TPLs from conda). GAMESS RI-MP2
  needs an environment-provided NVIDIA HPC SDK `nvfortran` (not conda).
- validate.sh runs at `HPCPERF_GPUS=1` by default; upstream ctests that call
  `mpirun -np 2` themselves need the PRRTE slot relaxation as a command-local
  `env` (Branson pattern), never an export.

## Validation principles

- Official small scientific workflows (upstream decks, verbatim or with a
  documented derivation), never library unit tests as "the benchmark"; unit
  tests are dependency probes and are reported as such.
- Criteria are fixed **before** the runs, taken from upstream's own test
  definitions where they exist (regtest tolerances, `check_scalars.py`,
  geos-ats metrics, `fcompare` tolerances ...) and cross-checked against
  upstream's published baselines/references first. If a criterion turns out to
  be misread, fix it before the run and record the correction in the README.
- FAIL on: non-zero exit, timeout, NaN/Inf, incomplete output, missing GPU
  evidence, launcher audit mismatch, stale results. Correctness at N GPUs =
  official reference **and** consistency with the 1-GPU run. GPU evidence must
  be more than "links libcudart" (device timers, offload banners, audit).
- Every new checker gets negative tests (`level3/tools/tests/test_l3_validators.sh`,
  `level3/tools/tests/run_all.sh`).
- Record failed attempts (archive under `.deps/.../ATTEMPT-*`, README section),
  the real cause, and what remains open. Timing numbers from problems too small
  to scale are "completeness records", not scaling results -- say so.

## Change policy

- No kernel, algorithm, precision, solver-placement or tolerance changes in
  upstream code. Patch classes: A none, B build-system, C node adaptation
  (no patch, in build.sh), D minimal source back-port/fix with upstream
  provenance, E anything else (forbidden without the user).
- Prefer stable releases with the upstream-documented dependency combination;
  a `develop` snapshot needs a fixed SHA and a written reason (see GEOS).
- Native builds; Spack is rejected per app for recorded reasons (recipes cap
  `cuda_arch` at 90, inert GPU variants, 2025-05 local Spack).

## Pitfalls that cost hours (keep them in mind)

Shell / process
- The login shell **exports a `grep` shell function** into child processes
  (`l3_common.sh` does `unset -f grep`; use `/usr/bin/grep` in ad-hoc commands).
- `pkill -f`/`pgrep -f` self-kill the calling shell (exit 144) whenever the
  pattern text also appears later in the same command line -- kills go in their
  own command. Never edit a script and launch it in the same response.
- Under `set -e -o pipefail` an empty `grep` inside `$( ... )` aborts the script
  silently (`cuobjdump`, `ls -d missing`, log parsing): wrap with
  `{ cmd || true; }`.
- Background task output is buffered by `grep`; read progress from run dirs,
  manifests and logs, not from the task file. Serialize GPU jobs through a lock
  (atomic `mkdir`), never let two GPU workloads share devices unknowingly.

Conda environment leaks
- Conda exports `CFLAGS/CXXFLAGS/LDFLAGS` (`-march=nocona`, `-isystem`,
  `-Wl,--disable-new-dtags -rpath <conda lib>`), `AR`, `CMAKE_GENERATOR=Ninja`,
  `C_INCLUDE_PATH`/`LIBRARY_PATH` with foreign envs; MPI wrappers add the conda
  rpath first. Consequences seen: CP2K linked to the conda pthreads OpenBLAS
  (needed `CP2K_BLAS_VENDOR=CUSTOM` + rpath order + ldd guard), GEOS picked a
  32-bit `metis.h` from the conda include dir (put TPL `-I` first), nekRS took
  `$AR` as the archive command. Clear them for system-GCC builds.
- Conda `mpif90` needs `OMPI_FC=/usr/bin/gfortran`; mixed GCC13/gfortran14 LTO
  needs `-fno-lto`; conda GCC links PIE (Fortran objects need `-fPIC`); conda's
  cmake activation replaces a user `CMAKE_PREFIX_PATH`.
- OpenBLAS 0.3.30 misdetects the Xeon 8570 (`TARGET=SAPPHIRERAPIDS`); the
  node's zlib-ng CMake package references a missing `libz.a`
  (`-DCMAKE_IGNORE_PATH=/usr/lib64/cmake/ZLIB;/lib64/cmake/ZLIB`).

CUDA 13.2 / Blackwell
- cicc unrolls thread-strided FEM loops into huge PTX on sm_100 (MFEM
  `-Xcicc=-O1` patch, Laghos `MFEM_UNROLL(1)`; GEOS' heaviest TU takes ~26 min).
- Removed APIs: `cudaDeviceProp::memoryClockRate` (BLT smoke test back-port),
  Thrust 3.2 header changes in older hypre (nekRS patches). RAJA 2026.07's AVX2
  tensor layer does not compile under nvcc 13.2 + GCC 14 (`RAJA_ENABLE_VECTORIZATION=OFF`).
- Pinned submodules may not know sm_100 (Nyx's AMReX pin drops SM>=10):
  check `cuobjdump` archs of every built library, don't trust the flag.
- LLVM >= 21 builds the NVPTX device runtime only via
  `LLVM_RUNTIME_TARGETS=...;nvptx64-nvidia-cuda`; official LLVM binaries ship no
  offload runtime. QMCPACK on this stack uses ~320 MB device memory per walker
  (open issue): keep <= 300 walkers/GPU.

Level 3 build scripts
- `l3_fingerprint_text` hashes patch FILES: pass `$HERE/patches/<name>`, never the
  bare basenames from the lock (SPECFEM3D/nekRS aborted on their first rebuild).
- A `build.sh` must work on an empty profile tree: never read install-layout facts
  (RPATH dirs, a library path) before the stage that creates them (CP2K did, and
  could only re-run on top of an existing toolchain).
- Tpetra/Kokkos apps hang at >1 rank when Open MPI reports CUDA-aware device
  buffers through smcuda on this site: MiniEM runs with
  `TPETRA_ASSUME_GPU_AWARE_MPI=0`.
- The site NVIDIA HPC SDK's shipped `localrc` targets a GCC 8 that RHEL 10 no
  longer has: generate a per-user one (`makelocalrc -gcc /usr/bin/gcc ... -x -d
  <dir>`, `NVLOCALRC=<dir>/localrc`) before using `nvfortran`.

Build systems
- GEOS `scripts/config-build.py` **deletes an existing build tree**; configure
  once, reconfigure with plain `cmake -D... -C hostconfig` (`-D` before `-C`);
  forced cache entries need `cmake -U`. DBCSR's cmake breaks inside a git
  worktree (`.git` file) -- build on scratch. dftfe's `p4est-setup.sh` hardcodes
  Cray wrappers. ELPA's configure needs `-march=native` and ScaLAPACK paths in
  `LDFLAGS`; its analytic test programs print "Maximum error in
  eigenvalues/eigenvectors" (limits 5e-14 / 6e-10), not residual/orthogonality.
- Restart/HDF5 comparisons: a GPU build stores LvArrays with another
  `__permutation__` than a CPU baseline; compare logical arrays (geos-ats
  `permute_array`). Compile-time GPU defaults (GEOS `LinearSolverParameters`)
  legitimately differ from CPU baselines -- report, don't gate.
- AMReX `particle_compare` cannot compare across rank counts and exits 0 on
  FAIL; WarpX 26.09 uses CODATA 2022; single-node smcuda GPU-aware MPI is 3x
  slower for WarpX but 2.5x faster for LAMMPS -- record, don't generalise.

## Current state (2026-09-15)

- Level 1: 50/50 CUDA validated (ctest per benchmark directory; the aggregate
  `cmake -S level1` configure has no top-level `enable_testing()`).
- Level 2: 24/24 CUDA single-GPU working/validated on dgx003 (clean-clone
  protocol, 2026-09-15). 11 applications (amg2023, branson, cloverleaf,
  examinimd, exampm, haccabanapm, kripke, laghos, remhos, tealeaf, miniem) are
  the subset with verified 4-rank x 4-B200 single-node multi-GPU correctness.
  Comb, Quicksilver, SW4lite and GAMESS RI-MP2 have MPI execution paths but
  1-GPU validation only so far; hipBone and miniWeather have MPI paths without
  a verified multi-GPU result yet. MiniEM is validated (Trilinos built by
  `setup_level2_deps.sh trilinos`). GAMESS RI-MP2 needs an environment-provided
  NVIDIA HPC SDK `nvfortran` (not in the conda env; on dgx003 the site 25.7
  install with a per-user `makelocalrc` for GCC 14; CUDA arch auto-detected).
  HIP: untested everywhere (no ROCm). Status: `level2/README.md`,
  `level2/SCALEOUT_AUDIT.md`.
- Level 3: 10 applications in `main` (PR #5 source freeze, PR #8 backend/
  profile isolation of generated state); source artifacts published as
  `level3-source-hpcperf-l3-v1-rc1` and anonymously verified;
  `tools/prepare_benchmark.sh level3 <app>` materializes `src/` + `deps/`.
  CUDA validated at 1/2/4 GPUs as documented per application in
  `level3/README.md` / `benchmark.yaml`; Nyx heat/cool I_R remains PENDING
  (rc 3, never a pass); GEOS retired; HIP UNTESTED; multi-node
  BLOCKED/UNVERIFIED; 8/40/80 GPUs are dry-run plans only. Never write
  "Level 3 fully accepted" or "all tests PASS".
- Remote (`origin`): `main` is the canonical entry point and contains the
  completed Level 1/2/3 bring-up infrastructure. Historical bring-up branches
  (`level2/miniapps`, `level3/apps`, `level3/full-apps-bringup`,
  `level3/second-batch-bringup`, `level3/source-freeze`, ...) are development
  provenance, not entry points: start new work from `main`.
- Open items (follow-ups, none resolved): Nyx I_R (CPU-vs-CPU and GPU-vs-GPU
  repeats at the original configuration, independent tolerance comparison);
  GEOS compositional-flow/well unit tests (retired application; modules
  UNVERIFIED); QMCPACK per-walker device memory / cuSOLVER (<= 300
  walkers/GPU); numerical acceptance of strong/weak runs; multi-node and HIP;
  2+/4-GPU validation of Comb/Quicksilver/SW4lite/GAMESS RI-MP2 and multi-GPU
  decks for hipBone/miniWeather; wrapping the run/profiler path in the clean
  environment; the site's device-buffer (smcuda) MPI path that hangs Tpetra
  (MiniEM runs with `TPETRA_ASSUME_GPU_AWARE_MPI=0`).

## Reporting

The user writes Chinese; answer in Chinese, keep file contents/commit messages
in English. A final report per round lists: what was completed per app; exact
versions/environment/build strategy; measured 1/2/4-GPU results with the
criteria; GPU compute/communication paths; private Toolkit/compiler
exceptions; strong/weak/size-sweep definitions and status; 40/80 dry-run and
multi-node boundary; patches, real failure causes, unfinished items; where the
raw logs/manifests/fingerprints are; local commits and the decisions needed
from the user. State failures plainly with their cause; never soften a FAIL.
