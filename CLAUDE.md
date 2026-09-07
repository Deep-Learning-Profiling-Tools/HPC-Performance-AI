# CLAUDE.md -- working notes for Claude Code in this repository

HPC-Performance-AI: a three-level GPU benchmark/application suite used to build
an AI framework for HPC performance prediction. Everything here was brought up
on one node (dgx003: 4x NVIDIA B200 / sm_100, CUDA 13.2.78, RHEL 10, 64 CPUs,
800 GB) and nothing is claimed beyond what actually ran there.

| Level | Content | Where the truth lives |
|---|---|---|
| `level1/` | 50 standalone GPU kernels (CMake, `-DBACKEND=CUDA\|HIP`, ctest validation) | `level1/README.md`, per-benchmark README |
| `level2/` | 20 mini-apps with upstream build systems + `build.sh/run.sh/validate.sh` | `level2/README.md`, `level2/SCALEOUT_AUDIT.md`, `level2/tools/README.md` |
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
  concurrently. Never run two `build.sh` of the same app at once (shared src).
- NFS project storage is slow for 100k-file trees (LLVM, CP2K toolchain):
  extract/build those on `/tmp/hpcperf-*-scratch/` and keep installs/logs under
  `.deps/`. Expect stale NFS file handles on `rm -rf`; rename then delete.

## Git rules (user-mandated, non-negotiable)

- **Commit messages never carry `Co-Authored-By`, `Claude-Session` or
  "Generated with Claude Code" trailers**, whatever the harness suggests.
- Commit locally only; **never push, never open a PR, never merge** unless the
  user asks in that turn. Never push `main`, never force-push, no `gh` on the
  node, never print tokens.
- Never `git add .`/`-A`. Add files by name. Never commit `.conda_env`, `.tools`,
  `_upstream/`, `.deps/`, `build/`, binaries, tarballs or large data (`.gitignore`
  covers them, but symlinked `.conda_env`/`.tools` in worktrees are untracked --
  leave them).
- One commit per application or infrastructure change, message = what/why with
  the measured facts. Branch names follow `CONTRIBUTING.md` (`level3/<app>`,
  `env/...`, `docs/...`).
- Worktrees: the main checkout and `../HPC-Performance-AI-b2` (branch
  `level3/second-batch-bringup`) share one repository; `git worktree list`
  before assuming which branch a path is on. `.deps/`, `build/`, `_upstream/`
  are per-worktree.

## Conventions per application (Level 2/3)

```
level3/<app>/
  fetch.sh      pinned upstream + dependency sources (tag/SHA, sha256 for tarballs) into _upstream/ and .deps downloads
  build.sh      idempotent, stage-marked (.hpcperf-stage-done), per-profile, writes BUILD_INFO.txt + .hpcperf-l3-fingerprint
  run.sh        [CUDA] [args]; cases via HPCPERF_<APP>_CASE; modes smoke|strong|weak; writes run_manifest.txt
  validate.sh   [CUDA]; HPCPERF_GPUS=N; prints the criteria and PASS/FAIL, exit 0/1
  <app>_check.py  the numeric checker (uses level3/tools/l3_check.py: require_finite, ValidationError)
  patches/      *.patch with header: source, rationale, conditions, impact, verification, class
  README.md     provenance, versions, node adaptations, cases, criteria, RESULTS with dates
```

- Profiles: `.deps/level3/<app>/<profile>/{src,build,install,logs,cache}` via
  `l3_paths_profile` (`level3/tools/l3_common.sh`); app build trees under
  `build/level3/<app>/<profile>/`, runs under `.../run/<case>.<mode>.np<N>[.t<T>]/`,
  dry-runs under `.../run/.dryrun/` (never touch real results).
- `l3_common.sh` helpers you should reuse rather than reinvent:
  `l3_isolate_build_env` (strip Level 2 prefixes), `l3_clean_conda_build_env`
  (clear conda CFLAGS/LDFLAGS/AR/CMAKE_GENERATOR and `C_INCLUDE_PATH`/`LIBRARY_PATH`
  for system-GCC builds), `l3_binary_backend_check` (cuobjdump archs, works for
  static cudart), `l3_rundir`, `l3_run_id`, `l3_manifest`, `l3_fingerprint_*`,
  `l3_scale_mode`, `hpcperf_ranks`, `hpcperf_topology`, `hpcperf_forbid_args`.
- Launch only through `level2/tools/hpcperf_mpi_launch.sh --gpus N
  [--cpus-per-rank C] --bind wrapper -- <exe> ...`: one MPI rank per GPU, per-rank
  `CUDA_VISIBLE_DEVICES`, nvidia-smi audit ("N verified, 0 mismatch, 0
  unverified" is required evidence). Interface: `HPCPERF_GPUS=N|all`,
  `HPCPERF_NODES`, `HPCPERF_GPUS_PER_NODE`, `HPCPERF_CPUS_PER_RANK`,
  `HPCPERF_SCALE_MODE=smoke|strong|weak`, `HPCPERF_SITE_PROFILE`,
  `HPCPERF_DRY_RUN=1`. Requested GPUs == used GPUs; a rank count that cannot
  partition the problem is **refused**, never silently changed; 8/40/80 GPUs
  exist only as HYPOTHETICAL dry-runs (`HPCPERF_NODES=N/4`).
- Do not move the common runtime, do not refactor the launcher for an app;
  extend it only with tests in `level2/tools/tests/run_all.sh`.
- Regression campaigns never overwrite historical results: set
  `HPCPERF_L3_RUN_SUBDIR=run.regress-<sha>` (every run.sh/validate.sh builds its
  run directories under `build/level3/<app>/<profile>/$L3_RUN_SUBDIR`).
- Any tool that records its process environment (CP2K's toolchain installer,
  nsys/ncu, env-logging build systems) runs through `l3_clean_env_exec` /
  `level3/tools/l3_clean_env.sh` (allow-listed `env -i`): the login shell carries
  credentials that must never land in a `declare -x` dump or a profiler report.
  Never print a full `env` into a log; report variable names only.

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

## Current state (2026-09-06)

- Level 1: 50/50 validated. Level 2: 19/20 (MiniEM/Trilinos pending the user);
  10 MPI apps verified at 4 ranks x 4 B200; scale-out launcher in place.
- Level 3 first batch (`level3/full-apps-bringup`): LAMMPS, SPARTA, WarpX,
  SPECFEM3D, nekRS validated at 1/2/4 GPUs. Second batch
  (`level3/second-batch-bringup`, worktree `-b2`): Nyx, CP2K, QMCPACK, DFT-FE,
  GEOS validated at 1/2/4 GPUs -- `level3/SECOND_BATCH_STATUS.md`.
- Remote (`origin`): `main` = PR #3 (`level3/apps`, Level 3 prep); branches
  `level2/miniapps`, `level3/apps`, `infra/launcher-correctness` are pushed.
  **Local only**: `level3/full-apps-bringup` (first batch, 2 commits on top of
  `main`) and `level3/second-batch-bringup` (second batch, 16 more) -- the user
  decides when they are pushed or turned into PRs.
- Open items the user must decide on: QMCPACK per-walker device-memory anomaly;
  GEOS compositional-flow/well unit tests failing on the GPU build (modules
  UNVERIFIED); official Nyx/GEOS decks too small to show scaling; MiniEM/Trilinos
  for Level 2; pushing/PRs for the two Level 3 branches.

## Reporting

The user writes Chinese; answer in Chinese, keep file contents/commit messages
in English. A final report per round lists: what was completed per app; exact
versions/environment/build strategy; measured 1/2/4-GPU results with the
criteria; GPU compute/communication paths; private Toolkit/compiler
exceptions; strong/weak/size-sweep definitions and status; 40/80 dry-run and
multi-node boundary; patches, real failure causes, unfinished items; where the
raw logs/manifests/fingerprints are; local commits and the decisions needed
from the user. State failures plainly with their cause; never soften a FAIL.
