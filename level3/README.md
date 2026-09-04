# Level 3: Production / End-to-End HPC Applications

Level 3 covers production or end-to-end HPC applications: multi-GPU (and,
once the site transport is validated, multi-node) runs through the common
launcher, real input decks, job scripts, and their own dependency stacks.

Status: **integration in progress** on branch `level3/apps`; no application
is integrated yet (the application list is being confirmed). Nothing in this
directory is claimed validated.

## Conventions

- **Dependency profile.** Level 3 builds its dependencies into its own
  profile, never into the validated Level 2 tree:
  ```bash
  export HPCPERF_DEPS_PROFILE=level3      # .deps/level3/{build,install,logs}; sources in .deps/src are shared
  source hpcperf_env.sh                    # exposes ONLY the level3 prefixes (level2 ones are removed)
  ```
  Switching back (`HPCPERF_DEPS_PROFILE=level2 source hpcperf_env.sh`) removes
  the level3 prefixes again; user/system paths are never touched. Every install
  carries a build-time `.hpcperf-fingerprint` (schema 2: profile, GPU arch,
  compiler, full CUDA version, MPI, patch hashes) and a stale fingerprint fails
  fast instead of being reused.
- **Execution model.** One MPI rank per GPU via
  `level2/tools/hpcperf_mpi_launch.sh` (`HPCPERF_GPUS=N|all`,
  `HPCPERF_CPUS_PER_RANK`, site profiles, fail-fast on oversubscription,
  expected/observed GPU audit). Applications must not hard-code `mpirun`
  options or GPU ordinals.
- **Per-application layout** (reference, adapted per app):
  ```
  level3/<application>/
  ├── README.md          provenance (repo, commit, license), changes, deps, validation, status
  ├── build.sh           builds against the level3 dependency profile
  ├── run.sh             HPCPERF_GPUS / HPCPERF_SCALE_MODE aware; launches via the common launcher
  ├── validate.sh        correctness check with a PASS/FAIL line
  ├── configs/           input decks / problem definitions (smoke / strong / weak)
  ├── jobs/              scheduler job scripts (site-specific)
  └── <sources or integration files>
  ```
- **Status vocabulary** is the same as Level 2 (Build / Smoke 1-GPU /
  Distributed model / Selectable N / 1-node multi-GPU / Multi-node / Strong /
  Weak / CUDA / HIP / constraints): a result is recorded only where it was
  actually run; 40/80-GPU and multi-node remain dry-run/BLOCKED until the
  transport is validated.
