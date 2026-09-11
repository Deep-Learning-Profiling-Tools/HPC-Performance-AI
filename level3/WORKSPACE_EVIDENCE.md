# Level 3 evidence matrix: source, materialization, build, agent edit, science, GPU binding, multi-GPU, remote

Updated 2026-09-11 (third update of that day: ExaCA upstream-test attribution) for HEAD of `level3/source-freeze` (node dgx003, 4x B200, CUDA 13.2.78, Slurm job 9773959).
Each column is a separate claim with its own evidence; a green cell in one column never implies another.
Levels: **VERIFIED** (evidence reproduced by a recorded command on this branch), **HISTORICAL** (result of an
earlier bring-up on a tree proven content-equivalent to the artifact, `provenance/equivalence*.md`; not re-run),
**NOT_RUN** (no evidence of that kind exists), **UNVERIFIED** (claimed by upstream, not exercised here).

| app (variant) | source equivalence + materialization | canonical build from the artifact | independent workspace build (Ninja incremental) | agent-edit effective (recompile + validate) | scientific validation (version / profile) | GPU binding audit | multi-GPU in a workspace | remote fetch |
|---|---|---|---|---|---|---|---|---|
| lammps | VERIFIED (equivalence 13,893 identical; prepare from staging via cache; check 17/17) | HISTORICAL (2026-09-07 regression fc4d2a1 from the equivalent `_upstream` tree, CMake/Ninja) | **VERIFIED** loop-001 outside the repository: build FAIL on injected `#error`, restore -> 90 s incremental build | **VERIFIED** (new binary sha256 after a one-line edit, validate PASS; build provenance `built_this_iteration`) | VERIFIED in the workspace: 1 GPU + 2 GPU smoke PASS (2026-09-10/11, upstream reference log, bit-identical); 4 GPU HISTORICAL (fc4d2a1) | VERIFIED 2 ranks (separate 3000-step run: 2 verified / 0 mismatch); the ~1 s validation runs end before the sampler observes them and are reported `unverified` (observation gap, not a mismatch) | VERIFIED 2 GPU (iteration 7); 4 GPU NOT_RUN in the workspace | NOT_RUN (unpublished) |
| sparta | VERIFIED (prepare test 2026-09-10, check 17/17) | HISTORICAL (fc4d2a1, CMake) | NOT_RUN | NOT_RUN | HISTORICAL 1/2/4 GPU PASS (fc4d2a1, 2026-09-07) | HISTORICAL (regression audits) | NOT_RUN | NOT_RUN |
| warpx | VERIFIED | HISTORICAL (fc4d2a1, CMake + AMReX from deps/) | NOT_RUN | NOT_RUN | HISTORICAL 1/2/4 GPU PASS (fc4d2a1) | HISTORICAL | NOT_RUN | NOT_RUN |
| specfem3d | VERIFIED | HISTORICAL (fc4d2a1; autotools in-tree -> build-side copy of src/) | NOT_RUN (Make/autotools: incremental behaviour of an agent edit not exercised) | NOT_RUN | HISTORICAL 1/2/4 GPU PASS (fc4d2a1) | HISTORICAL | NOT_RUN | NOT_RUN |
| nekrs (hypregpu) | VERIFIED (both variants materialized in turn, check 17/17 each) | HISTORICAL (fc4d2a1; build-side copy) | NOT_RUN | NOT_RUN | HISTORICAL 1/2/4 GPU PASS cimode 2 and 3 (fc4d2a1) | HISTORICAL | NOT_RUN | NOT_RUN |
| nekrs (cpucoarse) | VERIFIED | HISTORICAL (fc4d2a1) | NOT_RUN | NOT_RUN | HISTORICAL 1/2/4 GPU PASS cimode 2 (fc4d2a1) | HISTORICAL | NOT_RUN | NOT_RUN |
| nyx | VERIFIED | HISTORICAL (fc4d2a1, CMake + AMReX/SUNDIALS) | NOT_RUN | NOT_RUN | HISTORICAL: MiniSB + LyA-adiabatic 1/2/4 GPU PASS (re-run 2026-09-07), LyA heat/cool **PENDING** (STATE_AND_PARTICLES_PASS; I_R_CHECK_PENDING) -- unchanged | HISTORICAL | NOT_RUN | NOT_RUN |
| cp2k | VERIFIED | HISTORICAL (2026-09-05/06, toolchain on /tmp scratch + make; not re-run at fc4d2a1) | NOT_RUN (make-based; toolchain copy) | NOT_RUN | HISTORICAL 1/2/4 GPU PASS (2026-09-05/06) | HISTORICAL | NOT_RUN | NOT_RUN |
| qmcpack | VERIFIED | HISTORICAL (2026-09-05/06, CMake, private LLVM offload toolchain) | NOT_RUN | NOT_RUN | HISTORICAL 1/2/4 GPU PASS (2026-09-05/06); walker-memory scale limit recorded -- unchanged | HISTORICAL | NOT_RUN | NOT_RUN |
| dftfe | VERIFIED | HISTORICAL (2026-09-05/06; build-side copy for `include/git_info.h`); the fingerprint text changed at the freeze (`bundle_tree=`), so the next build.sh run will refuse the existing install until it is removed | NOT_RUN | NOT_RUN | HISTORICAL 1/2/4 GPU PASS (2026-09-05/06) | HISTORICAL | NOT_RUN | NOT_RUN |
| geos (RETIRED) | VERIFIED (check 17/17 on the local tree; artifact in retired storage only) | HISTORICAL (2026-09-05/06) | NOT_RUN | NOT_RUN | HISTORICAL 1/2/4 GPU PASS; 5 flow/well unit tests FAIL -- unchanged, out of the suite | HISTORICAL | NOT_RUN | never (not published) |
| exaca | VERIFIED (freeze 2026-09-10, equivalence 109 + 1476 identical, check 17/17) | **VERIFIED** (109 s from the materialized artifact: Kokkos 4.7.04 + json + ExaCA, CMake/Ninja) | NOT_RUN (no agent workspace created yet) | NOT_RUN | project-defined statistical validation of the dirsolid smoke case: calibration 2026-09-10 (8 runs) + independent holdout 2026-09-11, 9/9 PASS (`exaca/README.md`, `references/holdout.json`). Upstream tests: kernel unit tests 30/52 as upstream runs them, all 22 failures attributed (7 test-fixture, 13 host-space-in-CUDA-build, 2 CTest GPU-binding, 0 application, 0 unresolved) and passing after test-side patches / in a Serial-only build; upstream's two small **full-application** cases PASS at 1/2/4 GPUs within upstream tolerances (`references/upstream_unit_test_matrix.json`). Release acceptance: ON_HOLD -> restored 2026-09-11 | VERIFIED 1/2/4 ranks (calibration and holdout runs: N verified / 0 mismatch) | NOT_RUN (canonical directory only) | NOT_RUN |

Notes.
- "Canonical build from the artifact" for the nine migrated applications is HISTORICAL by construction: the
  builds that produced the recorded validations read the `_upstream` checkouts or the private patched copies,
  which the freeze proved byte-equivalent to the artifacts (`provenance/equivalence*.md`, 14/14 EQUIVALENT); the
  build.sh migrations to `$HERE/src` were checked statically (check 7/15) and, for LAMMPS, exercised in the
  workspace. A rebuild of the other eight from `src/` was not performed this round (not requested; hours of
  dependency builds).
- The LAMMPS Ninja closed loop is not extrapolated: Make/autotools applications and the build-side-copy
  applications (SPECFEM3D, nekRS, DFT-FE, CP2K, GEOS) keep NOT_RUN for the agent-edit column.
- GPU binding: the launcher audit samples `nvidia-smi --query-compute-apps` while the ranks run; runs shorter
  than the sampling window report "unverified" (an observation gap), never "verified". Only runs with
  "N verified, 0 mismatch" count as binding evidence.
- "Workspace ready" means: `src/` (and, for the selected variant, `deps/`) materialized with the marker of that
  variant, `check_workspace` PASS 17/17, and -- for iterations > 0 -- the trusted baseline copy under
  `<repo>/.hpcperf/workspace_baselines/`. A bare `src/` directory is not readiness. An agent's legitimate
  modifications inside the modifiable scope are never reverted to iteration 0 by any check.
- Remote fetch has two recorded steps, both anonymous and with an empty cache
  (`tools/artifacts/remote_fetch_check.sh`): `--mode plan-url` verifies a freshly published asset against the
  **lock's** expected size/hashes (the URL comes from the reviewed plan, the identity never does), and
  `--mode lock-entry` proves the ordinary user entry (clean clone of the lock-update commit + plain
  `tools/prepare_benchmark.sh`). `REMOTE_FETCH_VERIFIED` means the second one passed. Nothing is published, so
  both are NOT_RUN for every artifact.
- Build provenance of a validated binary is recorded per iteration by `tools/validate_workspace.sh`:
  `built_this_iteration`, `verified_from_build_record` (a trusted build record for the same source hash,
  backend and variant contains that binary) or `UNVERIFIED`. A `--skip-build` iteration never asserts that the
  current source edit was compiled: iteration 8/9 of the LAMMPS workspace are recorded as `UNVERIFIED` for
  exactly that reason, even though their numerical verdict is PASS.
- Every iteration record lists the run manifests that were appended **during that iteration** (run id, ranks,
  exit code, binary sha256, launcher audit), so a verdict can never inherit an older run's binary.
