# Level 3 first batch -- correctness / reproducibility fixes

Round after the da35285 review. Base for this work: HEAD was
`da352857561daa8f754161a856d0d53875d6f3ad` (verified; working tree clean at
start). No application source, patch, input, or tolerance was changed to obtain
a PASS; the five applications, their patches, inputs and prior results are kept.

Shared mechanism lives in `level3/tools/l3_common.sh` (+ `l3_check.py`); the
CPU-only tests are `level3/tools/tests/` (`run_all.sh` -> `test_l3_infra.sh`,
13/13 passing).

## 1. False PASS / exit codes

| Review point | Fix | Where |
|---|---|---|
| A failed run must not PASS on a stale log | validators delete/rewrite only this-run output; run.sh removes its target log before launching; validators require the run's real exit code == 0 | all `validate.sh`; `lammps/sparta run.sh` `rm -f "$LOG"` |
| SPECFEM solver failure swallowed by `\| tee \| grep \|\| true` | solver now runs to a file and its exit code is captured directly (`rc=0; cmd > log \|\| rc=$?`); the grep is display-only afterwards | `specfem3d/run.sh` |
| Separate execution from log filtering; capture launcher/app/validator real exit codes | validators run the app into a stdout file and gate on `rc`; `l3_capture` returns the command's status, not tee's; LAMMPS/SPARTA run.sh use `\|\| rc=$?` (not `; rc=$?` which `set -e` would abort) so the code and manifest are always recorded | `l3_common.sh` `l3_capture`; every `run.sh`/`validate.sh` |
| timeout / missing file / analysis exception / nonzero -> FAIL; validate only new output | `timeout` wraps each run (rc 124 -> FAIL); missing output -> FAIL; python raises `ValidationError` -> FAIL | every `validate.sh` |

Negative test: `test_l3_infra.sh` case 3 shows a failed run (rc=1) FAILs the gate
even when a PASS-looking stale log is present; case 2 shows `l3_capture` returns
the real code (7), not tee's 0.

## 2. Numerical finiteness / completeness

| Review point | Fix |
|---|---|
| Reject NaN/Inf in data/reference/error | `l3_check.require_finite` names and rejects any non-finite compared quantity; used by LAMMPS/SPARTA/WarpX validators and the SPECFEM sample scan |
| LAMMPS: expected thermo fields + final step | requires Step 0 and Step 100 rows and the fields Temp/E_pair/TotEng/Press present, else FAIL |
| SPARTA: final step, complete stats interval, fields | requires the benchmark block to span steps 30..130 (equilibration boundary to final) and the fields Np/temp/Natt present |
| SPECFEM: all required reference traces, sampling range, comparison | requires the compared-trace count == number of REF_SEIS traces (12), and scans every produced `.semd` for non-finite samples before trusting the correlation |
| nekRS: complete cimode check set (not passed>0) | requires `passed+failed == EXPECT_CHECKS` (9) AND failed==0 AND rc==0 AND coarse-location matches the cimode |
| WarpX: final step, expected particle count, field completeness, reader robustness | reader FAILs on missing field, box/fab mismatch, a truncated FAB, or boxes not covering 100% of the domain; requires the langmuir plotfile at step 40 and the uniform-plasma NP series to reach step 10 with a constant finite count |
| Adapted-subset labelling | LAMMPS/SPARTA validators print "adapted subset" and name exactly what upstream check they re-implement |
| CPU-only negative tests | `test_l3_infra.sh` (NaN/Inf rejection, rc-gate, dry-run sentinel, patch/cache) |

Thresholds unchanged (LAMMPS 1e-8/1e-5; SPARTA Np-exact/2%/15%; WarpX 5e-2/1e-11;
SPECFEM 0.8/1%/0.01s; nekRS EPS 0.3).

## 3. Result management

| Review point | Fix |
|---|---|
| Unique run_id; full stdout/stderr, command, exit code, source/binary/input hash | `l3_run_id` + `l3_manifest` write `run_manifest.txt` per real run with run_id, exit_code, binary+input sha256, backend, ranks, sizes, timer; the launcher already logs the command and per-rank GPU audit into the captured stdout |
| backend, GPU/rank/node, CPU/GPU binding, transport, timer, validation | recorded in the manifest and in the launcher lines of the captured stdout |
| dry-run must not delete/overwrite/rewrite real results; sentinel test | `l3_rundir` routes a dry-run to a `.dryrun/` scratch dir and refuses paths outside `build/level3/`; WarpX/SPECFEM/nekRS run.sh (which `rm -rf`'d the real dir before the dry-run check) now go through it; LAMMPS/SPARTA redirect their log into `.dryrun/`. Verified live: an 8-GPU dry-run left a real `log.smoke.np1` untouched and used `.dryrun/` (`test_l3_infra` case 4 + the live sentinel run) |
| Historical UNKNOWN exit codes stay UNKNOWN | not back-filled; the review bundle already labels them UNKNOWN |

## 4. Fingerprint / cache

| Review point | Fix |
|---|---|
| Patch full path + ordered content hash, not basename | `l3_fingerprint_text` records `patch[i]=<name> sha256=<hash>` in order and a `patch_series_sha256` over the ordered contents (schema bumped l3-1 -> l3-2) |
| Missing / unhashable patch -> error | `l3_fingerprint_text` returns non-zero on a missing patch; `nekrs/build.sh` also checks each patch exists before building |
| Source-cache key includes upstream SHA + patch content hash; renamed-but-changed invalidates | nekRS src stamp is now `SHA <ordered-patch-content-hash>` (was basenames) |
| build/install/cache isolated by backend/toolchain/dependency/config | per-app `.deps/level3/<app>/{src,build,install,logs}`; nekRS is further split per variant (`hypregpu` legacy paths, others under `.deps/level3/nekrs/<variant>/` with their own build dir and JIT cache) |
| Verify binary backend before running (no HIP request on a CUDA install) | `l3_binary_backend_check` (libcudart vs libamdhip64) available in `l3_common.sh` |
| Post-hoc fingerprints marked | `l3_fingerprint_write` stamps `built=<UTC> (build-time record)`; nothing back-dates |

Negative test: `test_l3_infra` case 5 (missing patch -> error; same-name changed
content -> different series hash; empty series -> `none`).

## 5. Dependency isolation

| Review point | Fix / finding |
|---|---|
| Check LAMMPS/SPARTA actual Kokkos helper source | The recorded (polluted-env) `CMakeCache.txt` had `Kokkos_NVCC_WRAPPER`/`Kokkos_COMPILE_LAUNCHER` pointing at **Level 2's** `.deps/install/kokkos`; the actual compile/link commands had **0** Level 2 references and used the bundled `nvcc_wrapper`+includes (so the binaries were clean, the cache entry was an inert stale detection). |
| Reconfigure without Level 2 prefixes; rebuild only affected apps | `l3_isolate_build_env` strips `$R/.deps/install/` from CMAKE_PREFIX_PATH/LD_LIBRARY_PATH in every build.sh. LAMMPS + SPARTA rebuilt in the isolated env (221 s / 570 s): `CMakeCache.txt` now has 0 Level 2 refs and `Kokkos_NVCC_WRAPPER` points at the bundled Kokkos. WarpX/nekRS/SPECFEM already had 0 refs (CXX = conda/mpicxx, autotools); the isolation call was added to their build.sh too but they were not rebuilt. Re-validated LAMMPS/SPARTA 1/2/4 -> identical numbers, PASS. |
| env_profiles base/head 9/11 tracked separately, not green | unchanged Level 2 issue (conda cmake activation drops a user CMAKE_PREFIX_PATH); documented in the review bundle `50_issues/env_profiles`; NOT a Level 3 regression and NOT marked passing here. |

## 6. Status / build strategy

| Review point | Fix |
|---|---|
| Separate PASS (smoke/analytic) from COMPLETED (strong/weak) | status table distinguishes validated-correctness runs from run-completed runs; strong/weak remain COMPLETED unless a numerical criterion applies |
| printed-values-equal != full bitwise | wording corrected: LAMMPS reports the four state variables agree to printed precision, not full-state bitwise identity |
| Don't widen to untested algorithm paths | LAMMPS stays LJ; WarpX stays FFT=OFF Yee PIC; no new solver paths added |
| Spack doc vs local facts; concretization NOT_RUN | recorded in the review bundle `50_issues/build_strategy` (local Spack 1.0.0.dev0, recipe versions) and `COMPATIBILITY.md`; no concretization run this round |
| Missing container runtime is an environment limit, not a feasibility verdict | stated as such; only podman present, not evaluated |

## nekRS-specific (see COMPATIBILITY.md)

- Confirmed the six logs' `COARSE SOLVER LOCATION: CPU` against upstream source
  defaults and `ci.inc`: cimode 2 = CPU coarse (GPU main app), cimode 3 = DEVICE
  (GPU) coarse.
- The cimode-2 "9/9" validates the CUDA main app + CPU coarse only. GPU HYPRE is
  now separately verified with cimode 3 (`hypregpu` variant): 9/9, coarse=DEVICE,
  at 1 and 4 GPUs.
- The `pair`/`reverse_iterator` build errors are missing-include (visibility)
  issues; only `thrust::not1` is a genuinely removed API. Patches are labelled
  project-local (no upstream backport SHA located).
- Minimal candidate `cpucoarse` (`ENABLE_HYPRE_GPU=OFF`, **0 patches**, isolated
  variant tree, 113 s build): cimode 2 (CPU coarse) PASS 1/2/4 GPU (9/9,
  coarse=CPU, main app on distinct GPUs); cimode 3 (DEVICE requested) is
  **explicitly rejected** by nekRS (`HYPRE+DEVICE not enabled! Recompile with
  -DENABLE_HYPRE_GPU=ON`, exit 1) -- no silent CPU fallback. So the three patches
  are needed only for GPU HYPRE; the current CPU-coarse workload needs none.
- `hypregpu` (patched) cimode 3 (DEVICE/GPU coarse) PASS 1/4 GPU (9/9,
  coarse=DEVICE): the GPU HYPRE coarse solve the patches enable is verified
  correct, not merely compiled.
