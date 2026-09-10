# Migration: Git LFS source bundles (scheme 2, abandoned) -> external source artifacts (scheme 3)

DEPRECATED DESIGN: Git LFS source distribution was abandoned before release (decision of 2026-09-10; the
scheme-2 design existed only on the unpushed branch `level3/source-freeze`, commits 0128eab / 4bca9dc /
142aea0 of 2026-09-08; no LFS object was ever added to git or pushed, git-lfs was never installed on the
build node). The replacement is described in [EXTERNAL_ARTIFACT_DESIGN.md](EXTERNAL_ARTIFACT_DESIGN.md);
the artifacts and their status in [SOURCE_ARTIFACTS.md](SOURCE_ARTIFACTS.md).

## A. KEEP_AND_REUSE (source-freeze / workspace core, not LFS-specific)

| file | role in scheme 3 |
|---|---|
| `tools/hpcperf_source.py` | tree hashing (`hpcperf-tree-1`), manifest, credential/artifact scan rules, deterministic tar (`hpcperf-tar-1`), restricted extractor -- unchanged; the LFS-pointer helpers were replaced by cache/download helpers |
| `tools/freeze_benchmark_source.py` | freeze from `provenance/freeze_spec*.yaml` -- output redirected to the local artifact staging; writes the schema-2 lock |
| `tools/compare_source_trees.py` | equivalence proof of the frozen tree vs the validated tree (unchanged) |
| `tools/hpcperf_materialize.py` / `tools/prepare_benchmark.sh` | materialization logic (tree check, DIRTY refusal, safety checks, atomic rename) kept; artifact resolution now = `--artifact` > cache > immutable URL |
| `tools/check_workspace.py` | contract checks kept and extended (17 checks, agent mode) |
| `tools/create_agent_workspace.sh` | workspace copy logic kept; trusted baseline + `--dest` added |
| `tools/loc_report.py`, `tools/readme_source_section.py` | LOC by ownership, README/audit sections (updated to the schema-2 fields) |
| `level3/<app>/provenance/{freeze_spec*.yaml, upstream*.lock, patch_series*.txt, original_vs_baseline*.diff, SOURCE_MANIFEST*.json, LICENSES*.md, equivalence*.{json,md}, LOC*.{json,md}}` | unchanged provenance (the freezes were not repeated; the archives' content is identical) |
| `level3/<app>/provenance/source.lock*.yaml` | kept, rewritten from schema `hpcperf-source-lock-1` to `hpcperf-source-lock-2` (all provenance fields preserved; a `migration` block records the old archive path and that the sha256 is identical) |
| `level3/<app>/benchmark.yaml`, `optimization_scope.yaml` | kept; the identity block `source_bundle` (archive path) became `source_artifact` (filename, sizes, hashes, publish status) |
| `level3/tools/l3_common.sh` lock queries, `l3_require_materialized` | kept (read the schema-2 fields) |
| `tools/tests/test_source_tools.sh` | kept and rewritten for scheme 3 (62 checks) |

## B. MIGRATE_THEN_DELETE (the eleven scheme-2 archives)

Procedure per archive (`tools/artifacts/migrate_from_lfs_bundle.py`): copy `level3/<app>/archives/<old>.tar.zst`
-> `$HPCPERF_ARTIFACT_STAGING/level3/<app>/hpcperf-l3-v1/<app>[-<variant>]-hpcperf-l3-v1.tar.zst` (byte-identical;
only the name and location change) -> verify size + zstd magic + sha256 against the old lock -> full extraction
in scratch: layout, `source_tree_sha256`, entry count, escaping symlinks, secret/build-output scan
(`verify_artifact.py --full`) -> write `artifact.json` + `SHA256SUMS` in the staging entry -> rewrite the
lock to schema 2, `benchmark.yaml` identity, freeze spec version -> **prepare test**: remove the canonical
`src/`(+`deps/`), `tools/prepare_benchmark.sh level3 <app> --artifact <staging copy>` (through the cache),
`check_workspace.py` on the canonical directory -> only then delete the old archive (`--delete-old`, which
re-verifies the staging copy first). Copy first, verify, then delete -- never move-then-check.

<!-- hpcperf:migration-table:begin -->
| application | variant | old scheme-2 archive | scheme-3 artifact (staging) | size | archive sha256 | tree sha256 | copy+verify | prepare test | check | old archive deleted |
|---|---|---|---|---|---|---|---|---|---|---|
| lammps | - | `archives/source_bundle.tar.zst` | `staging/level3/lammps/hpcperf-l3-v1/lammps-hpcperf-l3-v1.tar.zst` | 110.0 MB | `4f6e1096de3a6671…` (identical) | `d7549c2f6d1b5b6c…` | PASS | PASS (prototype: manual) | PASS 17/17 | yes |
| sparta | - | `archives/source_bundle.tar.zst` | `staging/level3/sparta/hpcperf-l3-v1/sparta-hpcperf-l3-v1.tar.zst` | 20.3 MB | `ea3be3032b3d6a41…` (identical) | `63519f0e3e9ac974…` | PASS | PASS | PASS 17/17 | yes |
| warpx | - | `archives/source_bundle.tar.zst` | `staging/level3/warpx/hpcperf-l3-v1/warpx-hpcperf-l3-v1.tar.zst` | 13.6 MB | `d4eca1ca57907dd2…` (identical) | `e90d20b2a7d82634…` | PASS | PASS | PASS 17/17 | yes |
| specfem3d | - | `archives/source_bundle.tar.zst` | `staging/level3/specfem3d/hpcperf-l3-v1/specfem3d-hpcperf-l3-v1.tar.zst` | 236.8 MB | `62b9739b1440a1cb…` (identical) | `9bdc4eed3a3593e4…` | PASS | PASS | PASS 17/17 | yes |
| nekrs | hypregpu | `archives/hypregpu.source.tar.zst` | `staging/level3/nekrs/hpcperf-l3-v1/nekrs-hypregpu-hpcperf-l3-v1.tar.zst` | 45.9 MB | `83c721f69eb47bea…` (identical) | `76e6ad3cdd624a90…` | PASS | PASS | PASS 17/17 | yes |
| nekrs | cpucoarse | `archives/cpucoarse.source.tar.zst` | `staging/level3/nekrs/hpcperf-l3-v1/nekrs-cpucoarse-hpcperf-l3-v1.tar.zst` | 45.9 MB | `a0b1d0a9a8a45edb…` (identical) | `6bde03184c09b236…` | PASS | PASS | PASS 17/17 | yes |
| nyx | - | `archives/source_bundle.tar.zst` | `staging/level3/nyx/hpcperf-l3-v1/nyx-hpcperf-l3-v1.tar.zst` | 75.7 MB | `218ba208e5d270ab…` (identical) | `71b64ddffc1e5c5f…` | PASS | PASS | PASS 17/17 | yes |
| cp2k | - | `archives/source_bundle.tar.zst` | `staging/level3/cp2k/hpcperf-l3-v1/cp2k-hpcperf-l3-v1.tar.zst` | 201.2 MB | `d6776dd4fa3107d4…` (identical) | `d877d2d4de4643ce…` | PASS | PASS | PASS 17/17 | yes |
| qmcpack | - | `archives/source_bundle.tar.zst` | `staging/level3/qmcpack/hpcperf-l3-v1/qmcpack-hpcperf-l3-v1.tar.zst` | 215.0 MB | `6c1585bd027f44f4…` (identical) | `f0dbdc83677ff4a6…` | PASS | PASS | PASS 17/17 | yes |
| dftfe | - | `archives/source_bundle.tar.zst` | `staging/level3/dftfe/hpcperf-l3-v1/dftfe-hpcperf-l3-v1.tar.zst` | 174.5 MB | `0259dfec2c9459f7…` (identical) | `54bea98826f91a4c…` | PASS | PASS | PASS 17/17 | yes |
| geos | - | `archives/source_bundle.tar.zst` | `retired storage/level3/geos/hpcperf-l3-v1/geos-hpcperf-l3-v1.tar.zst` | 362.6 MB | `cc5a6bfaf27504a2…` (identical) | `208e8f98027e5a5f…` | PASS | n/a (retired, not re-materialized) | PASS 17/17 | yes |
<!-- hpcperf:migration-table:end -->

GEOS: retired from the default suite; its archive was NOT migrated into the scheme-3 staging. The
byte-identical copy (verified the same way) lives in controlled local research storage outside the staging
(`hpcperf-artifacts-retired/`, not enumerated by any catalog, `publish: never`); the lock was rewritten to
schema 2 with `redistribution_status: blocked`, `suite_status: retired`, and the worktree archive was deleted.

## C. DELETE_OBSOLETE (removed from the branch in this round)

| item | disposition |
|---|---|
| `.gitattributes` (`level3/*/archives/*.tar.zst filter=lfs ...`) | deleted (the file only carried the LFS rule) |
| `.gitignore` guard `level3/*/archives/*.tar.zst` and the "Git LFS source bundles" block | replaced by the scheme-3 block (`.artifacts/`, `.hpcperf/`, `level3/.materialize-staging/`, `*.tar.zst`) |
| `tools/hpcperf_source.py`: `LFS_POINTER_PREFIX`, `is_lfs_pointer`, `lfs_pointer_text` | deleted (replaced by artifact/cache helpers) |
| `tools/hpcperf_materialize.py`: LFS-pointer detection / `git lfs pull` hint (exit 4) | deleted (exit 4 now = artifact unavailable / unpublished / offline miss) |
| `tools/source_archives_report.py` (LFS-ready column, LFS status section, quota text) -> `level3/SOURCE_ARCHIVES.md` | deleted; replaced by `tools/artifacts/generate_release_manifest.py` -> `level3/SOURCE_ARTIFACTS.md` |
| `tools/check_workspace.py` check 10 "archive present / LFS pointer" | replaced by "benchmark.yaml identity == lock; no scheme-2 fields (source_bundle, archives/)" |
| `level3/<app>/archives/` directories and `archives/*.tar.zst` | deleted after the prepare test (table above) |
| lock fields `archive.path: archives/...`, benchmark.yaml `source_bundle.archive`, freeze spec `archive:` | dropped by the lock migration |
| `git lfs pull` user flow in `level3/README.md`, `CLAUDE.md`, per-application README sections | rewritten to `tools/prepare_benchmark.sh level3 <app>` |
| scheme-2 intermediate report (LFS sizes/quota, "LFS ready" statuses) | superseded by SOURCE_ARTIFACTS.md; the historical numbers are in the git history of the branch only |

## D. REVIEW_MANUALLY

| item | note |
|---|---|
| Git history of the branch (commits 0128eab, 4bca9dc, 142aea0 mention "Git LFS") | historical documentation of a design that was never released; the commits are kept (no history rewrite); the branch is unpushed |
| `level3/geos/` and the retired archive copy | maintainer decides whether the local retired copy is kept or deleted once GEOS is definitively replaced |
| `level3/exaca/` | replacement candidate; admission criteria and status in `exaca/README.md` |

## Residual scan (branch, tracked files)

Patterns searched: `Git LFS`, `git-lfs`, `git lfs`, `LFS pointer`, `LFS quota`, `filter=lfs`,
`archives/source_bundle`, `LFS_READY`, `LFS object`, `git lfs pull`.

<!-- hpcperf:lfs-scan:begin -->
Scan of the tracked files on 2026-09-10 (`git grep`, case-insensitive): 31 lines.

- **active dependency**: 0
- **historical documentation**: 31
  - `.gitignore:30:# the maintainer's staging / the external artifact storage (never in git, never in Git LFS)`
  - `CLAUDE.md:63:  artifacts (`*.tar.zst`) never enter git (no Git LFS either): they live in the`
  - `level3/APPLICATION_AUDIT.md:365:- official_inputs_datasets: 582 XML (`*_smoke.xml`, `*_benchmark.xml`; solidMechanics, singlePhaseFlow, compositionalM`
  - `level3/APPLICATION_AUDIT.md:371:- expected_input_data_size: 108 MB in-repo; optional LFS data and baseline tarball (sizes unpublished)`
  - `level3/README.md:127:external artifact storage (never in git, never in Git LFS):`
  - `level3/README.md:132:the abandoned Git LFS design and its migration: [LFS_TO_ARTIFACT_MIGRATION.md](LFS_TO_ARTIFACT_MIGRATION.md).`
  - `level3/cp2k/provenance/source.lock.yaml:379:  from_scheme: git-lfs source bundle (design abandoned 2026-09-10, never committed, never pushed)`
  - `level3/cp2k/provenance/source.lock.yaml:380:  old_archive_path: archives/source_bundle.tar.zst`
  - `level3/dftfe/provenance/source.lock.yaml:382:  from_scheme: git-lfs source bundle (design abandoned 2026-09-10, never committed, never pushed)`
  - `level3/dftfe/provenance/source.lock.yaml:383:  old_archive_path: archives/source_bundle.tar.zst`
  - `level3/geos/provenance/source.lock.yaml:586:  from_scheme: git-lfs source bundle (design abandoned 2026-09-10, never committed, never pushed)`
  - `level3/geos/provenance/source.lock.yaml:587:  old_archive_path: archives/source_bundle.tar.zst`
  - `level3/lammps/provenance/source.lock.yaml:161:  from_scheme: git-lfs source bundle (design abandoned 2026-09-10, never committed, never pushed)`
  - `level3/lammps/provenance/source.lock.yaml:162:  old_archive_path: archives/source_bundle.tar.zst`
  - `level3/nekrs/provenance/source.lock.cpucoarse.yaml:198:  from_scheme: git-lfs source bundle (design abandoned 2026-09-10, never committed, never pushe`
  - `level3/nekrs/provenance/source.lock.hypregpu.yaml:227:  from_scheme: git-lfs source bundle (design abandoned 2026-09-10, never committed, never pushed`
  - `level3/nyx/provenance/source.lock.yaml:193:  from_scheme: git-lfs source bundle (design abandoned 2026-09-10, never committed, never pushed)`
  - `level3/nyx/provenance/source.lock.yaml:194:  old_archive_path: archives/source_bundle.tar.zst`
  - `level3/qmcpack/provenance/source.lock.yaml:213:  from_scheme: git-lfs source bundle (design abandoned 2026-09-10, never committed, never pushed)`
  - `level3/qmcpack/provenance/source.lock.yaml:214:  old_archive_path: archives/source_bundle.tar.zst`
  - `level3/sparta/provenance/source.lock.yaml:146:  from_scheme: git-lfs source bundle (design abandoned 2026-09-10, never committed, never pushed)`
  - `level3/sparta/provenance/source.lock.yaml:147:  old_archive_path: archives/source_bundle.tar.zst`
  - `level3/specfem3d/provenance/source.lock.yaml:163:  from_scheme: git-lfs source bundle (design abandoned 2026-09-10, never committed, never pushed)`
  - `level3/specfem3d/provenance/source.lock.yaml:164:  old_archive_path: archives/source_bundle.tar.zst`
  - `level3/warpx/provenance/source.lock.yaml:175:  from_scheme: git-lfs source bundle (design abandoned 2026-09-10, never committed, never pushed)`
  - `level3/warpx/provenance/source.lock.yaml:176:  old_archive_path: archives/source_bundle.tar.zst`
  - `tools/check_workspace.py:32:     materialization marker agrees with the lock (variant, tree hash); no Git LFS metadata`
  - `tools/check_workspace.py:299:        if any(k in lock for k in ("lfs", "git_lfs")) or any(k in lock["artifact"] for k in ("lfs", "lfs_pointer")):`
  - `tools/check_workspace.py:300:            ok16 = False; d16.append("Git LFS metadata present")`
  - `tools/check_workspace.py:301:    rec(16, "artifact publish status consistent, marker agrees with the lock, no LFS metadata", ok16, "; ".join(d16))`
  - `tools/hpcperf_source.py:371:        raise SourceError(f"artifact is not a zstd stream ({os.path.basename(path)}): wrong file, LFS pointer or HTML erro`
- **obsolete**: 0
<!-- hpcperf:lfs-scan:end -->

Classification: **active dependency** = none allowed (any hit of this class fails the migration);
**historical documentation** = this file, `EXTERNAL_ARTIFACT_DESIGN.md`, README sentences saying "never in
Git LFS", `tools/artifacts/migrate_from_lfs_bundle.py` (its name and docstring describe what it migrates
from), the `migration` block of the locks; **obsolete** = everything of section C (removed).
