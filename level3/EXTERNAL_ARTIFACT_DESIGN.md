# Level 3 source distribution: project-controlled external source artifacts (scheme 3)

Decided 2026-09-10 (meeting with Keren). Scheme 2 (Git LFS source archives, designed 2026-09-08, never
committed, never pushed) is abandoned; its migration is recorded in [LFS_TO_ARTIFACT_MIGRATION.md](LFS_TO_ARTIFACT_MIGRATION.md).
Level 1 and Level 2 are untouched (their kernel / mini-app sources stay in git).

```
PROJECT-CONTROLLED EXTERNAL SOURCE ARTIFACTS  +  AUTOMATIC MATERIALIZATION  +  SELF-CONTAINED AGENT WORKSPACE
```

## 1. Lifecycle

| step | what | where | tool |
|---|---|---|---|
| A freeze | exact upstream source (committed blobs of a pinned checkout) + approved compatibility patches (pre-applied) + benchmark-specific source dependencies (pinned git trees / sha256-pinned release tarballs) -> deterministic `<app>[-<variant>]-<source_version>.tar.zst` | maintainer's local artifact staging (`$HPCPERF_ARTIFACT_STAGING/level3/<app>/<source_version>/`, outside the git worktree) | `tools/freeze_benchmark_source.py` (spec `level3/<app>/provenance/freeze_spec*.yaml`) |
| B publish | the artifact goes to project-controlled external artifact storage (GitHub Release assets, institutional/S3-compatible object storage, Zenodo, other project-controlled https hosting -- provider NOT decided yet); the lock then records the immutable https URL | external storage | `tools/artifacts/publish_artifacts.sh` (this round: `--dry-run` only, no adapter, nothing uploaded) |
| C git | harness + contract + provenance only: `README.md benchmark.yaml optimization_scope.yaml build.sh run.sh validate.sh inputs/ references/ configs/ provenance/{source.lock.yaml, SOURCE_MANIFEST.json, upstream.lock, patch_series.txt, original_vs_baseline.diff, LICENSES.md, equivalence.md, LOC.md}` -- never `src/`, `deps/`, an archive or an LFS pointer | GitHub repository | -- |
| D materialize | `tools/prepare_benchmark.sh level3 <app>`: source.lock -> local content-addressed cache -> (cache miss) immutable URL -> size + sha256 -> restricted extraction outside the benchmark directory -> `source_tree_sha256` -> safety scan -> atomic `level3/<app>/{src,deps}` | user clone | `tools/prepare_benchmark.sh` / `tools/hpcperf_materialize.py` |
| E optimize | `tools/create_agent_workspace.sh level3 <app> <run-id>` -> `workspaces/<run-id>/level3/<app>/` (real copies, never symlinks to the canonical tree); the agent's cwd; iteration 0 must pass `tools/check_workspace.py`; later iterations are checked in agent mode and validated by `tools/validate_workspace.sh` (trusted harness) | per run | `tools/create_agent_workspace.sh`, `tools/check_workspace.py`, `tools/validate_workspace.sh` |

Acceptance criterion of the scheme: **every Level 3 benchmark becomes a complete source-level workspace
before optimization starts; the agent modifies, builds, runs and validates entirely inside that workspace,
except for explicitly declared system/runtime dependencies (CUDA/ROCm, compilers, MPI, Slurm, driver, site
UCX/libfabric, system runtime libraries, `.conda_env`).**

## 2. Artifact format

- Name: `<app>-<source_version>.tar.zst` (`lammps-hpcperf-l3-v1.tar.zst`); a variant carries its own artifact
  when its tree differs (`nekrs-hypregpu-hpcperf-l3-v1.tar.zst`, `nekrs-cpucoarse-hpcperf-l3-v1.tar.zst`).
- Top-level entries: `src/` (application source + upstream-bundled dependency source, patches applied),
  `deps/` (benchmark-specific source dependencies), optionally `ARTIFACT_MANIFEST.json` (descriptive, not part
  of the identity). Never `level3/<app>/...` prefixes.
- Deterministic: GNU tar written by the tool (`hpcperf-tar-1`: entries sorted by path bytes, mtime fixed to
  2024-01-01T00:00:00Z, uid/gid 0, empty user/group names, mode 0755/0644, symlinks kept), `zstd -19
  --single-thread` (version recorded). Re-freezing identical content reproduces `source_tree_sha256` and, with
  the same zstd, the archive sha256 (`--verify-determinism` builds the archive twice and compares).
- Content rules: source only, from an allowlist (committed blobs of pinned checkouts, declared submodules,
  sha256-pinned files); no build/runtime/profiler output, no binaries, no credentials, no session logs, no
  `.env`/`toolchain.env`, no absolute-path dependency. The freeze scans names and contents (rule names + paths
  only are reported; a hit fails the freeze) and refuses symlinks escaping the tree.
- Two identities, recorded separately: `archive_sha256` (the file) and `source_tree_sha256` (algorithm
  `hpcperf-tree-1`: sorted paths, file content / symlink target; no mtime, uid, mode).

## 3. `provenance/source.lock[.variant].yaml` (schema `hpcperf-source-lock-2`)

```yaml
schema: hpcperf-source-lock-2 / schema_version: 2
benchmark: {name, level: 3, application, variant, source_version}
upstream: {repository, tag, commit (40-hex)}
artifact: {filename, format: tar.zst, layout: [src/, deps/], size, sha256, source_tree_sha256, tree_algorithm,
           tar_format, compression: {tool: zstd, level: 19, single_thread, version}, uncompressed_size,
           tar_entries, file_count, symlink_count, immutable: true,
           primary: {url: null, status: unpublished} | {url: https://.../<filename>, status: published},
           mirrors: [https://...]}
cache: {content_addressed: true, layout: sha256/<archive_sha256>.tar.zst, env: HPCPERF_ARTIFACT_CACHE, default: $HPC_PERFORMANCE_AI_ROOT/.artifacts}
materialized_tree: {sha256, algorithm, entries, layout}
patches: [{path, sha256, category, upstream_reference, component, files}]
dependencies: {bundled, benchmark_specific, environment_provided}
components, equivalence, licenses, license_notes
redistribution_status: cleared | blocked | review     (+ redistribution_notes)
source_scope: {application_owned, bundled, benchmark_specific, test, exclude}   (from optimization_scope.yaml)
scan_allow, freeze: {tool_version, timestamp}, migration (for locks migrated from scheme 2)
```
`tools/hpcperf_lock.py:validate_lock` rejects: a non-40-hex commit, a filename outside the convention, a URL
while `unpublished`, a non-https or floating (`latest`, branch) URL, a URL not ending with the filename, an
unknown status, any `/tmp/`, `/home/`, `/projects/`, `$HOME`, `file://` location in the file, Git LFS keys.
Unpublished artifacts are recorded as `{url: null, status: unpublished}` -- never a fabricated future URL.

`benchmark.yaml` carries the derived identity (`source_version`, `source_tree_sha256`, `source_artifact:
{filename, archive_sha256, source_tree_sha256, size, uncompressed_size, file_count, source_version, primary_url,
publish_status}` or `variants.<v>.{...}`), `upstream_version`, `redistribution_status`, `suite_status`
(retained | candidate | retired).

## 4. Local staging vs publication

- `HPCPERF_ARTIFACT_STAGING` (maintainer side, outside the worktree, not `/tmp` as the only copy):
  `level3/<app>/<source_version>/{<artifact>, artifact.json, SHA256SUMS}`; `artifact.json` records per artifact
  size, sha256, tree sha256, origin (freeze | migrated-from-lfs-bundle), `status: LOCAL_ARTIFACT_VERIFIED`,
  `remote_status: REMOTE_ARTIFACT_UNPUBLISHED`. Users never depend on the staging path.
- Statuses: `FROZEN` -> `LOCAL_ARTIFACT_VERIFIED` (`verify_artifact.py --full`: size, magic, sha256,
  extraction, layout, tree hash, scan) -> `LOCAL_MATERIALIZATION_VERIFIED` (prepare + check_workspace on the
  canonical directory) -> `AGENT_WORKSPACE_VERIFIED` (real closed loop) ; remote: `REMOTE_ARTIFACT_UNPUBLISHED`
  until a published artifact has been downloaded again and verified (`REMOTE_FETCH_VERIFIED`). `REMOTE_READY`
  is not a status.
- Catalog `tools/artifacts/artifact_catalog.yaml`: suite membership (retained / candidate / retired), naming,
  publish requirements. `generate_release_manifest.py` -> `tools/artifacts/release_manifest.json` +
  [SOURCE_ARTIFACTS.md](SOURCE_ARTIFACTS.md). `publish_artifacts.sh --dry-run` prints per artifact:
  application, variant, local path, source version, size, archive sha256, tree sha256, intended remote
  filename, intended release, license status, publish status, PLAN/REFUSE. REFUSE when
  `redistribution_status != cleared`, hash/size mismatch, tree mismatch, scan failure, benchmark.yaml/lock
  mismatch, missing artifact, retired application, already published. A real publish is refused until the
  maintainer selects the provider and an adapter exists; `prepare_benchmark.sh` will keep consuming plain
  immutable https URLs (no `gh`/`aws` CLI required from users).

## 5. Content-addressed cache and `prepare_benchmark.sh`

`HPCPERF_ARTIFACT_CACHE` (default `<repo>/.artifacts/`, gitignored): `sha256/<archive_sha256>.tar.zst`,
entries are real copies (never hard links to the staging) made read-only after verification; downloads stream
into `.partial/` and are size/magic/sha256-checked before the atomic rename -- a failed transfer never becomes a
cache entry. Resolution order: `--artifact FILE` (verified, then copied into the cache) > cache > `primary.url`
> `mirrors`; `--offline` forbids remote fetches (cache miss = exit 4). Every step verifies sha256; a file name
or size alone never qualifies an artifact.

Materialization (`tools/hpcperf_materialize.py`): existing `src/`/`deps/` are re-hashed -- identical:
`READY`; different: `DIRTY` (exit 3; never overwritten without `--force-rematerialize`); extraction happens in
`level3/.materialize-staging/` (outside the benchmark directory, same filesystem) with a restricted extractor
(only `src/`, `deps/`, `ARTIFACT_MANIFEST.json`; no absolute paths, `..`, hard links, devices), then
`source_tree_sha256`, layout, escaping-symlink, credential/env-dump/session/profiler/build-output scans, then
the atomic renames and the marker `.hpcperf-materialized.yaml` (variant, source version, tree hash, artifact
name/sha256, origin, status). Statuses printed: `NOT_PREPARED, LOCAL_ARTIFACT_VERIFIED,
REMOTE_FETCH_VERIFIED, TREE_VERIFIED, MATERIALIZED, READY, DIRTY, INVALID`. Exit codes: 0, 1 (INVALID/error),
2 (usage), 3 (DIRTY), 4 (artifact unavailable / unpublished / offline miss), 5 (hash/size/tree mismatch,
unsafe content).

After a successful prepare the benchmark directory is source-level self-contained: `build.sh`/`run.sh`/
`validate.sh` read only `$HERE/src` and `$HERE/deps` (`l3_require_materialized` fails plainly with the prepare
hint when they are absent; nothing is fetched, cloned or patched at build time; `_upstream/`, another checkout,
`.deps/.../src`, a home directory or `/tmp` are never sources). Builds that write into their source tree use a
build-side copy (SPECFEM3D, nekRS, DFT-FE, GEOS, CP2K toolchain). The tests simulate "artifact, cache and
staging unavailable after prepare" by design: build/run/validate never touch them.

## 6. Agent workspace and the trusted harness

- `tools/create_agent_workspace.sh level3 <app> <run-id> [--dest DIR] [--variant V] [--link-prebuilt-deps]`:
  a self-contained root (`hpcperf_env.sh`, `level2/tools`, `level3/tools` copies; `.conda_env`, `.tools`,
  `.deps/install` environment symlinks) with `level3/<app>/` as a REAL copy (`cp --reflink=auto`, else copy)
  of `src/ deps/ build.sh run.sh validate.sh benchmark.yaml optimization_scope.yaml inputs/ references/
  configs/ provenance/`; never `src -> ../../canonical/src`. Builds, installs and results are private to the
  run (`<root>/build/`, `<root>/.deps/level3/<app>/`); different models/runs/iterations share nothing writable.
  Recorded: `workspace.yaml` (run_id, benchmark, variant, source_version, canonical_source_tree_sha256,
  workspace_initial_tree_sha256, creation_timestamp, iteration 0) and `workspace_baseline.json` (sha256 of every
  file of the benchmark copy; read-only, outside the agent's cwd, mirrored to `<repo>/.hpcperf/workspace_baselines/`).
- PRE-AGENT (iteration 0): `check_workspace.py` baseline mode -- the source hash must equal the canonical
  `source_tree_sha256` (17 checks, all PASS).
- POST-AGENT: `check_workspace.py --agent-mode --iteration N`: files inside `optimization_scope.yaml:modifiable`
  may differ (modified/added/deleted lists, `initial_source_hash`, `current_source_hash`, diff recorded);
  every readonly/excluded/unclassified file (`validate.sh`, `run.sh`, `build.sh`, `benchmark.yaml`,
  `optimization_scope.yaml`, `provenance/**`, `inputs/**`, `references/**`, dependency source) must still equal
  the trusted baseline -> otherwise READONLY TAMPERING. Neither "one changed line blocks every build" nor
  "prepare runs before build and overwrites the agent's edit" can happen: build.sh never calls prepare, and the
  agent-mode check is the gate the harness uses.
- `tools/validate_workspace.sh level3 <app> <workspace-benchmark-dir> [--iteration N] [--skip-build] [-- args]`
  (trusted harness): agent-mode check against the trusted baseline (repository copy preferred) -> REFUSED
  (exit 3, nothing runs) on tampering; otherwise build inside the workspace (incremental: only changed files
  recompile), run the unchanged validator on the workspace's own binary, record
  `reports/iter-N.{check.json,diff,verdict.yaml}` (verdict, exit code, source hashes, modified files, binary
  sha256 from `run_manifest.txt`).

## 7. Versioning

Any change of application source, patch series, benchmark-specific or bundled dependency source produces a
new `source_tree_sha256` and a new `source_version` (`hpcperf-l3-v1` -> `hpcperf-l3-v2`) and therefore a new
artifact name; published artifacts are immutable (a same-named artifact is never overwritten; the freeze
refuses a staging entry with a different sha256 under the same name). Metadata-only changes (lock fields,
README, catalog) do not touch the source identity. The release name proposed for the first publication is
`level3-source-hpcperf-l3-v1` (catalog); provider, naming, visibility and timing are the maintainer's decision.

## 8. License / redistribution

An external artifact is a redistribution. Before an artifact enters the publish plan: license audit
(`provenance/LICENSES*.md`, `redistribution_status: cleared` in the lock), secret scan, archive hash, tree hash
and provenance checks all PASS. GEOS is retired from the default suite (ParMETIS 4.0.3 redistribution
constraint + replacement decision): no scheme-3 artifact, `redistribution_status: blocked`, `suite_status:
retired`, code/provenance/results kept in git, its old archive kept only in controlled local research storage.

## 9. Tools

| tool | role |
|---|---|
| `tools/hpcperf_source.py` | tree hash, manifest, scan rules, deterministic tar, restricted extractor, cache/download helpers |
| `tools/hpcperf_lock.py` | lock schema 2: build, validate, identity for benchmark.yaml, staging metadata |
| `tools/freeze_benchmark_source.py`, `tools/compare_source_trees.py` | freeze (spec-driven) + equivalence proof against the validated tree |
| `tools/prepare_benchmark.sh` -> `tools/hpcperf_materialize.py` | user-side materialization (cache, download, verify, atomic placement) |
| `tools/artifacts/verify_artifact.py` | verify an artifact against its lock (staging entry, cache entry or file) |
| `tools/artifacts/migrate_from_lfs_bundle.py` | one-time scheme-2 -> scheme-3 migration (copy, verify, rewrite lock, delete old) |
| `tools/artifacts/artifact_catalog.yaml`, `generate_release_manifest.py`, `publish_artifacts.sh` (+ `publish_plan.py`) | suite catalog, release manifest / SOURCE_ARTIFACTS.md, publish preflight |
| `tools/create_agent_workspace.sh`, `tools/check_workspace.py`, `tools/validate_workspace.sh` | workspace, contract checks (baseline / agent mode), trusted validation |
| `tools/loc_report.py`, `tools/readme_source_section.py` | LOC by ownership, README/audit sections |
| `tools/tests/test_source_tools.sh` | 62 static checks of all of the above (run by `level3/tools/tests/run_all.sh`) |
