# Distributing the golden store (`tools/golden/golden_artifacts.py`)

The frozen references (`level<N>/<benchmark>/reference/*.json`, committed) name the raw bytes they compare
against by sha256; those bytes live in the content-addressed golden store
(`$HPCPERF_GOLDEN_STORE`, default `<repo>/.golden-store/sha256/<aa>/<sha256>.bin`, git-ignored). A fresh clone has
no store, so every tolerance-rule golden check FAILs with "golden blob ... missing" (by design: there is no
fallback). This tool distributes the store as a small set of deterministic bundles, the same way Level 3
distributes its source artifacts (`level3/EXTERNAL_ARTIFACT_DESIGN.md`): local staging outside the worktree,
a committed lock, GitHub Release assets once the maintainer publishes, sha256 verification at every step.

State on 2026-10-27 (HEAD `cb5f78f`): 82 committed references name 103 distinct blobs, 8,550,402,964 bytes,
packed into **30 bundles, 2,747,748,740 bytes** (zstd -3, 31.5 % of the 8,718,643,200 tar bytes).

## User workflow: clone, fetch, verify, run

```bash
git clone ... && cd HPC-Performance-AI && source hpcperf_env.sh
python3 tools/golden/golden_artifacts.py fetch                      # every committed reference (2.7 GB download, 8.6 GB store)
python3 tools/golden/golden_artifacts.py fetch level1/adam level3/nyx   # or only what some benchmarks need
python3 tools/golden/golden_artifacts.py verify                     # every named blob present, size + sha256 correct
cmake --build build/adam/cuda && ctest ...                          # the golden checks now find their blobs
```

`fetch` resolves each needed bundle from `--from DIR` (a directory of bundle files, e.g. the maintainer's
staging), then the content-addressed cache (`$HPCPERF_ARTIFACT_CACHE`, default `<repo>/.artifacts`,
`sha256/<bundle sha256>.tar.zst`, shared with Level 3), then the lock's published URL and mirrors
(`hpcperf_source.download_to_cache`, the code path of `tools/prepare_benchmark.sh`; anonymous https, nothing
else is contacted; `--offline` forbids it). Until the bundles are published only `--from`/the cache work, and a
missing bundle is reported as unavailable (exit 4). Selections: a level (`level1`), a benchmark
(`level1/adam`) or one reference file. Blobs already present and correct are skipped (a second `fetch` reads
no bundle); a corrupted blob is replaced (REPAIRED). Downloads stay in the cache unless `--drop-downloads`.

| exit | meaning |
|---|---|
| 0 | everything selected is present and verified |
| 1 | failure (verify: something missing or corrupted; lock-check: stale/invalid lock) |
| 2 | usage (e.g. a selection that matches no committed reference) |
| 4 | a needed bundle is unavailable: not in `--from`, not cached, unpublished, `--offline`, or the blob is in no bundle of the lock (`NOT_IN_LOCK`: the lock is stale) |
| 5 | integrity: a bundle/member/lock hash or size mismatch, unsafe or unexpected archive content |

## Trust model and the order of checks

The **committed references are the root of trust** (read from `git ls-tree HEAD`, never from the working tree:
the same rule `hpcperf_golden.py check` applies to a reference). The lock adds the bundle identities and
locations; it cannot make a wrong blob acceptable:

1. the lock is validated (schema, every bundle's `members_digest` recomputed from its member list, the bundle
   name derived from group/part/digest, `tar_size` from the member sizes, publish states, https and
   non-floating URLs ending with the bundle name, no URL while unpublished, no node-private path);
2. the bundle file's size, zstd magic and sha256 equal the lock **before** extraction;
3. the stream is decompressed through `zstd -d` into a restricted reader: every member must be a regular file
   named `sha256/<aa>/<sha256>.bin`, listed in the lock's members of this bundle, of the recorded size, not
   duplicated, without pax headers; links, devices, directories, absolute or `..` paths are refused before a
   byte is written; the decompressed size must equal `tar_size` (bomb guard) and trailing data must be zero;
4. every member is written to `<store>/.incoming/<run>/`, fsync'ed, and its content sha256 must equal its name;
   its size must equal the size the committed reference records (`bytes` / `archive.size`);
5. the compressed stream is hashed again while it is decompressed (the file must not change between 2 and 3);
6. only then is each blob renamed atomically into `sha256/<aa>/` (mode 0444). A failure at any step installs
   nothing from that bundle; a bundle member that no committed reference names is not installed.

## Bundle format (`hpcperf-golden-tar-1`)

* one group per benchmark (`level1/adam`, `level3/nyx`, ...); a group is split into parts when a bundle could
  exceed the GitHub release-asset limit (2 GiB): first-fit decreasing by blob size, a part takes blobs while
  `ZSTD_compressBound(tar size) < 2^31`, so the limit holds for any data (today only `burgers_equation`
  splits: one 2,125,440,000-byte blob alone, two 531 MB blobs together). A single blob too large for one asset
  is refused (blob splitting is not implemented; the largest blob today, 2,125,440,000 bytes, is 13.7 MB below
  that bound);
* a bundle is self-contained: a blob shared by two benchmarks (3 blobs of `energy` and `pressure`) is in both
  bundles (168 MB of duplicated tar bytes, ~0 after compression);
* name `golden-<level>-<benchmark>[-p<i>of<n>]-<members_digest[:16]>.tar.zst`; `members_digest` = sha256 over
  the format, the compression flags, the group and the sorted `(sha256, size)` members, so a different member
  set always has a different name and a published name never needs to change its bytes;
* POSIX ustar written by the tool itself (not by Python's tarfile, so the bytes do not depend on the Python
  version): regular-file members only, sorted by name, mode 0644, uid/gid 0, empty user/group, mtime
  1704067200 (2024-01-01, as `hpcperf-tar-1`), no directory entries, two zero blocks, padded to 10240 bytes;
  readable by GNU tar;
* compressed with `zstd -3 --single-thread` from a stdin stream. Level 19 was measured on 64 MB slices of the
  largest blobs: 0-4 percentage points smaller (89 % -> 85 % at best) at 50-65x the time, so -3.

**Reproducibility, measured**: three packs of the real store (two runs plus `--verify-determinism`, which packs
every bundle twice) gave byte-identical bundles, SHA256SUMS and lock (lock sha256 `b1a79520...`); Python
3.13 instead of 3.12 gave the identical lock. zstd 1.5.6 instead of 1.5.7 changed 23 of the 30 bundles: the
compressed bytes are reproducible for a fixed zstd version only, which is recorded in the lock
(`bundle_format.compression.version`; pack warns when it differs and refuses to give a published name other
bytes). Consumers never depend on this: they verify the sha256 of whatever file they got.

## The lock (`tools/golden/golden_artifacts.lock.yaml`, schema `hpcperf-golden-lock-1`)

No timestamp, host or path: the same committed references, store and zstd give the same lock bytes.

```yaml
schema: hpcperf-golden-lock-1
reference_scan: {source, schemas_with_blobs, references_with_blobs, digest}
bundle_format: {naming, tar: {format: hpcperf-golden-tar-1, ...}, compression: {tool: zstd, level: 3, flags, version},
                asset_limit_bytes: 2147483648, partition}
cache: {env: HPCPERF_ARTIFACT_CACHE, default: <repo>/.artifacts, layout: sha256/<bundle sha256>.tar.zst}
release: {provider: github-release, proposed_tag: golden-hpcperf-g1-rc1, status: unpublished}
totals: {bundles, distinct_blobs, blob_bytes, member_entries, bundle_bytes}
bundles:
- {name, group, part, parts, sha256, size, tar_size, members_digest, members: [{sha256, size}],
   content_scan: {archive_blobs, credential_hits, node_private_members},
   redistribution: {status: review | cleared | blocked, notes},
   primary: {url: null, status: unpublished} | {url: https://.../<name>, status: published}, mirrors: []}
references:                                   # reference -> its blobs and bundles: fetch reads only what a selection needs
  level1/adam/reference/n10000-t200.json: {json_sha256, schema, blobs: [...], bundles: [...]}
```

`lock-check` compares the lock with the committed references: the bundle layout the references require must be
exactly the lock's (otherwise STALE: re-pack); a reference whose JSON changed without a blob change is only a
"metadata drift" note. A re-pack carries `primary`, `mirrors` and `redistribution` over for every bundle whose
name and sha256 are unchanged.

## Maintainer workflow (nothing here uploads)

1. `hpcperf_golden.py make ...` / `nyx_reference.py` freeze a reference; commit the reference JSONs.
2. `python3 tools/golden/golden_artifacts.py pack --staging "$HPCPERF_ARTIFACT_STAGING/golden" [--verify-determinism]`
   (reads the committed tree; refuses a missing or corrupted blob, an unknown schema that mentions the store,
   credential-looking content in an archive blob, a staging inside the worktree, other bytes under an existing
   staged name); writes the bundles, `SHA256SUMS`, `golden_bundles.json` (status `LOCAL_ARTIFACT_VERIFIED`:
   every staged bundle was read back member by member) and the lock. 20 s for the whole store on dgx003.
3. `golden_artifacts.py lock-check` must PASS; commit the lock (with the references or right after them).
4. Redistribution review: every bundle starts as `review`; the maintainer sets `cleared` (or `blocked`) in the
   lock. The pack notes what it found: the 4 Nyx archives carry `plt00010/job_info` text with node-private
   build/run paths (`/projects/...`, build directory, compiler paths) and the build host name. They are part of
   the frozen bytes (changing them means regenerating the Nyx references), so the choice is: publish as is,
   regenerate without them, or keep the Nyx bundle unpublished.
5. `golden_artifacts.py plan --tag <tag> --commit <40-hex> --json ...` lists PLAN/REFUSE per bundle (staged file
   verified, redistribution cleared, lock current, not already published) with the planned
   `https://github.com/<owner>/<repo>/releases/download/<tag>/<bundle>` URLs and SHA256SUMS lines.
6. Upload: only with the user's explicit per-run authorization (`HPCPERF_CONFIRM_UPLOAD=yes`) through the
   release adapter; that adapter does not read this plan yet (see below).
7. Anonymous check from a clean clone with an empty store and cache, then record `primary: {url, status:
   published}` per bundle in a separate reviewed commit. Published names are immutable: a new member set is
   always a new name, so a later round uploads only the bundles whose names changed.

## Proposed integration (not applied; files owned elsewhere)

* `tools/prepare_benchmark.sh`: accept `golden [SELECTION...] [--from DIR] [--offline]` and exec
  `golden_artifacts.py fetch`; `level1|level2 <b>` could fetch that benchmark's bundle, and
  `level3 nyx` could fetch `level3/nyx` after materializing the source.
* `tools/golden/hpcperf_golden.py`: in the "golden blob ... missing" FAIL line and in `status` BLOB-MISSING, add
  the hint `fetch it: python3 tools/golden/golden_artifacts.py fetch <level>/<benchmark>`.
* `tools/artifacts/artifact_catalog.yaml`: a `golden_store:` block (lock path, naming, staging layout
  `$HPCPERF_ARTIFACT_STAGING/golden/<bundle>`, proposed tag, publish requirements = lock-check PASS,
  staged sha256/size == lock, redistribution cleared, not yet published).
* `tools/artifacts/github_release_publish.py`: `asset_paths()` assumes `staging/level3/<app>/<version>/<file>` and
  one SOURCE_MANIFEST per asset; a plan of schema `hpcperf-golden-release-plan-1` needs the staging path
  `golden/<filename>`, no manifest, the lock as an extra asset.
* `tools/artifacts/remote_fetch_check.sh`: a `golden` mode (clean clone, empty store and cache, no credentials,
  `golden_artifacts.py fetch` + `verify`, then `hpcperf_golden.py status`).
* `tools/golden/tests/run_all.sh`: run `test_golden_artifacts.sh`; once the lock is committed, a
  `golden_artifacts.py lock-check` line makes a reference committed without a re-pack visible.

## Limits

* Content scan: archive blobs (the Nyx tars) member by member; raw numeric arrays are not text and are not
  scanned. No signature beyond sha256: the trust anchor is the git history of the references and the lock.
* Needs `git`, `zstd` and Python 3 with PyYAML (all in `.conda_env`); CPU only.
