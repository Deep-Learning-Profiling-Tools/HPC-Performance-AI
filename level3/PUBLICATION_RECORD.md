# Level 3 source-artifact publication record

What was published, when, how it was checked, and what it does **not** claim. Written after the fact from the
publisher's own machine-readable results; every number here comes from a check that ran.

## The release

| | |
|---|---|
| Tag | `level3-source-hpcperf-l3-v1-rc1`, lightweight, resolved through `/git/ref/tags` to the plan target |
| Release | id 387492391, **prerelease**, public |
| Target commit | `16dcf18a605c21dfcfb4aaddb2a35aef39bff8ee` |
| Release plan | `level3/RELEASE_PLAN.json` at commit `89ee00411bd8bca2f02d038f8a134b1ef61d48c1`, sha256 `52be4c5370ac6b67f4f324ad9556982ac4857e2e67fa2c5e76490a73ef3e1eb6` |
| Assets | 24: 11 source archives, `SHA256SUMS`, 11 `SOURCE_MANIFEST.<artifact>.json`, `RELEASE_PLAN.json` |
| Archive bytes | 1,197,001,594 |
| Excluded | GEOS: retired over ParMETIS redistribution, never published |
| Draft uploaded and verified | 2026-09-12T07:33:48Z |
| Published as prerelease | 2026-09-12T07:36:56Z |

The same plan file and the same `--expect-plan-sha256` value were used for every publisher invocation, from the
preflight through the public release, so a plan edited mid-flight would have aborted the run.

## Checks that actually ran

1. **Read-only access check** before anything else: repository readable, push permission present, and both
   `/releases/tags/<tag>` and `/git/ref/tags/<tag>` returned 404, i.e. no tag or release was reused.
2. **Local preflight**: every archive matched the plan's size and sha256, every working-tree `SOURCE_MANIFEST`
   matched the plan, and `SHA256SUMS` was derived from the plan rather than from the files.
3. **Draft upload**: each of the 24 assets was uploaded, its API response checked for status, JSON shape,
   object identity and `state == uploaded`, then re-downloaded **with credentials** and compared byte for byte.
   That is recorded as `DRAFT_UPLOADED_AND_VERIFIED` and is explicitly *not* evidence of anonymous availability.
4. **Pre-publish re-verification**: the complete asset set was listed again, and the remote content of all 24
   assets was verified by API digest before the release was made public. Size alone is never accepted.
5. **Publication** as a prerelease, followed by a post-publish re-verification of the release object and the
   asset set, and resolution of the created Git tag to the plan's target commit.
6. **Anonymous plan-URL verification** of all 11 archives, including both nekRS variants: downloaded with no
   credential in the process and an empty cache, then checked against size, archive sha256 and
   `source_tree_sha256` taken **only** from the lock. 11/11 PASS. Records:
   `level3/<app>/provenance/remote_artifact_verification*.yaml`.
7. **Ordinary-user verification**: clean clone of the lock-update commit, empty cache, no credential, plain
   `tools/prepare_benchmark.sh`. Records: `level3/<app>/provenance/remote_fetch_verification*.yaml`.

## One defect found during publication

The first anonymous run failed for all 11 archives. The cause was in this repository's own tooling, not in the
artifacts: `verify_published_artifact.py` did not create a caller-supplied `--cache`/`--scratch` directory, and
`remote_fetch_check.sh` always hands it paths inside a fresh `mktemp -d`. Every failure occurred **after** a
correct download whose size and sha256 had already been verified against the lock. The fix creates both
directories; test 4a2 in the release mock suite covers it. No archive, hash or release asset was changed, and no
check was relaxed or skipped to get past it.

## What this release does not claim

* It is a **source-artifact prerelease**. It does not assert that the ten applications are validated on any
  machine other than this one, nor that HIP, multi-node, or the 8/40/80-GPU plans work anywhere.
* The code is **not merged into `main`**. A plain clone of the default branch does not contain these tools; check out
  the branch `level3/source-freeze` until PR #5 is merged, then `main`. The release tag is the identity of the commit the
  artifacts were cut from (its locks still say `unpublished`), not a harness entry point: a plain `prepare` at that tag
  does not download.
* Scientific status is unchanged by publication: Nyx `LyA` heat/cool stays `I_R_CHECK_PENDING`, eight
  applications keep historical build evidence rather than a workspace rebuild, ExaCA's acceptance remains the
  project-defined `dirsolid` protocol plus upstream's two small official cases, and unmodified upstream ExaCA
  tests still fail under CTest's own launcher in this environment.

## Credential note

The upload used the GitHub token present in the agent environment. That token is the one the security record of
2026-09-07 lists as exposed, with rotation not confirmed at the time of use; the maintainer reviewed that fact
and authorized this specific publication anyway on 2026-09-12. No credential value was printed, copied into any
file, or placed on a command line, and the anonymous verification steps ran with the token removed from the
environment. Rotation remains the maintainer's action and is **not** claimed here.

## Extension round ext1 (2026-10-21): VPIC, PIConGPU, Cholla

| | |
|---|---|
| Tag | `level3-source-hpcperf-l3-v1-ext1`, lightweight, resolved through `/git/ref/tags` to the plan target |
| Release | id 419976733, **prerelease**, public (the rc1 release and its 24 assets are untouched) |
| Target commit | `7e2a220b5682b4562e066d77e4feea51b80803ba` (branch `level3/cholla`: the three applications' provenance with `unpublished` locks) |
| Release plan | `level3/RELEASE_PLAN.json` at commit `7d485b0`, sha256 `8fd5ba642341d24224ba7ec4863727c2d936f4cfbe38747f114421d084f9bde7` (generated with the new `--app vpic,picongpu,cholla` filter of `release_plan.py`: the ten rc1 artifacts are listed as `NOT_IN_THIS_ROUND`) |
| Assets | 8: 3 source archives, `SHA256SUMS`, 3 `SOURCE_MANIFEST.<artifact>.json`, `RELEASE_PLAN.json` |
| Archive bytes | 162,526,367 (vpic 1,937,373; picongpu 160,232,498; cholla 356,496) |
| Draft uploaded and verified | 2026-10-21 (every asset re-downloaded with credentials and compared) |
| Published as prerelease | 2026-10-21, re-verified by API digest and `--verify --redownload` (3/3 archives sha256 ok) |

Checks that ran, in order: the three staging archives verified against their locks (`verify_artifact.py` PASS 3/3) and
re-extracted for the plan (`--full`); the publisher's mock suite (45/45); preflight; draft upload with authenticated
re-download; publication; anonymous plan-URL verification 3/3 (`provenance/remote_artifact_verification.yaml`); the
lock update commits (one per application branch: `level3/vpic` 9f5e0b7, `level3/picongpu-rebased` 8b22efa,
`level3/cholla` 2938e9c), merged forward through the stacked branches; the ordinary-user verification
(`remote_fetch_check.sh --mode lock-entry`: clean clone, empty cache, no credential, plain `prepare_benchmark.sh`)
3/3 `REMOTE_FETCH_VERIFIED` with `check_workspace` 15/15 (`provenance/remote_fetch_verification.yaml`).

One defect found: the first ordinary-user run failed `check_workspace` check 10 for all three applications because
their `benchmark.yaml` `inputs` / `references` lists held globs and descriptions instead of literal paths (the check
tests each entry with `os.path.exists`). Fixed in the descriptors (literal paths; one commit per application); no
artifact, hash, lock or release asset changed. The download and hash verification had already passed before the
failing check in every case. The code of this round (the three applications, their locks and records) is in `main`
since 2026-10-22 (PRs #22 / #25 / #26 merged through the stacked branches and brought to `main` together).
