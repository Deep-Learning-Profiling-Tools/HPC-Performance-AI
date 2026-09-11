#!/usr/bin/env bash
# remote_fetch_check.sh -- REMOTE_FETCH_VERIFIED test for one published artifact (run AFTER publication, never
# by this round): clean clone at the reviewed commit, EMPTY cache, no credentials, ordinary https download
# through prepare_benchmark.sh, tree hash verified; writes provenance/remote_fetch_verification[.variant].yaml
# into the given repository checkout only on success.
#
#   tools/artifacts/remote_fetch_check.sh <app> [--variant V] --commit <sha> --record-into <repo-checkout> [--clone-url URL]
#
# Refuses to run with GITHUB_TOKEN / GH_TOKEN / a git credential helper in the environment (a draft-release
# download that only works with a token is NOT an anonymous fetch). Uses a throwaway clone under $TMPDIR; the
# maintainer's staging and the original machine's cache must not be reachable (HPCPERF_ARTIFACT_STAGING and
# HPCPERF_ARTIFACT_CACHE are unset; the cache is a fresh empty directory).
set -euo pipefail
APP=""; VARIANT=""; COMMIT=""; RECORD=""; URL="https://github.com/Deep-Learning-Profiling-Tools/HPC-Performance-AI.git"
while [ $# -gt 0 ]; do case "$1" in --variant) VARIANT=$2; shift 2;; --commit) COMMIT=$2; shift 2;; --record-into) RECORD=$2; shift 2;; --clone-url) URL=$2; shift 2;; -*) echo "unknown option $1" >&2; exit 2;; *) APP=$1; shift;; esac; done
[ -n "$APP" ] && [ -n "$COMMIT" ] && [ -n "$RECORD" ] || { echo "usage: $0 <app> [--variant V] --commit <sha> --record-into <repo-checkout>" >&2; exit 2; }
for v in GITHUB_TOKEN GH_TOKEN; do [ -z "${!v:-}" ] || { echo "remote_fetch_check: $v is set -- an authenticated download is not an anonymous fetch; unset it" >&2; exit 2; }; done
unset HPCPERF_ARTIFACT_STAGING HPCPERF_ARTIFACT_CACHE
T="$(mktemp -d "${TMPDIR:-/tmp}/hpcperf-remote-fetch-XXXXXX")"; trap 'rm -rf "$T"' EXIT
export HPCPERF_ARTIFACT_CACHE="$T/empty-cache"; mkdir -p "$HPCPERF_ARTIFACT_CACHE"
git -c credential.helper= clone -q "$URL" "$T/clone" && git -C "$T/clone" checkout -q "$COMMIT"
VARG=(); [ -n "$VARIANT" ] && VARG=(--variant "$VARIANT")
( cd "$T/clone" && ./tools/prepare_benchmark.sh level3 "$APP" "${VARG[@]}" ) 2>&1 | tee "$T/prepare.log"
grep -q 'STATUS REMOTE_FETCH_VERIFIED' "$T/prepare.log" && grep -q 'STATUS MATERIALIZED' "$T/prepare.log" || { echo "remote_fetch_check: the artifact was not fetched from the published URL and materialized" >&2; exit 1; }
( cd "$T/clone" && python3 tools/check_workspace.py "level3/$APP" "${VARG[@]}" --quick ) || { echo "remote_fetch_check: check_workspace failed on the fetched tree" >&2; exit 1; }
SFX=""; [ -n "$VARIANT" ] && SFX=".$VARIANT"
OUT="$RECORD/level3/$APP/provenance/remote_fetch_verification$SFX.yaml"
{ echo "schema: hpcperf-remote-fetch-verification-1"; echo "benchmark: level3/$APP"; echo "variant: ${VARIANT:-null}"; echo "commit: $COMMIT"; echo "verdict: PASS"
  echo "method: clean clone, empty cache, no credentials, prepare_benchmark.sh download from the lock's primary url, size+sha256+source_tree_sha256 verified, check_workspace PASS"
  echo "utc: $(date -u +%FT%TZ)"; echo "host: $(hostname)"; } > "$OUT"
echo "remote_fetch_check: PASS -> $OUT"
