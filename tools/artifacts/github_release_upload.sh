#!/usr/bin/env bash
# github_release_upload.sh -- guarded entry point for publishing the Level 3 source artifacts as GitHub Release
# assets. All checking logic lives in tools/artifacts/github_release_publish.py (strict HTTP status, JSON shape
# and object-identity checks; assets are never overwritten or deleted). UNTESTED against the live GitHub API;
# exercised only by the CPU-only mock suite tools/artifacts/tests/test_release_publish_mock.sh.
#
#   tools/artifacts/github_release_upload.sh --plan level3/RELEASE_PLAN.json --preflight
#   HPCPERF_CONFIRM_UPLOAD=yes tools/artifacts/github_release_upload.sh --plan ... --draft
#   HPCPERF_CONFIRM_UPLOAD=yes tools/artifacts/github_release_upload.sh --plan ... --publish-prerelease <id>
#   tools/artifacts/github_release_upload.sh --plan ... --verify <id> [--redownload]
#
# --preflight and --verify need no authorization; --draft and --publish-prerelease refuse to run without
# HPCPERF_CONFIRM_UPLOAD=yes, which the maintainer sets per run. Publishing never edits a source lock: the lock
# update (real URL + status) is a separate reviewed commit made after the anonymous download check
# (tools/artifacts/remote_fetch_check.sh --mode plan-url) passed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAN=""; MODE=""; RID=""; EXTRA=()
while [ $# -gt 0 ]; do case "$1" in
    --plan) PLAN=$2; shift 2;;
    --preflight) MODE=preflight; shift;;
    --draft) MODE=draft; shift;;
    --publish-prerelease) MODE=publish; RID=$2; shift 2;;
    --verify) MODE=verify; RID=$2; shift 2;;
    --redownload) EXTRA+=(--redownload); shift;;
    --staging) EXTRA+=(--staging "$2"); shift 2;;
    --expect-plan-sha256) EXTRA+=(--expect-plan-sha256 "$2"); shift 2;;
    --json-out) EXTRA+=(--json-out "$2"); shift 2;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "github_release_upload: unknown option $1" >&2; exit 2;; esac; done
[ -n "$PLAN" ] && [ -n "$MODE" ] || { sed -n '2,20p' "$0" >&2; exit 2; }
case "$MODE" in draft|publish)
    [ "${HPCPERF_CONFIRM_UPLOAD:-}" = yes ] || { echo "github_release_upload: REFUSED -- set HPCPERF_CONFIRM_UPLOAD=yes only with the maintainer's explicit authorization for this upload" >&2; exit 3; };;
esac
exec python3 "$HERE/github_release_publish.py" --plan "$PLAN" --mode "$MODE" ${RID:+--release-id "$RID"} "${EXTRA[@]}"
