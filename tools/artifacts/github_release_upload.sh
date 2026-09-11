#!/usr/bin/env bash
# github_release_upload.sh -- GitHub Release adapter for the Level 3 source artifacts. UNTESTED against the live
# API in this repository (written 2026-09-11 for the future publication step; no release exists yet).
#
#   HPCPERF_CONFIRM_UPLOAD=yes tools/artifacts/github_release_upload.sh --plan level3/RELEASE_PLAN.json --draft
#   HPCPERF_CONFIRM_UPLOAD=yes tools/artifacts/github_release_upload.sh --plan level3/RELEASE_PLAN.json --publish-prerelease <release-id>
#
# Guards: refuses without HPCPERF_CONFIRM_UPLOAD=yes (the maintainer's explicit authorization for THIS run),
# without GITHUB_TOKEN, when the plan's target commit is not on origin, when an asset of the same name already
# exists on the release (assets are never overwritten), or when a local file fails size/sha256 against the plan.
# --draft creates a DRAFT release on the target commit and uploads every planned asset + SHA256SUMS +
# SOURCE_MANIFEST files, then re-downloads each asset (authenticated) and compares sha256/size -- that is an
# upload check, NOT the anonymous REMOTE_FETCH_VERIFIED (tools/artifacts/remote_fetch_check.sh after publishing).
# --publish-prerelease flips an existing draft to a published prerelease. It never edits locks: the lock update
# (primary.url/status) is a separate reviewed commit after remote_fetch_check.sh passed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; R="$(cd "$HERE/../.." && pwd)"
PLAN=""; MODE=""; REL_ID=""; STAGING="${HPCPERF_ARTIFACT_STAGING:-}"
while [ $# -gt 0 ]; do case "$1" in --plan) PLAN=$2; shift 2;; --draft) MODE=draft; shift;; --publish-prerelease) MODE=publish; REL_ID=$2; shift 2;; --staging) STAGING=$2; shift 2;; *) echo "unknown option $1" >&2; exit 2;; esac; done
[ "${HPCPERF_CONFIRM_UPLOAD:-}" = yes ] || { echo "github_release_upload: REFUSED -- set HPCPERF_CONFIRM_UPLOAD=yes only with the maintainer's explicit authorization for this upload" >&2; exit 3; }
[ -n "${GITHUB_TOKEN:-}" ] || { echo "github_release_upload: GITHUB_TOKEN required" >&2; exit 2; }
[ -n "$PLAN" ] && [ -f "$PLAN" ] && [ -n "$MODE" ] || { echo "usage: see header" >&2; exit 2; }
[ -n "$STAGING" ] || { echo "github_release_upload: --staging or HPCPERF_ARTIFACT_STAGING required" >&2; exit 2; }
API=https://api.github.com; REPO="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["repository"])' "$PLAN")"
TAG="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tag"])' "$PLAN")"; COMMIT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["target_commit"])' "$PLAN")"
gh_api() { curl -sS -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" "$@"; }
if [ "$MODE" = draft ]; then
    git -C "$R" fetch -q origin && git -C "$R" branch -r --contains "$COMMIT" | grep -q origin/ || { echo "github_release_upload: target commit $COMMIT is not on origin -- push the reviewed branch first" >&2; exit 2; }
    gh_api "$API/repos/$REPO/releases/tags/$TAG" -o /dev/null -w '%{http_code}' | grep -q '^404$' || { echo "github_release_upload: a release with tag $TAG already exists -- tags/assets are never reused" >&2; exit 2; }
    T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
    # local preflight: every planned asset present with the planned size + sha256
    python3 - "$PLAN" "$STAGING" "$T" <<'PY'
import json, os, sys, hashlib
plan, stg, T = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
sums = []
for a in plan["assets"]:
    p = os.path.join(stg, "level3", a["artifact"].split(".")[0], a["source_version"], a["filename"])
    h = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) else None
    if h != a["sha256"] or os.path.getsize(p) != a["size_bytes"]:
        sys.exit(f"preflight FAIL: {a['filename']} size/sha256 differ from the plan")
    sums.append(a["sha256sums_line"]); open(os.path.join(T, "files.txt"), "a").write(p + "\n")
open(os.path.join(T, "SHA256SUMS"), "w").write("\n".join(sums) + "\n"); print(f"preflight ok: {len(sums)} assets")
PY
    REL="$(gh_api -X POST "$API/repos/$REPO/releases" -d "$(python3 -c 'import json,sys; print(json.dumps({"tag_name": sys.argv[1], "target_commitish": sys.argv[2], "name": sys.argv[1], "draft": True, "prerelease": True, "body": "Level 3 source artifacts (scheme 3), release candidate. See level3/RELEASE_PLAN.md at the target commit."}))' "$TAG" "$COMMIT")")"
    REL_ID="$(echo "$REL" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')"; UPLOAD="$(echo "$REL" | python3 -c 'import json,sys; print(json.load(sys.stdin)["upload_url"].split("{")[0])')"
    echo "draft release id $REL_ID created on $COMMIT"
    upload() { local f=$1 name=$2; gh_api -X POST -H "Content-Type: application/octet-stream" "$UPLOAD?name=$name" --data-binary @"$f" -o "$T/resp.json" -w '%{http_code}' | grep -q '^201$' || { echo "upload of $name failed"; cat "$T/resp.json"; exit 1; }
        local url; url="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["url"])' "$T/resp.json")"
        gh_api -L -H "Accept: application/octet-stream" "$url" -o "$T/dl.bin"; cmp -s "$f" "$T/dl.bin" && echo "  $name uploaded and re-downloaded identical (authenticated check)" || { echo "  $name re-download differs"; exit 1; }; }
    while read -r f; do upload "$f" "$(basename "$f")"; done < "$T/files.txt"
    upload "$T/SHA256SUMS" SHA256SUMS
    python3 - "$PLAN" "$R" "$T" <<'PY'
import json, os, shutil, sys
plan, R, T = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
with open(os.path.join(T, "manifests.txt"), "w") as out:
    for a in plan["assets"]:
        src = os.path.join(R, a["source_manifest"]["path"]); dst = os.path.join(T, a["source_manifest"]["planned_asset"]); shutil.copyfile(src, dst); out.write(dst + "\n")
PY
    while read -r f; do upload "$f" "$(basename "$f")"; done < "$T/manifests.txt"
    cp "$PLAN" "$T/RELEASE_PLAN.json"; upload "$T/RELEASE_PLAN.json" RELEASE_PLAN.json
    echo "draft release $REL_ID complete: assets uploaded and re-downloaded (authenticated). Next: --publish-prerelease $REL_ID, then tools/artifacts/remote_fetch_check.sh (anonymous)."
else
    gh_api -X PATCH "$API/repos/$REPO/releases/$REL_ID" -d '{"draft": false, "prerelease": true}' -o /dev/null -w '%{http_code}\n'
    echo "release $REL_ID published as prerelease; run tools/artifacts/remote_fetch_check.sh per artifact before touching any lock"
fi
