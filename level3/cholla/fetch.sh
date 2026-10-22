#!/usr/bin/env bash
# Fetch Cholla at the recorded release tag into _upstream/level3/cholla (gitignored, read-only reference;
# FREEZE-TIME ONLY -- build.sh never calls this). Idempotent; a checkout at another commit is an error,
# never silently reused. The identity is the v3.0.0 release commit (2024-02-02, the latest tag; the main
# branch holds only four CI/docker commits after it). The cholla-tests-data submodule (fiducial outputs of
# upstream's gtest system tests) is NOT fetched: the registered inputs are checked against analytic
# solutions and this build's frozen references (inputs.yaml), and the bundle holds no fiducial data.
#
#   ./fetch.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
UPSTREAM_URL="https://github.com/cholla-hydro/cholla.git"
UPSTREAM_REF="v3.0.0"
UPSTREAM_SHA="8478c9ede8a5e2cb72d6481a50023666d489a763"
DEST="$R/_upstream/level3/cholla"
if [ -d "$DEST/.git" ]; then
    have="$(git -C "$DEST" rev-parse HEAD)"
    if [ "$have" = "$UPSTREAM_SHA" ]; then echo "fetch.sh: $DEST already at $UPSTREAM_REF ($UPSTREAM_SHA)"; exit 0; fi
    echo "fetch.sh: $DEST is at $have, not the recorded $UPSTREAM_SHA; remove it to re-fetch" >&2; exit 1
fi
mkdir -p "$(dirname "$DEST")"
echo "fetch.sh: cloning $UPSTREAM_URL, checking out $UPSTREAM_REF ($UPSTREAM_SHA)"
git clone --quiet "$UPSTREAM_URL" "$DEST"
git -C "$DEST" checkout --quiet "$UPSTREAM_SHA"
have="$(git -C "$DEST" rev-parse HEAD)"
[ "$have" = "$UPSTREAM_SHA" ] || { echo "fetch.sh: checkout resolved to $have, expected $UPSTREAM_SHA" >&2; exit 1; }
[ "$(git -C "$DEST" describe --tags --exact-match 2>/dev/null)" = "$UPSTREAM_REF" ] || { echo "fetch.sh: $UPSTREAM_SHA is not tag $UPSTREAM_REF" >&2; exit 1; }
echo "fetch.sh: ok -> $DEST ($UPSTREAM_REF = $UPSTREAM_SHA)"
