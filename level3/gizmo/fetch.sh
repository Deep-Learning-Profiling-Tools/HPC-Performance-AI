#!/usr/bin/env bash
# Fetch GIZMO (pfhopkins/gizmo) at the recorded commit of the gpu_kokkos_unified branch into
# _upstream/level3/gizmo (gitignored, read-only reference; FREEZE-TIME ONLY -- build.sh never calls this).
# Idempotent; a checkout at another commit is an error, never silently reused. The branch is the
# maintainer's active Kokkos CPU/GPU line and carries no release tags: the identity is the commit.
#
#   ./fetch.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
UPSTREAM_URL="https://github.com/pfhopkins/gizmo.git"
UPSTREAM_REF="gpu_kokkos_unified"
UPSTREAM_SHA="e9d2d0bea23b08872f747b780cdcedb5847a2bff"
DEST="$R/_upstream/level3/gizmo"
if [ -d "$DEST/.git" ]; then
    have="$(git -C "$DEST" rev-parse HEAD)"
    if [ "$have" = "$UPSTREAM_SHA" ]; then echo "fetch.sh: $DEST already at $UPSTREAM_REF ($UPSTREAM_SHA)"; exit 0; fi
    echo "fetch.sh: $DEST is at $have, not the recorded $UPSTREAM_SHA; remove it to re-fetch" >&2; exit 1
fi
mkdir -p "$(dirname "$DEST")"
echo "fetch.sh: cloning $UPSTREAM_URL @ $UPSTREAM_REF, checking out $UPSTREAM_SHA"
git clone --quiet --branch "$UPSTREAM_REF" "$UPSTREAM_URL" "$DEST"
git -C "$DEST" checkout --quiet "$UPSTREAM_SHA"
have="$(git -C "$DEST" rev-parse HEAD)"
[ "$have" = "$UPSTREAM_SHA" ] || { echo "fetch.sh: checkout resolved to $have, expected $UPSTREAM_SHA" >&2; exit 1; }
echo "fetch.sh: ok -> $DEST ($UPSTREAM_SHA)"
