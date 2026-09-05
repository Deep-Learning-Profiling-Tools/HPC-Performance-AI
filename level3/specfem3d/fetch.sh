#!/usr/bin/env bash
# Fetch SPECFEM3D Cartesian at the latest release tag into
# _upstream/level3/specfem3d (gitignored, read-only). Idempotent; a checkout
# at another commit is an error, never silently reused.
#
#   ./fetch.sh
#
# The two source patches in patches/ are backports of upstream `devel` commits
# (CUDA 13 `deviceOverlap` guard, Blackwell device block); their provenance is
# the devel snapshot recorded in DEVEL_SNAPSHOT below (not checked out here).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
UPSTREAM_URL="https://github.com/SPECFEM/specfem3d.git"
UPSTREAM_TAG="v4.1.1"
UPSTREAM_SHA="c67d3ae7d4bfc5ac75cb9e5601d93afa262d3d8d"
DEVEL_SNAPSHOT="cc2e9ffa7e7cb5338e05f5a7df81cfbe60e00683"   # devel @ 2026-07-24, source of the patches
DEST="$R/_upstream/level3/specfem3d"

if [ -d "$DEST/.git" ]; then
    have="$(git -C "$DEST" rev-parse HEAD)"
    if [ "$have" = "$UPSTREAM_SHA" ]; then echo "fetch.sh: $DEST already at $UPSTREAM_TAG ($UPSTREAM_SHA)"; exit 0; fi
    echo "fetch.sh: $DEST is at $have, not the recorded $UPSTREAM_SHA ($UPSTREAM_TAG); remove it to re-fetch" >&2; exit 1
fi
mkdir -p "$(dirname "$DEST")"
echo "fetch.sh: cloning $UPSTREAM_URL @ $UPSTREAM_TAG (shallow; submodules m4/flexwin/pyCMT3D not needed)"
git clone --quiet --depth 1 --branch "$UPSTREAM_TAG" "$UPSTREAM_URL" "$DEST"
have="$(git -C "$DEST" rev-parse HEAD)"
[ "$have" = "$UPSTREAM_SHA" ] || { echo "fetch.sh: tag $UPSTREAM_TAG resolved to $have, expected $UPSTREAM_SHA" >&2; exit 1; }
echo "fetch.sh: ok -> $DEST ($UPSTREAM_SHA); patches backported from devel $DEVEL_SNAPSHOT"
