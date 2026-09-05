#!/usr/bin/env bash
# Fetch nekRS at the latest release tag into _upstream/level3/nekRS
# (gitignored, read-only). nekRS vendors all third-party libraries as squashed
# subtrees (OCCA, HYPRE, gslib, Nek5000, ADIOS2, ...), so no submodules and no
# configure-time downloads are involved. Idempotent; a checkout at another
# commit is an error, never silently reused.
#
#   ./fetch.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
UPSTREAM_URL="https://github.com/Nek5000/nekRS.git"
UPSTREAM_TAG="v26.0"
UPSTREAM_SHA="96b3cf9e5bacede16568826c04a21bc0fe50dc7d"
DEST="$R/_upstream/level3/nekRS"

if [ -d "$DEST/.git" ]; then
    have="$(git -C "$DEST" rev-parse HEAD)"
    if [ "$have" = "$UPSTREAM_SHA" ]; then echo "fetch.sh: $DEST already at $UPSTREAM_TAG ($UPSTREAM_SHA)"; exit 0; fi
    echo "fetch.sh: $DEST is at $have, not the recorded $UPSTREAM_SHA ($UPSTREAM_TAG); remove it to re-fetch" >&2; exit 1
fi
mkdir -p "$(dirname "$DEST")"
echo "fetch.sh: cloning $UPSTREAM_URL @ $UPSTREAM_TAG (shallow)"
git clone --quiet --depth 1 --branch "$UPSTREAM_TAG" "$UPSTREAM_URL" "$DEST"
have="$(git -C "$DEST" rev-parse HEAD)"
[ "$have" = "$UPSTREAM_SHA" ] || { echo "fetch.sh: tag $UPSTREAM_TAG resolved to $have, expected $UPSTREAM_SHA" >&2; exit 1; }
echo "fetch.sh: ok -> $DEST ($UPSTREAM_SHA)"
