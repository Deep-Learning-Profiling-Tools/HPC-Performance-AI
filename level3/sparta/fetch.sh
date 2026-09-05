#!/usr/bin/env bash
# Fetch SPARTA at the recorded release into _upstream/level3/sparta
# (gitignored, read-only reference). Idempotent; a checkout at another commit
# is an error, never silently reused.
#
#   ./fetch.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
UPSTREAM_URL="https://github.com/sparta/sparta.git"
UPSTREAM_TAG="27Aug2026"
UPSTREAM_SHA="95b9abaa8bd548991cc3c3f1c58b34722f7ade74"
DEST="$R/_upstream/level3/sparta"
if [ -d "$DEST/.git" ]; then
    have="$(git -C "$DEST" rev-parse HEAD)"
    [ "$have" = "$UPSTREAM_SHA" ] && { echo "fetch.sh: $DEST already at $UPSTREAM_TAG ($UPSTREAM_SHA)"; exit 0; }
    echo "fetch.sh: $DEST is at $have, not the recorded $UPSTREAM_SHA ($UPSTREAM_TAG); remove it to re-fetch" >&2; exit 1
fi
mkdir -p "$(dirname "$DEST")"
echo "fetch.sh: cloning $UPSTREAM_URL @ $UPSTREAM_TAG (shallow)"
git clone --quiet --depth 1 --branch "$UPSTREAM_TAG" "$UPSTREAM_URL" "$DEST"
have="$(git -C "$DEST" rev-parse HEAD)"
[ "$have" = "$UPSTREAM_SHA" ] || { echo "fetch.sh: tag $UPSTREAM_TAG resolved to $have, expected $UPSTREAM_SHA" >&2; exit 1; }
echo "fetch.sh: ok -> $DEST ($UPSTREAM_SHA)"
