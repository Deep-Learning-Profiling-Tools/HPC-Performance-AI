#!/usr/bin/env bash
# Fetch PIConGPU (ComputationalRadiationPhysics/picongpu) at the recorded commit of the dev branch into
# _upstream/level3/picongpu (gitignored, read-only reference; FREEZE-TIME ONLY -- build.sh never calls this).
# Idempotent; a checkout at another commit is an error, never silently reused. The third-party libraries the
# build needs (alpaka, mallocMC, nlohmann_json, toml11, cmake-modules) are vendored in thirdParty/; the only
# git submodule (thirdParty/catch2, unit tests) is not needed and not fetched.
#
#   ./fetch.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
UPSTREAM_URL="https://github.com/ComputationalRadiationPhysics/picongpu.git"
UPSTREAM_REF="dev"
UPSTREAM_SHA="93cba5b61b628659a6ec8181f5e2dd28b8006f29"
DEST="$R/_upstream/level3/picongpu"
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
