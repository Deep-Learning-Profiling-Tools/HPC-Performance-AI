#!/usr/bin/env bash
# Fetch WarpX and the AMReX release it pins into _upstream/level3/{WarpX,amrex}
# (gitignored, read-only). WarpX 26.09 pins AMReX 26.09 (cmake/dependencies/
# AMReX.cmake / dependencies.json); the AMReX checkout is used as
# -DWarpX_amrex_src so no configure-time download is needed. Idempotent; a
# checkout at another commit is an error, never silently reused.
#
#   ./fetch.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# Official repository (ECP-WarpX/WarpX redirects here since the project moved).
WARPX_URL="https://github.com/BLAST-WarpX/warpx.git"; WARPX_TAG="26.09"; WARPX_SHA="0c62c75e53a9ad08241535444bd7e53fd1deba88"
AMREX_URL="https://github.com/AMReX-Codes/amrex.git";  AMREX_TAG="26.09"; AMREX_SHA="a52ca73324ac2c7b65ec04f131e6df99eec9c576"

fetch_one() { # url tag sha dest
    local url=$1 tag=$2 sha=$3 dest=$4 have
    if [ -d "$dest/.git" ]; then
        have="$(git -C "$dest" rev-parse HEAD)"
        [ "$have" = "$sha" ] && { echo "fetch.sh: $dest already at $tag ($sha)"; return 0; }
        echo "fetch.sh: $dest is at $have, not the recorded $sha ($tag); remove it to re-fetch" >&2; return 1
    fi
    mkdir -p "$(dirname "$dest")"
    echo "fetch.sh: cloning $url @ $tag (shallow)"
    git clone --quiet --depth 1 --branch "$tag" "$url" "$dest"
    have="$(git -C "$dest" rev-parse HEAD)"
    [ "$have" = "$sha" ] || { echo "fetch.sh: tag $tag resolved to $have, expected $sha" >&2; return 1; }
    echo "fetch.sh: ok -> $dest ($sha)"
}
fetch_one "$WARPX_URL" "$WARPX_TAG" "$WARPX_SHA" "$R/_upstream/level3/WarpX"
fetch_one "$AMREX_URL" "$AMREX_TAG" "$AMREX_SHA" "$R/_upstream/level3/amrex"
