#!/usr/bin/env bash
# FREEZE-TIME ONLY. Fetch the ExaCA upstream release and the pinned Kokkos release into the read-only
# reference checkouts _upstream/level3/{ExaCA,kokkos} (gitignored) -- the inputs of
# tools/freeze_benchmark_source.py, which produces the source artifact exaca-<source_version>.tar.zst
# (src/ = ExaCA, deps/kokkos = Kokkos, deps/json = the nlohmann json release tarball, sha256-pinned in
# provenance/freeze_spec.yaml). build.sh/run.sh/validate.sh never read these checkouts: they use
# level3/exaca/{src,deps}, materialized by tools/prepare_benchmark.sh. Nothing is built here.
# Re-running is idempotent; a checkout at a different commit is an error (delete it to re-fetch).
#
#   ./fetch.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

fetch() { # <url> <tag> <sha> <dest>
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
# Official repositories and the selected releases (see README.md / provenance/source.lock.yaml).
fetch https://github.com/LLNL/ExaCA.git 2.1.0 d26e59cd51e241a327c5267d43fd70537e5425f7 "$R/_upstream/level3/ExaCA"
fetch https://github.com/kokkos/kokkos.git 4.7.04 82799e4577568f9666bde36265ac15d78da3e6c8 "$R/_upstream/level3/kokkos"
