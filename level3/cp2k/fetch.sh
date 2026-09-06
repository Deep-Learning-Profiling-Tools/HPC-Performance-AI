#!/usr/bin/env bash
# Fetch CP2K at the pinned release (read-only checkout under _upstream/level3/cp2k).
#
#   CP2K v2026.2   67b5da876dd6a76b8b021d5a04d1c81ba79a4c50  (tag v2026.2, 2026-07-15; CMake-only build,
#                  ships tools/toolchain/install_cp2k_toolchain.sh with DBCSR 2.10.0 pinned)
#
# DBCSR 2.10.0 (sha256 3d897220fbb4498215331efad6905eb7744881b4cf04eb5c5fb4db7c48a56ef9) and the
# other dependencies are downloaded by the toolchain script itself (build.sh) into
# the private toolchain copy -- nothing is fetched into the repository.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
DEST="$R/_upstream/level3/cp2k"
TAG=v2026.2; SHA=67b5da876dd6a76b8b021d5a04d1c81ba79a4c50
if [ ! -d "$DEST/.git" ]; then
    mkdir -p "$(dirname "$DEST")"
    git clone -q --branch "$TAG" --depth 1 https://github.com/cp2k/cp2k.git "$DEST"
fi
got="$(git -C "$DEST" rev-parse HEAD)"
[ "$got" = "$SHA" ] || { echo "fetch.sh: $DEST is at $got, expected $SHA ($TAG)" >&2; exit 1; }
echo "# CP2K $TAG $got -> $DEST (data/ $(du -sh "$DEST/data" | cut -f1), benchmarks/ $(du -sh "$DEST/benchmarks" | cut -f1))"
