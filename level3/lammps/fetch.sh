#!/usr/bin/env bash
# FREEZE-TIME ONLY. Fetch the LAMMPS upstream source at the recorded stable
# release into the read-only reference checkout _upstream/level3/lammps
# (gitignored) -- the input of tools/freeze_benchmark_source.py, which produces
# the source artifact (in the maintainer's staging, never in git). build.sh/run.sh/validate.sh never read this
# checkout: they use level3/lammps/src, materialized by tools/prepare_benchmark.sh.
# Nothing is built here. Re-running is idempotent; a checkout at a different
# commit is an error (delete it to re-fetch), never silently reused.
#
#   ./fetch.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

# Official repository and the selected stable release (see README.md).
UPSTREAM_URL="https://github.com/lammps/lammps.git"
UPSTREAM_TAG="stable_22Jul2025_update6"
UPSTREAM_SHA="9c5ab448c78a14fd534619622162ba418d6a1fb1"
DEST="$R/_upstream/level3/lammps"

if [ -d "$DEST/.git" ]; then
    have="$(git -C "$DEST" rev-parse HEAD)"
    if [ "$have" = "$UPSTREAM_SHA" ]; then
        echo "fetch.sh: $DEST already at $UPSTREAM_TAG ($UPSTREAM_SHA)"; exit 0
    fi
    echo "fetch.sh: $DEST is at $have, not the recorded $UPSTREAM_SHA ($UPSTREAM_TAG); remove it to re-fetch" >&2
    exit 1
fi
mkdir -p "$(dirname "$DEST")"
echo "fetch.sh: cloning $UPSTREAM_URL @ $UPSTREAM_TAG (shallow)"
git clone --quiet --depth 1 --branch "$UPSTREAM_TAG" "$UPSTREAM_URL" "$DEST"
have="$(git -C "$DEST" rev-parse HEAD)"
[ "$have" = "$UPSTREAM_SHA" ] || { echo "fetch.sh: tag $UPSTREAM_TAG resolved to $have, expected $UPSTREAM_SHA" >&2; exit 1; }
echo "fetch.sh: ok -> $DEST ($UPSTREAM_SHA)"
