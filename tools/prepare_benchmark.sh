#!/usr/bin/env bash
# prepare_benchmark.sh -- materialize a frozen benchmark source bundle into its benchmark directory.
#
#   tools/prepare_benchmark.sh level3 <app> [--variant NAME] [--force-rematerialize]
#
# After `git clone` + `git lfs pull`, this turns level3/<app>/archives/<bundle>.tar.zst into
# level3/<app>/src (+ deps): the complete, patched, frozen application/benchmark-specific source the
# benchmark's build.sh reads. Idempotent when src/deps already match the recorded source_tree_sha256;
# refuses to touch a modified tree unless --force-rematerialize is given explicitly (build.sh never
# calls this with --force). See tools/hpcperf_materialize.py for the exact checks.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
[ $# -ge 2 ] || { echo "usage: $0 level3 <app> [--variant NAME] [--force-rematerialize]" >&2; exit 2; }
LEVEL="$1"; APP="$2"; shift 2
[ "$LEVEL" = level3 ] || { echo "prepare_benchmark: only level3 benchmarks are materialized from source bundles (got '$LEVEL')" >&2; exit 2; }
DIR="$R/$LEVEL/$APP"
[ -d "$DIR" ] || { echo "prepare_benchmark: $DIR does not exist" >&2; exit 2; }
command -v zstd >/dev/null || { echo "prepare_benchmark: zstd not found on PATH (source hpcperf_env.sh)" >&2; exit 2; }
exec python3 "$HERE/hpcperf_materialize.py" "$DIR" "$@"
