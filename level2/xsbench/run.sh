#!/usr/bin/env bash
# Run the standard XSBench benchmark problem.
#
#   ./run.sh [CUDA|HIP] [extra XSBench args]      (default backend: CUDA)
#
# Standard problem = upstream's GPU default: event-based simulation
# (the only mode implemented in the CUDA/HIP sources), Hoogenboom-Martin
# "large" reactor model (355 nuclides, 11,303 gridpoints/nuclide, unionized
# energy grid), 17,000,000 macroscopic cross-section lookups, baseline kernel:
#
#   ./XSBench -m event -s large
#
# Extra arguments are appended and therefore override the defaults
# (e.g. ./run.sh CUDA -l 100000000  or  ./run.sh CUDA -k 6). The binary runs
# from its build directory so that optional -b write/read data files land there.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

# Load the project environment first (see build.sh for why it comes first).
if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$R" ] && [ -f "$R/hpcperf_env.sh" ]; then
    set +eu
    # shellcheck disable=SC1091
    source "$R/hpcperf_env.sh" 2>/dev/null
    set -eu
fi

BACKEND="CUDA"
case "$(printf '%s' "${1:-}" | tr '[:lower:]' '[:upper:]')" in
    CUDA) BACKEND="CUDA"; shift ;;
    HIP)  BACKEND="HIP";  shift ;;
esac
BUILD_DIR="$R/build/level2/xsbench/$(printf '%s' "$BACKEND" | tr '[:upper:]' '[:lower:]')"
EXE="$BUILD_DIR/XSBench"

if [ ! -x "$EXE" ]; then
    echo "error: $EXE not found -- run $HERE/build.sh $BACKEND first" >&2
    exit 1
fi

cd "$BUILD_DIR"
exec ./XSBench -m event -s large "$@"
