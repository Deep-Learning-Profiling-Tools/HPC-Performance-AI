#!/usr/bin/env bash
# Native CUDA/HIP SW4lite point-source solve with MPI spatial decomposition.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null
set -euo pipefail

BACKEND=CUDA
case "${1:-}" in CUDA|cuda|HIP|hip) BACKEND="${1^^}"; shift;; esac
MODEL="${BACKEND,,}"; BUILD="$R/build/level2/sw4lite/$MODEL"
if [ "$BACKEND" = CUDA ]; then EXE="$BUILD/sw4lite"; else EXE="$BUILD/app/sw4lite"; fi
[ -x "$EXE" ] || { echo "run.sh: $EXE not found -- run $HERE/build.sh $BACKEND first" >&2; exit 1; }
# shellcheck disable=SC1091
source "$HERE/../tools/hpcperf_launch_common.sh"
export HPCPERF_GPU_BACKEND="$BACKEND"
N_RANKS="$(hpcperf_ranks sw4lite no)" || exit 2
INPUT="${HPCPERF_SW4LITE_INPUT:-$HERE/inputs/pointsource.in}"
[ -f "$INPUT" ] || { echo "run.sh: input not found: $INPUT" >&2; exit 1; }
mkdir -p "$BUILD/run"; cd "$BUILD/run"
echo "== SW4lite $BACKEND: ranks=$N_RANKS input=$INPUT"
exec "$HPCPERF_LAUNCHER_BIN" --gpus "$N_RANKS" --bind wrapper -- "$EXE" "$INPUT" "$@"
