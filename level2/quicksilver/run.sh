#!/usr/bin/env bash
# CORAL-2 P1 physics with a weak-scaled MPI domain, one rank per GPU.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null
set -euo pipefail

BACKEND=CUDA
case "${1:-}" in CUDA|cuda|HIP|hip) BACKEND="${1^^}"; shift;; esac
MODEL="${BACKEND,,}"; BUILD="$R/build/level2/quicksilver/$MODEL"; EXE="$BUILD/qs"
[ -x "$EXE" ] || { echo "run.sh: $EXE not found -- run $HERE/build.sh $BACKEND first" >&2; exit 1; }
# shellcheck disable=SC1091
source "$HERE/../tools/hpcperf_launch_common.sh"
export HPCPERF_GPU_BACKEND="$BACKEND"
N_RANKS="$(hpcperf_ranks quicksilver no)" || exit 2
read -r PX PY PZ <<< "$(hpcperf_topology quicksilver "$N_RANKS" --dims 3)"
LOCAL="${HPCPERF_QUICKSILVER_CELLS_PER_RANK:-8}"
PPR="${HPCPERF_QUICKSILVER_PARTICLES_PER_RANK:-100000}"
STEPS="${HPCPERF_QUICKSILVER_STEPS:-20}"
for value in "$LOCAL" "$PPR" "$STEPS"; do case "$value" in ''|*[!0-9]*|0) echo "run.sh: cells, particles, and steps must be positive integers" >&2; exit 2;; esac; done
NX=$(( LOCAL * PX )); NY=$(( LOCAL * PY )); NZ=$(( LOCAL * PZ )); PARTICLES=$(( PPR * N_RANKS ))
INPUT="${HPCPERF_QUICKSILVER_INPUT:-$HERE/inputs/coral2_p1_profile.inp}"
[ -f "$INPUT" ] || { echo "run.sh: input not found: $INPUT" >&2; exit 1; }
hpcperf_forbid_args quicksilver -i --inputFile -X --lx -Y --ly -Z --lz -n --nParticles -N --nSteps -x --nx -y --ny -z --nz -I --xDom -J --yDom -K --zDom -- "$@" || exit 2

mkdir -p "$BUILD/run"
cd "$BUILD/run"
echo "== Quicksilver $BACKEND: ranks=$N_RANKS topology=${PX}x${PY}x${PZ} mesh=${NX}x${NY}x${NZ} particles=$PARTICLES steps=$STEPS"
exec "$HPCPERF_LAUNCHER_BIN" --gpus "$N_RANKS" --bind wrapper -- "$EXE" \
  -i "$INPUT" -X "$NX" -Y "$NY" -Z "$NZ" -n "$PARTICLES" -N "$STEPS" \
  -x "$NX" -y "$NY" -z "$NZ" -I "$PX" -J "$PY" -K "$PZ" "$@"
