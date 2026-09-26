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
# Registered inputs (inputs.yaml, tools/inputs/hpcperf_inputs.py): HPCPERF_QUICKSILVER_INPUT_ID=<id>.
# A registered upstream deck (verbatim: true) carries its own mesh/particle/step counts and is
# passed with -i only, on exactly one rank; the derived default input reuses the size knobs above
# with the values recorded in inputs.yaml. The id is mutually exclusive with the size/deck knobs.
INPUT_ID="${HPCPERF_QUICKSILVER_INPUT_ID:-}"
VERBATIM=false
if [ -n "$INPUT_ID" ]; then
    for v in HPCPERF_QUICKSILVER_INPUT HPCPERF_QUICKSILVER_CELLS_PER_RANK HPCPERF_QUICKSILVER_PARTICLES_PER_RANK HPCPERF_QUICKSILVER_STEPS; do
        [ -z "${!v:-}" ] || { echo "run.sh: HPCPERF_QUICKSILVER_INPUT_ID=$INPUT_ID and $v are mutually exclusive (the input id defines deck and sizes)" >&2; exit 2; }
    done
    TOOL="$R/tools/inputs/hpcperf_inputs.py"
    DECK_REL="$(python3 "$TOOL" param "$HERE" "$INPUT_ID" deck)" || exit 2
    VERBATIM="$(python3 "$TOOL" param "$HERE" "$INPUT_ID" verbatim | tr '[:upper:]' '[:lower:]')" || exit 2
    INPUT="$HERE/$DECK_REL"
    if [ "$VERBATIM" = true ]; then
        [ "$N_RANKS" -eq 1 ] || { echo "run.sh: input '$INPUT_ID' is an upstream single-rank deck (xDom*yDom*zDom = 1 inside the deck); HPCPERF_GPUS=$N_RANKS is refused, never silently re-decomposed" >&2; exit 2; }
    else
        LOCAL="$(python3 "$TOOL" param "$HERE" "$INPUT_ID" cells_per_rank)" || exit 2
        PPR="$(python3 "$TOOL" param "$HERE" "$INPUT_ID" particles_per_rank)" || exit 2
        STEPS="$(python3 "$TOOL" param "$HERE" "$INPUT_ID" steps)" || exit 2
        NX=$(( LOCAL * PX )); NY=$(( LOCAL * PY )); NZ=$(( LOCAL * PZ )); PARTICLES=$(( PPR * N_RANKS ))
    fi
fi
[ -f "$INPUT" ] || { echo "run.sh: input not found: $INPUT" >&2; exit 1; }
hpcperf_forbid_args quicksilver -i --inputFile -X --lx -Y --ly -Z --lz -n --nParticles -N --nSteps -x --nx -y --ny -z --nz -I --xDom -J --yDom -K --zDom -- "$@" || exit 2

mkdir -p "$BUILD/run"
cd "$BUILD/run"
if [ "$VERBATIM" = true ]; then
    echo "== Quicksilver $BACKEND: ranks=$N_RANKS input=$INPUT_ID deck=$INPUT (verbatim upstream deck: mesh/particles/steps from the deck)"
    exec "$HPCPERF_LAUNCHER_BIN" --gpus "$N_RANKS" --bind wrapper -- "$EXE" -i "$INPUT" "$@"
fi
echo "== Quicksilver $BACKEND: ranks=$N_RANKS topology=${PX}x${PY}x${PZ} mesh=${NX}x${NY}x${NZ} particles=$PARTICLES steps=$STEPS${INPUT_ID:+ input=$INPUT_ID}"
exec "$HPCPERF_LAUNCHER_BIN" --gpus "$N_RANKS" --bind wrapper -- "$EXE" \
  -i "$INPUT" -X "$NX" -Y "$NY" -Z "$NZ" -n "$PARTICLES" -N "$STEPS" \
  -x "$NX" -y "$NY" -z "$NZ" -I "$PX" -J "$PY" -K "$PZ" "$@"
