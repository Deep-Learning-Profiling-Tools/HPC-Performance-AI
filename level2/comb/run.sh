#!/usr/bin/env bash
# One coupled MPI-decomposed Comb halo exchange, one rank per GPU.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null
set -euo pipefail

BACKEND=CUDA
case "${1:-}" in CUDA|cuda|HIP|hip) BACKEND="${1^^}"; shift;; esac
MODEL="${BACKEND,,}"
EXE="$R/build/level2/comb/$MODEL/bin/comb"
[ -x "$EXE" ] || { echo "run.sh: $EXE not found -- run $HERE/build.sh $BACKEND first" >&2; exit 1; }

# shellcheck disable=SC1091
source "$HERE/../tools/hpcperf_launch_common.sh"
export HPCPERF_GPU_BACKEND="$BACKEND"
N_RANKS="$(hpcperf_ranks comb no)" || exit 2
read -r PX PY PZ <<< "$(hpcperf_topology comb "$N_RANKS" --dims 3)"
LOCAL="${HPCPERF_COMB_LOCAL_SIZE:-128}"
CYCLES="${HPCPERF_COMB_CYCLES:-100}"
VARS="${HPCPERF_COMB_VARIABLES:-3}"
for value in "$LOCAL" "$CYCLES" "$VARS"; do
    case "$value" in ''|*[!0-9]*|0) echo "run.sh: Comb sizes/cycles/variables must be positive integers" >&2; exit 2;; esac
done
GX=$(( LOCAL * PX )); GY=$(( LOCAL * PY )); GZ=$(( LOCAL * PZ ))
GPU_AWARE="-${MODEL}_aware_mpi"
hpcperf_forbid_args comb -divide -periodic -ghost -vars -cycles -comm -exec -memory \
  -cuda_aware_mpi -hip_aware_mpi -- "$@" || exit 2

cd "$R/build/level2/comb/$MODEL"
echo "== Comb $BACKEND: ranks=$N_RANKS topology=${PX}x${PY}x${PZ} global=${GX}x${GY}x${GZ} local=${LOCAL}^3 cycles=$CYCLES"
exec "$HPCPERF_LAUNCHER_BIN" --gpus "$N_RANKS" --bind wrapper -- "$EXE" \
  "${GX}_${GY}_${GZ}" -divide "${PX}_${PY}_${PZ}" -periodic 1_1_1 \
  -ghost 2_2_2 -vars "$VARS" -cycles "$CYCLES" \
  -comm disable all -comm enable mpi -exec disable all -exec enable "$MODEL" \
  -memory disable all -memory mesh enable "${MODEL}_device" \
  -memory buffer enable "${MODEL}_device" "$GPU_AWARE" "$@"
