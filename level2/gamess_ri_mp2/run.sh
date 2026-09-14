#!/usr/bin/env bash
# MPI distributes one RI-MP2 pair domain per rank; each rank owns one GPU.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null
set -euo pipefail

BACKEND=CUDA
case "${1:-}" in CUDA|cuda|HIP|hip) BACKEND="${1^^}"; shift;; esac
MODEL="${BACKEND,,}"; BUILD="$R/build/level2/gamess_ri_mp2/$MODEL"
if [ "$BACKEND" = CUDA ]; then EXE="$BUILD/rimp2-cublas"; else EXE="$BUILD/rimp2-hipblas"; fi
[ -x "$EXE" ] || { echo "run.sh: $EXE not found -- run $HERE/build.sh $BACKEND first" >&2; exit 1; }
# shellcheck disable=SC1091
source "$HERE/../tools/hpcperf_launch_common.sh"
export HPCPERF_GPU_BACKEND="$BACKEND"
N_RANKS="$(hpcperf_ranks gamess_ri_mp2 no)" || exit 2
INPUT="${HPCPERF_GAMESS_INPUT:-w30.rand}"
NQVV="${HPCPERF_GAMESS_NQVV:-30}"
case "$NQVV" in ''|*[!0-9]*|0) echo "run.sh: HPCPERF_GAMESS_NQVV must be a positive integer" >&2; exit 2;; esac
if [[ "$INPUT" == *.kern ]] && [ ! -f "$HERE/inputs/$INPUT" ]; then
    echo "run.sh: kernel input must be a basename present in $HERE/inputs (missing $INPUT)" >&2; exit 1
fi
# Upstream retires ranks above NACT before entering the distributed pair loop.
# Refuse that idle-rank configuration so every requested GPU contributes.
case "$INPUT" in
  benz.rand|benz.kern) MAX_RANKS=15;;
  cor.rand|cor.kern) MAX_RANKS=54;;
  c60.rand|c60.kern|w30.rand|w30.kern) MAX_RANKS=120;;
  w60.rand|w60.kern) MAX_RANKS=240;;
  *[!0-9]*|'') MAX_RANKS=0;;
  *) MAX_RANKS=$(( 4 * INPUT ));;
esac
if [ "$MAX_RANKS" -gt 0 ] && [ "$N_RANKS" -gt "$MAX_RANKS" ]; then
    echo "run.sh: requested $N_RANKS GPUs, but input $INPUT has only $MAX_RANKS active-orbital work partitions" >&2
    exit 2
fi
cd "$HERE/inputs"
echo "== GAMESS RI-MP2 $BACKEND: ranks=$N_RANKS input=$INPUT NQVV=$NQVV"
exec "$HPCPERF_LAUNCHER_BIN" --gpus "$N_RANKS" --bind wrapper -- "$EXE" "$INPUT" "$NQVV" "$@"
