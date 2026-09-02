#!/usr/bin/env bash
# Run the CloverLeaf standard benchmark problem.
#
#   ./run.sh [CUDA|HIP] [extra cloverleaf args]
#
# Default deck: InputDecks/clover_bm16.in (3840 x 3840 cells, 2955 steps,
# built-in reference check "test_problem 5"). Override with
# HPCPERF_CLOVERLEAF_DECK=<path> or append e.g. `--file InputDecks/clover_bm16_short.in`
# (later options win). Ranks: HPCPERF_NP (default 1; >1 adds --oversubscribe).
# clover.out is written to the build tree, not the cwd.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null
set -euo pipefail

BACKEND=CUDA
case "${1:-}" in
  CUDA|cuda|HIP|hip) BACKEND="${1^^}"; shift ;;
esac
MODEL="${BACKEND,,}"

BUILD="$R/build/level2/cloverleaf/$MODEL"
EXE="$BUILD/$MODEL-cloverleaf"
if [ ! -x "$EXE" ]; then
  echo "run.sh: $EXE not found -- run $HERE/build.sh $BACKEND first" >&2
  exit 1
fi

DECK="${HPCPERF_CLOVERLEAF_DECK:-$HERE/InputDecks/clover_bm16.in}"
OUT="$BUILD/clover.out"
NP="${HPCPERF_NP:-1}"
MPIRUN=(mpirun -np "$NP")
if [ "$NP" -gt 1 ]; then MPIRUN+=(--oversubscribe); fi

echo "== CloverLeaf $BACKEND: ${MPIRUN[*]} $EXE --file $DECK --out $OUT $*"
exec "${MPIRUN[@]}" "$EXE" --file "$DECK" --out "$OUT" "$@"
