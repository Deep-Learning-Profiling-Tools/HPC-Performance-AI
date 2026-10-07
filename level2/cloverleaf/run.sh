#!/usr/bin/env bash
# Run the CloverLeaf standard benchmark problem.
#
#   ./run.sh [CUDA|HIP] [extra cloverleaf args]
#
# Default deck: InputDecks/clover_bm16.in (3840 x 3840 cells, 2955 steps,
# built-in reference check "test_problem 5"). Override with
# HPCPERF_CLOVERLEAF_DECK=<path> or append e.g. `--file InputDecks/clover_bm16_short.in`
# (later options win), or HPCPERF_CLOVERLEAF_INPUT=<id> for a registered input (inputs.yaml;
# mutually exclusive with the deck variable and extra arguments).
# (later options win). Ranks: HPCPERF_NP (default 1; multi-GPU via level2/tools/hpcperf_mpi_launch.sh).
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

INPUT_ID="${HPCPERF_CLOVERLEAF_INPUT:-}"
if [ -n "$INPUT_ID" ]; then
    # A registered input (inputs.yaml, tools/inputs/hpcperf_inputs.py) defines the deck; it is
    # refused together with HPCPERF_CLOVERLEAF_DECK or extra cloverleaf arguments (nothing is
    # silently overridden). The application log clover.out is per input and, because CloverLeaf
    # writes the field-summary rows only there, it is appended to stdout after the run.
    if [ -n "${HPCPERF_CLOVERLEAF_DECK:-}" ] || [ $# -gt 0 ]; then
        echo "run.sh: HPCPERF_CLOVERLEAF_INPUT=$INPUT_ID is mutually exclusive with HPCPERF_CLOVERLEAF_DECK and extra arguments" >&2
        exit 2
    fi
    DECK_REL="$(python3 "$R/tools/inputs/hpcperf_inputs.py" param "$HERE" "$INPUT_ID" deck)" || exit 2
    DECK="$HERE/$DECK_REL"
    OUT="$BUILD/clover.input.$INPUT_ID.out"
else
    DECK="${HPCPERF_CLOVERLEAF_DECK:-$HERE/InputDecks/clover_bm16.in}"
    OUT="$BUILD/clover.out"
fi
[ -f "$DECK" ] || { echo "run.sh: deck $DECK not found" >&2; exit 1; }
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../tools" && pwd)/hpcperf_launch_common.sh"
N_RANKS="$(hpcperf_ranks cloverleaf no)" || exit 2   # HPCPERF_GPUS (or legacy HPCPERF_NP); no scale modes yet
hpcperf_forbid_args cloverleaf --device -- "$@" || exit 2   # per-rank GPU comes from the launcher's wrapper
# One rank per GPU through the common launcher; CloverLeaf's --device is one
# shared value for all ranks, so the wrapper narrows CUDA_VISIBLE_DEVICES.
MPIRUN=("$HPCPERF_LAUNCHER_BIN" --gpus "$N_RANKS" --bind wrapper --)

echo "== CloverLeaf $BACKEND: ${MPIRUN[*]} $EXE --file $DECK --out $OUT $*${INPUT_ID:+ input=$INPUT_ID}"
if [ -n "$INPUT_ID" ]; then
    rm -f "$OUT"      # never a stale application log
    rc=0; "${MPIRUN[@]}" "$EXE" --file "$DECK" --out "$OUT" || rc=$?
    if [ -z "${HPCPERF_DRY_RUN:-}" ] && [ -f "$OUT" ]; then echo "### clover.out ($OUT)"; cat "$OUT"; fi
    exit "$rc"
fi
exec "${MPIRUN[@]}" "$EXE" --file "$DECK" --out "$OUT" "$@"
