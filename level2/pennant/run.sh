#!/usr/bin/env bash
# Run PENNANT (CUDA, single GPU) on one input deck.
#
#   ./run.sh [CUDA] [<deck>.pnt]
#
# Default deck: test/sedovbig/sedovbig.pnt (the Sedov blast wave on the 540 x 540 rectangular mesh =
# 291,600 zones, run to tstop = 1.0 -- upstream's large Sedov problem). Any other upstream deck can be
# given (test/<name>/<name>.pnt); PENNANT takes exactly one argument, so nothing else is accepted.
# HPCPERF_PENNANT_INPUT=<id> selects a registered input (inputs.yaml, tools/inputs/README.md): its deck
# is the argument; the selector is refused together with an explicit deck argument.
#
# PENNANT writes its outputs (<name>.xy per-zone profile, Ensight Gold <name>.case/.geo/.zr/.ze/.zp) next
# to the deck, named after it, so the deck is copied byte-identically into the run directory
# $R/build/level2/pennant/cuda/run/<name>/ and the program runs there on the copy's absolute path
# (never inside the vendored tree). One run directory per deck name; a rerun overwrites it.
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
# Registered inputs: HPCPERF_PENNANT_INPUT=<id> supplies the deck as this script's argument.
# shellcheck disable=SC1091
source "$R/tools/inputs/hpcperf_input_selector.sh"
hpcperf_apply_input "$HERE" HPCPERF_PENNANT_INPUT || exit 2

BACKEND="CUDA"
case "$(printf '%s' "${1:-}" | tr '[:lower:]' '[:upper:]')" in
    CUDA) BACKEND="CUDA"; shift ;;
    HIP)  echo "run.sh: no HIP build of PENNANT here (upstream branch 'hip' is not vendored)" >&2; exit 1 ;;
esac
BUILD_DIR="$R/build/level2/pennant/cuda"
EXE="$BUILD_DIR/build/pennant"
if [ ! -x "$EXE" ]; then
    echo "run.sh: $EXE not found -- run $HERE/build.sh CUDA first" >&2
    exit 1
fi

if [ -n "${HPCPERF_PENNANT_INPUT:-}" ]; then
    if [ $# -gt 0 ]; then
        echo "run.sh: HPCPERF_PENNANT_INPUT=$HPCPERF_PENNANT_INPUT is mutually exclusive with a deck argument" >&2
        exit 2
    fi
    set -- "${HPCPERF_INPUT_ARGS[@]}"
fi
case $# in
    0) DECK="$HERE/test/sedovbig/sedovbig.pnt" ;;
    1) DECK="$1" ;;
    *) echo "run.sh: PENNANT takes exactly one deck argument (got $#: $*)" >&2; exit 2 ;;
esac
case "$DECK" in /*) ;; *) DECK="$PWD/$DECK" ;; esac
[ -f "$DECK" ] || { echo "run.sh: deck $DECK not found" >&2; exit 1; }
case "$DECK" in *.pnt) ;; *) echo "run.sh: the deck must be a .pnt file (got $DECK)" >&2; exit 2 ;; esac
NAME="$(basename "$DECK" .pnt)"
RUN_DIR="$BUILD_DIR/run/$NAME"
mkdir -p "$RUN_DIR"
cp -p "$DECK" "$RUN_DIR/$NAME.pnt"
rm -f "$RUN_DIR/$NAME.xy"            # the check reads only THIS run's profile, never a stale one
cd "$RUN_DIR"
echo "== PENNANT $BACKEND: deck=$DECK (copy $RUN_DIR/$NAME.pnt)${HPCPERF_PENNANT_INPUT:+ input=$HPCPERF_PENNANT_INPUT}"
exec "$EXE" "$RUN_DIR/$NAME.pnt"
