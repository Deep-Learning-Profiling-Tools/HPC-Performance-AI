#!/usr/bin/env bash
# Run the standard TeaLeaf benchmark problem on the CUDA or HIP build.
#
#   ./run.sh [CUDA|HIP] [extra tealeaf args...]
#
# Default deck: Benchmarks/tea_bm_5.in (4000x4000 cells, 10 timesteps, CG
# solver, eps 1e-15) -- the standard TeaLeaf "bm_5" benchmark. Runs with
# `mpirun -np 1` when the binary was built with MPI (build.sh default).
#
# Environment overrides:
#   HPCPERF_TEALEAF_DECK  input deck (absolute path, or a name relative to
#                         level2/tealeaf, e.g. Benchmarks/tea_bm_4.in)
#   HPCPERF_NP            number of MPI ranks (default 1)
#
# Other upstream decks (all validated against tea.problems by the app):
#   Benchmarks/tea_bm_4.in      1000x1000 @ 10 steps  (~1 s on B200)
#   Benchmarks/tea_bm_5e_2.in   2000x2000 @ 10 steps
#   Benchmarks/tea_bm_5.in      4000x4000 @ 10 steps  (default, ~25 s)
#   Benchmarks/tea_bm_6.in      8000x8000 @ 10 steps  (several minutes)
#   Benchmarks/tea_bm_5e_{1,2,4,8}_{2,4}.in  -- same grids, 2 or 4 steps
#   tea.in                      512x512 @ 20 steps    (used by validate.sh)
# tea_bm_1..3 (10, 250, 500 cells) have no entry in tea.problems and therefore
# report FAILED at the end; they are not meaningful GPU workloads anyway.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true

set -euo pipefail

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
[ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level2/tealeaf/$MODEL"
EXE="$BUILD_DIR/${MODEL}-tealeaf"

if [ ! -x "$EXE" ]; then
    echo "run.sh: $EXE not found -- run ./build.sh $BACKEND first" >&2
    exit 1
fi

DECK="${HPCPERF_TEALEAF_DECK:-Benchmarks/tea_bm_5.in}"
case "$DECK" in /*) ;; *) DECK="$HERE/$DECK" ;; esac
if [ ! -f "$DECK" ]; then
    echo "run.sh: deck $DECK not found" >&2
    exit 1
fi

# tea.out (the log file) goes under the build tree, not the caller's cwd.
OUT_DIR="$BUILD_DIR/run"
mkdir -p "$OUT_DIR"

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../tools" && pwd)/hpcperf_launch_common.sh"
N_RANKS="$(hpcperf_ranks tealeaf no)" || exit 2   # HPCPERF_GPUS (or legacy HPCPERF_NP); no scale modes yet
hpcperf_forbid_args tealeaf --device -d -- "$@" || exit 2   # per-rank GPU comes from the launcher's wrapper
LAUNCH=()
if grep -q '^ENABLE_MPI:[A-Z]*=ON' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    # One rank per GPU through the common launcher; TeaLeaf's --device is one
    # shared value for all ranks, so the wrapper narrows CUDA_VISIBLE_DEVICES.
    LAUNCH=("$HPCPERF_LAUNCHER_BIN" --gpus "$N_RANKS" --bind wrapper --)
elif [ "$N_RANKS" -ne 1 ]; then
    echo "run.sh: this TeaLeaf build has ENABLE_MPI=OFF but $N_RANKS ranks were requested" >&2
    exit 2
fi

echo "# TeaLeaf $BACKEND: ${LAUNCH[*]:-} $EXE --file $DECK"
exec "${LAUNCH[@]}" "$EXE" \
    --file "$DECK" \
    --problems "$HERE/tea.problems" \
    --out "$OUT_DIR/tea.out" \
    "$@"
