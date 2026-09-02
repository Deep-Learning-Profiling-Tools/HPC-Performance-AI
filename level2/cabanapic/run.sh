#!/usr/bin/env bash
# Run the standard CabanaPIC benchmark problem on the CUDA or HIP build.
#
#   ./run.sh [CUDA|HIP]
#
# Standard problem (deck decks/hpcperf_weibel.cxx, compiled in by build.sh):
# the upstream Weibel / filamentation instability -- two counter-streaming
# electron beams on a periodic 1 x ny x 1 grid, electromagnetic field solver,
# double precision -- at GPU size:
#
#   ny = 512 cells, nppc = 2000 particles/cell  -> 1,024,000 particles
#   num_steps = 96000                            -> t = 117.3 / omega_pe
#
# Upstream's built-in deck is the same problem at ny = 32, nppc = 100,
# num_steps = 6000 (3200 particles). Only these three numbers differ: the grid
# is 16x finer (dt shrinks with it, so 16x the steps cover exactly the same
# physical time, through growth and saturation of the instability) and there
# are 20x more particles per cell. v0, box lengths, n0, boundaries and the
# time-step rule are untouched. About 72 s on a B200.
#
# Environment overrides (read by the deck at start-up; no rebuild needed):
#   HPCPERF_NY / HPCPERF_NPPC / HPCPERF_NUM_STEPS
#                            problem size (defaults above; the upstream
#                            problem is 32 / 100 / 6000, ~2 s)
#   HPCPERF_CABANAPIC_RUNDIR directory to run in (created if needed, kept
#                            afterwards). Default: a fresh directory under
#                            ${TMPDIR:-/tmp}, deleted after the results have
#                            been copied to build/level2/cabanapic/<model>/run/.
#                            CabanaPIC opens, appends and closes energies.txt
#                            in the working directory on EVERY time step, which
#                            on an NFS working directory costs ~2-3 ms per step
#                            -- more than the physics itself (see README).
#
# CabanaPIC makes no MPI calls (libmpi is only linked in through Cabana), so
# the binary is run directly, without mpirun. Per-step particle/field dumps are
# compiled out by build.sh (PARTICLE_DUMP_INTERVAL=0); the per-step energies
# (step, time, E energy, B energy) go to energies.txt and to the log.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true

set -euo pipefail

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
[ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level2/cabanapic/$MODEL"
EXE="$BUILD_DIR/example/cbnpic"

if [ ! -x "$EXE" ]; then
    echo "run.sh: $EXE not found -- run ./build.sh $BACKEND first" >&2
    exit 1
fi

export HPCPERF_NY="${HPCPERF_NY:-512}"
export HPCPERF_NPPC="${HPCPERF_NPPC:-2000}"
export HPCPERF_NUM_STEPS="${HPCPERF_NUM_STEPS:-96000}"

OUT_DIR="$BUILD_DIR/run"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"

if [ -n "${HPCPERF_CABANAPIC_RUNDIR:-}" ]; then
    RUN_DIR="$HPCPERF_CABANAPIC_RUNDIR"
    mkdir -p "$RUN_DIR"
    KEEP=1
else
    RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hpcperf-cabanapic.XXXXXX")"
    KEEP=0
fi
cleanup() { [ "$KEEP" -eq 0 ] && rm -rf "$RUN_DIR"; return 0; }
trap cleanup EXIT

# CabanaPIC appends to energies.txt, so stale files must go first.
rm -f "$RUN_DIR/energies.txt" "$RUN_DIR/partloc" "$RUN_DIR/ex1d"

echo "# CabanaPIC $BACKEND: $EXE  (ny=$HPCPERF_NY nppc=$HPCPERF_NPPC num_steps=$HPCPERF_NUM_STEPS)"
echo "# working directory: $RUN_DIR; log: $LOG"
cd "$RUN_DIR"
t0="$(date +%s.%N)"
set +e
"$EXE" > "$LOG" 2>&1
rc=$?
set -e
t1="$(date +%s.%N)"
WALL="$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.2f", b - a }')"

# Run header (everything the app prints before the per-step energy lines).
grep -v -E '^[0-9]+ [0-9.]+ ' "$LOG" | grep -v -E '^\s*$' || true
echo "----------------------------------------------------------------"
if [ -f energies.txt ]; then
    cp energies.txt "$OUT_DIR/energies.txt"
    echo "# energies.txt: $(wc -l < energies.txt) steps written (copied to $OUT_DIR/energies.txt)"
    echo "# last line (step time E_energy B_energy): $(tail -1 energies.txt)"
fi
echo "# CabanaPIC $BACKEND ny=$HPCPERF_NY nppc=$HPCPERF_NPPC num_steps=$HPCPERF_NUM_STEPS: wall time ${WALL} s (exit code $rc)"
exit "$rc"
