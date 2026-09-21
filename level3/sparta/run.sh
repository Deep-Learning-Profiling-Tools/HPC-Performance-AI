#!/usr/bin/env bash
# Run the SPARTA collisional-flow benchmark (upstream bench/in.collide) on N GPUs.
#
#   ./run.sh [CUDA|HIP] [extra spa args...]
#
# Execution model (upstream Section_accelerate): one MPI rank per GPU, KOKKOS
# package on the device (`-k on g 1 -sf kk`), particles/grid distributed by
# SPARTA's own `balance_grid rcb part` -- any rank count is legal. Ranks go
# through the common launcher with the per-rank GPU wrapper (each rank sees
# exactly one GPU; mapping audited). GPU-aware MPI (`-pk kokkos gpu/aware
# yes`, SPARTA's default on GPUs) matches this repository's CUDA-aware
# Open MPI; HPCPERF_SPARTA_GPU_AWARE=no disables it.
#
# Resource / size controls (common Level 3 parameters):
#   HPCPERF_GPUS=N|all        ranks = GPUs (default 1)
#   HPCPERF_SCALE_MODE        smoke | strong | weak   (default smoke)
#     smoke  : upstream deck as shipped: 10x10x10 cells, 10 particles/cell =
#              10,000 particles; 30 equilibration + 100 benchmark steps
#              (reference log bench/log.7Jul14.collide.icc.10K.1)
#     strong : ONE fixed global grid, S^3 cells (S=HPCPERF_SPARTA_STRONG,
#              default 100: 1,000,000 cells = 10,000,000 particles), split by
#              SPARTA over the ranks
#     weak   : fixed work per rank, L^3 cells per rank (L=HPCPERF_SPARTA_LOCAL,
#              default 50: 125,000 cells = 1,250,000 particles/rank); grid
#              L*PX x L*PY x L*PZ with PXxPYxPZ from hpcperf_topology.py
#   HPCPERF_SPARTA_GPU_AWARE  yes|no (default yes)
#   HPCPERF_SPARTA_INPUT      a registered input id (inputs.yaml: in.collide/in.free/in.sphere at
#                             the upstream reference sizes); mutually exclusive with strong/weak
# The deck is upstream's bench/in.collide, unmodified; sizes enter through its
# own -var x/y/z variables (particles = 10 * cells by construction).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"; [ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
PROFILE="$(l3_backend_profile SPARTA "$MODEL")"
l3_paths_profile sparta "$PROFILE" "$MODEL" || exit 2     # same derivation as build.sh: binary, install and run tree of ONE profile
BUILD_DIR="$L3_BUILD"
EXE="$(find "$BUILD_DIR" -maxdepth 2 -name "spa_kokkos_$MODEL" -type f 2>/dev/null | head -1)"
[ -n "$EXE" ] && [ -x "$EXE" ] || { echo "run.sh: spa_kokkos_$MODEL not found under $BUILD_DIR -- run ./build.sh $BACKEND first (profile $PROFILE)" >&2; exit 1; }
l3_fingerprint_expect_backend "$L3_INSTALL" "$MODEL" || exit 1
l3_require_materialized "$HERE" || exit 3
SRC="$HERE/src"      # frozen source bundle (the deck bench/in.collide and its species files live inside it)

N_RANKS="$(hpcperf_ranks sparta yes)" || exit 2
hpcperf_forbid_args sparta -in -i -var -v -k -kokkos -sf -suffix -pk -package -log -- "$@" || exit 2
MODE="$(l3_scale_mode sparta)" || exit 2
GAM="${HPCPERF_SPARTA_GPU_AWARE:-yes}"

case "$MODE" in
    smoke)  X=10; Y=10; Z=10 ;;
    strong) S="${HPCPERF_SPARTA_STRONG:-100}"; X=$S; Y=$S; Z=$S ;;
    weak)   L="${HPCPERF_SPARTA_LOCAL:-50}"
            TOPO="$(hpcperf_topology sparta "$N_RANKS")" || exit 2
            read -r PX PY PZ <<< "$TOPO"
            X=$((L * PX)); Y=$((L * PY)); Z=$((L * PZ)) ;;
esac
DECK=in.collide; LABEL="$MODE"
INPUT_ID="${HPCPERF_SPARTA_INPUT:-}"
if [ -n "$INPUT_ID" ]; then
    # A registered input (inputs.yaml, tools/inputs/hpcperf_inputs.py) fixes the deck and its
    # x/y/z size; it is refused together with the scale modes and their size knobs so that
    # nothing is silently overridden. The deck must be one of the frozen bench decks.
    if [ "$MODE" != smoke ] || [ -n "${HPCPERF_SPARTA_STRONG:-}${HPCPERF_SPARTA_LOCAL:-}" ]; then
        echo "run.sh: HPCPERF_SPARTA_INPUT=$INPUT_ID is mutually exclusive with HPCPERF_SCALE_MODE=strong|weak and HPCPERF_SPARTA_STRONG/LOCAL" >&2
        exit 2
    fi
    param() { python3 "$R/tools/inputs/hpcperf_inputs.py" param "$HERE" "$INPUT_ID" "$1"; }
    DECK="$(param deck)" || exit 2
    case "$DECK" in in.collide|in.free|in.sphere) ;; *) echo "run.sh: input deck '$DECK' is not a frozen bench deck" >&2; exit 2 ;; esac
    X="$(param x)" || exit 2; Y="$(param y)" || exit 2; Z="$(param z)" || exit 2
    LABEL="input.$INPUT_ID"
fi
CELLS=$((X * Y * Z)); PARTS=$((10 * CELLS))     # nominal (in.sphere starts empty and fills by inflow)
RUN_DIR="$BUILD_DIR/$L3_RUN_SUBDIR"
[ -n "${HPCPERF_DRY_RUN:-}" ] && RUN_DIR="$RUN_DIR/.dryrun"   # dry-run never overwrites real results
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/log.$LABEL.np$N_RANKS.sparta"
rm -f "$LOG"      # validate only against THIS run's output; never a stale log
echo "# SPARTA $BACKEND profile=$PROFILE: mode=$MODE${INPUT_ID:+ input=$INPUT_ID deck=$DECK} ranks=$N_RANKS grid=${X}x${Y}x${Z} = $CELLS cells, $PARTS particles ($((PARTS / N_RANKS))/rank), gpu-aware=$GAM, log=$LOG"
cd "$SRC/bench"   # ar.species / ar.vss / data.sphere are referenced relative to the deck
RUN_ID="$(l3_run_id)"
rc=0
"$L3_LAUNCHER" --gpus "$N_RANKS" --bind wrapper -- \
    "$EXE" -k on g 1 -sf kk -pk kokkos gpu/aware "$GAM" \
    -in "$DECK" -var x "$X" -var y "$Y" -var z "$Z" -log "$LOG" -echo none "$@" || rc=$?
if [ -z "${HPCPERF_DRY_RUN:-}" ]; then
    l3_manifest "$RUN_DIR" "run_id=$RUN_ID" "app=sparta" "backend=$BACKEND" "profile=$PROFILE" "mode=$MODE" \
        "input_id=${INPUT_ID:-}" "deck=$DECK" \
        "ranks=$N_RANKS" "cells=$CELLS" "particles=$PARTS" "gpu_aware=$GAM" "exit_code=$rc" \
        "binary=$EXE" "binary_sha256=$(l3_sha_file "$EXE")" "input=$SRC/bench/$DECK" \
        "input_sha256=$(l3_sha_file "$SRC/bench/$DECK")" \
        "fingerprint=$L3_INSTALL/.hpcperf-l3-fingerprint" "fingerprint_sha256=$(l3_sha_file "$L3_INSTALL/.hpcperf-l3-fingerprint")" \
        "log=$LOG" "utc=$(date -u +%FT%TZ)"
fi
exit "$rc"
