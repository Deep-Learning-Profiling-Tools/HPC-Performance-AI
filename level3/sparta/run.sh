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
CELLS=$((X * Y * Z)); PARTS=$((10 * CELLS))
RUN_DIR="$BUILD_DIR/$L3_RUN_SUBDIR"
[ -n "${HPCPERF_DRY_RUN:-}" ] && RUN_DIR="$RUN_DIR/.dryrun"   # dry-run never overwrites real results
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/log.$MODE.np$N_RANKS.sparta"
rm -f "$LOG"      # validate only against THIS run's output; never a stale log
echo "# SPARTA $BACKEND profile=$PROFILE: mode=$MODE ranks=$N_RANKS grid=${X}x${Y}x${Z} = $CELLS cells, $PARTS particles ($((PARTS / N_RANKS))/rank), gpu-aware=$GAM, log=$LOG"
cd "$SRC/bench"   # ar.species / ar.vss are referenced relative to the deck
RUN_ID="$(l3_run_id)"
rc=0
"$L3_LAUNCHER" --gpus "$N_RANKS" --bind wrapper -- \
    "$EXE" -k on g 1 -sf kk -pk kokkos gpu/aware "$GAM" \
    -in in.collide -var x "$X" -var y "$Y" -var z "$Z" -log "$LOG" -echo none "$@" || rc=$?
if [ -z "${HPCPERF_DRY_RUN:-}" ]; then
    l3_manifest "$RUN_DIR" "run_id=$RUN_ID" "app=sparta" "backend=$BACKEND" "profile=$PROFILE" "mode=$MODE" \
        "ranks=$N_RANKS" "cells=$CELLS" "particles=$PARTS" "gpu_aware=$GAM" "exit_code=$rc" \
        "binary=$EXE" "binary_sha256=$(l3_sha_file "$EXE")" "input=$SRC/bench/in.collide" \
        "input_sha256=$(l3_sha_file "$SRC/bench/in.collide")" \
        "fingerprint=$L3_INSTALL/.hpcperf-l3-fingerprint" "fingerprint_sha256=$(l3_sha_file "$L3_INSTALL/.hpcperf-l3-fingerprint")" \
        "log=$LOG" "utc=$(date -u +%FT%TZ)"
fi
exit "$rc"
