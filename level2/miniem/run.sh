#!/usr/bin/env bash
# Run MiniEM (Trilinos Panzer mini-em, BlockPrec driver): implicit Maxwell time stepping with the MueLu
# RefMaxwell block preconditioner on an inline-generated 3D mesh.
#
#   ./run.sh [CUDA] [extra BlockPrec args...]
#
# Distributed model: native MPI (Tpetra/Kokkos), ONE MPI RANK PER GPU, launched through
# level2/tools/hpcperf_mpi_launch.sh (site transport, GPU binding audit, no silent GPU sharing). The mesh
# factory (Panzer_STK inline mesh, "X/Y/Z Procs = -1") decomposes the global element grid over the ranks itself.
#
# Resource / size controls (common Level 2 parameters):
#   HPCPERF_GPUS=N|all      ranks = GPUs to use (default 1)
#   HPCPERF_SCALE_MODE      smoke | strong | weak   (default weak)
#     smoke  : upstream's `Maxwell_MueLu order1` case: maxwell.xml, 15^3 hex elements, 1 time step
#     strong : upstream's performance deck maxwell-large.xml (tet mesh) with a fixed GLOBAL element grid of
#              HPCPERF_MINIEM_GLOBAL^3 (default 64^3), HPCPERF_MINIEM_STEPS time steps (default 3, as the
#              upstream MiniEM-BlockPrec_RefMaxwell_Performance tests)
#     weak   : the same deck with a fixed LOCAL grid per rank: HPCPERF_MINIEM_N^3 elements (default 48^3),
#              global grid = local x the process topology from hpcperf_topology.py
#   HPCPERF_MINIEM_N        weak local elements per dimension (default 48)
#   HPCPERF_MINIEM_GLOBAL   strong global elements per dimension (default 64)
#   HPCPERF_MINIEM_STEPS    time steps for strong/weak (default 3)
#   HPCPERF_MINIEM_DECK     deck name inside the build's decks/ directory (overrides the mode's deck)
#   HPCPERF_GPUS/HPCPERF_NP mismatch is an error (one rank per GPU).
#
# Solver: --solver=MueLu (RefMaxwell) --linAlgebra=Tpetra, i.e. upstream's GPU configuration; extra args are
# appended and win. Run from <build>/decks so the solver-parameter files referenced by the decks resolve.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true
# Registered inputs (inputs.yaml): HPCPERF_MINIEM_INPUT=<id> supplies this script's knobs / extra arguments
# (tools/inputs/README.md); it is refused together with a conflicting pre-set knob or an unknown id.
# shellcheck disable=SC1091
source "$R/tools/inputs/hpcperf_input_selector.sh"
hpcperf_apply_input "$HERE" HPCPERF_MINIEM_INPUT || exit 2

set -euo pipefail

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
[ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level2/miniem/$MODEL"
EXE="$BUILD_DIR/PanzerMiniEM_BlockPrec"
DECKS="$BUILD_DIR/decks"

if [ ! -x "$EXE" ]; then
    echo "run.sh: $EXE not found -- run ./build.sh $BACKEND first" >&2
    exit 1
fi

MODE="${HPCPERF_SCALE_MODE:-weak}"

# shellcheck disable=SC1091
source "$R/level2/tools/hpcperf_launch_common.sh"
hpcperf_forbid_args miniem --x-elements --y-elements --z-elements --inputFile -- "$@" || exit 2   # size/deck only via the checked variables
N_RANKS="$(hpcperf_ranks miniem yes)" || exit 2

STEPS="${HPCPERF_MINIEM_STEPS:-3}"
case "$MODE" in
    smoke)
        DECK="${HPCPERF_MINIEM_DECK:-maxwell.xml}"; STEPS="${HPCPERF_MINIEM_STEPS:-1}"; SIZE=()
        DESC="deck=$DECK (15^3 hex, upstream Maxwell_MueLu order1)" ;;
    strong)
        DECK="${HPCPERF_MINIEM_DECK:-maxwell-large.xml}"; G="${HPCPERF_MINIEM_GLOBAL:-64}"
        SIZE=(--x-elements="$G" --y-elements="$G" --z-elements="$G")
        DESC="deck=$DECK global=${G}x${G}x${G} elements" ;;
    weak)
        DECK="${HPCPERF_MINIEM_DECK:-maxwell-large.xml}"; NL="${HPCPERF_MINIEM_N:-48}"
        TOPO="$(hpcperf_topology miniem "$N_RANKS")" || exit 2
        read -r PX PY PZ <<< "$TOPO"
        SIZE=(--x-elements=$((NL * PX)) --y-elements=$((NL * PY)) --z-elements=$((NL * PZ)))
        DESC="deck=$DECK local=${NL}^3 x topology ${PX}x${PY}x${PZ} = global $((NL*PX))x$((NL*PY))x$((NL*PZ)) elements" ;;
    *) echo "run.sh: HPCPERF_SCALE_MODE must be smoke|strong|weak (got '$MODE')" >&2; exit 2 ;;
esac
[ -f "$DECKS/$DECK" ] || { echo "run.sh: deck $DECK not found in $DECKS" >&2; exit 2; }

# MPI on device buffers: Tpetra asks Open MPI whether it is CUDA-aware (this conda build says yes through
# smcuda) and then hands GPU pointers to MPI. On this site that path hangs at 4 ranks inside the RefMaxwell
# setup (all ranks spinning in a Tpetra import) -- the same smcuda GPU-aware transport the Level 2/3 notes flag
# as slow or unreliable. Every other Level 2 dependency is built with GPU-aware MPI off, so MiniEM does the same:
# Tpetra stages communication buffers through the host. Override with TPETRA_ASSUME_GPU_AWARE_MPI=1 to test.
export TPETRA_ASSUME_GPU_AWARE_MPI="${TPETRA_ASSUME_GPU_AWARE_MPI:-0}"
echo "# MiniEM $BACKEND: mode=$MODE ranks=$N_RANKS $DESC steps=$STEPS solver=MueLu(RefMaxwell)/Tpetra gpu_aware_mpi=$TPETRA_ASSUME_GPU_AWARE_MPI"
cd "$DECKS"
exec "$R/level2/tools/hpcperf_mpi_launch.sh" --gpus "$N_RANKS" -- \
    "$EXE" --inputFile="$DECK" --solver=MueLu --linAlgebra=Tpetra --numTimeSteps="$STEPS" "${SIZE[@]}" --stacked-timer ${HPCPERF_INPUT_ARGS[@]+"${HPCPERF_INPUT_ARGS[@]}"} "$@"
