#!/usr/bin/env bash
# Run the standard AMG2023 benchmark problem on the CUDA or HIP build.
#
#   ./run.sh [CUDA|HIP] [extra amg args...]
#
# Distributed model: hypre BoomerAMG on a 3D grid block-decomposed over a
# px x py x pz process topology, ONE MPI RANK PER GPU, launched through
# level2/tools/hpcperf_mpi_launch.sh (site transport, GPU binding audit,
# no silent GPU sharing). hypre binds each rank to its GPU itself
# (hypre_bind_device); the launcher only audits.
#
# Resource / size controls (common Level 2 parameters):
#   HPCPERF_GPUS=N|all      ranks = GPUs to use (default 1)
#   HPCPERF_SCALE_MODE      smoke | strong | weak   (default weak)
#     weak   : fixed LOCAL grid per rank, HPCPERF_AMG_N^3 (default 256^3);
#              global grid grows with the topology  [the historical deck:
#              1 rank x 256^3 local == the Level 2 single-GPU benchmark]
#     strong : fixed GLOBAL grid, HPCPERF_AMG_GLOBAL^3 (default 512^3),
#              divided over the topology (each factor must divide it)
#     smoke  : fixed 128^3 local, seconds-fast correctness/bring-up run
#   HPCPERF_AMG_N           weak/smoke local points per dim (default 256/128)
#   HPCPERF_AMG_GLOBAL      strong global points per dim (default 512)
#   HPCPERF_AMG_P="px py pz" explicit topology override (product must equal
#                           the rank count; default: hpcperf_topology.py)
#   HPCPERF_AMG_PROBLEM     1 (27pt, AMG-GMRES) or 2 (7pt, AMG-PCG); default 1
#   HPCPERF_GPUS/HPCPERF_NP mismatch is an error (one rank per GPU).
#
# Constraint checked here: hypre is built with 32-bit HYPRE_Int, so the global
# unknown count (px*nx)*(py*ny)*(pz*nz) must stay below 2^31.
#
# Extra args are appended and win (amg's parser takes the last occurrence),
# e.g. `./run.sh CUDA -printstats`.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true

set -euo pipefail

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
[ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level2/amg2023/$MODEL"
EXE="$BUILD_DIR/amg"

if [ ! -x "$EXE" ]; then
    echo "run.sh: $EXE not found -- run ./build.sh $BACKEND first" >&2
    exit 1
fi

PROBLEM="${HPCPERF_AMG_PROBLEM:-1}"
MODE="${HPCPERF_SCALE_MODE:-weak}"

# ----- rank count (one per GPU); HPCPERF_AMG_P may imply it for compatibility
# shellcheck disable=SC1091
source "$R/level2/tools/hpcperf_launch_common.sh"
hpcperf_forbid_args amg2023 -P -n -- "$@" || exit 2   # topology/size only via the checked variables
if [ -n "${HPCPERF_AMG_P:-}" ]; then
    read -r PX PY PZ <<< "$HPCPERF_AMG_P"
    P_PROD=$((PX * PY * PZ))
    if [ -z "${HPCPERF_GPUS:-}${HPCPERF_NP:-}" ]; then export HPCPERF_GPUS="$P_PROD"; fi
fi
N_RANKS="$(hpcperf_ranks amg2023 yes)" || exit 2
if [ -n "${HPCPERF_AMG_P:-}" ] && [ "$N_RANKS" -ne "$P_PROD" ]; then
    echo "run.sh: HPCPERF_AMG_P=$HPCPERF_AMG_P (product $P_PROD) does not match the rank count $N_RANKS (HPCPERF_GPUS)" >&2
    exit 2
fi

# ----- topology
if [ -z "${HPCPERF_AMG_P:-}" ]; then
    if [ "$MODE" = strong ]; then
        G="${HPCPERF_AMG_GLOBAL:-512}"
        TOPO="$(hpcperf_topology amg2023 "$N_RANKS" --divides "$G,$G,$G")" || exit 2
    else
        TOPO="$(hpcperf_topology amg2023 "$N_RANKS")" || exit 2
    fi
    read -r PX PY PZ <<< "$TOPO"
fi

# ----- local size per mode
case "$MODE" in
    weak)   NX="${HPCPERF_AMG_N:-256}"; NY=$NX; NZ=$NX ;;
    smoke)  NX="${HPCPERF_AMG_N:-128}"; NY=$NX; NZ=$NX ;;
    strong) G="${HPCPERF_AMG_GLOBAL:-512}"
            NX=$((G / PX)); NY=$((G / PY)); NZ=$((G / PZ))
            [ $((NX * PX)) -eq "$G" ] && [ $((NY * PY)) -eq "$G" ] && [ $((NZ * PZ)) -eq "$G" ] \
                || { echo "run.sh: topology ${PX}x${PY}x${PZ} does not divide global ${G}^3" >&2; exit 2; } ;;
    *) echo "run.sh: HPCPERF_SCALE_MODE must be smoke|strong|weak (got '$MODE')" >&2; exit 2 ;;
esac

# ----- 32-bit HYPRE_Int guard: global unknowns < 2^31
GLOBAL_UNKNOWNS=$(( (NX * PX) * (NY * PY) * (NZ * PZ) ))
if [ "$GLOBAL_UNKNOWNS" -ge 2147483648 ]; then
    echo "run.sh: global grid $((NX*PX))x$((NY*PY))x$((NZ*PZ)) = $GLOBAL_UNKNOWNS unknowns" >&2
    echo "run.sh: exceeds the 32-bit HYPRE_Int limit (2^31) of this hypre build;" >&2
    echo "run.sh: reduce HPCPERF_AMG_N / HPCPERF_AMG_GLOBAL or the rank count." >&2
    exit 2
fi

echo "# AMG2023 $BACKEND: mode=$MODE ranks=$N_RANKS topology=${PX}x${PY}x${PZ} local=${NX}x${NY}x${NZ} global=$((NX*PX))x$((NY*PY))x$((NZ*PZ)) ($GLOBAL_UNKNOWNS unknowns)"
exec "$R/level2/tools/hpcperf_mpi_launch.sh" --gpus "$N_RANKS" --bind app -- \
    "$EXE" -P "$PX" "$PY" "$PZ" -n "$NX" "$NY" "$NZ" -problem "$PROBLEM" "$@"
