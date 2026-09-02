#!/usr/bin/env bash
# Run the standard AMG2023 benchmark problem on the CUDA or HIP build.
#
#   ./run.sh [CUDA|HIP] [extra amg args...]
#
# Default problem: Problem 1 (upstream default) -- 3D diffusion problem on a
# cuboid, 27-point stencil, solved with AMG-GMRES(100), relative residual
# stopping criterion 1e-12; one MPI rank, 256 x 256 x 256 local grid
# (16,777,216 unknowns). ~9 s wall clock and ~30 GB of device memory on a
# B200. The amg-doc reference GPU study (1 V100) sweeps this problem from
# n=50 to n=200 in steps of 10; 256 is a B200-sized point on the same curve.
#
# Extra args are appended to the command line; amg's parser lets the last
# occurrence win, so e.g. `./run.sh CUDA -n 320 320 320` or
# `./run.sh CUDA -problem 2` override the defaults. `-printstats` adds
# hypre's AMG setup/convergence tables.
#
# Environment overrides:
#   HPCPERF_AMG_N        local grid points per dimension (default 256)
#   HPCPERF_AMG_PROBLEM  1 (27pt, AMG-GMRES) or 2 (7pt, AMG-PCG); default 1
#   HPCPERF_AMG_P        MPI process topology "px py pz" (default "1 1 1")
#   HPCPERF_NP           number of MPI ranks (default: px*py*pz)
#
# Note: `-n` is the LOCAL size per rank, so the global grid is
# (px*nx) x (py*ny) x (pz*nz). hypre here is built with 32-bit HYPRE_Int, so
# keep the global unknown count below 2^31.

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

N="${HPCPERF_AMG_N:-256}"
PROBLEM="${HPCPERF_AMG_PROBLEM:-1}"
read -r PX PY PZ <<< "${HPCPERF_AMG_P:-1 1 1}"
NP="${HPCPERF_NP:-$((PX * PY * PZ))}"

LAUNCH=(mpirun -np "$NP")
# mpirun inside this Slurm allocation needs --oversubscribe for >1 rank.
[ "$NP" -gt 1 ] && LAUNCH+=(--oversubscribe)

echo "# AMG2023 $BACKEND: ${LAUNCH[*]} $EXE -P $PX $PY $PZ -n $N $N $N -problem $PROBLEM $*"
exec "${LAUNCH[@]}" "$EXE" \
    -P "$PX" "$PY" "$PZ" \
    -n "$N" "$N" "$N" \
    -problem "$PROBLEM" \
    "$@"
