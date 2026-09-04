#!/usr/bin/env bash
# Run the standard miniWeather benchmark problem (YAKL parallel_for variant).
#
#   ./run.sh [CUDA|HIP] [mpirun options]      (default backend: CUDA)
#
# Standard problem (fixed at build time by build.sh, see there): rising
# thermal (DATA_SPEC_THERMAL) on upstream's GPU grid of 2048 x 1024 cells,
# 1000 s of model time (~30,700 time steps of dt=0.0326 s), double precision,
# no NetCDF output (OUT_FREQ=-1), one MPI rank on one GPU:
#
#   mpirun -np 1 ./parallelfor
#
# miniWeather takes no command-line arguments. Extra arguments, if given,
# REPLACE the default "-np 1" mpirun options (e.g. via level2/tools/hpcperf_mpi_launch.sh).
# The binary runs from its build directory so that output.nc (if OUT_FREQ >= 0
# was configured) lands there. Prints upstream's normal output: device name,
# grid/dt, "CPU Time: <s> sec" (wall time of the time-step loop) and the
# relative mass / total-energy changes d_mass / d_te.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

BACKEND="CUDA"
case "$(printf '%s' "${1:-}" | tr '[:lower:]' '[:upper:]')" in
    CUDA) BACKEND="CUDA"; shift ;;
    HIP)  BACKEND="HIP";  shift ;;
esac
BUILD_DIR="$R/build/level2/miniweather/$(printf '%s' "$BACKEND" | tr '[:upper:]' '[:lower:]')"
EXE="$BUILD_DIR/parallelfor"

if [ ! -x "$EXE" ]; then
    echo "error: $EXE not found -- run $HERE/build.sh $BACKEND first" >&2
    exit 1
fi

if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$R" ] && [ -f "$R/hpcperf_env.sh" ]; then
    # conda's activate.d scripts are not `set -u`/`set -e` safe; relax while sourcing.
    set +eu
    # shellcheck disable=SC1091
    source "$R/hpcperf_env.sh" 2>/dev/null
    set -eu
fi

if [ "$#" -eq 0 ]; then
    set -- -np 1
fi

cd "$BUILD_DIR"
echo "== mpirun $* ./parallelfor   (build dir: $BUILD_DIR)"
exec mpirun "$@" ./parallelfor
