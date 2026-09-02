#!/usr/bin/env bash
# Validate miniWeather with upstream's own correctness check.
#
#   ./validate.sh [CUDA|HIP]          (default: CUDA)
#
# miniWeather has no reference solution; upstream checks conservation instead
# (README "Checking for Correctness", enforced by cpp/build/check_output.sh,
# which is what `make test` / ctest YAKL_TEST runs). The check runs the
# `parallelfor_test` executable -- upstream's fixed validation problem
# (NX=100, NZ=50, SIM_TIME=400 s, OUT_FREQ=400, DATA_SPEC_THERMAL, the exact
# configuration for which the energy criterion is stated) -- with
# TEST_MPI_COMMAND="mpirun -np 1" and parses the two summary lines it prints:
#
#   d_mass  relative change of the domain-integrated mass:
#           must not be NaN and |d_mass| < 1e-9   (upstream C++ tolerance;
#           mass is conserved to machine precision by the finite-volume scheme)
#   d_te    relative change of the domain-integrated total energy:
#           must not be NaN, must be NEGATIVE (hyper-viscosity only removes
#           energy) and |d_te| < 4.5e-5
#
# The test binary also writes output.nc (two frames) through PnetCDF, which
# exercises the parallel I/O path; it is written into the build directory.
# Prints a final PASS/FAIL line, exit 0/1. Full program output is kept in
# $R/build/level2/miniweather/<backend>/validate.log.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

BACKEND="$(printf '%s' "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
case "$BACKEND" in
    CUDA|HIP) ;;
    *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac
BUILD_DIR="$R/build/level2/miniweather/$(printf '%s' "$BACKEND" | tr '[:upper:]' '[:lower:]')"
EXE="$BUILD_DIR/parallelfor_test"
CHECK="$HERE/cpp/build/check_output.sh"

if [ ! -x "$EXE" ]; then
    echo "error: $EXE not found -- run $HERE/build.sh $BACKEND first" >&2
    echo "FAIL: miniweather $BACKEND (not built)"
    exit 1
fi

if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$R" ] && [ -f "$R/hpcperf_env.sh" ]; then
    # conda's activate.d scripts are not `set -u` safe; relax while sourcing.
    set +u
    # shellcheck disable=SC1091
    source "$R/hpcperf_env.sh" 2>/dev/null
    set -u
fi

cd "$BUILD_DIR"
export TEST_MPI_COMMAND="mpirun -np 1"
MASS_TOL=1e-9
TE_TOL=4.5e-5

# 1) Run the validation problem once with the output visible (check_output.sh
#    deletes its own capture on success).
echo "== $TEST_MPI_COMMAND ./parallelfor_test   (NX=100 NZ=50 SIM_TIME=400 THERMAL; log: $BUILD_DIR/validate.log)"
$TEST_MPI_COMMAND ./parallelfor_test > validate.log 2>&1
rc=$?
grep -E "^(nx_glob|dx,dz|dt:|CPU Time|d_mass|d_te)" validate.log || true
if [ "$rc" -ne 0 ]; then
    echo "   run failed (exit $rc) -- see $BUILD_DIR/validate.log"
    echo "FAIL: miniweather $BACKEND (parallelfor_test exited $rc)"
    exit 1
fi

# 2) Upstream's check: |d_mass| < 1e-9, d_te < 0 and |d_te| < 4.5e-5, no NaN
#    (same command and tolerances as the CMake test YAKL_TEST).
echo "== $CHECK ./parallelfor_test $MASS_TOL $TE_TOL"
"$CHECK" ./parallelfor_test "$MASS_TOL" "$TE_TOL"
crc=$?

dmass="$(grep d_mass validate.log | awk '{print $2}')"
dte="$(grep d_te validate.log | awk '{print $2}')"
if [ "$crc" -eq 0 ]; then
    echo "PASS: miniweather $BACKEND (d_mass=$dmass, |d_mass|<$MASS_TOL; d_te=$dte, negative and |d_te|<$TE_TOL; upstream check_output.sh criteria)"
    exit 0
else
    echo "FAIL: miniweather $BACKEND (check_output.sh exit $crc; d_mass=$dmass d_te=$dte; tolerances |d_mass|<$MASS_TOL, d_te<0, |d_te|<$TE_TOL)"
    exit 1
fi
