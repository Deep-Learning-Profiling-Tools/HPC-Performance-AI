#!/usr/bin/env bash
# Validate the SHAW CUDA or HIP build with upstream's own regression test.
#
#   ./validate.sh [CUDA|HIP]
#
# Runs the upstream ctest `fomInnerDomain` (tests/fomInnerDomain) in the build
# tree: shawExe on the 21x51 mesh (tests/fullMesh21x51), dt = 1 s, 150 steps,
# sinusoidal source at 1100 km depth (period 40 s, delay 10 s), single-layer
# material (rho 2000, vs 5000), ASCII snapshots of the velocity and stress
# fields at every step plus a seismogram at 4 receivers. Upstream's test.cmake
# then runs `python compare.py` (numpy.allclose) on the full snapshot matrices
# against the gold files shipped with the test:
#     snaps_vp_0 (1071 velocity dofs x 150 steps) vs snaps_vp_0_gold, atol 1e-13
#     snaps_sp_0 (2070 stress dofs   x 150 steps) vs snaps_sp_0_gold, atol 1e-10
# (the seismogram is written but not compared upstream). The gold files are
# stored xz-compressed in the repo and decompressed by CMake at configure time.
# This script forwards ctest's verdict as a final PASS/FAIL line and exit code,
# and additionally prints the observed max |difference| of each field.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true

set -uo pipefail

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level2/shaw/$MODEL"
TEST_DIR="$BUILD_DIR/tests/fomInnerDomain"

if [ ! -x "$BUILD_DIR/shawExe" ] || [ ! -f "$BUILD_DIR/CTestTestfile.cmake" ]; then
    echo "validate.sh: $BUILD_DIR/shawExe (or its ctest files) not found -- run ./build.sh $BACKEND first" >&2
    echo "SHAW $BACKEND validation (fomInnerDomain): FAIL"
    exit 1
fi

LOG="$BUILD_DIR/validate.log"
echo "# SHAW $BACKEND: ctest --test-dir $BUILD_DIR -R fomInnerDomain --output-on-failure"
ctest --test-dir "$BUILD_DIR" -R fomInnerDomain --output-on-failure 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

echo "----------------------------------------------------------------"
# Informative summary of the comparison ctest just performed (same files, same
# tolerances as tests/fomInnerDomain/test.cmake).
if [ -f "$TEST_DIR/snaps_vp_0" ] && [ -f "$TEST_DIR/snaps_sp_0" ]; then
    (cd "$TEST_DIR" && python3 - <<'PY' || true
import numpy as np
for f, tol in (("snaps_vp_0", 1e-13), ("snaps_sp_0", 1e-10)):
    try:
        a = np.loadtxt(f, skiprows=1); b = np.loadtxt(f + "_gold", skiprows=1)
        ok = a.shape == b.shape and np.allclose(a, b, atol=tol)
        print("%-10s shape %-12s max|gold| %.3e  max|diff| %.3e  atol %g  %s"
              % (f, str(a.shape), np.abs(b).max(), np.abs(a - b).max(), tol, "ok" if ok else "MISMATCH"))
    except Exception as e:
        print("%-10s could not compare: %s" % (f, e))
PY
    )
fi

if [ "$rc" -eq 0 ] && grep -q '100% tests passed' "$LOG"; then
    echo "SHAW $BACKEND validation (fomInnerDomain): PASS"
    exit 0
else
    echo "SHAW $BACKEND validation (fomInnerDomain): FAIL (ctest exit code $rc, see $LOG)"
    exit 1
fi
