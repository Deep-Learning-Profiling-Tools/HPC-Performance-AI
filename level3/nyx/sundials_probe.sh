#!/usr/bin/env bash
# Independent SUNDIALS 7.2.1 + CUDA 13.2 check BEFORE it is wired into Nyx's
# heating/cooling build: build the commit Nyx pins (subprojects/sundials,
# 5c53be85 = v7.2.1) with ENABLE_CUDA and its CUDA examples/tests enabled, then
# run SUNDIALS' own CUDA test set (ctest -R cuda: each example compares its
# output with the shipped .out answer file through SUNDIALS' testRunner).
#
#   ./sundials_probe.sh          HPCPERF_BUILD_JOBS=N (default 16)
# Build tree: .deps/level3/nyx/cuda132-gcc133-heatcool/build/sundials-test; log logs/sundials-ctest.log
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
l3_isolate_build_env
SRC="$R/_upstream/level3/Nyx/subprojects/sundials"; [ -f "$SRC/CMakeLists.txt" ] || { echo "sundials_probe.sh: run fetch.sh" >&2; exit 1; }
ARCH="${HPCPERF_CUDA_ARCH:-$(l3_gpu_arch)}"; GCC_MM="$(l3_version_mm "$("$CXX" -dumpfullversion)")"
l3_paths_profile nyx "cuda$(l3_version_mm "$(l3_cuda_version)")-gcc${GCC_MM}-heatcool"
B="$L3_BUILD_DEPS/sundials-test"; JOBS="${HPCPERF_BUILD_JOBS:-16}"
echo "# SUNDIALS $(git -C "$SRC" rev-parse HEAD) (v$(/usr/bin/grep -oE 'PACKAGE_VERSION_(MAJOR|MINOR|PATCH) "[0-9]+"' "$SRC/CMakeLists.txt" | /usr/bin/grep -oE '[0-9]+' | paste -sd.)) CUDA sm_$ARCH test build -> $B"
mkdir -p "$B"
cmake -S "$SRC" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_CXX_STANDARD=17 \
    -DENABLE_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=$ARCH" "-DCMAKE_CUDA_HOST_COMPILER=$CXX" -DSUNDIALS_INDEX_SIZE=32 -DSUNDIALS_BUILD_PACKAGE_FUSED_KERNELS=ON \
    -DENABLE_MPI=OFF -DENABLE_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON \
    -DBUILD_ARKODE=ON -DBUILD_CVODE=ON -DBUILD_KINSOL=OFF -DBUILD_IDA=OFF -DBUILD_IDAS=OFF -DBUILD_CVODES=OFF \
    -DEXAMPLES_ENABLE_C=ON -DEXAMPLES_ENABLE_CXX=ON -DEXAMPLES_ENABLE_CUDA=ON -DEXAMPLES_INSTALL=OFF -DBUILD_TESTING=ON \
    -DSUNDIALS_TEST_ENABLE_DEV_TESTS=ON -DSUNDIALS_TEST_ENABLE_DIFF_OUTPUT=ON \
    > "$L3_LOGS/sundials-test-configure.log" 2>&1 || { tail -30 "$L3_LOGS/sundials-test-configure.log"; echo "sundials_probe.sh: configure failed" >&2; exit 1; }
cmake --build "$B" -j "$JOBS" > "$L3_LOGS/sundials-test-build.log" 2>&1 || { tail -30 "$L3_LOGS/sundials-test-build.log"; echo "sundials_probe.sh: build failed" >&2; exit 1; }
# SUNDIALS_TEST_ENABLE_DEV_TESTS registers every example as a test whose output is compared
# with the shipped answer file (examples/*/*.out) -- this is SUNDIALS' own regression check.
( cd "$B" && ctest -N | /usr/bin/grep -iE 'cuda|cusolver' ) > "$L3_LOGS/sundials-ctest-list.log" 2>&1 || true
( cd "$B" && CUDA_VISIBLE_DEVICES=0 ctest -R 'cuda|cusolver' --timeout 600 --output-on-failure -j 1 ) > "$L3_LOGS/sundials-ctest.log" 2>&1 || true
/usr/bin/grep -E 'tests passed|Failed|Timeout|Not Run' "$L3_LOGS/sundials-ctest.log" | head -12
if /usr/bin/grep -qE '100% tests passed, 0 tests failed out of [1-9]' "$L3_LOGS/sundials-ctest.log"; then
    echo "sundials_probe.sh: PASS -- $(/usr/bin/grep -oE 'out of [0-9]+' "$L3_LOGS/sundials-ctest.log" | head -1) SUNDIALS CUDA examples/tests on sm_$ARCH with CUDA $(l3_cuda_version)"
    echo "PASS $(date -u +%FT%TZ) $(/usr/bin/grep -E 'tests passed' "$L3_LOGS/sundials-ctest.log")" > "$B/.hpcperf-probe-pass"; exit 0
fi
echo "sundials_probe.sh: FAIL -- see $L3_LOGS/sundials-ctest.log"; exit 1
