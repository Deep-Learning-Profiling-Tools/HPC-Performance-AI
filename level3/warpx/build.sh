#!/usr/bin/env bash
# Build WarpX (3D, MPI, CUDA or HIP) with upstream's native CMake superbuild,
# using the AMReX 26.09 checkout WarpX pins (no configure-time downloads).
#
#   ./build.sh [CUDA|HIP]        (default CUDA)
#
# Layout (Level 3 isolation): sources level3/warpx/src (WarpX) and
# level3/warpx/deps/amrex (AMReX 26.09) -- the frozen source bundle materialized
# by tools/prepare_benchmark.sh (identity in provenance/); build
# build/level3/warpx/<cuda|hip>, install .deps/level3/warpx/install
# (+ .hpcperf-l3-fingerprint), logs .deps/level3/warpx/logs. AMReX is built
# by WarpX's superbuild from deps/amrex (-DWarpX_amrex_src) -- it is WarpX's
# private copy, nothing is shared with other Level 3 applications. No source
# is read from outside the benchmark directory; nothing is fetched or patched.
#
# Configuration (bring-up subset of the documented options): WarpX_COMPUTE=CUDA,
# CMAKE_CUDA_ARCHITECTURES=100 (sm_100), WarpX_DIMS=3, WarpX_MPI=ON,
# WarpX_OPENPMD=OFF (plotfile output only; no HDF5/ADIOS2), WarpX_QED=OFF
# (PICSAR-QED not needed by the uniform-plasma/laser benchmarks; avoids a
# download), WarpX_PYTHON=OFF, WarpX_FFT=OFF (no PSATD). Host compiler conda
# GCC 13.3.0 (upstream: GCC 12+, NVCC 12.4+; upstream's Perlmutter profile
# uses GCC 13 with NVCC 13.2.78). Modification class: A (build options only).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
l3_isolate_build_env    # Level 3 builds must not see Level 2 .deps/install prefixes

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
l3_require_materialized "$HERE" || exit 3
SRC="$HERE/src"; AMREX="$HERE/deps/amrex"
[ -f "$SRC/CMakeLists.txt" ] && [ -f "$AMREX/CMakeLists.txt" ] || { echo "build.sh: src/ or deps/amrex incomplete -- run tools/prepare_benchmark.sh level3 warpx" >&2; exit 3; }
SHA="$(l3_source_commit "$HERE")"; AMREX_SHA="$(l3_component_commit "$HERE" deps/amrex)"; TREE_SHA="$(l3_source_tree_sha "$HERE")"
l3_paths warpx
BUILD_DIR="$R/build/level3/warpx/$MODEL"
JOBS="${HPCPERF_BUILD_JOBS:-32}"

case "$BACKEND" in
    CUDA)
        ARCH="${HPCPERF_CUDA_ARCH:-$(l3_gpu_arch)}"
        GPU_FLAGS=(-DWarpX_COMPUTE=CUDA "-DCMAKE_CUDA_ARCHITECTURES=$ARCH" "-DCMAKE_CUDA_HOST_COMPILER=$CXX")
        ARCHNOTE="sm_$ARCH" ;;
    HIP)
        command -v hipcc >/dev/null 2>&1 || { echo "build.sh: HIP requested but hipcc not found -- HIP build is UNTESTED on this machine (no ROCm)" >&2; exit 1; }
        ARCH="${HPCPERF_HIP_ARCH:-gfx950}"
        GPU_FLAGS=(-DWarpX_COMPUTE=HIP "-DAMReX_AMD_ARCH=$ARCH" -DCMAKE_CXX_COMPILER=hipcc)
        ARCHNOTE="$ARCH" ;;
    *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac

CMAKE_OPTS="WarpX_COMPUTE=$BACKEND arch=$ARCHNOTE WarpX_DIMS=3 WarpX_MPI=ON WarpX_OPENPMD=OFF WarpX_QED=OFF WarpX_PYTHON=OFF WarpX_FFT=OFF WarpX_amrex_src=local BUILD_TESTING=OFF"
FP="$(l3_fingerprint_text warpx "$SHA" "$MODEL" "amrex=26.09($AMREX_SHA) picsar-qed=off openpmd=off" "$CMAKE_OPTS" "runtime(amrex.use_gpu_aware_mpi auto)")"
l3_fingerprint_check "$L3_INSTALL" "$FP" || exit 1

echo "# WarpX $BACKEND: upstream $SHA, AMReX $AMREX_SHA (26.09), frozen source tree $TREE_SHA, arch $ARCHNOTE, MPI $(mpirun --version 2>/dev/null | head -1)"
mkdir -p "$BUILD_DIR"
cmake -S "$SRC" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$L3_INSTALL" \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DWarpX_DIMS=3 -DWarpX_MPI=ON -DWarpX_OPENPMD=OFF -DWarpX_QED=OFF -DWarpX_PYTHON=OFF -DWarpX_FFT=OFF \
    -DWarpX_APP=ON -DWarpX_LIB=OFF -DWarpX_amrex_src="$AMREX" -DBUILD_TESTING=OFF \
    "${GPU_FLAGS[@]}" > "$L3_LOGS/configure-$MODEL.log" 2>&1 \
    || { tail -40 "$L3_LOGS/configure-$MODEL.log"; echo "build.sh: configure failed (log: $L3_LOGS/configure-$MODEL.log)" >&2; exit 1; }
t0=$(date +%s)
cmake --build "$BUILD_DIR" -j "$JOBS" > "$L3_LOGS/build-$MODEL.log" 2>&1 \
    || { tail -40 "$L3_LOGS/build-$MODEL.log"; echo "build.sh: build failed (log: $L3_LOGS/build-$MODEL.log)" >&2; exit 1; }
cmake --install "$BUILD_DIR" > "$L3_LOGS/install-$MODEL.log" 2>&1 || { echo "build.sh: install failed" >&2; exit 1; }
l3_fingerprint_write "$L3_INSTALL" "$FP"
EXE="$(find "$BUILD_DIR/bin" -maxdepth 1 -name 'warpx.3d*' -type f 2>/dev/null | head -1)"
echo "# built in $(( $(date +%s)-t0 )) s: ${EXE:-<warpx.3d* not found under $BUILD_DIR/bin>} (installed under $L3_INSTALL)"
echo "# compiler warning lines: $(grep -c 'warning' "$L3_LOGS/build-$MODEL.log" || true)"
