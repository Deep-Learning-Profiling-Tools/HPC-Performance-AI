#!/usr/bin/env bash
# Build PENNANT (LANL unstructured-mesh hydro mini-app, single-GPU CUDA branch) with its upstream GNU Makefile.
#
#   ./build.sh [CUDA]            (default and only backend: CUDA; the HIP port lives on another upstream
#                                 branch that is not vendored here -> exit 1)
#
# The upstream Makefile builds in place (build/ under the source directory), so the vendored tree
# (Makefile + src/) is mirrored with rsync into $R/build/level2/pennant/cuda and `make` runs there;
# level2/pennant/ itself stays clean. Every compiler-dependent setting of the upstream Makefile is a
# plain `:=` assignment, so build.sh overrides them on the make command line and no Makefile line is
# edited: the g++ flag set instead of the icpc one, `nvcc -ccbin $CXX` (conda GCC 13.3 as the host
# compiler), the detected GPU architecture instead of the hard-coded `-arch=sm_21` (unsupported since
# CUDA 9) and the CUDA install path for -lcudart.
#
# Environment knobs:
#   HPCPERF_CUDA_ARCH   CUDA arch override, e.g. 100 or sm_100 (default: nvidia-smi compute_cap)
#   MAKE_JOBS           parallel make jobs (default 8)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

# Load the project toolchain (conda GCC 13.3 + system CUDA 13.2) unless the calling shell already did;
# before any local variable is set (conda's activate hooks are not `set -u`-clean).
if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$R" ] && [ -f "$R/hpcperf_env.sh" ]; then
    set +eu
    # shellcheck disable=SC1091
    source "$R/hpcperf_env.sh" 2>/dev/null
    set -eu
fi
# tools/timing ROI markers (header-only; a no-op unless measured): tools/timing/roi/README.md.
# src/Driver.cc (host code, compiled by $CXX which honours CPATH) includes hpcperf_roi.h; the upstream
# .cu rule clears CPATH for nvcc on purpose and HydroGPU.cu does not need the header.
export CPATH="$R/tools/timing/roi${CPATH:+:$CPATH}"

BACKEND="$(printf '%s' "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
case "$BACKEND" in
    CUDA) ;;
    HIP)  echo "build.sh: PENNANT's HIP port is upstream branch 'hip', not vendored here (only the 'cuda' branch is); HIP build unavailable" >&2; exit 1 ;;
    *)    echo "usage: $0 [CUDA]" >&2; exit 2 ;;
esac

JOBS="${MAKE_JOBS:-8}"
BUILD_DIR="$R/build/level2/pennant/cuda"

ARCH="${HPCPERF_CUDA_ARCH:-}"
if [ -z "$ARCH" ]; then
    ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
fi
ARCH="${ARCH#sm_}"
if [ -z "$ARCH" ]; then
    echo "error: could not detect the GPU compute capability; set HPCPERF_CUDA_ARCH (e.g. 100)" >&2
    exit 1
fi
HOSTCXX="${CUDAHOSTCXX:-${CXX:-g++}}"
NVCC="$(command -v nvcc || true)"
[ -n "$NVCC" ] || { echo "error: nvcc not on PATH (source hpcperf_env.sh)" >&2; exit 1; }
CUDA_ROOT="$(cd "$(dirname "$NVCC")/.." && pwd)"

mkdir -p "$BUILD_DIR"
rsync -a --delete --exclude='build/' --exclude='*.o' "$HERE/Makefile" "$HERE/src" "$BUILD_DIR/"

echo "== PENNANT CUDA: sm_${ARCH}, host compiler ${HOSTCXX}, nvcc ${NVCC}"
echo "== build dir: $BUILD_DIR"
# Command-line assignments beat the Makefile's `:=` and `+=` lines (GNU make): the upstream file is unchanged.
#   CXX / CXXFLAGS_OPT   the Makefile's g++ set (-O3) instead of the icpc set (-O3 -fast -fno-alias)
#   CUDAC / CUDACFLAGS   nvcc with the same host compiler; upstream's --ptxas-options=-v -O3 -I. kept, arch replaced
#   LDFLAGS              upstream's -L<cuda>/lib64 -lcudart with the real toolkit path
make -C "$BUILD_DIR" -j"$JOBS" \
    CXX="$HOSTCXX" CXXFLAGS_OPT="-O3" \
    CUDAC="$NVCC -ccbin $HOSTCXX" CUDACFLAGS="-arch=sm_${ARCH} --ptxas-options=-v -O3 -I." \
    CUDA_INSTALL_PATH="$CUDA_ROOT" LDFLAGS="-L$CUDA_ROOT/lib64 -lcudart"
echo "== built: $BUILD_DIR/build/pennant"
