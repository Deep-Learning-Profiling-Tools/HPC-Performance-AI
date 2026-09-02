#!/usr/bin/env bash
# Build the P3-miniapps heat3d mini-app (CUDA or HIP programming model) out of tree.
#
#   ./build.sh [CUDA|HIP]        (default: CUDA)
#
# Build tree:  $R/build/level2/p3_heat3d/<cuda|hip>   (level2/p3_heat3d/ stays clean)
# Executable:  $R/build/level2/p3_heat3d/<cuda|hip>/miniapps/heat3d/thrust/heat3d
#
# Environment knobs:
#   HPCPERF_CUDA_ARCH   CUDA arch override, e.g. 100 or sm_100
#                       (default: detected from nvidia-smi compute_cap)
#   HPCPERF_HIP_ARCH    optional HIP architecture list, e.g. gfx90a
#   MAKE_JOBS           parallel build jobs (default 4)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

# Load the project toolchain (conda GCC 13.3 + system CUDA 13.2) unless the
# calling shell already sourced it. hpcperf_env.sh is idempotent. It must run
# before any local variables are set: conda's activate hooks are not
# `set -u`-clean and export generic names such as BUILD and HOST.
if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$R" ] && [ -f "$R/hpcperf_env.sh" ]; then
    set +eu
    # shellcheck disable=SC1091
    source "$R/hpcperf_env.sh" 2>/dev/null
    set -eu
fi

BACKEND="$(printf '%s' "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
case "$BACKEND" in
    CUDA|HIP) ;;
    *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac

JOBS="${MAKE_JOBS:-4}"
COMMON_ARGS=(
    -DAPPLICATION=heat3d
    -DBUILD_TESTING=OFF
    -DCMAKE_BUILD_TYPE=Release
)

if [ "$BACKEND" = "CUDA" ]; then
    BUILD_DIR="$R/build/level2/p3_heat3d/cuda"

    # GPU architecture: HPCPERF_CUDA_ARCH override, else detect from the GPU.
    ARCH="${HPCPERF_CUDA_ARCH:-}"
    if [ -z "$ARCH" ]; then
        ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                | head -1 | tr -d ' .')"
    fi
    ARCH="${ARCH#sm_}"
    if [ -z "$ARCH" ]; then
        echo "error: could not detect the GPU compute capability; set HPCPERF_CUDA_ARCH (e.g. 100)" >&2
        exit 1
    fi

    HOSTCXX="${CUDAHOSTCXX:-${CXX:-g++}}"

    echo "== P3-miniapps heat3d CUDA: sm_${ARCH}, host compiler ${HOSTCXX}, build dir ${BUILD_DIR}"
    cmake -S "$HERE" -B "$BUILD_DIR" "${COMMON_ARGS[@]}" \
        -DPROGRAMMING_MODEL=CUDA -DBACKEND=CUDA \
        -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
        -DCMAKE_CXX_COMPILER="$HOSTCXX" \
        -DCMAKE_CUDA_HOST_COMPILER="$HOSTCXX"
    cmake --build "$BUILD_DIR" -j"$JOBS"
    echo "== built: $BUILD_DIR/miniapps/heat3d/thrust/heat3d"
else
    BUILD_DIR="$R/build/level2/p3_heat3d/hip"

    if ! command -v hipcc >/dev/null 2>&1; then
        echo "error: hipcc not found on PATH; the HIP variant cannot be built on this machine (no ROCm)." >&2
        exit 1
    fi
    HIP_ARGS=()
    if [ -n "${HPCPERF_HIP_ARCH:-}" ]; then
        HIP_ARGS+=(-DCMAKE_HIP_ARCHITECTURES="$HPCPERF_HIP_ARCH")
    fi
    if [ -n "${ROCM_PATH:-}" ]; then
        HIP_ARGS+=(-DCMAKE_PREFIX_PATH="$ROCM_PATH")
    fi

    echo "== P3-miniapps heat3d HIP (untested here): build dir ${BUILD_DIR}"
    cmake -S "$HERE" -B "$BUILD_DIR" "${COMMON_ARGS[@]}" \
        -DPROGRAMMING_MODEL=HIP -DBACKEND=HIP \
        -DCMAKE_CXX_COMPILER=hipcc "${HIP_ARGS[@]}"
    cmake --build "$BUILD_DIR" -j"$JOBS"
    echo "== built: $BUILD_DIR/miniapps/heat3d/thrust/heat3d"
fi
