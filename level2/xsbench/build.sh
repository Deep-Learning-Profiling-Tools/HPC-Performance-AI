#!/usr/bin/env bash
# Build XSBench (CUDA or HIP variant) out of tree.
#
#   ./build.sh [CUDA|HIP]        (default: CUDA)
#
# The upstream Makefiles build in place, so the selected backend directory is
# mirrored (rsync) into  $R/build/level2/xsbench/<cuda|hip>  and `make` runs
# there; level2/xsbench/ itself stays clean.
#
# Environment knobs:
#   HPCPERF_CUDA_ARCH   CUDA arch override, e.g. 100 or sm_100
#                       (default: detected from nvidia-smi compute_cap)
#   HPCPERF_HIP_ARCH    optional HIP --offload-arch, e.g. gfx90a
#   MAKE_JOBS           parallel make jobs (default 4)
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

if [ "$BACKEND" = "CUDA" ]; then
    SRC_DIR="$HERE/cuda"
    BUILD_DIR="$R/build/level2/xsbench/cuda"

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

    mkdir -p "$BUILD_DIR"
    rsync -a --delete --exclude='*.o' --exclude='XSBench' "$SRC_DIR/" "$BUILD_DIR/"

    echo "== XSBench CUDA: sm_${ARCH}, host compiler ${HOSTCXX}"
    echo "== build dir: $BUILD_DIR"
    # SM_VERSION and CC are plain Makefile assignments, so command-line
    # overrides take effect without editing the upstream Makefile.
    make -C "$BUILD_DIR" -j"$JOBS" SM_VERSION="$ARCH" CC="nvcc -ccbin $HOSTCXX"
    echo "== built: $BUILD_DIR/XSBench"
else
    SRC_DIR="$HERE/hip"
    BUILD_DIR="$R/build/level2/xsbench/hip"

    if ! command -v hipcc >/dev/null 2>&1; then
        echo "error: hipcc not found on PATH -- HIP build is not possible on this machine (no ROCm)" >&2
        exit 1
    fi
    HIPCC="hipcc"
    if [ -n "${HPCPERF_HIP_ARCH:-}" ]; then
        HIPCC="hipcc --offload-arch=${HPCPERF_HIP_ARCH}"
    fi

    mkdir -p "$BUILD_DIR"
    rsync -a --delete --exclude='*.o' --exclude='XSBench' "$SRC_DIR/" "$BUILD_DIR/"

    echo "== XSBench HIP: compiler ${HIPCC}"
    echo "== build dir: $BUILD_DIR"
    # The HIP Makefile appends to CFLAGS (+=), so drop the conda toolchain's
    # environment CFLAGS/LDFLAGS to keep upstream's flags exactly.
    env -u CFLAGS -u LDFLAGS make -C "$BUILD_DIR" -j"$JOBS" CC="$HIPCC"
    echo "== built: $BUILD_DIR/XSBench"
fi
