#!/usr/bin/env bash
# Build TeaLeaf (UoB-HPC C++ version) for the CUDA or HIP programming model.
#
#   ./build.sh [CUDA|HIP] [extra cmake -D options...]
#
# Output tree: $R/build/level2/tealeaf/<cuda|hip>/<cuda|hip>-tealeaf
#
# Environment overrides:
#   HPCPERF_CUDA_ARCH    compute capability digits (e.g. 100 for sm_100);
#                        default: detected from nvidia-smi
#   HPCPERF_HIP_ARCH     AMD gfx target (e.g. gfx90a) passed as --offload-arch
#   HPCPERF_TEALEAF_MPI  ON (default) or OFF -- build with/without MPI
#   HPCPERF_BUILD_JOBS   parallel build jobs (default 4)
#   HIPCXX               path to hipcc (default: hipcc from PATH)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

# Load the project toolchain (conda GCC, system CUDA, CMake/Ninja, OpenMPI).
# Idempotent, so it is safe even if the caller already sourced it.
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true

set -euo pipefail

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
[ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level2/tealeaf/$MODEL"
MPI="${HPCPERF_TEALEAF_MPI:-ON}"
JOBS="${HPCPERF_BUILD_JOBS:-4}"

COMMON=(
    -S "$HERE" -B "$BUILD_DIR"
    -DMODEL="$MODEL"
    -DENABLE_MPI="$MPI"
    -DCMAKE_BUILD_TYPE=Release
)

case "$BACKEND" in
    CUDA)
        if [ -n "${HPCPERF_CUDA_ARCH:-}" ]; then
            ARCH="$HPCPERF_CUDA_ARCH"
        else
            ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
        fi
        if [ -z "$ARCH" ]; then
            echo "build.sh: could not detect GPU compute capability; set HPCPERF_CUDA_ARCH (e.g. 100)" >&2
            exit 1
        fi
        # Upstream passes the architecture itself via -arch=${CUDA_ARCH} (and sets
        # policy CMP0104 OLD). Our environment exports CUDAARCHS=native, which
        # would make CMake append a second, conflicting -arch=native; disable
        # CMake's own architecture handling so only upstream's flag is used.
        cmake "${COMMON[@]}" \
            -DCMAKE_C_COMPILER="$CC" \
            -DCMAKE_CXX_COMPILER="$CXX" \
            -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
            -DCUDA_ARCH="sm_${ARCH}" \
            -DCMAKE_CUDA_ARCHITECTURES=OFF \
            "$@"
        ;;
    HIP)
        # NOTE: no ROCm/hipcc on the development machine -- untested.
        HIPCXX="${HIPCXX:-$(command -v hipcc || echo hipcc)}"
        HIP_EXTRA=()
        if [ -n "${HPCPERF_HIP_ARCH:-}" ]; then
            HIP_EXTRA+=(-DCXX_EXTRA_FLAGS="--offload-arch=${HPCPERF_HIP_ARCH}")
        fi
        cmake "${COMMON[@]}" \
            -DCMAKE_CXX_COMPILER="$HIPCXX" \
            "${HIP_EXTRA[@]}" \
            "$@"
        ;;
    *)
        echo "usage: $0 [CUDA|HIP] [extra cmake options]" >&2
        exit 2
        ;;
esac

cmake --build "$BUILD_DIR" -j"$JOBS"
echo "Built: $BUILD_DIR/${MODEL}-tealeaf"
