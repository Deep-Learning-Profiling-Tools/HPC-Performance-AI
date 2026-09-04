#!/usr/bin/env bash
# Build Branson (LANL IMC proxy app) for the GPU: CUDA or HIP variant, out of tree.
#
#   ./build.sh [CUDA|HIP]        (default: CUDA)
#
# Configures upstream's CMake project (level2/branson/src) into
#   $R/build/level2/branson/<cuda|hip>
# with the project toolchain (conda GCC 13.3 + conda OpenMPI 5 + system CUDA 13.2).
# Branson is a header-only code base: main.cc is compiled as a CUDA (or HIP)
# source, pugixml is built as a static library, and the ctest unit tests
# (src/test, BUILD_TESTING=ON) are built too -- validate.sh runs them.
#
# Environment knobs:
#   HPCPERF_CUDA_ARCH   CUDA arch override, e.g. 100 or sm_100
#                       (default: detected from nvidia-smi compute_cap)
#   HPCPERF_HIP_ARCH    HIP architecture, e.g. gfx90a / gfx942 (default gfx942 = upstream's)
#   ROCM_PATH           ROCm install prefix for the HIP variant (default /opt/rocm)
#   MAKE_JOBS           parallel build jobs (default 4)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

BACKEND="$(printf '%s' "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
case "$BACKEND" in
    CUDA|HIP) ;;
    *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac

# Load the project toolchain unless the calling shell already sourced it.
# hpcperf_env.sh is idempotent.
if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$R" ] && [ -f "$R/hpcperf_env.sh" ]; then
    # conda's activate hooks are not `set -u`-clean: relax the shell options
    # around the source and restore them afterwards.
    set +eu
    # shellcheck disable=SC1091
    source "$R/hpcperf_env.sh" 2>/dev/null
    set -eu
fi

JOBS="${MAKE_JOBS:-4}"
SRC="$HERE/src"

# Options common to both variants. USE_UMPIRE / USE_CALIPER stay OFF (not
# needed, not available); USE_OPENMP is left at upstream's default (OFF: the
# transport phase runs on the GPU, the OpenMP threads only affect CPU
# transport). METIS is found in $CONDA_PREFIX via CMAKE_PREFIX_PATH (only used
# in domain-decomposed mode); HDF5 is found but Silo is not, so the optional
# Silo visualization output stays disabled.
COMMON_OPTS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER="${CC:-gcc}"
    -DCMAKE_CXX_COMPILER="${CXX:-g++}"
    -DCMAKE_PREFIX_PATH="${CONDA_PREFIX:-}"
    -DUSE_GPU=ON
    -DUSE_UMPIRE=OFF
    -DUSE_CALIPER=OFF
    -DBUILD_TESTING=ON
)

if [ "$BACKEND" = "CUDA" ]; then
    BUILD="$R/build/level2/branson/cuda"

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
    NVCC="$(command -v nvcc || true)"
    if [ -z "$NVCC" ]; then
        echo "error: nvcc not found on PATH" >&2
        exit 1
    fi
    # Upstream's CMake sets the CUDA architecture explicitly from CUDA_ARCH; do
    # not let the environment's CUDAARCHS=native compete with it.
    unset CUDAARCHS

    echo "== Branson CUDA: sm_${ARCH}, nvcc ${NVCC}, host compiler ${CXX:-g++}"
    echo "== build dir: $BUILD"
    cmake -S "$SRC" -B "$BUILD" "${COMMON_OPTS[@]}" \
        -DUSE_CUDA=ON \
        -DCUDA_ARCH="$ARCH" \
        -DCMAKE_CUDA_COMPILER="$NVCC"
    cmake --build "$BUILD" -j"$JOBS"
    echo "== built: $BUILD/BRANSON"
else
    BUILD="$R/build/level2/branson/hip"
    ROCM="${ROCM_PATH:-/opt/rocm}"
    HIP_ARCH="${HPCPERF_HIP_ARCH:-gfx942}"

    if [ ! -x "$ROCM/llvm/bin/clang++" ] && ! command -v hipcc >/dev/null 2>&1; then
        echo "error: no ROCm found (ROCM_PATH=$ROCM, no hipcc on PATH) -- HIP build is not possible on this machine" >&2
        exit 1
    fi
    export ROCM_PATH="$ROCM"

    echo "== Branson HIP: ${HIP_ARCH}, ROCM_PATH=${ROCM}, host compiler ${CXX:-g++}"
    echo "== build dir: $BUILD"
    # Upstream sets CMAKE_HIP_COMPILER to ${ROCM_PATH}/llvm/bin/clang++ itself.
    cmake -S "$SRC" -B "$BUILD" "${COMMON_OPTS[@]}" \
        -DUSE_HIP=ON \
        -DHIP_ARCH="$HIP_ARCH" \
        -DROCM_PATH="$ROCM"
    cmake --build "$BUILD" -j"$JOBS"
    echo "== built: $BUILD/BRANSON"
fi
