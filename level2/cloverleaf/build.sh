#!/usr/bin/env bash
# Build CloverLeaf (UoB-HPC C++ port, cuda or hip model).
#
#   ./build.sh [CUDA|HIP]        (default: CUDA)
#
# Build tree: $R/build/level2/cloverleaf/<cuda|hip>  (R = repository root).
# Environment: hpcperf_env.sh (conda GCC 13.3, system CUDA, conda OpenMPI).
# Override the detected GPU arch with HPCPERF_CUDA_ARCH=<cc without dot>
# (e.g. 90) for CUDA, or HPCPERF_HIP_ARCH=gfx90a for HIP.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null
set -euo pipefail

BACKEND="${1:-CUDA}"
BACKEND="${BACKEND^^}"

case "$BACKEND" in
  CUDA)
    ARCH="${HPCPERF_CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')}"
    if [ -z "$ARCH" ]; then
      echo "build.sh: could not detect GPU compute capability; set HPCPERF_CUDA_ARCH" >&2
      exit 1
    fi
    BUILD="$R/build/level2/cloverleaf/cuda"
    echo "== CloverLeaf CUDA build: sm_${ARCH} -> $BUILD"
    # -DCMAKE_CUDA_ARCHITECTURES=OFF: upstream passes -arch=${CUDA_ARCH} itself;
    # CUDAARCHS=native from hpcperf_env.sh would add a second, conflicting
    # -arch=native (nvcc warning "incompatible redefinition ... gpu-architecture").
    cmake -S "$HERE" -B "$BUILD" \
      -DMODEL=cuda \
      -DENABLE_MPI=ON \
      -DCMAKE_CXX_COMPILER="$CXX" \
      -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
      -DCUDA_ARCH="sm_${ARCH}" \
      -DCMAKE_CUDA_ARCHITECTURES=OFF \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build "$BUILD" -j4
    echo "== built: $BUILD/cuda-cloverleaf"
    ;;
  HIP)
    HIPCXX="${HIPCXX:-$(command -v hipcc || true)}"
    if [ -z "$HIPCXX" ]; then
      echo "build.sh: hipcc not found (no ROCm on this machine); HIP build untested" >&2
      exit 1
    fi
    BUILD="$R/build/level2/cloverleaf/hip"
    EXTRA=()
    if [ -n "${HPCPERF_HIP_ARCH:-}" ]; then
      EXTRA+=(-DCXX_EXTRA_FLAGS="--offload-arch=${HPCPERF_HIP_ARCH}")
    fi
    echo "== CloverLeaf HIP build: $HIPCXX -> $BUILD"
    cmake -S "$HERE" -B "$BUILD" \
      -DMODEL=hip \
      -DENABLE_MPI=ON \
      -DCMAKE_CXX_COMPILER="$HIPCXX" \
      -DCMAKE_BUILD_TYPE=Release \
      "${EXTRA[@]}"
    cmake --build "$BUILD" -j4
    echo "== built: $BUILD/hip-cloverleaf"
    ;;
  *)
    echo "usage: $0 [CUDA|HIP]" >&2
    exit 2
    ;;
esac
