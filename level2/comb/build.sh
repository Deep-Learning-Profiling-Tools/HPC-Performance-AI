#!/usr/bin/env bash
# Build Comb's native CUDA or HIP execution policy.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null
set -euo pipefail
# tools/timing ROI markers (header-only; a no-op unless measured): tools/timing/roi/README.md
export CPATH="$R/tools/timing/roi${CPATH:+:$CPATH}"

BACKEND="${1:-CUDA}"
BACKEND="${BACKEND^^}"

cuda_arch() {
    local arch="${HPCPERF_CUDA_ARCH:-${CUDA_ARCH:-}}"
    if [ -z "$arch" ] && command -v nvidia-smi >/dev/null 2>&1; then
        arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')"
    fi
    arch="${arch#sm_}"; arch="${arch//./}"
    case "$arch" in ''|*[!0-9]*) echo "build.sh: cannot determine CUDA architecture; set HPCPERF_CUDA_ARCH (for example 90)" >&2; return 1;; esac
    echo "$arch"
}

hip_arch() {
    local arch="${HPCPERF_HIP_ARCH:-${HIP_ARCH:-}}"
    [ -n "$arch" ] || arch="$(rocminfo 2>/dev/null | sed -n 's/^[[:space:]]*Name:[[:space:]]*\(gfx[0-9a-fA-F]*\).*/\1/p' | head -1)"
    case "$arch" in gfx[0-9a-fA-F]*) echo "$arch";; *) echo "build.sh: cannot determine AMD GPU architecture; set HPCPERF_HIP_ARCH (for example gfx942)" >&2; return 1;; esac
}

case "$BACKEND" in
  CUDA)
    NVCC="${NVCC:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}"
    [ -x "$NVCC" ] || NVCC="$(command -v nvcc || true)"
    [ -n "$NVCC" ] || { echo "build.sh: nvcc not found; source hpcperf_env.sh or set CUDA_HOME" >&2; exit 1; }
    CUDA_TOOLKIT="$(cd "$(dirname "$NVCC")/.." && pwd)"
    command -v mpicxx >/dev/null 2>&1 || { echo "build.sh: MPI C++ compiler not found (mpicxx)" >&2; exit 1; }
    ARCH="$(cuda_arch)"
    BUILD="$R/build/level2/comb/cuda"
    echo "== Comb CUDA build: sm_${ARCH} -> $BUILD"
    cmake -S "$HERE/upstream" -B "$BUILD" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS_RELEASE=-O3 -DCMAKE_CUDA_FLAGS_RELEASE=-O3 \
      -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
      -DCMAKE_CUDA_COMPILER="$NVCC" -DCMAKE_CUDA_HOST_COMPILER="$CXX" -DCMAKE_CUDA_ARCHITECTURES=OFF \
      -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_TOOLKIT" -DCUDAToolkit_ROOT="$CUDA_TOOLKIT" \
      -DMPI_C_COMPILER="$(command -v mpicc)" -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
      -DENABLE_MPI=ON -DENABLE_CUDA=ON -DENABLE_HIP=OFF -DENABLE_OPENMP=OFF \
      -DCOMB_ENABLE_RAJA=OFF -DCOMB_ENABLE_NV_TOOLS_EXT=OFF \
      -DCOMB_ENABLE_CALIPER=OFF -DCOMB_ENABLE_ADIAK=OFF -DCUDA_ARCH="sm_${ARCH}"
    cmake --build "$BUILD" -j"${HPCPERF_BUILD_JOBS:-4}"
    ;;
  HIP)
    HIPCXX="${HIPCXX:-$(command -v hipcc || true)}"
    [ -n "$HIPCXX" ] || { echo "build.sh: hipcc not found; install/source ROCm" >&2; exit 1; }
    command -v mpicxx >/dev/null 2>&1 || { echo "build.sh: MPI C++ compiler not found (mpicxx); use a ROCm-compatible MPI" >&2; exit 1; }
    ARCH="$(hip_arch)"
    BUILD="$R/build/level2/comb/hip"
    ROCM_ROOT="${ROCM_PATH:-${ROCM_ROOT:-$(cd "$(dirname "$HIPCXX")/.." && pwd)}}"
    echo "== Comb HIP build: $ARCH -> $BUILD"
    cmake -S "$HERE/upstream" -B "$BUILD" \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE=-O3 -DCMAKE_HIP_FLAGS_RELEASE=-O3 \
      -DCMAKE_C_COMPILER="${CC:-cc}" -DCMAKE_CXX_COMPILER="$HIPCXX" -DCMAKE_HIP_COMPILER="$HIPCXX" \
      -DMPI_C_COMPILER="$(command -v mpicc)" -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
      -DROCM_PATH="$ROCM_ROOT" -DCMAKE_HIP_ARCHITECTURES="$ARCH" -DAMDGPU_TARGETS="$ARCH" \
      -DENABLE_MPI=ON -DENABLE_CUDA=OFF -DENABLE_HIP=ON -DENABLE_OPENMP=OFF \
      -DCOMB_ENABLE_RAJA=OFF -DCOMB_ENABLE_ROCTX=OFF \
      -DCOMB_ENABLE_CALIPER=OFF -DCOMB_ENABLE_ADIAK=OFF
    cmake --build "$BUILD" -j"${HPCPERF_BUILD_JOBS:-4}"
    ;;
  *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2;;
esac

echo "== built: $BUILD/bin/comb"
