#!/usr/bin/env bash
# Build miniBUDE (level2/minibude) for the CUDA or HIP programming model.
#
#   ./build.sh [CUDA|HIP]        (default: CUDA)
#
# Build tree: $R/build/level2/minibude/<cuda|hip>  (R = repository root, gitignored)
# Environment: source $R/hpcperf_env.sh first (conda GCC 13, CUDA 13.2, CMake 3.28, Ninja).
#
# Overrides:
#   HPCPERF_CUDA_ARCH   CUDA compute capability, e.g. 100 or sm_100 (default: detected via nvidia-smi)
#   HPCPERF_HIP_ARCH    AMD GPU target, e.g. gfx90a (passed as --offload-arch; default: hipcc's default)
#   HIPCXX              HIP compiler (default: hipcc)
#   HPCPERF_BUILD_JOBS  parallel build jobs (default: 4)
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SRC_DIR}/../.." && pwd)"
BACKEND="${1:-CUDA}"
BACKEND_UPPER="$(echo "${BACKEND}" | tr '[:lower:]' '[:upper:]')"
JOBS="${HPCPERF_BUILD_JOBS:-4}"

case "${BACKEND_UPPER}" in
  CUDA)
    BUILD_DIR="${REPO_ROOT}/build/level2/minibude/cuda"
    if [[ -n "${HPCPERF_CUDA_ARCH:-}" ]]; then
      ARCH="${HPCPERF_CUDA_ARCH#sm_}"
    else
      ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
      if [[ -z "${ARCH}" ]]; then
        echo "build.sh: could not detect the GPU compute capability via nvidia-smi; set HPCPERF_CUDA_ARCH (e.g. 100)" >&2
        exit 1
      fi
    fi
    NVCC="$(command -v nvcc || true)"
    if [[ -z "${NVCC}" ]]; then
      echo "build.sh: nvcc not found on PATH (source hpcperf_env.sh first)" >&2
      exit 1
    fi
    echo "== miniBUDE CUDA build: arch sm_${ARCH}, nvcc ${NVCC}, host compiler ${CXX:-c++}"
    echo "== build dir: ${BUILD_DIR}"
    cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
      -DMODEL=cuda \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER="${CXX:-c++}" \
      -DCMAKE_CUDA_COMPILER="${NVCC}" \
      -DCUDA_ARCH="sm_${ARCH}"
    cmake --build "${BUILD_DIR}" -j "${JOBS}"
    echo "== built: ${BUILD_DIR}/cuda-bude"
    ;;
  HIP)
    # NOTE: no ROCm/hipcc on the development machine -- this path is untested.
    BUILD_DIR="${REPO_ROOT}/build/level2/minibude/hip"
    HIPCXX_BIN="${HIPCXX:-hipcc}"
    if ! command -v "${HIPCXX_BIN}" >/dev/null 2>&1; then
      echo "build.sh: HIP compiler '${HIPCXX_BIN}' not found (no ROCm on this machine?); set HIPCXX to override" >&2
      exit 1
    fi
    EXTRA=()
    if [[ -n "${HPCPERF_HIP_ARCH:-}" ]]; then
      EXTRA+=("-DCXX_EXTRA_FLAGS=--offload-arch=${HPCPERF_HIP_ARCH}")
    fi
    echo "== miniBUDE HIP build: compiler ${HIPCXX_BIN} ${HPCPERF_HIP_ARCH:+(--offload-arch=${HPCPERF_HIP_ARCH})}"
    echo "== build dir: ${BUILD_DIR}"
    cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
      -DMODEL=hip \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER="${HIPCXX_BIN}" \
      ${EXTRA[@]+"${EXTRA[@]}"}
    cmake --build "${BUILD_DIR}" -j "${JOBS}"
    echo "== built: ${BUILD_DIR}/hip-bude"
    ;;
  *)
    echo "usage: $0 [CUDA|HIP]" >&2
    exit 2
    ;;
esac
