#!/usr/bin/env bash
# Build Kripke (level2/kripke) for the CUDA or HIP programming model.
#
#   ./build.sh [CUDA|HIP]        (default: CUDA)
#
# Build tree: $R/build/level2/kripke/<cuda|hip>  (R = repository root, gitignored)
# Environment: source $R/hpcperf_env.sh first (conda GCC 13, CUDA 13.2, CMake 3.28,
#              Ninja, conda OpenMPI 5).
#
# Kripke is a BLT/CMake project.  Upstream's top-level CMakeLists.txt is a thin
# "superbuild" (ExternalProject) around cmake/kripke/CMakeLists.txt that only
# exists to optionally build Caliper/Adiak.  We configure the inner project
# directly (exactly what the superbuild does, minus Caliper), which gives one
# build tree, honours -j, and puts kripke.exe at the top of the build dir.
# The options mirror upstream's host-configs/llnl-toss4-H100-cuda12-gcc-vector.cmake
# (CHAI+Umpire memory management, RAJA CUDA back end, MPI on, OpenMP off).
#
# Overrides:
#   HPCPERF_CUDA_ARCH   CUDA compute capability, e.g. 100 or sm_100 (default: detected via nvidia-smi)
#   HPCPERF_HIP_ARCH    AMD GPU target, e.g. gfx942 (default: gfx942, as in upstream's MI300A host-config)
#   HIPCXX              HIP compiler (default: hipcc)
#   HPCPERF_BUILD_JOBS  parallel build jobs (default: 4)
#   HPCPERF_CUB_DIR     directory containing cub/cub.cuh (default: $CUDA_HOME/include/cccl, CUDA >= 12.x layout)
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SRC_DIR}/../.." && pwd)"
BACKEND="${1:-CUDA}"
BACKEND_UPPER="$(echo "${BACKEND}" | tr '[:lower:]' '[:upper:]')"
JOBS="${HPCPERF_BUILD_JOBS:-4}"

# Our environment exports CUDAARCHS=native; CMake would use it to seed
# CMAKE_CUDA_ARCHITECTURES.  We pass the architecture explicitly instead.
unset CUDAARCHS

# Options shared by both back ends (upstream defaults: C++17, no tests/examples/docs,
# CHAI/Umpire on, MPI via CMake FindMPI -> conda OpenMPI, OpenMP off).
COMMON_OPTS=(
  -DKRIPKE_SOURCE_ROOT="${SRC_DIR}"
  -DCMAKE_BUILD_TYPE=Release
  -DBLT_CXX_STD=c++17
  -DENABLE_CHAI=On
  -DENABLE_MPI=On
  -DENABLE_OPENMP=Off
  -DENABLE_TESTS=Off
  -DENABLE_GTEST=Off
  -DENABLE_EXAMPLES=Off
  -DENABLE_BENCHMARKS=Off
  -DENABLE_DOCS=Off
  -DCMAKE_CXX_FLAGS_RELEASE="-O3 -ffast-math"
)

case "${BACKEND_UPPER}" in
  CUDA)
    BUILD_DIR="${REPO_ROOT}/build/level2/kripke/cuda"
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
    # RAJA (bundled, v2025.03.2) requires the CUB that ships with the CUDA toolkit
    # (RAJA_ENABLE_EXTERNAL_CUB=VersionDependent -> ON for CUDA >= 11).  Its
    # FindCUB.cmake only looks in ${CUB_DIR} and ${CUDA_TOOLKIT_ROOT_DIR}/include;
    # since CUDA 12.x the headers live under include/cccl, so point CUB_DIR there.
    CUDA_ROOT="${CUDA_HOME:-$(dirname "$(dirname "${NVCC}")")}"
    CUB_DIR="${HPCPERF_CUB_DIR:-${CUDA_ROOT}/include/cccl}"
    if [[ ! -f "${CUB_DIR}/cub/cub.cuh" ]]; then
      if [[ -f "${CUDA_ROOT}/include/cub/cub.cuh" ]]; then
        CUB_DIR="${CUDA_ROOT}/include"
      else
        echo "build.sh: cub/cub.cuh not found under ${CUB_DIR} or ${CUDA_ROOT}/include; set HPCPERF_CUB_DIR" >&2
        exit 1
      fi
    fi
    echo "== Kripke CUDA build: arch sm_${ARCH}, nvcc ${NVCC}, host compiler ${CXX:-c++}, CUB ${CUB_DIR}"
    echo "== build dir: ${BUILD_DIR}"
    cmake -S "${SRC_DIR}/cmake/kripke" -B "${BUILD_DIR}" \
      "${COMMON_OPTS[@]}" \
      -DCMAKE_C_COMPILER="${CC:-cc}" \
      -DCMAKE_CXX_COMPILER="${CXX:-c++}" \
      -DCMAKE_CUDA_COMPILER="${NVCC}" \
      -DCMAKE_CUDA_HOST_COMPILER="${CXX:-c++}" \
      -DCMAKE_CUDA_ARCHITECTURES="${ARCH}" \
      -DCUB_DIR="${CUB_DIR}" \
      -DENABLE_CUDA=On \
      -DCMAKE_CUDA_FLAGS="-restrict --expt-relaxed-constexpr" \
      -DCMAKE_CUDA_FLAGS_RELEASE="-O3 --expt-extended-lambda --expt-relaxed-constexpr"
    cmake --build "${BUILD_DIR}" -j "${JOBS}"
    echo "== built: ${BUILD_DIR}/kripke.exe"
    ;;
  HIP)
    # NOTE: no ROCm/hipcc on the development machine -- this path is untested.
    # Mirrors upstream's host-configs/llnl-toss4-MI300A-rocm6-adams.cmake.
    BUILD_DIR="${REPO_ROOT}/build/level2/kripke/hip"
    HIPCXX_BIN="${HIPCXX:-hipcc}"
    if ! command -v "${HIPCXX_BIN}" >/dev/null 2>&1; then
      echo "build.sh: HIP compiler '${HIPCXX_BIN}' not found (no ROCm on this machine?); set HIPCXX to override" >&2
      exit 1
    fi
    HIP_ARCH="${HPCPERF_HIP_ARCH:-gfx942}"
    ROCM_ROOT="${ROCM_PATH:-$(dirname "$(dirname "$(command -v "${HIPCXX_BIN}")")")}"
    echo "== Kripke HIP build: compiler ${HIPCXX_BIN}, arch ${HIP_ARCH}, ROCm ${ROCM_ROOT}"
    echo "== build dir: ${BUILD_DIR}"
    cmake -S "${SRC_DIR}/cmake/kripke" -B "${BUILD_DIR}" \
      "${COMMON_OPTS[@]}" \
      -DCMAKE_C_COMPILER="${CC:-cc}" \
      -DCMAKE_CXX_COMPILER="${HIPCXX_BIN}" \
      -DCMAKE_HIP_COMPILER="${HIPCXX_BIN}" \
      -DROCM_PATH="${ROCM_ROOT}" \
      -DCMAKE_HIP_ARCHITECTURES="${HIP_ARCH}" \
      -DGPU_TARGETS="${HIP_ARCH}" \
      -DAMD_GPU_TARGETS="${HIP_ARCH}" \
      -DENABLE_HIP=On
    cmake --build "${BUILD_DIR}" -j "${JOBS}"
    echo "== built: ${BUILD_DIR}/kripke.exe"
    ;;
  *)
    echo "usage: $0 [CUDA|HIP]" >&2
    exit 2
    ;;
esac
