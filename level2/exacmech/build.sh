#!/usr/bin/env bash
# Build ExaCMech + its orientation_evolution miniapp (level2/exacmech) for CUDA or HIP.
#
#   ./build.sh [CUDA|HIP]        (default: CUDA)
#
# Build tree: $R/build/level2/exacmech/<cuda|hip>  (R = repository root, gitignored)
# Environment: source $R/hpcperf_env.sh first (conda GCC 13, CUDA 13.2, CMake 3.28, Ninja).
#
# ExaCMech is a BLT/CMake project.  On a GPU back end its top-level CMakeLists.txt
# builds the bundled SNLS with USE_BATCH_SOLVERS=ON, which requires the RAJA
# Portability Suite: RAJA + camp (from $R/.deps/install/raja, built by
# $R/setup_level2_deps.sh) and Umpire + CHAI (+ the fmt bundled with Umpire),
# expected in $R/.deps/install/umpire and $R/.deps/install/chai -- see README.md
# ("Dependencies") for the exact commands that produce them.
#
# Overrides:
#   HPCPERF_CUDA_ARCH   CUDA compute capability, e.g. 100 or sm_100 (default: detected via nvidia-smi)
#   HPCPERF_HIP_ARCH    AMD GPU target, e.g. gfx942 (default: gfx942)
#   HIPCXX              HIP compiler (default: hipcc)
#   HPCPERF_BUILD_JOBS  parallel build jobs (default: 4)
#   HPCPERF_RAJA_DIR    RAJA+camp install prefix   (default: $R/.deps/install/raja)
#   HPCPERF_UMPIRE_DIR  Umpire(+fmt) install prefix (default: $R/.deps/install/umpire)
#   HPCPERF_CHAI_DIR    CHAI install prefix        (default: $R/.deps/install/chai)
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SRC_DIR}/../.." && pwd)"
BACKEND="${1:-CUDA}"
BACKEND_UPPER="$(echo "${BACKEND}" | tr '[:lower:]' '[:upper:]')"
JOBS="${HPCPERF_BUILD_JOBS:-4}"

# Our environment exports CUDAARCHS=native; CMake would use it to seed
# CMAKE_CUDA_ARCHITECTURES.  We pass the architecture explicitly instead.
unset CUDAARCHS

DEPS="${REPO_ROOT}/.deps/install"
RAJA_ROOT="${HPCPERF_RAJA_DIR:-${DEPS}/raja}"
UMPIRE_ROOT="${HPCPERF_UMPIRE_DIR:-${DEPS}/umpire}"
CHAI_ROOT="${HPCPERF_CHAI_DIR:-${DEPS}/chai}"

# cmake_dir <install prefix> <package>  ->  <prefix>/lib{,64}/cmake/<package>
cmake_dir() {
  local d
  for d in "$1/lib/cmake/$2" "$1/lib64/cmake/$2" "$1/share/$2/cmake"; do
    if [[ -d "${d}" ]]; then echo "${d}"; return 0; fi
  done
  echo "build.sh: ${2} CMake package not found under ${1} (looked in lib/cmake, lib64/cmake, share)." >&2
  echo "          RAJA/camp come from ${REPO_ROOT}/setup_level2_deps.sh; Umpire and CHAI must be built" >&2
  echo "          into ${DEPS}/{umpire,chai} (see level2/exacmech/README.md, 'Dependencies')." >&2
  exit 1
}
RAJA_CMAKE="$(cmake_dir "${RAJA_ROOT}" raja)"
CAMP_CMAKE="$(cmake_dir "${RAJA_ROOT}" camp)"
UMPIRE_CMAKE="$(cmake_dir "${UMPIRE_ROOT}" umpire)"
FMT_CMAKE="$(cmake_dir "${UMPIRE_ROOT}" fmt)"
CHAI_CMAKE="$(cmake_dir "${CHAI_ROOT}" chai)"

# Options shared by both back ends.
#  - BLT_CXX_STD=c++20: upstream sets C++17, but RAJA/camp/Umpire/CHAI v2026.07 headers
#    need C++20 (concepts); BLT applies this after the project's own setting.
#  - USE_BATCH_SOLVERS / SNLS_USE_RAJA_PORT_SUITE: upstream derives these inside the SNLS
#    subdirectory, i.e. after the top-level third-party lookup has already run, so on a
#    fresh configure the top level never imports the chai/umpire/fmt targets that
#    src/ecmech and miniapp link against ("fmt::fmt ... target was not found").  Passing
#    them on the command line makes the first configure behave like upstream's second one.
#  - ENABLE_TESTS=OFF: the test/ trees and BLT's bundled googletest are not shipped here.
COMMON_OPTS=(
  -DCMAKE_BUILD_TYPE=Release
  -DBLT_CXX_STD=c++20
  -DENABLE_TESTS=OFF
  -DENABLE_MINIAPPS=ON
  -DENABLE_OPENMP=ON
  -DENABLE_PYTHON=OFF
  -DBUILD_SHARED_LIBS=OFF
  -DBUILD_STATIC_LIBS=ON
  -DUSE_BATCH_SOLVERS=ON
  -DSNLS_USE_RAJA_PORT_SUITE=ON
  -DRAJA_DIR="${RAJA_CMAKE}"
  -DCAMP_DIR="${CAMP_CMAKE}"
  -DUMPIRE_DIR="${UMPIRE_CMAKE}"
  -DFMT_DIR="${FMT_CMAKE}"
  -DCHAI_DIR="${CHAI_CMAKE}"
)

case "${BACKEND_UPPER}" in
  CUDA)
    BUILD_DIR="${REPO_ROOT}/build/level2/exacmech/cuda"
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
    echo "== ExaCMech CUDA build: arch sm_${ARCH}, nvcc ${NVCC}, host compiler ${CXX:-c++}"
    echo "== RAJA ${RAJA_CMAKE}"
    echo "== Umpire ${UMPIRE_CMAKE}  CHAI ${CHAI_CMAKE}"
    echo "== build dir: ${BUILD_DIR}"
    cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
      "${COMMON_OPTS[@]}" \
      -DCMAKE_CXX_COMPILER="${CXX:-c++}" \
      -DCMAKE_CUDA_COMPILER="${NVCC}" \
      -DCMAKE_CUDA_HOST_COMPILER="${CXX:-c++}" \
      -DCMAKE_CUDA_ARCHITECTURES="${ARCH}" \
      -DENABLE_CUDA=ON
    cmake --build "${BUILD_DIR}" -j "${JOBS}"
    echo "== built: ${BUILD_DIR}/bin/orientation_evolution"
    ;;
  HIP)
    # NOTE: no ROCm/hipcc on the development machine -- this path is untested.
    # It also needs HIP builds of RAJA/camp, Umpire and CHAI (point HPCPERF_RAJA_DIR,
    # HPCPERF_UMPIRE_DIR, HPCPERF_CHAI_DIR at them); the CUDA builds in .deps will not link.
    BUILD_DIR="${REPO_ROOT}/build/level2/exacmech/hip"
    HIPCXX_BIN="${HIPCXX:-hipcc}"
    if ! command -v "${HIPCXX_BIN}" >/dev/null 2>&1; then
      echo "build.sh: HIP compiler '${HIPCXX_BIN}' not found (no ROCm on this machine?); set HIPCXX to override" >&2
      exit 1
    fi
    HIP_ARCH="${HPCPERF_HIP_ARCH:-gfx942}"
    ROCM_ROOT="${ROCM_PATH:-$(dirname "$(dirname "$(command -v "${HIPCXX_BIN}")")")}"
    echo "== ExaCMech HIP build: compiler ${HIPCXX_BIN}, arch ${HIP_ARCH}, ROCm ${ROCM_ROOT}"
    echo "== build dir: ${BUILD_DIR}"
    cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
      "${COMMON_OPTS[@]}" \
      -DCMAKE_CXX_COMPILER="${HIPCXX_BIN}" \
      -DCMAKE_HIP_COMPILER="${HIPCXX_BIN}" \
      -DROCM_PATH="${ROCM_ROOT}" \
      -DCMAKE_HIP_ARCHITECTURES="${HIP_ARCH}" \
      -DGPU_TARGETS="${HIP_ARCH}" \
      -DAMD_GPU_TARGETS="${HIP_ARCH}" \
      -DENABLE_HIP=ON
    cmake --build "${BUILD_DIR}" -j "${JOBS}"
    echo "== built: ${BUILD_DIR}/bin/orientation_evolution"
    ;;
  *)
    echo "usage: $0 [CUDA|HIP]" >&2
    exit 2
    ;;
esac
