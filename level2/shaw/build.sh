#!/usr/bin/env bash
# Build SHAW (Pressio, Kokkos + Kokkos Kernels) for the CUDA or HIP programming model.
#
#   ./build.sh [CUDA|HIP] [extra cmake -D options...]
#
# Output tree: $R/build/level2/shaw/<cuda|hip>/shawExe
#
# SHAW is a Kokkos code: the same source tree is the CUDA variant when built
# against a CUDA-enabled Kokkos/Kokkos Kernels (this repo: $R/.deps/install/
# kokkos and kokkos-kernels, CUDA + OpenMP + Serial backends) and the HIP
# variant when built against a HIP-enabled Kokkos/Kokkos Kernels.
#
# Environment overrides:
#   HPCPERF_CUDA_ARCH    compute capability digits (e.g. 100 for sm_100);
#                        default: detected from nvidia-smi. Only used for a
#                        consistency check -- the architecture is baked into
#                        the Kokkos install the code is linked against.
#   HPCPERF_KOKKOS_ROOT          Kokkos install prefix (CUDA build)
#   HPCPERF_KOKKOSKERNELS_ROOT   Kokkos Kernels install prefix (CUDA build)
#   HPCPERF_KOKKOS_HIP_ROOT          Kokkos install prefix (HIP build)
#   HPCPERF_KOKKOSKERNELS_HIP_ROOT   Kokkos Kernels install prefix (HIP build)
#   HPCPERF_YAMLCPP_ROOT yaml-cpp install prefix (default: $CONDA_PREFIX)
#   HPCPERF_BUILD_JOBS   parallel build jobs (default 4)
#   HIPCXX               path to hipcc (default: hipcc from PATH)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

# Load the project toolchain (conda GCC, system CUDA, CMake/Ninja, yaml-cpp).
# Idempotent, so it is safe even if the caller already sourced it.
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true

set -euo pipefail

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
[ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level2/shaw/$MODEL"
JOBS="${HPCPERF_BUILD_JOBS:-4}"
YAMLCPP_ROOT="${HPCPERF_YAMLCPP_ROOT:-${CONDA_PREFIX:-}}"

if [ -z "$YAMLCPP_ROOT" ] || [ ! -f "$YAMLCPP_ROOT/lib/cmake/yaml-cpp/yaml-cpp-config.cmake" ]; then
    echo "build.sh: yaml-cpp not found under '$YAMLCPP_ROOT' (set HPCPERF_YAMLCPP_ROOT or source hpcperf_env.sh)" >&2
    exit 1
fi

# Upstream sets CMAKE_CXX_STANDARD 14; Kokkos 5 requires C++20. The Kokkos CMake
# config raises the standard itself (Kokkos_CXX_STANDARD=20 in KokkosConfigCommon),
# so upstream's CMakeLists.txt is left untouched. -DCMAKE_CXX_STANDARD=20 is passed
# explicitly anyway so the requirement is visible on the command line.
COMMON=(
    -S "$HERE" -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_STANDARD=20
    -DYAMLCPP_DIR="$YAMLCPP_ROOT"
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
        KOKKOS_ROOT="${HPCPERF_KOKKOS_ROOT:-$R/.deps/install/kokkos}"
        KK_ROOT="${HPCPERF_KOKKOSKERNELS_ROOT:-$R/.deps/install/kokkos-kernels}"
        KOKKOS_CFG="$(ls "$KOKKOS_ROOT"/lib*/cmake/Kokkos/KokkosConfig.cmake 2>/dev/null | head -1 || true)"
        if [ -z "$KOKKOS_CFG" ] || [ ! -d "$KK_ROOT" ]; then
            echo "build.sh: Kokkos / Kokkos Kernels not found under $KOKKOS_ROOT / $KK_ROOT" >&2
            echo "          (run $R/setup_level2_deps.sh, or set HPCPERF_KOKKOS_ROOT / HPCPERF_KOKKOSKERNELS_ROOT)" >&2
            exit 1
        fi
        # The GPU architecture is a property of the Kokkos install, not of this
        # build: check that the installed Kokkos was built for the detected GPU.
        KOKKOS_ARCH="$(sed -n 's/^set(Kokkos_ARCH \(.*\))$/\1/p' "$(dirname "$KOKKOS_CFG")"/KokkosConfigCommon.cmake 2>/dev/null | head -1 || true)"
        echo "build.sh: GPU compute capability ${ARCH}; Kokkos install $KOKKOS_ROOT (Kokkos_ARCH=${KOKKOS_ARCH:-unknown})"
        # nvcc_wrapper forwards to NVCC_WRAPPER_DEFAULT_COMPILER (= $CXX, conda g++,
        # exported by hpcperf_env.sh) for host code.
        export NVCC_WRAPPER_DEFAULT_COMPILER="${NVCC_WRAPPER_DEFAULT_COMPILER:-${CXX:-g++}}"
        cmake "${COMMON[@]}" \
            -DCMAKE_CXX_COMPILER="$KOKKOS_ROOT/bin/nvcc_wrapper" \
            -DKOKKOSKERNELS_DIR="$KK_ROOT" \
            -DCMAKE_PREFIX_PATH="$KK_ROOT;$KOKKOS_ROOT" \
            "$@"
        ;;
    HIP)
        # NOTE: no ROCm/hipcc and no HIP-enabled Kokkos on the development machine -- untested.
        HIPCXX="${HIPCXX:-$(command -v hipcc || echo hipcc)}"
        KOKKOS_ROOT="${HPCPERF_KOKKOS_HIP_ROOT:-$R/.deps/install/kokkos-hip}"
        KK_ROOT="${HPCPERF_KOKKOSKERNELS_HIP_ROOT:-$R/.deps/install/kokkos-kernels-hip}"
        if ! command -v "$HIPCXX" >/dev/null 2>&1; then
            echo "build.sh HIP: hipcc not found ($HIPCXX). SHAW is a Kokkos code: a HIP build needs ROCm" >&2
            echo "  plus Kokkos and Kokkos Kernels built with Kokkos_ENABLE_HIP=ON installed at" >&2
            echo "  $KOKKOS_ROOT and $KK_ROOT (override with HPCPERF_KOKKOS_HIP_ROOT /" >&2
            echo "  HPCPERF_KOKKOSKERNELS_HIP_ROOT). Untested on this machine (no ROCm)." >&2
            exit 1
        fi
        if [ ! -d "$KOKKOS_ROOT" ] || [ ! -d "$KK_ROOT" ]; then
            echo "build.sh HIP: HIP-enabled Kokkos / Kokkos Kernels not found under $KOKKOS_ROOT / $KK_ROOT" >&2
            exit 1
        fi
        cmake "${COMMON[@]}" \
            -DCMAKE_CXX_COMPILER="$HIPCXX" \
            -DKOKKOSKERNELS_DIR="$KK_ROOT" \
            -DCMAKE_PREFIX_PATH="$KK_ROOT;$KOKKOS_ROOT" \
            "$@"
        ;;
    *)
        echo "usage: $0 [CUDA|HIP] [extra cmake options]" >&2
        exit 2
        ;;
esac

cmake --build "$BUILD_DIR" -j"$JOBS"
echo "Built: $BUILD_DIR/shawExe"
