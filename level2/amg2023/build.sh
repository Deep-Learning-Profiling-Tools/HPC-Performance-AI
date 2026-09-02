#!/usr/bin/env bash
# Build AMG2023 (LLNL) against hypre for the CUDA or HIP programming model.
#
#   ./build.sh [CUDA|HIP] [extra cmake -D options...]
#
# Output tree: $R/build/level2/amg2023/<cuda|hip>/amg
#
# AMG2023 itself contains NO device code -- it is a single C driver (amg.c)
# that calls hypre's IJ/BoomerAMG/Krylov interfaces. All GPU kernels live
# inside libHYPRE, so this build compiles no .cu/.hip source and there is no
# GPU architecture flag to pass: the target architecture is whatever hypre was
# compiled for (this repo's hypre is sm_100). The detected compute capability
# is only reported here, and compared against hypre's, as a sanity check.
#
# Environment overrides:
#   HPCPERF_HYPRE_PREFIX  hypre install prefix (default $R/.deps/install/hypre)
#   HPCPERF_CUDA_ARCH     compute capability digits (e.g. 100 for sm_100);
#                         default: detected from nvidia-smi. Informational only.
#   HPCPERF_BUILD_JOBS    parallel build jobs (default 4)
#   ROCM_PATH             ROCm root for the HIP build (default /opt/rocm)

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
BUILD_DIR="$R/build/level2/amg2023/$MODEL"
JOBS="${HPCPERF_BUILD_JOBS:-4}"
HYPRE_PREFIX="${HPCPERF_HYPRE_PREFIX:-$R/.deps/install/hypre}"

if [ ! -f "$HYPRE_PREFIX/include/HYPRE_config.h" ]; then
    echo "build.sh: hypre not found at $HYPRE_PREFIX" >&2
    echo "          (build it with $R/setup_level2_deps.sh, or set HPCPERF_HYPRE_PREFIX)" >&2
    exit 1
fi

COMMON=(
    -S "$HERE" -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER="$CC"
    -DCMAKE_CXX_COMPILER="$CXX"
    -DAMG_WITH_MPI=ON
    -DHYPRE_PREFIX="$HYPRE_PREFIX"
)

case "$BACKEND" in
    CUDA)
        if ! grep -q '^#define HYPRE_USING_CUDA' "$HYPRE_PREFIX/include/HYPRE_config.h"; then
            echo "build.sh: $HYPRE_PREFIX was not built with CUDA (HYPRE_USING_CUDA undefined)." >&2
            echo "          AMG2023 runs on the GPU only through hypre." >&2
            exit 1
        fi
        ARCH="${HPCPERF_CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')}"
        if [ -n "$ARCH" ]; then
            echo "# GPU compute capability detected: sm_${ARCH}"
            if command -v cuobjdump >/dev/null 2>&1; then
                HARCH="$(cuobjdump -lelf "$HYPRE_PREFIX"/lib*/libHYPRE.a 2>/dev/null |
                         grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')"
                [ -n "$HARCH" ] && echo "# hypre libHYPRE.a contains code for: ${HARCH}"
            fi
        fi
        cmake "${COMMON[@]}" -DAMG_WITH_CUDA=ON -DAMG_WITH_HIP=OFF "$@"
        ;;
    HIP)
        # NOTE: no ROCm/hipcc on the development machine -- this branch is
        # untested. It also needs a hypre built with --with-hip, which this
        # repo does not provide (point HPCPERF_HYPRE_PREFIX at one).
        if ! command -v hipcc >/dev/null 2>&1; then
            echo "build.sh: hipcc not found -- no ROCm on this machine; HIP build is untested." >&2
            exit 1
        fi
        if ! grep -q '^#define HYPRE_USING_HIP' "$HYPRE_PREFIX/include/HYPRE_config.h"; then
            echo "build.sh: $HYPRE_PREFIX was not built with HIP (HYPRE_USING_HIP undefined)." >&2
            echo "          Set HPCPERF_HYPRE_PREFIX to a hypre configured with --with-hip." >&2
            exit 1
        fi
        cmake "${COMMON[@]}" \
            -DAMG_WITH_CUDA=OFF -DAMG_WITH_HIP=ON \
            -DCMAKE_PREFIX_PATH="${ROCM_PATH:-/opt/rocm}" \
            "$@"
        ;;
    *)
        echo "usage: $0 [CUDA|HIP] [extra cmake options]" >&2
        exit 2
        ;;
esac

cmake --build "$BUILD_DIR" -j"$JOBS"
echo "Built: $BUILD_DIR/amg"
