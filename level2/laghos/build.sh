#!/usr/bin/env bash
# Build Laghos (CEED) against this repository's MFEM for the CUDA or HIP
# programming model, using the unmodified upstream CMakeLists.txt.
#
#   ./build.sh [CUDA|HIP] [extra cmake -D options...]
#
# Output tree: $R/build/level2/laghos/<cuda|hip>/laghos   (+ the small `sedov`
# exact-solution tool that upstream's CMakeLists builds alongside it)
#
# Laghos has no .cu files of its own: every kernel is written with MFEM's
# mfem::forall / MFEM_FORALL macros, and upstream's CMakeLists.txt compiles the
# .cpp sources as CUDA (or HIP) whenever the MFEM it finds was built with
# MFEM_USE_CUDA (MFEM_USE_HIP). The GPU backend is then chosen at run time
# with `-d cuda` (or `-d gpu`, which MFEM maps to the compiled backend).
#
# Environment overrides:
#   HPCPERF_MFEM_PREFIX   MFEM install prefix (default $R/.deps/install/mfem)
#   HPCPERF_HYPRE_PREFIX  hypre install prefix (default $R/.deps/install/hypre)
#   HPCPERF_CUDA_ARCH     compute capability digits (e.g. 100 for sm_100);
#                         default: detected from nvidia-smi
#   HPCPERF_HIP_ARCH      HIP architecture for the HIP build (default gfx942,
#                         upstream's default)
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
BUILD_DIR="$R/build/level2/laghos/$MODEL"
JOBS="${HPCPERF_BUILD_JOBS:-4}"
MFEM_PREFIX="${HPCPERF_MFEM_PREFIX:-$R/.deps/install/mfem}"
HYPRE_PREFIX="${HPCPERF_HYPRE_PREFIX:-$R/.deps/install/hypre}"

MFEM_CONFIG="$(ls "$MFEM_PREFIX"/lib*/cmake/mfem/MFEMConfig.cmake 2>/dev/null | head -1 || true)"
if [ -z "$MFEM_CONFIG" ]; then
    echo "build.sh: MFEM CMake package not found under $MFEM_PREFIX/lib*/cmake/mfem" >&2
    echo "          (build it with $R/setup_level2_deps.sh, or set HPCPERF_MFEM_PREFIX)" >&2
    exit 1
fi
if ! ls "$HYPRE_PREFIX"/lib*/cmake/HYPRE/HYPREConfig.cmake >/dev/null 2>&1; then
    echo "build.sh: hypre CMake package not found under $HYPRE_PREFIX/lib*/cmake/HYPRE" >&2
    echo "          (build it with $R/setup_level2_deps.sh, or set HPCPERF_HYPRE_PREFIX)" >&2
    exit 1
fi

mfem_flag() { # $1 = MFEM_USE_* name; echoes ON/OFF as recorded in MFEMConfig.cmake
    sed -n "s/^set($1 \(.*\))$/\1/p" "$MFEM_CONFIG" | head -1
}

# <Pkg>_ROOT (CMP0074, NEW since CMake 3.12) makes find_package(MFEM) and
# find_package(HYPRE) look in these prefixes first; upstream's CMakeLists then
# derives the extra include dir "${MFEM_DIR}/../../../include/mfem" from the
# resolved <prefix>/lib*/cmake/mfem, which is correct for this layout.
COMMON=(
    -S "$HERE" -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_COMPILER="$CXX"
    -DMFEM_ROOT="$MFEM_PREFIX"
    -DHYPRE_ROOT="$HYPRE_PREFIX"
)

case "$BACKEND" in
    CUDA)
        if [ "$(mfem_flag MFEM_USE_CUDA)" != "ON" ]; then
            echo "build.sh: MFEM at $MFEM_PREFIX was not built with MFEM_USE_CUDA=ON." >&2
            echo "          Laghos runs on the GPU only through MFEM's CUDA backend." >&2
            exit 1
        fi
        ARCH="${HPCPERF_CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')}"
        if [ -z "$ARCH" ]; then
            echo "build.sh: could not detect the GPU compute capability (no nvidia-smi?);" >&2
            echo "          set HPCPERF_CUDA_ARCH (e.g. 100 for sm_100)." >&2
            exit 1
        fi
        echo "# GPU compute capability: sm_${ARCH}"
        if command -v cuobjdump >/dev/null 2>&1; then
            MARCH="$(cuobjdump -lelf "$MFEM_PREFIX"/lib*/libmfem.a 2>/dev/null |
                     grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')"
            [ -n "$MARCH" ] && echo "# MFEM libmfem.a contains device code for: ${MARCH}"
        fi
        # Upstream's CMakeLists defaults CMAKE_CUDA_ARCHITECTURES to 70 before
        # enable_language(CUDA) runs, so it has to be passed explicitly (the
        # CUDAARCHS=native environment default would otherwise be ignored).
        cmake "${COMMON[@]}" \
            -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
            -DCMAKE_CUDA_HOST_COMPILER="$CXX" \
            "$@"
        ;;
    HIP)
        # NOTE: no ROCm/hipcc on the development machine -- this branch is
        # untested. It also needs an MFEM (and hypre) built with HIP, which
        # this repo does not provide (point HPCPERF_MFEM_PREFIX /
        # HPCPERF_HYPRE_PREFIX at them).
        if ! command -v hipcc >/dev/null 2>&1; then
            echo "build.sh: hipcc not found -- no ROCm on this machine; HIP build is untested." >&2
            exit 1
        fi
        if [ "$(mfem_flag MFEM_USE_HIP)" != "ON" ]; then
            echo "build.sh: MFEM at $MFEM_PREFIX was not built with MFEM_USE_HIP=ON." >&2
            echo "          Set HPCPERF_MFEM_PREFIX to an MFEM configured with -DMFEM_USE_HIP=ON." >&2
            exit 1
        fi
        cmake "${COMMON[@]}" \
            -DCMAKE_HIP_ARCHITECTURES="${HPCPERF_HIP_ARCH:-gfx942}" \
            -DCMAKE_HIP_COMPILER="$(command -v hipcc)" \
            -DCMAKE_PREFIX_PATH="${ROCM_PATH:-/opt/rocm}" \
            "$@"
        ;;
    *)
        echo "usage: $0 [CUDA|HIP] [extra cmake options]" >&2
        exit 2
        ;;
esac

# A full build takes about one minute at -j4 on the B200 machine (the longest
# translation unit, laghos.cpp, needs ~20 s). NOTE: this relies on the
# MFEM_UNROLL(1) lines added to laghos_solver.cpp (see README, "Changes from
# upstream"): with the unmodified file nvcc's device front end ran 15 min and
# emitted 107 MB of PTX for sm_100, after which ptxas exceeded the 64 GB
# memory limit of the Slurm job and was killed (signal 9).
cmake --build "$BUILD_DIR" -j"$JOBS"
echo "Built: $BUILD_DIR/laghos"
