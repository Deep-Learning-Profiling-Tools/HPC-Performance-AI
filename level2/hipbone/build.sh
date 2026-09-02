#!/usr/bin/env bash
# Build hipBone (OCCA-based Nekbone port) out of tree.
#
#   ./build.sh [CUDA|HIP]        (default: CUDA)
#
# hipBone has a single source tree: its kernels are OCCA kernel-language
# files (okl/, libs/*/okl/) that OCCA JIT-compiles at run time for whichever
# backend is selected with `hipBone -m CUDA|HIP|Serial|...`. The "CUDA
# variant" is therefore hipBone linked against an OCCA built with CUDA
# support, and the "HIP variant" is the same sources linked against an OCCA
# built with HIP support. This script builds the bundled OCCA (occa/, its own
# Makefile) with exactly one GPU backend enabled and then hipBone itself.
#
# The upstream makefiles build in place, so the whole level2/hipbone/ tree is
# mirrored (rsync) into  $R/build/level2/hipbone/<cuda|hip>  and `make` runs
# there; level2/hipbone/ itself stays clean.
#
# Environment knobs:
#   HPCPERF_CUDA_ARCH   informational only -- OCCA queries the device at run
#                       time and passes -arch=sm_<cc> to nvcc itself, so the
#                       build does not need (or bake in) a GPU architecture.
#   HIP_PATH / ROCM_PATH  ROCm location for the HIP build (default /opt/rocm)
#   OPENBLAS_DIR        OpenBLAS library dir (default: $CONDA_PREFIX/lib)
#   MAKE_JOBS           parallel make jobs (default 4)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

BACKEND="$(printf '%s' "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
case "$BACKEND" in
    CUDA|HIP) ;;
    *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac

# Load the project toolchain (conda GCC 13.3 + OpenMPI 5 + system CUDA 13.2)
# unless the calling shell already sourced it. hpcperf_env.sh is idempotent.
if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$R" ] && [ -f "$R/hpcperf_env.sh" ]; then
    # conda's activate hooks are not `set -u`-clean: relax the shell options
    # around the source and restore them afterwards.
    set +eu
    # shellcheck disable=SC1091
    source "$R/hpcperf_env.sh" 2>/dev/null
    set -eu
fi

JOBS="${MAKE_JOBS:-4}"
LC="$(printf '%s' "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD="$R/build/level2/hipbone/$LC"
# OCCA writes its JIT cache to $OCCA_CACHE_DIR (default ~/.occa) as soon as
# libocca is loaded -- `make` runs `bin/occa info` at the end of the OCCA
# build -- so point it into the build tree. hipBone itself later switches the
# cache to <exe dir>/.occa (or $HIPBONE_CACHE_DIR); run.sh/validate.sh set both.
export OCCA_CACHE_DIR="$R/build/level2/hipbone/occa-cache"

for tool in mpicxx rsync make; do
    command -v "$tool" >/dev/null 2>&1 || { echo "error: $tool not found on PATH" >&2; exit 1; }
done
OPENBLAS_DIR="${OPENBLAS_DIR:-${CONDA_PREFIX:-}/lib}"
if [ ! -e "$OPENBLAS_DIR/libopenblas.so" ] && [ ! -e "$OPENBLAS_DIR/libopenblas.a" ]; then
    echo "error: libopenblas not found in OPENBLAS_DIR=$OPENBLAS_DIR" >&2
    exit 1
fi

# --- OCCA backend selection --------------------------------------------------
# OCCA's Makefile auto-detects every backend whose headers/libraries it can
# find (here that would also switch on OpenCL via the CUDA toolkit's CL/cl.h).
# Enable exactly the requested GPU backend and disable the others explicitly;
# OpenMP (CPU) stays auto-detected as upstream intends. Header/library search
# paths go through OCCA's own OCCA_INCLUDE_PATH / OCCA_LIBRARY_PATH knobs.
export OCCA_OPENCL_ENABLED=0 OCCA_DPCPP_ENABLED=0 OCCA_METAL_ENABLED=0
if [ "$BACKEND" = "CUDA" ]; then
    command -v nvcc >/dev/null 2>&1 || { echo "error: nvcc not found on PATH" >&2; exit 1; }
    CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
    export OCCA_CUDA_ENABLED=1 OCCA_HIP_ENABLED=0
    export OCCA_INCLUDE_PATH="$CUDA_HOME/include"
    export OCCA_LIBRARY_PATH="$CUDA_HOME/lib64/stubs"   # libcuda.so link stub (driver API)
    ARCH="${HPCPERF_CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')}"
    echo "== hipBone CUDA: OCCA CUDA backend, CUDA_HOME=$CUDA_HOME (GPU sm_${ARCH:-?}; kernels are JIT-compiled at run time)"
else
    HIP_PATH="${HIP_PATH:-${ROCM_PATH:-/opt/rocm}}"
    if [ ! -x "$HIP_PATH/bin/hipconfig" ]; then
        echo "error: $HIP_PATH/bin/hipconfig not found -- HIP build is not possible on this machine (no ROCm)" >&2
        exit 1
    fi
    export HIP_PATH
    export OCCA_CUDA_ENABLED=0 OCCA_HIP_ENABLED=1
    export OCCA_INCLUDE_PATH="$HIP_PATH/include"
    export OCCA_LIBRARY_PATH="$HIP_PATH/lib"
    echo "== hipBone HIP: OCCA HIP backend, HIP_PATH=$HIP_PATH (kernels are JIT-compiled at run time)"
fi
echo "== build dir: $BUILD"

mkdir -p "$BUILD"
rsync -a --delete \
    --exclude='*.o' --exclude='*.a' --exclude='/hipBone' \
    --exclude='/occa/lib/' --exclude='/occa/obj/' --exclude='/occa/bin/occa' \
    --exclude='/occa/include/occa/defines/compiledDefines.hpp' --exclude='/occa/.compiledDefines' \
    --exclude='/occa/include/occa/core/codegen/' --exclude='/occa/src/core/codegen/' \
    --exclude='/occa/src/occa/internal/utils/codegen/' --exclude='/occa/include/occa/defines/codegen/' \
    --exclude='/.occa/' --exclude='*.log' \
    "$HERE/" "$BUILD/"
# OCCA's Makefile globs tests/src at parse time; the tests are not shipped
# here, so give it an empty directory to keep `find` quiet.
mkdir -p "$BUILD/occa/tests/src"

# --- 1. OCCA (occa/Makefile) --------------------------------------------------
# OCCA replaces its own release flags (-O3 -march=native) with $CXXFLAGS when
# that variable is set; the conda environment exports generic -O2 flags, so
# run make with the conda CXXFLAGS/CFLAGS/LDFLAGS/CPPFLAGS unset and let OCCA
# pick its upstream defaults for the project compiler ($CXX = conda GCC 13.3).
echo "== [1/2] OCCA (CXX=${CXX:-g++})"
env -u CXXFLAGS -u CFLAGS -u LDFLAGS -u CPPFLAGS \
    make -C "$BUILD/occa" -j"$JOBS" CXX="${CXX:-g++}" CC="${CC:-gcc}"

# --- 2. hipBone (makefile / make.top) ----------------------------------------
# make.top: HIPBONE_CXX = mpic++ (conda OpenMPI wrapper around $CXX),
# -O3 -march=native -std=c++17 -fopenmp; OPENBLAS_DIR is read from the env.
echo "== [2/2] hipBone (mpic++ -> $(mpicxx --showme:command 2>/dev/null || echo '?'), OPENBLAS_DIR=$OPENBLAS_DIR)"
OPENBLAS_DIR="$OPENBLAS_DIR" make -C "$BUILD" -j"$JOBS" hipBone

echo "== built: $BUILD/hipBone"
