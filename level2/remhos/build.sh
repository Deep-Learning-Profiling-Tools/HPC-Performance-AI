#!/usr/bin/env bash
# Build Remhos (CEED) against the repository's MFEM for the CUDA or HIP
# programming model.
#
#   ./build.sh [CUDA|HIP] [extra make variables, e.g. REMHOS_DEBUG=YES]
#
# Output tree: $R/build/level2/remhos/<cuda|hip>/remhos
#
# Build route: the upstream GNU makefile (the build Remhos' README and CI use).
# It takes the compiler, flags and link line from MFEM's installed config.mk
# (share/mfem/config.mk), so Remhos' own device kernels (mfem::forall in
# remhos_lo.cpp etc.) are compiled with the same nvcc/hipcc flags and GPU
# architecture as MFEM itself. The upstream CMakeLists.txt is NOT used: it is
# a developer-local file (hard-coded ccache launcher, ../mfem include path,
# /usr/include/hypre, -lfmt, no find_package(MFEM), no CUDA) -- see README.md.
#
# The makefile compiles in-source (objects next to the .cpp files, executable
# in the current directory), so the sources and the makefile are copied into
# the build tree first; level2/remhos/ itself stays untouched. `cp -p` keeps
# the time stamps, so an unchanged source is not recompiled on the next run.
#
# Environment overrides:
#   HPCPERF_MFEM_PREFIX  MFEM install prefix (default $R/.deps/install/mfem);
#                        must contain share/mfem/config.mk (GNU-make config)
#   HPCPERF_CUDA_ARCH    compute capability digits (e.g. 100 for sm_100);
#                        default: detected from nvidia-smi. Informational only:
#                        the architecture is fixed by MFEM's config.mk.
#   HPCPERF_BUILD_JOBS   parallel build jobs (default 4)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

# Load the project toolchain (conda GCC, system CUDA, OpenMPI). Idempotent.
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true

set -euo pipefail

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
[ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level2/remhos/$MODEL"
JOBS="${HPCPERF_BUILD_JOBS:-4}"
MFEM_PREFIX="${HPCPERF_MFEM_PREFIX:-$R/.deps/install/mfem}"
CONFIG_MK="$MFEM_PREFIX/share/mfem/config.mk"
TEST_MK="$MFEM_PREFIX/share/mfem/test.mk"

if [ ! -f "$CONFIG_MK" ]; then
    echo "build.sh: MFEM GNU-make config not found: $CONFIG_MK" >&2
    echo "          (build MFEM with $R/setup_level2_deps.sh, or set HPCPERF_MFEM_PREFIX)" >&2
    exit 1
fi
if ! grep -Eq '^MFEM_USE_MPI[[:space:]]*=[[:space:]]*YES' "$CONFIG_MK"; then
    echo "build.sh: $MFEM_PREFIX was built without MPI (MFEM_USE_MPI != YES); Remhos needs a parallel MFEM." >&2
    exit 1
fi

case "$BACKEND" in
    CUDA)
        if ! grep -Eq '^MFEM_USE_CUDA[[:space:]]*=[[:space:]]*YES' "$CONFIG_MK"; then
            echo "build.sh: $MFEM_PREFIX was not built with CUDA (MFEM_USE_CUDA != YES)." >&2
            echo "          Remhos runs on the GPU only through MFEM's device backend." >&2
            exit 1
        fi
        ARCH="${HPCPERF_CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')}"
        MARCH="$(grep -o 'arch=compute_[0-9a-z]*' "$CONFIG_MK" | head -1 | sed 's/arch=compute_//')"
        [ -n "$ARCH" ]  && echo "# GPU compute capability detected: sm_${ARCH}"
        [ -n "$MARCH" ] && echo "# MFEM config.mk compiles for:     compute_${MARCH} (Remhos device code uses the same flags)"
        if [ -n "$ARCH" ] && [ -n "$MARCH" ] && [ "$ARCH" != "$MARCH" ]; then
            echo "# WARNING: GPU is sm_${ARCH} but MFEM (and therefore Remhos) is compiled for compute_${MARCH};" >&2
            echo "#          the architecture is fixed by the MFEM build -- rebuild MFEM to change it." >&2
        fi
        ;;
    HIP)
        # NOTE: no ROCm/hipcc on the development machine -- this branch is
        # untested. It needs an MFEM built with MFEM_USE_HIP=YES (hipcc as
        # MFEM_CXX in config.mk), which this repo does not provide: point
        # HPCPERF_MFEM_PREFIX at one. The make invocation is otherwise
        # identical -- Remhos has no CUDA- or HIP-specific source.
        if ! command -v hipcc >/dev/null 2>&1; then
            echo "build.sh: hipcc not found -- no ROCm on this machine; HIP build is untested." >&2
            exit 1
        fi
        if ! grep -Eq '^MFEM_USE_HIP[[:space:]]*=[[:space:]]*YES' "$CONFIG_MK"; then
            echo "build.sh: $MFEM_PREFIX was not built with HIP (MFEM_USE_HIP != YES)." >&2
            echo "          Set HPCPERF_MFEM_PREFIX to an MFEM configured with MFEM_USE_HIP=YES." >&2
            exit 1
        fi
        ;;
    *)
        echo "usage: $0 [CUDA|HIP] [extra make variables]" >&2
        exit 2
        ;;
esac

mkdir -p "$BUILD_DIR"
# Stage the (byte-identical) upstream sources and makefile in the build tree.
for f in "$HERE"/remhos*.cpp "$HERE"/remhos*.hpp "$HERE"/makefile; do
    b="$(basename "$f")"
    if [ ! -f "$BUILD_DIR/$b" ] || ! cmp -s "$f" "$BUILD_DIR/$b"; then
        cp -p "$f" "$BUILD_DIR/$b"
    fi
done

echo "# make -j$JOBS remhos MFEM_DIR=$MFEM_PREFIX CONFIG_MK=$CONFIG_MK TEST_MK=$TEST_MK $*"
( cd "$BUILD_DIR" && \
  make -j"$JOBS" remhos \
       MFEM_DIR="$MFEM_PREFIX" CONFIG_MK="$CONFIG_MK" TEST_MK="$TEST_MK" "$@" \
  2>&1 | tee "$BUILD_DIR/build.log" )

echo "Built: $BUILD_DIR/remhos"
