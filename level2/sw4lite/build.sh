#!/usr/bin/env bash
# Build geodynamics/SW4lite CUDA or the ECP-linked AMD HIP port.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null
set -euo pipefail
# tools/timing ROI markers (header-only; a no-op unless measured): tools/timing/roi/README.md
export CPATH="$R/tools/timing/roi${CPATH:+:$CPATH}"

BACKEND="${1:-CUDA}"; BACKEND="${BACKEND^^}"
MPICXX="${MPICXX:-$(command -v mpicxx || true)}"
[ -n "$MPICXX" ] || { echo "build.sh: MPI C++ compiler not found (mpicxx)" >&2; exit 1; }

case "$BACKEND" in
  CUDA)
    NVCC="${NVCC:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}"; [ -x "$NVCC" ] || NVCC="$(command -v nvcc || true)"; [ -n "$NVCC" ] || { echo "build.sh: nvcc not found" >&2; exit 1; }
    ARCH="${HPCPERF_CUDA_ARCH:-${CUDA_ARCH:-}}"
    [ -n "$ARCH" ] || ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
    ARCH="${ARCH#sm_}"; ARCH="${ARCH//./}"
    case "$ARCH" in ''|*[!0-9]*) echo "build.sh: set HPCPERF_CUDA_ARCH (for example 90)" >&2; exit 1;; esac
    FC="${HPCPERF_FC:-$(command -v gfortran || true)}"; [ -n "$FC" ] || { echo "build.sh: Fortran compiler not found (set HPCPERF_FC)" >&2; exit 1; }
    BUILD="$R/build/level2/sw4lite/cuda"
    echo "== SW4lite CUDA build: sm_${ARCH} -> $BUILD"
    cmake -S "$HERE/source/cuda" -B "$BUILD" \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
      -DCMAKE_CUDA_COMPILER="$NVCC" -DCMAKE_CUDA_HOST_COMPILER="$CXX" \
      -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_Fortran_COMPILER="$FC" \
      -DMPI_CXX_COMPILER="$MPICXX" -DCMAKE_PREFIX_PATH="${CONDA_PREFIX:-}"
    cmake --build "$BUILD" -j"${HPCPERF_BUILD_JOBS:-4}"
    EXE="$BUILD/sw4lite"
    ;;
  HIP)
    HIPCXX="${HIPCXX:-$(command -v hipcc || true)}"; [ -n "$HIPCXX" ] || { echo "build.sh: hipcc not found; install/source ROCm" >&2; exit 1; }
    command -v hipify-perl >/dev/null 2>&1 || { echo "build.sh: hipify-perl not found; the authoritative SW4lite HIP build requires it" >&2; exit 1; }
    ARCH="${HPCPERF_HIP_ARCH:-${HIP_ARCH:-}}"
    [ -n "$ARCH" ] || ARCH="$(rocminfo 2>/dev/null | sed -n 's/^[[:space:]]*Name:[[:space:]]*\(gfx[0-9a-fA-F]*\).*/\1/p' | head -1)"
    case "$ARCH" in gfx[0-9a-fA-F]*) ;; *) echo "build.sh: set HPCPERF_HIP_ARCH (for example gfx942)" >&2; exit 1;; esac
    MPI_PREFIX="$(cd "$(dirname "$MPICXX")/.." && pwd)"
    [ -d "$MPI_PREFIX/include" ] && [ -d "$MPI_PREFIX/lib" ] || { echo "build.sh: cannot derive MPI include/lib directories from $MPICXX; set MPICXX to the MPI wrapper" >&2; exit 1; }
    BUILD="$R/build/level2/sw4lite/hip"; WORK="$BUILD/source"
    if [ -d "$WORK" ]; then cmake -E remove_directory "$WORK"; fi
    mkdir -p "$BUILD"
    cmake -E copy_directory "$HERE/source/hip" "$WORK"
    echo "== SW4lite HIP build: $ARCH -> $BUILD"
    make -C "$WORK" -f Makefile.hip -j"${HPCPERF_BUILD_JOBS:-4}" \
      CXX="$HIPCXX" MPIPATH="$MPI_PREFIX" MPIINC="$MPI_PREFIX/include" MPILIB="$MPI_PREFIX/lib" \
      builddir="$BUILD/app" HIP_ARCH_FLAGS="--offload-arch=$ARCH"
    EXE="$BUILD/app/sw4lite"
    ;;
  *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2;;
esac
echo "== built: $EXE"
