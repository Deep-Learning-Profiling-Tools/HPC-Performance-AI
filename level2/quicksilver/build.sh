#!/usr/bin/env bash
# Build the LLNL CUDA branch or the official LLNL AMD-HIP branch.
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

detect_cuda_arch() {
    local a="${HPCPERF_CUDA_ARCH:-${CUDA_ARCH:-}}"
    [ -n "$a" ] || a="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
    a="${a#sm_}"; a="${a//./}"
    case "$a" in ''|*[!0-9]*) echo "build.sh: set HPCPERF_CUDA_ARCH (for example 90)" >&2; return 1;; esac
    echo "$a"
}
detect_hip_arch() {
    local a="${HPCPERF_HIP_ARCH:-${HIP_ARCH:-}}"
    [ -n "$a" ] || a="$(rocminfo 2>/dev/null | sed -n 's/^[[:space:]]*Name:[[:space:]]*\(gfx[0-9a-fA-F]*\).*/\1/p' | head -1)"
    case "$a" in gfx[0-9a-fA-F]*) echo "$a";; *) echo "build.sh: set HPCPERF_HIP_ARCH (for example gfx942)" >&2; return 1;; esac
}

mpi_flags() {
    "$MPICXX" --showme:"$1" 2>/dev/null || "$MPICXX" -show 2>/dev/null
}

case "$BACKEND" in
  CUDA)
    NVCC="${NVCC:-${CUDA_HOME:-/usr/local/cuda}/bin/nvcc}"
    [ -x "$NVCC" ] || NVCC="$(command -v nvcc || true)"
    [ -n "$NVCC" ] || { echo "build.sh: nvcc not found; source hpcperf_env.sh or set CUDA_HOME" >&2; exit 1; }
    ARCH="$(detect_cuda_arch)"; SRC="$HERE/source/cuda"; BUILD="$R/build/level2/quicksilver/cuda"
    MPI_COMPILE="$(mpi_flags compile)"; MPI_LINK_RAW="$(mpi_flags link)"
    MPI_COMPILE="${MPI_COMPILE//-pthread/-Xcompiler=-pthread}"
    MPI_LINK=""
    for flag in $MPI_LINK_RAW; do
        case "$flag" in
          -pthread) MPI_LINK+=" -Xcompiler=-pthread";;
          -Wl,*) rest="${flag#-Wl,}"; IFS=',' read -r -a parts <<< "$rest"; for part in "${parts[@]}"; do MPI_LINK+=" -Xlinker=$part"; done;;
          *) MPI_LINK+=" $flag";;
        esac
    done
    COMPILER="$NVCC"; CXXFLAGS="-std=c++14 -O3 -arch=sm_${ARCH} -rdc=true -ccbin=$CXX -I$SRC -I$BUILD"
    CPPFLAGS="-x cu -DHAVE_MPI -DHAVE_ASYNC_MPI -DHAVE_CUDA $MPI_COMPILE"
    COMMIT=eb68bb8d6fc53de1f65011d4e79ff2ed0dd60f3b; BRANCH=master
    ;;
  HIP)
    COMPILER="${HIPCXX:-$(command -v hipcc || true)}"
    [ -n "$COMPILER" ] || { echo "build.sh: hipcc not found; install/source ROCm" >&2; exit 1; }
    ARCH="$(detect_hip_arch)"; SRC="$HERE/source/hip"; BUILD="$R/build/level2/quicksilver/hip"
    MPI_COMPILE="$(mpi_flags compile)"; MPI_LINK="$(mpi_flags link)"
    CXXFLAGS="-std=c++14 -O3 --offload-arch=$ARCH -fgpu-rdc -I$SRC -I$BUILD"
    CPPFLAGS="-DHAVE_HIP=1 -DHAVE_MPI -DHAVE_ASYNC_MPI -DMaxIt=15 $MPI_COMPILE"
    COMMIT=c1e29f2a34f6505969720e2ea72b9fa25ef00743; BRANCH=AMD-HIP
    ;;
  *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2;;
esac

mkdir -p "$BUILD"
touch "$BUILD/.depend"
echo "== Quicksilver $BACKEND build: $ARCH -> $BUILD"
make -C "$BUILD" -f "$SRC/Makefile" VPATH="$SRC" -j"${HPCPERF_BUILD_JOBS:-4}" \
  CXX="$COMPILER" CXXFLAGS="$CXXFLAGS" CPPFLAGS="$CPPFLAGS" LDFLAGS="$MPI_LINK" \
  GITHASH="$COMMIT" GITVERS="$BRANCH"
echo "== built: $BUILD/qs"
