#!/usr/bin/env bash
# Build the MiniEM driver (Trilinos Panzer mini-em, BlockPrec example) against the project's Trilinos install.
#
#   ./build.sh [CUDA] [extra cmake -D options...]
#
# Output tree: $R/build/level2/miniem/cuda/PanzerMiniEM_BlockPrec (+ decks/ with the upstream input files)
#
# MiniEM is a Trilinos subpackage: the physics (Panzer closures, equation sets, RefMaxwell solver setup) is
# the library PanzerMiniEM inside Trilinos, and the executable is a ~800-line driver that upstream builds only
# through TriBITS. The Trilinos dependency (Panzer + STK adapters, MueLu, Teko, Belos, Ifpack2, Amesos2, Tpetra,
# Kokkos 5.2.1 CUDA + Serial, SEACAS Exodus/Ioss) is built once by `setup_level2_deps.sh trilinos` into
# .deps/install/trilinos (multi-hour); this script only compiles the vendored driver (src/, byte-identical to
# upstream except the tools/timing ROI marker lines in main.cpp; src/UPSTREAM_SHA256SUMS keeps upstream's
# checksums) with the standalone src/CMakeLists.txt.
#
# Compiler recipe = the one the dependency was built with: mpicxx driving Trilinos' nvcc_wrapper (OMPI_CXX),
# host compiler = the project GCC ($CXX). The Panzer/Phalanx headers contain device code, so the driver cannot
# be compiled with a plain host compiler.
#
# Environment overrides:
#   HPCPERF_TRILINOS_PREFIX  Trilinos install prefix (default $R/.deps/install/trilinos)
#   HPCPERF_BUILD_JOBS       parallel build jobs (default 4)
# HIP: not available -- the Trilinos dependency is built for CUDA only on this site (untested elsewhere).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true

set -euo pipefail
# tools/timing ROI markers (header-only; a no-op unless measured): tools/timing/roi/README.md
export CPATH="$R/tools/timing/roi${CPATH:+:$CPATH}"

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
[ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level2/miniem/$MODEL"
JOBS="${HPCPERF_BUILD_JOBS:-4}"
TRILINOS_PREFIX="${HPCPERF_TRILINOS_PREFIX:-$R/.deps/install/trilinos}"

case "$BACKEND" in
    CUDA) ;;
    HIP)  echo "build.sh: MiniEM HIP is not available: the Trilinos dependency (setup_level2_deps.sh) is built for CUDA only" >&2; exit 1 ;;
    *)    echo "build.sh: unknown backend '$BACKEND' (CUDA)" >&2; exit 2 ;;
esac

CFG="$(ls "$TRILINOS_PREFIX"/lib*/cmake/Trilinos/TrilinosConfig.cmake 2>/dev/null | head -1 || true)"
if [ -z "$CFG" ]; then
    echo "build.sh: Trilinos not found under $TRILINOS_PREFIX (no lib*/cmake/Trilinos/TrilinosConfig.cmake)" >&2
    echo "          build it with: source hpcperf_env.sh && ./setup_level2_deps.sh trilinos   (or set HPCPERF_TRILINOS_PREFIX)" >&2
    exit 1
fi
[ -f "$TRILINOS_PREFIX/.hpcperf-built" ] || { echo "build.sh: $TRILINOS_PREFIX has no .hpcperf-built marker (incomplete dependency build)" >&2; exit 1; }
NVW="$TRILINOS_PREFIX/bin/nvcc_wrapper"
[ -x "$NVW" ] || NVW="$R/.deps/src/trilinos/packages/kokkos/bin/nvcc_wrapper"
[ -x "$NVW" ] || { echo "build.sh: nvcc_wrapper not found (looked in $TRILINOS_PREFIX/bin and .deps/src/trilinos/packages/kokkos/bin)" >&2; exit 1; }
command -v mpicxx >/dev/null || { echo "build.sh: mpicxx not on PATH -- source hpcperf_env.sh" >&2; exit 1; }

ARCH="${HPCPERF_CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')}"
[ -n "$ARCH" ] && echo "# GPU compute capability detected: sm_${ARCH} (the Trilinos dependency was built for: $(sed -n 's/^profile=.*arch=//p' "$TRILINOS_PREFIX/.hpcperf-fingerprint" 2>/dev/null))"
echo "# Trilinos: $(cat "$TRILINOS_PREFIX/.hpcperf-commit" 2>/dev/null) at $TRILINOS_PREFIX"

export OMPI_CXX="$NVW" NVCC_WRAPPER_DEFAULT_COMPILER="${CXX:-g++}"
cmake -S "$HERE/src" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_CXX_STANDARD=20 \
    -DTrilinos_DIR="$(dirname "$CFG")" \
    "$@"
cmake --build "$BUILD_DIR" -j "$JOBS"
echo "# built: $BUILD_DIR/PanzerMiniEM_BlockPrec ($(ls "$BUILD_DIR/decks" | wc -l) decks in $BUILD_DIR/decks)"
