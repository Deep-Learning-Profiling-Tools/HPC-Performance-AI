#!/usr/bin/env bash
# Build SPARTA (KOKKOS package, CUDA or HIP) with upstream's native CMake and
# the Kokkos it bundles (lib/kokkos) -- upstream's own kokkos_cuda/kokkos_hip
# preset recipe, with the GPU architecture chosen for this node instead of the
# preset's hard-coded HOPPER90.
#
#   ./build.sh [CUDA|HIP]        (default CUDA)
#
# Layout (one frozen source tree, generated state isolated per backend profile):
# source level3/sparta/src (frozen source bundle materialized by
# tools/prepare_benchmark.sh; identity in provenance/; backend-independent),
# profile <cuda|hip> (override HPCPERF_SPARTA_PROFILE; must name the backend),
# build build/level3/sparta/<profile>, install .deps/level3/sparta/<profile>/install
# (+ .hpcperf-l3-fingerprint), logs .deps/level3/sparta/<profile>/logs. Application
# source is read ONLY from $HERE/src; nothing is fetched, cloned or patched here.
#
# Recipe = upstream cmake/presets/kokkos_common.cmake (PKG_KOKKOS, BUILD_MPI,
# -O3) loaded with -C, plus the settings of cmake/presets/kokkos_cuda.cmake
# given explicitly: nvcc_wrapper as CXX (host compiler = conda GCC 13.3.0),
# Kokkos_ENABLE_CUDA, Kokkos_ENABLE_SERIAL, FFT_KOKKOS=CUFFT, and
# Kokkos_ARCH_BLACKWELL100 (sm_100) instead of HOPPER90. SPARTA requires
# C++20 with the KOKKOS package (cmake/CMakeLists.txt).
#
# Modification class: A (no upstream file modified; the arch differs from the
# shipped preset only through command-line cache entries).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
l3_isolate_build_env    # Level 3 builds must not see Level 2 .deps/install prefixes

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
l3_require_materialized "$HERE" || exit 3
SRC="$HERE/src"
[ -f "$SRC/cmake/CMakeLists.txt" ] || { echo "build.sh: $SRC is not a SPARTA source tree -- run tools/prepare_benchmark.sh level3 sparta" >&2; exit 3; }
SHA="$(l3_source_commit "$HERE")"; TREE_SHA="$(l3_source_tree_sha "$HERE")"
KOKKOS_VER="$(sed -n 's/^set(Kokkos_VERSION_\(MAJOR\|MINOR\|PATCH\) \([0-9]*\))/\2/p' "$SRC/lib/kokkos/CMakeLists.txt" | paste -sd.)"
PROFILE="$(l3_backend_profile SPARTA "$MODEL")"
l3_paths_profile sparta "$PROFILE" "$MODEL" || exit 2
BUILD_DIR="$L3_BUILD"
JOBS="${HPCPERF_BUILD_JOBS:-32}"

case "$BACKEND" in
    CUDA)
        ARCH="${HPCPERF_CUDA_ARCH:-$(l3_gpu_arch)}"
        case "$ARCH" in
            100) KARCH=BLACKWELL100;; 120) KARCH=BLACKWELL120;; 90) KARCH=HOPPER90;; 80) KARCH=AMPERE80;;
            *) echo "build.sh: no Kokkos arch mapping for compute capability '$ARCH' (set HPCPERF_CUDA_ARCH)" >&2; exit 2;;
        esac
        export NVCC_WRAPPER_DEFAULT_COMPILER="$CXX"
        GPU_FLAGS=(-DCMAKE_CXX_COMPILER="$SRC/lib/kokkos/bin/nvcc_wrapper"
                   -DKokkos_ENABLE_CUDA=ON "-DKokkos_ARCH_$KARCH=ON" -DFFT_KOKKOS=CUFFT)
        FFTK=CUFFT ;;
    HIP)
        command -v hipcc >/dev/null 2>&1 || { echo "build.sh: HIP requested but hipcc not found -- HIP build is UNTESTED on this machine (no ROCm)" >&2; exit 1; }
        KARCH="${HPCPERF_HIP_ARCH:-AMD_GFX950}"
        GPU_FLAGS=(-DCMAKE_CXX_COMPILER=hipcc -DKokkos_ENABLE_HIP=ON "-DKokkos_ARCH_$KARCH=ON" -DFFT_KOKKOS=HIPFFT)
        FFTK=HIPFFT ;;
    *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac

CMAKE_OPTS="preset=kokkos_common BUILD_MPI=ON PKG_KOKKOS=ON CXX_STANDARD=20 Kokkos_ENABLE_${BACKEND}=ON Kokkos_ARCH_${KARCH} Kokkos_ENABLE_SERIAL=ON Kokkos_ENABLE_OPENMP=OFF FFT_KOKKOS=$FFTK"
FP="$(l3_fingerprint_text sparta "$SHA" "$MODEL" "kokkos(bundled)=$KOKKOS_VER" "$CMAKE_OPTS" "runtime(-pk kokkos gpu/aware)")"
l3_fingerprint_check "$L3_INSTALL" "$FP" || exit 1

echo "# SPARTA $BACKEND profile=$PROFILE: upstream $SHA (frozen source tree $TREE_SHA), bundled Kokkos $KOKKOS_VER, arch $KARCH, MPI $(mpirun --version 2>/dev/null | head -1), install=$L3_INSTALL"
mkdir -p "$BUILD_DIR"
cmake -S "$SRC/cmake" -B "$BUILD_DIR" -G Ninja -C "$SRC/cmake/presets/kokkos_common.cmake" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$L3_INSTALL" -DCMAKE_CXX_STANDARD=20 \
    -DSPARTA_MACHINE="kokkos_$MODEL" -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=OFF \
    "${GPU_FLAGS[@]}" > "$L3_LOGS/configure-$MODEL.log" 2>&1 \
    || { tail -30 "$L3_LOGS/configure-$MODEL.log"; echo "build.sh: configure failed (log: $L3_LOGS/configure-$MODEL.log)" >&2; exit 1; }
t0=$(date +%s)
cmake --build "$BUILD_DIR" -j "$JOBS" > "$L3_LOGS/build-$MODEL.log" 2>&1 \
    || { tail -30 "$L3_LOGS/build-$MODEL.log"; echo "build.sh: build failed (log: $L3_LOGS/build-$MODEL.log)" >&2; exit 1; }
cmake --install "$BUILD_DIR" > "$L3_LOGS/install-$MODEL.log" 2>&1 || { echo "build.sh: install failed" >&2; exit 1; }
l3_fingerprint_write "$L3_INSTALL" "$FP"
EXE="$(find "$BUILD_DIR" -maxdepth 2 -name "spa_kokkos_$MODEL" -type f | head -1)"
echo "# built in $(( $(date +%s)-t0 )) s: ${EXE:-<spa_kokkos_$MODEL not found under $BUILD_DIR>} (installed under $L3_INSTALL)"
echo "# compiler warning lines: $(grep -c 'warning' "$L3_LOGS/build-$MODEL.log" || true)"
