#!/usr/bin/env bash
# Build VPIC (vpic-kokkos, Kokkos CUDA backend) with upstream's native CMake and the Kokkos it pins as a
# submodule (src/kokkos, BUILD_INTERNAL_KOKKOS=ON -- upstream's arch/CUDA-Release recipe with the GPU
# architecture of this node instead of ADA89), then compile every registered input deck
# (decks/<id>.cxx, inputs.yaml) with the build tree's own deck compiler (bin/vpic) into
# $BUILD/decks/<id>.Linux. VPIC decks are compile-time inputs: a deck is a C++ file linked against
# libvpic, so each registered input is a separate executable.
#
#   ./build.sh [CUDA|HIP]        (default CUDA)
#
# Layout (one frozen source tree, generated state isolated per backend profile):
#   source   level3/vpic/src (frozen source bundle materialized by tools/prepare_benchmark.sh; identity in
#            provenance/; backend-independent; Kokkos in src/kokkos)
#   profile  <cuda|hip> (override HPCPERF_VPIC_PROFILE; must name the backend)
#   build    build/level3/vpic/<profile>   (libvpic, bin/vpic, the integrated tests, decks/<id>.Linux)
#   install  .deps/level3/vpic/<profile>/install (+ .hpcperf-l3-fingerprint), logs .deps/level3/vpic/<profile>/logs
# Application source is read ONLY from $HERE/src; nothing is fetched, cloned or patched here.
#
# Recipe = upstream arch/CUDA-Release (nvcc_wrapper as CXX, BUILD_INTERNAL_KOKKOS, ENABLE_KOKKOS_CUDA,
# ENABLE_INTEGRATED_TESTS, -rdynamic) with Kokkos_ARCH_BLACKWELL100 (sm_100) instead of ADA89, the
# Kokkos OpenMP backend OFF (one host thread per rank; the GPU does the work), unit tests OFF (Catch2
# download), GPU-aware MPI OFF (upstream default; single node, host-staged exchanges).
#
# Modification class: A (no upstream file modified; cache entries only).
#
# Environment overrides:
#   HPCPERF_CUDA_ARCH   numeric compute capability (default: detected; 100 -> Kokkos_ARCH_BLACKWELL100)
#   HPCPERF_HIP_ARCH    Kokkos AMD arch name for HIP (default AMD_GFX950); HIP is UNTESTED here
#   HPCPERF_BUILD_JOBS  parallel jobs (default 32)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
l3_isolate_build_env    # Level 3 builds must not see Level 2 .deps/install prefixes
l3_clean_conda_build_env    # conda's CFLAGS/CXXFLAGS/LDFLAGS (-march=nocona, rpaths) would reach nvcc_wrapper through CMake

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
l3_require_materialized "$HERE" || exit 3
SRC="$HERE/src"
[ -f "$SRC/CMakeLists.txt" ] && [ -f "$SRC/kokkos/CMakeLists.txt" ] || { echo "build.sh: $SRC is not a vpic-kokkos source tree with its kokkos submodule -- run tools/prepare_benchmark.sh level3 vpic" >&2; exit 3; }
SHA="$(l3_source_commit "$HERE")"; TREE_SHA="$(l3_source_tree_sha "$HERE")"
KOKKOS_VER="$(sed -n 's/^set(Kokkos_VERSION_\(MAJOR\|MINOR\|PATCH\) \([0-9]*\))/\2/p' "$SRC/kokkos/CMakeLists.txt" | paste -sd.)"
PROFILE="$(l3_backend_profile VPIC "$MODEL")"
l3_paths_profile vpic "$PROFILE" "$MODEL" || exit 2
BUILD_DIR="$L3_BUILD"
JOBS="${HPCPERF_BUILD_JOBS:-32}"
TOOL="$R/tools/inputs/hpcperf_inputs.py"

case "$BACKEND" in
    CUDA)
        ARCH="${HPCPERF_CUDA_ARCH:-$(l3_gpu_arch)}"
        case "$ARCH" in
            100) KARCH=BLACKWELL100;; 120) KARCH=BLACKWELL120;; 90) KARCH=HOPPER90;; 80) KARCH=AMPERE80;; 89) KARCH=ADA89;;
            *) echo "build.sh: no Kokkos arch mapping for compute capability '$ARCH' (set HPCPERF_CUDA_ARCH)" >&2; exit 2;;
        esac
        export NVCC_WRAPPER_DEFAULT_COMPILER="$CXX"
        GPU_FLAGS=(-DCMAKE_CXX_COMPILER="$SRC/kokkos/bin/nvcc_wrapper" -DENABLE_KOKKOS_CUDA=ON "-DKokkos_ARCH_$KARCH=ON") ;;
    HIP)
        command -v hipcc >/dev/null 2>&1 || { echo "build.sh: HIP requested but hipcc not found -- HIP build is UNTESTED on this machine (no ROCm)" >&2; exit 1; }
        KARCH="${HPCPERF_HIP_ARCH:-AMD_GFX950}"; L3_FP_ARCH="$KARCH"
        GPU_FLAGS=(-DCMAKE_CXX_COMPILER=hipcc -DENABLE_KOKKOS_HIP=ON "-DKokkos_ARCH_$KARCH=ON") ;;
    *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac

CMAKE_OPTS="BUILD_INTERNAL_KOKKOS=ON ENABLE_KOKKOS_${BACKEND}=ON Kokkos_ARCH_${KARCH} ENABLE_KOKKOS_OPENMP=OFF ENABLE_INTEGRATED_TESTS=ON ENABLE_UNIT_TESTS=OFF VPIC_ENABLE_GPU_AWARE_MPI=OFF CMAKE_BUILD_TYPE=Release CMAKE_CXX_FLAGS=-rdynamic"
DECKS="$(python3 "$TOOL" list "$HERE" | awk '{print $1}' | paste -sd,)"
FP="$(l3_fingerprint_text vpic "$SHA" "$MODEL" "kokkos(submodule)=$KOKKOS_VER" "$CMAKE_OPTS decks=$DECKS" "runtime(host-staged MPI, 1 rank per GPU)")"
l3_fingerprint_check "$L3_INSTALL" "$FP" || exit 1

echo "# VPIC $BACKEND profile=$PROFILE: upstream $SHA (frozen source tree $TREE_SHA), Kokkos $KOKKOS_VER (submodule, internal build), arch $KARCH, MPI $(mpirun --version 2>/dev/null | head -1), build=$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cmake -S "$SRC" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$L3_INSTALL" \
    -DBUILD_INTERNAL_KOKKOS=ON -DENABLE_KOKKOS_OPENMP=OFF -DENABLE_INTEGRATED_TESTS=ON -DENABLE_UNIT_TESTS=OFF \
    -DVPIC_ENABLE_GPU_AWARE_MPI=OFF -DCMAKE_CXX_FLAGS="-rdynamic" "${GPU_FLAGS[@]}" > "$L3_LOGS/configure-$MODEL.log" 2>&1 \
    || { tail -30 "$L3_LOGS/configure-$MODEL.log"; echo "build.sh: configure failed (log: $L3_LOGS/configure-$MODEL.log)" >&2; exit 1; }
t0=$(date +%s)
cmake --build "$BUILD_DIR" -j "$JOBS" > "$L3_LOGS/build-$MODEL.log" 2>&1 \
    || { tail -30 "$L3_LOGS/build-$MODEL.log"; echo "build.sh: build failed (log: $L3_LOGS/build-$MODEL.log)" >&2; exit 1; }
[ -x "$BUILD_DIR/bin/vpic" ] || { echo "build.sh: $BUILD_DIR/bin/vpic (the deck compiler) was not generated" >&2; exit 1; }

# The registered decks: compiled with the build tree's deck compiler (bin/vpic = nvcc_wrapper + libvpic +
# Kokkos flags, upstream's documented way to build an input deck). The deck's sha256 is recorded next to
# the executable; run.sh refuses a deck executable whose source has changed since.
mkdir -p "$BUILD_DIR/decks"
for id in ${DECKS//,/ }; do
    deck="$HERE/$(python3 "$TOOL" param "$HERE" "$id" deck)"
    [ -f "$deck" ] || { echo "build.sh: deck $deck of input $id missing" >&2; exit 1; }
    cp -f "$deck" "$BUILD_DIR/decks/$id.cxx"
    ( cd "$BUILD_DIR/decks" && "$BUILD_DIR/bin/vpic" "$id.cxx" > "$L3_LOGS/deck-$id-$MODEL.log" 2>&1 ) \
        || { tail -20 "$L3_LOGS/deck-$id-$MODEL.log"; echo "build.sh: deck $id failed to compile (log: $L3_LOGS/deck-$id-$MODEL.log)" >&2; exit 1; }
    [ -x "$BUILD_DIR/decks/$id.Linux" ] || { echo "build.sh: $BUILD_DIR/decks/$id.Linux not produced" >&2; exit 1; }
    l3_sha_file "$deck" > "$BUILD_DIR/decks/$id.deck.sha256"
done
cmake --install "$BUILD_DIR" > "$L3_LOGS/install-$MODEL.log" 2>&1 || { echo "build.sh: install failed" >&2; exit 1; }
l3_fingerprint_write "$L3_INSTALL" "$FP"
echo "# built in $(( $(date +%s)-t0 )) s: libvpic + bin/vpic + decks [$DECKS] under $BUILD_DIR/decks (installed under $L3_INSTALL)"
echo "# compiler warning lines: $(grep -c 'warning' "$L3_LOGS/build-$MODEL.log" || true)"
