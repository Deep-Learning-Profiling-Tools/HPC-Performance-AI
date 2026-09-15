#!/usr/bin/env bash
# Build ExaCA (Kokkos CUDA or HIP) with upstream's native CMake from the frozen source artifact:
#   deps/kokkos                  -> Kokkos install (stage 1)
#   deps/json/json-3.12.0.tar.xz -> nlohmann_json install (stage 2; replaces ExaCA's FetchContent download)
#   src/                         -> ExaCA + ExaCA-GrainAnalysis (stage 3)
#
#   ./build.sh [CUDA|HIP]        (default CUDA)
#
# Layout (one frozen source tree; generated state isolated per backend profile; nothing shared with Level 2):
#   source    level3/exaca/{src,deps}                    (materialized by tools/prepare_benchmark.sh; identity in provenance/;
#                                                        backend-independent -- never copied per backend)
#   profile   <cuda|hip>   (override HPCPERF_EXACA_PROFILE; a profile must name its backend)
#   build     build/level3/exaca/<profile>               (ExaCA) and .deps/level3/exaca/<profile>/build/ (dependencies)
#   install   .deps/level3/exaca/<profile>/install/{kokkos,json,exaca}  (+ .hpcperf-l3-fingerprint, .hpcperf-stage-done)
#   logs      .deps/level3/exaca/<profile>/logs
# This script reads application source ONLY from $HERE/src and $HERE/deps; it never fetches, clones or patches.
# ExaCA and its dependencies build out-of-source (nothing is written into src/ or deps/).
#
# Modification class: A (no upstream source modified; build flags only).
#
# Environment overrides:
#   HPCPERF_CUDA_ARCH   numeric compute capability (default: detected; 100 -> Kokkos_ARCH_BLACKWELL100)
#   HPCPERF_HIP_ARCH    Kokkos AMD arch name for HIP (default AMD_GFX950); HIP is UNTESTED here (no ROCm)
#   HPCPERF_BUILD_JOBS  parallel jobs (default 32)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
l3_isolate_build_env

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
l3_require_materialized "$HERE" || exit 3
SRC="$HERE/src"; DEPS="$HERE/deps"
[ -f "$SRC/CMakeLists.txt" ] && [ -f "$SRC/src/runCA.hpp" ] || { echo "build.sh: $SRC is not an ExaCA source tree -- run tools/prepare_benchmark.sh level3 exaca" >&2; exit 3; }
[ -f "$DEPS/kokkos/CMakeLists.txt" ] || { echo "build.sh: $DEPS/kokkos missing -- the artifact carries Kokkos as a benchmark-specific dependency" >&2; exit 3; }
JSON_TAR="$DEPS/json/json-3.12.0.tar.xz"
[ -f "$JSON_TAR" ] || { echo "build.sh: $JSON_TAR missing" >&2; exit 3; }
SHA="$(l3_source_commit "$HERE")"; TREE_SHA="$(l3_source_tree_sha "$HERE")"; KOKKOS_SHA="$(l3_component_commit "$HERE" deps/kokkos)"
KOKKOS_VER="$(sed -n 's/^set(Kokkos_VERSION_\(MAJOR\|MINOR\|PATCH\) \([0-9]*\))/\2/p' "$DEPS/kokkos/CMakeLists.txt" | paste -sd.)"
PROFILE="$(l3_backend_profile EXACA "$MODEL")"
l3_paths_profile exaca "$PROFILE" "$MODEL" || exit 2
BUILD_DIR="$L3_BUILD"
JOBS="${HPCPERF_BUILD_JOBS:-32}"
# the profile root separates the backends; no per-backend suffix inside the install any more
KINST="$L3_INSTALL/kokkos"; JINST="$L3_INSTALL/json"; EINST="$L3_INSTALL/exaca"

case "$BACKEND" in
    CUDA)
        ARCH="${HPCPERF_CUDA_ARCH:-$(l3_gpu_arch)}"
        case "$ARCH" in
            100) KARCH=BLACKWELL100;; 120) KARCH=BLACKWELL120;; 90) KARCH=HOPPER90;; 80) KARCH=AMPERE80;;
            *) echo "build.sh: no Kokkos arch mapping for compute capability '$ARCH' (set HPCPERF_CUDA_ARCH)" >&2; exit 2;;
        esac
        export NVCC_WRAPPER_DEFAULT_COMPILER="$CXX"
        CXX_FOR_KOKKOS="$DEPS/kokkos/bin/nvcc_wrapper"
        GPU_FLAGS=(-DKokkos_ENABLE_CUDA=ON "-DKokkos_ARCH_$KARCH=ON" -DKokkos_ENABLE_CUDA_LAMBDA=ON)
        ARCHNOTE="sm_$ARCH ($KARCH)" ;;
    HIP)
        command -v hipcc >/dev/null 2>&1 || { echo "build.sh: HIP requested but hipcc not found -- HIP build is UNTESTED on this machine (no ROCm)" >&2; exit 1; }
        KARCH="${HPCPERF_HIP_ARCH:-AMD_GFX950}"; CXX_FOR_KOKKOS=hipcc; L3_FP_ARCH="$KARCH"     # configured arch -> fingerprint
        GPU_FLAGS=(-DKokkos_ENABLE_HIP=ON "-DKokkos_ARCH_$KARCH=ON")
        ARCHNOTE="$KARCH" ;;
    *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac
CMAKE_OPTS="Kokkos_ENABLE_${BACKEND}=ON Kokkos_ARCH_${KARCH} Kokkos_ENABLE_SERIAL=ON ExaCA_REQUIRE_EXTERNAL_JSON=ON ExaCA_ENABLE_TESTING=OFF Finch=OFF CMAKE_BUILD_TYPE=Release"
FP="$(l3_fingerprint_text exaca "$SHA" "$MODEL" "kokkos=$KOKKOS_VER@${KOKKOS_SHA:0:12} nlohmann_json=3.12.0 source_tree=$TREE_SHA" "$CMAKE_OPTS" "n/a (host-staged halo exchange)")"
l3_fingerprint_check "$L3_INSTALL" "$FP" || exit 1
echo "# ExaCA $BACKEND profile=$PROFILE: upstream $SHA (frozen source tree $TREE_SHA), Kokkos $KOKKOS_VER (deps/kokkos @ ${KOKKOS_SHA:0:12}), arch $ARCHNOTE, MPI $(mpirun --version 2>/dev/null | head -1), install=$L3_INSTALL"
t0=$(date +%s)

# stage 1: Kokkos (from deps/kokkos, out-of-source)
if [ ! -f "$KINST/.hpcperf-stage-done" ]; then
    KB="$L3_BUILD_DEPS/kokkos"; mkdir -p "$KB"
    cmake -S "$DEPS/kokkos" -B "$KB" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$KINST" \
        -DCMAKE_CXX_COMPILER="$CXX_FOR_KOKKOS" -DCMAKE_CXX_STANDARD=17 -DKokkos_ENABLE_SERIAL=ON "${GPU_FLAGS[@]}" > "$L3_LOGS/kokkos-configure-$MODEL.log" 2>&1 \
        || { tail -20 "$L3_LOGS/kokkos-configure-$MODEL.log"; echo "build.sh: Kokkos configure failed" >&2; exit 1; }
    cmake --build "$KB" -j "$JOBS" > "$L3_LOGS/kokkos-build-$MODEL.log" 2>&1 || { tail -20 "$L3_LOGS/kokkos-build-$MODEL.log"; echo "build.sh: Kokkos build failed" >&2; exit 1; }
    cmake --install "$KB" > "$L3_LOGS/kokkos-install-$MODEL.log" 2>&1 || { echo "build.sh: Kokkos install failed" >&2; exit 1; }
    echo "kokkos=$KOKKOS_VER commit=$KOKKOS_SHA backend=$MODEL arch=$KARCH" > "$KINST/.hpcperf-stage-done"
    echo "# stage 1 Kokkos $KOKKOS_VER installed ($(( $(date +%s)-t0 )) s)"
else
    echo "# stage 1 Kokkos: reusing $KINST ($(cat "$KINST/.hpcperf-stage-done"))"
fi
# stage 2: nlohmann_json (header-only; the exact release tarball from deps/, extracted on the build side)
if [ ! -f "$JINST/.hpcperf-stage-done" ]; then
    JB="$L3_BUILD_DEPS/json"; rm -rf "$JB"; mkdir -p "$JB/src"
    tar -xJf "$JSON_TAR" -C "$JB/src" --strip-components=1
    cmake -S "$JB/src" -B "$JB/build" -DCMAKE_INSTALL_PREFIX="$JINST" -DJSON_BuildTests=OFF -DJSON_Install=ON > "$L3_LOGS/json-configure.log" 2>&1 \
        || { tail -20 "$L3_LOGS/json-configure.log"; echo "build.sh: json configure failed" >&2; exit 1; }
    cmake --install "$JB/build" > "$L3_LOGS/json-install.log" 2>&1 || { echo "build.sh: json install failed" >&2; exit 1; }
    echo "nlohmann_json=3.12.0 sha256=$(sha256sum "$JSON_TAR" | cut -d' ' -f1)" > "$JINST/.hpcperf-stage-done"
    echo "# stage 2 nlohmann_json 3.12.0 installed"
else
    echo "# stage 2 nlohmann_json: reusing $JINST"
fi
# stage 3: ExaCA (always rebuilt from $SRC; incremental via Ninja)
mkdir -p "$BUILD_DIR"
cmake -S "$SRC" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$EINST" \
    -DCMAKE_CXX_COMPILER="$CXX_FOR_KOKKOS" -DCMAKE_CXX_STANDARD=17 -DCMAKE_PREFIX_PATH="$KINST;$JINST" \
    -DExaCA_REQUIRE_EXTERNAL_JSON=ON -DExaCA_ENABLE_TESTING=OFF -DExaCA_REQUIRE_FINCH=OFF > "$L3_LOGS/configure-$MODEL.log" 2>&1 \
    || { tail -30 "$L3_LOGS/configure-$MODEL.log"; echo "build.sh: configure failed (log: $L3_LOGS/configure-$MODEL.log)" >&2; exit 1; }
cmake --build "$BUILD_DIR" -j "$JOBS" > "$L3_LOGS/build-$MODEL.log" 2>&1 \
    || { tail -30 "$L3_LOGS/build-$MODEL.log"; echo "build.sh: build failed (log: $L3_LOGS/build-$MODEL.log)" >&2; exit 1; }
cmake --install "$BUILD_DIR" > "$L3_LOGS/install-$MODEL.log" 2>&1 || { echo "build.sh: install failed" >&2; exit 1; }
l3_fingerprint_write "$L3_INSTALL" "$FP"
echo "# built in $(( $(date +%s)-t0 )) s: $EINST/bin/ExaCA (+ ExaCA-GrainAnalysis), data in $EINST/share/ExaCA"
echo "# compiler warning lines: $(grep -c 'warning' "$L3_LOGS/build-$MODEL.log" || true)"
