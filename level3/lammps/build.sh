#!/usr/bin/env bash
# Build LAMMPS (KOKKOS package, CUDA or HIP) with upstream's native CMake and
# the Kokkos version LAMMPS bundles (lib/kokkos) -- the upstream-supported path.
#
#   ./build.sh [CUDA|HIP]        (default CUDA)
#
# Layout (one frozen source tree; generated state isolated per backend profile; nothing shared with Level 2):
#   source    level3/lammps/src                         (frozen source bundle materialized by tools/prepare_benchmark.sh;
#                                                       the app owns its Kokkos in src/lib/kokkos; identity in provenance/;
#                                                       backend-independent -- never copied per backend)
#   profile   <cuda|hip> or <variant>.<cuda|hip> with HPCPERF_LAMMPS_VARIANT=reaxff (+REAXFF package)
#             (override HPCPERF_LAMMPS_PROFILE; a profile must name its backend)
#   build     build/level3/lammps/<profile>
#   install   .deps/level3/lammps/<profile>/install     (+ .hpcperf-l3-fingerprint)
#   logs      .deps/level3/lammps/<profile>/logs
# This script reads application source ONLY from $HERE/src; it never fetches, clones or patches.
#
# Toolchain: conda GCC 13.3.0 as nvcc_wrapper host compiler (LAMMPS documents
# GCC >= 8 and C++17), system CUDA, conda Open MPI 5.0.10 (CUDA-aware).
# Packages: KOKKOS + MOLECULE, KSPACE (pppm/kk needs FFT_KOKKOS=CUFFT/HIPFFT),
# MANYBODY, RIGID, GRANULAR -- the set the bench/ inputs need.
#
# Modification class: A (no upstream source modified; build flags only).
#
# Environment overrides:
#   HPCPERF_CUDA_ARCH   numeric compute capability (default: detected; 100 -> Kokkos_ARCH_BLACKWELL100)
#   HPCPERF_HIP_ARCH    Kokkos AMD arch name for HIP (default AMD_GFX950); HIP is UNTESTED here
#   HPCPERF_BUILD_JOBS  parallel jobs (default 32)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# conda activation scripts are not set -u safe
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
l3_isolate_build_env    # Level 3 builds must not see Level 2 .deps/install prefixes

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
l3_require_materialized "$HERE" || exit 3
SRC="$HERE/src"
[ -f "$SRC/cmake/CMakeLists.txt" ] || { echo "build.sh: $SRC is not a LAMMPS source tree -- run tools/prepare_benchmark.sh level3 lammps" >&2; exit 3; }
SHA="$(l3_source_commit "$HERE")"; TREE_SHA="$(l3_source_tree_sha "$HERE")"
KOKKOS_VER="$(sed -n 's/^set(Kokkos_VERSION_\(MAJOR\|MINOR\|PATCH\) \([0-9]*\))/\2/p' "$SRC/lib/kokkos/CMakeLists.txt" | paste -sd.)"
# Build variant (HPCPERF_LAMMPS_VARIANT): "" = the default package set (profile <backend>);
# "reaxff" = the same set plus the REAXFF package (profile reaxff.<backend>) for the CORAL-2
# ReaxFF/HNS workload. A variant is its own profile: own build/install/logs/cache trees and
# its own fingerprint (the package list is part of it); the default profile is never rebuilt
# or extended by it, and the frozen source tree is shared read-only.
VARIANT="${HPCPERF_LAMMPS_VARIANT:-}"
case "$VARIANT" in ""|reaxff) ;; *) echo "build.sh: HPCPERF_LAMMPS_VARIANT must be empty or 'reaxff' (got '$VARIANT')" >&2; exit 2;; esac
PROFILE="$(l3_backend_profile LAMMPS "$MODEL" "$VARIANT")"
l3_paths_profile lammps "$PROFILE" "$MODEL" || exit 2
BUILD_DIR="$L3_BUILD"
JOBS="${HPCPERF_BUILD_JOBS:-32}"
PKGS=(-DPKG_KOKKOS=yes -DPKG_MOLECULE=yes -DPKG_KSPACE=yes -DPKG_MANYBODY=yes -DPKG_RIGID=yes -DPKG_GRANULAR=yes)
PKGLIST="KOKKOS,MOLECULE,KSPACE,MANYBODY,RIGID,GRANULAR"
if [ "$VARIANT" = reaxff ]; then PKGS+=(-DPKG_REAXFF=yes); PKGLIST="$PKGLIST,REAXFF"; fi

case "$BACKEND" in
    CUDA)
        ARCH="${HPCPERF_CUDA_ARCH:-$(l3_gpu_arch)}"
        case "$ARCH" in
            100) KARCH=BLACKWELL100;; 120) KARCH=BLACKWELL120;; 90) KARCH=HOPPER90;; 80) KARCH=AMPERE80;;
            *) echo "build.sh: no Kokkos arch mapping for compute capability '$ARCH' (set HPCPERF_CUDA_ARCH)" >&2; exit 2;;
        esac
        export NVCC_WRAPPER_DEFAULT_COMPILER="$CXX"
        GPU_FLAGS=(-DCMAKE_CXX_COMPILER="$SRC/lib/kokkos/bin/nvcc_wrapper"
                   -DKokkos_ENABLE_CUDA=yes "-DKokkos_ARCH_$KARCH=yes" -DFFT_KOKKOS=CUFFT)
        ARCHNOTE="sm_$ARCH ($KARCH)" ;;
    HIP)
        command -v hipcc >/dev/null 2>&1 || { echo "build.sh: HIP requested but hipcc not found -- HIP build is UNTESTED on this machine (no ROCm)" >&2; exit 1; }
        KARCH="${HPCPERF_HIP_ARCH:-AMD_GFX950}"; L3_FP_ARCH="$KARCH"     # the arch this build configures goes into the fingerprint
        GPU_FLAGS=(-DCMAKE_CXX_COMPILER=hipcc -DKokkos_ENABLE_HIP=yes "-DKokkos_ARCH_$KARCH=yes" -DFFT_KOKKOS=HIPFFT)
        ARCHNOTE="$KARCH" ;;
    *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac

# Image output libs are disabled: the node has a libjpeg runtime but no headers
# (jpeglib.h), and dump image is not part of any benchmark here.
CMAKE_OPTS="BUILD_MPI=yes BUILD_OMP=yes CXX_STANDARD=17 Kokkos_ENABLE_${BACKEND}=yes Kokkos_ARCH_${KARCH} Kokkos_ENABLE_OPENMP=yes Kokkos_ENABLE_SERIAL=yes FFT=KISS FFT_KOKKOS=${GPU_FLAGS[-1]#-DFFT_KOKKOS=} WITH_JPEG=no WITH_PNG=no PKGS=$PKGLIST"
FP="$(l3_fingerprint_text lammps "$SHA" "$MODEL" "kokkos(bundled)=$KOKKOS_VER" "$CMAKE_OPTS" "runtime(-pk kokkos gpu/aware)")"
l3_fingerprint_check "$L3_INSTALL" "$FP" || exit 1

echo "# LAMMPS $BACKEND profile=$PROFILE: upstream $SHA (frozen source tree $TREE_SHA), bundled Kokkos $KOKKOS_VER, arch $ARCHNOTE, MPI $(mpirun --version 2>/dev/null | head -1), install=$L3_INSTALL"
mkdir -p "$BUILD_DIR"
cmake -S "$SRC/cmake" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$L3_INSTALL" \
    -DCMAKE_CXX_STANDARD=17 -DBUILD_MPI=yes -DBUILD_OMP=yes -DLAMMPS_MACHINE="kokkos_$MODEL" \
    -DKokkos_ENABLE_OPENMP=yes -DKokkos_ENABLE_SERIAL=yes -DFFT=KISS \
    -DWITH_JPEG=no -DWITH_PNG=no -DWITH_GZIP=yes \
    "${GPU_FLAGS[@]}" "${PKGS[@]}" > "$L3_LOGS/configure-$MODEL.log" 2>&1 \
    || { tail -30 "$L3_LOGS/configure-$MODEL.log"; echo "build.sh: configure failed (log: $L3_LOGS/configure-$MODEL.log)" >&2; exit 1; }
t0=$(date +%s)
cmake --build "$BUILD_DIR" -j "$JOBS" > "$L3_LOGS/build-$MODEL.log" 2>&1 \
    || { tail -30 "$L3_LOGS/build-$MODEL.log"; echo "build.sh: build failed (log: $L3_LOGS/build-$MODEL.log)" >&2; exit 1; }
cmake --install "$BUILD_DIR" > "$L3_LOGS/install-$MODEL.log" 2>&1 || { echo "build.sh: install failed" >&2; exit 1; }
l3_fingerprint_write "$L3_INSTALL" "$FP"
echo "# built in $(( $(date +%s)-t0 )) s: $BUILD_DIR/lmp_kokkos_$MODEL (installed under $L3_INSTALL)"
echo "# compiler warning lines: $(grep -c 'warning' "$L3_LOGS/build-$MODEL.log" || true)"
