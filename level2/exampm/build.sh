#!/usr/bin/env bash
# Build ExaMPM (Cabana Grid + Kokkos, MPI) for the CUDA or HIP programming model.
#
#   ./build.sh [CUDA|HIP] [extra cmake -D options...]
#
# Output tree: $R/build/level2/exampm/<cuda|hip>/examples/{DamBreak,FreeFall}
#
# ExaMPM is a Kokkos/Cabana application: the same src/ + examples/ are the
# CUDA and the HIP variant, the backend is chosen by the Kokkos/Cabana
# installation it is linked against. CUDA uses the project's Kokkos 5.2.1 and
# Cabana 0.8.0 in $R/.deps/install/{kokkos,cabana} (built by
# setup_level2_deps.sh with Serial+OpenMP+CUDA for the GPU present at that
# time; Cabana with Core+Grid, MPI, parallel HDF5, heFFTe) and compiles through
# Kokkos' nvcc_wrapper. HIP expects a ROCm-enabled Kokkos/Cabana in
# $R/.deps/install/{kokkos-hip,cabana-hip} and hipcc -- neither exists on the
# development machine, so the HIP form is untested.
#
# Environment overrides:
#   HPCPERF_CUDA_ARCH     compute capability digits (e.g. 100 for sm_100);
#                         default: detected from nvidia-smi. Only checked
#                         against the architecture baked into the Kokkos
#                         install (Kokkos, not the app, fixes the GPU arch).
#   HPCPERF_BUILD_JOBS    parallel build jobs (default 4)
#   HPCPERF_KOKKOS_ROOT / HPCPERF_CABANA_ROOT
#                         alternative Kokkos / Cabana installs
#   HIPCXX                path to hipcc (default: hipcc from PATH)

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
BUILD_DIR="$R/build/level2/exampm/$MODEL"
JOBS="${HPCPERF_BUILD_JOBS:-4}"

COMMON=(
    -S "$HERE" -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release
)

check_installs() {
    for d in "$@"; do
        if [ ! -d "$d/lib64/cmake" ] && [ ! -d "$d/lib/cmake" ] && [ ! -d "$d/share/cmake" ]; then
            return 1
        fi
    done
}

case "$BACKEND" in
    CUDA)
        KOKKOS_ROOT="${HPCPERF_KOKKOS_ROOT:-$R/.deps/install/kokkos}"
        CABANA_ROOT="${HPCPERF_CABANA_ROOT:-$R/.deps/install/cabana}"
        if ! check_installs "$KOKKOS_ROOT" "$CABANA_ROOT"; then
            echo "build.sh: $KOKKOS_ROOT and/or $CABANA_ROOT is not a Kokkos/Cabana install -- run ./setup_level2_deps.sh kokkos cabana first" >&2
            exit 1
        fi
        if [ -n "${HPCPERF_CUDA_ARCH:-}" ]; then
            ARCH="$HPCPERF_CUDA_ARCH"
        else
            ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
        fi
        if [ -z "$ARCH" ]; then
            echo "build.sh: could not detect GPU compute capability; set HPCPERF_CUDA_ARCH (e.g. 100)" >&2
            exit 1
        fi
        # The GPU architecture is a property of the Kokkos installation
        # (Kokkos_ARCH_<name>=ON at Kokkos configure time, propagated to every
        # application through nvcc_wrapper); the app cannot override it. Check
        # that the install matches this GPU and say so if it does not.
        case "$ARCH" in
            70) WANT=VOLTA70;;   75) WANT=TURING75;;  80) WANT=AMPERE80;;
            86) WANT=AMPERE86;;  89) WANT=ADA89;;     90) WANT=HOPPER90;;
            100) WANT=BLACKWELL100;; 120) WANT=BLACKWELL120;;
            *) WANT="";;
        esac
        HAVE="$(sed -n 's/^set(Kokkos_ARCH \(.*\))$/\1/p' "$KOKKOS_ROOT"/lib*/cmake/Kokkos/KokkosConfigCommon.cmake 2>/dev/null | head -1)"
        if [ -n "$WANT" ] && [ -n "$HAVE" ] && [ "$WANT" != "$HAVE" ]; then
            echo "build.sh: WARNING: GPU is sm_${ARCH} (Kokkos_ARCH_${WANT}) but $KOKKOS_ROOT was built for Kokkos_ARCH_${HAVE};" >&2
            echo "          rebuild Kokkos + Cabana with ./setup_level2_deps.sh kokkos cabana" >&2
        else
            echo "build.sh: GPU sm_${ARCH}; Kokkos install $KOKKOS_ROOT (Kokkos_ARCH ${HAVE:-unknown}); Cabana install $CABANA_ROOT"
        fi
        # nvcc_wrapper forwards host code to $NVCC_WRAPPER_DEFAULT_COMPILER
        # (hpcperf_env.sh sets it to the conda g++) and device code to nvcc.
        # Upstream requires CMake 3.12, so <Pkg>_ROOT is honoured (CMP0074).
        cmake "${COMMON[@]}" \
            -DCMAKE_CXX_COMPILER="$KOKKOS_ROOT/bin/nvcc_wrapper" \
            -DKokkos_ROOT="$KOKKOS_ROOT" \
            -DCabana_ROOT="$CABANA_ROOT" \
            "$@"
        ;;
    HIP)
        # NOTE: no ROCm/hipcc on the development machine -- untested.
        KOKKOS_ROOT="${HPCPERF_KOKKOS_ROOT:-$R/.deps/install/kokkos-hip}"
        CABANA_ROOT="${HPCPERF_CABANA_ROOT:-$R/.deps/install/cabana-hip}"
        HIPCXX="${HIPCXX:-$(command -v hipcc || echo hipcc)}"
        if ! command -v "$HIPCXX" >/dev/null 2>&1; then
            echo "build.sh: HIP requested but hipcc was not found (no ROCm on this machine); set HIPCXX or install ROCm" >&2
            exit 1
        fi
        if ! check_installs "$KOKKOS_ROOT" "$CABANA_ROOT"; then
            echo "build.sh: HIP requested but $KOKKOS_ROOT and/or $CABANA_ROOT does not exist -- a HIP-enabled Kokkos 5.2.1 (Kokkos_ENABLE_HIP=ON, Kokkos_ARCH_AMD_<gfx>=ON, CXX=hipcc) and a Cabana 0.8.0 (Core+Grid, MPI, HDF5) built against it must be installed there first" >&2
            exit 1
        fi
        cmake "${COMMON[@]}" \
            -DCMAKE_CXX_COMPILER="$HIPCXX" \
            -DKokkos_ROOT="$KOKKOS_ROOT" \
            -DCabana_ROOT="$CABANA_ROOT" \
            "$@"
        ;;
    *)
        echo "usage: $0 [CUDA|HIP] [extra cmake options]" >&2
        exit 2
        ;;
esac

cmake --build "$BUILD_DIR" -j"$JOBS"
echo "Built: $BUILD_DIR/examples/DamBreak and $BUILD_DIR/examples/FreeFall"
