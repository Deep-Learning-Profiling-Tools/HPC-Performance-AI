#!/usr/bin/env bash
# Build HACCabanaPM (pmkokkos: Cabana + Kokkos + heFFTe + parallel HDF5) for
# the CUDA or HIP programming model.
#
#   ./build.sh [CUDA|HIP] [extra cmake -D options...]
#
# Output tree: $R/build/level2/haccabanapm/<cuda|hip>/{pm_ic,pm_run,snapshot_inspect,snapshot_check,pm_single_kick}
#
# HACCabanaPM is a Kokkos/Cabana application: the same source tree is the CUDA
# and the HIP variant, the backend is chosen by the Kokkos + Cabana installs it
# is linked against. CUDA uses the project's Kokkos 5.2.1 and Cabana 0.8.0
# (Core + Grid, MPI, heFFTe 2.4.1 FFTW+cuFFT, HDF5) in
# $R/.deps/install/{kokkos,cabana,heffte} (built by setup_level2_deps.sh with
# Serial+OpenMP+CUDA for the GPU present at that time) and compiles through
# Kokkos' nvcc_wrapper. HIP expects a ROCm-enabled Kokkos/Cabana/heFFTe in
# $R/.deps/install/{kokkos-hip,cabana-hip,heffte-hip} and hipcc -- none of which
# exists on the development machine, so the HIP form is untested.
#
# Upstream's GoogleTest suite is off by default (-DPMKOKKOS_ENABLE_TESTS=OFF;
# the conda environment has no GTest). Set HPCPERF_HACCABANAPM_TESTS=ON to
# build it if GTest is available.
#
# Environment overrides:
#   HPCPERF_CUDA_ARCH          compute capability digits (e.g. 100 for sm_100);
#                              default: detected from nvidia-smi. Only checked
#                              against the architecture baked into the Kokkos
#                              install (Kokkos, not the app, fixes the GPU arch).
#   HPCPERF_HACCABANAPM_TESTS  ON/OFF (default OFF) -> PMKOKKOS_ENABLE_TESTS
#   HPCPERF_BUILD_JOBS         parallel build jobs (default 4)
#   HPCPERF_KOKKOS_ROOT / HPCPERF_CABANA_ROOT / HPCPERF_HEFFTE_ROOT
#                              alternative Kokkos / Cabana / heFFTe installs
#   HIPCXX                     path to hipcc (default: hipcc from PATH)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

# Load the project toolchain (conda GCC, system CUDA, CMake/Ninja, OpenMPI,
# HDF5, FFTW). Idempotent, so it is safe even if the caller already sourced it.
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true

set -euo pipefail

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
[ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level2/haccabanapm/$MODEL"
TESTS="${HPCPERF_HACCABANAPM_TESTS:-OFF}"
JOBS="${HPCPERF_BUILD_JOBS:-4}"

COMMON=(
    -S "$HERE" -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release
    -DPMKOKKOS_ENABLE_TESTS="$TESTS"
)

# Kokkos/heFFTe install their CMake package under lib*/cmake, the header-only
# Cabana under share/cmake.
check_installs() {
    for d in "$@"; do
        if [ ! -d "$d/lib64/cmake" ] && [ ! -d "$d/lib/cmake" ] && [ ! -d "$d/share/cmake" ]; then
            echo "build.sh: $d is not a CMake package install ($BACKEND needs Kokkos, Cabana and heFFTe there) -- run ./setup_level2_deps.sh kokkos heffte cabana first" >&2
            exit 1
        fi
    done
}

case "$BACKEND" in
    CUDA)
        KOKKOS_ROOT="${HPCPERF_KOKKOS_ROOT:-$R/.deps/install/kokkos}"
        CABANA_ROOT="${HPCPERF_CABANA_ROOT:-$R/.deps/install/cabana}"
        HEFFTE_ROOT="${HPCPERF_HEFFTE_ROOT:-$R/.deps/install/heffte}"
        check_installs "$KOKKOS_ROOT" "$CABANA_ROOT" "$HEFFTE_ROOT"
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
            echo "          rebuild Kokkos, heFFTe and Cabana with ./setup_level2_deps.sh kokkos heffte cabana" >&2
        else
            echo "build.sh: GPU sm_${ARCH}; Kokkos install $KOKKOS_ROOT (Kokkos_ARCH ${HAVE:-unknown})"
        fi
        # nvcc_wrapper forwards host code to $NVCC_WRAPPER_DEFAULT_COMPILER
        # (hpcperf_env.sh sets it to the conda g++) and device code to nvcc.
        # Cabana's config pulls in Kokkos, MPI, heFFTe and HDF5 transitively.
        cmake "${COMMON[@]}" \
            -DCMAKE_CXX_COMPILER="$KOKKOS_ROOT/bin/nvcc_wrapper" \
            -DCMAKE_PREFIX_PATH="$KOKKOS_ROOT;$CABANA_ROOT;$HEFFTE_ROOT" \
            "$@"
        ;;
    HIP)
        # NOTE: no ROCm/hipcc on the development machine -- untested.
        KOKKOS_ROOT="${HPCPERF_KOKKOS_ROOT:-$R/.deps/install/kokkos-hip}"
        CABANA_ROOT="${HPCPERF_CABANA_ROOT:-$R/.deps/install/cabana-hip}"
        HEFFTE_ROOT="${HPCPERF_HEFFTE_ROOT:-$R/.deps/install/heffte-hip}"
        HIPCXX="${HIPCXX:-$(command -v hipcc || echo hipcc)}"
        if ! command -v "$HIPCXX" >/dev/null 2>&1; then
            echo "build.sh: HIP requested but hipcc was not found (no ROCm on this machine); set HIPCXX or install ROCm" >&2
            exit 1
        fi
        for d in "$KOKKOS_ROOT" "$CABANA_ROOT" "$HEFFTE_ROOT"; do
            if [ ! -d "$d/lib64/cmake" ] && [ ! -d "$d/lib/cmake" ] && [ ! -d "$d/share/cmake" ]; then
                echo "build.sh: HIP requested but $d does not exist -- a HIP-enabled Kokkos 5.2.1 (Kokkos_ENABLE_HIP=ON, Kokkos_ARCH_AMD_<gfx>=ON, CXX=hipcc), heFFTe (Heffte_ENABLE_ROCM=ON) and Cabana 0.8.0 (Grid + heFFTe + HDF5) must be installed there first" >&2
                exit 1
            fi
        done
        cmake "${COMMON[@]}" \
            -DCMAKE_CXX_COMPILER="$HIPCXX" \
            -DCMAKE_PREFIX_PATH="$KOKKOS_ROOT;$CABANA_ROOT;$HEFFTE_ROOT" \
            "$@"
        ;;
    *)
        echo "usage: $0 [CUDA|HIP] [extra cmake options]" >&2
        exit 2
        ;;
esac

cmake --build "$BUILD_DIR" -j"$JOBS"
echo "Built: $BUILD_DIR/{pm_ic,pm_run,snapshot_inspect,snapshot_check,pm_single_kick}"
