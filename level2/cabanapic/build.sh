#!/usr/bin/env bash
# Build CabanaPIC (Cabana + Kokkos) for the CUDA or HIP programming model.
#
#   ./build.sh [CUDA|HIP] [DECK] [extra cmake -D options...]
#
# Output tree: $R/build/level2/cabanapic/<cuda|hip>/
#   example/cbnpic                     the benchmark binary (deck compiled in)
#   tests/energy_comparison/2stream-em upstream energy-comparison test (validate.sh)
#   tests/decks/custom_init            upstream 30-step smoke test
#
# CabanaPIC is a Kokkos application: the same src/ is the CUDA and the HIP
# variant, the backend is chosen by the Kokkos/Cabana installation it is linked
# against. CUDA uses the project's Kokkos 5.2.1 + Cabana 0.8.0 in
# $R/.deps/install/{kokkos,cabana} (built by setup_level2_deps.sh with
# Serial+OpenMP+CUDA for the GPU present at that time) and compiles through
# Kokkos' nvcc_wrapper. HIP expects a ROCm-enabled Kokkos/Cabana in
# $R/.deps/install/{kokkos-hip,cabana-hip} and hipcc -- neither exists on the
# development machine, so the HIP form is untested.
#
# The input deck is a C++ file compiled into the binary (CabanaPIC has no
# runtime input file). DECK is an absolute path, a path relative to this
# directory (decks/custom_init.cxx) or a bare name (custom_init). Default:
# decks/hpcperf_weibel.cxx -- the upstream Weibel problem with its three size
# parameters overridable at run time (see run.sh).
#
# Environment overrides:
#   HPCPERF_DECK             deck, same forms as the DECK argument
#   HPCPERF_REAL_TYPE        double (default) or float. Upstream's CMake default
#                            is float, but upstream CI validates with double and
#                            the float gold file fails on the GPU (see README).
#   HPCPERF_PARTICLE_DUMP_INTERVAL
#                            0 (default) -- never write the per-step particle and
#                            field dumps "partloc"/"ex1d"; 1 = upstream behaviour
#                            (every step), N = every N steps. Diagnostics only.
#   HPCPERF_CABANAPIC_TESTS  ON (default) or OFF -- also build upstream tests/
#                            (needed by validate.sh; no GTest involved)
#   HPCPERF_CUDA_ARCH        compute capability digits (e.g. 100 for sm_100);
#                            default: detected from nvidia-smi. Only checked
#                            against the architecture baked into the Kokkos
#                            install (Kokkos, not the app, fixes the GPU arch).
#   HPCPERF_BUILD_JOBS       parallel build jobs (default 4)
#   HPCPERF_KOKKOS_ROOT / HPCPERF_CABANA_ROOT
#                            alternative Kokkos / Cabana installs
#   HIPCXX                   path to hipcc (default: hipcc from PATH)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

# Load the project toolchain (conda GCC, system CUDA, CMake/Ninja, OpenMPI).
# Idempotent, so it is safe even if the caller already sourced it.
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true

set -euo pipefail

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
[ $# -gt 0 ] && shift
# A second positional argument that is not a -D option is the deck.
DECK="${HPCPERF_DECK:-decks/hpcperf_weibel.cxx}"
if [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; then
    DECK="$1"; shift
fi
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level2/cabanapic/$MODEL"
JOBS="${HPCPERF_BUILD_JOBS:-4}"
REAL_TYPE="${HPCPERF_REAL_TYPE:-double}"
DUMP_INTERVAL="${HPCPERF_PARTICLE_DUMP_INTERVAL:-0}"
TESTS="${HPCPERF_CABANAPIC_TESTS:-ON}"

# Resolve the deck to an absolute path (CMake compiles it into cbnpic).
case "$DECK" in
    /*) ;;
    *)  if   [ -f "$HERE/$DECK" ];             then DECK="$HERE/$DECK"
        elif [ -f "$HERE/decks/$DECK" ];       then DECK="$HERE/decks/$DECK"
        elif [ -f "$HERE/decks/$DECK.cxx" ];   then DECK="$HERE/decks/$DECK.cxx"
        fi ;;
esac
if [ ! -f "$DECK" ]; then
    echo "build.sh: deck '$DECK' not found (available: $(cd "$HERE/decks" && ls *.cxx | tr '\n' ' '))" >&2
    exit 1
fi
case "$REAL_TYPE" in float|double) ;; *) echo "build.sh: HPCPERF_REAL_TYPE must be float or double" >&2; exit 1;; esac

COMMON=(
    -S "$HERE" -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE=Release
    -DINPUT_DECK="$DECK"
    -DREAL_TYPE="$REAL_TYPE"
    -DPARTICLE_DUMP_INTERVAL="$DUMP_INTERVAL"
    -DENABLE_TESTS="$TESTS"
)

case "$BACKEND" in
    CUDA)
        KOKKOS_ROOT="${HPCPERF_KOKKOS_ROOT:-$R/.deps/install/kokkos}"
        CABANA_ROOT="${HPCPERF_CABANA_ROOT:-$R/.deps/install/cabana}"
        if [ ! -d "$KOKKOS_ROOT/lib64/cmake" ] && [ ! -d "$KOKKOS_ROOT/lib/cmake" ]; then
            echo "build.sh: $KOKKOS_ROOT is not a Kokkos install -- run ./setup_level2_deps.sh kokkos first" >&2
            exit 1
        fi
        if [ ! -f "$CABANA_ROOT/share/cmake/Cabana/CabanaConfig.cmake" ] && [ ! -d "$CABANA_ROOT/lib64/cmake/Cabana" ] && [ ! -d "$CABANA_ROOT/lib/cmake/Cabana" ]; then
            echo "build.sh: $CABANA_ROOT is not a Cabana install -- run ./setup_level2_deps.sh cabana first" >&2
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
            echo "build.sh: GPU sm_${ARCH}; Kokkos install $KOKKOS_ROOT (Kokkos_ARCH ${HAVE:-unknown})"
        fi
        echo "build.sh: deck $DECK; REAL_TYPE=$REAL_TYPE; PARTICLE_DUMP_INTERVAL=$DUMP_INTERVAL; tests $TESTS"
        # nvcc_wrapper forwards host code to $NVCC_WRAPPER_DEFAULT_COMPILER
        # (hpcperf_env.sh sets it to the conda g++) and device code to nvcc.
        # Upstream's cmake_minimum_required(3.9) predates CMP0074, so
        # <Pkg>_ROOT would be ignored; hand the two installs over via
        # CMAKE_PREFIX_PATH instead (hpcperf_env.sh also exports them, plus
        # heFFTe which Cabana's config file requires, there). No
        # -DCMAKE_CXX_STANDARD is needed: Kokkos' exported cxx_std_20 compile
        # feature already raises the effective standard from upstream's 14 to 20.
        cmake "${COMMON[@]}" \
            -DCMAKE_CXX_COMPILER="$KOKKOS_ROOT/bin/nvcc_wrapper" \
            -DCMAKE_PREFIX_PATH="$KOKKOS_ROOT;$CABANA_ROOT" \
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
        for d in "$KOKKOS_ROOT" "$CABANA_ROOT"; do
            if [ ! -d "$d" ]; then
                echo "build.sh: HIP requested but $d does not exist -- a HIP-enabled Kokkos 5.2.1 (Kokkos_ENABLE_HIP=ON, Kokkos_ARCH_AMD_<gfx>=ON, CXX=hipcc) and a Cabana 0.8.0 built against it must be installed there first" >&2
                exit 1
            fi
        done
        echo "build.sh: deck $DECK; REAL_TYPE=$REAL_TYPE; PARTICLE_DUMP_INTERVAL=$DUMP_INTERVAL; tests $TESTS"
        cmake "${COMMON[@]}" \
            -DCMAKE_CXX_COMPILER="$HIPCXX" \
            -DCMAKE_PREFIX_PATH="$KOKKOS_ROOT;$CABANA_ROOT" \
            "$@"
        ;;
    *)
        echo "usage: $0 [CUDA|HIP] [DECK] [extra cmake options]" >&2
        exit 2
        ;;
esac

cmake --build "$BUILD_DIR" -j"$JOBS"
echo "Built: $BUILD_DIR/example/cbnpic"
