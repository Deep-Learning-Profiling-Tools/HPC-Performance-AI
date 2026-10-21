#!/usr/bin/env bash
# Build GIZMO (gpu_kokkos_unified branch: Kokkos CUDA offload of the gravity tree, density/gradient/hydro
# loops and cooling) with upstream's GNU Makefile, once per registered Config.sh: GIZMO's physics selection
# is the compile-time Config.sh, so every registered problem (inputs/<config>/Config.sh, the `config` of an
# input in inputs.yaml) is a separate executable GIZMO-<config>; inputs that register the same Config.sh
# with another parameter file (a shorter integration of the same problem) share it.
#
#   ./build.sh [CUDA] [ID ...]        (default CUDA and the configurations of every registered input)
#
# Layout (one frozen source tree, generated state isolated per backend profile):
#   source   level3/gizmo/src (frozen source bundle materialized by tools/prepare_benchmark.sh; identity in
#            provenance/; backend-independent) -- read ONLY; nothing is fetched, cloned or patched here
#   profile  <cuda> (override HPCPERF_GIZMO_PROFILE; must name the backend)
#   build    build/level3/gizmo/<profile>/<config>/src   a copy of src/ with inputs/<config>/Config.sh as its
#            Config.sh (upstream's Makefile writes objects and the generated headers next to the sources)
#   install  .deps/level3/gizmo/<profile>/install/bin/GIZMO-<config> (+ .hpcperf-l3-fingerprint), logs under
#            .deps/level3/gizmo/<profile>/logs
#
# Recipe = upstream's "Vista" (NVIDIA GPU, nvcc_wrapper) system block of src/Makefile expressed as make
# command-line variables (SYSTYPE=HPCPERF matches no block, so the defaults are replaced by these): conda
# GCC 13.3 through mpicc/mpicxx (-std=c++20: Kokkos 5.2 needs C++20), OpenMP, the repository's Kokkos install
# (.deps/install/kokkos, CUDA sm_<arch> + OpenMP backends; nvcc_wrapper with NVCC_WRAPPER_DEFAULT_COMPILER=
# mpicxx, --expt-relaxed-constexpr --expt-extended-lambda), conda HDF5 and FFTW, system zlib (-idirafter
# /usr/include: a plain -I shadows glibc's math.h for CCCL), -L/usr/lib64 for libcuda/libz, PERL=/usr/bin/perl
# (the conda environment ships no perl). Modification class: A for the build (no upstream file edited here;
# the two source patches of the frozen bundle are provenance/patch_series.txt).
#
# Environment overrides:
#   HPCPERF_CUDA_ARCH   numeric compute capability (default: detected)
#   HPCPERF_BUILD_JOBS  parallel jobs (default 16)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
l3_isolate_build_env
l3_clean_conda_build_env

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
case "$BACKEND" in CUDA) shift $(( $# > 0 ? 1 : 0 ));; HIP) echo "build.sh: HIP: untested here (no ROCm); the Kokkos HIP recipe is upstream's Frontier block" >&2; exit 1;; *) if [ -d "$HERE/inputs/$1" ]; then BACKEND=CUDA; else echo "usage: $0 [CUDA] [ID ...]" >&2; exit 2; fi;; esac
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
l3_require_materialized "$HERE" || exit 3
SRC="$HERE/src"
[ -f "$SRC/src/Makefile" ] && [ -f "$SRC/src/core/run.cc" ] || { echo "build.sh: $SRC is not a GIZMO source tree -- run tools/prepare_benchmark.sh level3 gizmo" >&2; exit 3; }
SHA="$(l3_source_commit "$HERE")"; TREE_SHA="$(l3_source_tree_sha "$HERE")"
PROFILE="$(l3_backend_profile GIZMO "$MODEL")"
l3_paths_profile gizmo "$PROFILE" "$MODEL" || exit 2
JOBS="${HPCPERF_BUILD_JOBS:-16}"
TOOL="$R/tools/inputs/hpcperf_inputs.py"
ARCH="${HPCPERF_CUDA_ARCH:-$(l3_gpu_arch)}"
KP="$R/.deps/install/kokkos"
[ -x "$KP/bin/nvcc_wrapper" ] && [ -f "$KP/lib64/libkokkoscore.a" ] || { echo "build.sh: the repository Kokkos install $KP (bin/nvcc_wrapper, lib64/libkokkoscore.a) is missing -- build the Level 2 Kokkos dependency first" >&2; exit 3; }
KVER="$(sed -n 's/^#define KOKKOS_VERSION \([0-9]*\)$/\1/p' "$KP/include/KokkosCore_config.h" | awk '{printf "%d.%d.%d", $1/10000, ($1/100)%100, $1%100}')"
CUDA_ROOT="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
CONDA="${CONDA_PREFIX:?build.sh: CONDA_PREFIX unset (source hpcperf_env.sh)}"
command -v mpicxx >/dev/null && command -v mpicc >/dev/null || { echo "build.sh: mpicc/mpicxx not on PATH" >&2; exit 3; }
[ -x /usr/bin/perl ] || { echo "build.sh: /usr/bin/perl missing (upstream's Makefile generates headers with perl)" >&2; exit 3; }
export NVCC_WRAPPER_DEFAULT_COMPILER=mpicxx

IDS=("$@"); [ ${#IDS[@]} -gt 0 ] || read -r -a IDS <<< "$(python3 "$TOOL" list "$HERE" | awk '{print $1}' | paste -sd' ')"
# the configurations to build = the distinct `config` directories of the selected inputs (inputs/<config>/Config.sh)
CFGS=()
for id in "${IDS[@]}"; do
    rel="$(python3 "$TOOL" param "$HERE" "$id" config)" || { echo "build.sh: $id is not a registered input (inputs.yaml)" >&2; exit 2; }
    c="$(basename "$(dirname "$rel")")"; [ -f "$HERE/inputs/$c/Config.sh" ] && [ "$rel" = "inputs/$c/Config.sh" ] || { echo "build.sh: $id: config $rel is not inputs/<config>/Config.sh" >&2; exit 2; }
    [[ " ${CFGS[*]-} " == *" $c "* ]] || CFGS+=("$c")
done
CONFIGS=""; for c in "${CFGS[@]}"; do CONFIGS="$CONFIGS${CONFIGS:+,}$c=$(l3_sha_file "$HERE/inputs/$c/Config.sh" | cut -c1-12)"; done
MAKEVARS="SYSTYPE=HPCPERF CXX=mpicxx-std=c++20 OPTIMIZE=-O2-fopenmp kokkos=$KVER(sm_$ARCH,OpenMP) nvcc_wrapper hdf5+fftw=conda"
FP="$(l3_fingerprint_text gizmo "$SHA" "$MODEL" "kokkos=$KVER" "$MAKEVARS configs[$CONFIGS]" "runtime(1 rank per GPU, Kokkos CudaUVM arena)")"
l3_fingerprint_check "$L3_INSTALL" "$FP" || exit 1

echo "# GIZMO $BACKEND profile=$PROFILE: upstream $SHA (frozen source tree $TREE_SHA), Kokkos $KVER ($KP), sm_$ARCH, MPI $(mpirun --version 2>/dev/null | head -1), inputs [${IDS[*]}] -> configurations [${CFGS[*]}]"
mkdir -p "$L3_BUILD" "$L3_INSTALL/bin" "$L3_LOGS"
t0=$(date +%s)
for id in "${CFGS[@]}"; do
    CFG="$HERE/inputs/$id/Config.sh"
    B="$L3_BUILD/$id/src"
    rm -rf "$B"; mkdir -p "$B"
    rsync -a --exclude='.git' "$SRC/src/" "$B/"        # upstream builds in place: a private copy per configuration
    cp -f "$CFG" "$B/Config.sh"
    echo "# $id: Config.sh $(l3_sha_file "$CFG" | cut -c1-12) -> $B"
    ( cd "$B" && make -j "$JOBS" \
        SYSTYPE=HPCPERF CC=mpicc CXX="mpicxx -std=c++20" FC=mpif90 \
        OPTIMIZE="-O2 -Wall -fopenmp -idirafter /usr/include" \
        KOKKOS_PATH="$KP" KOKKOS_CPPFLAGS="-I$KP/include" \
        KOKKOS_CXXFLAGS="--expt-relaxed-constexpr --expt-extended-lambda -arch=sm_$ARCH" \
        KOKKOS_LDFLAGS="-L$KP/lib64 -L$CUDA_ROOT/lib64 -Wl,-rpath,$CUDA_ROOT/lib64 -L/usr/lib64" \
        KOKKOS_LIBS="-lkokkoscontainers -lkokkoscore -lkokkossimd -lcudart -lcuda" \
        GPU_CXX="$KP/bin/nvcc_wrapper --std=c++20" \
        GPU_CFLAGS="-I$KP/include --expt-relaxed-constexpr --expt-extended-lambda -arch=sm_$ARCH" \
        GPU_LDFLAGS="-L$KP/lib64 -L$CUDA_ROOT/lib64 -Wl,-rpath,$CUDA_ROOT/lib64 -L/usr/lib64" \
        FFTW_INCL="-I$CONDA/include" FFTW_LIBS="-L$CONDA/lib -Wl,-rpath,$CONDA/lib" \
        HDF5INCL="-I$CONDA/include -DH5_USE_16_API" HDF5LIB="-L$CONDA/lib -Wl,-rpath,$CONDA/lib -lhdf5 -lz" \
        MPICHLIB= PERL=/usr/bin/perl > "$L3_LOGS/build-$id-$MODEL.log" 2>&1 ) \
        || { tail -30 "$L3_LOGS/build-$id-$MODEL.log"; echo "build.sh: $id failed (log: $L3_LOGS/build-$id-$MODEL.log)" >&2; exit 1; }
    [ -x "$B/GIZMO" ] || { echo "build.sh: $B/GIZMO not produced" >&2; exit 1; }
    cp -f "$B/GIZMO" "$L3_INSTALL/bin/GIZMO-$id"
    l3_sha_file "$CFG" > "$L3_INSTALL/bin/GIZMO-$id.config.sha256"
done
l3_fingerprint_write "$L3_INSTALL" "$FP"
echo "# built in $(( $(date +%s)-t0 )) s: $L3_INSTALL/bin/GIZMO-{${CFGS[*]// /,}}"
