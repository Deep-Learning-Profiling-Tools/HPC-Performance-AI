#!/usr/bin/env bash
# Build PIConGPU (alpaka CUDA backend) with upstream's own pic-create / pic-build for every registered input set, plus
# the two libraries the conda environment lacks (Boost program_options, GSL) from the bundled release tarballs.
#
#   ./build.sh [CUDA] [ID ...]        (default CUDA and every registered input set)
#
# Layout (one frozen source tree, generated state isolated per backend profile):
#   source   level3/picongpu/src (frozen source bundle materialized by tools/prepare_benchmark.sh; identity in
#            provenance/; thirdParty/ bundled) -- read ONLY; nothing is fetched, cloned or patched here
#   deps     level3/picongpu/deps/{boost_1_86_0,gsl-2.8}.tar.gz (part of the bundle) -> .deps/level3/picongpu/<profile>/install/{boost,gsl}
#   profile  <cuda> (override HPCPERF_PICONGPU_PROFILE; must name the backend)
#   build    build/level3/picongpu/<profile>/<set>   the input set (pic-create of inputs/<set>/) with its .build directory
#   install  .deps/level3/picongpu/<profile>/install/<set>/bin/picongpu (+ .hpcperf-l3-fingerprint), logs under logs/
#
# A PIConGPU "input set" (inputs/<set>/include/picongpu/param/*.param + cmakeFlags) is compile-time: the physics,
# species, pusher, shapes and precision are template parameters; grid size, step count, periodicity and plugins are
# runtime arguments (inputs.yaml). pic-build = cmake on src/include/picongpu with -DPIC_EXTENSION_PATH=<set>
# (upstream's documented way), here with the CUDA backend for the detected compute capability
# (alpaka_ACC_GPU_CUDA_ENABLE, CUDA_ONLY_MODE), CMAKE_CUDA_HOST_COMPILER = conda g++, Boost_DIR = the built Boost.
# Modification class: A (no upstream file edited; cmake cache entries only).
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
case "$BACKEND" in CUDA) shift $(( $# > 0 ? 1 : 0 ));; HIP) echo "build.sh: HIP: untested here (no ROCm); pic-build -b hip is upstream's path" >&2; exit 1;; *) if [ -d "$HERE/inputs/$1" ]; then BACKEND=CUDA; else echo "usage: $0 [CUDA] [ID ...]" >&2; exit 2; fi;; esac
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
l3_require_materialized "$HERE" || exit 3
SRC="$HERE/src"
[ -x "$SRC/bin/pic-build" ] && [ -f "$SRC/include/picongpu/CMakeLists.txt" ] || { echo "build.sh: $SRC is not a PIConGPU source tree -- run tools/prepare_benchmark.sh level3 picongpu" >&2; exit 3; }
[ -f "$HERE/deps/boost_1_86_0.tar.gz" ] && [ -f "$HERE/deps/gsl-2.8.tar.gz" ] || { echo "build.sh: deps/boost_1_86_0.tar.gz or deps/gsl-2.8.tar.gz missing (part of the bundle)" >&2; exit 3; }
SHA="$(l3_source_commit "$HERE")"; TREE_SHA="$(l3_source_tree_sha "$HERE")"
PROFILE="$(l3_backend_profile PICONGPU "$MODEL")"
l3_paths_profile picongpu "$PROFILE" "$MODEL" || exit 2
JOBS="${HPCPERF_BUILD_JOBS:-16}"
TOOL="$R/tools/inputs/hpcperf_inputs.py"
ARCH="${HPCPERF_CUDA_ARCH:-$(l3_gpu_arch)}"
CXX_HOST="${CXX:-$(command -v g++)}"
IDS=("$@"); [ ${#IDS[@]} -gt 0 ] || read -r -a IDS <<< "$(python3 "$TOOL" list "$HERE" | awk '{print $1}' | paste -sd' ')"
# the registered input SETS (several registered inputs may share one param set)
SETS=""; for id in "${IDS[@]}"; do s="$(python3 "$TOOL" param "$HERE" "$id" paramset)"; case ",$SETS," in *",$s,"*) ;; *) SETS="$SETS${SETS:+,}$s";; esac; done
SETSHA=""; for s in ${SETS//,/ }; do SETSHA="$SETSHA${SETSHA:+,}$s=$(find "$HERE/inputs/$s/include" "$HERE/inputs/$s/cmakeFlags" -type f | sort | xargs cat | sha256sum | cut -c1-12)"; done
CMAKE_OPTS="alpaka_ACC_GPU_CUDA_ENABLE=ON alpaka_ACC_GPU_CUDA_ONLY_MODE=ON alpaka_CUDA_EXPT_EXTENDED_LAMBDA=ON CMAKE_CUDA_ARCHITECTURES=$ARCH CMAKE_BUILD_TYPE=Release"
FP="$(l3_fingerprint_text picongpu "$SHA" "$MODEL" "boost=1.86.0(program_options) gsl=2.8" "$CMAKE_OPTS sets[$SETSHA]" "runtime(1 rank per GPU, alpaka CUDA)")"
l3_fingerprint_check "$L3_INSTALL" "$FP" || exit 1

echo "# PIConGPU $BACKEND profile=$PROFILE: upstream $SHA (frozen source tree $TREE_SHA), sm_$ARCH, host compiler $CXX_HOST, MPI $(mpirun --version 2>/dev/null | head -1), input sets [$SETS]"
mkdir -p "$L3_BUILD" "$L3_INSTALL" "$L3_LOGS"
t0=$(date +%s)
# [1] Boost (program_options + atomic; headers) and GSL from the bundled release tarballs
if [ ! -f "$L3_INSTALL/boost/lib/libboost_program_options.a" ]; then
    rm -rf "$L3_BUILD/boost-src"; mkdir -p "$L3_BUILD/boost-src"
    tar -xzf "$HERE/deps/boost_1_86_0.tar.gz" -C "$L3_BUILD/boost-src" --strip-components=1
    ( cd "$L3_BUILD/boost-src" && ./bootstrap.sh --with-libraries=program_options,atomic --prefix="$L3_INSTALL/boost" > "$L3_LOGS/boost-bootstrap.log" 2>&1 \
      && ./b2 -j "$JOBS" install > "$L3_LOGS/boost-build.log" 2>&1 ) || { tail -20 "$L3_LOGS/boost-build.log"; echo "build.sh: Boost build failed" >&2; exit 1; }
fi
if [ ! -f "$L3_INSTALL/gsl/lib/libgsl.a" ]; then
    rm -rf "$L3_BUILD/gsl-src"; mkdir -p "$L3_BUILD/gsl-src"
    tar -xzf "$HERE/deps/gsl-2.8.tar.gz" -C "$L3_BUILD/gsl-src" --strip-components=1
    ( cd "$L3_BUILD/gsl-src" && ./configure --prefix="$L3_INSTALL/gsl" --disable-shared > "$L3_LOGS/gsl-configure.log" 2>&1 \
      && make -j "$JOBS" > "$L3_LOGS/gsl-build.log" 2>&1 && make install > "$L3_LOGS/gsl-install.log" 2>&1 ) || { tail -20 "$L3_LOGS/gsl-build.log"; echo "build.sh: GSL build failed" >&2; exit 1; }
fi
# [2] every registered input set: pic-create (upstream's merge of the default parameters with the set's) + pic-build
export PICSRC="$SRC" PIC_BACKEND="cuda:$ARCH" PIC_EXAMPLES="$SRC/share/picongpu/examples"
export CMAKE_PREFIX_PATH="$L3_INSTALL/boost:$L3_INSTALL/gsl${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
for s in ${SETS//,/ }; do
    B="$L3_BUILD/$s"; rm -rf "$B"
    "$SRC/bin/pic-create" "$HERE/inputs/$s" "$B" > "$L3_LOGS/pic-create-$s.log" 2>&1 || { tail -5 "$L3_LOGS/pic-create-$s.log"; echo "build.sh: pic-create $s failed" >&2; exit 1; }
    ( cd "$B" && "$SRC/bin/pic-build" -b "cuda:$ARCH" -j "$JOBS" \
        -c "-DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_HOST_COMPILER=$CXX_HOST -DCMAKE_CXX_COMPILER=$CXX_HOST -DBoost_DIR=$L3_INSTALL/boost/lib/cmake/Boost-1.86.0 -DGSL_ROOT_DIR=$L3_INSTALL/gsl -Dalpaka_CUDA_EXPT_EXTENDED_LAMBDA=ON" \
        > "$L3_LOGS/pic-build-$s.log" 2>&1 ) || { tail -30 "$L3_LOGS/pic-build-$s.log"; echo "build.sh: pic-build $s failed (log: $L3_LOGS/pic-build-$s.log)" >&2; exit 1; }
    [ -x "$B/bin/picongpu" ] || { echo "build.sh: $B/bin/picongpu not produced" >&2; exit 1; }
    mkdir -p "$L3_INSTALL/$s/bin"; cp -f "$B/bin/picongpu" "$L3_INSTALL/$s/bin/picongpu"
    find "$HERE/inputs/$s/include" "$HERE/inputs/$s/cmakeFlags" -type f | sort | xargs cat | sha256sum | cut -d' ' -f1 > "$L3_INSTALL/$s/bin/picongpu.paramset.sha256"
done
l3_fingerprint_write "$L3_INSTALL" "$FP"
echo "# built in $(( $(date +%s)-t0 )) s: $L3_INSTALL/{${SETS}}/bin/picongpu (Boost + GSL under $L3_INSTALL)"
