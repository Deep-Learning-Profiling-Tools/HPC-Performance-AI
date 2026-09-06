#!/usr/bin/env bash
# Build QMCPACK v4.4.0 with upstream's recommended NVIDIA GPU configuration:
# LLVM/Clang OpenMP target offload + CUDA (QMC_GPU="openmp;cuda", QMC_GPU_ARCHS=sm_100),
# using the private LLVM 23.1.0 toolchain from toolchain/build_llvm.sh (probe:
# toolchain/probe_offload.sh must have PASSED first -- checked here).
#
#   ./build.sh                       HPCPERF_BUILD_JOBS=N (default 32)
#   HPCPERF_QMCPACK_COMPLEX=1        also build the complex-valued executable
#
# Stages (profile clang231-cuda132-offload, .deps/level3/qmcpack/<profile>/):
#   openblas 0.3.30 (gcc/gfortran, single-threaded)  -> install/openblas
#   HDF5 1.14.5 (parallel, built with the clang MPI wrappers) -> install/hdf5
#   Boost 1.90.0 headers                              -> install/boost-1.90.0
#   QMCPACK real (+complex)                           -> build/level3/qmcpack/<profile>/{real,complex}, install/qmcpack
# Compilers: clang/clang++ 23.1.0 through the conda Open MPI wrappers (OMPI_CC/OMPI_CXX);
# CUDA parts through nvcc 13.2 with clang as host compiler (QMCPACK adds
# --allow-unsupported-compiler itself); FFTW3 and libxml2 from the project conda env
# / system (header/library only); BLAS/LAPACK = private OpenBLAS. Full precision,
# MPI on, unit tests built. Modification class: A/C (build options, environment).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
l3_isolate_build_env
l3_clean_conda_build_env
PROFILE="${HPCPERF_QMCPACK_PROFILE:-clang231-cuda132-offload}"
l3_paths_profile qmcpack "$PROFILE"
SRC="$R/_upstream/level3/qmcpack"; DL="$R/.deps/level3/qmcpack/downloads"
[ -f "$SRC/CMakeLists.txt" ] || { echo "build.sh: run $HERE/fetch.sh" >&2; exit 1; }
LLVM="$L3_INSTALL/llvm"; CLANG="$LLVM/bin/clang"; CLANGXX="$LLVM/bin/clang++"
[ -x "$CLANGXX" ] || { echo "build.sh: private LLVM missing ($LLVM) -- run toolchain/build_llvm.sh" >&2; exit 1; }
[ -f "$LLVM/OFFLOAD_PROBE.txt" ] && /usr/bin/grep -q '^PASS' "$LLVM/OFFLOAD_PROBE.txt" || { echo "build.sh: offload probe has not PASSED (toolchain/probe_offload.sh) -- refusing to build QMCPACK on an unverified offload toolchain" >&2; exit 1; }
SHA="$(git -C "$SRC" rev-parse HEAD)"
ARCH="${HPCPERF_CUDA_ARCH:-$(l3_gpu_arch)}"; CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
JOBS="${HPCPERF_BUILD_JOBS:-32}"; INST="$L3_INSTALL"; BLD="$L3_BUILD_DEPS"
export OMPI_CC="$CLANG" OMPI_CXX="$CLANGXX" OMPI_FC=/usr/bin/gfortran
export LD_LIBRARY_PATH="$LLVM/lib/x86_64-unknown-linux-gnu:$LLVM/lib:${LD_LIBRARY_PATH:-}"
unset CMAKE_GENERATOR CUDAARCHS
CLANG_V="$("$CLANG" --version | head -1 | /usr/bin/grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
OMPI_V="$(mpirun --version | head -1 | /usr/bin/grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
DEPS="llvm=$CLANG_V(offload sm_$ARCH, $(cat "$LLVM/OFFLOAD_PROBE.txt" | cut -c1-30)) hdf5=1.14.5(parallel) boost=1.90.0(headers) openblas=0.3.30 fftw3=conda libxml2=system openmpi=$OMPI_V cuda=$(l3_cuda_version) profile=$PROFILE"
OPTS="QMC_GPU=openmp;cuda QMC_GPU_ARCHS=sm_$ARCH QMC_MPI=ON QMC_COMPLEX=OFF(+ON on request) QMC_MIXED_PRECISION=OFF ENABLE_PHDF5=ON BUILD_UNIT_TESTS=ON CMAKE_BUILD_TYPE=Release CXX=mpicxx(clang++)"
FP="$(l3_fingerprint_text qmcpack "$SHA" cuda "$DEPS" "$OPTS" "not-used(walker-parallel; MPI on host buffers)")"
l3_fingerprint_check "$INST" "$FP" || exit 1
echo "# QMCPACK profile=$PROFILE: qmcpack $SHA (v4.4.0), clang $CLANG_V, $(mpirun --version | head -1), CUDA $(l3_cuda_version) sm_$ARCH, -j$JOBS; expected 30-60 min + deps ~15 min"
T0=$(date +%s)
run() { local log=$1; shift; "$@" > "$L3_LOGS/$log" 2>&1 || { tail -40 "$L3_LOGS/$log"; echo "build.sh: FAILED: $* (log $L3_LOGS/$log)" >&2; exit 1; }; }
untar() { # untar <tarball> <dest-parent> -> prints the extracted top-level directory (tolerates a leading ./ entry)
    local tb=$1 dst=$2 top; mkdir -p "$dst"
    top="$(tar -tf "$tb" | sed 's|^\./||' | /usr/bin/grep -v '^$' | head -1 | cut -d/ -f1)"
    [ -n "$top" ] && [ "$top" != . ] || { echo "untar: cannot determine top-level directory of $tb" >&2; return 1; }
    [ -d "$dst/$top" ] || tar -xf "$tb" -C "$dst"; echo "$dst/$top"; }

if [ ! -f "$INST/openblas/.hpcperf-stage-done" ]; then
    d="$(untar "$DL/OpenBLAS-0.3.30.tar.gz" "$L3_SRC")"
    run openblas-make.log make -C "$d" -j "$JOBS" USE_THREAD=0 USE_OPENMP=0 DYNAMIC_ARCH=0 NO_AFFINITY=1 TARGET="${HPCPERF_OPENBLAS_TARGET:-SAPPHIRERAPIDS}" CC=/usr/bin/gcc FC=/usr/bin/gfortran
    run openblas-install.log make -C "$d" PREFIX="$INST/openblas" install
    touch "$INST/openblas/.hpcperf-stage-done"
fi
BLAS="$INST/openblas/lib/libopenblas.so"
if [ ! -f "$INST/hdf5/.hpcperf-stage-done" ]; then
    d="$(untar "$DL/hdf5-1.14.5.tar.gz" "$L3_SRC")"; mkdir -p "$BLD/hdf5"
    run hdf5-configure.log cmake -S "$d" -B "$BLD/hdf5" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$(command -v mpicc)" \
        -DHDF5_ENABLE_PARALLEL=ON -DHDF5_BUILD_CPP_LIB=OFF -DHDF5_BUILD_FORTRAN=OFF -DHDF5_BUILD_JAVA=OFF -DHDF5_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
        -DHDF5_BUILD_TOOLS=ON -DBUILD_SHARED_LIBS=ON -DHDF5_ENABLE_Z_LIB_SUPPORT=ON -DHDF5_ENABLE_SZIP_SUPPORT=OFF -DCMAKE_INSTALL_PREFIX="$INST/hdf5" \
        "-DCMAKE_IGNORE_PATH=/usr/lib64/cmake/ZLIB;/lib64/cmake/ZLIB"   # the node's zlib-ng-compat CMake package references a libz.a that is not installed; use FindZLIB (/usr/lib64/libz.so + /usr/include/zlib.h)
    run hdf5-build.log cmake --build "$BLD/hdf5" -j "$JOBS"; run hdf5-install.log cmake --install "$BLD/hdf5"
    touch "$INST/hdf5/.hpcperf-stage-done"
fi
if [ ! -f "$INST/boost-1.90.0/.hpcperf-stage-done" ]; then
    tar -xf "$DL/boost-1.90.0-b2-nodocs.tar.xz" -C "$INST" boost-1.90.0/boost boost-1.90.0/LICENSE_1_0.txt 2>/dev/null || tar -xf "$DL/boost-1.90.0-b2-nodocs.tar.xz" -C "$INST"
    [ -f "$INST/boost-1.90.0/boost/version.hpp" ] || { echo "build.sh: Boost headers not found after extraction" >&2; exit 1; }
    touch "$INST/boost-1.90.0/.hpcperf-stage-done"
fi
T1=$(date +%s)
build_qmc() { # build_qmc <real|complex>
    local kind=$1 cplx=OFF; [ "$kind" = complex ] && cplx=ON
    mkdir -p "$L3_BUILD/$kind"
    run "qmcpack-$kind-configure.log" cmake -S "$SRC" -B "$L3_BUILD/$kind" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$(command -v mpicc)" -DCMAKE_CXX_COMPILER="$(command -v mpicxx)" "-DCMAKE_CUDA_COMPILER=$CUDA_ROOT/bin/nvcc" \
        -DQMC_GPU="openmp;cuda" "-DQMC_GPU_ARCHS=sm_$ARCH" -DQMC_MPI=ON -DQMC_COMPLEX="$cplx" -DQMC_MIXED_PRECISION=OFF \
        -DENABLE_PHDF5=ON "-DHDF5_ROOT=$INST/hdf5" "-DBOOST_ROOT=$INST/boost-1.90.0" \
        "-DBLAS_LIBRARIES=$BLAS" "-DLAPACK_LIBRARIES=$BLAS" -DBUILD_UNIT_TESTS=ON -DQMC_GPU_VISIBILITY_VARIABLE=CUDA_VISIBLE_DEVICES \
        "-DCMAKE_INSTALL_PREFIX=$INST/qmcpack-$kind"
    run "qmcpack-$kind-build.log" cmake --build "$L3_BUILD/$kind" -j "$JOBS"
    run "qmcpack-$kind-install.log" cmake --install "$L3_BUILD/$kind"
    local exe; exe="$(ls "$INST/qmcpack-$kind"/bin/qmcpack* | head -1)"
    local archs; archs="$(cuobjdump --list-elf "$exe" 2>/dev/null | /usr/bin/grep -o 'sm_[0-9]*' | sort -u | paste -sd,)"
    echo "qmcpack_$kind exe=$exe sha256=$(l3_sha_file "$exe") cuobjdump_archs=${archs:-none} offload_libs=$(ldd "$exe" | /usr/bin/grep -oE 'lib(omptarget|omp|cudart|cublas|cusolver)[^ ]*' | sort -u | paste -sd,)" >> "$INST/BUILD_INFO.txt"
    [ -n "$archs" ] && [ "$archs" != "sm_$ARCH" ] && { echo "build.sh: $exe embeds '$archs' != sm_$ARCH" >&2; exit 1; }
    return 0
}
: > "$INST/BUILD_INFO.txt"
build_qmc real
[ -n "${HPCPERF_QMCPACK_COMPLEX:-}" ] && build_qmc complex
T2=$(date +%s)
l3_fingerprint_write "$INST" "$FP"
{ echo "profile=$PROFILE qmcpack=$SHA utc=$(date -u +%FT%TZ) jobs=$JOBS seconds: deps=$((T1 - T0)) qmcpack=$((T2 - T1))"; echo "deps: $DEPS"; echo "options: $OPTS"; } >> "$INST/BUILD_INFO.txt"
cat "$INST/BUILD_INFO.txt"; echo "# QMCPACK built in $((T2 - T0)) s"
