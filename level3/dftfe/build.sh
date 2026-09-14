#!/usr/bin/env bash
# Build DFT-FE 1.2.0 (real-valued executable; complex optional) with its full
# dependency stack, following upstream's install_DFTFE recipe (frontierDevelop
# dftfe2.sh / setupUser.sh) adapted to this node: system GCC 14.2.1 for
# C/C++/Fortran, conda Open MPI 5.0.10 (wrappers redirected to that GCC), CUDA
# 13.2 sm_100. Everything lives in one profile tree; each stage has a done-marker
# so the script can be re-run (and re-parallelised) without redoing finished work.
#
#   ./build.sh                          HPCPERF_BUILD_JOBS=N (default 16)
#   HPCPERF_DFTFE_COMPLEX=1             also build the complex-valued executable (k-points)
#   HPCPERF_DFTFE_ELPA_GPU=OFF          diagnostic variant: ELPA without NVIDIA kernels
#                                       (profile suffix -elpacpu; NEVER a silent substitute --
#                                       the default profile builds the GPU ELPA)
#
# Stages  (install/<pkg>, build/<pkg>, logs/<pkg>-*.log):
#   openblas 0.3.30 (single-threaded) -> scalapack 2.2.2 -> libxc 7.0.0 -> spglib -> alglib 4.06.0
#   -> p4est 2.8.7 (dftfe's p4est-setup.sh: FAST+DEBUG) -> kokkos 4.6.00 (Serial, for deal.II)
#   -> deal.II 9.6.2 (MPI, p4est, 64-bit indices, LAPACK=openblas, bundled boost, no TBB/taskflow)
#   -> ELPA 2026.02.001 (--enable-nvidia-gpu-kernels --with-NVIDIA-GPU-compute-capability=sm_100,
#      test programs built for elpa_probe.sh) -> DFT-FE (WITH_GPU=ON GPU_LANG=cuda GPU_VENDOR=nvidia,
#      CMAKE_CUDA_ARCHITECTURES=100, WITH_DCCL=OFF (no NCCL), WITH_GPU_AWARE_MPI=OFF, USE_64BIT_INT=ON)
# Modification classes: A (build options) + C (compiler/MPI wrapper environment). No source patched.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
l3_isolate_build_env
l3_clean_conda_build_env   # system GCC 14 build: drop conda's CFLAGS/LDFLAGS/AR/...
export CC=/usr/bin/gcc CXX=/usr/bin/g++ FC=/usr/bin/gfortran F77=/usr/bin/gfortran F90=/usr/bin/gfortran
export OMPI_CC=/usr/bin/gcc OMPI_CXX=/usr/bin/g++ OMPI_FC=/usr/bin/gfortran CUDAHOSTCXX=/usr/bin/g++
unset CMAKE_GENERATOR   # autotools/ExternalProject-style stages below drive make/ninja explicitly
for c in mpicc mpicxx mpifort mpirun cmake ninja nvcc python3; do command -v $c >/dev/null || { echo "build.sh: $c not in PATH" >&2; exit 1; }; done
# Sources come ONLY from the frozen bundle materialized here (tools/prepare_benchmark.sh): src/ = DFT-FE
# 1.2.0 with the isnan patch already applied, deps/<pkg>/<tarball or checkout> = the pinned dependency
# sources. Nothing is fetched, cloned or patched by this script.
l3_require_materialized "$HERE" || exit 3
SRC="$HERE/src"; DL="$HERE/deps"
# deal.II 9.6.2: the newest deal.II whose API release 1.2.0 still compiles against (9.7 removed Utilities::MPI::create_group,
# Triangulation::load(name, autopartition) and VtkFlags::ZlibCompressionLevel, all used by 1.2.0; 9.5.2 is the release's own pin;
# 9.7.1 is what the current install_DFTFE recipe pairs with the *develop* branch) -- see README. Only 9.6.2 is bundled.
DEALII_VER="${HPCPERF_DFTFE_DEALII:-9.6.2}"
[ -f "$SRC/CMakeLists.txt" ] && [ -f "$DL/dealii/dealii-$DEALII_VER.tar.gz" ] || { echo "build.sh: src/ or deps/dealii incomplete -- run tools/prepare_benchmark.sh level3 dftfe" >&2; exit 3; }
for tb in openblas/OpenBLAS-0.3.30.tar.gz scalapack/v2.2.2.tar.gz libxc/libxc-7.0.0.tar.gz alglib/alglib-4.06.0.cpp.gpl.tgz p4est/p4est-2.8.7.tar.gz p4est/p4est-setup.sh kokkos/4.6.00.tar.gz elpa/elpa-2026.02.001.tar.gz spglib/CMakeLists.txt; do
    [ -f "$DL/$tb" ] || { echo "build.sh: deps/$tb missing from the materialized bundle" >&2; exit 3; }
done
SHA="$(l3_source_commit "$HERE")"; TREE_SHA="$(l3_source_tree_sha "$HERE")"
# The source compatibility patch (patches/0001-std-isnan.patch) is part of the frozen baseline
# (provenance/patch_series.txt); its content hash stays in the fingerprint exactly as before.
DFTFE_PATCH_SHA=""
for P in "$HERE"/patches/000*.patch; do
    [ -f "$P" ] || continue
    DFTFE_PATCH_SHA="$DFTFE_PATCH_SHA${DFTFE_PATCH_SHA:+,}$(basename "$P" | cut -d- -f1)=$(sha256sum "$P" | cut -c1-12)"
done
[ "$(l3_lock_patches "$HERE")" = "$(for P in "$HERE"/patches/000*.patch; do basename "$P"; done | paste -sd' ')" ] || { echo "build.sh: patch series in the lock file differs from patches/ -- the frozen tree is not the expected baseline" >&2; exit 3; }
ARCH="${HPCPERF_CUDA_ARCH:-$(l3_gpu_arch)}"; [ -n "$ARCH" ] || { echo "build.sh: no GPU arch detected" >&2; exit 1; }
ELPA_GPU="$(echo "${HPCPERF_DFTFE_ELPA_GPU:-ON}" | tr '[:lower:]' '[:upper:]')"
GCC_MM="$(l3_version_mm "$(/usr/bin/gcc -dumpfullversion)")"; OMPI_V="$(mpirun --version | head -1 | /usr/bin/grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
PROFILE="${HPCPERF_DFTFE_PROFILE:-cuda$(l3_version_mm "$(l3_cuda_version)")-gcc${GCC_MM}-ompi$(echo "$OMPI_V" | tr -d .)$( [ "$ELPA_GPU" = OFF ] && echo -elpacpu || true)}"
l3_paths_profile dftfe "$PROFILE"
JOBS="${HPCPERF_BUILD_JOBS:-16}"; INST="$L3_INSTALL"; BLD="$L3_BUILD_DEPS"; SRCD="$L3_SRC"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
DEPS_DESC="dftfe_patches[$DFTFE_PATCH_SHA] openblas=0.3.30 scalapack=2.2.2 libxc=7.0.0 spglib=02159eef alglib=4.06.0 p4est=2.8.7 kokkos=4.6.00(serial) dealii=$DEALII_VER elpa=2026.02.001(nvidia-gpu-kernels=$ELPA_GPU sm_$ARCH) gcc=$(/usr/bin/gcc -dumpfullversion) openmpi=$OMPI_V bundle_tree=$(echo "$TREE_SHA" | cut -c1-16) profile=$PROFILE"
DFTFE_OPTS="WITH_GPU=ON GPU_LANG=cuda GPU_VENDOR=nvidia CMAKE_CUDA_ARCHITECTURES=$ARCH WITH_DCCL=OFF WITH_GPU_AWARE_MPI=OFF WITH_COMPLEX=OFF(+ON if HPCPERF_DFTFE_COMPLEX) USE_64BIT_INT=ON HIGHERQUAD_PSP=OFF WITH_TESTING=OFF BUILD_SHARED_LIBS=ON CMAKE_BUILD_TYPE=Release"
FP="$(l3_fingerprint_text dftfe "$SHA" cuda "$DEPS_DESC" "$DFTFE_OPTS" "WITH_GPU_AWARE_MPI=OFF")"
l3_fingerprint_check "$INST" "$FP" || exit 1
echo "# DFT-FE profile=$PROFILE: dftfe $SHA (1.2.0, frozen source tree $TREE_SHA), gcc $(/usr/bin/gcc -dumpfullversion), $(mpirun --version | head -1), CUDA $(l3_cuda_version) sm_$ARCH, -j$JOBS; expected 2.5-4 h (deal.II dominates), ~10 GB under $L3_DEPS"
T0=$(date +%s)
done_marker() { [ -f "$INST/$1/.hpcperf-stage-done" ]; }
finish() { touch "$INST/$1/.hpcperf-stage-done"; echo "# stage $1 done ($(( $(date +%s) - T0 )) s since start)"; }
untar() { # untar <tarball> <dest-parent> -> prints extracted dir
    local tb=$1 dst=$2 top; mkdir -p "$dst"; top="$(tar -tzf "$tb" | head -1 | cut -d/ -f1)"
    [ -d "$dst/$top" ] || tar -xzf "$tb" -C "$dst"; echo "$dst/$top"
}
run() { # run <log> <cmd...>
    local log=$1; shift; "$@" > "$L3_LOGS/$log" 2>&1 || { tail -40 "$L3_LOGS/$log"; echo "build.sh: FAILED: $* (log $L3_LOGS/$log)" >&2; exit 1; }
}

# 1. OpenBLAS (BLAS+LAPACK for ScaLAPACK/deal.II/ELPA/DFT-FE), single-threaded, gfortran 14
if ! done_marker openblas; then
    d="$(untar "$DL/openblas/OpenBLAS-0.3.30.tar.gz" "$SRCD")"
    run openblas-make.log make -C "$d" -j "$JOBS" USE_THREAD=0 USE_OPENMP=0 DYNAMIC_ARCH=0 NO_AFFINITY=1 TARGET="${HPCPERF_OPENBLAS_TARGET:-SAPPHIRERAPIDS}" CC=/usr/bin/gcc FC=/usr/bin/gfortran
    run openblas-install.log make -C "$d" PREFIX="$INST/openblas" install
    finish openblas
fi
BLAS="$INST/openblas/lib/libopenblas.so"
# 2. ScaLAPACK 2.2.2 (reference), on OpenBLAS, MPI via the wrappers
if ! done_marker scalapack; then
    d="$(untar "$DL/scalapack/v2.2.2.tar.gz" "$SRCD")"; mkdir -p "$BLD/scalapack"
    run scalapack-configure.log cmake -S "$d" -B "$BLD/scalapack" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF -DBUILD_TESTING=OFF \
        -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_Fortran_COMPILER=/usr/bin/gfortran -DMPI_C_COMPILER="$(command -v mpicc)" -DMPI_Fortran_COMPILER="$(command -v mpifort)" \
        -DCMAKE_C_FLAGS="-fPIC -Wno-error=implicit-function-declaration" -DCMAKE_Fortran_FLAGS="-fPIC -fallow-argument-mismatch" \
        -DUSE_OPTIMIZED_LAPACK_BLAS=ON -DBLAS_LIBRARIES="$BLAS" -DLAPACK_LIBRARIES="$BLAS" -DCMAKE_INSTALL_PREFIX="$INST/scalapack"
    run scalapack-build.log cmake --build "$BLD/scalapack" -j "$JOBS"; run scalapack-install.log cmake --install "$BLD/scalapack"
    finish scalapack
fi
# 3. libxc 7.0.0
if ! done_marker libxc; then
    d="$(untar "$DL/libxc/libxc-7.0.0.tar.gz" "$SRCD")"; mkdir -p "$BLD/libxc"
    run libxc-configure.log cmake -S "$d" -B "$BLD/libxc" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
        -DCMAKE_C_FLAGS="-O2 -fPIC" -DCMAKE_CXX_FLAGS="-O2 -fPIC" -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF -DENABLE_FORTRAN=OFF -DCMAKE_INSTALL_PREFIX="$INST/libxc"
    run libxc-build.log cmake --build "$BLD/libxc" -j "$JOBS"; run libxc-install.log cmake --install "$BLD/libxc"
    finish libxc
fi
# 4. spglib (pinned commit)
if ! done_marker spglib; then
    mkdir -p "$BLD/spglib"
    run spglib-configure.log cmake -S "$DL/spglib" -B "$BLD/spglib" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
        -DSPGLIB_WITH_Fortran=OFF -DSPGLIB_WITH_Python=OFF -DSPGLIB_WITH_TESTS=OFF -DCMAKE_INSTALL_PREFIX="$INST/spglib"
    run spglib-build.log cmake --build "$BLD/spglib" -j "$JOBS"; run spglib-install.log cmake --install "$BLD/spglib"
    finish spglib
fi
# 5. ALGLIB 4.06.0 (C++ sources -> shared library, as upstream's recipe)
if ! done_marker alglib; then
    d="$SRCD/alglib-cpp"; [ -d "$d" ] || { mkdir -p "$SRCD"; tar -xzf "$DL/alglib/alglib-4.06.0.cpp.gpl.tgz" -C "$SRCD"; }
    mkdir -p "$INST/alglib"
    ( cd "$d/src" && /usr/bin/g++ -o "$INST/alglib/libAlglib.so" -shared -fPIC -O2 ./*.cpp && cp ./*.h "$INST/alglib/" ) > "$L3_LOGS/alglib.log" 2>&1 || { tail -20 "$L3_LOGS/alglib.log"; echo "build.sh: ALGLIB failed" >&2; exit 1; }
    finish alglib
fi
# 6. p4est 2.8.7 with dftfe's p4est-setup.sh (FAST + DEBUG trees, MPI, zlib required)
if ! done_marker p4est; then
    [ -f /usr/include/zlib.h ] || { echo "build.sh: /usr/include/zlib.h missing (zlib-devel) -- p4est/deal.II need zlib" >&2; exit 1; }
    mkdir -p "$SRCD/p4est"; cp "$DL/p4est/p4est-2.8.7.tar.gz" "$DL/p4est/p4est-setup.sh" "$SRCD/p4est/"; chmod +x "$SRCD/p4est/p4est-setup.sh"
    # p4est 2.8.7 writes its configured header to <build>/config/p4est_config.h; dftfe's p4est-setup.sh (written for
    # older p4est layouts) looks for <build>/src/p4est_config.h in its zlib check -> point the check at the 2.8.7 location
    sed -i 's|/src/p4est_config.h"|/config/p4est_config.h"|g' "$SRCD/p4est/p4est-setup.sh"
    /usr/bin/grep -q 'config/p4est_config.h' "$SRCD/p4est/p4est-setup.sh" || { echo "build.sh: p4est-setup.sh zlib-check path adaptation failed" >&2; exit 1; }
    # dftfe's p4est-setup.sh hardcodes the Cray wrappers (configure ... CC=cc CXX=CC FC=ftn F77=ftn "$@"); the
    # trailing "$@" lets us override them with this node's Open MPI wrappers (last assignment wins in configure)
    ( cd "$SRCD/p4est" && CFLAGS="-fPIC -O2" ./p4est-setup.sh p4est-2.8.7.tar.gz "$INST/p4est" CC=mpicc CXX=mpicxx FC=mpifort F77=mpifort LIBS=-lm )   # LIBS=-lm: the Cray wrappers link libm implicitly, gcc/mpicc do not (libsc examples: undefined sin/sqrt) > "$L3_LOGS/p4est.log" 2>&1 || { tail -30 "$L3_LOGS/p4est.log"; echo "build.sh: p4est failed" >&2; exit 1; }
    [ -f "$INST/p4est/FAST/lib/libp4est.so" ] || [ -f "$INST/p4est/FAST/lib/libp4est.a" ] || { echo "build.sh: p4est FAST install missing" >&2; exit 1; }
    finish p4est
fi
# 7. Kokkos 4.6.00 (Serial; deal.II's Kokkos dependency -- CPU only, as upstream)
if ! done_marker kokkos; then
    d="$(untar "$DL/kokkos/4.6.00.tar.gz" "$SRCD")"; mkdir -p "$BLD/kokkos"
    run kokkos-configure.log cmake -S "$d" -B "$BLD/kokkos" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CXX_FLAGS="-O2 -fPIC" -DKokkos_ENABLE_SERIAL=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX="$INST/kokkos"
    run kokkos-build.log cmake --build "$BLD/kokkos" -j "$JOBS"; run kokkos-install.log cmake --install "$BLD/kokkos"
    finish kokkos
fi
# 8. deal.II (version: DEALII_VER above)
if ! done_marker dealii; then
    d="$(untar "$DL/dealii/dealii-$DEALII_VER.tar.gz" "$SRCD")"; mkdir -p "$BLD/dealii"
    run dealii-configure.log cmake -S "$d" -B "$BLD/dealii" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$(command -v mpicc)" -DCMAKE_CXX_COMPILER="$(command -v mpicxx)" -DCMAKE_Fortran_COMPILER="$(command -v mpifort)" \
        -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_FLAGS="-std=c++17" -DDEAL_II_CXX_FLAGS_RELEASE=-O2 -DDEAL_II_ALLOW_PLATFORM_INTROSPECTION=OFF \
        -DDEAL_II_WITH_MPI=ON -DDEAL_II_WITH_64BIT_INDICES=ON -DDEAL_II_WITH_COMPLEX_VALUES=ON -DDEAL_II_FORCE_BUNDLED_BOOST=ON \
        -DDEAL_II_WITH_TASKFLOW=OFF -DDEAL_II_WITH_TBB=OFF -DDEAL_II_COMPONENT_EXAMPLES=OFF \
        -DP4EST_DIR="$INST/p4est" -DKOKKOS_DIR="$INST/kokkos" -DDEAL_II_WITH_KOKKOS=ON \
        -DDEAL_II_WITH_LAPACK=ON -DLAPACK_DIR="$INST/openblas" -DLAPACK_FOUND=true -DLAPACK_LIBRARIES="$BLAS" \
        -DCMAKE_INSTALL_PREFIX="$INST/dealii"
    run dealii-build.log cmake --build "$BLD/dealii" -j "$JOBS"; run dealii-install.log cmake --install "$BLD/dealii"
    finish dealii
fi
# 9. ELPA 2026.02.001 (autotools). NVIDIA GPU kernels for sm_100 (generic --generate-code path of
#    ELPA's configure). Test programs are built (make check TESTS=) for elpa_probe.sh; not run here.
#    ELPA expects the host SIMD flags in CFLAGS (its configure probes AVX512 intrinsics with the given
#    CFLAGS and aborts otherwise; EL10's GCC defaults to x86-64-v3 = AVX2 only): -march=native on this
#    Xeon 8570 enables the AVX-512 CPU kernels -- a node-specific binary, like OpenBLAS's TARGET.
#    LDFLAGS carries the ScaLAPACK/OpenBLAS library paths as well: ELPA's later CUDA checks (cublas) link
#    with LIBS=-lscalapack but replace LDFLAGS by the CUDA path, so without them "cannot find -lscalapack".
if ! done_marker elpa; then
    d="$(untar "$DL/elpa/elpa-2026.02.001.tar.gz" "$SRCD")"; mkdir -p "$BLD/elpa"
    GPUOPTS=(--disable-nvidia-gpu-kernels)
    [ "$ELPA_GPU" = ON ] && GPUOPTS=(--enable-nvidia-gpu-kernels "--with-NVIDIA-GPU-compute-capability=sm_$ARCH" "--with-cuda-path=$CUDA_ROOT")
    ( cd "$BLD/elpa" && "$d/configure" --prefix="$INST/elpa" CC="$(command -v mpicc)" CXX="$(command -v mpicxx)" FC="$(command -v mpifort)" \
        CFLAGS="-O2 -fPIC -march=native" CXXFLAGS="-O2 -fPIC -std=c++17 -march=native" FCFLAGS="-O2 -fPIC -march=native" \
        SCALAPACK_LDFLAGS="-L$INST/scalapack/lib -lscalapack -L$INST/openblas/lib -lopenblas -Wl,-rpath,$INST/scalapack/lib -Wl,-rpath,$INST/openblas/lib" \
        SCALAPACK_FCFLAGS="-I$INST/scalapack/include" \
        LDFLAGS="-L$INST/scalapack/lib -L$INST/openblas/lib -Wl,-rpath,$INST/scalapack/lib -Wl,-rpath,$INST/openblas/lib" \
        --enable-shared --disable-static --enable-c-tests=no --enable-cpp-tests=no --enable-option-checking=fatal --disable-openmp \
        "${GPUOPTS[@]}" ) > "$L3_LOGS/elpa-configure.log" 2>&1 || { tail -40 "$L3_LOGS/elpa-configure.log"; echo "build.sh: ELPA configure failed" >&2; exit 1; }
    run elpa-build.log make -C "$BLD/elpa" -j "$JOBS"
    run elpa-tests-build.log make -C "$BLD/elpa" -j "$JOBS" check TESTS=
    run elpa-install.log make -C "$BLD/elpa" install
    ls "$BLD/elpa"/validate_* > "$INST/elpa/TEST_PROGRAMS.txt" 2>/dev/null || true
    finish elpa
fi
# 10. DFT-FE (real; complex on request). DFT-FE's CMake writes include/git_info.h INTO its source directory,
#     so the application is configured from a build-side copy of the frozen tree (the frozen src/ stays pristine).
SRC_BUILD="$SRCD/dftfe-src"
if [ ! -f "$SRC_BUILD/.hpcperf-src-stamp" ] || [ "$(cat "$SRC_BUILD/.hpcperf-src-stamp")" != "$SHA tree=$TREE_SHA" ]; then
    rm -rf "$SRC_BUILD"; mkdir -p "$SRC_BUILD"; rsync -a "$SRC/" "$SRC_BUILD/"; echo "$SHA tree=$TREE_SHA" > "$SRC_BUILD/.hpcperf-src-stamp"
fi
build_dftfe() { # build_dftfe <real|complex>
    local kind=$1 cplx=OFF; [ "$kind" = complex ] && cplx=ON
    mkdir -p "$L3_BUILD/$kind"
    run "dftfe-$kind-configure.log" cmake -S "$SRC_BUILD" -B "$L3_BUILD/$kind" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CXX_COMPILER="$(command -v mpicxx)" -DCMAKE_CXX_FLAGS="-fPIC" -DCMAKE_CXX_FLAGS_RELEASE="-O2" \
        -DDEAL_II_DIR="$INST/dealii" -DALGLIB_DIR="$INST/alglib" -DLIBXC_DIR="$INST/libxc" -DSPGLIB_DIR="$INST/spglib" \
        -DXML_LIB_DIR=/usr/lib64 -DXML_INCLUDE_DIR=/usr/include/libxml2 -DCMAKE_PREFIX_PATH="$INST/elpa" \
        -DWITH_MDI=OFF -DWITH_TORCH=OFF -DWITH_CUSTOMIZED_DEALII=OFF -DWITH_DCCL=OFF \
        -DWITH_COMPLEX="$cplx" -DWITH_GPU=ON -DGPU_LANG=cuda -DGPU_VENDOR=nvidia -DWITH_GPU_AWARE_MPI=OFF \
        "-DCMAKE_CUDA_ARCHITECTURES=$ARCH" -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
        -DWITH_TESTING=OFF -DHIGHERQUAD_PSP=OFF -DBUILD_SHARED_LIBS=ON -DUSE_64BIT_INT=ON
    run "dftfe-$kind-build.log" cmake --build "$L3_BUILD/$kind" -j "$JOBS"
    mkdir -p "$INST/bin"; cp -f "$L3_BUILD/$kind/dftfe" "$INST/bin/dftfe_$kind"
    # (cuobjdump exits nonzero on a host-only binary: keep the substitutions from aborting the script under pipefail)
    local archs; archs="$( { cuobjdump --list-elf "$INST/bin/dftfe_$kind" 2>/dev/null || true; } | { /usr/bin/grep -o 'sm_[0-9]*' || true; } | sort -u | paste -sd,)"
    [ -n "$archs" ] || archs="$(for l in "$L3_BUILD/$kind"/libdftfe*.so*; do if [ -f "$l" ]; then cuobjdump --list-elf "$l" 2>/dev/null | /usr/bin/grep -o 'sm_[0-9]*' || true; fi; done | sort -u | paste -sd,)"
    [ "$archs" = "sm_$ARCH" ] || { echo "build.sh: dftfe_$kind device code '$archs' != sm_$ARCH" >&2; exit 1; }
    echo "dftfe_$kind sha256=$(l3_sha_file "$INST/bin/dftfe_$kind") device_archs=$archs" >> "$INST/BUILD_INFO.txt"
}
: > "$INST/BUILD_INFO.txt"
build_dftfe real
[ -n "${HPCPERF_DFTFE_COMPLEX:-}" ] && build_dftfe complex
l3_fingerprint_write "$INST" "$FP"
{
    echo "profile=$PROFILE dftfe=$SHA utc=$(date -u +%FT%TZ) jobs=$JOBS total_seconds=$(( $(date +%s) - T0 ))"
    echo "deps: $DEPS_DESC"; echo "dftfe options: $DFTFE_OPTS"
    echo "elpa nvidia kernels: $ELPA_GPU ($(/usr/bin/grep -c 'nvidia' "$L3_LOGS/elpa-configure.log" 2>/dev/null || true) configure lines mention nvidia)"
} >> "$INST/BUILD_INFO.txt"
cat "$INST/BUILD_INFO.txt"
echo "# DFT-FE built in $(( $(date +%s) - T0 )) s -> $INST/bin"
