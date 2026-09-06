#!/usr/bin/env bash
# Build GEOS (develop snapshot b7a0f133) with upstream's thirdPartyLibs superbuild
# (TPL 9b55672 = tag 361-1070) and a node host-config, in a private profile tree.
#
#   ./build.sh [tpl|geos|all]      (default all)   HPCPERF_BUILD_JOBS=N (default 32)
#
# Stages:
#   [0] OpenBLAS 0.3.30 (single-threaded; BLAS/LAPACK for hypre/SuiteSparse/SuperLU_DIST)
#   [1] thirdPartyLibs superbuild: RAJA/CHAI/Umpire/camp 2026.07.0 (CUDA sm_100), hypre f1374fb6
#       (--with-cuda --enable-cusparse --enable-cusolver --enable-unified-memory --with-umpire,
#       gpu-arch 100, mixedint), hypredrive, conduit 0.9.5, HDF5 1.12.1, silo 4.11.1, VTK,
#       pugixml, ParMETIS/METIS, SuperLU_DIST, SuiteSparse 5.10.1, Scotch, fmt
#       (Trilinos/PETSc/Caliper/MathPresso/Doxygen/Uncrustify OFF)      -> install/tpl
#   [2] GEOS: scripts/config-build.py with the same host-config + GEOS_TPL_DIR -> install/geos
# Toolchain: system GCC 14.2.1 (C/C++; CUDA host), conda Open MPI 5.0.10 (wrappers redirected
# via OMPI_CC/OMPI_CXX), CUDA 13.2, sm_100. Profile cuda132-gcc142-ompi5010.
# Modification class: A/B (host-config + build options; no source patched unless a
# patches/ file exists and is listed below -- none at the time of writing).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
l3_isolate_build_env
l3_clean_conda_build_env   # system GCC 14 build: drop conda's CFLAGS/LDFLAGS/AR/...
export CC=/usr/bin/gcc CXX=/usr/bin/g++ FC=/usr/bin/gfortran OMPI_CC=/usr/bin/gcc OMPI_CXX=/usr/bin/g++ OMPI_FC=/usr/bin/gfortran CUDAHOSTCXX=/usr/bin/g++
unset CMAKE_GENERATOR
STAGE="${1:-all}"
GEOS_SRC="$R/_upstream/level3/GEOS"; TPL_SRC="$R/_upstream/level3/thirdPartyLibs"
[ -f "$GEOS_SRC/src/CMakeLists.txt" ] && [ -f "$TPL_SRC/CMakeLists.txt" ] || { echo "build.sh: run $HERE/fetch.sh first" >&2; exit 1; }
GEOS_SHA="$(git -C "$GEOS_SRC" rev-parse HEAD)"; TPL_SHA="$(git -C "$TPL_SRC" rev-parse HEAD)"
# TPL build-system patches (headers in patches/*.patch): 0001 fixes upstream's 65-character superlu_dist URL hash,
# 0002 makes RAJA_ENABLE_VECTORIZATION overridable (nvcc 13.2 + GCC 14/x86-64-v3 cannot compile RAJA's AVX2 tensor layer),
# 0003 gives the hdf5 step the superbuild's sub-project generator and build/install commands like every other step.
# Applied to the TPL checkout idempotently; their hashes are recorded in the fingerprint.
# GEOS-side patch: BLT submodule smoke test vs CUDA 13 (back-port of LLNL/blt 38b46203), same idempotent scheme
BLT_SRC="$GEOS_SRC/src/cmake/blt"
for BLT_PATCH in "$HERE"/patches/geos-blt-*.patch; do
    [ -f "$BLT_PATCH" ] || continue
    if git -C "$BLT_SRC" apply --check --reverse "$BLT_PATCH" >/dev/null 2>&1; then :
    elif git -C "$BLT_SRC" apply --check "$BLT_PATCH" >/dev/null 2>&1; then git -C "$BLT_SRC" apply "$BLT_PATCH"; echo "# applied $(basename "$BLT_PATCH") to the GEOS BLT submodule"
    else echo "build.sh: $BLT_PATCH is neither applied nor applicable to BLT $(git -C "$BLT_SRC" rev-parse --short HEAD)" >&2; exit 1; fi
    GEOS_PATCH_SHA="${GEOS_PATCH_SHA:-}${GEOS_PATCH_SHA:+,}$(basename "$BLT_PATCH" .patch)=$(sha256sum "$BLT_PATCH" | cut -c1-12)"
done
TPL_PATCH_SHA=""
for TPL_PATCH in "$HERE"/patches/000*.patch; do
    if git -C "$TPL_SRC" apply --check --reverse "$TPL_PATCH" >/dev/null 2>&1; then :
    elif git -C "$TPL_SRC" apply --check "$TPL_PATCH" >/dev/null 2>&1; then git -C "$TPL_SRC" apply "$TPL_PATCH"; echo "# applied $(basename "$TPL_PATCH") to the thirdPartyLibs checkout"
    else echo "build.sh: $TPL_PATCH is neither applied nor applicable to thirdPartyLibs $TPL_SHA" >&2; exit 1; fi
    TPL_PATCH_SHA="$TPL_PATCH_SHA${TPL_PATCH_SHA:+,}$(basename "$TPL_PATCH" | cut -d- -f1)=$(sha256sum "$TPL_PATCH" | cut -c1-12)"
done
ARCH="${HPCPERF_CUDA_ARCH:-$(l3_gpu_arch)}"; [ "$ARCH" = 100 ] || { echo "build.sh: host-config is written for sm_100 (got sm_$ARCH)" >&2; exit 1; }
GCC_MM="$(l3_version_mm "$(/usr/bin/gcc -dumpfullversion)")"; OMPI_V="$(mpirun --version | head -1 | /usr/bin/grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
PROFILE="${HPCPERF_GEOS_PROFILE:-cuda$(l3_version_mm "$(l3_cuda_version)")-gcc${GCC_MM}-ompi$(echo "$OMPI_V" | tr -d .)}"
l3_paths_profile geos "$PROFILE"
JOBS="${HPCPERF_BUILD_JOBS:-32}"
HC="$HERE/host-configs/gmu-hopper-gcc14-ompi5010-cuda132-sm100.cmake"
MPI_BIN="$(dirname "$(command -v mpicc)")"; CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
OPENBLAS="$L3_INSTALL/openblas/lib/libopenblas.so"
HCDEFS=(-D "HPCPERF_MPI_BIN=$MPI_BIN" -D "HPCPERF_OPENBLAS_LIB=$OPENBLAS" -D "HPCPERF_CUDA_ROOT=$CUDA_ROOT")
DEPS="geos_patches[${GEOS_PATCH_SHA:-none}] tpl=361-1070($TPL_SHA)+patches[$TPL_PATCH_SHA] raja/chai/umpire=2026.07.0 hypre=f1374fb6(cuda,unified-memory,umpire,gpu-arch=$ARCH) openblas=0.3.30 gcc=$(/usr/bin/gcc -dumpfullversion) openmpi=$OMPI_V profile=$PROFILE"
OPTS="host-config=$(basename "$HC") RAJA_ENABLE_VECTORIZATION=OFF ENABLE_HYPREDRV=OFF(host-config+TPL) tpl_metis_include_first ENABLE_TESTS=ON(upstream default) ENABLE_CUDA=ON CUDA_ARCH=sm_$ARCH GEOS_LA_INTERFACE=Hypre ENABLE_HYPRE_DEVICE=CUDA ENABLE_TRILINOS=OFF ENABLE_PETSC=OFF ENABLE_OPENMP=OFF ENABLE_CALIPER=OFF ENABLE_MATHPRESSO=OFF ENABLE_VTK=ON ENABLE_SUPERLU_DIST=ON ENABLE_SUITESPARSE=ON ENABLE_SCOTCH=ON CMAKE_BUILD_TYPE=Release"
FP="$(l3_fingerprint_text geos "$GEOS_SHA" cuda "$DEPS" "$OPTS" "ENABLE_HYPRE_GPU_AWARE_MPI=OFF (GEOS pinned host buffers)")"
l3_fingerprint_check "$L3_INSTALL" "$FP" || exit 1
echo "# GEOS profile=$PROFILE: GEOS $GEOS_SHA, TPL $TPL_SHA, gcc $(/usr/bin/gcc -dumpfullversion), $(mpirun --version | head -1), CUDA $(l3_cuda_version) sm_$ARCH, -j$JOBS"
echo "# resources: TPL superbuild ~2-3 h (VTK, hypre-CUDA, RAJA suite; downloads ~500 MB), GEOS ~1.5-2.5 h; ~25 GB under $L3_DEPS"
T0=$(date +%s)
run() { local log=$1; shift; "$@" > "$L3_LOGS/$log" 2>&1 || { tail -50 "$L3_LOGS/$log"; echo "build.sh: FAILED: $* (log $L3_LOGS/$log)" >&2; exit 1; }; }

# [0] OpenBLAS
if [ ! -f "$L3_INSTALL/openblas/.hpcperf-stage-done" ]; then
    TB="$R/.deps/level3/dftfe/downloads/OpenBLAS-0.3.30.tar.gz"
    [ -f "$TB" ] || { mkdir -p "$L3_DEPS/downloads"; TB="$L3_DEPS/downloads/OpenBLAS-0.3.30.tar.gz"; curl -sSL -o "$TB" https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.30/OpenBLAS-0.3.30.tar.gz; }
    echo "$(sha256sum "$TB" | cut -d' ' -f1) OpenBLAS-0.3.30.tar.gz" > "$L3_LOGS/openblas.sha256"
    d="$L3_SRC/OpenBLAS-0.3.30"; [ -d "$d" ] || tar -xzf "$TB" -C "$L3_SRC"
    run openblas-make.log make -C "$d" -j "$JOBS" USE_THREAD=0 USE_OPENMP=0 DYNAMIC_ARCH=0 NO_AFFINITY=1 TARGET="${HPCPERF_OPENBLAS_TARGET:-SAPPHIRERAPIDS}" CC=/usr/bin/gcc FC=/usr/bin/gfortran
    run openblas-install.log make -C "$d" PREFIX="$L3_INSTALL/openblas" install
    touch "$L3_INSTALL/openblas/.hpcperf-stage-done"
fi
T1=$(date +%s)
# [1] thirdPartyLibs superbuild
if [ "$STAGE" != geos ] && [ ! -f "$L3_INSTALL/tpl/.hpcperf-stage-done" ]; then
    # config-build.py DELETES an existing build directory before configuring; configure only once and let
    # a re-run continue the ninja build (CMakeLists changes -- e.g. the TPL patch -- trigger cmake's own re-run)
    # vectorization: see patches/0002 (upstream's own choice for ROCm); NUM_PROC = make -j inside the TPL steps;
    # hypredrive (optional hypre driver library, GEOS' ENABLE_HYPREDRV defaults to OFF and is not used here) is left out:
    # upstream's TPL step compiles it against a hypre built with Umpire but without Umpire's include path (fails on
    # `umpire/config.hpp: No such file`) -- not needed for the beam workflow, so disabled rather than patched.
    TPLDEFS=(-D RAJA_ENABLE_VECTORIZATION:BOOL=OFF -D ENABLE_HYPREDRV:BOOL=OFF -D "NUM_PROC=$JOBS")
    if [ ! -f "$L3_BUILD_DEPS/tpl/build.ninja" ]; then
        ( cd "$TPL_SRC" && python3 scripts/config-build.py -hc "$HC" -bt Release -bp "$L3_BUILD_DEPS/tpl" -ip "$L3_INSTALL/tpl" -n "${HCDEFS[@]}" "${TPLDEFS[@]}" ) > "$L3_LOGS/tpl-configure.log" 2>&1 \
            || { tail -40 "$L3_LOGS/tpl-configure.log"; echo "build.sh: TPL configure failed" >&2; exit 1; }
    else  # existing tree: re-assert the cache values without config-build.py (which would delete the tree)
        run tpl-reconfigure.log cmake "${TPLDEFS[@]}" "$L3_BUILD_DEPS/tpl"
    fi
    run tpl-build.log ninja -C "$L3_BUILD_DEPS/tpl" -j "$JOBS"
    for d in raja chai hypre hdf5 conduit vtk pugixml suitesparse superlu_dist parmetis scotch; do [ -d "$L3_INSTALL/tpl/$d" ] || echo "build.sh: WARNING TPL install lacks $d" >&2; done
    [ -d "$L3_INSTALL/tpl/hypre" ] && [ -d "$L3_INSTALL/tpl/raja" ] || { echo "build.sh: TPL install incomplete (hypre/raja missing)" >&2; exit 1; }
    for l in "$L3_INSTALL"/tpl/hypre/lib/libHYPRE*.a "$L3_INSTALL"/tpl/raja/lib/libRAJA*.a; do [ -f "$l" ] && echo "$(basename "$l"): $(cuobjdump --list-elf "$l" 2>/dev/null | /usr/bin/grep -o 'sm_[0-9]*' | sort -u | paste -sd,)"; done > "$L3_LOGS/tpl-device-archs.txt" || true
    touch "$L3_INSTALL/tpl/.hpcperf-stage-done"
fi
T2=$(date +%s)
[ "$STAGE" = tpl ] && { echo "# TPL stage done ($((T2 - T0)) s)"; exit 0; }
# [2] GEOS
mkdir -p "$L3_BUILD"
if [ ! -f "$L3_BUILD/build.ninja" ]; then   # same reason as above: config-build.py deletes an existing build tree
    ( cd "$GEOS_SRC" && python3 scripts/config-build.py -hc "$HC" -bt Release -bp "$L3_BUILD" -ip "$L3_INSTALL/geos" -n "${HCDEFS[@]}" -D "HPCPERF_GEOS_TPL_DIR=$L3_INSTALL/tpl" -D "GEOS_TPL_DIR=$L3_INSTALL/tpl" ) > "$L3_LOGS/geos-configure.log" 2>&1 \
        || { tail -60 "$L3_LOGS/geos-configure.log"; echo "build.sh: GEOS configure failed" >&2; exit 1; }
else  # existing tree: re-read the host-config (cache FORCE values) without config-build.py
    run geos-reconfigure.log cmake "${HCDEFS[@]}" -D "HPCPERF_GEOS_TPL_DIR=$L3_INSTALL/tpl" -D "GEOS_TPL_DIR=$L3_INSTALL/tpl" -C "$HC" "$L3_BUILD"   # -D before -C: the host-config reads HPCPERF_*
fi
run geos-build.log ninja -C "$L3_BUILD" -j "$JOBS"
run geos-install.log ninja -C "$L3_BUILD" install
T3=$(date +%s)
EXE="$L3_INSTALL/geos/bin/geosx"; [ -x "$EXE" ] || EXE="$(find "$L3_INSTALL/geos/bin" -maxdepth 1 -type f -name 'geos*' | head -1)"
[ -n "$EXE" ] && [ -x "$EXE" ] || { echo "build.sh: GEOS executable not found under $L3_INSTALL/geos/bin" >&2; exit 1; }
archs="$(cuobjdump --list-elf "$EXE" 2>/dev/null | /usr/bin/grep -o 'sm_[0-9]*' | sort -u | paste -sd,)"
[ -n "$archs" ] || archs="$(for l in "$L3_INSTALL"/geos/lib/*.so "$L3_INSTALL"/geos/lib64/*.so; do [ -f "$l" ] && cuobjdump --list-elf "$l" 2>/dev/null | /usr/bin/grep -o 'sm_[0-9]*'; done | sort -u | paste -sd,)"
[ "$archs" = "sm_$ARCH" ] || { echo "build.sh: GEOS device code '$archs' != sm_$ARCH" >&2; exit 1; }
l3_fingerprint_write "$L3_INSTALL" "$FP"
{
    echo "profile=$PROFILE geos=$GEOS_SHA tpl=$TPL_SHA utc=$(date -u +%FT%TZ) jobs=$JOBS"
    echo "seconds: openblas=$((T1 - T0)) tpl=$((T2 - T1)) geos=$((T3 - T2)) total=$((T3 - T0))"
    echo "geos exe=$EXE sha256=$(l3_sha_file "$EXE") device_archs=$archs"
    [ -f "$L3_LOGS/tpl-device-archs.txt" ] && sed 's/^/tpl: /' "$L3_LOGS/tpl-device-archs.txt"
    "$EXE" --help 2>&1 | head -3 | sed 's/^/help: /'
} > "$L3_INSTALL/BUILD_INFO.txt"; cat "$L3_INSTALL/BUILD_INFO.txt"
echo "# GEOS built: openblas $((T1 - T0)) s, TPL $((T2 - T1)) s, GEOS $((T3 - T2)) s"
