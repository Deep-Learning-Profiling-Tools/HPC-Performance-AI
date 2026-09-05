#!/usr/bin/env bash
# Build nekRS (OCCA/CUDA backend, HYPRE on GPU) with upstream's CMake, from a
# private copy of the vendored source tree.
#
#   ./build.sh [CUDA|HIP]        (default CUDA)
#
# Layout (Level 3 isolation): read-only clone _upstream/level3/nekRS; patched
# private source copy .deps/level3/nekrs/src (upstream's third-party libraries
# -- OCCA, HYPRE 2.32, gslib, Nek5000, LAPACK -- are vendored in-tree and built
# by nekRS' own superbuild, nothing is shared with other applications); build
# build/level3/nekrs/<cuda|hip>; install .deps/level3/nekrs/install
# (= NEKRS_HOME, with nekrs.conf recording the JIT toolchain); logs
# .deps/level3/nekrs/logs.
#
# Toolchain (upstream: GNU >= 9.1, MPI-3.1 with Fortran bindings, CMake >= 3.21,
# CUDA >= 12): CC/CXX/FC = conda Open MPI wrappers (mpicc/mpicxx/mpif90 around
# conda GCC 13.3.0). The conda environment has no gfortran, so the Fortran
# wrapper is pointed at the system gfortran 14.2.1 through OMPI_FC (class C,
# environment only); the conda MPI Fortran modules (.mod format 15) load under
# gfortran 14 and the mixed link (gfortran-14 objects + conda libgfortran) was
# tested with a 2-rank MPI Fortran program before this recipe was written.
#
# Options vs upstream defaults (class A, documented CMake options):
# OCCA_ENABLE_HIP/DPCPP=OFF (CUDA only; no ROCm/SYCL here), ENABLE_ADIOS=OFF
# (ADIOS2 checkpoint backend not needed; native .fld output stays),
# NEKRS_BUILD_FLOAT=OFF (skip the second, fp32 solver build), ENABLE_CVODE
# off (default). Patch (class B, 1 line): cmake/hypre.cmake adds sm_100 to the
# HYPRE device architectures for CUDA >= 13 (upstream lists 80 90 only, which
# would leave HYPRE's device kernels without Blackwell code); OKL kernels are
# JIT-compiled by OCCA for the device it finds at run time (sm_100 here).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
l3_isolate_build_env    # Level 3 builds must not see Level 2 .deps/install prefixes

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
UP="$R/_upstream/level3/nekRS"
[ -f "$UP/CMakeLists.txt" ] || { echo "build.sh: $UP missing -- run $HERE/fetch.sh first" >&2; exit 1; }
SHA="$(git -C "$UP" rev-parse HEAD)"
l3_paths nekrs

# --- variant selection (multi-variant build/install/cache isolation) ----------
# HPCPERF_NEKRS_HYPRE_GPU=ON|OFF selects whether the vendored HYPRE is built with
# its CUDA device backend (GPU coarse solve possible) or host-only (CPU coarse
# only). This is INDEPENDENT of OCCA_ENABLE_CUDA: the main application is on the
# GPU either way. The default variant 'hypregpu' (ENABLE_HYPRE_GPU=ON) keeps the
# existing legacy paths so the already-validated install is untouched; any other
# variant gets a fully separate src/build/install/logs and its own JIT cache.
HYPRE_GPU="${HPCPERF_NEKRS_HYPRE_GPU:-ON}"
case "$HYPRE_GPU" in ON|OFF) : ;; *) echo "build.sh: HPCPERF_NEKRS_HYPRE_GPU must be ON or OFF" >&2; exit 2 ;; esac
VARIANT="${HPCPERF_NEKRS_VARIANT:-$([ "$HYPRE_GPU" = ON ] && echo hypregpu || echo cpucoarse)}"
if [ "$VARIANT" = hypregpu ]; then
    BUILD_DIR="$R/build/level3/nekrs/$MODEL"                 # legacy layout (unchanged)
else
    L3_DEPS="$L3_R/.deps/level3/nekrs/$VARIANT"
    L3_SRC="$L3_DEPS/src"; L3_INSTALL="$L3_DEPS/install"; L3_LOGS="$L3_DEPS/logs"
    mkdir -p "$L3_SRC" "$L3_INSTALL" "$L3_LOGS"
    BUILD_DIR="$R/build/level3/nekrs/$VARIANT.$MODEL"
fi
JOBS="${HPCPERF_BUILD_JOBS:-32}"
SYS_FC="${HPCPERF_SYSTEM_GFORTRAN:-/usr/bin/gfortran}"
[ -x "$SYS_FC" ] || { echo "build.sh: no gfortran at $SYS_FC (set HPCPERF_SYSTEM_GFORTRAN); the conda env has none" >&2; exit 1; }
export OMPI_FC="$SYS_FC"
# Mixed GCC majors (conda GCC 13 for C/C++, system gfortran 14): CMake's FortranCInterface detection
# compiles its probe objects with -flto=auto -ffat-lto-objects (GCC >= 12) and links them with the
# Fortran driver, whose lto1 rejects GCC 13 bytecode ("LTO version 13.1 instead of 14.0"). Linking with
# -fno-lto uses the fat objects' regular code instead (verified with a 2-language test program). nekRS
# itself does not use LTO, so this changes nothing else. Class C (link flag).
export LDFLAGS="${LDFLAGS:-} -fno-lto"
# The conda GCC links position-independent executables by default while the system gfortran emits
# non-PIE objects ("relocation R_X86_64_32S ... can not be used when making a PIE object"); every
# Fortran object that ends up in nekrs (Nek5000 interface, vendored LAPACK) therefore needs -fPIC.
# Class C (compile flag; no effect on numerics).
export FFLAGS="${FFLAGS:-} -fPIC"
# The conda environment exports AR=x86_64-conda-linux-gnu-ar; the vendored HYPRE's configure takes $AR
# verbatim as the full archive command (its default is "ar -rcu"), so the bare tool name makes every
# `ar libHYPRE_*.a ...` call fail with a usage error. Unset -> HYPRE's own default. Class C.
unset AR
# The three HYPRE patches are needed ONLY to compile HYPRE's CUDA device backend against CUDA 13:
#   0001 (class B, 1 line): add sm_100 to HYPRE_CUDA_SM (device SASS list).
#   0002 (class D, 2 lines): thrust::reduce_by_key result type -> auto (device_utils.c/csr_matop_device.c).
#   0003 (class D, 36 lines): explicit <thrust/iterator/reverse_iterator.h> + <thrust/pair.h> includes and
#         thrust::not1 -> thrust::not_fn, all in HYPRE's *device* sources / device_utils headers.
# All three touch code that is compiled by nvcc ONLY when ENABLE_HYPRE_GPU=ON. With ENABLE_HYPRE_GPU=OFF
# HYPRE is host-only and that code is not compiled, so NO patch is applied (the candidate tests whether
# the unused GPU component -- and therefore the patches -- can be dropped entirely).
if [ "$HYPRE_GPU" = ON ]; then
    PATCHES=("$HERE/patches/0001-hypre-cuda-sm100.patch" "$HERE/patches/0002-hypre-cuda13-thrust-pair.patch" "$HERE/patches/0003-hypre-cuda13-thrust3-compat.patch")
else
    PATCHES=()
fi

case "$BACKEND" in
    CUDA) command -v nvcc >/dev/null || { echo "build.sh: nvcc not on PATH" >&2; exit 1; }
          OCCA_FLAGS=(-DOCCA_ENABLE_CUDA=ON -DOCCA_ENABLE_HIP=OFF -DOCCA_ENABLE_DPCPP=OFF -DOCCA_ENABLE_OPENCL=OFF); ARCHNOTE="sm_$(l3_gpu_arch) (JIT at run time)" ;;
    HIP)  command -v hipcc >/dev/null 2>&1 || { echo "build.sh: HIP requested but hipcc not found -- HIP build is UNTESTED on this machine (no ROCm)" >&2; exit 1; }
          OCCA_FLAGS=(-DOCCA_ENABLE_CUDA=OFF -DOCCA_ENABLE_HIP=ON -DOCCA_ENABLE_DPCPP=OFF -DOCCA_ENABLE_OPENCL=OFF); ARCHNOTE="gfx950 (JIT at run time)" ;;
    *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac
CMAKE_OPTS="variant=$VARIANT ${OCCA_FLAGS[*]} ENABLE_HYPRE_GPU=$HYPRE_GPU ENABLE_ADIOS=OFF ENABLE_CVODE=OFF NEKRS_BUILD_FLOAT=OFF NEKRS_GPU_MPI=OFF(default; runtime NEKRS_GPU_MPI) CC=mpicc CXX=mpicxx FC=mpif90(OMPI_FC=$SYS_FC)"
PATCHNAMES=(); for p in "${PATCHES[@]}"; do PATCHNAMES+=("$(basename "$p")"); done
FP="$(l3_fingerprint_text nekrs "$SHA" "$MODEL" "vendored: occa=2.0.0-dev hypre=2.32.0 gslib nek5000 lapack (in-tree)" "$CMAKE_OPTS" "runtime(NEKRS_GPU_MPI, default 0)" "${PATCHNAMES[@]}")"
l3_fingerprint_check "$L3_INSTALL" "$FP" || exit 1

echo "# nekRS $BACKEND: variant=$VARIANT ENABLE_HYPRE_GPU=$HYPRE_GPU patches=${#PATCHES[@]} upstream $SHA (v26.0), arch $ARCHNOTE, install=$L3_INSTALL"
# private source copy (upstream clone stays pristine). The copy is ~280 MB in many small files (slow on
# this filesystem), so it is reused only when it already holds this upstream commit AND this exact patch
# series. The cache key is the upstream SHA plus the ORDERED patch-CONTENT hash, so editing a patch (even
# without renaming it) invalidates the copy. A missing patch is a hard error.
for p in "${PATCHES[@]}"; do [ -f "$p" ] || { echo "build.sh: patch $p missing" >&2; exit 1; }; done
PATCH_SERIES_HASH="$(for p in "${PATCHES[@]}"; do sha256sum "$p" | cut -d' ' -f1; done | sha256sum | cut -d' ' -f1)"
STAMP="$SHA $PATCH_SERIES_HASH"
if [ -f "$L3_SRC/.hpcperf-src-stamp" ] && [ "$(cat "$L3_SRC/.hpcperf-src-stamp")" = "$STAMP" ]; then
    echo "# reusing patched source copy $L3_SRC ($STAMP)"
else
    rm -rf "$L3_SRC"; mkdir -p "$L3_SRC"
    rsync -a --exclude .git "$UP/" "$L3_SRC/"    # examples/ and doc/ are installed by nekRS' CMake
    for p in "${PATCHES[@]}"; do
        (cd "$L3_SRC" && patch -p1 --forward --silent < "$p") || { echo "build.sh: patch $(basename "$p") failed to apply" >&2; exit 1; }
        echo "# applied $(basename "$p")"
    done
    echo "$STAMP" > "$L3_SRC/.hpcperf-src-stamp"
fi
# fresh configure every time: CMake caches CMAKE_EXE_LINKER_FLAGS and the Fortran/C detection results
# from the first configure of a build directory, so environment fixes would otherwise not take effect.
# HPCPERF_L3_INCREMENTAL=1 keeps an existing build tree (only for re-running install/fingerprint after a
# late failure with an unchanged toolchain).
[ -n "${HPCPERF_L3_INCREMENTAL:-}" ] || rm -rf "$BUILD_DIR"; mkdir -p "$BUILD_DIR"
# upstream's build.sh uses CMake's default generator (Unix Makefiles); the conda environment exports
# CMAKE_GENERATOR=Ninja, under which nekRS' generated build file is invalid ("bad $-escape" in a
# vendored-library rule), so the generator is pinned to upstream's
CC=mpicc CXX=mpicxx FC=mpif90 cmake -S "$L3_SRC" -B "$BUILD_DIR" -G "Unix Makefiles" -Wfatal-errors \
    -DCMAKE_INSTALL_PREFIX="$L3_INSTALL" \
    "${OCCA_FLAGS[@]}" -DENABLE_HYPRE_GPU="$HYPRE_GPU" -DENABLE_ADIOS=OFF -DENABLE_CVODE=OFF -DNEKRS_BUILD_FLOAT=OFF \
    > "$L3_LOGS/configure-$MODEL.log" 2>&1 \
    || { tail -40 "$L3_LOGS/configure-$MODEL.log"; echo "build.sh: configure failed (log: $L3_LOGS/configure-$MODEL.log)" >&2; exit 1; }
t0=$(date +%s)
cmake --build "$BUILD_DIR" --target install -j "$JOBS" > "$L3_LOGS/build-$MODEL.log" 2>&1 \
    || { tail -40 "$L3_LOGS/build-$MODEL.log"; echo "build.sh: build failed (log: $L3_LOGS/build-$MODEL.log)" >&2; exit 1; }
l3_fingerprint_write "$L3_INSTALL" "$FP"
echo "# built + installed in $(( $(date +%s)-t0 )) s: $L3_INSTALL/bin/nekrs (NEKRS_HOME=$L3_INSTALL)"
echo "# compiler warning lines: $(grep -c 'warning' "$L3_LOGS/build-$MODEL.log" || true)"
grep -E 'OCCA_CXX|OCCA_CUDA_COMPILER|NEKRS_FC|NEKRS_GPU_MPI|BACKEND' "$L3_INSTALL/nekrs.conf" 2>/dev/null | sed 's/^/# nekrs.conf: /'
