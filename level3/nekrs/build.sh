#!/usr/bin/env bash
# Build nekRS (OCCA/CUDA backend, HYPRE on GPU) with upstream's CMake, from a
# private copy of the vendored source tree.
#
#   ./build.sh [CUDA|HIP]        (default CUDA)
#
# Layout (one frozen source tree per variant; generated state isolated per
# variant x backend profile): frozen source bundle level3/nekrs/src, materialized
# by tools/prepare_benchmark.sh for ONE variant (hypregpu = v26.0 + the three
# HYPRE CUDA-13 patches; cpucoarse = exact v26.0) -- the variant of the
# materialized tree must match HPCPERF_NEKRS_VARIANT; no patch is applied here.
# Profile = <variant>.<backend>: hypregpu.cuda, cpucoarse.cuda (validated),
# cpucoarse.hip (defined, untested: no ROCm here); hypregpu.hip does not exist
# and is refused (override HPCPERF_NEKRS_PROFILE must still name the backend).
# A build-side copy .deps/level3/nekrs/<profile>/src protects the frozen tree
# (upstream's third-party libraries -- OCCA, HYPRE 2.32, gslib, Nek5000, LAPACK
# -- are vendored in-tree and built by nekRS' own superbuild, nothing is shared
# with other applications or profiles); build build/level3/nekrs/<profile>;
# install .deps/level3/nekrs/<profile>/install (= NEKRS_HOME, with nekrs.conf
# recording the JIT toolchain); logs .deps/level3/nekrs/<profile>/logs; OCCA
# JIT cache .deps/level3/nekrs/<profile>/cache (run.sh).
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
l3_require_materialized "$HERE" || exit 3
UP="$HERE/src"
[ -f "$UP/CMakeLists.txt" ] || { echo "build.sh: $UP is not a nekRS tree -- run tools/prepare_benchmark.sh level3 nekrs --variant <hypregpu|cpucoarse>" >&2; exit 3; }

# --- variant selection (multi-variant build/install/cache isolation) ----------
# HPCPERF_NEKRS_HYPRE_GPU=ON|OFF selects whether the vendored HYPRE is built with
# its CUDA device backend (GPU coarse solve possible) or host-only (CPU coarse
# only). This is INDEPENDENT of OCCA_ENABLE_CUDA: the main application is on the
# GPU either way. Every variant x backend combination is its own profile
# (<variant>.<backend>) with a fully separate src copy, build, install, logs and
# JIT cache; nothing is shared between variants or between backends.
HYPRE_GPU="${HPCPERF_NEKRS_HYPRE_GPU:-ON}"
case "$HYPRE_GPU" in ON|OFF) : ;; *) echo "build.sh: HPCPERF_NEKRS_HYPRE_GPU must be ON or OFF" >&2; exit 2 ;; esac
VARIANT="${HPCPERF_NEKRS_VARIANT:-$([ "$HYPRE_GPU" = ON ] && echo hypregpu || echo cpucoarse)}"
MATERIALIZED="$(l3_materialized_variant "$HERE")"
[ "$MATERIALIZED" = "$VARIANT" ] || { echo "build.sh: the materialized source is variant '${MATERIALIZED:-unknown}' but variant '$VARIANT' was requested -- each variant is its own frozen tree: tools/prepare_benchmark.sh level3 nekrs --variant $VARIANT (--force-rematerialize discards the other variant's tree)" >&2; exit 3; }
SHA="$(l3_source_commit "$HERE" "$VARIANT")"; TREE_SHA="$(l3_source_tree_sha "$HERE" "$VARIANT")"
PROFILE="$(l3_backend_profile NEKRS "$MODEL" "$VARIANT")"     # <variant>.<backend>, e.g. hypregpu.cuda
# variant x backend: hypregpu is defined for CUDA only. Its frozen tree carries the three HYPRE CUDA-13
# patches and ENABLE_HYPRE_GPU=ON compiles HYPRE's CUDA device backend, so a HIP hypregpu configuration
# does not exist in this benchmark: it is refused here, before any profile directory is created.
# cpucoarse (host-only HYPRE) is the variant that pairs with the OCCA HIP backend (untested: no ROCm here).
if [ "$VARIANT" = hypregpu ] && [ "$MODEL" != cuda ]; then
    echo "build.sh: variant hypregpu is CUDA-only (HYPRE CUDA-13 patches, ENABLE_HYPRE_GPU=ON); backend $BACKEND is not defined for it -- use HPCPERF_NEKRS_VARIANT=cpucoarse HPCPERF_NEKRS_HYPRE_GPU=OFF for $BACKEND" >&2; exit 2
fi
l3_paths_profile nekrs "$PROFILE" "$MODEL" || exit 2
BUILD_DIR="$L3_BUILD"
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
# HYPRE is host-only and that code is not compiled, so the cpucoarse variant carries NO patch (the candidate
# tests whether the unused GPU component -- and therefore the patches -- can be dropped entirely).
# The patch series is part of the frozen variant tree (provenance/patch_series.<variant>.txt); only its
# names enter the fingerprint here (same identity as the validated installs).
PATCHNAMES=($(l3_lock_patches "$HERE" "$VARIANT"))
if [ "$HYPRE_GPU" = ON ]; then [ "${#PATCHNAMES[@]}" -eq 3 ] || { echo "build.sh: hypregpu tree must carry the 3 HYPRE patches, lock lists: ${PATCHNAMES[*]:-none}" >&2; exit 3; }
else [ "${#PATCHNAMES[@]}" -eq 0 ] || { echo "build.sh: cpucoarse tree must be unpatched, lock lists: ${PATCHNAMES[*]}" >&2; exit 3; }; fi

case "$BACKEND" in
    CUDA) command -v nvcc >/dev/null || { echo "build.sh: nvcc not on PATH" >&2; exit 1; }
          OCCA_FLAGS=(-DOCCA_ENABLE_CUDA=ON -DOCCA_ENABLE_HIP=OFF -DOCCA_ENABLE_DPCPP=OFF -DOCCA_ENABLE_OPENCL=OFF); ARCHNOTE="sm_$(l3_gpu_arch) (JIT at run time)" ;;
    HIP)  command -v hipcc >/dev/null 2>&1 || { echo "build.sh: HIP requested but hipcc not found -- HIP build is UNTESTED on this machine (no ROCm)" >&2; exit 1; }
          OCCA_FLAGS=(-DOCCA_ENABLE_CUDA=OFF -DOCCA_ENABLE_HIP=ON -DOCCA_ENABLE_DPCPP=OFF -DOCCA_ENABLE_OPENCL=OFF); ARCHNOTE="gfx950 (JIT at run time)" ;;
    *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac
CMAKE_OPTS="variant=$VARIANT ${OCCA_FLAGS[*]} ENABLE_HYPRE_GPU=$HYPRE_GPU ENABLE_ADIOS=OFF ENABLE_CVODE=OFF NEKRS_BUILD_FLOAT=OFF NEKRS_GPU_MPI=OFF(default; runtime NEKRS_GPU_MPI) CC=mpicc CXX=mpicxx FC=mpif90(OMPI_FC=$SYS_FC)"
FP="$(l3_fingerprint_text nekrs "$SHA" "$MODEL" "vendored: occa=2.0.0-dev hypre=2.32.0 gslib nek5000 lapack (in-tree)" "$CMAKE_OPTS" "runtime(NEKRS_GPU_MPI, default 0)" "${PATCHNAMES[@]}")"
l3_fingerprint_check "$L3_INSTALL" "$FP" || exit 1

echo "# nekRS $BACKEND profile=$PROFILE: variant=$VARIANT ENABLE_HYPRE_GPU=$HYPRE_GPU patches(pre-applied)=${#PATCHNAMES[@]} upstream $SHA (v26.0), frozen source tree $TREE_SHA, arch $ARCHNOTE, build-side copy $L3_SRC, install=$L3_INSTALL"
# build-side copy of the frozen tree (the frozen tree stays pristine; nekRS' superbuild is kept away from
# it). The copy is ~220 MB in many small files (slow on this filesystem), so it is reused only when it holds
# exactly this frozen tree (cache key = source_tree_sha256 of the materialized bundle).
STAMP="$SHA tree=$TREE_SHA"
if [ -f "$L3_SRC/.hpcperf-src-stamp" ] && [ "$(cat "$L3_SRC/.hpcperf-src-stamp")" = "$STAMP" ]; then
    echo "# reusing build-side source copy $L3_SRC ($STAMP)"
else
    rm -rf "$L3_SRC"; mkdir -p "$L3_SRC"
    rsync -a "$UP/" "$L3_SRC/"    # examples/ and doc/ are installed by nekRS' CMake
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
