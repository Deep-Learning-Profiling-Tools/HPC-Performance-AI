#!/usr/bin/env bash
# Build Nekbone (Nek5000's CG proxy, upstream branch cuda-openacc: OpenACC + CUDA Fortran) with upstream's own
# makenek / makefile.template, once per registered input.
#
#   ./build.sh [CUDA|CPU] [ID ...]
#
#   CUDA  (default) the GPU build upstream calls makenek.cuda: -acc -cuda (OpenACC solver, CUDA Fortran ax kernel)
#   CPU   upstream's makenek.mpi build of the same source without -acc/-cuda: the CPU code path (cg, ax).
#         Never measured; validate.sh compares it with the CUDA build as upstream's test harness does, and the
#         frozen correctness references (reference/<id>.json, check_cg.py) are made from it.
#
# Nekbone's problem maxima (polynomial order lx1, elements per rank lelt) are compile-time constants of the
# SIZE file, so every registered input (inputs.yaml, inputs/<id>/SIZE + data.rea) is a build of its own in
# $R/build/level2/nekbone/<cuda|cpu>/<id>/ (SIZE copy, generated makenek + makefile, obj/, nekbone). ID is a
# directory under inputs/ (default: all of them) or under test/ (test/nek_gpu1, the upstream example validate.sh
# uses). The vendored tree (src/, test/) is read-only: makenek writes the makefile and the objects into the
# build directory (SOURCE_ROOT points at src/).
#
# Toolchain: NVIDIA HPC SDK nvfortran / nvc (nvfortran on PATH, or HPCPERF_NVHPC_BIN=<sdk>/compilers/bin; the
# site's nvfortran may need NVLOCALRC, see README.md) with the repository's Open MPI (mpifort --showme flags,
# as level2/gamess_ri_mp2/build.sh: Nekbone's Fortran 77 `include 'mpif.h'` needs no Fortran MPI module).
# Flags (upstream makenek.cuda / makenek.mpi with the 2018 PGI spellings replaced by their current ones):
#   G           -acc -cuda -gpu=cc<ARCH> -Minfo=accel   (-cuda = -Mcuda, -gpu=ccNN = -ta=nvidia:ccNN)
#               plus -mcmodel=medium (the static arrays of the registered sizes exceed 2 GB) and the include
#               paths of MPI and of the ROI module; upstream's optimisation levels (-O2 / -O3 per file).
#   USR_LFLAGS  the same device flags, the MPI link line and the tools/timing ROI objects.
# Environment knobs:
#   HPCPERF_CUDA_ARCH   CUDA arch override, e.g. 100 or sm_100 (default: nvidia-smi compute_cap)
#   HPCPERF_NVHPC_BIN   directory holding nvfortran / nvc when they are not on PATH
#   MAKE_JOBS           parallel make jobs (default 8)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$R" ] && [ -f "$R/hpcperf_env.sh" ]; then
    set +eu
    # shellcheck disable=SC1091
    source "$R/hpcperf_env.sh" 2>/dev/null
    set -eu
fi
[ -n "${HPCPERF_NVHPC_BIN:-}" ] && export PATH="$HPCPERF_NVHPC_BIN:$PATH"

BACKEND="$(printf '%s' "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
case "$BACKEND" in
    CUDA|CPU) shift $(( $# > 0 ? 1 : 0 )) ;;
    HIP)  echo "build.sh: Nekbone's GPU port is OpenACC + CUDA Fortran (NVIDIA HPC SDK); no HIP build" >&2; exit 1 ;;
    *)    if [ -d "$HERE/inputs/$1" ] || [ -d "$HERE/test/$1" ]; then BACKEND=CUDA; else echo "usage: $0 [CUDA|CPU] [ID ...]" >&2; exit 2; fi ;;
esac
MODEL="$(printf '%s' "$BACKEND" | tr '[:upper:]' '[:lower:]')"
JOBS="${MAKE_JOBS:-8}"

NVFORTRAN="${NVFORTRAN:-$(command -v nvfortran || true)}"
NVC="${NVC:-$(command -v nvc || true)}"
[ -n "$NVFORTRAN" ] && [ -n "$NVC" ] || { echo "build.sh: nvfortran / nvc not found; Nekbone's OpenACC + CUDA Fortran port needs the NVIDIA HPC SDK (PATH or HPCPERF_NVHPC_BIN)" >&2; exit 1; }
# makenek.inc identifies the compiler by the bare command name (a path holding "mp", e.g. .../compilers/bin,
# is taken for an MPI wrapper): the compilers go into makenek by name, their directory onto PATH.
export PATH="$(dirname "$NVFORTRAN"):$PATH"
NVFORTRAN_CMD="$(basename "$NVFORTRAN")"; NVC_CMD="$(basename "$NVC")"
MPIFC="${MPIFC:-$(command -v mpifort || true)}"
[ -n "$MPIFC" ] || { echo "build.sh: mpifort (the repository's Open MPI) not found; source hpcperf_env.sh" >&2; exit 1; }
read -r -a MPI_COMPILE <<< "$($MPIFC --showme:compile 2>/dev/null)"
read -r -a MPI_LINK_RAW <<< "$($MPIFC --showme:link 2>/dev/null)"
MPI_LINK=()
for flag in "${MPI_LINK_RAW[@]}"; do          # the Fortran-module libraries belong to the MPI's own compiler
    case "$flag" in -lmpi_usempif08|-lmpi_usempi_ignore_tkr) ;; *) MPI_LINK+=("$flag");; esac
done

DEVFLAGS=""
if [ "$BACKEND" = CUDA ]; then
    ARCH="${HPCPERF_CUDA_ARCH:-}"
    [ -n "$ARCH" ] || ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
    ARCH="${ARCH#sm_}"
    [ -n "$ARCH" ] || { echo "error: could not detect the GPU compute capability; set HPCPERF_CUDA_ARCH (e.g. 100)" >&2; exit 1; }
    DEVFLAGS="-acc -cuda -gpu=cc${ARCH}"
fi

IDS=("$@")
if [ ${#IDS[@]} -eq 0 ]; then
    for d in "$HERE"/inputs/*/; do IDS+=("$(basename "$d")"); done
fi
for ID in "${IDS[@]}"; do
    if [ -f "$HERE/inputs/$ID/SIZE" ]; then CASE="$HERE/inputs/$ID"
    elif [ -f "$HERE/test/$ID/SIZE" ]; then CASE="$HERE/test/$ID"
    else echo "build.sh: no inputs/$ID/SIZE or test/$ID/SIZE" >&2; exit 2; fi
    BUILD="$R/build/level2/nekbone/$MODEL/$ID"
    mkdir -p "$BUILD"
    rm -rf "$BUILD/obj" "$BUILD/makefile" "$BUILD/nekbone" "$BUILD/compiler.out" "$BUILD/hpcperf_roi.mod"
    cp -p "$CASE/SIZE" "$BUILD/SIZE"
    # Nekbone's Fortran 77 `include 'mpif.h'`: the repository MPI's mpif.h pulls in mpif-sizeof.h (MPI_SIZEOF
    # interfaces with up to 15 array dimensions), which nvfortran rejects ("More than seven dimensions");
    # Nekbone never calls MPI_SIZEOF, so a build-local copy of mpif.h without that one include line is put in
    # front of the MPI include directory (the other mpif-*.h parts -- constants, handles, externals -- are the
    # MPI's own, so the binary links and runs against the repository's Open MPI).
    MPI_INC="$(for f in "${MPI_COMPILE[@]}"; do case "$f" in -I*) [ -f "${f#-I}/mpif.h" ] && { echo "${f#-I}"; break; };; esac; done)"
    [ -n "$MPI_INC" ] || { echo "build.sh: no mpif.h in the include directories of $MPIFC --showme:compile" >&2; exit 1; }
    mkdir -p "$BUILD/mpif77"
    sed "/include 'mpif-sizeof.h'/d" "$MPI_INC/mpif.h" > "$BUILD/mpif77/mpif.h"
    echo "== Nekbone $BACKEND: $ID (SIZE $(grep -o 'lx1=[0-9]*' "$CASE/SIZE" | head -1), $(grep -o 'lelt=[0-9]*' "$CASE/SIZE" | head -1)) -> $BUILD"
    # tools/timing ROI markers: the Fortran module and its C entry points (NVTX backend; tools/timing/roi/README.md)
    "$NVFORTRAN" -O2 -module "$BUILD" -c "$R/tools/timing/roi/hpcperf_roi.f90" -o "$BUILD/hpcperf_roi.o"
    "${CC:-cc}" -O2 -I"$R/tools/timing/roi" -c "$R/tools/timing/roi/hpcperf_roi_fortran.c" -o "$BUILD/hpcperf_roi_fortran.o"
    # upstream's makenek script for this build mode, with its site settings replaced (the "DON'T TOUCH" part is
    # upstream's: it sources src/makenek.inc, which generates the makefile from src/makefile.template, and runs make)
    if [ "$BACKEND" = CUDA ]; then TEMPLATE="$HERE/test/nek_gpu1/makenek.cuda"; else TEMPLATE="$HERE/test/nek_gpu1/makenek.mpi"; fi
    G="$DEVFLAGS${DEVFLAGS:+ -Minfo=accel} -mcmodel=medium -I$BUILD -I$BUILD/mpif77 ${MPI_COMPILE[*]}"
    LF="$DEVFLAGS ${MPI_LINK[*]} $BUILD/hpcperf_roi.o $BUILD/hpcperf_roi_fortran.o"
    sed -e "s?^SOURCE_ROOT=.*?SOURCE_ROOT=\"$HERE/src\"?" \
        -e "s?^F77=.*?F77=\"$NVFORTRAN_CMD\"?" \
        -e "s?^CC=.*?CC=\"$NVC_CMD\"?" \
        -e "s?^#\{0,1\}USR_LFLAGS=.*?USR_LFLAGS=\"$LF\"?" \
        -e "s?^#\{0,1\}G=.*?G=\"$G\"?" \
        "$TEMPLATE" > "$BUILD/makenek"
    chmod +x "$BUILD/makenek"
    ( cd "$BUILD" && MAKEFLAGS="-j$JOBS" ./makenek > makenek.log 2>&1 ) || true     # makenek always exits 0
    if [ ! -x "$BUILD/nekbone" ]; then
        echo "build.sh: $ID: no nekbone executable; last lines of $BUILD/compiler.out:" >&2
        tail -30 "$BUILD/compiler.out" >&2 2>/dev/null || tail -30 "$BUILD/makenek.log" >&2
        exit 1
    fi
    echo "== built: $BUILD/nekbone"
done
