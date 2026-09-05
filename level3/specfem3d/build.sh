#!/usr/bin/env bash
# Build SPECFEM3D Cartesian (meshfem3D, decompose_mesh, generate_databases,
# specfem3D) with upstream's autotools build, CUDA-enabled, from a private
# copy of the source tree.
#
#   ./build.sh [CUDA|HIP]        (default CUDA)
#
# Layout (Level 3 isolation): read-only clone _upstream/level3/specfem3d;
# patched private source copy .deps/level3/specfem3d/src (autotools builds
# in-tree: obj/ and bin/ live there); install .deps/level3/specfem3d/install/bin;
# logs .deps/level3/specfem3d/logs. Only dependency besides MPI/CUDA is SCOTCH,
# bundled (external_libs/scotch_5.1.12b) and built by the same make.
#
# Toolchain: CC = conda GCC 13.3.0 (also nvcc's host compiler, first `gcc` on
# PATH as upstream's Makefile expects), FC = system gfortran 14.2.1 (the conda
# env has no gfortran), MPIFC = conda Open MPI's mpif90 pointed at that
# gfortran via OMPI_FC (class C). The conda MPI Fortran module loads under
# gfortran 14 (.mod format 15; verified with a 2-rank MPI Fortran test).
#
# Modifications (all recorded in patches/, provenance upstream devel
# cc2e9ffa, 2026-07-24):
#   class D  0001-cuda13-deviceOverlap-guard.patch -- v4.1.1 reads
#            cudaDeviceProp.deviceOverlap, a field removed in CUDA 13; upstream
#            devel guards it (CUDA_VERSION < 13000) and prints asyncEngineCount
#            instead. Diagnostic output only, no numerics. 10 lines.
#   class D  0002-blackwell-device-block.patch -- upstream devel's
#            GPU_DEVICE_Blackwell block (same content as Hopper's:
#            #undef USE_LAUNCH_BOUNDS). 8 lines.
#   class B  v4.1.1's configure knows --with-cuda=cuda4..cuda12 (cuda12 =
#            sm_90 + GPU_DEVICE_Hopper); devel added cuda13 = sm_100 +
#            GPU_DEVICE_Blackwell. Regenerating configure needs autoreconf and
#            the m4 submodule (absent), so the same result is obtained by
#            configuring with --with-cuda=cuda12 and overriding the GENCODE
#            make variable on the command line with upstream devel's cuda13
#            value. No upstream build file is edited.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
UP="$R/_upstream/level3/specfem3d"
[ -f "$UP/configure" ] || { echo "build.sh: $UP missing -- run $HERE/fetch.sh first" >&2; exit 1; }
SHA="$(git -C "$UP" rev-parse HEAD)"
l3_paths specfem3d
JOBS="${HPCPERF_BUILD_JOBS:-32}"
SYS_FC="${HPCPERF_SYSTEM_GFORTRAN:-/usr/bin/gfortran}"
[ -x "$SYS_FC" ] || { echo "build.sh: no gfortran at $SYS_FC (set HPCPERF_SYSTEM_GFORTRAN); the conda env has none" >&2; exit 1; }
export OMPI_FC="$SYS_FC"
PATCHES=("$HERE/patches/0001-cuda13-deviceOverlap-guard.patch" "$HERE/patches/0002-blackwell-device-block.patch")

case "$BACKEND" in
    CUDA)
        command -v nvcc >/dev/null || { echo "build.sh: nvcc not on PATH" >&2; exit 1; }
        ARCH="${HPCPERF_CUDA_ARCH:-$(l3_gpu_arch)}"
        CONF_GPU=(--with-cuda=cuda12)
        # = upstream devel's cuda13 GENCODE (sm_100 SASS + compute_100 PTX), written as two -gencode flags so
        # that no shell quoting travels through the make command line
        GENCODE="-gencode=arch=compute_${ARCH},code=sm_${ARCH} -gencode=arch=compute_${ARCH},code=compute_${ARCH} -DGPU_DEVICE_Blackwell"
        ARCHNOTE="sm_$ARCH" ;;
    HIP)
        command -v hipcc >/dev/null 2>&1 || { echo "build.sh: HIP requested but hipcc not found -- HIP build is UNTESTED on this machine (no ROCm)" >&2; exit 1; }
        # v4.1.1 knows --with-hip=MI8..MI250 (gfx803..gfx90a) only; devel added MI300/MI350 (gfx942/gfx950)
        CONF_GPU=(--with-hip=MI250); GENCODE=""; ARCHNOTE="gfx950 (UNTESTED; v4.1.1 has no MI350 option)" ;;
    *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac
CMAKE_OPTS="configure: FC=$SYS_FC CC=$CC MPIFC=mpif90(OMPI_FC=$SYS_FC) --with-mpi ${CONF_GPU[*]} USE_BUNDLED_SCOTCH=1; make GENCODE=${GENCODE:-default}"
PATCHNAMES=(); for p in "${PATCHES[@]}"; do PATCHNAMES+=("$(basename "$p")"); done
FP="$(l3_fingerprint_text specfem3d "$SHA" "$MODEL" "scotch=5.1.12b (bundled)" "$CMAKE_OPTS" "no (host-staged halo exchange in v4.1.1)" "${PATCHNAMES[@]}")"
l3_fingerprint_check "$L3_INSTALL" "$FP" || exit 1

echo "# SPECFEM3D $BACKEND: upstream $SHA (v4.1.1), arch $ARCHNOTE, MPI $(mpirun --version 2>/dev/null | head -1), FC $($SYS_FC --version | head -1), CC $($CC --version | head -1)"
rm -rf "$L3_SRC"; mkdir -p "$L3_SRC"
# private source copy (configure needs the top-level DATA/ defaults, which point into EXAMPLES/); doc/ stays in the clone
rsync -a --exclude .git --exclude doc "$UP/" "$L3_SRC/"
for p in "${PATCHES[@]}"; do
    (cd "$L3_SRC" && patch -p1 --forward --silent < "$p") || { echo "build.sh: patch $(basename "$p") failed to apply" >&2; exit 1; }
    echo "# applied $(basename "$p")"
done
cd "$L3_SRC"
CUDA_ROOT="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
# mpi.h for the nvcc-compiled GPU sources (configure's auto-detection via `mpif90 -showme:incdirs` did not
# reach the nvcc command line here); MPI_INC is the documented configure variable for this
MPI_INC_DIR="$(mpicc -showme:incdirs 2>/dev/null | awk '{print $1}')"; [ -f "$MPI_INC_DIR/mpi.h" ] || MPI_INC_DIR="$(dirname "$(dirname "$(command -v mpicc)")")/include"
[ -f "$MPI_INC_DIR/mpi.h" ] || { echo "build.sh: cannot locate mpi.h (looked in $MPI_INC_DIR)" >&2; exit 1; }
FC="$SYS_FC" CC="$CC" MPIFC=mpif90 MPI_INC="$MPI_INC_DIR" CUDA_INC="$CUDA_ROOT/include" CUDA_LIB="$CUDA_ROOT/lib64" USE_BUNDLED_SCOTCH=1 \
    ./configure --with-mpi "${CONF_GPU[@]}" > "$L3_LOGS/configure-$MODEL.log" 2>&1 \
    || { tail -40 "$L3_LOGS/configure-$MODEL.log"; echo "build.sh: configure failed (log: $L3_LOGS/configure-$MODEL.log)" >&2; exit 1; }
# bundled SCOTCH: the generated Makefile.inc enables gzip-compressed mesh files (-DCOMMON_FILE_COMPRESS_GZ,
# -lz); the conda GCC has no zlib.h in its sysroot and the meshes here are uncompressed -> build SCOTCH
# without that optional feature (class B, generated file only; upstream sources untouched)
sed -i -e 's/ -DCOMMON_FILE_COMPRESS_GZ//' -e 's/ -lz\b//' external_libs/scotch/src/Makefile.inc
t0=$(date +%s)
if [ -n "$GENCODE" ]; then MAKEVARS=("GENCODE=$GENCODE"); else MAKEVARS=(); fi
make -j "$JOBS" "${MAKEVARS[@]}" all > "$L3_LOGS/build-$MODEL.log" 2>&1 \
    || { grep -n -i 'error' "$L3_LOGS/build-$MODEL.log" | head -20 || true; echo "build.sh: build failed (log: $L3_LOGS/build-$MODEL.log)" >&2; exit 1; }
mkdir -p "$L3_INSTALL/bin"; cp -f bin/x* "$L3_INSTALL/bin/"
l3_fingerprint_write "$L3_INSTALL" "$FP"
echo "# built in $(( $(date +%s)-t0 )) s: $(ls "$L3_INSTALL/bin" | tr '\n' ' ')"
echo "# compiler warning lines: $(grep -c -i 'warning' "$L3_LOGS/build-$MODEL.log" || true)"
