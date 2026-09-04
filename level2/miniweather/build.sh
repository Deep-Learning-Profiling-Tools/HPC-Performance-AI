#!/usr/bin/env bash
# Build miniWeather (C++ YAKL parallel_for variant) for CUDA or HIP, out of tree.
#
#   ./build.sh [CUDA|HIP]        (default: CUDA)
#
# Configures upstream's cpp/CMakeLists.txt (YAKL backend selected through
# YAKL_ARCH) into  $R/build/level2/miniweather/<cuda|hip>  and builds the two
# executables upstream defines for the parallel_for version:
#   parallelfor       benchmark problem (size set at configure time, see below)
#   parallelfor_test  fixed validation problem (NX=100 NZ=50 SIM_TIME=400
#                     OUT_FREQ=400 DATA_SPEC_THERMAL, hard-coded upstream)
#
# The problem size is a compile-time constant in miniWeather (const.h reads
# _NX/_NZ/_SIM_TIME/_OUT_FREQ/_DATA_SPEC), so it is chosen here. Defaults are
# upstream's GPU grid (cpp/build/cmake_*_gpu.sh: 2048x1024, no NetCDF output)
# with upstream's default simulation time (1000 s), ~1 minute on a B200:
#   MINIWEATHER_NX         cells in x                 (default 2048)
#   MINIWEATHER_NZ         cells in z, keep NX = 2*NZ (default 1024)
#   MINIWEATHER_SIM_TIME   simulated seconds          (default 1000)
#   MINIWEATHER_OUT_FREQ   NetCDF output interval [s], -1 = no file output (default -1)
#   MINIWEATHER_DATA_SPEC  initial condition          (default DATA_SPEC_THERMAL)
# Other knobs:
#   HPCPERF_CUDA_ARCH   CUDA arch override, e.g. 100 or sm_100
#                       (default: detected from nvidia-smi compute_cap)
#   HPCPERF_HIP_ARCH    HIP --offload-arch (default gfx90a; HIP is untested here)
#   PNETCDF_PATH        PnetCDF prefix (default $CONDA_PREFIX, conda libpnetcdf)
#   MAKE_JOBS           parallel build jobs (default 4)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

BACKEND="$(printf '%s' "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
case "$BACKEND" in
    CUDA|HIP) ;;
    *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac

# Load the project toolchain (conda GCC 13.3 + OpenMPI 5 + PnetCDF, system
# CUDA 13.2) unless the calling shell already sourced it. hpcperf_env.sh is
# idempotent.
if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$R" ] && [ -f "$R/hpcperf_env.sh" ]; then
    # conda's activate.d scripts are not `set -u`/`set -e` safe; relax while sourcing.
    set +eu
    # shellcheck disable=SC1091
    source "$R/hpcperf_env.sh" 2>/dev/null
    set -eu
fi

JOBS="${MAKE_JOBS:-4}"
SRC="$HERE/cpp"
BUILD_DIR="$R/build/level2/miniweather/$(printf '%s' "$BACKEND" | tr '[:upper:]' '[:lower:]')"

NX="${MINIWEATHER_NX:-2048}"
NZ="${MINIWEATHER_NZ:-1024}"
SIM_TIME="${MINIWEATHER_SIM_TIME:-1000}"
OUT_FREQ="${MINIWEATHER_OUT_FREQ:--1}"
DATA_SPEC="${MINIWEATHER_DATA_SPEC:-DATA_SPEC_THERMAL}"

# PnetCDF (required by the code for its NetCDF output path, linked always).
PNETCDF_PATH="${PNETCDF_PATH:-${CONDA_PREFIX:-}}"
if [ -z "$PNETCDF_PATH" ] || [ ! -f "$PNETCDF_PATH/include/pnetcdf.h" ]; then
    echo "error: pnetcdf.h not found under '${PNETCDF_PATH:-<unset>}/include'; set PNETCDF_PATH" >&2
    exit 1
fi

# MPI include directories for the device compiler (nvcc/hipcc do not go through
# the mpicxx wrapper for the compile step; mpicxx is used for linking).
command -v mpicxx >/dev/null 2>&1 || { echo "error: mpicxx not found on PATH" >&2; exit 1; }
MPI_INC=""
for d in $(mpicxx --showme:incdirs 2>/dev/null); do
    [ "$d" = "$PNETCDF_PATH/include" ] || MPI_INC="$MPI_INC -I$d"
done

# Common upstream build variables (cpp/build/cmake_summit_gnu.sh style).
COMMON_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_COMPILER=mpicxx
    -DCMAKE_C_COMPILER=mpicc
    -DNX="$NX" -DNZ="$NZ" -DSIM_TIME="$SIM_TIME" -DOUT_FREQ="$OUT_FREQ" -DDATA_SPEC="$DATA_SPEC"
)

if [ "$BACKEND" = "CUDA" ]; then
    # GPU architecture: HPCPERF_CUDA_ARCH override, else detect from the GPU.
    SM_ARCH="${HPCPERF_CUDA_ARCH:-}"
    if [ -z "$SM_ARCH" ]; then
        SM_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                | head -1 | tr -d ' .')"
    fi
    SM_ARCH="${SM_ARCH#sm_}"
    if [ -z "$SM_ARCH" ]; then
        echo "error: could not detect the GPU compute capability; set HPCPERF_CUDA_ARCH (e.g. 100)" >&2
        exit 1
    fi

    echo "== miniWeather CUDA (YAKL_ARCH=CUDA): sm_${SM_ARCH}, host compiler ${CUDAHOSTCXX:-${CXX:-g++}}"
    echo "== problem: NX=$NX NZ=$NZ SIM_TIME=$SIM_TIME OUT_FREQ=$OUT_FREQ DATA_SPEC=$DATA_SPEC"
    echo "== build dir: $BUILD_DIR"
    # Flags as in upstream cpp/build/cmake_summit_gnu.sh / cmake_thatchroof_gnu_gpu.sh
    # (-DHAVE_MPI -DNO_INFORM -O3 --use_fast_math -arch sm_XX -I<pnetcdf>). The
    # MPI include dirs are passed with -I instead of "-ccbin mpic++" because CMake
    # already sets nvcc's host compiler from $CUDAHOSTCXX.
    mkdir -p "$BUILD_DIR"
    cmake -S "$SRC" -B "$BUILD_DIR" "${COMMON_ARGS[@]}" \
        -DYAKL_ARCH=CUDA \
        -DYAKL_CUDA_FLAGS="-DHAVE_MPI -DNO_INFORM -O3 --use_fast_math -arch sm_${SM_ARCH}${MPI_INC} -I${PNETCDF_PATH}/include" \
        -DLDFLAGS="-L${PNETCDF_PATH}/lib -lpnetcdf"
else
    if ! command -v hipcc >/dev/null 2>&1; then
        echo "error: hipcc not found on PATH -- HIP build is not possible on this machine (no ROCm)" >&2
        exit 1
    fi
    ROCM_PATH="${ROCM_PATH:-$(cd "$(dirname "$(command -v hipcc)")/.." && pwd)}"
    HIP_ARCH="${HPCPERF_HIP_ARCH:-gfx90a}"
    # YAKL's HIP backend compiles the .cpp sources with the C++ compiler using
    # "-x hip", so mpicxx must wrap hipcc (Open MPI honours OMPI_CXX).
    export OMPI_CXX=hipcc

    echo "== miniWeather HIP (YAKL_ARCH=HIP): --offload-arch=${HIP_ARCH}, ROCM_PATH=${ROCM_PATH} (untested)"
    echo "== problem: NX=$NX NZ=$NZ SIM_TIME=$SIM_TIME OUT_FREQ=$OUT_FREQ DATA_SPEC=$DATA_SPEC"
    echo "== build dir: $BUILD_DIR"
    # Flags follow upstream cpp/build/cmake_crusher_amd_gpu.sh.
    mkdir -p "$BUILD_DIR"
    cmake -S "$SRC" -B "$BUILD_DIR" "${COMMON_ARGS[@]}" \
        -DYAKL_ARCH=HIP \
        -DYAKL_HIP_FLAGS="-DHAVE_MPI -DNO_INFORM -O3 -ffast-math --rocm-path=${ROCM_PATH} --offload-arch=${HIP_ARCH} -x hip -Wno-unused-result${MPI_INC} -I${PNETCDF_PATH}/include" \
        -DLDFLAGS="-L${PNETCDF_PATH}/lib -lpnetcdf --rocm-path=${ROCM_PATH} -L${ROCM_PATH}/lib -lamdhip64"
fi

# Only the parallel_for executables are built (YAKL's unused Fortran interface
# library, which the YAKL CMake project also defines, is skipped).
cmake --build "$BUILD_DIR" -j"$JOBS" --target parallelfor parallelfor_test

# Upstream's ctest entry (YAKL_TEST) runs ./check_output.sh from the build dir.
cp "$SRC/build/check_output.sh" "$BUILD_DIR/check_output.sh"
echo "== built: $BUILD_DIR/parallelfor  $BUILD_DIR/parallelfor_test"
