#!/usr/bin/env bash
# Build Nyx 26.09 natively (CMake) against a PRIVATE AMReX 26.09 install and, for
# the heating/cooling variant, a private SUNDIALS 7.2.1 (the commit Nyx pins).
# Staged build, all inside one profile tree -- nothing shared with WarpX/Level 2:
#   [1] SUNDIALS (heatcool only)  .deps/level3/nyx/<profile>/install/sundials
#   [2] AMReX 26.09               .deps/level3/nyx/<profile>/install/amrex
#   [3] Nyx                       build/level3/nyx/<profile>  -> install/bin/nyx_*
#
#   ./build.sh [CUDA|HIP|CPU]          (default CUDA)
#   HPCPERF_NYX_HEATCOOL=NO|YES        (default NO -> adiabatic variant; YES -> SUNDIALS CVODE + CUDA fused kernels)
#   HPCPERF_NYX_PROFILE=<name>         (override the derived profile name)
#   HPCPERF_BUILD_JOBS=N               (default 32)
#
# Profiles: <backend><toolkit>-gcc<ver>-<variant>, e.g.
#   cuda132-gcc133-adiabatic   CUDA 13.2 / conda GCC 13.3 / Nyx_HEATCOOL=NO, sm_100
#   cuda132-gcc133-heatcool    same + Nyx_HEATCOOL=YES (SUNDIALS 7.2.1, ENABLE_CUDA, fused kernels)
#   cpu-gcc133-adiabatic       Nyx_GPU_BACKEND=NONE reference build; its AMReX also builds the
#                              plotfile tools (amrex_fcompare/fnan/fvolumesum/...) and this script
#                              adds particle_compare -- the official comparison tools validate.sh uses
#
# Why an external AMReX 26.09 instead of Nyx's submodule pin (6e875b7c): the
# pin's CMake drops SM >= 10.0 (convert_cuda_archs) and autodetects 8.6+PTX on
# this node -> an sm_86 binary (first attempt, removed; see README). AMReX 26.09
# (a52ca73, 21 commits ahead / 0 behind the pin) resolves sm_100 correctly. Nyx
# requires AMReX >= 20.11 and consumes it through find_package(AMReX CONFIG).
# AMReX options = exactly what Nyx's superbuild would set for the same Nyx options
# (cmake/NyxSetupAMReX.cmake: 3D, DOUBLE, PARTICLES(PDOUBLE), MPI, no OMP, no
# Fortran/PROBINIT, LINEAR_SOLVERS, SUNDIALS iff HEATCOOL, GPU backend); SUNDIALS
# options = cmake/NyxSetupSUNDIALS.cmake (CVODE only, index 32, fused kernels).
# Nyx options mirror upstream's GPU CI (Nyx_HYDRO=YES Nyx_MPI=YES Nyx_OMP=NO,
# CMAKE_CXX_STANDARD=17). Modification class: A (build options; out-of-source
# builds; no file of any checkout is modified).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
l3_isolate_build_env    # never see Level 2 .deps/install prefixes

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
HC="$(echo "${HPCPERF_NYX_HEATCOOL:-NO}" | tr '[:lower:]' '[:upper:]')"
case "$HC" in YES|NO) ;; *) echo "build.sh: HPCPERF_NYX_HEATCOOL must be YES or NO" >&2; exit 2;; esac
VARIANT=adiabatic; [ "$HC" = YES ] && VARIANT=heatcool
SRC="$R/_upstream/level3/Nyx"; AMREX_SRC="$R/_upstream/level3/amrex"
[ -f "$SRC/CMakeLists.txt" ] && [ -f "$AMREX_SRC/CMakeLists.txt" ] || { echo "build.sh: sources missing -- run $HERE/fetch.sh first" >&2; exit 1; }
[ "$HC" = NO ] || [ -f "$SRC/subprojects/sundials/CMakeLists.txt" ] || { echo "build.sh: sundials submodule missing -- run $HERE/fetch.sh" >&2; exit 1; }
SHA="$(git -C "$SRC" rev-parse HEAD)"; AMREX_SHA="$(git -C "$AMREX_SRC" rev-parse HEAD)"
SUNDIALS_SHA="$(git -C "$SRC/subprojects/sundials" rev-parse HEAD 2>/dev/null || echo none)"
GCC_MM="$(l3_version_mm "$("$CXX" -dumpfullversion 2>/dev/null || "$CXX" -dumpversion)")"

case "$BACKEND" in
    CUDA)
        ARCH="${HPCPERF_CUDA_ARCH:-$(l3_gpu_arch)}"; [ -n "$ARCH" ] || { echo "build.sh: cannot determine GPU arch (no GPU?) -- set HPCPERF_CUDA_ARCH" >&2; exit 1; }
        MODEL=cuda; ARCHNOTE="sm_$ARCH"; AMREX_GPU=CUDA
        ARCH_FLAGS=("-DCMAKE_CUDA_ARCHITECTURES=$ARCH" "-DCMAKE_CUDA_HOST_COMPILER=$CXX")
        PROFILE_DEFAULT="cuda$(l3_version_mm "$(l3_cuda_version)")-gcc${GCC_MM}-${VARIANT}" ;;
    HIP)
        command -v hipcc >/dev/null 2>&1 || { echo "build.sh: HIP requested but hipcc not found -- HIP build is UNTESTED on this machine (no ROCm)" >&2; exit 1; }
        ARCH="${HPCPERF_HIP_ARCH:-gfx950}"; MODEL=hip; ARCHNOTE="$ARCH"; AMREX_GPU=HIP
        ARCH_FLAGS=("-DAMReX_AMD_ARCH=$ARCH" -DCMAKE_CXX_COMPILER=hipcc)
        PROFILE_DEFAULT="hip-${ARCH}-${VARIANT}" ;;
    CPU|NONE)
        MODEL=cpu; ARCHNOTE=host; AMREX_GPU=NONE; ARCH_FLAGS=()
        PROFILE_DEFAULT="cpu-gcc${GCC_MM}-${VARIANT}" ;;
    *) echo "usage: $0 [CUDA|HIP|CPU]" >&2; exit 2 ;;
esac
PROFILE="${HPCPERF_NYX_PROFILE:-$PROFILE_DEFAULT}"
l3_paths_profile nyx "$PROFILE"
BUILD_DIR="$L3_BUILD"
JOBS="${HPCPERF_BUILD_JOBS:-32}"
AMREX_PREFIX="$L3_INSTALL/amrex"; SUND_PREFIX="$L3_INSTALL/sundials"
TOOLS=OFF; [ "$MODEL" = cpu ] && TOOLS=ON

AMREX_OPTS="AMReX_SPACEDIM=3 AMReX_PRECISION=DOUBLE AMReX_PARTICLES=ON AMReX_PARTICLES_PRECISION=DOUBLE AMReX_MPI=ON AMReX_OMP=OFF AMReX_FORTRAN=OFF AMReX_PROBINIT=OFF AMReX_LINEAR_SOLVERS=ON AMReX_EB=OFF AMReX_FFT=OFF AMReX_SUNDIALS=$( [ "$HC" = YES ] && echo ON || echo OFF) AMReX_GPU_BACKEND=$AMREX_GPU arch=$ARCHNOTE AMReX_PLOTFILE_TOOLS=$TOOLS"
CMAKE_OPTS="Nyx_GPU_BACKEND=$AMREX_GPU arch=$ARCHNOTE Nyx_HYDRO=YES Nyx_HEATCOOL=$HC Nyx_MPI=YES Nyx_OMP=NO Nyx_SINGLE_PRECISION_PARTICLES=NO CMAKE_CXX_STANDARD=17 CMAKE_BUILD_TYPE=Release amrex=external($AMREX_OPTS) sundials=$( [ "$HC" = YES ] && echo "external(ENABLE_CUDA=$( [ "$MODEL" = cuda ] && echo ON || echo OFF) INDEX_SIZE=32 FUSED_KERNELS=$( [ "$MODEL" = cuda ] && echo ON || echo OFF) CVODE only)" || echo off)"
DEPS="amrex=26.09($AMREX_SHA) [Nyx submodule pin $(git -C "$SRC/subprojects/amrex" rev-parse HEAD 2>/dev/null || echo unknown) not used: no sm_100 through CMake] sundials=$( [ "$HC" = YES ] && echo "7.2.1($SUNDIALS_SHA)" || echo off) profile=$PROFILE"
FP="$(l3_fingerprint_text nyx "$SHA" "$MODEL" "$DEPS" "$CMAKE_OPTS" "runtime(amrex.use_gpu_aware_mpi default)")"
l3_fingerprint_check "$L3_INSTALL" "$FP" || exit 1

COMMON=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX")
[ "$MODEL" = hip ] && COMMON=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DCMAKE_C_COMPILER="$CC")
echo "# Nyx $BACKEND profile=$PROFILE: Nyx $SHA (26.09), AMReX $AMREX_SHA (26.09, external), sundials $( [ "$HC" = YES ] && echo "$SUNDIALS_SHA (7.2.1)" || echo off), arch $ARCHNOTE, host $CXX, MPI $(mpirun --version 2>/dev/null | head -1)"
echo "# resources: -j$JOBS; expected AMReX 5-10 min + Nyx 2-5 min (+ SUNDIALS ~3 min); trees $L3_BUILD_DEPS, $BUILD_DIR"
t0=$(date +%s)

stage() { # stage <name> <src> <build> <log-prefix> [cmake options...]
    local name=$1 src=$2 bld=$3 log=$4; shift 4
    mkdir -p "$bld"
    cmake -S "$src" -B "$bld" "${COMMON[@]}" "$@" > "$L3_LOGS/$log-configure.log" 2>&1 \
        || { tail -40 "$L3_LOGS/$log-configure.log"; echo "build.sh: $name configure failed (log: $L3_LOGS/$log-configure.log)" >&2; exit 1; }
    cmake --build "$bld" -j "$JOBS" > "$L3_LOGS/$log-build.log" 2>&1 \
        || { tail -40 "$L3_LOGS/$log-build.log"; echo "build.sh: $name build failed (log: $L3_LOGS/$log-build.log)" >&2; exit 1; }
}

# [1] SUNDIALS 7.2.1 (Nyx's pinned submodule commit), options from cmake/NyxSetupSUNDIALS.cmake
if [ "$HC" = YES ] && [ ! -f "$SUND_PREFIX/.hpcperf-stage-done" ]; then
    SUND_GPU=(-DENABLE_CUDA=OFF)
    [ "$MODEL" = cuda ] && SUND_GPU=(-DENABLE_CUDA=ON -DSUNDIALS_INDEX_SIZE=32 -DSUNDIALS_BUILD_PACKAGE_FUSED_KERNELS=ON "${ARCH_FLAGS[@]}")
    [ "$MODEL" = hip ] && SUND_GPU=(-DENABLE_HIP=ON -DSUNDIALS_BUILD_PACKAGE_FUSED_KERNELS=ON)
    stage SUNDIALS "$SRC/subprojects/sundials" "$L3_BUILD_DEPS/sundials" sundials \
        -DCMAKE_INSTALL_PREFIX="$SUND_PREFIX" -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON \
        -DEXAMPLES_ENABLE_C=OFF -DEXAMPLES_ENABLE_CXX=OFF -DEXAMPLES_INSTALL=OFF -DENABLE_MPI=OFF -DENABLE_OPENMP=OFF \
        -DBUILD_ARKODE=OFF -DBUILD_KINSOL=OFF -DBUILD_IDA=OFF -DBUILD_IDAS=OFF -DBUILD_CVODES=OFF -DBUILD_TESTING=OFF \
        "${SUND_GPU[@]}"
    cmake --install "$L3_BUILD_DEPS/sundials" > "$L3_LOGS/sundials-install.log" 2>&1 || { echo "build.sh: SUNDIALS install failed" >&2; exit 1; }
    touch "$SUND_PREFIX/.hpcperf-stage-done"
fi
t1=$(date +%s)

# [2] AMReX 26.09, component set identical to what Nyx's superbuild sets for these Nyx options
if [ ! -f "$AMREX_PREFIX/.hpcperf-stage-done" ]; then
    AMREX_SUND=(-DAMReX_SUNDIALS=OFF); [ "$HC" = YES ] && AMREX_SUND=(-DAMReX_SUNDIALS=ON "-DSUNDIALS_ROOT=$SUND_PREFIX")
    stage AMReX "$AMREX_SRC" "$L3_BUILD_DEPS/amrex" amrex \
        -DCMAKE_INSTALL_PREFIX="$AMREX_PREFIX" -DBUILD_SHARED_LIBS=OFF \
        -DAMReX_SPACEDIM=3 -DAMReX_PRECISION=DOUBLE -DAMReX_PARTICLES=ON -DAMReX_PARTICLES_PRECISION=DOUBLE \
        -DAMReX_MPI=ON -DAMReX_OMP=OFF -DAMReX_FORTRAN=OFF -DAMReX_PROBINIT=OFF -DAMReX_LINEAR_SOLVERS=ON \
        -DAMReX_EB=OFF -DAMReX_FFT=OFF -DAMReX_AMRDATA=OFF -DAMReX_BUILD_TUTORIALS=OFF -DAMReX_INSTALL=ON \
        "-DAMReX_GPU_BACKEND=$AMREX_GPU" -DAMReX_PLOTFILE_TOOLS="$TOOLS" "${ARCH_FLAGS[@]}" "${AMREX_SUND[@]}"
    cmake --install "$L3_BUILD_DEPS/amrex" > "$L3_LOGS/amrex-install.log" 2>&1 || { echo "build.sh: AMReX install failed" >&2; exit 1; }
    if [ "$MODEL" = cuda ]; then
        archs="$(/usr/bin/grep -h 'AMREX_CUDA_ARCHS:INTERNAL' "$L3_BUILD_DEPS/amrex/CMakeCache.txt" | cut -d= -f2)"
        [ "$archs" = "$ARCH" ] || { echo "build.sh: AMReX resolved CUDA archs '$archs', expected '$ARCH' -- refusing" >&2; exit 1; }
    fi
    touch "$AMREX_PREFIX/.hpcperf-stage-done"
fi
t2=$(date +%s)

# [3] Nyx against the private AMReX (+SUNDIALS). ENABLE_CUDA=ON makes Nyx include
# AMReXTargetHelpers (setup_target_for_cuda_compilation) in the external-AMReX branch.
NYX_GPU=(-DNyx_GPU_BACKEND="$AMREX_GPU")
[ "$MODEL" = cuda ] && NYX_GPU+=(-DENABLE_CUDA=ON "${ARCH_FLAGS[@]}")
[ "$MODEL" = hip ] && NYX_GPU+=("${ARCH_FLAGS[@]}")
NYX_SUND=(); [ "$HC" = YES ] && NYX_SUND=("-DSUNDIALS_ROOT=$SUND_PREFIX")
# Nyx_SINGLE_PRECISION_PARTICLES=NO: double-precision particles, as upstream's nightly
# regression builds (GNU make default; LyA-adiabatic passes USE_SINGLE_PRECISION_PARTICLES=FALSE
# explicitly) -- the CMake default would be single precision (PSINGLE).
stage Nyx "$SRC" "$BUILD_DIR" nyx \
    "-DAMReX_ROOT=$AMREX_PREFIX" "-DCMAKE_PREFIX_PATH=$AMREX_PREFIX" \
    -DNyx_HYDRO=YES -DNyx_HEATCOOL="$HC" -DNyx_MPI=YES -DNyx_OMP=NO -DNyx_SINGLE_PRECISION_PARTICLES=NO \
    "${NYX_GPU[@]}" "${NYX_SUND[@]}"
/usr/bin/grep -q 'AMReX found: configuration file located at' "$L3_LOGS/nyx-configure.log" \
    || { echo "build.sh: Nyx did not pick up the external AMReX (would have fallen back to the submodule)" >&2; exit 1; }
t3=$(date +%s)

mkdir -p "$L3_INSTALL/bin"
for exe in Exec/MiniSB/nyx_MiniSB Exec/LyA/nyx_LyA Exec/AMR-density/nyx_AMR-density; do
    [ -x "$BUILD_DIR/$exe" ] && cp -f "$BUILD_DIR/$exe" "$L3_INSTALL/bin/"
done
[ -x "$L3_INSTALL/bin/nyx_MiniSB" ] && [ -x "$L3_INSTALL/bin/nyx_LyA" ] || { echo "build.sh: nyx_MiniSB / nyx_LyA not produced under $BUILD_DIR/Exec" >&2; exit 1; }
if [ "$MODEL" = cuda ]; then
    for exe in "$L3_INSTALL"/bin/nyx_*; do
        got="$(cuobjdump --list-elf "$exe" 2>/dev/null | /usr/bin/grep -o 'sm_[0-9a-z]*' | sort -u | paste -sd,)"
        [ "$got" = "sm_$ARCH" ] || { echo "build.sh: $(basename "$exe") embeds '$got', expected sm_$ARCH -- refusing to install" >&2; exit 1; }
    done
fi
if [ "$MODEL" = cpu ]; then
    for t in fcompare fnan fvolumesum fextrema fvarnames ftime; do
        [ -x "$AMREX_PREFIX/bin/amrex_$t" ] && cp -f "$AMREX_PREFIX/bin/amrex_$t" "$L3_INSTALL/bin/"
    done
    [ -x "$L3_INSTALL/bin/amrex_fcompare" ] || { echo "build.sh: amrex_fcompare not installed by AMReX (AMReX_PLOTFILE_TOOLS)" >&2; exit 1; }
    # particle_compare (AMReX Tools/Postprocessing/C_Src) against the installed AMReX
    PC_SRC="$L3_SRC/particle_compare"; rm -rf "$PC_SRC"; mkdir -p "$PC_SRC"
    cp "$AMREX_SRC/Tools/Postprocessing/C_Src/particle_compare.cpp" "$PC_SRC/"
    cat > "$PC_SRC/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.24)
project(particle_compare C CXX)   # AMReXConfig's find_dependency(MPI) needs the C language enabled
find_package(AMReX REQUIRED CONFIG)
add_executable(particle_compare particle_compare.cpp)
target_link_libraries(particle_compare PRIVATE AMReX::amrex_3d)
EOF
    stage particle_compare "$PC_SRC" "$L3_BUILD_DEPS/particle_compare" particle_compare "-DAMReX_ROOT=$AMREX_PREFIX"
    cp -f "$L3_BUILD_DEPS/particle_compare/particle_compare" "$L3_INSTALL/bin/"
fi
l3_fingerprint_write "$L3_INSTALL" "$FP"
{
    echo "profile=$PROFILE backend=$MODEL variant=$VARIANT jobs=$JOBS utc=$(date -u +%FT%TZ)"
    echo "seconds: sundials=$((t1 - t0)) amrex=$((t2 - t1)) nyx=$((t3 - t2)) total=$(( $(date +%s) - t0 ))"
    echo "amrex=$AMREX_SHA (26.09) options: $AMREX_OPTS"
    [ "$HC" = YES ] && echo "sundials=$SUNDIALS_SHA (7.2.1)"
    for b in "$L3_INSTALL"/bin/*; do echo "$(basename "$b") sha256=$(l3_sha_file "$b")"; done
    if [ "$MODEL" = cuda ]; then
        echo "AMREX_CUDA_ARCHS=$(/usr/bin/grep -h 'AMREX_CUDA_ARCHS:INTERNAL' "$L3_BUILD_DEPS/amrex/CMakeCache.txt" | cut -d= -f2)"
        echo "cuobjdump(nyx_MiniSB): $(cuobjdump --list-elf "$L3_INSTALL/bin/nyx_MiniSB" 2>/dev/null | /usr/bin/grep -o 'sm_[0-9a-z]*' | sort -u | paste -sd,) (cudart static; ldd shows $(ldd "$L3_INSTALL/bin/nyx_MiniSB" | /usr/bin/grep -oE 'lib(cudart|cuda|cusparse|cublas|curand)[^ ]*' | sort -u | paste -sd, || echo none))"
    fi
} > "$L3_INSTALL/BUILD_INFO.txt"
echo "# built in $(( $(date +%s) - t0 )) s (sundials $((t1 - t0)), amrex $((t2 - t1)), nyx $((t3 - t2))): $(ls "$L3_INSTALL/bin" | paste -sd' ') (installed under $L3_INSTALL)"
echo "# compiler warning lines (nyx): $(/usr/bin/grep -c 'warning' "$L3_LOGS/nyx-build.log" || true)"
[ "$MODEL" = cuda ] && /usr/bin/grep -E 'AMREX_CUDA_ARCHS|cuobjdump' "$L3_INSTALL/BUILD_INFO.txt"
exit 0
