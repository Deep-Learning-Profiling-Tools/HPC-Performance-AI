#!/usr/bin/env bash
# Build CP2K v2026.2 (psmp: MPI + OpenMP + CUDA) with upstream's own dependency
# bootstrap (tools/toolchain/install_cp2k_toolchain.sh, "NATIVE+toolchain") and
# upstream's CMake, in a private profile tree. Staged:
#   [A] toolchain: OpenBLAS, ScaLAPACK, FFTW3, libint (lmax 5), libxc, LIBXSMM/LIBXS,
#       spglib, DBCSR 2.10.0 (CPU + CUDA sm_100)   -> .deps/level3/cp2k/<profile>/install/toolchain
#   [B] DBCSR official tests (its own ctest suite, BUILD_TESTING=ON, CUDA sm_100, MPI+OpenMP)
#       -> build/dbcsr-test; results in logs/dbcsr-ctest.log  (gate: all tests must pass)
#   [C] CP2K CMake configure/build/install                    -> install/cp2k/bin/cp2k.psmp
#
#   ./build.sh [CUDA]                 (HIP: not attempted here -- no ROCm on this node)
#   HPCPERF_BUILD_JOBS=N              (default 32; also NPROCS_OVERWRITE for the toolchain)
#   HPCPERF_CP2K_PROFILE=<name>       (override the derived profile name)
#   HPCPERF_CP2K_SKIP_DBCSR_TEST=1    (debug only: skip stage B; the fingerprint then records dbcsr_tests=SKIPPED)
#
# Toolchain (profile cuda132-gcc142-ompi5010):
#   compilers   system GCC 14.2.1 (gcc/g++/gfortran) for everything incl. the nvcc host
#               compiler (the conda GCC 13.3 has no gfortran; a single GCC for C/C++/
#               Fortran avoids the mixed-GCC LTO/PIE issues met by SPECFEM/nekRS)
#   MPI         conda Open MPI 5.0.10 (the site-validated launcher/transport); wrappers
#               redirected to the system GCC via OMPI_CC/OMPI_CXX/OMPI_FC (probe: C and
#               mpi_f08 Fortran programs build and run with 2 ranks)
#   CUDA        13.2.78, sm_100 (--gpu-ver=B200 -> ARCH_NUM 100 via the backported upstream
#               patch in patches/; CP2K itself gets -DCMAKE_CUDA_ARCHITECTURES=100, which
#               v2026.2 accepts natively)
#   options     --with-elpa=no --with-cosma=no --with-sirius=no --with-tblite=no --with-libvori=no
#               (minimal GPW-DFT stack: the selected cases use OT, not diagonalisation);
#               everything the toolchain would download beyond that is left off
# Modification classes: B (toolchain scripts: patches/0001-*, upstream backport) + C
# (OMPI_* compiler redirection, NPROCS_OVERWRITE). No CP2K or DBCSR source changed by us;
# the DBCSR CMakeLists/parameters edit is upstream's own sed (see the patch header).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
l3_isolate_build_env
l3_clean_conda_build_env   # conda's LDFLAGS carry -Wl,--disable-new-dtags/-rpath <conda lib>: they made cp2k.psmp/libcp2k.so
                           # resolve libopenblas.so.0 to the conda OpenBLAS (pthreads build) instead of the toolchain's (found 2026-09-06)

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
[ "$BACKEND" = CUDA ] || { echo "build.sh: only CUDA is implemented for CP2K here (HIP: no ROCm on this node, UNTESTED)" >&2; exit 2; }
# Sources come ONLY from the frozen bundle materialized here (tools/prepare_benchmark.sh): src/ = CP2K
# v2026.2 whose tools/toolchain already carries the B200 back-port; deps/cp2k-toolchain-dist/ = the package
# tarballs the toolchain installer would download. Nothing is fetched, cloned or patched by this script.
l3_require_materialized "$HERE" || exit 3
SRC="$HERE/src"; TC_DIST="$HERE/deps/cp2k-toolchain-dist"
[ -f "$SRC/CMakeLists.txt" ] && [ -f "$SRC/tools/toolchain/install_cp2k_toolchain.sh" ] && [ -d "$TC_DIST" ] || { echo "build.sh: src/ or deps/cp2k-toolchain-dist incomplete -- run tools/prepare_benchmark.sh level3 cp2k" >&2; exit 3; }
SHA="$(l3_source_commit "$HERE")"; TREE_SHA="$(l3_source_tree_sha "$HERE")"
# system GCC 14 toolchain for C/C++/Fortran; conda Open MPI wrappers redirected to it
export CC=/usr/bin/gcc CXX=/usr/bin/g++ FC=/usr/bin/gfortran F90=/usr/bin/gfortran F77=/usr/bin/gfortran
export OMPI_CC=/usr/bin/gcc OMPI_CXX=/usr/bin/g++ OMPI_FC=/usr/bin/gfortran CUDAHOSTCXX=/usr/bin/g++
for c in gcc g++ gfortran; do [ "$(command -v $c)" = "/usr/bin/$c" ] || { echo "build.sh: '$c' resolves to $(command -v $c), expected /usr/bin/$c" >&2; exit 1; }; done
for c in mpicc mpic++ mpifort mpiexec cmake python3 nvcc; do command -v $c >/dev/null || { echo "build.sh: $c not in PATH" >&2; exit 1; }; done
GCC_MM="$(l3_version_mm "$(/usr/bin/gcc -dumpfullversion)")"
OMPI_V="$(mpirun --version 2>/dev/null | head -1 | /usr/bin/grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
ARCH="${HPCPERF_CUDA_ARCH:-$(l3_gpu_arch)}"; [ "$ARCH" = 100 ] || { echo "build.sh: this profile is defined for sm_100 (B200); detected sm_$ARCH -- set HPCPERF_CUDA_ARCH deliberately if you mean it" >&2; exit 1; }
GPUVER=B200
PROFILE="${HPCPERF_CP2K_PROFILE:-cuda$(l3_version_mm "$(l3_cuda_version)")-gcc${GCC_MM}-ompi$(echo "$OMPI_V" | tr -d .)}"
l3_paths_profile cp2k "$PROFILE" cuda || exit 2
JOBS="${HPCPERF_BUILD_JOBS:-32}"; export NPROCS_OVERWRITE="$JOBS"
# The toolchain's private source/build copy lives on the node's LOCAL disk, outside any git
# work tree: DBCSR 2.10.0's cmake/GetGitRevisionDescription.cmake walks up from its source
# directory, finds this repository's `.git` *file* (git worktree) and mis-resolves the
# absolute gitdir it points to as a relative path -> "file failed to open for reading" and
# a configure abort. Everything the later stages need (install prefix, setup/toolchain.conf,
# logs) stays under the profile tree; $L3_SRC/toolchain is a symlink to the scratch copy.
TC_SCRATCH="${HPCPERF_CP2K_TOOLCHAIN_SCRATCH:-/tmp/hpcperf-l3-b2-scratch/cp2k-toolchain/$PROFILE}"
mkdir -p "$(dirname "$TC_SCRATCH")"; [ -L "$L3_SRC/toolchain" ] || { rm -rf "$L3_SRC/toolchain"; ln -sfn "$TC_SCRATCH" "$L3_SRC/toolchain"; }
TC_SRC="$TC_SCRATCH"; TC_INSTALL="$L3_INSTALL/toolchain"; CP2K_PREFIX="$L3_INSTALL/cp2k"
PATCHES=("$HERE/patches/0001-toolchain-b200-backport-cp2k-378b2fab.patch")   # already applied in the frozen src/tools/toolchain; content hash kept in the fingerprint
[ "$(l3_lock_patches "$HERE")" = "$(basename "${PATCHES[0]}")" ] || { echo "build.sh: the lock's patch series ($(l3_lock_patches "$HERE")) differs from the expected $(basename "${PATCHES[0]}")" >&2; exit 3; }
TC_OPTS=(--install-dir="$TC_INSTALL" --mpi-mode=openmpi --math-mode=openblas --with-gcc=system --with-openmpi=system --with-cmake=system
         --enable-cuda=yes --gpu-ver=$GPUVER --libint-lmax=5
         --with-openblas=install --with-scalapack=install --with-fftw=install --with-libint=install --with-libxc=install
         --with-libxsmm=install --with-libxs=install --with-dbcsr=install --with-spglib=install
         --with-elpa=no --with-cosma=no --with-sirius=no --with-tblite=no --with-libvori=no --with-hdf5=no --with-plumed=no
         --with-libtorch=no --with-gsl=no --with-dftd4=no --with-spla=no --with-spfft=no --with-gauxc=no --with-libsmeagol=no
         --with-deepmd=no --with-ace=no --with-greenx=no --with-trexio=no --with-libfci=no --with-mcl=no --with-libgint=no --with-cusolvermp=no)
# Installed binaries carry an RPATH to the toolchain library directories (and libcp2k.so) so that the BLAS/LAPACK,
# ScaLAPACK, FFTW, libxc, ... actually used at run time are the toolchain's, independent of LD_LIBRARY_PATH ordering;
# run.sh verifies the resolution with ldd before every run.
TC_RPATH="$( { ls -d "$TC_INSTALL"/*/lib "$TC_INSTALL"/*/lib64 2>/dev/null || true; } | paste -sd';')"   # (ls exits 2 when no lib64 exists: keep set -e/pipefail quiet)
# BLAS/LAPACK: the toolchain's static OpenBLAS (its own convention, MATH_LIBS="-l:libopenblas.a") through CP2K's CUSTOM
# vendor -- a dynamic -lopenblas resolved at run time to whichever libopenblas.so.0 the loader met first (the conda MPI
# wrapper puts its rpath before ours), which was the conda pthreads OpenBLAS in attempts 1 and 2.
TC_OPENBLAS_A="$(ls "$TC_INSTALL"/openblas-*/lib/libopenblas.a 2>/dev/null | head -1)"
[ -f "$TC_OPENBLAS_A" ] || { echo "build.sh: toolchain libopenblas.a not found under $TC_INSTALL" >&2; exit 1; }
TC_RPATH_COLON="${TC_RPATH//;/:}"
CP2K_CMAKE=(-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON "-DCMAKE_INSTALL_PREFIX=$CP2K_PREFIX" "-DCP2K_DATA_DIR=$SRC/data"
            "-DCMAKE_INSTALL_RPATH=$CP2K_PREFIX/lib;$TC_RPATH" -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
            "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,$CP2K_PREFIX/lib:$TC_RPATH_COLON" "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath,$CP2K_PREFIX/lib:$TC_RPATH_COLON"
            -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_Fortran_COMPILER=/usr/bin/gfortran
            -DCP2K_USE_MPI=ON -DCP2K_USE_MPI_F08=ON -DCP2K_USE_FFTW3=ON -DCP2K_USE_LIBXC=ON -DCP2K_USE_LIBINT2=ON
            -DCP2K_USE_LIBXS=ON -DCP2K_USE_LIBXSMM=ON -DCP2K_USE_SPGLIB=ON
            -DCP2K_USE_ELPA=OFF -DCP2K_USE_COSMA=OFF -DCP2K_USE_SIRIUS=OFF -DCP2K_USE_TBLITE=OFF -DCP2K_USE_VORI=OFF -DCP2K_USE_DFTD4=OFF
            -DCP2K_USE_HDF5=OFF -DCP2K_USE_PLUMED=OFF -DCP2K_USE_LIBTORCH=OFF -DCP2K_USE_GAUXC=OFF -DCP2K_USE_GREENX=OFF -DCP2K_USE_TREXIO=OFF
            -DCP2K_USE_ACE=OFF -DCP2K_USE_DEEPMD=OFF -DCP2K_USE_LIBFCI=OFF -DCP2K_USE_MIMIC=OFF -DCP2K_USE_LIBSMEAGOL=OFF -DCP2K_USE_SPLA=OFF
            -DCP2K_BLAS_VENDOR=CUSTOM "-DCP2K_BLAS_LINK_LIBRARIES=$TC_OPENBLAS_A" "-DCP2K_LAPACK_LINK_LIBRARIES=$TC_OPENBLAS_A" -DCP2K_SCALAPACK_VENDOR=GENERIC
            -DCP2K_USE_ACCEL=CUDA "-DCMAKE_CUDA_ARCHITECTURES=$ARCH" -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++)
CMAKE_OPTS="blas=CUSTOM:toolchain-libopenblas.a(static) install_rpath=toolchain-first ${CP2K_CMAKE[*]} | toolchain: ${TC_OPTS[*]}"
DEPS="toolchain(install_cp2k_toolchain.sh v2026.2 + backport 378b2fab) dbcsr=2.10.0(sha256 3d897220fbb4498215331efad6905eb7744881b4cf04eb5c5fb4db7c48a56ef9; B200 entry=arch 100, libsmm_acc parameters=H100 reused) openblas/scalapack/fftw3/libint(lmax5)/libxc/libxsmm/libxs/spglib=toolchain pins gcc=$(/usr/bin/gcc -dumpfullversion) openmpi=$OMPI_V profile=$PROFILE dbcsr_tests=$( [ -n "${HPCPERF_CP2K_SKIP_DBCSR_TEST:-}" ] && echo SKIPPED || echo required)"
FP="$(l3_fingerprint_text cp2k "$SHA" cuda "$DEPS" "$CMAKE_OPTS" "not-used(DBCSR/DBM communicate through host buffers)" "${PATCHES[@]}")"
l3_fingerprint_check "$L3_INSTALL" "$FP" || exit 1

echo "# CP2K $BACKEND profile=$PROFILE: cp2k $SHA (v2026.2, frozen source tree $TREE_SHA), gcc $(/usr/bin/gcc -dumpfullversion), gfortran $(/usr/bin/gfortran -dumpfullversion), $(mpirun --version | head -1), CUDA $(l3_cuda_version) sm_$ARCH, -j$JOBS"
echo "# resources: toolchain ~1.5-3 h (libint lmax 5 dominates), DBCSR tests ~10 min, CP2K ~40-90 min; disk ~10-15 GB under $L3_DEPS"
t0=$(date +%s)

# [A] toolchain (build-side copy of the frozen src/tools/toolchain, which already carries the B200 back-port;
#     the frozen tree is never written to). The package tarballs of the bundle are pre-seeded into the
#     installer's build directory: retrieve_package() finds them, verifies the same sha256 and does not download.
if [ ! -f "$TC_INSTALL/.hpcperf-stage-done" ]; then
    if [ ! -f "$TC_SRC/.hpcperf-src-stamp" ] || [ "$(cat "$TC_SRC/.hpcperf-src-stamp")" != "$SHA tree=$TREE_SHA" ]; then
        rm -rf "$TC_SRC"; mkdir -p "$TC_SRC"; cp -r "$SRC/tools/toolchain/." "$TC_SRC/"
        echo "$SHA tree=$TREE_SHA" > "$TC_SRC/.hpcperf-src-stamp"
    fi
    mkdir -p "$TC_SRC/build"; cp -n "$TC_DIST"/* "$TC_SRC/build/"
    echo "# [A] toolchain: ${TC_OPTS[*]}"
    # the toolchain installer writes `declare -x` of its whole environment into <install>/toolchain.env:
    # run it under the allow-listed environment so that no login-shell secret can end up in that file
    ( cd "$TC_SRC" && unset CMAKE_GENERATOR && l3_clean_env_exec ./install_cp2k_toolchain.sh "${TC_OPTS[@]}" ) > "$L3_LOGS/toolchain.log" 2>&1 \
        || { tail -60 "$L3_LOGS/toolchain.log"; echo "build.sh: toolchain failed (log: $L3_LOGS/toolchain.log)" >&2; exit 1; }
    [ -f "$TC_INSTALL/setup" ] && [ -f "$TC_INSTALL/toolchain.conf" ] || { echo "build.sh: toolchain produced no setup/toolchain.conf under $TC_INSTALL" >&2; exit 1; }
    /usr/bin/grep -q 'GPU_ARCH_NUMBER_B200 100' "$TC_SRC/build/dbcsr-2.10.0/CMakeLists.txt" || { echo "build.sh: DBCSR CMakeLists did not receive the B200 (arch 100) entry" >&2; exit 1; }
    touch "$TC_INSTALL/.hpcperf-stage-done"
fi
t1=$(date +%s)
set +u; # shellcheck disable=SC1091
source "$TC_INSTALL/setup"; set -u
# the toolchain's setup may re-point compilers; keep the system GCC and the conda MPI wrappers
export CC=/usr/bin/gcc CXX=/usr/bin/g++ FC=/usr/bin/gfortran OMPI_CC=/usr/bin/gcc OMPI_CXX=/usr/bin/g++ OMPI_FC=/usr/bin/gfortran

# [B] DBCSR official tests on the GPU (from the toolchain's patched DBCSR source)
DB_SRC="$TC_SRC/build/dbcsr-2.10.0"; DB_TEST="$L3_BUILD_DEPS/dbcsr-test"
if [ -z "${HPCPERF_CP2K_SKIP_DBCSR_TEST:-}" ] && [ ! -f "$DB_TEST/.hpcperf-stage-done" ]; then
    [ -f "$DB_SRC/CMakeLists.txt" ] || { echo "build.sh: DBCSR source $DB_SRC missing" >&2; exit 1; }
    echo "# [B] DBCSR 2.10.0 tests: USE_ACCEL=cuda WITH_GPU=$GPUVER (arch $ARCH) USE_MPI=ON USE_OPENMP=ON BUILD_TESTING=ON"
    mkdir -p "$DB_TEST"
    cmake -S "$DB_SRC" -B "$DB_TEST" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON -DWITH_EXAMPLES=OFF \
        -DUSE_MPI=ON -DUSE_MPI_F08=ON -DUSE_OPENMP=ON -DUSE_LIBXS=ON -DUSE_LIBXSMM=ON -DUSE_ACCEL=cuda "-DWITH_GPU=$GPUVER" \
        -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_Fortran_COMPILER=/usr/bin/gfortran \
        -DTEST_MPI_RANKS="${HPCPERF_CP2K_DBCSR_TEST_RANKS:-4}" -DTEST_OMP_THREADS=4 > "$L3_LOGS/dbcsr-test-configure.log" 2>&1 \
        || { tail -30 "$L3_LOGS/dbcsr-test-configure.log"; echo "build.sh: DBCSR test configure failed" >&2; exit 1; }
    cmake --build "$DB_TEST" -j "$JOBS" > "$L3_LOGS/dbcsr-test-build.log" 2>&1 || { tail -30 "$L3_LOGS/dbcsr-test-build.log"; echo "build.sh: DBCSR test build failed" >&2; exit 1; }
    for l in "$DB_TEST"/src/libdbcsr*.so "$DB_TEST"/src/libdbcsr*.a; do [ -f "$l" ] || continue; echo "$(basename "$l"): $(cuobjdump --list-elf "$l" 2>/dev/null | /usr/bin/grep -o 'sm_[0-9]*' | sort -u | paste -sd,)"; done > "$L3_LOGS/dbcsr-test-archs.txt" || true
    # ctest with the launcher's single-node transport. DBCSR's tests call `mpiexec -n 4` directly
    # (no launcher wrapper): each rank binds device rank%ndev -> 4 ranks on 4 distinct GPUs
    # (ranks <= GPUs). The Slurm allocation exposes 1 task slot, so PRRTE's slot accounting is
    # relaxed for these launches exactly as the common launcher does (--map-by ...:OVERSUBSCRIBE
    # bookkeeping; no GPU sharing).
    ( cd "$DB_TEST" && OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,sm,smcuda OMP_NUM_THREADS=4 \
        PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe OMPI_MCA_rmaps_base_oversubscribe=true \
        ctest --output-on-failure --timeout 1800 -j 1 ) > "$L3_LOGS/dbcsr-ctest.log" 2>&1 || true
    if /usr/bin/grep -qE '100% tests passed, 0 tests failed' "$L3_LOGS/dbcsr-ctest.log"; then
        touch "$DB_TEST/.hpcperf-stage-done"; /usr/bin/grep -E 'tests passed|Total Test time' "$L3_LOGS/dbcsr-ctest.log"
    else
        /usr/bin/grep -E 'tests passed|Failed|\*\*\*|Not Run|Timeout' "$L3_LOGS/dbcsr-ctest.log" | head -20
        echo "build.sh: DBCSR official tests did not all pass on the GPU -- CP2K is not built on top of an unverified DBCSR (log: $L3_LOGS/dbcsr-ctest.log)" >&2; exit 1
    fi
fi
t2=$(date +%s)

# [C] CP2K
echo "# [C] CP2K cmake: ${CP2K_CMAKE[*]}"
mkdir -p "$L3_BUILD"
cmake -S "$SRC" -B "$L3_BUILD" -G Ninja "${CP2K_CMAKE[@]}" > "$L3_LOGS/cp2k-configure.log" 2>&1 \
    || { tail -60 "$L3_LOGS/cp2k-configure.log"; echo "build.sh: CP2K configure failed (log: $L3_LOGS/cp2k-configure.log)" >&2; exit 1; }
cmake --build "$L3_BUILD" -j "$JOBS" > "$L3_LOGS/cp2k-build.log" 2>&1 \
    || { tail -60 "$L3_LOGS/cp2k-build.log"; echo "build.sh: CP2K build failed (log: $L3_LOGS/cp2k-build.log)" >&2; exit 1; }
cmake --install "$L3_BUILD" > "$L3_LOGS/cp2k-install.log" 2>&1 || { echo "build.sh: CP2K install failed" >&2; exit 1; }
t3=$(date +%s)
EXE="$CP2K_PREFIX/bin/cp2k.psmp"
[ -x "$EXE" ] || { echo "build.sh: $EXE not produced" >&2; exit 1; }
# device code lives in libcp2k.so (the psmp executable itself carries none); a `[ -f ] &&` chain
# must not propagate a failing last test into the substitution under set -e
archs="$(for l in "$EXE" "$CP2K_PREFIX"/lib/libcp2k*.so* "$CP2K_PREFIX"/lib64/libcp2k*.so*; do if [ -f "$l" ]; then cuobjdump --list-elf "$l" 2>/dev/null | /usr/bin/grep -o 'sm_[0-9]*' || true; fi; done | sort -u | paste -sd,)"
[ "$archs" = "sm_$ARCH" ] || { echo "build.sh: CP2K device code embeds '$archs', expected sm_$ARCH -- refusing" >&2; exit 1; }
l3_fingerprint_write "$L3_INSTALL" "$FP"
{
    echo "profile=$PROFILE cp2k=$SHA utc=$(date -u +%FT%TZ) jobs=$JOBS"
    echo "seconds: toolchain=$((t1 - t0)) dbcsr_tests=$((t2 - t1)) cp2k=$((t3 - t2)) total=$(( $(date +%s) - t0 ))"
    echo "cp2k.psmp sha256=$(l3_sha_file "$EXE") device_archs=$archs"
    echo "cp2k version line: $(LD_LIBRARY_PATH="$CP2K_PREFIX/lib:$CP2K_PREFIX/lib64:${LD_LIBRARY_PATH:-}" "$EXE" --version 2>/dev/null | head -3 | paste -sd'|' || true)"
    echo "runtime note: cp2k.psmp has no rpath to libcp2k.so -- run.sh sources $TC_INSTALL/setup and prepends $CP2K_PREFIX/lib to LD_LIBRARY_PATH"
    echo "toolchain.conf:"; sed 's/^/  /' "$TC_INSTALL/toolchain.conf"
    [ -f "$L3_LOGS/dbcsr-test-archs.txt" ] && { echo "dbcsr test libs:"; sed 's/^/  /' "$L3_LOGS/dbcsr-test-archs.txt"; }
    /usr/bin/grep -E 'tests passed' "$L3_LOGS/dbcsr-ctest.log" 2>/dev/null | sed 's/^/dbcsr ctest: /'
} > "$L3_INSTALL/BUILD_INFO.txt"
echo "# built: toolchain $((t1 - t0)) s, DBCSR tests $((t2 - t1)) s, CP2K $((t3 - t2)) s -> $EXE ($archs)"
exit 0
