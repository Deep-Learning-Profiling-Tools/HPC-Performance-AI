#!/usr/bin/env bash
# Private LLVM/Clang OpenMP-offload toolchain for QMCPACK (upstream's documented
# GPU compiler for NVIDIA: "For NVIDIA GPUs, LLVM clang"; QMC_GPU="openmp;cuda").
#
#   LLVM 23.1.0 (llvmorg-23.1.0, 2026-08-25; llvm-project-23.1.0.src.tar.xz
#   sha256 ab1f0e3ec52448c33e8782eaf0422504b87c7b016b22514653ee0d8fcee479ff)
#   projects clang;lld, runtimes openmp;offload (host) + the GPU runtimes target
#   nvptx64-nvidia-cuda with libc;openmp (LLVM >= 21 builds the OpenMP device runtime
#   libompdevice through LLVM_RUNTIME_TARGETS; LIBOMPTARGET_DEVICE_ARCHITECTURES is
#   unused since LLVM 21 -- the first attempt without the GPU runtime target produced no
#   device runtime and clang failed with "no library 'libomptarget-nvptx.bc' found"),
#   CUDA 13.2 (clang 23 lists CUDA_132 as FULLY_SUPPORTED),
#   host compiler system GCC 14.2.1 (clang's default GCC toolchain on this node).
#
# Why a source build: the official binary release LLVM-23.1.0-Linux-X64.tar.xz
# (sha256 18da30f7..., 2.0 GB) ships libomp.so and the clang-offload-* tools but
# NO libomptarget / device runtime (.deps/level3/qmcpack/downloads/LLVM-binary-release-check.txt);
# the node's own /home clang+llvm is an old, broken (libtinfo.so.5) install with
# nvptx bitcode up to sm_60 only. Spack's llvm+cuda would also build from source.
#
#   ./build_llvm.sh            profile clang231-cuda132-offload, -j${HPCPERF_BUILD_JOBS:-24}
# Install: .deps/level3/qmcpack/clang231-cuda132-offload/install/llvm  (~2-3 GB, ~1 h)
# Nothing under the project conda env or the system is modified.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../../.." && pwd)"
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
PROFILE="${HPCPERF_QMCPACK_PROFILE:-clang231-cuda132-offload}"
l3_paths_profile qmcpack "$PROFILE" cuda || exit 2
VER=23.1.0; TARBALL="$R/.deps/level3/qmcpack/downloads/llvm-project-$VER.src.tar.xz"
EXPECT_SHA=ab1f0e3ec52448c33e8782eaf0422504b87c7b016b22514653ee0d8fcee479ff
PREFIX="$L3_INSTALL/llvm"
# Source + build trees on the node's LOCAL disk (the project filesystem is NFS and
# extracting/compiling the ~150k-file LLVM tree there took >1 h just to unpack); the
# install, logs and BUILD_INFO go to the profile tree on the project filesystem. The scratch is
# workspace x tarball x profile specific (l3_local_scratch_dir; HPCPERF_LLVM_SCRATCH overrides);
# the pre-2026-09-15 shared /tmp/hpcperf-l3-b2-scratch/qmcpack-llvm is legacy local state, never read.
SCRATCH="${HPCPERF_LLVM_SCRATCH:-$(l3_local_scratch_dir qmcpack-llvm "$EXPECT_SHA" "$PROFILE")}"; mkdir -p "$SCRATCH"
SRC="$SCRATCH/llvm-project-$VER.src"; BUILD="$SCRATCH/build"
echo "$SCRATCH" > "$L3_SRC/LLVM_SCRATCH_LOCATION.txt"
JOBS="${HPCPERF_BUILD_JOBS:-24}"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda-13.2}"
ARCH="${HPCPERF_CUDA_ARCH:-$(l3_gpu_arch)}"; [ -n "$ARCH" ] || ARCH=100
# clean host environment: system GCC 14 as host compiler; cmake/ninja from the project conda env
# (tools only -- conda is NOT on CMAKE_PREFIX_PATH so no conda libraries are linked)
CONDA_BIN="$R/.conda_env/bin"
export PATH="/usr/bin:/usr/sbin:/bin:$CUDA_ROOT/bin"; unset CMAKE_PREFIX_PATH LD_LIBRARY_PATH CPATH LIBRARY_PATH CMAKE_GENERATOR CUDAARCHS CC CXX
CMAKE="$CONDA_BIN/cmake"; NINJA="$CONDA_BIN/ninja"
[ -x "$CMAKE" ] && [ -x "$NINJA" ] || { echo "build_llvm.sh: cmake/ninja not found under $CONDA_BIN" >&2; exit 1; }

if [ -f "$PREFIX/.hpcperf-stage-done" ]; then echo "# LLVM $VER already installed under $PREFIX"; exit 0; fi
[ -f "$TARBALL" ] || { echo "build_llvm.sh: $TARBALL missing (download llvm-project-$VER.src.tar.xz from the llvmorg-$VER release)" >&2; exit 1; }
got="$(sha256sum "$TARBALL" | cut -d' ' -f1)"; [ "$got" = "$EXPECT_SHA" ] || { echo "build_llvm.sh: tarball sha256 $got != $EXPECT_SHA" >&2; exit 1; }
if [ ! -f "$SRC/.hpcperf-extracted" ]; then rm -rf "$SRC"; tar -xJf "$TARBALL" -C "$SCRATCH"; touch "$SRC/.hpcperf-extracted"; fi
# the device-architecture variable name changed across LLVM releases; use the one this tree defines
DEVARCH_VAR=LIBOMPTARGET_DEVICE_ARCHITECTURES
/usr/bin/grep -rqs 'OFFLOAD_DEVICE_ARCHITECTURES' "$SRC/offload/CMakeLists.txt" "$SRC/offload/DeviceRTL/CMakeLists.txt" && \
    ! /usr/bin/grep -rqs 'LIBOMPTARGET_DEVICE_ARCHITECTURES' "$SRC/offload/CMakeLists.txt" "$SRC/offload/DeviceRTL/CMakeLists.txt" && DEVARCH_VAR=OFFLOAD_DEVICE_ARCHITECTURES

echo "# LLVM $VER -> $PREFIX: host gcc $(/usr/bin/gcc -dumpfullversion), CUDA $CUDA_ROOT, device arch sm_$ARCH ($DEVARCH_VAR), -j$JOBS (expect ~1 h, ~15 GB build tree)"
mkdir -p "$BUILD" "$L3_LOGS"
t0=$(date +%s)
"$CMAKE" -S "$SRC/llvm" -B "$BUILD" -G Ninja "-DCMAKE_MAKE_PROGRAM=$NINJA" \
    -DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=$PREFIX" \
    -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DLLVM_ENABLE_PROJECTS="clang;lld" -DLLVM_ENABLE_RUNTIMES="openmp;offload" \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_DOCS=OFF \
    -DCLANG_INCLUDE_TESTS=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_LIBEDIT=OFF \
    -DLLVM_PARALLEL_LINK_JOBS=4 -DLLVM_INSTALL_UTILS=ON \
    "-DCUDAToolkit_ROOT=$CUDA_ROOT" "-D${DEVARCH_VAR}=sm_$ARCH" \
    -DOPENMP_ENABLE_LIBOMPTARGET=ON -DLIBOMP_OMPT_SUPPORT=ON \
    -DLLVM_RUNTIME_TARGETS="default;nvptx64-nvidia-cuda" \
    -DRUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_RUNTIMES="libc;openmp" \
    > "$L3_LOGS/llvm-configure.log" 2>&1 || { tail -40 "$L3_LOGS/llvm-configure.log"; echo "build_llvm.sh: configure failed" >&2; exit 1; }
"$CMAKE" --build "$BUILD" -j "$JOBS" > "$L3_LOGS/llvm-build.log" 2>&1 || { tail -40 "$L3_LOGS/llvm-build.log"; echo "build_llvm.sh: build failed (log $L3_LOGS/llvm-build.log)" >&2; exit 1; }
"$CMAKE" --install "$BUILD" > "$L3_LOGS/llvm-install.log" 2>&1 || { echo "build_llvm.sh: install failed" >&2; exit 1; }
t1=$(date +%s)
# evidence that the offload runtime and the sm_100 device runtime exist
OMPT="$(find "$PREFIX/lib" -name 'libomptarget.so*' | head -1)"; DEVRT="$(find "$PREFIX/lib" -name 'libomptarget*nvptx*' -o -name 'libomptarget.devicertl.a' -o -name 'libompdevice*' 2>/dev/null | head -3 | paste -sd,)"
[ -n "$OMPT" ] || { echo "build_llvm.sh: libomptarget.so not installed -- offload runtime missing" >&2; exit 1; }
{
    echo "llvm=$VER tarball_sha256=$EXPECT_SHA host_gcc=$(/usr/bin/gcc -dumpfullversion) cuda=$CUDA_ROOT device_arch=sm_$ARCH ($DEVARCH_VAR) jobs=$JOBS seconds=$((t1 - t0)) utc=$(date -u +%FT%TZ)"
    echo "clang: $("$PREFIX/bin/clang" --version | head -1)"
    echo "libomptarget: $OMPT"; echo "device runtime: ${DEVRT:-<none found>}"
    echo "offload-arch: $("$PREFIX/bin/offload-arch" 2>&1 | head -2 | paste -sd,)"
} > "$PREFIX/BUILD_INFO.txt"; cat "$PREFIX/BUILD_INFO.txt"
touch "$PREFIX/.hpcperf-stage-done"
echo "# LLVM built in $((t1 - t0)) s"
