#!/usr/bin/env bash
# HPC-Performance-AI: build the framework libraries that the Level 2 mini-apps
# are written against (Kokkos, RAJA, hypre, MFEM, Cabana, heFFTe, ...).
#
# Everything is built from pinned upstream release tags with the project
# toolchain (conda GCC 13.3.0 + system CUDA from hpcperf_env.sh) and installed
# under <repo>/.deps -- nothing is written outside the repository.
#
# Dependency PROFILES (HPCPERF_DEPS_PROFILE, default "level2"):
#   level2 (default)  .deps/src/<dep>  .deps/build/<dep>  .deps/install/<dep>
#                     -- the validated Level 2 tree; unchanged layout.
#   <name>            .deps/src/<dep>  .deps/<name>/build/<dep>  .deps/<name>/install/<dep>
#                     -- an isolated tree (e.g. HPCPERF_DEPS_PROFILE=level3), so
#                        Level 3 can pick its own dependency set/flags without
#                        touching the Level 2 installs. hpcperf_env.sh exposes
#                        exactly ONE profile's prefixes (same variable).
#   Sources (.deps/src) are shared: they are pinned tags plus patches/.
#
# Each install carries two markers:
#   .hpcperf-built        the pinned tag that was built
#   .hpcperf-fingerprint  what it was built WITH: profile, backend, GPU arch,
#                         compiler, CUDA, MPI, GPU-aware-MPI flag, sha256 of
#                         every applied patch. is_built() requires the tag to
#                         match AND the fingerprint to match the current
#                         environment; a mismatch FAILS FAST (no silent reuse
#                         of a stale install). Installs made before
#                         fingerprinting existed have no fingerprint: they are
#                         accepted with a warning ("legacy") and can be stamped
#                         post-hoc with --stamp-existing (clearly labelled as
#                         recorded after the fact, not at build time).
#
# Usage:
#   source hpcperf_env.sh
#   ./setup_level2_deps.sh                # build every dependency (idempotent)
#   ./setup_level2_deps.sh kokkos cabana  # build a subset (in dependency order)
#   ./setup_level2_deps.sh --list         # show pinned versions
#   ./setup_level2_deps.sh --stamp-existing  # fingerprint legacy installs
#   HPCPERF_DEPS_LIBMODE=1 source setup_level2_deps.sh   # functions only (tests)
#
# Prerequisites: ./setup_env.sh has been run and hpcperf_env.sh is sourced
# (provides CC/CXX, nvcc, mpicxx, cmake, ninja, and the conda METIS/FFTW/HDF5).
if [ "${HPCPERF_DEPS_LIBMODE:-0}" != 1 ]; then set -euo pipefail; fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
DEPS="$ROOT/.deps"
PROFILE="${HPCPERF_DEPS_PROFILE:-level2}"
SRC="$DEPS/src"
if [ "$PROFILE" = level2 ]; then
  BLD="$DEPS/build"; INST="$DEPS/install"; LOGS="$DEPS/logs"
else
  case "$PROFILE" in (*[!A-Za-z0-9_.-]*|"") echo "invalid HPCPERF_DEPS_PROFILE '$PROFILE'" >&2; exit 2;; esac
  BLD="$DEPS/$PROFILE/build"; INST="$DEPS/$PROFILE/install"; LOGS="$DEPS/$PROFILE/logs"
fi
JOBS="${HPCPERF_DEPS_JOBS:-$(nproc)}"

say()  { printf '\n[level2-deps] %s\n' "$*"; }
warn() { printf '[level2-deps] WARNING: %s\n' "$*" >&2; }
fail() { printf '[level2-deps] ERROR: %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------- pinned versions
# name|git url|tag
PINS=(
  "kokkos|https://github.com/kokkos/kokkos.git|5.2.1"
  "kokkos-kernels|https://github.com/kokkos/kokkos-kernels.git|5.2.1"
  "heffte|https://github.com/icl-utk-edu/heffte.git|v2.4.1"
  "cabana|https://github.com/ECP-copa/Cabana.git|0.8.0"
  "raja|https://github.com/LLNL/RAJA.git|v2026.07.0"
  "umpire|https://github.com/LLNL/Umpire.git|v2026.07.0"
  "chai|https://github.com/LLNL/CHAI.git|v2026.07.0"
  "hypre|https://github.com/hypre-space/hypre.git|v3.2.0"
  "mfem|https://github.com/mfem/mfem.git|v4.10"
)
ORDER=(kokkos kokkos-kernels heffte cabana raja umpire chai hypre mfem)

pin_field() { # $1=name $2=field(2=url,3=tag)
  local p; for p in "${PINS[@]}"; do
    [ "${p%%|*}" = "$1" ] && { echo "$p" | cut -d'|' -f"$2"; return; }
  done; fail "unknown dependency '$1'"
}

# ------------------------------------------------------- markers/fingerprints
done_marker() { echo "$INST/$1/.hpcperf-built"; }
fp_file()     { echo "$INST/$1/.hpcperf-fingerprint"; }
first_line()  { "$@" 2>/dev/null | head -n 1 || true; }

# current_fingerprint <name>: what a build of <name> in THIS environment/profile
# would be fingerprinted as. Pure (no side effects); tolerant of missing tools.
current_fingerprint() {
  local name=$1 p
  echo "dep=$name tag=$(pin_field "$name" 3)"
  echo "profile=$PROFILE backend=cuda arch=sm_${CUDA_ARCH:-unknown}"
  echo "compiler=$(first_line "${CXX:-c++}" --version)"
  echo "cuda=$(first_line nvcc --version | sed -n 's/.*release \([^,]*\),.*/\1/p')"
  echo "mpi=$(first_line mpirun --version)"
  echo "gpu_aware_mpi=off"
  for p in "$ROOT/patches/$name"-*.patch; do
    [ -e "$p" ] || continue                       # no patches is a normal case
    echo "patch=$(basename "$p") sha256=$(sha256sum "$p" | cut -d' ' -f1)"
  done
  return 0
}

# mark_built <name>: write the fingerprint first (atomically), and only when
# that succeeded write the completion marker. Returns non-zero (marker NOT
# written) if anything could not be written. Works with or without patches.
mark_built() {
  local name=$1 dir fp tmp
  dir="$INST/$name"; fp="$(fp_file "$name")"
  [ -d "$dir" ] || return 1
  tmp="$(mktemp "$dir/.hpcperf-fingerprint.XXXXXX" 2>/dev/null)" || return 1
  if ! current_fingerprint "$name" > "$tmp"; then rm -f "$tmp"; return 1; fi
  mv -f "$tmp" "$fp" || { rm -f "$tmp"; return 1; }
  pin_field "$name" 3 > "$(done_marker "$name")" || return 1
  return 0
}

# check_fingerprint <name>: 0 if the recorded fingerprint matches the current
# environment (or is absent = legacy, accepted with a warning), 1 on mismatch
# (prints the differing lines). 'stamped=' lines are informational only.
check_fingerprint() {
  local name=$1 fp diffout
  fp="$(fp_file "$name")"
  if [ ! -f "$fp" ]; then
    warn "$name: install has no .hpcperf-fingerprint (built before fingerprinting existed); accepted as legacy -- run '$0 --stamp-existing' to record one"
    return 0
  fi
  diffout="$(diff <(grep -v '^stamped=' "$fp") <(current_fingerprint "$name" | grep -v '^stamped=') || true)"
  if [ -n "$diffout" ]; then
    printf '[level2-deps] fingerprint mismatch for %s (recorded < vs current >):\n%s\n' "$name" "$diffout" >&2
    return 1
  fi
  return 0
}

# is_built <name>: tag marker matches AND fingerprint is consistent. A stale
# install (built with a different toolchain/arch/patches) fails fast here.
is_built() {
  local name=$1 m
  m="$(done_marker "$name")"
  [ -f "$m" ] || return 1
  [ "$(cat "$m")" = "$(pin_field "$name" 3)" ] || return 1
  check_fingerprint "$name" || fail "$name is installed in profile '$PROFILE' but was built with a different configuration; refusing to reuse it silently. Remove $INST/$name to rebuild, or use another HPCPERF_DEPS_PROFILE."
  return 0
}

# stamp_existing: post-hoc fingerprint for legacy installs (labelled as such).
stamp_existing() {
  local d name
  for d in "$INST"/*/; do
    d="${d%/}"; name="$(basename "$d")"
    [ -f "$d/.hpcperf-built" ] || continue
    if [ -f "$d/.hpcperf-fingerprint" ]; then say "$name: already fingerprinted"; continue; fi
    { current_fingerprint "$name"; echo "stamped=post-hoc $(date -u +%Y-%m-%dT%H:%MZ) (recorded after the build, from the current environment; not a build-time record)"; } > "$d/.hpcperf-fingerprint"
    say "$name: fingerprint stamped post-hoc"
  done
}

# fetch <name>: check out the pinned tag into $SRC/<name>, record its commit.
fetch() {
  local name=$1 url tag
  url="$(pin_field "$name" 2)"; tag="$(pin_field "$name" 3)"
  if [ ! -d "$SRC/$name/.git" ]; then
    say "fetching $name @ $tag"
    git clone --quiet --depth 1 --branch "$tag" "$url" "$SRC/$name"
  fi
  git -C "$SRC/$name" rev-parse HEAD > "$SRC/$name.commit"
  apply_patches "$name"
}

# apply_patches <name>: apply every patches/<name>-*.patch to the checkout.
# Idempotent: a patch that is already in the tree (reverse-applies cleanly) is
# skipped; anything else that fails to apply aborts. No patches is normal.
apply_patches() {
  local name=$1 p
  for p in "$ROOT/patches/$name"-*.patch; do
    [ -e "$p" ] || continue
    if git -C "$SRC/$name" apply --check --reverse "$p" >/dev/null 2>&1; then
      say "$name: patch $(basename "$p") already applied"
    elif git -C "$SRC/$name" apply "$p"; then
      say "$name: applied patch $(basename "$p")"
    else
      fail "$name: patch $(basename "$p") does not apply (see patches/README.md)"
    fi
  done
}

# cmake_build <name> <source subdir or .> <cmake args...>
cmake_build() {
  local name=$1 sub=$2; shift 2
  local log="$LOGS/$name.log"
  rm -rf "$BLD/$name"; mkdir -p "$BLD/$name"
  say "configuring $name (log: $log)"
  cmake -S "$SRC/$name/$sub" -B "$BLD/$name" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INST/$name" \
        -DCMAKE_PREFIX_PATH="$CONDA;$INST" \
        -DBUILD_SHARED_LIBS=OFF \
        "$@" > "$log" 2>&1 || { tail -40 "$log"; fail "$name configure failed"; }
  say "building $name"
  cmake --build "$BLD/$name" -j "$JOBS" >> "$log" 2>&1 || { tail -40 "$log"; fail "$name build failed"; }
  cmake --install "$BLD/$name" >> "$log" 2>&1 || fail "$name install failed"
  cp "$SRC/$name.commit" "$INST/$name/.hpcperf-commit"
  mark_built "$name" || fail "$name: could not write .hpcperf-fingerprint / .hpcperf-built under $INST/$name"
  say "$name installed -> $INST/$name"
}

# Kokkos_ENABLE_IMPL_VIEW_LEGACY: Kokkos >= 4.7 switched View to an mdspan-based
# implementation that does not yet support the custom layouts Cabana's AoSoA
# needs; Cabana (0.8.0 and current master) refuses to configure without the
# legacy View implementation. It is the same View code Kokkos < 4.7 shipped.
build_kokkos() {
  # Kokkos is compiled through nvcc_wrapper (upstream-recommended for GCC host
  # compilers). Kokkos 5.x requires C++20. DEPRECATED_CODE_4 keeps the 4.x API
  # surface the mini-apps were written against.
  cmake_build kokkos . \
    -DCMAKE_CXX_COMPILER="$SRC/kokkos/bin/nvcc_wrapper" \
    -DCMAKE_CXX_STANDARD=20 \
    -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
    -DKokkos_ENABLE_DEPRECATED_CODE_4=ON -DKokkos_ENABLE_DEPRECATION_WARNINGS=OFF \
    -DKokkos_ENABLE_IMPL_VIEW_LEGACY=ON \
    "-DKokkos_ARCH_${KOKKOS_ARCH}=ON"
}

build_kokkos_kernels() {
  cmake_build kokkos-kernels . \
    -DCMAKE_CXX_COMPILER="$NVCC_WRAPPER" \
    -DKokkos_ROOT="$INST/kokkos" \
    -DKokkosKernels_ENABLE_TESTS=OFF -DKokkosKernels_ENABLE_EXAMPLES=OFF \
    -DKokkosKernels_INST_DOUBLE=ON -DKokkosKernels_INST_FLOAT=OFF \
    -DKokkosKernels_INST_LAYOUTLEFT=ON -DKokkosKernels_INST_LAYOUTRIGHT=OFF \
    -DKokkosKernels_INST_ORDINAL_INT=ON -DKokkosKernels_INST_OFFSET_INT=ON \
    -DKokkosKernels_INST_OFFSET_SIZE_T=ON \
    -DKokkosKernels_ENABLE_TPL_CUBLAS=OFF -DKokkosKernels_ENABLE_TPL_CUSPARSE=OFF -DKokkosKernels_ENABLE_TPL_CUSOLVER=OFF
}

build_heffte() {
  cmake_build heffte . \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_CUDA_HOST_COMPILER="$CXX" -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DHeffte_ENABLE_CUDA=ON -DHeffte_ENABLE_FFTW=ON \
    -DHeffte_ENABLE_AVX=OFF -DHeffte_ENABLE_TESTING=OFF \
    -DFFTW_ROOT="$CONDA"
}

build_cabana() {
  cmake_build cabana . \
    -DCMAKE_CXX_COMPILER="$NVCC_WRAPPER" \
    -DKokkos_ROOT="$INST/kokkos" -DHeffte_ROOT="$INST/heffte" \
    -DCabana_ENABLE_GRID=ON -DCabana_REQUIRE_MPI=ON -DCabana_REQUIRE_HEFFTE=ON \
    -DCabana_ENABLE_TESTING=OFF -DCabana_ENABLE_EXAMPLES=OFF \
    -DCabana_ENABLE_PERFORMANCE_TESTING=OFF
}

build_raja() {
  # RAJA pulls camp as a submodule (not included in a tag-only shallow clone).
  git -C "$SRC/raja" submodule update --init --depth 1 --recursive >> "$LOGS/raja.log" 2>&1 || true
  cmake_build raja . \
    -DCMAKE_CXX_COMPILER="$CXX" -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CUDA_HOST_COMPILER="$CXX" -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DENABLE_CUDA=ON -DRAJA_ENABLE_CUDA=ON -DENABLE_OPENMP=ON \
    -DRAJA_ENABLE_TESTS=OFF -DRAJA_ENABLE_EXAMPLES=OFF -DRAJA_ENABLE_EXERCISES=OFF \
    -DRAJA_ENABLE_BENCHMARKS=OFF -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF
}

# Umpire and CHAI (RAJA Portability Suite, same release as RAJA/camp) are what
# ExaCMech's GPU path (CHAI ManagedArray/managed_ptr + Umpire allocators) needs.
# Both use BLT (submodule) and Umpire bundles fmt; -DBLT_CXX_STD=c++20 matches
# the C++20 RAJA/camp headers. They find camp/RAJA via CMAKE_PREFIX_PATH=$INST.
build_umpire() {
  git -C "$SRC/umpire" submodule update --init --depth 1 blt src/tpl/umpire/fmt >> "$LOGS/umpire.log" 2>&1 || true
  cmake_build umpire . \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_CUDA_HOST_COMPILER="$CXX" -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DBLT_CXX_STD=c++20 -DENABLE_CUDA=ON -DENABLE_OPENMP=ON -DUMPIRE_ENABLE_C=OFF \
    -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF -DENABLE_DOCS=OFF -DENABLE_BENCHMARKS=OFF \
    -Dcamp_DIR="$INST/raja/lib/cmake/camp"
}

build_chai() {
  git -C "$SRC/chai" submodule update --init --depth 1 blt >> "$LOGS/chai.log" 2>&1 || true
  cmake_build chai . \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_CUDA_HOST_COMPILER="$CXX" -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DBLT_CXX_STD=c++20 -DENABLE_CUDA=ON -DENABLE_OPENMP=ON \
    -DCHAI_ENABLE_RAJA_PLUGIN=ON -DCHAI_ENABLE_MANAGED_PTR=ON \
    -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF -DENABLE_DOCS=OFF -DENABLE_BENCHMARKS=OFF \
    -Dcamp_DIR="$INST/raja/lib/cmake/camp" -Draja_DIR="$INST/raja/lib/cmake/raja" \
    -Dumpire_DIR="$INST/umpire/lib64/cmake/umpire" -Dfmt_DIR="$INST/umpire/lib64/cmake/fmt"
}

build_hypre() {
  cmake_build hypre src \
    -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_CUDA_HOST_COMPILER="$CXX" -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DHYPRE_ENABLE_CUDA=ON -DHYPRE_ENABLE_MPI=ON \
    -DHYPRE_ENABLE_UNIFIED_MEMORY=OFF -DHYPRE_ENABLE_GPU_AWARE_MPI=OFF \
    -DHYPRE_ENABLE_CUSPARSE=ON -DHYPRE_ENABLE_CUBLAS=ON -DHYPRE_ENABLE_CURAND=ON \
    -DHYPRE_ENABLE_UMPIRE=OFF \
    -DHYPRE_BUILD_TESTS=OFF -DHYPRE_BUILD_EXAMPLES=OFF
}

build_mfem() {
  cmake_build mfem . \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_CUDA_HOST_COMPILER="$CXX" -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DMFEM_USE_MPI=ON -DMFEM_USE_CUDA=ON -DMFEM_USE_METIS=ON -DMFEM_USE_METIS_5=ON \
    -DHYPRE_DIR="$INST/hypre" -DMETIS_DIR="$CONDA" \
    -DMFEM_ENABLE_EXAMPLES=OFF -DMFEM_ENABLE_MINIAPPS=OFF -DMFEM_ENABLE_TESTING=OFF
}

# Library mode: functions only (tests source this file with HPCPERF_DEPS_LIBMODE=1).
if [ "${HPCPERF_DEPS_LIBMODE:-0}" = 1 ]; then return 0 2>/dev/null || exit 0; fi

# ------------------------------------------------------------------ commands
if [ "${1:-}" = "--list" ]; then
  for p in "${PINS[@]}"; do IFS='|' read -r n u t <<<"$p"; printf '%-16s %-10s %s\n' "$n" "$t" "$u"; done
  exit 0
fi

# --------------------------------------------------------------- environment
[ -n "${HPC_PERFORMANCE_AI_ROOT:-}" ] || fail "source hpcperf_env.sh first"
command -v nvcc   >/dev/null || fail "nvcc not on PATH (CUDA_HOME=${CUDA_HOME:-unset})"
command -v mpicxx >/dev/null || fail "mpicxx not found -- rerun ./setup_env.sh (installs conda OpenMPI)"
command -v cmake  >/dev/null || fail "cmake not found"
CUDA_ARCH="${HPCPERF_CUDA_ARCH:-}"   # numeric, e.g. 100 for sm_100
if [ -z "$CUDA_ARCH" ]; then
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
  [ -n "$cc" ] || fail "cannot detect GPU compute capability; set HPCPERF_CUDA_ARCH=<e.g. 100>"
  CUDA_ARCH="$cc"
fi
# Kokkos names its GPU architectures; map the numeric capability.
case "$CUDA_ARCH" in
  70) KOKKOS_ARCH=VOLTA70;;    75) KOKKOS_ARCH=TURING75;;
  80) KOKKOS_ARCH=AMPERE80;;   86) KOKKOS_ARCH=AMPERE86;;  89) KOKKOS_ARCH=ADA89;;
  90) KOKKOS_ARCH=HOPPER90;;   100) KOKKOS_ARCH=BLACKWELL100;; 120) KOKKOS_ARCH=BLACKWELL120;;
  *) fail "no Kokkos architecture mapping for compute capability $CUDA_ARCH";;
esac
CONDA="${CONDA_PREFIX:?conda env not active}"
mkdir -p "$SRC" "$BLD" "$INST" "$LOGS"

if [ "${1:-}" = "--stamp-existing" ]; then stamp_existing; exit 0; fi

say "profile=$PROFILE install=$INST toolchain: CC=$CC CXX=$CXX nvcc=$(nvcc --version | grep -o 'release [0-9.]*') CUDA arch=sm_$CUDA_ARCH ($KOKKOS_ARCH) jobs=$JOBS"

NVCC_WRAPPER="$INST/kokkos/bin/nvcc_wrapper"
export NVCC_WRAPPER_DEFAULT_COMPILER="$CXX"

# ------------------------------------------------------------------- driver
TARGETS=("$@"); [ ${#TARGETS[@]} -gt 0 ] || TARGETS=("${ORDER[@]}")
for t in "${TARGETS[@]}"; do
  pin_field "$t" 3 >/dev/null
  if is_built "$t"; then say "$t already built ($(pin_field "$t" 3)) -- skipping"; continue; fi
  fetch "$t"
  "build_$(echo "$t" | tr '-' '_')"
done
say "done. Installed in profile '$PROFILE': $(ls "$INST" | tr '\n' ' ')"
