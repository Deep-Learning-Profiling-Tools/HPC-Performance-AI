#!/bin/bash
# l3_common.sh -- shared helpers for Level 3 application wrappers. Source it.
#
# Dependency isolation: every Level 3 application owns a private tree
#     $R/.deps/level3/<app>/{src,build,install,logs}
# (never a shared install root, so Kokkos/AMReX/MPI/hypre versions of
# different applications cannot pollute each other), plus its upstream
# checkout under $R/_upstream/level3/<Name> and its own build tree under
# $R/build/level3/<app>/<backend>. Nothing here touches the Level 2 tree.
#
# Fingerprint: an install is stamped with .hpcperf-l3-fingerprint recording
# application, upstream commit, dependency versions, compiler, CUDA/ROCm, GPU
# arch, MPI, CMake options, GPU-aware-MPI option, patch hashes, site profile,
# Spack lock hash and container image hash where applicable. A recorded
# fingerprint that does not match the requested configuration FAILS FAST
# (l3_fingerprint_check) -- stale installs are never reused silently.
#
# Runtime: launches go through the common launcher. Until the shared runtime
# tools move to tools/runtime/ (proposal in tools/runtime/README.md) the
# location is a single variable, HPCPERF_RUNTIME_DIR, defaulting to
# level2/tools -- nothing is copied or moved, so Level 2 is not disturbed.

# Some interactive environments export a `grep` shell FUNCTION (a ugrep wrapper with
# -I/--ignore-files) into child processes; it changes grep's semantics (binary-file
# handling, exit codes) inside these scripts. Level 3 scripts want the real grep.
unset -f grep 2>/dev/null || true

L3_R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
L3_TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; export L3_TOOLS   # for the validators' python (l3_check.py)

# L3_RUN_SUBDIR: name of the run-directory tree under each application build tree
# ($R/build/level3/<app>/<profile>/<L3_RUN_SUBDIR>/<case>.<mode>.np<N>). Default "run".
# HPCPERF_L3_RUN_SUBDIR=run.regress-<sha> sends a regression campaign into a fresh
# sibling tree so that historical results are never overwritten (a plain name, no
# path separators). run.sh and validate.sh of every application use $L3_RUN_SUBDIR.
L3_RUN_SUBDIR="${HPCPERF_L3_RUN_SUBDIR:-run}"
case "$L3_RUN_SUBDIR" in ""|*/*|.|..|.*) echo "l3_common: invalid HPCPERF_L3_RUN_SUBDIR '$L3_RUN_SUBDIR' (plain directory name expected)" >&2; return 2 2>/dev/null || exit 2;; esac
export L3_RUN_SUBDIR
HPCPERF_RUNTIME_DIR="${HPCPERF_RUNTIME_DIR:-$L3_R/level2/tools}"
L3_LAUNCHER="$HPCPERF_RUNTIME_DIR/hpcperf_mpi_launch.sh"
L3_TOPOLOGY="$HPCPERF_RUNTIME_DIR/hpcperf_topology.py"
# shellcheck disable=SC1091
source "$HPCPERF_RUNTIME_DIR/hpcperf_launch_common.sh"

# l3_paths <app>: exports L3_APP, L3_UPSTREAM, L3_DEPS, L3_SRC, L3_BUILD_DEPS, L3_INSTALL, L3_LOGS
l3_paths() {
    L3_APP="$1"
    L3_DEPS="$L3_R/.deps/level3/$L3_APP"
    L3_SRC="$L3_DEPS/src"; L3_BUILD_DEPS="$L3_DEPS/build"; L3_INSTALL="$L3_DEPS/install"; L3_LOGS="$L3_DEPS/logs"
    mkdir -p "$L3_SRC" "$L3_BUILD_DEPS" "$L3_INSTALL" "$L3_LOGS"
}

# l3_paths_profile <app> <profile>: second-batch layout -- one private tree per
# *configuration profile* (compiler/Toolkit/backend/key-dependency variant):
#     $R/.deps/level3/<app>/<profile>/{src,build,install,logs,cache}
# exports L3_APP, L3_PROFILE, L3_DEPS, L3_SRC, L3_BUILD_DEPS, L3_INSTALL, L3_LOGS,
# L3_CACHE and L3_BUILD (= $R/build/level3/<app>/<profile>, the application build
# tree). Different profiles never share a mutable source tree or an install.
l3_paths_profile() {
    L3_APP="$1"; L3_PROFILE="$2"
    case "$L3_PROFILE" in ""|*/*|.*) echo "l3_paths_profile: invalid profile name '$L3_PROFILE'" >&2; return 2;; esac
    L3_DEPS="$L3_R/.deps/level3/$L3_APP/$L3_PROFILE"
    L3_SRC="$L3_DEPS/src"; L3_BUILD_DEPS="$L3_DEPS/build"; L3_INSTALL="$L3_DEPS/install"; L3_LOGS="$L3_DEPS/logs"; L3_CACHE="$L3_DEPS/cache"
    L3_BUILD="$L3_R/build/level3/$L3_APP/$L3_PROFILE"
    mkdir -p "$L3_SRC" "$L3_BUILD_DEPS" "$L3_INSTALL" "$L3_LOGS" "$L3_CACHE"
}

# l3_clean_conda_build_env: the project conda env exports its own compiler-driving
# variables (CFLAGS/CXXFLAGS/LDFLAGS with -march=nocona -mtune=haswell and conda
# -isystem/-rpath paths, AR/RANLIB/NM/LD = conda binutils, CMAKE_ARGS, ...). They are
# right for the conda GCC, wrong for a build that deliberately uses the system GCC
# 14 toolchain (they made OpenBLAS fail with "target specific option mismatch" on
# its AVX512 kernels and would pin -march=nocona on everything). Call after
# hpcperf_env.sh in build scripts that select the system compilers; PATH, CUDA and
# the MPI wrappers are left alone.
l3_clean_conda_build_env() {
    unset CFLAGS CXXFLAGS FFLAGS FCFLAGS FORTRANFLAGS CPPFLAGS LDFLAGS \
          DEBUG_CFLAGS DEBUG_CXXFLAGS DEBUG_CPPFLAGS DEBUG_FFLAGS DEBUG_FORTRANFLAGS \
          AR RANLIB NM LD STRIP AS CPP OBJCOPY OBJDUMP READELF SIZE STRINGS ADDR2LINE ELFEDIT GPROF CXXFILT LD_GOLD \
          HOST BUILD CMAKE_ARGS MESON_ARGS GCC_AR GCC_NM GCC_RANLIB GXX GCC GFORTRAN F77 F90 F95 \
          CONDA_BUILD_SYSROOT CC_FOR_BUILD CXX_FOR_BUILD \
          C_INCLUDE_PATH CPLUS_INCLUDE_PATH CPATH LIBRARY_PATH 2>/dev/null || true   # login-shell leftovers pointing at foreign conda envs
    echo "# l3: conda build variables (CFLAGS/LDFLAGS/AR/... ) cleared for a system-toolchain build"
}

# l3_version_mm <version string>: "13.2.78" -> "132", "13.3.0" -> "133" (profile-name component)
l3_version_mm() { echo "$1" | awk -F. '{printf "%s%s", $1, $2}'; }

# l3_clean_env_exec [--] <cmd...>
#   Runs a command under an environment reduced to an explicit allow-list (scheduler,
#   MPI, CUDA, compilers/build flags, conda/tool paths, HPCPERF_*, OpenMP/BLAS/UCX
#   knobs, locale). Everything else from the login shell -- in particular API keys,
#   tokens, agent/session variables -- is NOT passed on. Use it around every tool
#   that dumps or records its process environment (upstream toolchain installers
#   that write `declare -x` files, profilers such as nsys/ncu, `env`-recording
#   build systems). The allow-list is printed by `l3_clean_env_exec --list`.
#   Implementation: level3/tools/l3_clean_env.sh (also usable stand-alone).
l3_clean_env_exec() { "$L3_TOOLS/l3_clean_env.sh" "$@"; }

# l3_isolate_build_env: remove the Level 2 dependency prefixes (everything under
# $L3_R/.deps/install/, the validated Level 2 tree) from CMAKE_PREFIX_PATH and
# LD_LIBRARY_PATH before a Level 3 configure, so a Level 3 build can never pick
# up a Level 2 Kokkos/RAJA/hypre. Each Level 3 app owns its dependencies
# (bundled, or under .deps/level3/<app>). Call once in build.sh after sourcing
# hpcperf_env.sh. Does not touch the launcher (level2/tools, not .deps/install).
l3_isolate_build_env() {
    local before_c="${CMAKE_PREFIX_PATH:-}" before_l="${LD_LIBRARY_PATH:-}"
    [ -n "$before_c" ] && export CMAKE_PREFIX_PATH="$(tr ':' '\n' <<<"$before_c" | grep -v "$L3_R/.deps/install/" | paste -sd:)"
    [ -n "$before_l" ] && export LD_LIBRARY_PATH="$(tr ':' '\n' <<<"$before_l" | grep -v "$L3_R/.deps/install/" | paste -sd:)"
    local n; n="$(tr ':' '\n' <<<"${CMAKE_PREFIX_PATH:-}" | grep -c "$L3_R/.deps/install/" || true)"
    echo "# l3: build env isolated from Level 2 prefixes (CMAKE_PREFIX_PATH .deps/install entries remaining: $n)"
}

l3_first_line() { "$@" 2>/dev/null | head -n 1 || true; }
l3_cuda_version() { nvcc --version 2>/dev/null | sed -n 's/^Cuda compilation tools, release [^,]*, V\([0-9][0-9.]*\).*$/\1/p' | head -n 1; }
l3_gpu_arch() { # numeric compute capability of GPU 0, e.g. 100
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .'
}
l3_site_profile() {
    if [ -n "${HPCPERF_SITE_PROFILE:-}" ]; then echo "$HPCPERF_SITE_PROFILE"; return; fi
    case "$(hostname -s)" in dgx003|hopper*|gpu0*) echo gmu-hopper ;; *) echo generic ;; esac
}

# l3_fingerprint_text <app> <upstream_sha> <backend> "<deps versions>" "<cmake options>" <gpu_aware_mpi> [patch files...]
#   Prints the fingerprint for the configuration about to be built. Patches are
#   recorded IN THE ORDER GIVEN with their content sha256 (the source-cache key
#   depends on both the upstream SHA and this ordered patch-content hash, so a
#   same-named patch whose bytes change invalidates the cache). A patch path
#   that does not exist, or whose hash cannot be taken, is a hard error -- the
#   fingerprint is never written with a silently-missing patch.
l3_fingerprint_text() {
    local app=$1 sha=$2 backend=$3 deps=$4 cmakeopts=$5 gam=$6; shift 6
    local p h
    echo "schema=l3-2"
    echo "application=$app"
    echo "upstream_commit=$sha"
    echo "backend=$backend arch=sm_$(l3_gpu_arch)"
    echo "dependencies=$deps"
    echo "compiler=${CXX:-c++} ($(l3_first_line "${CXX:-c++}" --version))"
    echo "fortran=${FC:-gfortran} ($(l3_first_line "${FC:-gfortran}" --version))"
    echo "cuda=$(l3_cuda_version)"
    echo "rocm=${ROCM_VERSION:-none}"
    echo "mpi=$(l3_first_line mpirun --version)"
    echo "cmake_options=$cmakeopts"
    echo "gpu_aware_mpi=$gam"
    echo "site_profile=$(l3_site_profile)"
    echo "spack_lock_sha256=${L3_SPACK_LOCK_SHA:-none}"
    echo "container_image_sha256=${L3_CONTAINER_SHA:-none}"
    local idx=0 ordered=""
    for p in "$@"; do
        idx=$((idx+1))
        if [ ! -f "$p" ]; then
            echo "l3: patch file '$p' not found -- refusing to fingerprint a build with a missing patch" >&2
            return 1
        fi
        h="$(sha256sum "$p" 2>/dev/null | cut -d' ' -f1)"
        [ -n "$h" ] || { echo "l3: could not hash patch '$p'" >&2; return 1; }
        echo "patch[$idx]=$(basename "$p") sha256=$h"
        ordered="$ordered$h"
    done
    # ordered content hash of the whole patch series (empty series -> the literal 'none')
    echo "patch_series_sha256=$( [ -n "$ordered" ] && printf '%s' "$ordered" | sha256sum | cut -d' ' -f1 || echo none )"
}

# l3_fingerprint_check <install_dir> <expected_text>
#   0 = no fingerprint yet (fresh build) or identical; 1 = mismatch (prints diff, caller must fail).
l3_fingerprint_check() {
    local dir=$1 expected=$2 fp="$1/.hpcperf-l3-fingerprint" diffout
    [ -f "$fp" ] || return 0
    diffout="$(diff <(grep -v '^built=' "$fp") <(printf '%s\n' "$expected") || true)"
    if [ -n "$diffout" ]; then
        echo "l3: fingerprint mismatch for $dir (recorded < vs requested >):" >&2
        echo "$diffout" >&2
        echo "l3: refusing to reuse a differently-configured install; remove $dir or change the request" >&2
        return 1
    fi
    return 0
}

# l3_fingerprint_write <install_dir> <text>: written only after a successful build+install.
l3_fingerprint_write() {
    local dir=$1 text=$2
    { printf '%s\n' "$text"; echo "built=$(date -u +%Y-%m-%dT%H:%MZ) (build-time record)"; } > "$dir/.hpcperf-l3-fingerprint.tmp" \
        && mv -f "$dir/.hpcperf-l3-fingerprint.tmp" "$dir/.hpcperf-l3-fingerprint"
}

# l3_scale_mode <app>: validated HPCPERF_SCALE_MODE (smoke|strong|weak; default smoke)
l3_scale_mode() {
    local m="${HPCPERF_SCALE_MODE:-smoke}"
    case "$m" in smoke|strong|weak) echo "$m";; *) echo "$1/run.sh: HPCPERF_SCALE_MODE must be smoke|strong|weak (got '$m')" >&2; return 2;; esac
}

# ---------------------------------------------------------------------------
# Result management (correctness / reproducibility)
# ---------------------------------------------------------------------------

# l3_run_id: a unique id for one real execution (UTC, pid, random).
l3_run_id() { echo "$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}"; }

# l3_rundir <intended-real-path>
#   Echoes the directory run.sh should actually write into, and prepares it.
#   In a real run: rm -rf the intended dir and recreate it (fresh output only).
#   In a dry-run (HPCPERF_DRY_RUN set): NEVER touch the real dir -- a throwaway
#   sibling under .dryrun/ is used, so planning can never delete or overwrite a
#   real result. Refuses to operate on a path that is not under a Level 3
#   build/ tree (guards against an accidental rm of the wrong directory).
l3_rundir() {
    local real=$1 base parent
    case "$real" in
        "$L3_R"/build/level3/*) : ;;
        *) echo "l3_rundir: refusing to manage '$real' (not under $L3_R/build/level3/)" >&2; return 2;;
    esac
    if [ -n "${HPCPERF_DRY_RUN:-}" ]; then
        parent="$(dirname "$real")"; base="$(basename "$real")"
        real="$parent/.dryrun/$base"
        rm -rf "$real"; mkdir -p "$real"
    else
        rm -rf "$real"; mkdir -p "$real"
    fi
    printf '%s\n' "$real"
}

# l3_capture <logfile> -- <cmd...>
#   Runs the command, copies combined stdout+stderr to <logfile>, and returns
#   the command's real exit status (NOT tee's). Nothing is swallowed; callers
#   check the status. Use this instead of `cmd | grep ... || true`.
l3_capture() {
    local log=$1; shift
    [ "${1:-}" = -- ] && shift
    mkdir -p "$(dirname "$log")"
    set -o pipefail
    "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set +o pipefail
    return "$rc"
}

# l3_manifest <run_dir> key=value ...
#   Appends structured provenance for one real run. Records the run_id once.
l3_manifest() {
    local dir=$1; shift
    local f="$dir/run_manifest.txt"
    { for kv in "$@"; do echo "$kv"; done; } >> "$f"
}

# l3_sha_file <path>: sha256 of a file, or the literal MISSING.
l3_sha_file() { [ -f "$1" ] && sha256sum "$1" 2>/dev/null | cut -d' ' -f1 || echo MISSING; }

# l3_binary_backend_check <exe> <expected: cuda|hip>
#   Fails if the binary's GPU backend does not match what was requested (so a
#   HIP request can never run a CUDA install and vice versa). Evidence: the
#   linked runtime libraries (libcudart / libamdhip64); a binary that links the
#   CUDA runtime statically (CMake's default CUDA_RUNTIME_LIBRARY=Static, e.g.
#   AMReX-based apps) is accepted when cuobjdump finds embedded device code.
l3_binary_backend_check() {
    local exe=$1 want=$2 libs
    [ -x "$exe" ] || { echo "l3: $exe not executable" >&2; return 1; }
    libs="$(ldd "$exe" 2>/dev/null || true)"
    case "$want" in
        cuda) if ! grep -q 'libcudart' <<<"$libs"; then
                  # capture first, then grep: under the callers' `set -o pipefail` a `cuobjdump | grep -q`
                  # pipe reports cuobjdump's SIGPIPE (grep -q exits early) as a failure on large binaries
                  local elf=""; command -v cuobjdump >/dev/null 2>&1 && elf="$(cuobjdump --list-elf "$exe" 2>/dev/null || true)"
                  if grep -q 'sm_' <<<"$elf"; then
                      : # static cudart with embedded CUDA device code
                  else
                      echo "l3: $exe is not a CUDA binary (no libcudart linked, no embedded CUDA ELF) but CUDA was requested" >&2; return 1
                  fi
              fi
              grep -q 'libamdhip64' <<<"$libs" && { echo "l3: $exe links libamdhip64 (HIP) but CUDA was requested" >&2; return 1; } ;;
        hip)  grep -q 'libamdhip64' <<<"$libs" || { echo "l3: $exe is not a HIP binary (no libamdhip64 linked) but HIP was requested" >&2; return 1; } ;;
        *) return 0 ;;
    esac
    return 0
}
