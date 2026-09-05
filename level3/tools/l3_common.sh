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

L3_R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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
#   Prints the fingerprint for the configuration about to be built.
l3_fingerprint_text() {
    local app=$1 sha=$2 backend=$3 deps=$4 cmakeopts=$5 gam=$6; shift 6
    local p
    echo "schema=l3-1"
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
    for p in "$@"; do
        [ -e "$p" ] || continue
        echo "patch=$(basename "$p") sha256=$(sha256sum "$p" | cut -d' ' -f1)"
    done
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
