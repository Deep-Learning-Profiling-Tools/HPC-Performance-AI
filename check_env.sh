#!/usr/bin/env bash
# HPC-Performance-AI environment check.
#
# Usage:
#   source hpcperf_env.sh
#   ./check_env.sh                 # environment version report (read-only)
#   ./check_env.sh --mpi-cuda [NP] # ALSO run the CUDA-aware MPI runtime smoke
#                                  # test (default NP=2; builds a tiny binary)
#
# Compares the active environment against the versions this suite was
# validated with, printing [OK]/[WARN]/[FAIL]/[INFO] per component and
# Expected/Actual on any mismatch. The plain form is read-only; --mpi-cuda
# additionally compiles and runs level2/tools/mpi_cuda_check.
set -u

DO_MPI_CUDA=0; MPI_CUDA_NP=2
while [ $# -gt 0 ]; do
    case "$1" in
        --mpi-cuda) DO_MPI_CUDA=1; shift
                    case "${1:-}" in ''|-*) : ;; *) MPI_CUDA_NP="$1"; shift ;; esac ;;
        *) echo "check_env.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

# Validated environment (what the pinned environment.yml + dev machine provide)
EXP_GCC="13.3.0"
EXP_CMAKE="3.28.4"
EXP_PY="3.12.3"
EXP_NINJA="1.13.2"
EXP_CLOC="2.06"
EXP_CUDA_MM="13.2"      # CUDA major.minor the suite was validated with
EXP_NCU="2026.1.1"      # optional (profiling only)
EXP_OMPI="5.0.10"       # Level 2 (conda Open MPI, CUDA-aware build)

ROOT="${HPC_PERFORMANCE_AI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
STATUS=0
ok()   { printf '[OK]   %s\n' "$*"; }
info() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*"; STATUS=1; }
check_version() { # $1 label, $2 expected, $3 actual, $4 severity-on-mismatch(warn|fail)
    local label="$1" exp="$2" act="$3" sev="${4:-warn}"
    if [ -z "$act" ]; then
        fail "$label: not found (expected $exp)"
    elif [ "$act" = "$exp" ]; then
        ok "$label $act"
    else
        "$sev" "$label mismatch -- Expected: $exp  Actual: $act"
    fi
}

# --- OS (informational; the suite was validated on RHEL 10 / kernel 6.12)
info "OS: $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") (validated on Red Hat Enterprise Linux 10.0)"

# --- environment manager / conda
CONDA_CMD="${CONDA_EXE:-}"
[ -x "$CONDA_CMD" ] || CONDA_CMD="$(command -v conda 2>/dev/null || true)"
if [ -n "$CONDA_CMD" ] && [ -x "$CONDA_CMD" ]; then
    ok "Environment Manager: Conda ($("$CONDA_CMD" --version 2>/dev/null))"
else
    fail "Environment Manager: conda not found (no CONDA_EXE, not on PATH)"
fi
if [ "${CONDA_PREFIX:-}" = "$ROOT/.conda_env" ]; then
    ok "Conda Prefix: $CONDA_PREFIX"
else
    fail "Conda Prefix: expected $ROOT/.conda_env, got '${CONDA_PREFIX:-<unset>}' (source hpcperf_env.sh first)"
fi

# --- compilers
check_version "GCC" "$EXP_GCC" "$(${CC:-gcc} -dumpfullversion 2>/dev/null)"
check_version "G++" "$EXP_GCC" "$(${CXX:-g++} -dumpfullversion 2>/dev/null)"
info "CC=${CC:-<unset>}"
info "CXX=${CXX:-<unset>}"
info "GCC source: ${HPCPERF_GCC_SOURCE:-<unset>} (from hpcperf_env.sh / .tools/compiler_source)"

# --- build tools
check_version "CMake" "$EXP_CMAKE" "$(cmake --version 2>/dev/null | sed -n 's/^cmake version //p')"
check_version "Ninja" "$EXP_NINJA" "$(ninja --version 2>/dev/null)"
check_version "cloc" "$EXP_CLOC" "$(cloc --version 2>/dev/null)"
check_version "Python" "$EXP_PY" "$(python --version 2>/dev/null | awk '{print $2}')"
case "$(command -v python)" in
    "$ROOT"/.conda_env/*) ok "python resolves into project env" ;;
    *) warn "python does not resolve into $ROOT/.conda_env: $(command -v python)" ;;
esac
if python -c "import numpy, scipy" 2>/dev/null; then
    ok "numpy/scipy importable (needed by verify.py checks)"
else
    fail "numpy/scipy not importable in the active python"
fi

# --- CUDA (system-provided; NOT installed by setup_env.sh)
if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
    ok "CUDA_HOME=$CUDA_HOME (-> $(readlink -f "$CUDA_HOME"))"
else
    fail "CUDA_HOME: '${CUDA_HOME:-<unset>}' has no nvcc (install CUDA >= 13 or export CUDA_HOME)"
fi
NVCC_PATH="$(command -v nvcc || true)"
if [ -n "$NVCC_PATH" ]; then
    ok "NVCC=$NVCC_PATH"
    ACT_CUDA="$(nvcc --version 2>/dev/null | sed -n 's/^Cuda compilation tools, release [^,]*, V//p')"
    case "$ACT_CUDA" in
        "$EXP_CUDA_MM".*) ok "CUDA Toolkit $ACT_CUDA (validated with $EXP_CUDA_MM.x)" ;;
        13.*) warn "CUDA Toolkit $ACT_CUDA -- suite validated with $EXP_CUDA_MM.x; other 13.x should work" ;;
        *)    warn "CUDA Toolkit $ACT_CUDA -- suite validated with $EXP_CUDA_MM.x; older toolkits are untested" ;;
    esac
else
    fail "nvcc not on PATH"
fi

# --- driver / GPU
if command -v nvidia-smi >/dev/null 2>&1; then
    ok "NVIDIA driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) (validated on NVIDIA B200)"
else
    fail "nvidia-smi not available (no GPU visible?)"
fi

# --- Nsight Compute (optional, profiling only)
ACT_NCU="$(ncu --version 2>/dev/null | sed -n 's/^Version \([0-9.]*\.[0-9]*\)\..*/\1/p' | head -1)"
if [ -n "$ACT_NCU" ]; then
    [ "$ACT_NCU" = "$EXP_NCU" ] && ok "Nsight Compute $ACT_NCU" || info "Nsight Compute $ACT_NCU (validated with $EXP_NCU)"
else
    info "Nsight Compute not found (optional; only needed for profiling)"
fi

# --- MPI (Level 2 mini-apps; conda Open MPI, installed by setup_env.sh)
if command -v mpicxx >/dev/null 2>&1; then
    check_version "Open MPI" "$EXP_OMPI" "$(mpirun --version 2>/dev/null | sed -n 's/^mpirun (Open MPI) //p')"
    case "$(command -v mpicxx)" in
        "$ROOT"/.conda_env/*) ok "mpicxx resolves into project env" ;;
        *) warn "mpicxx does not resolve into $ROOT/.conda_env: $(command -v mpicxx)" ;;
    esac
    info "MPI transport: pml=${OMPI_MCA_pml:-<default/UCX>} btl=${OMPI_MCA_btl:-<default>}; single-node SM profile is opt-in (HPCPERF_MPI_SINGLE_NODE=1). opal_cuda_support=${OMPI_MCA_opal_cuda_support:-<conf file: 0>} (run ./check_env.sh --mpi-cuda to confirm the runtime capability)"
else
    warn "mpicxx not found -- Level 2 mini-apps need MPI (re-run setup_env.sh to update .conda_env)"
fi

# --- Level 2 framework libraries (setup_level2_deps.sh -> .deps/install/<name>)
if [ -d "$ROOT/.deps/install" ]; then
    for _dep in "$ROOT"/.deps/install/*/; do
        _dep="${_dep%/}"; _name="${_dep##*/}"
        if [ -f "$_dep/.hpcperf-built" ]; then
            ok "Level 2 dep: $_name ($(cat "$_dep/.hpcperf-built" 2>/dev/null))"
        else
            warn "Level 2 dep: $_name present but not marked built (incomplete build?)"
        fi
    done
else
    info "Level 2 framework libraries not built (run ./setup_level2_deps.sh; only needed for Level 2)"
fi

# --- CUDA-aware MPI runtime capability (opt-in: --mpi-cuda)
if [ "$DO_MPI_CUDA" = "1" ]; then
    if ! command -v mpicc >/dev/null 2>&1; then
        fail "CUDA-aware MPI: mpicc not found (source hpcperf_env.sh first)"
    else
        info "CUDA-aware MPI requested: OMPI_MCA_opal_cuda_support=${OMPI_MCA_opal_cuda_support:-<conf file: 0>}"
        out="$("$ROOT/level2/tools/mpi_cuda_check/run.sh" "$MPI_CUDA_NP" 2>&1)"
        rc=$?
        echo "$out" | grep -E 'mpi_cuda_check:|rank [0-9]' | sed 's/^/       /'
        if [ "$rc" = 0 ]; then
            ok "CUDA-aware MPI runtime capability confirmed (device-buffer MPI, ${MPI_CUDA_NP} ranks, numerically checked)"
        else
            fail "CUDA-aware MPI runtime check failed (rc=$rc; see output above)"
        fi
    fi
fi

# --- ROCm (optional; HIP backends are unverified without it)
if command -v hipcc >/dev/null 2>&1; then
    ok "hipcc: $(hipcc --version 2>/dev/null | head -1)"
elif [ -d /opt/rocm ]; then
    warn "/opt/rocm exists but hipcc not on PATH"
else
    info "ROCm unavailable (expected on NVIDIA machines; HIP backends stay unverified)"
fi

exit "$STATUS"
