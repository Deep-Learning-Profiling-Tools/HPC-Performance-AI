#!/usr/bin/env bash
# HPC-Performance-AI project environment loader.
#
# Usage (every login / every new shell):
#   cd <repo>
#   source hpcperf_env.sh
#
# This script only LOADS the environment; one-time installation lives in
# ./setup_env.sh. Safe to source repeatedly (no PATH duplication). It never
# modifies shell startup files and works from any clone location.

# ------------------------------------------------------------- 1. project root
# Derived from this file's own location -- no hardcoded clone path.
export HPC_PERFORMANCE_AI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# Idempotent PATH-like helpers (no duplicate entries on repeated sourcing).
_hpcperf_prepend() { # $1 = var name, $2 = dir
    local _cur; eval "_cur=\${$1:-}"
    case ":${_cur}:" in
        *":$2:"*) ;;
        *) eval "export $1=\"$2\${_cur:+:\${_cur}}\"" ;;
    esac
}
_hpcperf_append() { # $1 = var name, $2 = dir
    local _cur; eval "_cur=\${$1:-}"
    case ":${_cur}:" in
        *":$2:"*) ;;
        *) eval "export $1=\"\${_cur:+\${_cur}:}$2\"" ;;
    esac
}

# ------------------------------------------- 2. project-local Conda environment
_HPCPERF_CONDA_ENV="$HPC_PERFORMANCE_AI_ROOT/.conda_env"

if [ -d "$_HPCPERF_CONDA_ENV" ]; then
    # Locate a conda base without relying on `conda init` / startup files:
    #   1. conda already on PATH   2. the project-local Miniforge that
    #   setup_env.sh bootstraps    3. a known site install (dev machine)
    _HPCPERF_CONDA_BASE=""
    if command -v conda >/dev/null 2>&1; then
        _HPCPERF_CONDA_BASE="$(conda info --base 2>/dev/null)"
    fi
    for _cand in "$HPC_PERFORMANCE_AI_ROOT/.tools/miniforge3" \
                 "/projects/kzhou6/bcui2/env_software/miniconda3"; do
        [ -n "$_HPCPERF_CONDA_BASE" ] && break
        [ -x "$_cand/bin/conda" ] && _HPCPERF_CONDA_BASE="$_cand"
    done
    if [ -n "$_HPCPERF_CONDA_BASE" ] && [ -f "$_HPCPERF_CONDA_BASE/etc/profile.d/conda.sh" ]; then
        . "$_HPCPERF_CONDA_BASE/etc/profile.d/conda.sh"
        if [ "${CONDA_PREFIX:-}" != "$_HPCPERF_CONDA_ENV" ]; then
            conda activate "$_HPCPERF_CONDA_ENV"
        fi
    else
        echo "[WARN] conda not found; project Conda environment not activated."
    fi
else
    echo "[WARN] Project Conda environment not found. Run ./setup_env.sh first."
fi

# Project-local tools (cloc installed by setup_env.sh).
if [ -d "$HPC_PERFORMANCE_AI_ROOT/.tools/bin" ]; then
    _hpcperf_prepend PATH "$HPC_PERFORMANCE_AI_ROOT/.tools/bin"
fi

# ------------------------------------------------------- 3. compiler selection
# setup_env.sh records the validated NVCC host compiler source in
# .tools/compiler_source ("conda" or "system"). Default: conda GCC 13.3.0.
_HPCPERF_CC_SOURCE=""
if [ -f "$HPC_PERFORMANCE_AI_ROOT/.tools/compiler_source" ]; then
    _HPCPERF_CC_SOURCE="$(cat "$HPC_PERFORMANCE_AI_ROOT/.tools/compiler_source")"
fi
if [ -z "$_HPCPERF_CC_SOURCE" ]; then
    if [ -x "$_HPCPERF_CONDA_ENV/bin/x86_64-conda-linux-gnu-cc" ]; then
        _HPCPERF_CC_SOURCE="conda"
    else
        _HPCPERF_CC_SOURCE="system"
    fi
fi
if [ "$_HPCPERF_CC_SOURCE" = "conda" ] && [ -x "$_HPCPERF_CONDA_ENV/bin/x86_64-conda-linux-gnu-cc" ]; then
    export CC="$_HPCPERF_CONDA_ENV/bin/x86_64-conda-linux-gnu-cc"
    export CXX="$_HPCPERF_CONDA_ENV/bin/x86_64-conda-linux-gnu-c++"
    export HPCPERF_GCC_SOURCE="conda"
else
    export CC=/usr/bin/gcc
    export CXX=/usr/bin/g++
    export HPCPERF_GCC_SOURCE="system"
fi
export CUDAHOSTCXX="$CXX"

# Some inherited shells carry broken gcc/g++ aliases; drop them so gcc/g++
# resolve through PATH in this session.
unalias gcc g++ 2>/dev/null || true

# -------------------------------------------------- 4-7. CUDA environment
# CUDA comes from the SYSTEM installation, never from conda. Default is
# /usr/local/cuda (the layout on the B200 dev machine); a machine with CUDA
# elsewhere can `export CUDA_HOME=/path/to/cuda` before sourcing this file.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ -x "$CUDA_HOME/bin/nvcc" ]; then
    _hpcperf_prepend PATH "$CUDA_HOME/bin"
    _hpcperf_prepend LD_LIBRARY_PATH "$CUDA_HOME/lib64"
else
    echo "[WARN] nvcc not found under CUDA_HOME=$CUDA_HOME -- CUDA benchmarks will not build."
    echo "       Install the CUDA toolkit or set CUDA_HOME before sourcing hpcperf_env.sh."
fi

# Nsight Compute python extras (profiling helpers; B200 dev machine layout).
if [ -d /opt/nvidia/nsight-compute/2026.1.1/extras/python ]; then
    _hpcperf_append PYTHONPATH /opt/nvidia/nsight-compute/2026.1.1/extras/python
fi

# CMake defaults (kept out of benchmark CMakeLists.txt):
#   - Ninja build backend
#   - CUDA architectures resolved from the local GPU ("native")
export CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"
export CUDAARCHS="${CUDAARCHS:-native}"

# ------------------------------------- 8. Level 2 framework libraries (.deps)
# setup_level2_deps.sh installs Kokkos, RAJA, hypre, MFEM, Cabana, heFFTe, ...
# into .deps/install/<name>. Expose every installed prefix to CMake so the
# Level 2 mini-apps find them with plain find_package().
if [ -d "$HPC_PERFORMANCE_AI_ROOT/.deps/install" ]; then
    for _dep in "$HPC_PERFORMANCE_AI_ROOT"/.deps/install/*/; do
        _dep="${_dep%/}"
        [ -f "$_dep/.hpcperf-built" ] || continue
        # The *environment* variable CMAKE_PREFIX_PATH is ':'-separated (like PATH).
        case ":${CMAKE_PREFIX_PATH:-}:" in
            *":$_dep:"*) ;;
            *) export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:+${CMAKE_PREFIX_PATH}:}$_dep" ;;
        esac
        [ -d "$_dep/lib" ]   && _hpcperf_prepend LD_LIBRARY_PATH "$_dep/lib"
        [ -d "$_dep/lib64" ] && _hpcperf_prepend LD_LIBRARY_PATH "$_dep/lib64"
    done
    unset _dep
fi
# Kokkos' nvcc_wrapper must call the same host compiler as everything else.
export NVCC_WRAPPER_DEFAULT_COMPILER="$CXX"

# MPI (conda Open MPI 5, used by the Level 2 mini-apps). Inside a Slurm
# allocation with fewer tasks than requested ranks, PRRTE refuses to start
# without oversubscription; allow it so `mpirun -np N` works as documented.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    export PRTE_MCA_rmaps_default_mapping_policy="${PRTE_MCA_rmaps_default_mapping_policy:-:oversubscribe}"
fi
# All Level 2 runs are single-node. Open MPI's default component selection
# (pml ucx, btl ofi/uct) probes every InfiniBand / libfabric device at start-up,
# which costs ~5 s in every MPI_Init on the dev machine; the shared-memory
# transports are all a single node needs (smcuda keeps CUDA IPC between ranks).
# Measured: MPI_Init 5.6 s -> 0.8 s. Set HPCPERF_MPI_SINGLE_NODE=0 (or set
# OMPI_MCA_pml / OMPI_MCA_btl yourself) for multi-node runs.
if [ "${HPCPERF_MPI_SINGLE_NODE:-1}" = "1" ]; then
    export OMPI_MCA_pml="${OMPI_MCA_pml:-ob1}"
    export OMPI_MCA_btl="${OMPI_MCA_btl:-self,sm,smcuda}"
fi

# CUDA-aware MPI. The conda Open MPI is built with CUDA support, but its
# etc/openmpi-mca-params.conf ships `opal_cuda_support = 0`, which makes the
# accelerator component return NULL before any CUDA call -- passing a device
# buffer to MPI_Send/Isend then segfaults (seen in ExaMiniMD, ExaMPM, and
# every Cabana multi-rank halo exchange). Turn it on for every shell; the
# environment overrides the conf file, and nothing in the repository is
# modified. Set OMPI_MCA_opal_cuda_support=false yourself to disable.
export OMPI_MCA_opal_cuda_support="${OMPI_MCA_opal_cuda_support:-true}"

# ----------------------------------------------- 9. optional ROCm environment
if [ -d /opt/rocm ]; then
    export ROCM_PATH=/opt/rocm
    _hpcperf_prepend PATH "$ROCM_PATH/bin"
    _hpcperf_prepend LD_LIBRARY_PATH "$ROCM_PATH/lib"
fi

unset _HPCPERF_CONDA_ENV _HPCPERF_CONDA_BASE _HPCPERF_CC_SOURCE _cand
