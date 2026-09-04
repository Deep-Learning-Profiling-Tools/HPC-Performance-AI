#!/usr/bin/env bash
# Run hipBone (OCCA Nekbone port) on the CORAL-2 single-GPU problem.
#
#   ./run.sh [CUDA|HIP] [hipBone args...]        (default: CUDA)
#
# Default problem (upstream README "CORAL-2 problem size ... on one GPU"):
#   mpirun -np 1 ./hipBone -m CUDA -nx 24 -ny 24 -nz 24 -p 14
# i.e. 24^3 = 13824 spectral elements of polynomial degree 14 per rank,
# 37,595,375 DOFs, 1000 warm-up CG iterations followed by 100 timed ones.
#
# Any extra arguments REPLACE the default problem specification (hipBone
# aborts if a setting is given twice, so they cannot simply be appended);
# `-m <mode>` is always added by this script, do not pass it. Example:
#   ./run.sh CUDA -nx 8 -ny 8 -nz 8 -p 8 -v
#
# Environment knobs:
#   HIPBONE_NP       number of MPI ranks (default 1; one GPU is visible here)
#   OMP_NUM_THREADS  host OpenMP threads (default 4, see README "Run")
#   OCCA_CACHE_DIR   OCCA kernel cache (default $R/build/level2/hipbone/occa-cache)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

BACKEND="$(printf '%s' "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
case "$BACKEND" in
    CUDA|HIP) ;;
    *) echo "usage: $0 [CUDA|HIP] [hipBone args...]" >&2; exit 2 ;;
esac
shift $(( $# > 0 ? 1 : 0 ))

if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$R" ] && [ -f "$R/hpcperf_env.sh" ]; then
    set +eu
    # shellcheck disable=SC1091
    source "$R/hpcperf_env.sh" 2>/dev/null
    set -eu
fi

LC="$(printf '%s' "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD="$R/build/level2/hipbone/$LC"
BIN="$BUILD/hipBone"
if [ ! -x "$BIN" ]; then
    echo "error: $BIN not found -- run $HERE/build.sh $BACKEND first" >&2
    exit 1
fi

# --- OCCA JIT environment -----------------------------------------------------
# hipBone's kernels are compiled the first time a (kernel, problem size, device)
# combination is seen and cached. Keep the cache inside the build tree (OCCA
# would otherwise use ~/.occa or <exe dir>/.occa) and make the JIT use the same
# toolchain as the build: OCCA_CXX for the host launcher / Serial kernels and
# the project's nvcc + host compiler for CUDA kernels.
export OCCA_CACHE_DIR="${OCCA_CACHE_DIR:-$R/build/level2/hipbone/occa-cache}"
export HIPBONE_CACHE_DIR="${HIPBONE_CACHE_DIR:-$OCCA_CACHE_DIR}"
export OCCA_CXX="${OCCA_CXX:-${CXX:-g++}}"
if [ "$BACKEND" = "CUDA" ]; then
    export OCCA_CUDA_COMPILER="${OCCA_CUDA_COMPILER:-nvcc -ccbin ${CUDAHOSTCXX:-${CXX:-g++}}}"
fi
# hipBone's host code has OpenMP regions on the CG critical path (halo
# exchange, libp::memory ops). On this shared 16-core node the OpenMP default
# (one thread per allocated core, all of them busy with other jobs) costs
# ~13 ms per CG iteration; a handful of threads is plenty for a GPU run.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../tools" && pwd)/hpcperf_launch_common.sh"
if [ -n "${HIPBONE_NP:-}" ] && [ -z "${HPCPERF_GPUS:-}${HPCPERF_NP:-}" ]; then export HPCPERF_NP="$HIPBONE_NP"; fi   # legacy alias
N_RANKS="$(hpcperf_ranks hipbone no)" || exit 2   # HPCPERF_GPUS (or legacy HPCPERF_NP/HIPBONE_NP); no scale modes yet

if [ $# -gt 0 ]; then
    ARGS=("$@")
else
    ARGS=(-nx 24 -ny 24 -nz 24 -p 14)
fi
# hipBone distributes the box over a px x py x pz rank grid: without explicit
# -px/-py/-pz the rank count must be a perfect cube (LIBP_ABORT otherwise);
# with them the product must equal the rank count. Checked here, not left to
# a crash inside the app.
if [ "$N_RANKS" -gt 1 ]; then
    _px=""; _py=""; _pz=""; _prev=""
    for _a in "${ARGS[@]}"; do
        case "$_prev" in -px) _px="$_a";; -py) _py="$_a";; -pz) _pz="$_a";; esac; _prev="$_a"
    done
    if [ -n "$_px$_py$_pz" ]; then
        [ -n "$_px" ] && [ -n "$_py" ] && [ -n "$_pz" ] || { echo "run.sh: give all three of -px -py -pz" >&2; exit 2; }
        [ $(( _px * _py * _pz )) -eq "$N_RANKS" ] || { echo "run.sh: -px $_px -py $_py -pz $_pz product $((_px*_py*_pz)) != $N_RANKS ranks" >&2; exit 2; }
    else
        _c="$(python3 -c 'import sys; n=int(sys.argv[1]); c=round(n**(1/3)); print(c if c**3==n else 0)' "$N_RANKS")"
        [ "$_c" -gt 0 ] || { echo "run.sh: $N_RANKS ranks is not a perfect cube; pass -px -py -pz with product $N_RANKS (e.g. from level2/tools/hpcperf_topology.py $N_RANKS)" >&2; exit 2; }
    fi
fi
# One rank per GPU through the common launcher; hipBone binds each rank to its
# node-local GPU itself (hostname-based local rank), so the launcher only audits.
MPIRUN=("$HPCPERF_LAUNCHER_BIN" --gpus "$N_RANKS" --bind app --)

cd "$BUILD"
echo "== hipBone $BACKEND: ${MPIRUN[*]} ./hipBone -m $BACKEND ${ARGS[*]}  (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
exec "${MPIRUN[@]}" ./hipBone -m "$BACKEND" "${ARGS[@]}"
