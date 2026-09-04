#!/usr/bin/env bash
# Run the standard ExaMiniMD benchmark (Lennard-Jones melt) on the CUDA or HIP
# build.
#
#   ./run.sh [CUDA|HIP] [extra ExaMiniMD args...]
#
# Distributed model: ExaMiniMD is a native-MPI molecular-dynamics mini-app --
# one global fcc lattice box, spatially decomposed by its Comm class, halo
# exchange of device buffers every step (CUDA-aware MPI required). One MPI
# rank per GPU; multi-rank launches go through
# level2/tools/hpcperf_mpi_launch.sh (Kokkos maps each rank to its GPU by
# local rank; the launcher audits the mapping).
#
# Resource / size controls (common Level 2 parameters):
#   HPCPERF_GPUS=N|all      ranks = GPUs (default 1)
#   HPCPERF_SCALE_MODE      smoke | strong | weak    (default strong)
#     strong : ONE fixed global box, HPCPERF_EXAMINIMD_LATTICE^3 unit cells
#              (default 160^3 = 16,384,000 atoms -- the historical Level 2
#              single-GPU deck), decomposed over the ranks
#     weak   : fixed per-rank volume: HPCPERF_EXAMINIMD_LOCAL^3 cells per rank
#              (default 100^3 = 4,000,000 atoms/rank); the global box is
#              L*PX x L*PY x L*PZ with PXxPYxPZ from hpcperf_topology.py
#     smoke  : upstream's own 40^3 deck, 100 steps, seconds-fast
#   HPCPERF_EXAMINIMD_LATTICE  strong-mode global cells per edge (default 160)
#   HPCPERF_EXAMINIMD_LOCAL    weak-mode cells per edge PER RANK (default 100)
#   HPCPERF_EXAMINIMD_STEPS    MD steps (default 1000; smoke 100)
#   HPCPERF_EXAMINIMD_THERMO   thermo interval (default 100)
#   HPCPERF_EXAMINIMD_DECK     run this input deck unchanged (size logic off)
#
# Constraint checked here: upstream's neighbor list is indexed with 32-bit
# ints PER RANK; ~20M atoms per rank is the practical ceiling (a 200^3
# single-rank box, 32M atoms, dies with cudaErrorIllegalAddress). The check
# is on atoms/rank, so larger totals are fine with more ranks.
#
# Useful extra args (src/input.cpp): --force-iteration NEIGH_FULL|NEIGH_HALF|
# CELL_FULL, --neigh-type 2D|CSR|CSR_MAPCONSTR, --comm-type MPI|SERIAL.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true
set -euo pipefail
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
[ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level2/examinimd/$MODEL"
EXE="$BUILD_DIR/src/ExaMiniMD"
if [ ! -x "$EXE" ]; then
    echo "run.sh: $EXE not found -- run ./build.sh $BACKEND first" >&2
    exit 1
fi
RUN_DIR="$BUILD_DIR/run"
mkdir -p "$RUN_DIR"

# shellcheck disable=SC1091
source "$R/level2/tools/hpcperf_launch_common.sh"
N_RANKS="$(hpcperf_ranks examinimd yes)" || exit 2
hpcperf_forbid_args examinimd -il -- "$@" || exit 2   # the deck is generated and size-checked here
MODE="${HPCPERF_SCALE_MODE:-strong}"
STEPS="${HPCPERF_EXAMINIMD_STEPS:-1000}"
THERMO="${HPCPERF_EXAMINIMD_THERMO:-100}"

if [ -n "${HPCPERF_EXAMINIMD_DECK:-}" ]; then
    DECK="$HPCPERF_EXAMINIMD_DECK"
    case "$DECK" in /*) ;; *) DECK="$HERE/$DECK" ;; esac
    [ -f "$DECK" ] || { echo "run.sh: deck $DECK not found" >&2; exit 1; }
else
    case "$MODE" in
        smoke)  LX=40; LY=40; LZ=40; STEPS="${HPCPERF_EXAMINIMD_STEPS:-100}" ;;
        strong) L="${HPCPERF_EXAMINIMD_LATTICE:-160}"; LX=$L; LY=$L; LZ=$L ;;
        weak)   L="${HPCPERF_EXAMINIMD_LOCAL:-100}"
                TOPO="$(hpcperf_topology examinimd "$N_RANKS")" || exit 2
                read -r PX PY PZ <<< "$TOPO"
                LX=$((L * PX)); LY=$((L * PY)); LZ=$((L * PZ)) ;;
        *) echo "run.sh: HPCPERF_SCALE_MODE must be smoke|strong|weak (got '$MODE')" >&2; exit 2 ;;
    esac
    ATOMS=$((4 * LX * LY * LZ))
    PER_RANK=$((ATOMS / N_RANKS))
    if [ "$PER_RANK" -gt 20000000 ]; then
        echo "run.sh: $PER_RANK atoms/rank ($ATOMS total over $N_RANKS ranks) exceeds the" >&2
        echo "run.sh: ~20M/rank 32-bit neighbor-list ceiling; use more ranks or a smaller box." >&2
        exit 2
    fi
    DECK="$RUN_DIR/in.lj.${LX}x${LY}x${LZ}.s${STEPS}"
    sed -e "s/^region[[:space:]].*/region\t\tbox block 0 $LX 0 $LY 0 $LZ/" \
        -e "s/^thermo[[:space:]].*/thermo $THERMO/" \
        -e "s/^run[[:space:]].*/run\t\t$STEPS/" \
        "$HERE/input/in.lj" > "$DECK"
    echo "# ExaMiniMD $BACKEND: mode=$MODE ranks=$N_RANKS box=${LX}x${LY}x${LZ} cells = $ATOMS atoms ($PER_RANK/rank), $STEPS steps"
fi

# Kokkos initializes its OpenMP host backend as well; keep it to one pinned
# thread per rank (all work runs on the GPU) and silence the OMP_PROC_BIND
# warning Kokkos prints otherwise.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-threads}"

COMM=(--comm-type SERIAL)
if grep -q '^USE_MPI:[A-Z]*=ON' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    COMM=(--comm-type MPI)
    # Upstream's CommMPI passes device (CudaSpace) buffers straight to
    # MPI_Send/MPI_Irecv, i.e. it needs a CUDA-aware MPI (hpcperf_env.sh sets
    # OMPI_MCA_opal_cuda_support=true; confirm with ./check_env.sh --mpi-cuda).
    export OMPI_MCA_opal_cuda_support="${OMPI_MCA_opal_cuda_support:-true}"
elif [ "$N_RANKS" -gt 1 ]; then
    echo "run.sh: build has USE_MPI=OFF but $N_RANKS ranks requested" >&2
    exit 2
fi
exec "$R/level2/tools/hpcperf_mpi_launch.sh" --gpus "$N_RANKS" --bind app -- \
    "$EXE" -il "$DECK" "${COMM[@]}" "$@"
