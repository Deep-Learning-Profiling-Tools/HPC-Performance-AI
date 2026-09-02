#!/usr/bin/env bash
# Run the standard ExaMiniMD benchmark (Lennard-Jones melt) on the CUDA or HIP
# build.
#
#   ./run.sh [CUDA|HIP] [extra ExaMiniMD args...]
#
# Default problem: upstream's input/in.lj (LAMMPS in.lj: fcc lattice at reduced
# density 0.8442, T = 1.4, LJ cutoff 2.5, NVE, neighbor rebuild every 20 steps)
# scaled from a 40^3 to a 160^3 unit-cell box (16,384,000 atoms) and run for
# 1000 steps with thermo output every 100 steps. That is ~10 s of time
# integration (~20 s wall including lattice setup) on one B200; upstream's
# 40^3 / 100-step deck runs in 0.04 s and is only useful as a smoke test.
# The scaled deck is generated with sed from input/in.lj into the build tree,
# so the copied upstream input stays untouched.
#
# Environment overrides:
#   HPCPERF_EXAMINIMD_LATTICE  unit cells per box edge (default 160; 4 atoms
#                              per cell, so N = 4*L^3). Keep L <= ~170:
#                              upstream indexes the 2D neighbor list with a
#                              32-bit int, and 200^3 (32 M atoms) crashes with
#                              cudaErrorIllegalAddress.
#   HPCPERF_EXAMINIMD_STEPS    number of MD steps (default 1000)
#   HPCPERF_EXAMINIMD_THERMO   thermo output interval (default 100)
#   HPCPERF_EXAMINIMD_DECK     run this input deck unchanged instead
#                              (absolute path or relative to level2/examinimd,
#                              e.g. input/in.lj); the three variables above
#                              are then ignored
#   HPCPERF_NP                 number of MPI ranks (default 1; ExaMiniMD
#                              uses one GPU per rank, so >1 only makes sense
#                              with several GPUs; needs CUDA-aware MPI, see
#                              OMPI_MCA_opal_cuda_support below)
#   OMP_NUM_THREADS etc.       Kokkos' host (OpenMP) backend is initialized
#                              too; defaults below keep it to one bound
#                              thread per rank and silence Kokkos' OMP warning
#
# Useful extra args (see src/input.cpp): --force-iteration NEIGH_FULL|
# NEIGH_HALF|CELL_FULL, --neigh-type 2D|CSR|CSR_MAPCONSTR, --comm-type
# MPI|SERIAL, --kokkos-num-threads=N, --kokkos-device-id=N.

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

# Generated decks and logs go under the build tree, not the caller's cwd.
RUN_DIR="$BUILD_DIR/run"
mkdir -p "$RUN_DIR"

if [ -n "${HPCPERF_EXAMINIMD_DECK:-}" ]; then
    DECK="$HPCPERF_EXAMINIMD_DECK"
    case "$DECK" in /*) ;; *) DECK="$HERE/$DECK" ;; esac
    if [ ! -f "$DECK" ]; then
        echo "run.sh: deck $DECK not found" >&2
        exit 1
    fi
else
    L="${HPCPERF_EXAMINIMD_LATTICE:-160}"
    STEPS="${HPCPERF_EXAMINIMD_STEPS:-1000}"
    THERMO="${HPCPERF_EXAMINIMD_THERMO:-100}"
    DECK="$RUN_DIR/in.lj.L${L}.s${STEPS}"
    sed -e "s/^region[[:space:]].*/region\t\tbox block 0 $L 0 $L 0 $L/" \
        -e "s/^thermo[[:space:]].*/thermo $THERMO/" \
        -e "s/^run[[:space:]].*/run\t\t$STEPS/" \
        "$HERE/input/in.lj" > "$DECK"
fi

# Kokkos initializes its OpenMP host backend as well; keep it to one pinned
# thread per rank (all work runs on the GPU) and silence the OMP_PROC_BIND
# warning Kokkos prints otherwise.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-threads}"

LAUNCH=()
COMM=(--comm-type SERIAL)
if grep -q '^USE_MPI:[A-Z]*=ON' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    NP="${HPCPERF_NP:-1}"
    LAUNCH=(mpirun -np "$NP")
    [ "$NP" -gt 1 ] && LAUNCH+=(--oversubscribe)
    COMM=(--comm-type MPI)
    # Upstream's CommMPI passes device (CudaSpace) buffers straight to
    # MPI_Send/MPI_Irecv, i.e. it needs a CUDA-aware MPI. The conda OpenMPI is
    # built with CUDA support but its openmpi-mca-params.conf turns it off
    # (opal_cuda_support = false); without this, any run with >1 rank
    # segfaults inside MPI_Send. Irrelevant for a single rank.
    export OMPI_MCA_opal_cuda_support="${OMPI_MCA_opal_cuda_support:-true}"
fi

echo "# ExaMiniMD $BACKEND: ${LAUNCH[*]:-} $EXE -il $DECK ${COMM[*]} $*"
exec "${LAUNCH[@]}" "$EXE" -il "$DECK" "${COMM[@]}" "$@"
