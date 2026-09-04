#!/usr/bin/env bash
# Run the ExaMPM dam-break benchmark on the CUDA or HIP build.
#
#   ./run.sh [CUDA|HIP] [cell_size ppc halo_cells dt t_final write_freq]
#
# Upstream's executables take exactly seven positional arguments
# (examples/dam_break.cpp, examples/free_fall.cpp):
#
#   DamBreak <cell_size> <particles_per_cell_dim> <halo_cells> <dt> <t_final> <write_freq> <exec_space>
#
# where write_freq is in time steps and exec_space is serial|openmp|cuda|hip.
# run.sh fills in the exec space from the chosen backend; the six numeric
# arguments can be overridden on the command line (all six or none).
#
# Default problem: upstream's CI/README deck "DamBreak 0.05 2 0 0.001 0.25 100"
# (a 1 m^3 box, water column in one corner, 15,360 particles, 2.8 s wall on a
# B200 -- mostly start-up) refined to a 0.01 m grid and run to t = 1.0 s:
#
#   DamBreak 0.01 2 0 0.001 1.0 1000 cuda
#
# i.e. a 100^3 cell grid, 2^3 particles per cell in the initial column
# (1,920,000 particles), initial dt 1e-3 s shrunk by the CFL controller to
# ~2e-4 s (about 4,600 steps), HDF5/XDMF particle dumps every 1000 steps
# (5 dumps of 107 MB each). ~13-14 s wall on one B200.
# A larger alternative (15.4 M particles, ~26 s): ./run.sh CUDA 0.005 2 0 0.001 0.25 2000
#
# Output: ExaMPM prints "Time <t> / <t_final>" every write_freq steps and
# writes particles_<step>.h5/.xmf into the current directory, so run.sh
# runs in $R/build/level2/exampm/<cuda|hip>/run (created, previous dumps
# removed) and never into the repository.
#
# Environment overrides:
#   HPCPERF_EXAMPM_EXAMPLE  DamBreak (default) or FreeFall (the second
#                           upstream example: a periodic box with a falling
#                           sphere; validate.sh uses it)
#   HPCPERF_NP              number of MPI ranks (default 1; ExaMPM uses one
#                           GPU per rank, so >1 only makes sense with several
#                           GPUs)
#   HPCPERF_EXAMPM_RUN_DIR  directory to run in / write the dumps to
#                           (default $R/build/level2/exampm/<cuda|hip>/run;
#                           validate.sh uses its own so the benchmark output
#                           survives)
#   OMP_NUM_THREADS etc.    Kokkos' host (OpenMP) backend is initialized too;
#                           defaults below keep it to one bound thread per rank
#                           and silence Kokkos' OMP_PROC_BIND warning
#   OMPI_MCA_opal_cuda_support
#                           see below; default true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true

set -euo pipefail

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
[ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
case "$BACKEND" in
    CUDA|HIP) ;;
    *) echo "usage: $0 [CUDA|HIP] [cell_size ppc halo_cells dt t_final write_freq]" >&2; exit 2 ;;
esac
BUILD_DIR="$R/build/level2/exampm/$MODEL"
EXAMPLE="${HPCPERF_EXAMPM_EXAMPLE:-DamBreak}"
EXE="$BUILD_DIR/examples/$EXAMPLE"

if [ ! -x "$EXE" ]; then
    echo "run.sh: $EXE not found -- run ./build.sh $BACKEND first (HPCPERF_EXAMPM_EXAMPLE must be DamBreak or FreeFall)" >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    ARGS=(0.01 2 0 0.001 1.0 1000)
elif [ $# -eq 6 ]; then
    ARGS=("$@")
else
    echo "run.sh: give either no problem arguments or all six: cell_size ppc halo_cells dt t_final write_freq" >&2
    exit 2
fi

# Kokkos initializes its OpenMP host backend as well; keep it to one pinned
# thread per rank (all work runs on the GPU) and silence the OMP_PROC_BIND
# warning Kokkos prints otherwise.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-threads}"

# Cabana's grid halo exchange and particle migration pass device (CudaSpace)
# buffers straight to MPI_Isend/MPI_Irecv, i.e. ExaMPM needs a CUDA-aware
# MPI whenever a rank has a neighbour -- including itself, which is the case
# for every periodic direction (FreeFall is periodic in all three). The conda
# OpenMPI 5.0.10 is built with CUDA support, but its etc/openmpi-mca-params.conf
# turns it off (opal_cuda_support = 0), which makes the "accelerator/cuda"
# component refuse to initialize and device-buffer sends segfault. Turning
# the MCA parameter back on per run is enough. The default DamBreak deck is
# non-periodic and a single rank has no neighbour, so it runs either way.
export OMPI_MCA_opal_cuda_support="${OMPI_MCA_opal_cuda_support:-true}"

NP="${HPCPERF_NP:-1}"
LAUNCH=(mpirun -np "$NP")
if [ "$NP" -gt 1 ] && [ "${HPCPERF_ALLOW_OVERSUBSCRIBE:-0}" = "1" ]; then
    echo "WARNING: DEBUG ONLY: GPU/rank oversubscription is enabled." >&2
    LAUNCH+=(--oversubscribe)
fi
# For one-rank-per-GPU multi-GPU runs use level2/tools/hpcperf_mpi_launch.sh.

# ExaMPM writes its particle dumps into the cwd: run inside the build tree.
RUN_DIR="${HPCPERF_EXAMPM_RUN_DIR:-$BUILD_DIR/run}"
mkdir -p "$RUN_DIR"
rm -f "$RUN_DIR"/particles_*.h5 "$RUN_DIR"/particles_*.xmf
cd "$RUN_DIR"

echo "# ExaMPM $BACKEND: ${LAUNCH[*]} $EXE ${ARGS[*]} $MODEL  (cwd $RUN_DIR)"
exec "${LAUNCH[@]}" "$EXE" "${ARGS[@]}" "$MODEL"
