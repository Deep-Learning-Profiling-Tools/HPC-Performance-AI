#!/usr/bin/env bash
# Run the standard HACCabanaPM benchmark on the CUDA or HIP build.
#
#   ./run.sh [CUDA|HIP] [extra Kokkos runtime options, e.g. --kokkos-num-threads=4]
#
# Two stages, exactly upstream's workflow (docs/RUNNING.md):
#   pm_ic   <indat> <ic.h5>                 Zel'dovich initial conditions
#   pm_run  <ic.h5> <evolved.h5> <indat>    DKD particle-mesh evolution
#
# Default indat: apps/demo/indat_bench_256.params -- upstream's demo/reference
# indat (HACC "M000" cosmology, seed 5126873, transfer function cmbM000.tf)
# with NG = NP = 256, RL = 256 Mpc/h and upstream's validated evolution window
# z = 200 -> 0 in 500 DKD steps (16.8 M particles; ~5 s pm_ic + ~18 s pm_run on
# one B200). Runs with `mpirun -np 1`; more ranks need GPU-aware MPI and one
# GPU per rank (see README).
#
# pm_run's per-step timing (PMKOKKOS_TIMING=1: one CSV line per step and a
# "pm_run_timing: summary ... median_s=... total_s=..." line, upstream's
# benchmark metric) is on by default; the script also prints its own wall
# times for the two stages.
#
# Outputs (never in the repo): $R/build/level2/haccabanapm/<cuda|hip>/run/
#   ic_<indat-name>.h5, evolved_<indat-name>.h5   (0.77 GB each at 256^3)
#
# Environment overrides:
#   HPCPERF_HACCABANAPM_INDAT   indat file (absolute path, or relative to
#                               level2/haccabanapm); apps/demo/indat.params is
#                               upstream's 128^3, 5-step demo (validate.sh uses it)
#   HPCPERF_NP                  MPI ranks (default 1; multi-GPU via level2/tools/hpcperf_mpi_launch.sh
#                               and needs PMKOKKOS_TOPOLOGY="Px,Py,Pz" unless
#                               the indat has TOPOLOGY)
#   HPCPERF_HACCABANAPM_TIMING  1 (default) or 0 -> PMKOKKOS_TIMING
#   HPCPERF_HACCABANAPM_TAG     output file tag (default: indat basename)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true

set -euo pipefail

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
[ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level2/haccabanapm/$MODEL"

for exe in pm_ic pm_run; do
    if [ ! -x "$BUILD_DIR/$exe" ]; then
        echo "run.sh: $BUILD_DIR/$exe not found -- run ./build.sh $BACKEND first" >&2
        exit 1
    fi
done

INDAT="${HPCPERF_HACCABANAPM_INDAT:-apps/demo/indat_bench_256.params}"
case "$INDAT" in /*) ;; *) INDAT="$HERE/$INDAT" ;; esac
if [ ! -f "$INDAT" ]; then
    echo "run.sh: indat $INDAT not found" >&2
    exit 1
fi
INDAT_DIR="$(cd "$(dirname "$INDAT")" && pwd)"
INDAT="$INDAT_DIR/$(basename "$INDAT")"
TAG="${HPCPERF_HACCABANAPM_TAG:-$(basename "${INDAT%.*}")}"

# pm_ic resolves INPUT_BASE_NAME (the CAMB transfer function) relative to the
# working directory, so both stages run from the indat's directory.
TF="$(sed -n 's/^INPUT_BASE_NAME[[:space:]]\+\([^[:space:]#]*\).*/\1/p' "$INDAT" | head -1)"
if [ -n "$TF" ] && [ ! -f "$INDAT_DIR/$TF" ]; then
    echo "run.sh: transfer function '$TF' (INPUT_BASE_NAME) not found next to $INDAT" >&2
    echo "        regenerate it with: python3 $HERE/tools/make_camb_transfer.py $INDAT_DIR/$TF" >&2
    exit 1
fi

OUT_DIR="$BUILD_DIR/run"
mkdir -p "$OUT_DIR"
IC="$OUT_DIR/ic_$TAG.h5"
EVOLVED="$OUT_DIR/evolved_$TAG.h5"

NP="${HPCPERF_NP:-1}"
LAUNCH=(mpirun -np "$NP")
if [ "$NP" -gt 1 ] && [ "${HPCPERF_ALLOW_OVERSUBSCRIBE:-0}" = "1" ]; then
    echo "WARNING: DEBUG ONLY: GPU/rank oversubscription is enabled." >&2
    LAUNCH+=(--oversubscribe)
fi
# For one-rank-per-GPU multi-GPU runs use level2/tools/hpcperf_mpi_launch.sh.

# Cabana's halo exchange, the Distributor migration and heFFTe hand device
# buffers straight to MPI (GPU-aware MPI is a hard requirement upstream, even
# at 1 rank: the periodic halos are exchanged with itself). The conda-forge
# OpenMPI is built with CUDA support but its etc/openmpi-mca-params.conf sets
# opal_cuda_support = 0, which makes MPI treat device pointers as host memory
# and segfault in opal_convertor_unpack. Turn it on unless the caller decided.
if [ "$BACKEND" = CUDA ]; then
    export OMPI_MCA_opal_cuda_support="${OMPI_MCA_opal_cuda_support:-true}"
fi
export PMKOKKOS_TIMING="${PMKOKKOS_TIMING:-${HPCPERF_HACCABANAPM_TIMING:-1}}"

cd "$INDAT_DIR"
echo "# HACCabanaPM $BACKEND: indat $INDAT (cwd $INDAT_DIR)"
echo "# ${LAUNCH[*]} $BUILD_DIR/pm_ic $INDAT $IC"
t0=$(date +%s.%N)
"${LAUNCH[@]}" "$BUILD_DIR/pm_ic" "$INDAT" "$IC" "$@"
t1=$(date +%s.%N)
printf '# pm_ic wall time: %.2f s\n' "$(echo "$t1 - $t0" | bc)"

echo "# ${LAUNCH[*]} $BUILD_DIR/pm_run $IC $EVOLVED $INDAT  (PMKOKKOS_TIMING=$PMKOKKOS_TIMING)"
t0=$(date +%s.%N)
"${LAUNCH[@]}" "$BUILD_DIR/pm_run" "$IC" "$EVOLVED" "$INDAT" "$@"
t1=$(date +%s.%N)
printf '# pm_run wall time: %.2f s\n' "$(echo "$t1 - $t0" | bc)"
echo "# outputs: $IC $EVOLVED"
