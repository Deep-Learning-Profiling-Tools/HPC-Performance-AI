#!/usr/bin/env bash
# Run the LAMMPS Lennard-Jones benchmark (upstream bench/in.lj) on N GPUs.
#
#   ./run.sh [CUDA|HIP] [extra lmp args...]
#
# Execution model (upstream Speed_kokkos): one MPI rank per GPU, KOKKOS
# package on the device, one host thread per rank. Ranks are launched through
# the common launcher with the per-rank GPU wrapper, so every rank sees exactly
# one GPU and LAMMPS is started with `-k on g 1`; the launcher audits the
# rank->GPU mapping. GPU-aware MPI (device-buffer halo exchange,
# `-pk kokkos gpu/aware on`) is the LAMMPS default and matches the CUDA-aware
# Open MPI of this repository; HPCPERF_LAMMPS_GPU_AWARE=off disables it.
#
# Resource / size controls (common Level 3 parameters):
#   HPCPERF_GPUS=N|all        ranks = GPUs (default 1)
#   HPCPERF_SCALE_MODE        smoke | strong | weak   (default smoke)
#     smoke  : upstream bench/in.lj as shipped: 20^3 fcc cells = 32,000 atoms,
#              100 steps; bring-up / correctness (reference log in bench/)
#     strong : ONE fixed global box, (20*S)^3 cells with S=HPCPERF_LAMMPS_STRONG
#              (default 8: 160^3 cells = 16,384,000 atoms), decomposed by
#              LAMMPS over the ranks (any N is legal; LAMMPS factors the grid)
#     weak   : fixed work per rank: (20*L)^3 cells per rank, L=HPCPERF_LAMMPS_LOCAL
#              (default 4: 80^3 cells = 2,048,000 atoms/rank); the box is
#              20*L*PX x 20*L*PY x 20*L*PZ with PXxPYxPZ from hpcperf_topology.py
#              and the same grid is passed to LAMMPS `processors`
#   HPCPERF_LAMMPS_STEPS      MD steps (default 100, the bench convention)
#   HPCPERF_LAMMPS_GPU_AWARE  on|off (default on)
#
# LAMMPS decomposes the box into a PxQxR processor grid automatically for any
# rank count; in weak mode the grid is set explicitly to match the box shape.
# Extra args are appended to the lmp command line; arguments that would change
# the validated problem (-in, -var x/y/z, -k, -sf, -pk) are rejected.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"; [ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
EXE="$R/build/level3/lammps/$MODEL/lmp_kokkos_$MODEL"
[ -x "$EXE" ] || { echo "run.sh: $EXE not found -- run ./build.sh $BACKEND first" >&2; exit 1; }
SRC="$R/_upstream/level3/lammps"

N_RANKS="$(hpcperf_ranks lammps yes)" || exit 2
hpcperf_forbid_args lammps -in -i -var -v -k -kokkos -sf -suffix -pk -package -log -- "$@" || exit 2
MODE="$(l3_scale_mode lammps)" || exit 2
STEPS="${HPCPERF_LAMMPS_STEPS:-100}"
GAM="${HPCPERF_LAMMPS_GPU_AWARE:-on}"

PROCS=()
case "$MODE" in
    smoke)  X=1; Y=1; Z=1 ;;
    strong) S="${HPCPERF_LAMMPS_STRONG:-8}"; X=$S; Y=$S; Z=$S ;;
    weak)   L="${HPCPERF_LAMMPS_LOCAL:-4}"
            TOPO="$(hpcperf_topology lammps "$N_RANKS")" || exit 2
            read -r PX PY PZ <<< "$TOPO"
            X=$((L * PX)); Y=$((L * PY)); Z=$((L * PZ))
            PROCS=(-var px "$PX" -var py "$PY" -var pz "$PZ") ;;
esac
ATOMS=$(( 4 * 20 * X * 20 * Y * 20 * Z ))
RUN_DIR="$R/build/level3/lammps/$MODEL/run"; mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/log.$MODE.np$N_RANKS.lammps"
# Derived deck (upstream bench/in.lj untouched): `run 100` -> `run ${steps}`,
# and in weak mode a `processors ${px} ${py} ${pz}` line before create_box so
# the rank grid matches the box shape. With steps=100 and no processors line
# the derived deck is semantically identical to upstream's.
IN="$RUN_DIR/in.lj.$MODE"
{
    if [ "${#PROCS[@]}" -gt 0 ]; then
        sed -e 's/^create_box.*/processors      ${px} ${py} ${pz}\n&/' -e 's/^run[[:space:]].*/run             ${steps}/' "$SRC/bench/in.lj"
    else
        sed -e 's/^run[[:space:]].*/run             ${steps}/' "$SRC/bench/in.lj"
    fi
} > "$IN"

echo "# LAMMPS $BACKEND: mode=$MODE ranks=$N_RANKS box=$((20*X))x$((20*Y))x$((20*Z)) fcc cells = $ATOMS atoms ($((ATOMS / N_RANKS))/rank), $STEPS steps, gpu-aware=$GAM, log=$LOG"
exec "$L3_LAUNCHER" --gpus "$N_RANKS" --bind wrapper -- \
    "$EXE" -k on g 1 t "${HPCPERF_CPUS_PER_RANK:-1}" -sf kk -pk kokkos newton on neigh half gpu/aware "$GAM" \
    -in "$IN" -var x "$X" -var y "$Y" -var z "$Z" "${PROCS[@]}" -var steps "$STEPS" \
    -log "$LOG" -echo none "$@"
