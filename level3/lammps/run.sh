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
VARIANT="${HPCPERF_LAMMPS_VARIANT:-}"     # "" (default package set) or reaxff -- see build.sh
PROFILE="$(l3_backend_profile LAMMPS "$MODEL" "$VARIANT")"
l3_paths_profile lammps "$PROFILE" "$MODEL" || exit 2     # same derivation as build.sh: binary, install and run tree of ONE profile
EXE="$L3_BUILD/lmp_kokkos_$MODEL"
[ -x "$EXE" ] || { echo "run.sh: $EXE not found -- run ./build.sh $BACKEND first (profile $PROFILE)" >&2; exit 1; }
l3_fingerprint_expect_backend "$L3_INSTALL" "$MODEL" || exit 1
l3_require_materialized "$HERE" || exit 3
SRC="$HERE/src"      # frozen source bundle (decks bench/in.lj live inside it; never _upstream)

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
DECK="in.lj"; LABEL="$MODE"
# Registered inputs (inputs.yaml, tools/inputs/hpcperf_inputs.py): HPCPERF_LAMMPS_INPUT=<id>
# selects one of the frozen bench/ decks with its recorded x/y/z replication and step count.
# The id replaces the smoke/strong/weak size policy, so it is refused together with
# HPCPERF_SCALE_MODE=strong|weak or with the size/step knobs (nothing is silently overridden).
INPUT_ID="${HPCPERF_LAMMPS_INPUT:-}"
if [ -n "$INPUT_ID" ]; then
    [ "$MODE" = smoke ] || { echo "run.sh: HPCPERF_LAMMPS_INPUT=$INPUT_ID and HPCPERF_SCALE_MODE=$MODE are mutually exclusive (the input id defines deck and size)" >&2; exit 2; }
    for v in HPCPERF_LAMMPS_STEPS HPCPERF_LAMMPS_STRONG HPCPERF_LAMMPS_LOCAL; do
        [ -z "${!v:-}" ] || { echo "run.sh: HPCPERF_LAMMPS_INPUT=$INPUT_ID and $v are mutually exclusive" >&2; exit 2; }
    done
    TOOL="$R/tools/inputs/hpcperf_inputs.py"
    DECK="$(python3 "$TOOL" param "$HERE" "$INPUT_ID" deck)" || exit 2
    X="$(python3 "$TOOL" param "$HERE" "$INPUT_ID" x)" || exit 2
    Y="$(python3 "$TOOL" param "$HERE" "$INPUT_ID" y)" || exit 2
    Z="$(python3 "$TOOL" param "$HERE" "$INPUT_ID" z)" || exit 2
    STEPS="$(python3 "$TOOL" param "$HERE" "$INPUT_ID" steps)" || exit 2
    ATOMS="$(python3 "$TOOL" param "$HERE" "$INPUT_ID" atoms)" || exit 2
    # deck_dir (default bench) names the directory of the frozen tree the deck lives in; a deck
    # that needs a build variant (ReaxFF) declares it and is refused on any other profile.
    DECK_DIR="$(python3 "$TOOL" param "$HERE" "$INPUT_ID" deck_dir 2>/dev/null)" || DECK_DIR=bench
    NEED_VARIANT="$(python3 "$TOOL" param "$HERE" "$INPUT_ID" variant 2>/dev/null)" || NEED_VARIANT=""
    case "$DECK_DIR/$DECK" in
        bench/in.lj|bench/in.eam|bench/in.chain|bench/in.rhodo|bench/in.chute|examples/reaxff/HNS/in.reaxff.hns) ;;
        *) echo "run.sh: input '$INPUT_ID' names deck '$DECK_DIR/$DECK', which is not one of the frozen upstream decks" >&2; exit 2 ;;
    esac
    [ -f "$SRC/$DECK_DIR/$DECK" ] || { echo "run.sh: $SRC/$DECK_DIR/$DECK missing from the frozen source tree" >&2; exit 3; }
    if [ "$NEED_VARIANT" != "$VARIANT" ]; then
        echo "run.sh: input '$INPUT_ID' needs build variant '${NEED_VARIANT:-default}' but profile $PROFILE is variant '${VARIANT:-default}' (set HPCPERF_LAMMPS_VARIANT=$NEED_VARIANT)" >&2; exit 2
    fi
    LABEL="input.$INPUT_ID"
else
    DECK_DIR=bench
fi
RUN_DIR="$L3_BUILD/$L3_RUN_SUBDIR"
# A dry-run must never touch real results: it writes its derived deck and would-be
# log into a throwaway .dryrun/ subdir instead of the real run directory.
[ -n "${HPCPERF_DRY_RUN:-}" ] && RUN_DIR="$RUN_DIR/.dryrun"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/log.$LABEL.np$N_RANKS.lammps"
rm -f "$LOG"      # validate only against THIS run's output; never a stale log left by a failed run
# Derived deck (upstream bench/ decks untouched): `run 100` -> `run ${steps}`,
# and in weak mode a `processors ${px} ${py} ${pz}` line before create_box so
# the rank grid matches the box shape. With steps=100 and no processors line
# the derived deck is semantically identical to upstream's. For a registered
# input the data file (read_data) and EAM potential (pair_coeff ... *.eam) names
# are made absolute so the deck can run from the run directory.
IN="$RUN_DIR/$DECK.$LABEL"
{
    if [ -n "$INPUT_ID" ]; then
        # read_data, EAM potentials and ReaxFF force-field files (pair_coeff * * ffield...) get absolute paths
        sed -e 's/^run[[:space:]].*/run             ${steps}/' \
            -e "s#^\(read_data[[:space:]]\{1,\}\)\([^[:space:]/]\{1,\}\)#\1$SRC/$DECK_DIR/\2#" \
            -e "s#^\(pair_coeff[[:space:]].*[[:space:]]\)\([A-Za-z0-9_.]*\.eam\)\([[:space:]]*\)\$#\1$SRC/$DECK_DIR/\2\3#" \
            -e "s#^\(pair_coeff[[:space:]]\{1,\}\*[[:space:]]\{1,\}\*[[:space:]]\{1,\}\)\(ffield[A-Za-z0-9_.]*\)#\1$SRC/$DECK_DIR/\2#" "$SRC/$DECK_DIR/$DECK"
    elif [ "${#PROCS[@]}" -gt 0 ]; then
        sed -e 's/^create_box.*/processors      ${px} ${py} ${pz}\n&/' -e 's/^run[[:space:]].*/run             ${steps}/' "$SRC/bench/in.lj"
    else
        sed -e 's/^run[[:space:]].*/run             ${steps}/' "$SRC/bench/in.lj"
    fi
} > "$IN"

if [ "$DECK_DIR" = bench ]; then GEOM="box=$((20*X))x$((20*Y))x$((20*Z)) fcc cells ="; else GEOM="replicate ${X}x${Y}x${Z} ="; fi
echo "# LAMMPS $BACKEND profile=$PROFILE: mode=$MODE${INPUT_ID:+ input=$INPUT_ID deck=$DECK_DIR/$DECK} ranks=$N_RANKS $GEOM $ATOMS atoms ($((ATOMS / N_RANKS))/rank), $STEPS steps, gpu-aware=$GAM, log=$LOG"
RUN_ID="$(l3_run_id)"
rc=0
"$L3_LAUNCHER" --gpus "$N_RANKS" --bind wrapper -- \
    "$EXE" -k on g 1 t "${HPCPERF_CPUS_PER_RANK:-1}" -sf kk -pk kokkos newton on neigh half gpu/aware "$GAM" \
    -in "$IN" -var x "$X" -var y "$Y" -var z "$Z" "${PROCS[@]}" -var steps "$STEPS" \
    -log "$LOG" -echo none "$@" || rc=$?
if [ -z "${HPCPERF_DRY_RUN:-}" ]; then
    l3_manifest "$RUN_DIR" "run_id=$RUN_ID" "app=lammps" "backend=$BACKEND" "profile=$PROFILE" "variant=${VARIANT:-default}" "mode=$MODE" "input_id=${INPUT_ID:-}" "deck=$DECK_DIR/$DECK" \
        "ranks=$N_RANKS" "atoms=$ATOMS" "steps=$STEPS" "gpu_aware=$GAM" "exit_code=$rc" \
        "binary=$EXE" "binary_sha256=$(l3_sha_file "$EXE")" "input=$IN" "input_sha256=$(l3_sha_file "$IN")" \
        "fingerprint=$L3_INSTALL/.hpcperf-l3-fingerprint" "fingerprint_sha256=$(l3_sha_file "$L3_INSTALL/.hpcperf-l3-fingerprint")" \
        "log=$LOG" "utc=$(date -u +%FT%TZ)"
fi
exit "$rc"
