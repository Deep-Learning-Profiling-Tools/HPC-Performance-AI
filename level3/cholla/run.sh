#!/usr/bin/env bash
# Run one registered Cholla input (a parameter file derived from upstream's examples/) on one GPU.
#
#   ./run.sh [CUDA]
#
# Execution model: one MPI rank per GPU (Cholla decomposes the grid over MPI ranks; MPI_CHOLLA build). The
# registered workloads are the one-rank runs -- the problem sizes are chosen for one B200 -- and run.sh
# refuses HPCPERF_GPUS other than 1 (a rank count the registry did not validate is never silently run).
# The rank goes through the common launcher with the per-rank GPU wrapper (the process sees exactly one
# GPU; mapping audited).
#
# Controls:
#   HPCPERF_CHOLLA_INPUT   a registered input id (inputs.yaml; default: the registry's default_input)
#   HPCPERF_GPUS=1         ranks = GPUs (only 1 is accepted, see above)
#   HPCPERF_SCALE_MODE     smoke only (the registered parameter files are the workloads)
# The run lives in its own directory ($L3_BUILD/$L3_RUN_SUBDIR/<id>.np1/): Cholla is given the registered
# parameter file (read in place from inputs/) and `outdir=<run dir>/out/` on its command line (upstream's
# own key=value override mechanism, the only override); it writes its HDF5 outputs under out/<n>/<n>.h5.0
# (the initial state and the final state at tout for every registered input), its run_output.log, and
# prints one 'n_step: ... timestep time = <ms>' line per time step (the timer, apptimers.py: cholla).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"; [ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
PROFILE="$(l3_backend_profile CHOLLA "$MODEL")"
l3_paths_profile cholla "$PROFILE" "$MODEL" || exit 2
BUILD_DIR="$L3_BUILD"
l3_fingerprint_expect_backend "$L3_INSTALL" "$MODEL" || exit 1
l3_require_materialized "$HERE" || exit 3
TOOL="$R/tools/inputs/hpcperf_inputs.py"

hpcperf_forbid_args cholla -- "$@" || exit 2
[ $# -eq 0 ] || { echo "run.sh: Cholla takes its parameter file from the registered input; extra arguments are refused ($*)" >&2; exit 2; }
MODE="$(l3_scale_mode cholla)" || exit 2
[ "$MODE" = smoke ] || { echo "run.sh: Cholla has no strong/weak derivation here (the registered parameter files are the workloads); HPCPERF_SCALE_MODE=$MODE is refused" >&2; exit 2; }
N_RANKS="$(hpcperf_ranks cholla no)" || exit 2
[ "$N_RANKS" -eq 1 ] || { echo "run.sh: the registered Cholla workloads are one-rank runs; HPCPERF_GPUS=$N_RANKS is refused (not adjusted)" >&2; exit 2; }
INPUT_ID="${HPCPERF_CHOLLA_INPUT:-$(python3 "$TOOL" list "$HERE" 2>/dev/null | awk '/\(default\)/{print $1}')}"
[ -n "$INPUT_ID" ] || { echo "run.sh: no input id (HPCPERF_CHOLLA_INPUT) and no default input in inputs.yaml" >&2; exit 2; }
param() { python3 "$TOOL" param "$HERE" "$INPUT_ID" "$1"; }
PARAMS_REL="$(param params)" || exit 2
INIT="$(param init)" || exit 2
GRID="$(param grid)" || exit 2
TOUT="$(param tout)" || exit 2
EXE="$L3_INSTALL/bin/cholla"
[ -x "$EXE" ] || { echo "run.sh: $EXE not found -- run ./build.sh $BACKEND (profile $PROFILE) first" >&2; exit 1; }
PARAMS="$HERE/$PARAMS_REL"
[ -f "$PARAMS" ] || { echo "run.sh: parameter file $PARAMS_REL missing" >&2; exit 3; }
grep -qE '^outdir=' "$PARAMS" && { echo "run.sh: $PARAMS_REL sets outdir; the registered files leave it to run.sh" >&2; exit 2; }

RUN_DIR="$(l3_rundir "$BUILD_DIR/$L3_RUN_SUBDIR/$INPUT_ID.np$N_RANKS")" || exit 2
mkdir -p "$RUN_DIR/out"
echo "# Cholla $BACKEND profile=$PROFILE: input=$INPUT_ID params=$PARAMS_REL init=$INIT grid=$GRID tout=$TOUT ranks=$N_RANKS run_dir=$RUN_DIR"
cd "$RUN_DIR"
RUN_ID="$(l3_run_id)"
rc=0
"$L3_LAUNCHER" --gpus "$N_RANKS" --bind wrapper -- "$EXE" "$PARAMS" "outdir=$RUN_DIR/out/" || rc=$?
FINAL="$(ls -d "$RUN_DIR"/out/*/ 2>/dev/null | sort -V | tail -1)"
if [ -z "${HPCPERF_DRY_RUN:-}" ]; then
    l3_manifest "$RUN_DIR" "run_id=$RUN_ID" "app=cholla" "backend=$BACKEND" "profile=$PROFILE" "mode=$MODE" \
        "input_id=$INPUT_ID" "params=$PARAMS_REL" "params_sha256=$(l3_sha_file "$PARAMS")" "init=$INIT" "grid=$GRID" "tout=$TOUT" \
        "ranks=$N_RANKS" "exit_code=$rc" "binary=$EXE" "binary_sha256=$(l3_sha_file "$EXE")" \
        "final_output=${FINAL:-none}" "final_output_sha256=$( [ -n "$FINAL" ] && l3_sha_file "$(ls "$FINAL"/*.h5.0 | head -1)" || echo MISSING)" \
        "fingerprint=$L3_INSTALL/.hpcperf-l3-fingerprint" "fingerprint_sha256=$(l3_sha_file "$L3_INSTALL/.hpcperf-l3-fingerprint")" \
        "utc=$(date -u +%FT%TZ)"
fi
exit "$rc"
