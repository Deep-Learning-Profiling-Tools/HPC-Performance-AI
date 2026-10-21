#!/usr/bin/env bash
# Run one registered PIConGPU input (a compiled input set with its runtime grid / step count / plugins) on one GPU.
#
#   ./run.sh [CUDA]
#
# Execution model: one MPI rank per GPU (alpaka CUDA backend); the registered inputs fix a 1 x 1 x 1 device
# decomposition (-d 1 1 1 is part of the registered arguments), so run.sh refuses HPCPERF_GPUS other than 1 -- a
# different decomposition would be another registered input. The rank goes through the common launcher with the
# per-rank GPU wrapper (the process sees exactly one GPU; mapping audited).
#
# Controls:
#   HPCPERF_PICONGPU_INPUT   a registered input id (inputs.yaml; default: the registry's default_input)
#   HPCPERF_GPUS=1           ranks = GPUs (only 1 is accepted, see above)
#   HPCPERF_SCALE_MODE       smoke only (the registered inputs are the workloads; no strong/weak derivation)
# The run lives in $L3_BUILD/$L3_RUN_SUBDIR/<id>.np1/ (PIConGPU writes its plugin files -- fields_energy.dat,
# chargeConservation.dat, e_energy_all.dat -- into the working directory); the registered arguments (grid, steps,
# periodicity, plugin periods) are the executable's whole command line. The run manifest records the input set's
# hash the executable was compiled from (build.sh) and refuses a stale executable.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"; [ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
PROFILE="$(l3_backend_profile PICONGPU "$MODEL")"
l3_paths_profile picongpu "$PROFILE" "$MODEL" || exit 2
BUILD_DIR="$L3_BUILD"
l3_fingerprint_expect_backend "$L3_INSTALL" "$MODEL" || exit 1
l3_require_materialized "$HERE" || exit 3
TOOL="$R/tools/inputs/hpcperf_inputs.py"

hpcperf_forbid_args picongpu -d -g -s --periodic -- "$@" || exit 2
[ $# -eq 0 ] || { echo "run.sh: PIConGPU's command line is the registered input's argument list; extra arguments are refused ($*)" >&2; exit 2; }
MODE="$(l3_scale_mode picongpu)" || exit 2
[ "$MODE" = smoke ] || { echo "run.sh: PIConGPU has no strong/weak derivation here (the registered inputs are the workloads); HPCPERF_SCALE_MODE=$MODE is refused" >&2; exit 2; }
N_RANKS="$(hpcperf_ranks picongpu no)" || exit 2
[ "$N_RANKS" -eq 1 ] || { echo "run.sh: every registered input fixes -d 1 1 1 (one device); HPCPERF_GPUS=$N_RANKS is refused, not adjusted" >&2; exit 2; }
INPUT_ID="${HPCPERF_PICONGPU_INPUT:-$(python3 "$TOOL" list "$HERE" 2>/dev/null | awk '/\(default\)/{print $1}')}"
[ -n "$INPUT_ID" ] || { echo "run.sh: no input id (HPCPERF_PICONGPU_INPUT) and no default input in inputs.yaml" >&2; exit 2; }
param() { python3 "$TOOL" param "$HERE" "$INPUT_ID" "$1"; }
SET="$(param paramset)" || exit 2
GRID="$(param grid)" || exit 2
STEPS="$(param steps)" || exit 2
mapfile -t ARGS < <(python3 "$TOOL" args "$HERE" "$INPUT_ID") || exit 2   # one argument per line
EXE="$L3_INSTALL/$SET/bin/picongpu"
[ -x "$EXE" ] || { echo "run.sh: $EXE not found -- run ./build.sh $BACKEND (profile $PROFILE) first" >&2; exit 1; }
SET_SHA="$(find "$HERE/inputs/$SET/include" "$HERE/inputs/$SET/cmakeFlags" -type f | sort | xargs cat | sha256sum | cut -d' ' -f1)"
BUILT_SHA="$(cat "$EXE.paramset.sha256" 2>/dev/null || echo MISSING)"
[ "$SET_SHA" = "$BUILT_SHA" ] || { echo "run.sh: $EXE was built from an input set hashing $BUILT_SHA, but inputs/$SET now hashes to $SET_SHA -- rebuild (./build.sh $BACKEND)" >&2; exit 1; }

RUN_DIR="$(l3_rundir "$BUILD_DIR/$L3_RUN_SUBDIR/$INPUT_ID.np$N_RANKS")" || exit 2
echo "# PIConGPU $BACKEND profile=$PROFILE: input=$INPUT_ID paramset=inputs/$SET grid=$GRID steps=$STEPS ranks=$N_RANKS args=[${ARGS[*]}] run_dir=$RUN_DIR"
cd "$RUN_DIR"
RUN_ID="$(l3_run_id)"
rc=0
"$L3_LAUNCHER" --gpus "$N_RANKS" --bind wrapper -- "$EXE" "${ARGS[@]}" || rc=$?
if [ -z "${HPCPERF_DRY_RUN:-}" ]; then
    l3_manifest "$RUN_DIR" "run_id=$RUN_ID" "app=picongpu" "backend=$BACKEND" "profile=$PROFILE" "mode=$MODE" \
        "input_id=$INPUT_ID" "paramset=inputs/$SET" "paramset_sha256=$SET_SHA" "grid=$GRID" "steps=$STEPS" "args=${ARGS[*]}" \
        "ranks=$N_RANKS" "exit_code=$rc" "binary=$EXE" "binary_sha256=$(l3_sha_file "$EXE")" \
        "fields_energy=$RUN_DIR/fields_energy.dat" "fields_energy_sha256=$(l3_sha_file "$RUN_DIR/fields_energy.dat")" \
        "fingerprint=$L3_INSTALL/.hpcperf-l3-fingerprint" "fingerprint_sha256=$(l3_sha_file "$L3_INSTALL/.hpcperf-l3-fingerprint")" \
        "utc=$(date -u +%FT%TZ)"
fi
exit "$rc"
