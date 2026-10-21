#!/usr/bin/env bash
# Run one registered GIZMO input (a compile-time Config.sh -- the executable GIZMO-<config> of build.sh -- with
# its parameter file and initial conditions) on one or more GPUs.
#
#   ./run.sh [CUDA]
#
# Execution model: one MPI rank per GPU (GIZMO's domain decomposition; the registered workloads are the
# one-rank runs, HPCPERF_GPUS=N is a strong-scaling run of the same problem). The rank goes through the
# common launcher with the per-rank GPU wrapper (the process sees exactly one GPU; mapping audited); the
# Kokkos CUDA backend holds the particle arena in CUDA unified memory (upstream's design).
#
# Controls:
#   HPCPERF_GIZMO_INPUT   a registered input id (inputs.yaml; default: the registry's default_input)
#   HPCPERF_GPUS=N        ranks = GPUs (default 1)
#   HPCPERF_SCALE_MODE    smoke only (the registered problems are the workloads; no strong/weak derivation)
# The run lives in its own directory ($L3_BUILD/$L3_RUN_SUBDIR/<id>.np<N>/): run.sh writes there a copy
# of the registered parameter file with InitCondFile pointing at ic.hdf5, a symlink to the frozen initial
# conditions (deps/ics/<file>.hdf5, part of the source bundle), and OutputDir at the run's own output/ --
# the only two lines that differ, both relative to the run directory (recorded as a generated file; the
# manifest hashes both and the real IC file). GIZMO writes
# snapshots, cpu.txt (the per-step wall-clock accounting the timer reads) and its other logs under output/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"; [ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
PROFILE="$(l3_backend_profile GIZMO "$MODEL")"
l3_paths_profile gizmo "$PROFILE" "$MODEL" || exit 2
BUILD_DIR="$L3_BUILD"
l3_fingerprint_expect_backend "$L3_INSTALL" "$MODEL" || exit 1
l3_require_materialized "$HERE" || exit 3
TOOL="$R/tools/inputs/hpcperf_inputs.py"

hpcperf_forbid_args gizmo -- "$@" || exit 2
[ $# -eq 0 ] || { echo "run.sh: GIZMO takes its parameter file from the registered input; extra arguments are refused ($*)" >&2; exit 2; }
MODE="$(l3_scale_mode gizmo)" || exit 2
[ "$MODE" = smoke ] || { echo "run.sh: GIZMO has no strong/weak derivation here (the registered problems are the workloads); HPCPERF_SCALE_MODE=$MODE is refused" >&2; exit 2; }
N_RANKS="$(hpcperf_ranks gizmo no)" || exit 2
INPUT_ID="${HPCPERF_GIZMO_INPUT:-$(python3 "$TOOL" list "$HERE" 2>/dev/null | awk '/\(default\)/{print $1}')}"
[ -n "$INPUT_ID" ] || { echo "run.sh: no input id (HPCPERF_GIZMO_INPUT) and no default input in inputs.yaml" >&2; exit 2; }
param() { python3 "$TOOL" param "$HERE" "$INPUT_ID" "$1"; }
CFG_REL="$(param config)" || exit 2
PARAMS_REL="$(param params)" || exit 2
IC_REL="$(param ic)" || exit 2               # deps/ics/<name>_ics.hdf5 of the materialized bundle
TIME_MAX="$(param time_max)" || exit 2
PARTICLES="$(param particles)" || exit 2
CFG_ID="$(basename "$(dirname "$CFG_REL")")"      # the executable is per Config.sh: inputs/<config>/Config.sh -> GIZMO-<config>
EXE="$L3_INSTALL/bin/GIZMO-$CFG_ID"
[ -x "$EXE" ] || { echo "run.sh: $EXE not found -- run ./build.sh $BACKEND $INPUT_ID (profile $PROFILE) first" >&2; exit 1; }
CFG_SHA="$(l3_sha_file "$HERE/$CFG_REL")"
BUILT_SHA="$(cat "$EXE.config.sha256" 2>/dev/null || echo MISSING)"
[ "$CFG_SHA" = "$BUILT_SHA" ] || { echo "run.sh: $EXE was built from a Config.sh with sha256 $BUILT_SHA, but $CFG_REL now hashes to $CFG_SHA -- rebuild (./build.sh $BACKEND $INPUT_ID)" >&2; exit 1; }
IC="$HERE/$IC_REL"
[ -f "$IC" ] || { echo "run.sh: initial conditions $IC missing (part of the frozen bundle: tools/prepare_benchmark.sh level3 gizmo)" >&2; exit 3; }
[ -f "$HERE/$PARAMS_REL" ] || { echo "run.sh: parameter file $PARAMS_REL missing" >&2; exit 3; }

RUN_DIR="$(l3_rundir "$BUILD_DIR/$L3_RUN_SUBDIR/$INPUT_ID.np$N_RANKS")" || exit 2
mkdir -p "$RUN_DIR/output"
PARAMS="$RUN_DIR/$INPUT_ID.params"
# the registered parameter file with the two run-location lines rewritten, both RELATIVE to the run directory as
# upstream's parameter files have them (GIZMO composes output names from OutputDir in a fixed-size buffer: an
# absolute path of ~100 characters produced clipped names): InitCondFile = `ic` -> the symlink ic.hdf5 to the
# frozen initial conditions (GIZMO appends .hdf5), OutputDir = output/ of the run
ln -sfn "$IC" "$RUN_DIR/ic.hdf5"
awk '$1=="InitCondFile" {print "InitCondFile", "ic"; next}
     $1=="OutputDir"    {print "OutputDir", "output"; next}
     {print}' "$HERE/$PARAMS_REL" > "$PARAMS"
grep -q "^InitCondFile ic$" "$PARAMS" && grep -q "^OutputDir output$" "$PARAMS" || { echo "run.sh: could not rewrite InitCondFile/OutputDir in $PARAMS_REL" >&2; exit 2; }
echo "# GIZMO $BACKEND profile=$PROFILE: input=$INPUT_ID config=$CFG_REL params=$PARAMS_REL ic=$IC_REL particles=$PARTICLES time_max=$TIME_MAX ranks=$N_RANKS run_dir=$RUN_DIR"
cd "$RUN_DIR"
RUN_ID="$(l3_run_id)"
rc=0
"$L3_LAUNCHER" --gpus "$N_RANKS" --bind wrapper -- "$EXE" "$INPUT_ID.params" || rc=$?
if [ -z "${HPCPERF_DRY_RUN:-}" ]; then
    l3_manifest "$RUN_DIR" "run_id=$RUN_ID" "app=gizmo" "backend=$BACKEND" "profile=$PROFILE" "mode=$MODE" \
        "input_id=$INPUT_ID" "config=$CFG_REL" "config_sha256=$CFG_SHA" "params=$PARAMS_REL" "params_sha256=$(l3_sha_file "$HERE/$PARAMS_REL")" \
        "input=$PARAMS" "input_sha256=$(l3_sha_file "$PARAMS")" "ic=$IC_REL" "ic_sha256=$(l3_sha_file "$IC")" \
        "time_max=$TIME_MAX" "particles=$PARTICLES" "ranks=$N_RANKS" "exit_code=$rc" "binary=$EXE" "binary_sha256=$(l3_sha_file "$EXE")" \
        "cpu_txt=$RUN_DIR/output/cpu.txt" "cpu_txt_sha256=$(l3_sha_file "$RUN_DIR/output/cpu.txt")" \
        "fingerprint=$L3_INSTALL/.hpcperf-l3-fingerprint" "fingerprint_sha256=$(l3_sha_file "$L3_INSTALL/.hpcperf-l3-fingerprint")" \
        "utc=$(date -u +%FT%TZ)"
fi
exit "$rc"
