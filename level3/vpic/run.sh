#!/usr/bin/env bash
# Run one registered VPIC input deck (a compiled deck executable, build.sh) on one GPU.
#
#   ./run.sh [CUDA|HIP]
#
# Execution model: one MPI rank per GPU; every registered deck fixes its domain topology to one rank
# (the decks are compile-time inputs: topology, grid, particles and step count are constants of the
# deck source, decks/<id>.cxx), so run.sh refuses HPCPERF_GPUS other than 1 -- a rank count the deck
# cannot partition is never silently changed. The rank goes through the common launcher with the
# per-rank GPU wrapper (the process sees exactly one GPU; mapping audited).
#
# Controls:
#   HPCPERF_VPIC_INPUT    a registered input id (inputs.yaml; default: the registry's default_input)
#   HPCPERF_GPUS=1        ranks = GPUs (only 1 is accepted, see above)
#   HPCPERF_SCALE_MODE    smoke only (the registered decks are the workloads; no strong/weak derivation)
# The deck executable runs in its own run directory ($L3_BUILD/$L3_RUN_SUBDIR/<id>.np1/): VPIC writes its
# energies diagnostic ("energies", one row per energies_interval steps: field and per-species energies)
# there; the registered decks disable every other dump. The run manifest records the deck source hash
# the executable was compiled from (build.sh) and refuses a stale executable.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"; [ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
PROFILE="$(l3_backend_profile VPIC "$MODEL")"
l3_paths_profile vpic "$PROFILE" "$MODEL" || exit 2     # same derivation as build.sh: binaries, install and run tree of ONE profile
BUILD_DIR="$L3_BUILD"
l3_fingerprint_expect_backend "$L3_INSTALL" "$MODEL" || exit 1
l3_require_materialized "$HERE" || exit 3
TOOL="$R/tools/inputs/hpcperf_inputs.py"

hpcperf_forbid_args vpic --restore --modify --tpp -- "$@" || exit 2
MODE="$(l3_scale_mode vpic)" || exit 2
[ "$MODE" = smoke ] || { echo "run.sh: VPIC has no strong/weak derivation (the registered decks are the workloads); HPCPERF_SCALE_MODE=$MODE is refused" >&2; exit 2; }
N_RANKS="$(hpcperf_ranks vpic no)" || exit 2
[ "$N_RANKS" -eq 1 ] || { echo "run.sh: every registered VPIC deck fixes a 1 x 1 x 1 rank topology; HPCPERF_GPUS=$N_RANKS cannot partition it (refused, not adjusted)" >&2; exit 2; }
INPUT_ID="${HPCPERF_VPIC_INPUT:-$(python3 "$TOOL" list "$HERE" 2>/dev/null | awk '/\(default\)/{print $1}')}"
[ -n "$INPUT_ID" ] || { echo "run.sh: no input id (HPCPERF_VPIC_INPUT) and no default input in inputs.yaml" >&2; exit 2; }
param() { python3 "$TOOL" param "$HERE" "$INPUT_ID" "$1"; }
DECK_REL="$(param deck)" || exit 2
DECK="$HERE/$DECK_REL"
STEPS="$(param num_step)" || exit 2
GRID="$(param grid)" || exit 2
PARTICLES="$(param particles)" || exit 2
EFILE="$(param energies_file)" || exit 2     # the deck's energies diagnostic, relative to the run directory
EXE="$BUILD_DIR/decks/$INPUT_ID.Linux"
[ -x "$EXE" ] || { echo "run.sh: $EXE not found -- run ./build.sh $BACKEND (profile $PROFILE) first" >&2; exit 1; }
DECK_SHA="$(l3_sha_file "$DECK")"
BUILT_SHA="$(cat "$BUILD_DIR/decks/$INPUT_ID.deck.sha256" 2>/dev/null || echo MISSING)"
[ "$DECK_SHA" = "$BUILT_SHA" ] || { echo "run.sh: $EXE was compiled from a deck with sha256 $BUILT_SHA, but $DECK_REL now hashes to $DECK_SHA -- rebuild (./build.sh $BACKEND)" >&2; exit 1; }

RUN_DIR="$(l3_rundir "$BUILD_DIR/$L3_RUN_SUBDIR/$INPUT_ID.np$N_RANKS")" || exit 2
echo "# VPIC $BACKEND profile=$PROFILE: input=$INPUT_ID deck=$DECK_REL ranks=$N_RANKS grid=$GRID particles=$PARTICLES num_step=$STEPS run_dir=$RUN_DIR"
cd "$RUN_DIR"
RUN_ID="$(l3_run_id)"
rc=0
"$L3_LAUNCHER" --gpus "$N_RANKS" --bind wrapper -- "$EXE" || rc=$?
if [ -z "${HPCPERF_DRY_RUN:-}" ]; then
    l3_manifest "$RUN_DIR" "run_id=$RUN_ID" "app=vpic" "backend=$BACKEND" "profile=$PROFILE" "mode=$MODE" \
        "input_id=$INPUT_ID" "deck=$DECK_REL" "deck_sha256=$DECK_SHA" "num_step=$STEPS" "grid=$GRID" "particles=$PARTICLES" \
        "ranks=$N_RANKS" "exit_code=$rc" "binary=$EXE" "binary_sha256=$(l3_sha_file "$EXE")" \
        "energies=$RUN_DIR/$EFILE" "energies_sha256=$(l3_sha_file "$RUN_DIR/$EFILE")" \
        "fingerprint=$L3_INSTALL/.hpcperf-l3-fingerprint" "fingerprint_sha256=$(l3_sha_file "$L3_INSTALL/.hpcperf-l3-fingerprint")" \
        "utc=$(date -u +%FT%TZ)"
fi
exit "$rc"
