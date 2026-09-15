#!/usr/bin/env bash
# Correctness check for the ExaCA build: directional-solidification smoke case (128^3 cells, the upstream
# Inp_DirSolidification.json physics with the decomposition-independent substrate) on N GPUs.
#
#   ./validate.sh [CUDA|HIP]        HPCPERF_GPUS=N (default 1) selects the rank count
#
# ExaCA has no shipped reference output; its cell captures use atomic compare-exchange, so the final GrainID
# field is NOT bitwise reproducible (measured: identical 1-GPU runs differ bitwise while every statistic below
# agrees). The criteria are therefore statistical invariants of the final microstructure, computed from the
# GrainID field the run writes (exaca_check.py) with tolerances derived from the measured spread (see README):
#   [1] completeness: run exits 0, the field and the log exist, DIMENSIONS == the deck, every cell solidified
#       (GrainID != 0), the log reports N ranks and ONE global decomposition in Y whose subdomains tile the box
#       exactly once (offset chain with the 1-cell halo overlap, first offset 0, last end == Ny) -- never N
#       independent copies, no missing or duplicated subdomain
#   [2] self-consistency: ExaCA's own VolFractionNucleated equals the value recomputed from the field (1e-3)
#   [3] vs the frozen reference statistics of the validated 1-GPU baseline run
#       (references/dirsolid_smoke.reference.json): grain count, nucleated-grain count and volume fraction,
#       top-layer grain count, mean <001>-to-z misorientation (bulk and top layer), mean grain volume
#   [4] with HPCPERF_GPUS>1: the N-rank statistics vs this build's 1-GPU run (rank-count independence)
# Rules enforced here: the run's real exit code is captured; a nonzero exit, a timeout, a missing field/log
# or a non-finite statistic is a FAIL (never a PASS on a stale result); run.sh recreates its run directory,
# so only THIS run's output is validated.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
N="${HPCPERF_GPUS:-1}"
l3_require_materialized "$HERE" || exit 3
REF="$HERE/references/dirsolid_smoke.reference.json"; TOL="$HERE/references/dirsolid_smoke.tolerances.json"
ORIENT="$HERE/src/examples/Substrate/GrainOrientationVectors.csv"
[ -f "$REF" ] && [ -f "$TOL" ] || { echo "validate.sh: reference statistics/tolerances missing under references/" >&2; exit 1; }
[ -f "$ORIENT" ] || { echo "validate.sh: $ORIENT missing (run tools/prepare_benchmark.sh level3 exaca)" >&2; exit 1; }
PROFILE="$(l3_backend_profile EXACA "$MODEL")"
l3_paths_profile exaca "$PROFILE" "$MODEL" || exit 2     # the run tree of the SAME profile build.sh/run.sh use
RUN_ROOT="$L3_BUILD/$L3_RUN_SUBDIR"
TIMEOUT="${HPCPERF_VALIDATE_TIMEOUT:-900}"
PY="$(l3_python_yaml)"
unset HPCPERF_SCALE_MODE
export HPCPERF_GPUS="$N"
echo "validate.sh: ExaCA $BACKEND smoke (dirsolid 128^3 cells, Inconel625, G=5e5 K/m, R=3e5 K/s) on $N GPU(s) [profile $PROFILE]"

run_once() { # <ngpu> <stdout-file>
    local ng=$1 out=$2 rc=0
    HPCPERF_GPUS="$ng" HPCPERF_SCALE_MODE=smoke timeout "$TIMEOUT" "$HERE/run.sh" "$BACKEND" > "$out" 2>&1 || rc=$?
    return $rc
}
mkdir -p "$RUN_ROOT"
VOUT="$RUN_ROOT/validate.smoke.np$N.stdout"
rc=0; run_once "$N" "$VOUT" || rc=$?
grep -aE '^#|hpcperf-launch: audit summary|ExaCA version|Kokkos version|Number of MPI ranks|Time spent performing CA|Error|error' "$VOUT" | head -12 || true
if [ "$rc" -eq 124 ]; then echo "validate.sh: FAIL -- run timed out after ${TIMEOUT}s"; exit 1; fi
[ "$rc" -eq 0 ] || { echo "validate.sh: FAIL -- run.sh exited $rc (see $VOUT)"; exit 1; }
D="$RUN_ROOT/dirsolid.smoke.np$N"; VTK="$D/dirsolid_smoke_np$N.vtk"; LOG="$D/dirsolid_smoke_np$N.json"
[ -f "$VTK" ] && [ -f "$LOG" ] || { echo "validate.sh: FAIL -- output field/log missing ($VTK, $LOG)"; exit 1; }
if [ "$N" -gt 1 ] && [ ! -f "$RUN_ROOT/dirsolid.smoke.np1/stats.json" ]; then
    echo "validate.sh: producing the 1-GPU run for the rank-count comparison"
    r1=0; run_once 1 "$RUN_ROOT/validate.smoke.np1.stdout" || r1=$?
    [ "$r1" -eq 0 ] || { echo "validate.sh: FAIL -- 1-GPU run failed (rc=$r1)"; exit 1; }
    "$PY" "$HERE/exaca_check.py" stats "$RUN_ROOT/dirsolid.smoke.np1/dirsolid_smoke_np1.vtk" "$RUN_ROOT/dirsolid.smoke.np1/dirsolid_smoke_np1.json" "$ORIENT" --json "$RUN_ROOT/dirsolid.smoke.np1/stats.json" > /dev/null || { echo "validate.sh: FAIL -- 1-GPU statistics"; exit 1; }
fi
"$PY" "$HERE/exaca_check.py" stats "$VTK" "$LOG" "$ORIENT" --json "$D/stats.json" > /dev/null || { echo "validate.sh: FAIL -- statistics of the $N-GPU field could not be computed"; exit 1; }

NP1=(); [ "$N" -gt 1 ] && NP1=(--np1 "$RUN_ROOT/dirsolid.smoke.np1/stats.json")
echo "validate.sh: protocol $(sed -n 's/^protocol_version: *//p' "$HERE/references/validation_protocol.yaml" 2>/dev/null | head -1) run_id=$(sed -n 's/^run_id=//p' "$D/run_manifest.txt" | tail -1)"
"$PY" "$HERE/exaca_check.py" validate "$D/stats.json" --ref "$REF" --tol "$TOL" --ranks "$N" "${NP1[@]}" --expect-dims 128,128,128 --json "$D/validation.json"
