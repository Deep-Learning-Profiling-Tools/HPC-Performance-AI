#!/usr/bin/env bash
# Validate the TeaLeaf CUDA or HIP build against upstream's reference results.
#
#   ./validate.sh [CUDA|HIP]
#
# Runs the upstream default deck tea.in (512x512 cells, 20 timesteps, CG,
# eps 1e-15). At the end of the run TeaLeaf's built-in check
# (driver/field_summary_driver.cpp, `check_result` is on by default) looks up
# the (x_cells, y_cells, end_step) entry in tea.problems -- here
# "512 512 20 1.034697091898282e+02" -- and compares the final global
# temperature sum against it with a 0.001 % relative tolerance, printing
# " This run PASSED/FAILED" and returning exit status 0/1. This script
# forwards that verdict as a final PASS/FAIL line and exit code.
#
# Environment overrides:
#   HPCPERF_TEALEAF_DECK  a different deck that has a tea.problems entry
#                         (e.g. Benchmarks/tea_bm_4.in, Benchmarks/tea_bm_5.in)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"

export HPCPERF_TEALEAF_DECK="${HPCPERF_TEALEAF_DECK:-tea.in}"

LOG_DIR="$R/build/level2/tealeaf/$MODEL/run"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/validate.log"

"$HERE/run.sh" "$BACKEND" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

echo "----------------------------------------------------------------"
grep -E '^ (Expected|Actual) ' "$LOG" || true
if [ "$rc" -eq 0 ] && grep -q '^ This run PASSED' "$LOG" && grep -q 'Outcome: PASSED' "$LOG"; then
    echo "TeaLeaf $BACKEND validation ($HPCPERF_TEALEAF_DECK): PASS"
    exit 0
else
    echo "TeaLeaf $BACKEND validation ($HPCPERF_TEALEAF_DECK): FAIL (exit code $rc)"
    exit 1
fi
