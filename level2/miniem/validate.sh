#!/usr/bin/env bash
# Validate the MiniEM CUDA build with upstream's own analytic-solution case.
#
#   ./validate.sh [CUDA]
#
# Criterion source: Trilinos ships `maxwell-analyticSolution.xml` (a Maxwell problem with an analytic forcing
# and the exact E field as a closure model) and the driver itself asserts, for that deck, that the L2 error of
# the computed E field against the exact field is below 0.065 (main.cpp, TEUCHOS_ASSERT_INEQUALITY on
# "L2 Error E maxwell - analyticSolution"); upstream's test `Maxwell_MueLu_order1_analytic` runs exactly this.
# This script reproduces that run and checks three things:
#
#   1. the driver completes (exit 0) -- which already includes upstream's 0.065 assertion,
#   2. the printed "L2 Error E maxwell - analyticSolution = <value>" is parsed here and re-checked against the
#      same 0.065 bound (so the criterion is visible in the log, not only implied by the exit code),
#   3. with HPCPERF_GPUS=N > 1, the same case at N ranks reproduces the 1-rank L2 error to a relative 1e-4
#      (the finite-element solution does not depend on the decomposition beyond solver tolerance and
#      floating-point reduction order): rank-count consistency, the check every Level 2 MPI app carries.
#
# GPU execution is recorded through the launcher's binding audit line ("audit summary: N verified, 0 mismatch");
# a run that ends before nvidia-smi samples it is "unverified" (observation gap), never a mismatch.
# The analytic case is 15^3 hex elements; its deck sets `final time` 5e-9, which the driver reaches in 6 implicit
# steps whatever --numTimeSteps says. Mini-EM's own timer reports about one second; the wall clock is dominated by
# process start-up (a ~480 MB statically linked binary read from the network file system, CUDA/Kokkos init).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

set -uo pipefail

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
LOG_DIR="$R/build/level2/miniem/$MODEL/run"
mkdir -p "$LOG_DIR"
BOUND=0.065           # upstream's assertion in main.cpp
REL_TOL=1e-4          # rank-count consistency of the L2 error
fail=0

run_case() { # <ranks> -> prints the L2 error, logs to $LOG_DIR/validate-np<ranks>.log
    local ranks="$1"
    local log="$LOG_DIR/validate-np${ranks}.log"
    HPCPERF_GPUS="$ranks" HPCPERF_SCALE_MODE=smoke HPCPERF_MINIEM_DECK=maxwell-analyticSolution.xml \
        "$HERE/run.sh" "$BACKEND" > "$log" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "  run failed at $ranks rank(s) (exit $rc); last lines of $log:"; tail -6 "$log" | sed 's/^/    /'
        return 1
    fi
    local err
    err="$(grep -E '^L2 Error E maxwell - analyticSolution = ' "$log" | tail -1 | awk '{print $NF}')"
    if [ -z "$err" ]; then
        echo "  no 'L2 Error E maxwell - analyticSolution' line in $log"; return 1
    fi
    echo "$err"
}

echo "=== MiniEM $BACKEND: analytic Maxwell case (maxwell-analyticSolution.xml, 15^3 hex), 1 rank"
E1="$(run_case 1)" || { fail=1; E1=""; }
if [ -n "$E1" ]; then
    if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) < float(sys.argv[2]) else 1)" "$E1" "$BOUND"; then
        echo "  L2 error E = $E1 < $BOUND (upstream bound): PASS"
    else
        echo "  L2 error E = $E1 >= $BOUND (upstream bound): FAIL"; fail=1
    fi
    audit="$(grep -o 'audit summary: .*' "$LOG_DIR/validate-np1.log" | tail -1)"
    echo "  GPU binding audit: ${audit:-not reported}"
fi

N="${HPCPERF_GPUS:-1}"
if [ "$N" != 1 ] && [ -n "$E1" ]; then
    echo "=== rank-count consistency: the same case at $N ranks"
    EN="$(run_case "$N")" || fail=1
    if [ -n "${EN:-}" ]; then
        if python3 -c "import sys; a,b,t=map(float,sys.argv[1:4]); sys.exit(0 if abs(a-b) <= t*abs(a) else 1)" "$E1" "$EN" "$REL_TOL"; then
            echo "  L2 error E at $N ranks = $EN vs 1 rank = $E1 (rel diff <= $REL_TOL): PASS"
        else
            echo "  L2 error E at $N ranks = $EN vs 1 rank = $E1: FAIL (rel diff > $REL_TOL)"; fail=1
        fi
        audit="$(grep -o 'audit summary: .*' "$LOG_DIR/validate-np${N}.log" | tail -1)"
        echo "  GPU binding audit: ${audit:-not reported}"
    fi
fi

if [ "$fail" -eq 0 ]; then
    echo "MiniEM $BACKEND validation ($N GPU${N:+s}, analytic Maxwell, L2 error < $BOUND): PASS"
    exit 0
else
    echo "MiniEM $BACKEND validation: FAIL"
    exit 1
fi
