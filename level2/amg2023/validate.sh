#!/usr/bin/env bash
# Validate the AMG2023 CUDA or HIP build against upstream's documented
# reference results.
#
#   ./validate.sh [CUDA|HIP]
#
# AMG2023 has no built-in verification, but amg-doc (section K, "Suggested Test
# Runs") tabulates reference runs on 1 NVIDIA V100 with hypre configured
# --with-cuda, sweeping the local grid size n x n x n on a single rank and
# listing the iteration count for each n. This script reproduces two points of
# that table and checks three things per run:
#
#   1. the solver converged: "Final Relative Residual Norm" is below the
#      stopping tolerance amg.c uses for that problem
#      (tol = 1e-12 for Problem 1, tol = 1e-8 for Problem 2; amg.c:122,457),
#   2. "Iterations" matches the amg-doc reference count within +/-2, which
#      shows the AMG hierarchy (coarsening/interpolation) is being built
#      correctly and not just that the Krylov method ground its way down,
#   3. the run really executed on the GPU: hypre's HYPRE_PrintDeviceInfo()
#      line ("Running on \"<device>\", Comp. Capability: ...") is compiled in
#      only when hypre is built with GPU support, and amg.c prints it right
#      after HYPRE_DeviceInitialize(); the default memory location and
#      execution policy are HYPRE_MEMORY_DEVICE / HYPRE_EXEC_DEVICE
#      (amg.c:154-155), i.e. setup and solve run on the device.
#
# Reference points used (amg-doc, "Results for Problem 1" / "Problem 2"):
#   Problem 1 (27pt, AMG-GMRES, tol 1e-12), n=100x100x100 -> 19 iterations
#             (the table gives 19 for every n from 50 to 200)
#   Problem 2 ( 7pt, AMG-PCG,   tol 1e-8 ), n=100x100x100 -> 30 iterations
#             (the table rises slowly from 30 at n=80 to 35 at n=320)
# Both use ~2.4 GB of device memory and a few seconds of wall clock here.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

set -uo pipefail

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
LOG_DIR="$R/build/level2/amg2023/$MODEL/run"
mkdir -p "$LOG_DIR"

N=100
ITER_TOL=2
fail=0

check_problem() {
    local problem="$1" ref_iters="$2" res_tol="$3"
    local log="$LOG_DIR/validate-problem${problem}.log"

    echo "=== Problem $problem: n = ${N}x${N}x${N}, 1 rank (reference: ${ref_iters} iterations, residual < ${res_tol})"
    HPCPERF_AMG_N="$N" HPCPERF_AMG_PROBLEM="$problem" \
        "$HERE/run.sh" "$BACKEND" > "$log" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "  run failed (exit $rc); last lines of $log:"
        tail -5 "$log" | sed 's/^/    /'
        fail=1
        return
    fi

    local iters res device
    iters="$(awk '/^Iterations = /{print $3}' "$log" | tail -1)"
    res="$(awk '/^Final Relative Residual Norm = /{print $6}' "$log" | tail -1)"
    device="$(grep -m1 '^Running on ' "$log" || true)"

    echo "  Iterations = ${iters:-<missing>}   Final Relative Residual Norm = ${res:-<missing>}"
    echo "  ${device:-<no hypre device line -- not a GPU build?>}"

    if [ -z "$iters" ] || [ -z "$res" ]; then
        echo "  -> FAIL: solver did not report iterations / residual"
        fail=1
        return
    fi
    if [ -z "$device" ]; then
        echo "  -> FAIL: hypre printed no device info; the run was not on a GPU"
        fail=1
        return
    fi
    if ! awk -v r="$res" -v t="$res_tol" 'BEGIN{exit !(r+0 < t+0)}'; then
        echo "  -> FAIL: residual $res is not below the solver tolerance $res_tol"
        fail=1
        return
    fi
    if ! awk -v i="$iters" -v r="$ref_iters" -v d="$ITER_TOL" \
             'BEGIN{exit !(i+0 >= r-d && i+0 <= r+d)}'; then
        echo "  -> FAIL: $iters iterations outside reference ${ref_iters} +/- ${ITER_TOL}"
        fail=1
        return
    fi
    echo "  -> ok (converged below $res_tol in $iters iterations, reference $ref_iters)"
}

check_problem 1 19 1e-12
check_problem 2 30 1e-8

echo "----------------------------------------------------------------"
if [ "$fail" -eq 0 ]; then
    echo "PASS: amg2023 $BACKEND (Problem 1 and 2 at n=${N}: converged below tolerance, iteration counts match the amg-doc single-GPU reference)"
    exit 0
else
    echo "FAIL: amg2023 $BACKEND (see $LOG_DIR/validate-problem*.log)"
    exit 1
fi
