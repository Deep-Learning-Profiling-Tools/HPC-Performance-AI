#!/usr/bin/env bash
# Validate the hipBone GPU build.
#
#   ./validate.sh [CUDA|HIP]        (default: CUDA)
#
# Upstream ships no reference solution; its "Verifying correctness" section
# says to run with -v and check that the CG residual norm after the fixed 100
# iterations is small (CORAL-2 Nekbone acceptance rule: "generally less than
# 1e-8"). This script runs the validation problem
#
#     -nx 3 -ny 3 -nz 3 -p 14        (27 elements, degree 14 = the CORAL-2
#                                     polynomial order, 68,921 DOFs)
#
# once on the GPU backend and once with OCCA's Serial (CPU) backend as the
# reference, both with -v, and checks:
#   1. both runs exit 0 and report exactly 100 CG iterations (hipBone runs
#      CG with tol = 0 for a fixed 100 iterations; fewer would mean a
#      NaN/zero residual, more is impossible);
#   2. the GPU final residual ||r_100|| is finite and < 1e-8 (CORAL-2 rule);
#   3. the GPU relative reduction ||r_100|| / ||r_0|| is <= 1e-9;
#   4. GPU vs Serial: initial residual norms agree to 1% and the final
#      residual norms agree to within a factor of 10 (|log10 ratio| <= 1).
# Check 4 is deliberately loose: hipBone's right-hand side is the
# pseudo-random sequence rhs[n] = sin(1e9*cos(1e9*n*n)), whose values depend
# on the last bits of the transcendental functions and therefore differ
# between the CPU libm and the CUDA math library (both backends see a
# different random realisation of the same statistical RHS, so ||r_0|| and
# ||r_100|| agree statistically, not bit-wise). A broken operator or
# gather-scatter shows up as a stagnating/NaN residual (checks 1-3), not as
# a small shift.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

BACKEND="$(printf '%s' "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
case "$BACKEND" in
    CUDA|HIP) ;;
    *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac

if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$R" ] && [ -f "$R/hpcperf_env.sh" ]; then
    set +u
    # shellcheck disable=SC1091
    source "$R/hpcperf_env.sh" 2>/dev/null
    set -u
fi

LC="$(printf '%s' "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD="$R/build/level2/hipbone/$LC"
BIN="$BUILD/hipBone"
if [ ! -x "$BIN" ]; then
    echo "error: $BIN not found -- run $HERE/build.sh $BACKEND first" >&2
    echo "FAIL: hipbone $BACKEND (not built)"
    exit 1
fi

# Same JIT environment as run.sh (see there).
export OCCA_CACHE_DIR="${OCCA_CACHE_DIR:-$R/build/level2/hipbone/occa-cache}"
export HIPBONE_CACHE_DIR="${HIPBONE_CACHE_DIR:-$OCCA_CACHE_DIR}"
export OCCA_CXX="${OCCA_CXX:-${CXX:-g++}}"
if [ "$BACKEND" = "CUDA" ]; then
    export OCCA_CUDA_COMPILER="${OCCA_CUDA_COMPILER:-nvcc -ccbin ${CUDAHOSTCXX:-${CXX:-g++}}}"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

PROBLEM=(-nx 3 -ny 3 -nz 3 -p 14)
LOGDIR="$R/build/level2/hipbone/validate"
mkdir -p "$LOGDIR"

FAIL=0
fail() { echo "  FAIL: $*"; FAIL=1; }
ok()   { echo "  ok:   $*"; }

# run_mode <mode> -> sets R0_<mode>, R100_<mode>, NITER_<mode>, RC_<mode>
run_mode() {
    local mode="$1" log="$LOGDIR/$1.log" rc
    echo "== mpirun -np 1 ./hipBone -m $mode ${PROBLEM[*]} -v   (log: $log)"
    ( cd "$BUILD" && mpirun -np 1 ./hipBone -m "$mode" "${PROBLEM[@]}" -v ) >"$log" 2>&1
    rc=$?
    local r0 r100 niter summary
    r0="$(sed -n 's/^CG: initial res norm \([0-9.eE+-]*\).*/\1/p' "$log" | head -1)"
    r100="$(sed -n 's/^CG: it 100, r norm \([0-9.eE+-]*\),.*/\1/p' "$log" | head -1)"
    # "hipBone: N, DOFs, elapsed, iterations, ..." (4th field)
    summary="$(grep -E '^hipBone: [0-9]+,' "$log" | head -1)"
    niter="$(printf '%s' "$summary" | awk -F', *' '{print $4}')"
    echo "   exit $rc | $(printf '%s' "$summary" | cut -d';' -f1)"
    echo "   initial res norm = ${r0:-?}   final (it 100) r norm = ${r100:-?}"
    printf -v "RC_$mode"    '%s' "$rc"
    printf -v "R0_$mode"    '%s' "${r0:-nan}"
    printf -v "R100_$mode"  '%s' "${r100:-nan}"
    printf -v "NITER_$mode" '%s' "${niter:--1}"
}

# float helpers: fnum() accepts only a plain finite decimal/e-notation number
# (hipBone prints nan/inf for non-finite residuals; awk would read those as 0),
# the awk comparisons handle e-notation.
fnum() { [[ "$1" =~ ^[+-]?[0-9]+(\.[0-9]*)?([eE][+-]?[0-9]+)?$ ]]; }
flt()  { awk -v a="$1" -v b="$2" 'BEGIN{ exit !((a+0) <  (b+0)) }'; }
fle()  { awk -v a="$1" -v b="$2" 'BEGIN{ exit !((a+0) <= (b+0)) }'; }
fdiv() { awk -v a="$1" -v b="$2" 'BEGIN{ printf "%.6e", (a+0)/(b+0) }'; }
flog10abs() { awk -v a="$1" 'BEGIN{ x=log(a+0)/log(10); if (x<0) x=-x; printf "%.3f", x }'; }

run_mode "$BACKEND"
run_mode Serial

GPU_RC="RC_$BACKEND"; GPU_R0="R0_$BACKEND"; GPU_R100="R100_$BACKEND"; GPU_NIT="NITER_$BACKEND"
GPU_RC="${!GPU_RC}"; GPU_R0="${!GPU_R0}"; GPU_R100="${!GPU_R100}"; GPU_NIT="${!GPU_NIT}"

echo "== checks"
# 1. exit codes and fixed iteration count
[ "$GPU_RC" -eq 0 ]    && ok "$BACKEND run exited 0"           || fail "$BACKEND run exited $GPU_RC"
[ "$RC_Serial" -eq 0 ] && ok "Serial run exited 0"             || fail "Serial run exited $RC_Serial"
[ "$GPU_NIT" = 100 ]   && ok "$BACKEND CG iterations = 100"    || fail "$BACKEND CG iterations = $GPU_NIT (expected 100)"
[ "$NITER_Serial" = 100 ] && ok "Serial CG iterations = 100"   || fail "Serial CG iterations = $NITER_Serial (expected 100)"
# 2. absolute residual (CORAL-2 rule)
if fnum "$GPU_R100" && flt "$GPU_R100" 1e-8; then ok "$BACKEND final r norm $GPU_R100 < 1e-8"
else fail "$BACKEND final r norm $GPU_R100 is not < 1e-8"; fi
if fnum "$R100_Serial" && flt "$R100_Serial" 1e-8; then ok "Serial final r norm $R100_Serial < 1e-8"
else fail "Serial final r norm $R100_Serial is not < 1e-8"; fi
# 3. relative reduction
if fnum "$GPU_R0" && fnum "$GPU_R100"; then
    RED="$(fdiv "$GPU_R100" "$GPU_R0")"
    if fle "$RED" 1e-9; then ok "$BACKEND ||r_100||/||r_0|| = $RED <= 1e-9"
    else fail "$BACKEND ||r_100||/||r_0|| = $RED > 1e-9"; fi
else fail "$BACKEND residuals not finite ($GPU_R0, $GPU_R100)"; fi
# 4. GPU vs Serial agreement
if fnum "$GPU_R0" && fnum "$R0_Serial"; then
    D0="$(awk -v a="$GPU_R0" -v b="$R0_Serial" 'BEGIN{ d=(a-b)/b; if (d<0) d=-d; printf "%.3e", d }')"
    if fle "$D0" 1e-2; then ok "initial res norm $BACKEND=$GPU_R0 vs Serial=$R0_Serial (rel diff $D0 <= 1e-2)"
    else fail "initial res norm $BACKEND=$GPU_R0 vs Serial=$R0_Serial (rel diff $D0 > 1e-2)"; fi
else fail "initial residuals not finite ($GPU_R0, $R0_Serial)"; fi
if fnum "$GPU_R100" && fnum "$R100_Serial"; then
    RATIO="$(fdiv "$GPU_R100" "$R100_Serial")"; L="$(flog10abs "$RATIO")"
    if fle "$L" 1; then ok "final r norm $BACKEND=$GPU_R100 vs Serial=$R100_Serial (ratio $RATIO, |log10| = $L <= 1)"
    else fail "final r norm $BACKEND=$GPU_R100 vs Serial=$R100_Serial (ratio $RATIO, |log10| = $L > 1)"; fi
else fail "final residuals not finite ($GPU_R100, $R100_Serial)"; fi

if [ "$FAIL" -eq 0 ]; then
    echo "PASS: hipbone $BACKEND (-nx 3 -ny 3 -nz 3 -p 14: 100 CG iterations, final r norm $GPU_R100 < 1e-8, Serial reference $R100_Serial)"
    exit 0
else
    echo "FAIL: hipbone $BACKEND (see $LOGDIR/*.log)"
    exit 1
fi
