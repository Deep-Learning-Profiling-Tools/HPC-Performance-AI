#!/usr/bin/env bash
# Correctness check for nekRS: upstream's own CI test of the ethier case
# (analytic Ethier-Steinman solution) run on N GPUs.
#
#   ./validate.sh [CUDA|HIP]        HPCPERF_GPUS=N (default 1)
#     HPCPERF_NEKRS_CIMODE=C        upstream CI mode (default 2)
#
# `nekrs --cimode C` fixes the solver settings for CI and, at the last step,
# checks the L2 errors of velocity/pressure/scalars against the exact solution
# (references in examples/ethier/ci.inc, EPS = 0.3) plus solver iteration
# counts. nekRS prints "CI test <name> passed|failed" per check and exits
# non-zero on any failure. This validator requires ALL of:
#   * run exited 0 (no timeout, no abort);
#   * the COMPLETE expected set of CI checks was produced (count matches the
#     per-cimode table below) -- not merely "some passed";
#   * zero failed checks;
#   * the coarse-solver LOCATION recorded in the log matches what the cimode
#     selects (CPU for mode 2; DEVICE for mode 3 -- i.e. the run must really
#     have exercised GPU HYPRE, not silently fallen back).
# The coarse-solver location/precision are extracted and printed for the record.
#
# IMPORTANT SCOPE NOTE: cimode 2 runs the HYPRE BoomerAMG coarse solve on the
# CPU (nekRS default). A PASS here validates the CUDA main application + a
# CPU coarse solve; it does NOT validate GPU HYPRE. Use HPCPERF_NEKRS_CIMODE=3
# (DEVICE coarse) to exercise the GPU HYPRE coarse solve.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
N="${HPCPERF_GPUS:-1}"
CIMODE="${HPCPERF_NEKRS_CIMODE:-2}"
TIMEOUT="${HPCPERF_VALIDATE_TIMEOUT:-2400}"
VARIANT="${HPCPERF_NEKRS_VARIANT:-$([ "${HPCPERF_NEKRS_HYPRE_GPU:-ON}" = ON ] && echo hypregpu || echo cpucoarse)}"
if [ "$VARIANT" = hypregpu ]; then VBD="$R/build/level3/nekrs/$MODEL"; else VBD="$R/build/level3/nekrs/$VARIANT.$MODEL"; fi
OUT="$VBD/$L3_RUN_SUBDIR/validate.cimode$CIMODE.np$N.log"

# complete CI-check counts and required coarse-solver location, per cimode
case "$CIMODE" in
    2) EXPECT_CHECKS=9; WANT_COARSE=CPU ;;
    3) EXPECT_CHECKS=9; WANT_COARSE=DEVICE ;;
    *) echo "validate.sh: expected CI-check count for cimode $CIMODE is not recorded here; add it after observing one run (refusing to guess)" >&2; exit 2 ;;
esac

export HPCPERF_GPUS="$N"
mkdir -p "$(dirname "$OUT")"
echo "validate.sh: nekRS $BACKEND variant=$VARIANT ethier --cimode $CIMODE (upstream CI mode; expect $EXPECT_CHECKS checks, coarse=$WANT_COARSE) on $N GPU(s)"
rc=0
HPCPERF_SCALE_MODE=smoke timeout "$TIMEOUT" "$HERE/run.sh" "$BACKEND" --cimode "$CIMODE" > "$OUT" 2>&1 || rc=$?
grep -aE '^#|hpcperf-launch: audit summary|CI test|L2 err|COARSE SOLVER LOCATION|COARSE SOLVER PRECISION|elapsedStepSum|ERROR|Abort|abort' "$OUT" | grep -aiv 'no error' | sed 's/^/  /' | tail -50
if [ "$rc" -eq 124 ]; then echo "validate.sh: FAIL -- run timed out after ${TIMEOUT}s"; exit 1; fi

COARSE="$(grep -a 'COARSE SOLVER LOCATION' "$OUT" | head -1 | sed 's/.*value: *//' | tr -d ' ' || true)"
PASSED=$(grep -a -c 'CI test .* passed' "$OUT" || true)
FAILED=$(grep -a -c 'CI test .* failed' "$OUT" || true)
TOTAL=$((PASSED + FAILED))
echo "  nekrs exit code $rc; CI checks passed=$PASSED failed=$FAILED total=$TOTAL (expected $EXPECT_CHECKS); coarse solver location=${COARSE:-UNKNOWN}"

ok=1
[ "$rc" -eq 0 ]                    || { echo "  run exited $rc"; ok=0; }
[ "$FAILED" -eq 0 ]                || { echo "  $FAILED CI check(s) failed"; ok=0; }
[ "$TOTAL" -eq "$EXPECT_CHECKS" ]  || { echo "  produced $TOTAL CI checks, expected the complete set of $EXPECT_CHECKS (incomplete run or changed CI)"; ok=0; }
[ "${COARSE:-}" = "$WANT_COARSE" ] || { echo "  coarse solver location is '${COARSE:-UNKNOWN}', expected '$WANT_COARSE' -- the intended solve path did not run (no silent fallback allowed)"; ok=0; }
if [ "$ok" -eq 1 ]; then
    echo "nekRS $BACKEND validation ($N GPU, ethier --cimode $CIMODE, $EXPECT_CHECKS/$EXPECT_CHECKS checks, coarse=$COARSE): PASS"; exit 0
fi
echo "nekRS $BACKEND validation ($N GPU, ethier --cimode $CIMODE): FAIL (log: $OUT)"; exit 1
