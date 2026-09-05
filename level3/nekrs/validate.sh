#!/usr/bin/env bash
# Correctness check for nekRS: upstream's own CI test of the ethier case
# (analytic Ethier-Steinman solution) run on N GPUs.
#
#   ./validate.sh [CUDA|HIP]        HPCPERF_GPUS=N (default 1)
#
# `nekrs --cimode 2` is one of the modes upstream's CI (.github/workflows/
# ci.yml) runs on this case: it fixes the solver settings (velocity solver
# +BLOCK, subcycling 1, tolerances 1e-12/1e-10) and, at the last step, checks
# the L2 errors of velocity, pressure and both scalars against the exact
# solution (reference values in examples/ethier/ci.inc: 2.77e-10, 7.14e-10,
# 7.49e-12, 7.22e-12; relative tolerance EPS = 0.3) plus the iteration counts
# of the pressure/velocity/scalar solves (+-1). nekRS prints "CI test <...>
# passed|failed" for each check and exits non-zero on any failure -- that
# verdict is used unchanged. The L2-error line the case prints itself
# ("... L2 err") is echoed for the record.
# Note: upstream runs this CI on CPUs with 2 ranks; here the GPU backend on N
# ranks is being validated against the same criteria.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
N="${HPCPERF_GPUS:-1}"
OUT="$R/build/level3/nekrs/$MODEL/run/smoke.np$N.cimode2.log"

export HPCPERF_GPUS="$N"
mkdir -p "$(dirname "$OUT")"
echo "validate.sh: nekRS $BACKEND ethier --cimode 2 (upstream CI mode) on $N GPU(s)"
set +e
HPCPERF_SCALE_MODE=smoke "$HERE/run.sh" "$BACKEND" --cimode 2 > "$OUT" 2>&1
rc=$?
set -e
grep -a -E '^#|hpcperf-launch: audit summary|CI test|L2 err|elapsedStepSum|total elapsed|ERROR|error' "$OUT" | grep -a -v 'no error' | sed 's/^/  /' | tail -40
FAILED=$(grep -a -c 'CI test .* failed' "$OUT" || true)
PASSED=$(grep -a -c 'CI test .* passed' "$OUT" || true)
echo "  nekrs exit code $rc; CI checks passed=$PASSED failed=$FAILED"
if [ "$rc" -eq 0 ] && [ "$FAILED" -eq 0 ] && [ "$PASSED" -gt 0 ]; then
    echo "nekRS $BACKEND validation ($N GPU, ethier --cimode 2 vs upstream CI references): PASS"; exit 0
fi
echo "nekRS $BACKEND validation ($N GPU, ethier --cimode 2 vs upstream CI references): FAIL (log: $OUT)"; exit 1
