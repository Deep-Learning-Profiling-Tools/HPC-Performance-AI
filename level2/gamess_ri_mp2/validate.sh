#!/usr/bin/env bash
# Enforce the upstream relative-energy tolerance and native BLAS backend.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null
set -uo pipefail

BACKEND="${1:-CUDA}"; BACKEND="${BACKEND^^}"; MODEL="${BACKEND,,}"
BUILD="$R/build/level2/gamess_ri_mp2/$MODEL"; LOG="$BUILD/validate.log"
"$HERE/run.sh" "$BACKEND" >"$LOG" 2>&1
rc=$?
grep -E 'running the code with|Number of MPI ranks|Rel. error|Wall time \(maximum\)|Passed|Failed|hpcperf-launch: audit summary' "$LOG" | tail -12 || true
if [ "$BACKEND" = CUDA ]; then marker='cublas on GPU'; else marker='HIPBLAS on GPU'; fi
error="$(sed -n 's/.*Rel\. error of computed MP2 corr\. energy =  *//p' "$LOG" | tail -1)"
python - "$error" <<'PY'
import sys
try:
    value = float(sys.argv[1])
except (ValueError, IndexError):
    raise SystemExit(1)
raise SystemExit(0 if value <= 1.0e-6 else 1)
PY
numeric_rc=$?
if [ "$rc" -eq 0 ] && [ "$numeric_rc" -eq 0 ] && grep -q "$marker" "$LOG" && grep -q 'Passed :-)' "$LOG"; then
    echo "GAMESS RI-MP2 $BACKEND validation: PASS"
    exit 0
fi
echo "GAMESS RI-MP2 $BACKEND validation: FAIL (see $LOG)"
exit 1
