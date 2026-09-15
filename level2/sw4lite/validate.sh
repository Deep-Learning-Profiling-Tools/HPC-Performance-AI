#!/usr/bin/env bash
# Compare the analytic point-source norms with the upstream reference.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null
set -uo pipefail

BACKEND="${1:-CUDA}"; BACKEND="${BACKEND^^}"; MODEL="${BACKEND,,}"
BUILD="$R/build/level2/sw4lite/$MODEL"; LOG="$BUILD/validate.log"
"$HERE/run.sh" "$BACKEND" >"$LOG" 2>&1
rc=$?
grep -E 'Running on [0-9]+ MPI tasks|Cuda devices found|Errors at time|hpcperf-launch: audit summary' "$LOG" | tail -8 || true
python - "$LOG" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"Errors at time\s+\S+\s+Linf\s*=\s*([0-9.eE+-]+)\s+L2\s*=\s*([0-9.eE+-]+)\s+norm of solution\s*=\s*([0-9.eE+-]+)", text)
if not m:
    raise SystemExit(1)
got = tuple(map(float, m.groups()))
ref = (0.569416, 0.0245361, 3.7439)
tol = (5e-6, 5e-7, 5e-5)
raise SystemExit(0 if all(abs(a-b) <= t for a, b, t in zip(got, ref, tol)) else 1)
PY
numeric_rc=$?
if [ "$rc" -eq 0 ] && [ "$numeric_rc" -eq 0 ] && grep -Eq '[1-9][0-9]* Cuda devices found' "$LOG"; then
    echo "SW4lite $BACKEND validation: PASS"
    exit 0
fi
echo "SW4lite $BACKEND validation: FAIL (see $LOG)"
exit 1
