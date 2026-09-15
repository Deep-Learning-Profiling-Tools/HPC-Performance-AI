#!/usr/bin/env bash
# Require every upstream CORAL-2 check and evidence of native GPU kernels.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null
set -uo pipefail

BACKEND="${1:-CUDA}"; BACKEND="${BACKEND^^}"; MODEL="${BACKEND,,}"
BUILD="$R/build/level2/quicksilver/$MODEL"; LOG="$BUILD/validate.log"
[ -x "$BUILD/qs" ] || { echo "Quicksilver $BACKEND validation: FAIL (build first)"; exit 1; }
"$HERE/run.sh" "$BACKEND" >"$LOG" 2>&1
rc=$?
grep -E '^(PASS|FAIL)::|cycleTracking_Kernel|hpcperf-launch: audit summary' "$LOG" | tail -10 || true
passes="$(grep -c '^PASS::' "$LOG" || true)"
if [ "$rc" -eq 0 ] && [ "$passes" -eq 4 ] && ! grep -q '^FAIL::' "$LOG" && grep -q '^cycleTracking_Kernel' "$LOG"; then
    echo "Quicksilver $BACKEND validation: PASS"
    exit 0
fi
echo "Quicksilver $BACKEND validation: FAIL (see $LOG; expected four CORAL PASS lines)"
exit 1
