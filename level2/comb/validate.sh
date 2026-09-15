#!/usr/bin/env bash
# Comb initializes halo cells analytically and asserts every received value.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null
set -uo pipefail

BACKEND="${1:-CUDA}"; BACKEND="${BACKEND^^}"; MODEL="${BACKEND,,}"
BUILD="$R/build/level2/comb/$MODEL"; LOG="$BUILD/validate.log"
[ -x "$BUILD/bin/comb" ] || { echo "Comb $BACKEND validation: FAIL (build first)"; exit 1; }
"$HERE/run.sh" "$BACKEND" >"$LOG" 2>&1
rc=$?
grep -E 'Starting test Comm mpi Mesh|test (pre|post)-comm|hpcperf-launch: audit summary' "$LOG" | tail -8 || true
if [ "$rc" -eq 0 ] && grep -q "Starting test Comm mpi Mesh $MODEL" "$LOG" && ! grep -qE 'Assertion|test (pre|post)-comm' "$LOG"; then
    echo "Comb $BACKEND validation: PASS"
    exit 0
fi
echo "Comb $BACKEND validation: FAIL (see $LOG)"
exit 1
