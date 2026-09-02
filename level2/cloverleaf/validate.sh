#!/usr/bin/env bash
# Validate CloverLeaf against the upstream built-in reference solutions.
#
#   ./validate.sh [CUDA|HIP]
#
# Runs two standard decks that carry a `test_problem` id and lets the upstream
# driver (driver/report.cpp) compare the final total kinetic energy against its
# hard-coded reference value (tolerance: within 0.001 %):
#   InputDecks/clover_bm_short.in    test_problem 2  (960 x 960,   87 steps, KE ref 1.19316898756307)
#   InputDecks/clover_bm16_short.in  test_problem 4  (3840 x 3840, 87 steps, KE ref 0.307475452287895)
# Each run must exit 0 and print "This test is considered PASSED".
# Prints PASS/FAIL at the end; exit code 0/1.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null
set -uo pipefail

BACKEND="${1:-CUDA}"
BACKEND="${BACKEND^^}"
MODEL="${BACKEND,,}"

BUILD="$R/build/level2/cloverleaf/$MODEL"
EXE="$BUILD/$MODEL-cloverleaf"
if [ ! -x "$EXE" ]; then
  echo "validate.sh: $EXE not found -- run $HERE/build.sh $BACKEND first" >&2
  echo "CloverLeaf $BACKEND validation: FAIL"
  exit 1
fi

DECKS=(clover_bm_short.in clover_bm16_short.in)
fail=0
for deck in "${DECKS[@]}"; do
  name="${deck%.in}"
  log="$BUILD/validate_${name}.log"
  echo "== $deck"
  mpirun -np 1 "$EXE" --file "$HERE/InputDecks/$deck" --out "$BUILD/validate_${name}.out" >"$log" 2>&1
  rc=$?
  grep -E "Test problem [0-9]+ is within|This test is considered|^ - (Problem|Outcome)|Wall clock" "$log" | tail -5
  if [ "$rc" -eq 0 ] && grep -q "This test is considered PASSED" "$log"; then
    echo "   -> $deck: PASSED (exit $rc)"
  else
    echo "   -> $deck: FAILED (exit $rc); see $log"
    fail=1
  fi
done

if [ "$fail" -eq 0 ]; then
  echo "CloverLeaf $BACKEND validation: PASS"
  exit 0
else
  echo "CloverLeaf $BACKEND validation: FAIL"
  exit 1
fi
