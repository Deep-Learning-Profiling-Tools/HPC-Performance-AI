#!/bin/bash
# Level 3 CPU-only regression tests (no GPU, no application, no build):
#   test_l3_infra.sh -- correctness/reproducibility helpers (l3_common.sh, l3_check.py):
#                       NaN/Inf rejection, real-exit-code capture, the failed-run
#                       gate that prevents a stale-log false PASS, the dry-run
#                       result-directory sentinel, and patch fingerprint / cache
#                       invalidation.
# Usage: level3/tools/tests/run_all.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rc=0
run() { echo "=== $1"; shift; bash "$@" || { echo "=== FAILED: $*"; rc=1; }; echo; }
run "l3 infra (correctness/reproducibility)" "$HERE/test_l3_infra.sh"
run "second-batch checkers (negative/positive)" "$HERE/test_l3_validators.sh"
run "Nyx strict comparator + validate.sh chain" "$HERE/test_nyx_validator.sh"
run "verdict classes / regression summary (rc 3 stays PENDING)" "$HERE/test_l3_verdict.sh"
[ $rc -eq 0 ] && echo "ALL LEVEL3 TEST GROUPS PASSED" || echo "SOME LEVEL3 TEST GROUPS FAILED"
exit $rc
