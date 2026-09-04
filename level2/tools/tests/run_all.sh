#!/bin/bash
# Run the Level 2 infrastructure regression tests that need NO GPU execution:
#   1. hpcperf_topology.py --self-test        (pure Python)
#   2. test_deps_markers.sh                   (setup_level2_deps.sh markers/fingerprints, sandboxed)
#   3. test_launcher_dryrun.sh                (launcher resource parsing; dry-run only)
#   4. test_run_guards.sh                     (run.sh interface guards; dry-run only, needs built apps)
# Usage: level2/tools/tests/run_all.sh   (source hpcperf_env.sh first for 3/4)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rc=0
run() { echo "=== $1"; shift; "$@" || { echo "=== FAILED: $*"; rc=1; }; echo; }
run "topology self-test"   "$HERE/../hpcperf_topology.py" --self-test
run "deps markers"         bash "$HERE/test_deps_markers.sh"
run "launcher dry-run"     bash "$HERE/test_launcher_dryrun.sh"
run "run.sh guards"        bash "$HERE/test_run_guards.sh"
[ $rc -eq 0 ] && echo "ALL TEST GROUPS PASSED" || echo "SOME TEST GROUPS FAILED"
exit $rc
