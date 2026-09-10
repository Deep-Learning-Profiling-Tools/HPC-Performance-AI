#!/usr/bin/env bash
# validate_workspace.sh -- TRUSTED validation of an agent workspace (run by the harness, never by the agent).
#
#   tools/validate_workspace.sh level3 <app> <workspace-benchmark-dir> [--iteration N] [--baseline FILE]
#                               [--skip-build] [-- <validate.sh args...>]
#
# 1. check_workspace.py --agent-mode against the trusted baseline (repo copy under .hpcperf/workspace_baselines/
#    preferred, else the workspace's own copy): any change outside the modifiable scope -- validate.sh, run.sh,
#    build.sh, benchmark.yaml, optimization_scope.yaml, provenance/**, inputs/**, references/**, dependency
#    source -- is READONLY TAMPERING: the validation is REFUSED (exit 3) and nothing is run.
# 2. build inside the workspace (its own build.sh -> workspace-private build/ and .deps/), so that the
#    binary under test is compiled from the workspace's current source (incremental: only changed files).
# 3. the workspace's validate.sh (identical to the canonical one by step 1) on the workspace's own binary.
# 4. record iteration, current source hash, modified files, binary sha256 (from run_manifest.txt) and the
#    verdict in <workspace-root>/reports/iter-<N>.{check.json,diff,verdict.yaml}.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ $# -ge 3 ] || { echo "usage: $0 level3 <app> <workspace-benchmark-dir> [--iteration N] [--baseline FILE] [--skip-build] [-- args]" >&2; exit 2; }
LEVEL=$1; APP=$2; WAPP="$(cd "$3" && pwd)"; shift 3
ITER=""; BASE=""; SKIP_BUILD=0
while [ $# -gt 0 ]; do case "$1" in --iteration) ITER=$2; shift 2;; --baseline) BASE=$2; shift 2;; --skip-build) SKIP_BUILD=1; shift;; --) shift; break;; *) echo "unknown option $1" >&2; exit 2;; esac; done
[ "$LEVEL" = level3 ] || { echo "validate_workspace: only level3" >&2; exit 2; }
WS="$(cd "$WAPP/../.." && pwd)"
[ -f "$WAPP/workspace.yaml" ] || { echo "validate_workspace: $WAPP is not an agent workspace (workspace.yaml missing)" >&2; exit 2; }
RUN_ID="$(sed -n 's/^run_id: //p' "$WAPP/workspace.yaml" | head -1)"
[ -n "$ITER" ] || ITER="$(( $(sed -n 's/^iteration: //p' "$WAPP/workspace.yaml" | head -1) + 1 ))"
REP="$WS/reports"; mkdir -p "$REP"
BARG=(); [ -n "$BASE" ] && BARG=(--baseline "$BASE")
echo "validate_workspace: run $RUN_ID iteration $ITER ($WAPP)"
# 1. trusted contract check
if ! python3 "$HERE/check_workspace.py" "$WAPP" --agent-mode --iteration "$ITER" "${BARG[@]}" --report "$REP/iter-$ITER.check.json" --diff "$REP/iter-$ITER.diff" --json "$REP/iter-$ITER.check_workspace.json" --quick; then
    echo "validate_workspace: REFUSED -- the workspace violates the contract (readonly tampering or broken layout); no build, no run, no verdict" >&2
    printf 'run_id: %s\niteration: %s\nverdict: REFUSED\nreason: check_workspace agent-mode FAIL (see iter-%s.check.json)\nutc: %s\n' "$RUN_ID" "$ITER" "$ITER" "$(date -u +%FT%TZ)" > "$REP/iter-$ITER.verdict.yaml"
    exit 3
fi
# 2. build from the workspace's current source
if [ "$SKIP_BUILD" -eq 0 ]; then
    echo "validate_workspace: building from the workspace source"
    if ! ( cd "$WAPP" && ./build.sh > "$REP/iter-$ITER.build.log" 2>&1 ); then
        tail -20 "$REP/iter-$ITER.build.log"
        printf 'run_id: %s\niteration: %s\nverdict: BUILD_FAIL\nsource_hash: %s\nutc: %s\n' "$RUN_ID" "$ITER" "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["current_source_hash"])' "$REP/iter-$ITER.check.json")" "$(date -u +%FT%TZ)" > "$REP/iter-$ITER.verdict.yaml"
        echo "validate_workspace: BUILD_FAIL (log: $REP/iter-$ITER.build.log)"
        exit 1
    fi
fi
# 3. the (unchanged) validator on the workspace's own binary
rc=0; ( cd "$WAPP" && ./validate.sh "$@" ) 2>&1 | tee "$REP/iter-$ITER.validate.log"; rc=${PIPESTATUS[0]}
case "$rc" in 0) V=PASS;; 1) V=FAIL;; 3) V=PENDING;; 4) V=UNSUPPORTED_LAYOUT;; *) V="FAIL(rc=$rc)";; esac
# 4. record
BIN="$(grep -h '^binary_sha256=' "$WS"/build/level3/"$APP"/*/run*/run_manifest.txt "$WS"/build/level3/"$APP"/*/*/run*/*/run_manifest.txt 2>/dev/null | tail -1 | cut -d= -f2)"
BINP="$(grep -h '^binary=' "$WS"/build/level3/"$APP"/*/run*/run_manifest.txt "$WS"/build/level3/"$APP"/*/*/run*/*/run_manifest.txt 2>/dev/null | tail -1 | cut -d= -f2)"
python3 - "$REP/iter-$ITER.check.json" "$REP/iter-$ITER.verdict.yaml" "$RUN_ID" "$ITER" "$V" "$rc" "${BIN:-unknown}" "${BINP:-unknown}" <<'PY'
import json, sys, datetime
chk = json.load(open(sys.argv[1]))
out = {"run_id": sys.argv[3], "iteration": int(sys.argv[4]), "verdict": sys.argv[5], "validate_exit_code": int(sys.argv[6]),
       "initial_source_hash": chk["initial_source_hash"], "current_source_hash": chk["current_source_hash"],
       "modified_files": chk["modified_files"], "added_files": chk["added_files"], "deleted_files": chk["deleted_files"],
       "binary": sys.argv[8], "binary_sha256": sys.argv[7], "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
import yaml; yaml.safe_dump(out, open(sys.argv[2], "w"), sort_keys=False)
print(f"validate_workspace: {sys.argv[5]} (iteration {sys.argv[4]}, source {chk['current_source_hash'][:16]}..., binary {sys.argv[7][:16]}..., {len(chk['modified_files'])} modified file(s))")
PY
exit "$rc"
