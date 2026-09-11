#!/usr/bin/env bash
# validate_workspace.sh -- TRUSTED validation of an agent workspace (run by the harness, never by the agent).
#
#   tools/validate_workspace.sh level3 <app> <workspace-benchmark-dir> [--iteration N] [--baseline FILE]
#                               [--backend CUDA|HIP] [--skip-build] [-- <validate.sh args...>]
#
# 1. check_workspace.py --agent-mode against the TRUSTED baseline: the repository copy
#    <repo>/.hpcperf/workspace_baselines/<run-id>.json or an explicit --baseline outside the workspace (the copy
#    inside the workspace is never trusted here): any change outside the modifiable scope -- validate.sh, run.sh,
#    build.sh, benchmark.yaml, optimization_scope.yaml, provenance/**, inputs/**, references/**, dependency
#    source, the harness copies of the workspace root -- is tampering: the validation is REFUSED (exit 6),
#    nothing is built or run, and the result never enters a scientific or performance summary.
# 2. build inside the workspace (its own build.sh -> workspace-private build/ and .deps/) for the SAME backend
#    and variant that step 3 validates; a failed build is BUILD_FAIL (exit 7). Every successful build appends a
#    trusted build record (source hash, backend, variant, sha256 of the produced executables) to
#    <workspace-root>/reports/build-records.json.
# 3. the workspace's validate.sh (identical to the canonical one by step 1) on the workspace's own binary;
#    its exit code is propagated unchanged: 0 PASS, 1 FAIL, 3 PENDING (Nyx heat/cool I_R_CHECK_PENDING),
#    4 UNSUPPORTED_LAYOUT -- numerical-layer outcomes.
# 4. record iteration, layer, verdict, exit code, source hashes, modified files and THIS iteration's runs in
#    <workspace-root>/reports/iter-<N>.{check.json,diff,verdict.yaml}. The runs of this iteration are identified
#    by comparing the set, size and content of run_manifest.txt files before and after step 3 -- never by taking
#    the last line of a historical manifest -- and, because run.sh APPENDS to a manifest, only the records
#    appended during this iteration are read; every one is listed with its run id, ranks, exit code, binary and
#    binary sha256.
#    Build provenance of the validated binary: `built_this_iteration` when step 2 ran, otherwise the binary
#    sha256 is looked up in the trusted build records: `verified_from_build_record` when a record for the
#    current source hash, backend and variant contains it, else `UNVERIFIED` -- `--skip-build` alone never
#    asserts that the current source modification was compiled.
# Exit-code layers (never mixed): 6 REFUSED (workspace integrity), 7 BUILD_FAIL (build), 0/1/3/4 (numerical,
# = validate.sh contract), 2 usage. These are hash/permission checks, not an OS sandbox.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ $# -ge 3 ] || { echo "usage: $0 level3 <app> <workspace-benchmark-dir> [--iteration N] [--baseline FILE] [--backend CUDA|HIP] [--skip-build] [-- args]" >&2; exit 2; }
LEVEL=$1; APP=$2; WAPP="$(cd "$3" 2>/dev/null && pwd)" || { echo "validate_workspace: $3 not a directory" >&2; exit 2; }; shift 3
ITER=""; BASE=""; SKIP_BUILD=0; BACKEND=""
while [ $# -gt 0 ]; do case "$1" in
    --iteration) ITER=$2; shift 2;; --baseline) BASE=$2; shift 2;; --backend) BACKEND=$2; shift 2;;
    --skip-build) SKIP_BUILD=1; shift;; --) shift; break;; *) echo "unknown option $1" >&2; exit 2;; esac; done
EXIT_REFUSED=6; EXIT_BUILD_FAIL=7
[ "$LEVEL" = level3 ] || { echo "validate_workspace: only level3" >&2; exit 2; }
WS="$(cd "$WAPP/../.." && pwd)"
[ -f "$WAPP/workspace.yaml" ] || { echo "validate_workspace: $WAPP is not an agent workspace (workspace.yaml missing)" >&2; exit 2; }
RUN_ID="$(sed -n 's/^run_id: //p' "$WAPP/workspace.yaml" | head -1)"
[ -n "$ITER" ] || ITER="$(( $(sed -n 's/^iteration: //p' "$WAPP/workspace.yaml" | head -1) + 1 ))"
REP="$WS/reports"; mkdir -p "$REP"
VERDICT="$REP/iter-$ITER.verdict.yaml"
refuse() { # <reason>
    printf 'run_id: %s\niteration: %s\nlayer: integrity\nverdict: REFUSED\nexit_code: %s\nreason: %s\nutc: %s\n' \
        "$RUN_ID" "$ITER" "$EXIT_REFUSED" "$1" "$(date -u +%FT%TZ)" > "$VERDICT"
    echo "validate_workspace: REFUSED -- $1" >&2; exit $EXIT_REFUSED; }
# backend/variant: build and validate must use the same ones. The backend is the first positional argument of
# both build.sh and validate.sh; a backend given after `--` must agree with --backend.
FIRST_ARG="${1:-}"
case "$(echo "$FIRST_ARG" | tr '[:lower:]' '[:upper:]')" in
    CUDA|HIP) ARG_BACKEND="$(echo "$FIRST_ARG" | tr '[:lower:]' '[:upper:]')"; shift;; *) ARG_BACKEND="";; esac
if [ -n "$BACKEND" ] && [ -n "$ARG_BACKEND" ] && [ "$BACKEND" != "$ARG_BACKEND" ]; then
    echo "validate_workspace: --backend $BACKEND and the validate.sh argument $ARG_BACKEND disagree -- build and validation must use one backend" >&2; exit 2
fi
BACKEND="${BACKEND:-${ARG_BACKEND:-CUDA}}"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
# variant (nekRS): the materialized marker decides; the benchmark's variant_env is exported so that build.sh,
# run.sh and validate.sh of this iteration cannot select different variants
VARIANT="$(sed -n 's/^variant: *//p' "$WAPP/.hpcperf-materialized.yaml" 2>/dev/null | sed "s/^'\(.*\)'$/\1/; s/^null$//" | head -1)"
VARIANT_ENV="$(sed -n 's/^variant_env: *//p' "$WAPP/benchmark.yaml" 2>/dev/null | head -1)"
if [ -n "$VARIANT" ] && [ -n "$VARIANT_ENV" ]; then
    CUR="$(printenv "$VARIANT_ENV" 2>/dev/null || true)"
    if [ -n "$CUR" ] && [ "$CUR" != "$VARIANT" ]; then
        refuse "environment $VARIANT_ENV=$CUR does not match the materialized variant $VARIANT (build and validation would diverge)"
    fi
    export "$VARIANT_ENV=$VARIANT"
fi
echo "validate_workspace: run $RUN_ID iteration $ITER backend $BACKEND${VARIANT:+ variant $VARIANT} ($WAPP)"
# 1. trusted contract check (the baseline inside the workspace is never accepted here)
case "$BASE" in "$WS"|"$WS"/*) refuse "--baseline must lie outside the workspace root";; esac
BARG=(); [ -n "$BASE" ] && BARG=(--baseline "$BASE")
if ! python3 "$HERE/check_workspace.py" "$WAPP" --agent-mode --iteration "$ITER" "${BARG[@]}" --report "$REP/iter-$ITER.check.json" --diff "$REP/iter-$ITER.diff" --json "$REP/iter-$ITER.check_workspace.json" --quick; then
    refuse "check_workspace agent-mode FAIL (readonly/harness tampering, untrusted baseline or broken layout; see iter-$ITER.check.json)"
fi
SRC_HASH="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["current_source_hash"])' "$REP/iter-$ITER.check.json")"
# manifest snapshot BEFORE the run: (path, sha256) of every existing run manifest under this workspace
SNAP_BEFORE="$REP/.iter-$ITER.manifests.before"
find "$WS/build" -name run_manifest.txt -type f 2>/dev/null | sort | while read -r f; do echo "$(sha256sum "$f" | cut -d' ' -f1) $(stat -c %s "$f") $f"; done > "$SNAP_BEFORE"
# 2. build from the workspace's current source, same backend/variant as the validation
BUILT_THIS_ITER=0
if [ "$SKIP_BUILD" -eq 0 ]; then
    echo "validate_workspace: building from the workspace source ($BACKEND)"
    if ! ( cd "$WAPP" && ./build.sh "$BACKEND" > "$REP/iter-$ITER.build.log" 2>&1 ); then
        tail -20 "$REP/iter-$ITER.build.log"
        printf 'run_id: %s\niteration: %s\nlayer: build\nverdict: BUILD_FAIL\nexit_code: %s\nbackend: %s\nvariant: %s\nsource_hash: %s\nbuild_log: %s\nutc: %s\n' \
            "$RUN_ID" "$ITER" "$EXIT_BUILD_FAIL" "$BACKEND" "${VARIANT:-null}" "$SRC_HASH" "reports/iter-$ITER.build.log" "$(date -u +%FT%TZ)" > "$VERDICT"
        echo "validate_workspace: BUILD_FAIL (build layer, exit $EXIT_BUILD_FAIL; log: $REP/iter-$ITER.build.log)"
        exit $EXIT_BUILD_FAIL
    fi
    BUILT_THIS_ITER=1
    python3 "$HERE/workspace_build_record.py" --workspace "$WS" --app "$APP" --iteration "$ITER" --backend "$BACKEND" \
        --variant "${VARIANT:-}" --source-hash "$SRC_HASH" --build-log "reports/iter-$ITER.build.log" --records "$REP/build-records.json"
fi
# 3. the (unchanged) validator on the workspace's own binary, same backend
rc=0; ( cd "$WAPP" && ./validate.sh "$BACKEND" "$@" ) 2>&1 | tee "$REP/iter-$ITER.validate.log"; rc=${PIPESTATUS[0]}
case "$rc" in 0) V=PASS;; 1) V=FAIL;; 3) V=PENDING;; 4) V=UNSUPPORTED_LAYOUT;; *) V="FAIL(rc=$rc)";; esac   # numerical layer, validate.sh contract
# 4. record: THIS iteration's runs = manifests that appeared or changed during step 3
python3 "$HERE/workspace_iteration_record.py" --workspace "$WS" --app "$APP" --benchmark-dir "$WAPP" --iteration "$ITER" \
    --run-id "$RUN_ID" --backend "$BACKEND" --variant "${VARIANT:-}" --verdict "$V" --exit-code "$rc" \
    --check "$REP/iter-$ITER.check.json" --snapshot-before "$SNAP_BEFORE" --records "$REP/build-records.json" \
    --built-this-iteration "$BUILT_THIS_ITER" --out "$VERDICT"
rm -f "$SNAP_BEFORE"
exit "$rc"
