#!/usr/bin/env bash
# remote_fetch_check.sh -- the two post-publication verifications of a Level 3 source artifact. Both are
# anonymous (no credentials) and use an empty cache; neither may touch the maintainer's staging or this
# machine's cache. Run only AFTER the artifact has been published.
#
#   # 1. plan-URL verification (immediately after publishing, BEFORE the lock update commit M):
#   tools/artifacts/remote_fetch_check.sh --mode plan-url --plan level3/RELEASE_PLAN.json --app lammps \
#        [--variant V] [--record-into <repo-checkout>]
#
#   # 2. ordinary user-entry verification (after M is pushed): clean clone of M, plain prepare_benchmark.sh
#   tools/artifacts/remote_fetch_check.sh --mode lock-entry --app lammps [--variant V] --commit <M> \
#        --record-into <repo-checkout> [--clone-url URL]
#
# Why two modes: the lock can only carry a real URL once the asset exists, and the ordinary entry can only be
# tested once the lock carries it. Mode 1 therefore takes the URL from the reviewed release plan while the
# expected size, archive sha256 and source_tree_sha256 come ONLY from the trusted lock metadata
# (tools/artifacts/verify_published_artifact.py) -- a URL can never redefine what the artifact must contain.
# Mode 2 then proves the normal user path end to end and is what REMOTE_FETCH_VERIFIED means.
# Records: provenance/remote_artifact_verification[.variant].yaml (mode 1) and
# provenance/remote_fetch_verification[.variant].yaml (mode 2), written only on success.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; R="$(cd "$HERE/../.." && pwd)"
MODE=""; APP=""; VARIANT=""; COMMIT=""; RECORD=""; PLAN=""; URL="https://github.com/Deep-Learning-Profiling-Tools/HPC-Performance-AI.git"
while [ $# -gt 0 ]; do case "$1" in
    --mode) MODE=$2; shift 2;; --app) APP=$2; shift 2;; --variant) VARIANT=$2; shift 2;;
    --commit) COMMIT=$2; shift 2;; --record-into) RECORD=$2; shift 2;; --plan) PLAN=$2; shift 2;;
    --clone-url) URL=$2; shift 2;; -*) echo "unknown option $1" >&2; exit 2;; *) APP=$1; shift;; esac; done
[ -n "$APP" ] && [ -n "$MODE" ] || { sed -n '2,22p' "$0" >&2; exit 2; }
for v in GITHUB_TOKEN GH_TOKEN; do [ -z "${!v:-}" ] || { echo "remote_fetch_check: $v is set -- an authenticated download is not an anonymous fetch; unset it" >&2; exit 2; }; done
unset HPCPERF_ARTIFACT_STAGING HPCPERF_ARTIFACT_CACHE
SFX=""; [ -n "$VARIANT" ] && SFX=".$VARIANT"
VARG=(); [ -n "$VARIANT" ] && VARG=(--variant "$VARIANT")
T="$(mktemp -d "${TMPDIR:-/tmp}/hpcperf-remote-fetch-XXXXXX")"; trap 'rm -rf "$T"' EXIT

if [ "$MODE" = plan-url ]; then
    [ -n "$PLAN" ] || { echo "remote_fetch_check: --plan is required in mode plan-url" >&2; exit 2; }
    LOCK="$R/level3/$APP/provenance/source.lock$SFX.yaml"
    [ -f "$LOCK" ] || { echo "remote_fetch_check: $LOCK missing" >&2; exit 2; }
    ART="$APP${VARIANT:+.$VARIANT}"
    PLAN_URL="$(python3 - "$PLAN" "$ART" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
for a in plan["assets"]:
    if a["artifact"] == sys.argv[2]:
        print(a["planned_url"]); break
else:
    sys.exit(f"artifact {sys.argv[2]} is not in the plan")
PY
)" || exit 2
    echo "remote_fetch_check: mode plan-url, artifact $ART, URL from the reviewed plan, identity from $LOCK"
    REC=""; [ -n "$RECORD" ] && REC="$RECORD/level3/$APP/provenance/remote_artifact_verification$SFX.yaml"
    python3 "$HERE/verify_published_artifact.py" --lock "$LOCK" --url "$PLAN_URL" ${REC:+--record "$REC"} --cache "$T/cache" --scratch "$T/extract"
    exit $?
fi

if [ "$MODE" = lock-entry ]; then
    [ -n "$COMMIT" ] && [ -n "$RECORD" ] || { echo "remote_fetch_check: --commit and --record-into are required in mode lock-entry" >&2; exit 2; }
    export HPCPERF_ARTIFACT_CACHE="$T/empty-cache"; mkdir -p "$HPCPERF_ARTIFACT_CACHE"
    echo "remote_fetch_check: mode lock-entry, clean clone at $COMMIT, empty cache, no credentials"
    git -c credential.helper= clone -q "$URL" "$T/clone"
    git -C "$T/clone" checkout -q "$COMMIT"
    ( cd "$T/clone" && ./tools/prepare_benchmark.sh level3 "$APP" "${VARG[@]}" ) 2>&1 | tee "$T/prepare.log"
    grep -q 'STATUS REMOTE_FETCH_VERIFIED' "$T/prepare.log" && grep -q 'STATUS MATERIALIZED' "$T/prepare.log" \
        || { echo "remote_fetch_check: the artifact was not fetched from the lock's published URL and materialized" >&2; exit 1; }
    ( cd "$T/clone" && python3 tools/check_workspace.py "level3/$APP" "${VARG[@]}" --quick ) \
        || { echo "remote_fetch_check: check_workspace failed on the fetched tree" >&2; exit 1; }
    OUT="$RECORD/level3/$APP/provenance/remote_fetch_verification$SFX.yaml"
    { echo "schema: hpcperf-remote-fetch-verification-1"; echo "benchmark: level3/$APP"; echo "variant: ${VARIANT:-null}";
      echo "verified_commit: $COMMIT"; echo "verdict: PASS";
      echo "method: clean clone of the lock-update commit, empty cache, no credentials, ordinary tools/prepare_benchmark.sh; size + archive sha256 + source_tree_sha256 verified from the lock; check_workspace PASS";
      echo "utc: $(date -u +%FT%TZ)"; echo "host: $(hostname)"; } > "$OUT"
    echo "remote_fetch_check: PASS -> $OUT"
    exit 0
fi
echo "remote_fetch_check: unknown --mode $MODE (plan-url | lock-entry)" >&2; exit 2
