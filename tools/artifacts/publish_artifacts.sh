#!/usr/bin/env bash
# publish_artifacts.sh -- publish plan / preflight for the Level 3 source artifacts (scheme 3).
#
#   tools/artifacts/publish_artifacts.sh --dry-run [--app a,b,...] [--staging DIR] [--full] [--catalog FILE]
#   tools/artifacts/publish_artifacts.sh --provider <name> --release <name> [--app ...]     (no adapter yet: REFUSED)
#
# For every artifact of the catalog's default suite (+ admitted candidates) it prints: application, variant,
# artifact local path (staging), source version, archive size, archive SHA256, source tree SHA256, intended
# remote filename, intended release/version, license (redistribution) status, publish status, and the decision
# PLAN or REFUSE with the reasons. REFUSE when: redistribution_status != cleared, archive hash/size mismatch,
# tree hash mismatch (--full re-extracts), secret scan failure, benchmark.yaml/source.lock mismatch, artifact
# missing from staging, retired application. Nothing is uploaded by this round: no provider adapter exists
# (prepare_benchmark.sh must keep working from plain immutable https URLs, without gh/aws CLIs), so a real
# publish request is refused until the maintainer selects the provider and an adapter is added.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY=0; PROVIDER=""; RELEASE=""; APPS=""; STAGING="${HPCPERF_ARTIFACT_STAGING:-}"; FULL=""; CATALOG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1; shift;;
        --provider) PROVIDER=$2; shift 2;;
        --release) RELEASE=$2; shift 2;;
        --app) APPS=$2; shift 2;;
        --staging) STAGING=$2; shift 2;;
        --full) FULL=--full; shift;;
        --catalog) CATALOG=$2; shift 2;;
        -h|--help) sed -n '2,20p' "$0"; exit 0;;
        *) echo "publish_artifacts: unknown option $1" >&2; exit 2;;
    esac
done
[ -n "$STAGING" ] || { echo "publish_artifacts: set HPCPERF_ARTIFACT_STAGING or pass --staging DIR (maintainer-side local staging)" >&2; exit 2; }
[ -d "$STAGING" ] || { echo "publish_artifacts: staging directory $STAGING does not exist" >&2; exit 2; }
if [ "$DRY" -eq 0 ]; then
    [ -n "$PROVIDER" ] || { echo "publish_artifacts: a real publish needs --provider <name> (or use --dry-run)" >&2; exit 2; }
fi
python3 "$HERE/publish_plan.py" --staging "$STAGING" ${APPS:+--app "$APPS"} ${RELEASE:+--release "$RELEASE"} ${CATALOG:+--catalog "$CATALOG"} $FULL
rc=$?
if [ "$DRY" -eq 0 ]; then
    echo
    echo "publish_artifacts: REFUSED -- no provider adapter is implemented for '$PROVIDER' in this round; nothing was uploaded."
    echo "publish_artifacts: the maintainer decides provider, release naming, visibility and upload timing; then an adapter is added and the plan above is executed artifact by artifact."
    exit 3
fi
exit $rc
