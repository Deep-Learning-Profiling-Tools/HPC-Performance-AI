#!/usr/bin/env bash
# prepare_benchmark.sh -- materialize a frozen Level 3 benchmark source artifact into its benchmark directory.
#
#   tools/prepare_benchmark.sh level3 <app> [--variant NAME] [--artifact FILE] [--cache-dir DIR] [--offline]
#                                           [--force-rematerialize] [--status]
#
# After `git clone`, this is the ONE command that turns level3/<app>/ into a source-level self-contained
# benchmark directory: it reads provenance/source.lock[.variant].yaml, finds the artifact
# (<app>[-<variant>]-<source_version>.tar.zst) in --artifact / the content-addressed cache
# ($HPCPERF_ARTIFACT_CACHE, default <repo>/.artifacts/sha256/<sha256>.tar.zst) / the recorded immutable
# https URL(s), verifies size + sha256, extracts it safely outside the benchmark directory, verifies
# source_tree_sha256, runs the safety checks and atomically places src/ (+ deps/). build.sh/run.sh/
# validate.sh then read application source ONLY from there; nothing is fetched at build time.
# Idempotent (READY when the tree already matches); a modified tree is refused (DIRTY, exit 3) unless
# --force-rematerialize is given explicitly -- build.sh never calls this script.
# Exit codes and status words: see tools/hpcperf_materialize.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
[ $# -ge 2 ] || { echo "usage: $0 level3 <app> [--variant NAME] [--artifact FILE] [--cache-dir DIR] [--offline] [--force-rematerialize] [--status]" >&2; exit 2; }
LEVEL="$1"; APP="$2"; shift 2
[ "$LEVEL" = level3 ] || { echo "prepare_benchmark: only level3 benchmarks are materialized from source artifacts (got '$LEVEL')" >&2; exit 2; }
case "$APP" in ""|*/*|.*) echo "prepare_benchmark: invalid application name '$APP'" >&2; exit 2;; esac
DIR="$R/$LEVEL/$APP"
[ -d "$DIR" ] || { echo "prepare_benchmark: $DIR does not exist" >&2; exit 2; }
command -v zstd >/dev/null || { echo "prepare_benchmark: zstd not found on PATH (source hpcperf_env.sh)" >&2; exit 2; }
exec python3 "$HERE/hpcperf_materialize.py" "$DIR" "$@"
