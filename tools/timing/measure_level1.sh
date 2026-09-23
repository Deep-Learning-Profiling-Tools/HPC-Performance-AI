#!/usr/bin/env bash
# measure_level1.sh -- measure the region of interest (ROI) of Level 1 benchmarks.
#
#   tools/timing/measure_level1.sh --build-root build/gcc13 all
#   tools/timing/measure_level1.sh --build-root build/gcc13 daxpy all_pairs_distance/n20000
#   tools/timing/measure_level1.sh --build-root build/gcc13 --dry-run all
#
# Selection: all | <benchmark> | <benchmark>/<case>. Cases: tools/timing/cases/level1*.tsv.
#
# Protocol per case (tools/timing/lib/engine.sh has the details):
#   1 warm-up run (discarded) + N clean runs (ROI timed by the markers, no profiler)
#   + 1 profiled run (device activity clipped to the same markers).
# The measured time is the ROI: the computation, without process start-up, set-up,
# warm-up and verification. HPCPERF_SKIP_VERIFY=1 is still set -- not for correctness
# (verification is outside the ROI) but because the CPU reference can take minutes.
#
# Options
#   --build-root DIR    Level 1 build tree (default: build/all or build/gcc13)
#   --clean-runs N      clean runs per case (default 5)
#   --warmup N          discarded warm-up runs (default 1)
#   --no-profile        no profiled run (ROI time only)
#   --collector NAME    auto (default) | nvidia_nsys | none
#   --backend B         CUDA (default) | HIP (UNTESTED)
#   --keep-verify       do not set HPCPERF_SKIP_VERIFY
#   --raw-root DIR      raw evidence root (default build/timing)
#   --results-root DIR  records, CSVs and the web page (default results/timing)
#   --no-summary        skip the summarize + report step after the runs
#   --dry-run           print what would run
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
LEVEL=1 CLEAN_RUNS=5 WARMUP_RUNS=1 PROFILED_RUNS=1 SKIP_VERIFY=1 DRY_RUN=0
COLLECTOR=auto BACKEND=CUDA BUILD_ROOT="" ENV_SCRIPT="" RAW_ROOT="$REPO/build/timing"
SELECT=()
die() { echo "measure_level1: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
    case "$1" in
        --build-root)  BUILD_ROOT="${2:?}"; shift 2 ;;
        --clean-runs)  CLEAN_RUNS="${2:?}"; shift 2 ;;
        --warmup)      WARMUP_RUNS="${2:?}"; shift 2 ;;
        --no-profile)  PROFILED_RUNS=0; shift ;;
        --collector)   COLLECTOR="${2:?}"; shift 2 ;;
        --backend)     BACKEND="${2:?}"; shift 2 ;;
        --keep-verify) SKIP_VERIFY=0; shift ;;
        --raw-root)    RAW_ROOT="${2:?}"; shift 2 ;;
        --results-root) RESULTS_ROOT="${2:?}"; shift 2 ;;
        --no-summary)  SUMMARIZE=0; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        -h|--help)     awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)            die "unknown option '$1' (try --help)" ;;
        *)             SELECT+=("$1"); shift ;;
    esac
done
[ "${#SELECT[@]}" -gt 0 ] || die "give a selection: all, <benchmark> or <benchmark>/<case> (try --help)"
case "$CLEAN_RUNS" in ''|*[!0-9]*|0) die "--clean-runs must be a positive number" ;; esac
case "$WARMUP_RUNS" in ''|*[!0-9]*) die "--warmup must be a number" ;; esac
if [ -z "$BUILD_ROOT" ]; then
    for cand in "$REPO/build/all" "$REPO/build/gcc13"; do
        [ -d "$cand/level1" ] && { BUILD_ROOT="$cand"; break; }
    done
fi
[ -n "$BUILD_ROOT" ] && [ -d "$BUILD_ROOT" ] || die "no Level 1 build tree; pass --build-root"
BUILD_ROOT="$(cd "$BUILD_ROOT" && pwd)"
# shellcheck source=lib/engine.sh
. "$HERE/lib/engine.sh"
engine_main "${SELECT[@]}"
