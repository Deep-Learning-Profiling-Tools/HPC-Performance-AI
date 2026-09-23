#!/usr/bin/env bash
# run_conformance.sh -- admission test for a platform's timing backend.
#
#   tools/timing/probes/conformance/run_conformance.sh [--collector NAME] [--backend CUDA|HIP]
#
# Builds probe.cu for the platform's backend, measures it through the SAME engine
# (lib/engine.sh) and summarize path the suite uses, and checks the result against
# expected.json (check.py). The verdict is written to tools/timing/platforms/<platform>.json;
# summarize.py flags every record of a platform without a passing verdict.
#
# A platform is supported by a collector only when this passes: exact 10/20/1 split of
# compute ops around the ROI, the excluded check carved out, copies in the right
# category, clean and profiled ROI consistent, and a sentinel planted in the caller's
# environment absent from everything the profiler wrote.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$(cd "$HERE/../.." && pwd)"
REPO="$(cd "$TOOLS/../.." && pwd)"
LEVEL=0 CLEAN_RUNS=3 WARMUP_RUNS=1 PROFILED_RUNS=1 SKIP_VERIFY=0 DRY_RUN=0
COLLECTOR=auto BACKEND=CUDA ENV_SCRIPT="" BUILD_ROOT="" RAW_ROOT="$REPO/build/timing"
while [ $# -gt 0 ]; do
    case "$1" in
        --collector) COLLECTOR="${2:?}"; shift 2 ;;
        --backend)   BACKEND="${2:?}"; shift 2 ;;
        --raw-root)  RAW_ROOT="${2:?}"; shift 2 ;;
        *) echo "run_conformance: unknown argument '$1'" >&2; exit 2 ;;
    esac
done
# shellcheck source=../../lib/engine.sh
. "$TOOLS/lib/engine.sh"

# A sentinel in the CALLER's environment: it must not reach anything the profiler writes.
export HPCPERF_CONFORMANCE_SENTINEL="hpcperf-sentinel-$$-$(date +%s%N)"

engine_setup
BUILD="$REPO/build/timing-conformance/$PLATFORM_ID"
mkdir -p "$BUILD"
case "$BACKEND" in
    CUDA) command -v nvcc >/dev/null 2>&1 || engine_die "nvcc not on PATH"
          nvcc -O2 -arch=native -I"$TOOLS/roi" -I"$HERE" -o "$BUILD/probe" "$HERE/probe.cu" \
              > "$BUILD/build.log" 2>&1 || engine_die "probe build failed, see $BUILD/build.log" ;;
    HIP)  command -v hipcc >/dev/null 2>&1 || engine_die "hipcc not on PATH (HIP path is UNVERIFIED)"
          hipcc -O2 -x hip -I"$TOOLS/roi" -I"$HERE" -o "$BUILD/probe" "$HERE/probe.cu" \
              > "$BUILD/build.log" 2>&1 || engine_die "probe build failed, see $BUILD/build.log" ;;
    *)    engine_die "backend $BACKEND has no conformance probe build (TPU: interface only)" ;;
esac

echo "run_conformance: platform=$PLATFORM_ID collector=$COLLECTOR backend=$BACKEND run_id=$RUN_ID"
measure_case 0 probe default "$BACKEND" - "$BUILD" 120 - "$(printf '%q' "$BUILD/probe")" \
    - - - - - "set-up, warm-up, the excluded check, the work after the ROI" excluded \
    "conformance probe"
python3 "$HERE/check.py" "$RAW_ROOT/level0/probe/default/$RUN_ID" "$HPCPERF_CONFORMANCE_SENTINEL"
