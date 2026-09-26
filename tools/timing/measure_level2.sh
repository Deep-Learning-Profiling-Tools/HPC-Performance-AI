#!/usr/bin/env bash
# measure_level2.sh -- measure the region of interest (ROI) of Level 2 mini-applications.
#
#   tools/timing/measure_level2.sh all
#   tools/timing/measure_level2.sh quicksilver amg2023/n128
#   tools/timing/measure_level2.sh --dry-run all
#
# Selection: all | <app> | <app>/<case>. Cases: tools/timing/cases/level2_*.tsv.
#
# Protocol per case: 1 clean run of level2/<app>/run.sh (ROI timed by the markers, FOM
# and launcher audit read from it, no profiler) + 1 profiled run (device activity clipped
# to the same markers). No warm-up, no statistical repeats, validate.sh never runs:
# Level 2 keeps verification there. nsys wraps run.sh from the OUTSIDE -- 20 of 24 run.sh
# end in `exec`, and a profiler inside the launcher's wrapper breaks its nvidia-smi pid
# audit; from outside the audit stays clean.
#
# Every run starts from `env -i`; the repository environment script is then sourced
# inside the clean environment so the applications find their toolchain.
#
# Options
#   --env-script F      environment loader, relative to the repo (default hpcperf_env.sh,
#                       or $HPCPERF_TIMING_ENV_SCRIPT)
#   --clean-runs N      clean runs per case (default 1)
#   --no-profile        no profiled run (ROI time and FOM only)
#   --collector NAME    auto (default) | nvidia_nsys | none
#   --backend B         CUDA (default) | HIP (UNTESTED)
#   --raw-root DIR      raw evidence root (default build/timing)
#   --results-root DIR  records, CSVs and the web page (default results/timing)
#   --no-summary        skip the summarize + report step after the runs
#   --dry-run           print what would run
#   --registry          measure the registered inputs (level*/<app>/inputs.yaml via the generated
#                       cases/level2_registry.tsv; SELECT = all | <app> | <app>/<input_id>)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
LEVEL=2 CLEAN_RUNS=1 WARMUP_RUNS=0 PROFILED_RUNS=1 SKIP_VERIFY=0 DRY_RUN=0 REGISTRY=0
COLLECTOR=auto BACKEND=CUDA BUILD_ROOT="" RAW_ROOT="$REPO/build/timing"
ENV_SCRIPT="${HPCPERF_TIMING_ENV_SCRIPT:-hpcperf_env.sh}"
SELECT=()
die() { echo "measure_level2: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
    case "$1" in
        --env-script)  ENV_SCRIPT="${2:?}"; shift 2 ;;
        --clean-runs)  CLEAN_RUNS="${2:?}"; shift 2 ;;
        --no-profile)  PROFILED_RUNS=0; shift ;;
        --collector)   COLLECTOR="${2:?}"; shift 2 ;;
        --backend)     BACKEND="${2:?}"; shift 2 ;;
        --raw-root)    RAW_ROOT="${2:?}"; shift 2 ;;
        --results-root) RESULTS_ROOT="${2:?}"; shift 2 ;;
        --no-summary)  SUMMARIZE=0; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        --registry)    REGISTRY=1; shift ;;
        -h|--help)     awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)            die "unknown option '$1' (try --help)" ;;
        *)             SELECT+=("$1"); shift ;;
    esac
done
[ "${#SELECT[@]}" -gt 0 ] || die "give a selection: all, <app> or <app>/<case> (try --help)"
case "$CLEAN_RUNS" in ''|*[!0-9]*|0) die "--clean-runs must be a positive number" ;; esac
# shellcheck source=lib/engine.sh
. "$HERE/lib/engine.sh"
engine_main "${SELECT[@]}"
