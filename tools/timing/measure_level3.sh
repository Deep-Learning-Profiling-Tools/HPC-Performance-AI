#!/usr/bin/env bash
# measure_level3.sh -- time the Level 3 applications with their own timers.
#
#   tools/timing/measure_level3.sh all
#   tools/timing/measure_level3.sh lammps cp2k/h2o128
#   tools/timing/measure_level3.sh --dry-run all
#
# Selection: all | <app> | <app>/<case>. Cases: tools/timing/cases/level3_*.tsv.
#
# Level 3 applications are full production codes and carry no ROI markers. The measured
# region of each one is the timer the application prints itself, fixed once per
# application with source citations in tools/timing/apptimers.py (the time-step loop,
# without start-up, set-up, the application's own warm-up step and final output). The
# record has the same schema as Level 1/2 (hpcperf-timing-2, roi.source = "app_timer").
#
# Protocol per case: 1 clean run of level3/<app>/run.sh (the timer, FOM and launcher
# audit are read from it, no profiler) + 1 profiled run -- except for applications whose
# row in cases/level3_apps.tsv says `profile = no (<reason>)`: QMCPACK is not profiled by
# default (its profiled run writes a 24 GB trace and cost ~55 of the 100 minutes of the
# first sweep for a 5-minute application; the reason is in the table and in its record).
# Without markers the profiled run gives the device activity of the WHOLE process (context:
# it includes set-up), except for applications that emit an NVTX range for their loop
# themselves (cases/level3_apps.tsv nvtx_roi), where it is clipped to that range. nsys wraps run.sh from the OUTSIDE, as for
# Level 2, so the launcher's GPU audit stays clean. validate.sh never runs here.
#
# Each run writes its run directory under build/level3/<app>/<profile>/run.timing-<run id>-<c0|prof>
# (HPCPERF_L3_RUN_SUBDIR), so no validated or historical run directory is touched; the files
# the timer is read from are copied into the raw evidence.
#
# Every run starts from `env -i`; the environment script is then sourced inside the clean
# environment so the applications find their toolchain (and the profiler records no login
# shell variables).
#
# Options
#   --env-script F      environment loader, relative to the repo (default hpcperf_env.sh,
#                       or $HPCPERF_TIMING_ENV_SCRIPT)
#   --clean-runs N      clean runs per case (default 1)
#   --no-profile        no profiled run for any case (the application's timer and FOM only)
#   --profile-all       also profile the applications the table does not profile by default
#   --collector NAME    auto (default) | nvidia_nsys | none
#   --backend B         CUDA (default) | HIP (UNTESTED)
#   --raw-root DIR      raw evidence root (default build/timing)
#   --results-root DIR  records, CSVs and the web page (default results/timing)
#   --no-summary        skip the summarize + report step after the runs
#   --dry-run           print what would run
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
LEVEL=3 CLEAN_RUNS=1 WARMUP_RUNS=0 PROFILED_RUNS=1 SKIP_VERIFY=0 DRY_RUN=0 FORCE_PROFILE=0
COLLECTOR=auto BACKEND=CUDA BUILD_ROOT="" RAW_ROOT="$REPO/build/timing"
ENV_SCRIPT="${HPCPERF_TIMING_ENV_SCRIPT:-hpcperf_env.sh}"
SELECT=()
die() { echo "measure_level3: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
    case "$1" in
        --env-script)  ENV_SCRIPT="${2:?}"; shift 2 ;;
        --clean-runs)  CLEAN_RUNS="${2:?}"; shift 2 ;;
        --no-profile)  PROFILED_RUNS=0; shift ;;
        --profile-all) FORCE_PROFILE=1; shift ;;
        --collector)   COLLECTOR="${2:?}"; shift 2 ;;
        --backend)     BACKEND="${2:?}"; shift 2 ;;
        --raw-root)    RAW_ROOT="${2:?}"; shift 2 ;;
        --results-root) RESULTS_ROOT="${2:?}"; shift 2 ;;
        --no-summary)  SUMMARIZE=0; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
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
