# tools/timing/lib/engine.sh -- the measurement engine shared by measure_level1.sh and
# measure_level2.sh. Sourced, not executed. The front-end sets the protocol variables
# below, then calls engine_main with the case selection.
#
# What one case costs, and why (the ROI is the unit of measurement):
#   WARMUP_RUNS    whole-process runs, discarded (first-touch file and JIT caches)
#   CLEAN_RUNS     no profiler; the ROI markers log their own timestamps
#                  (HPCPERF_ROI_LOG) -> the headline ROI wall time, the FOM, the audit
#   PROFILED_RUNS  0 or 1; the collector wraps the run -> the device picture, clipped
#                  to the same markers by the analysis
# Every run starts from `env -i` with an allow-list: profilers record the whole process
# environment into their reports (measured with nsys: 349 variables incl. session
# tokens from a login shell, 101 with the allow-list).
#
# Raw evidence: <RAW_ROOT>/level<L>/<app>/<case>/<run_id>/
#   run_meta.txt              case, protocol, platform, provenance (key=value)
#   warmup.<i>/run.log        discarded
#   clean.<i>/{run.log,run.txt,roi.<pid>}
#   prof/{run.log,run.txt,roi.<pid>,trace.*}
# tools/timing/summarize.py turns it into results/timing/.

ENGINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$(cd "$ENGINE_DIR/.." && pwd)"
REPO="$(cd "$TOOLS/../.." && pwd)"
# shellcheck source=collectors.sh
. "$ENGINE_DIR/collectors.sh"

: "${LEVEL:?}" "${CLEAN_RUNS:=1}" "${WARMUP_RUNS:=0}" "${PROFILED_RUNS:=1}"
: "${RAW_ROOT:=$REPO/build/timing}" "${COLLECTOR:=auto}" "${ENV_SCRIPT:=}" "${BUILD_ROOT:=}"
: "${BACKEND:=CUDA}" "${SKIP_VERIFY:=0}" "${DRY_RUN:=0}" "${PROFILE_TIMEOUT_FACTOR:=3}"

BASE_ALLOW="PATH HOME USER LOGNAME SHELL TERM LANG LC_ALL TMPDIR TZ LD_LIBRARY_PATH
            SLURM_JOB_ID SLURM_JOB_NODELIST SLURM_JOB_NUM_NODES SLURM_JOB_GPUS SLURM_GPUS_ON_NODE
            SLURM_GPUS_PER_TASK SLURM_CPUS_ON_NODE SLURM_JOB_CPUS_PER_NODE SLURM_TASKS_PER_NODE
            SLURM_LOCALID SLURM_PROCID"
# Beats the allow-list: a name matching it never leaves the parent, whatever lists it.
ENV_DENY='(TOKEN|SECRET|PASSWD|PASSWORD|CREDENTIAL|PRIVATE_KEY|API_KEY|SESSION)'

engine_die() { echo "measure_level${LEVEL}: $*" >&2; exit 2; }

# ---------------------------------------------------------------- clean environment
_env_allow() { echo $BASE_ALLOW $(backend_env_allow "$BACKEND"); }

_allowed_pairs() {      # NAME=VALUE for every allowed name that is set; denied names reported
    local n v
    DENIED_NAMES=""
    for n in $(_env_allow); do
        if printf '%s' "$n" | /usr/bin/grep -qE "$ENV_DENY"; then
            DENIED_NAMES="$DENIED_NAMES $n"
            continue
        fi
        eval "v=\${$n+set}"
        [ "${v:-}" = set ] && eval "printf '%s=%s\n' \"$n\" \"\$$n\""
    done
}

# run_clean <log> <cwd> <extra NAME=VALUE...> -- <cmd...>
# Starts from `env -i`, sources ENV_SCRIPT (if any) inside the clean environment, runs
# the command in <cwd>. Returns the command's exit code.
run_clean() {
    local log="$1" cwd="$2"; shift 2
    local -a pairs=() extra=()
    local line
    while IFS= read -r line; do pairs+=("$line"); done < <(_allowed_pairs)
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do extra+=("$1"); shift; done
    [ $# -gt 0 ] || engine_die "run_clean: missing -- before the command"
    shift
    env -i "${pairs[@]}" "${extra[@]}" \
        bash -c 'cd "$1" || exit 97; shift; if [ -n "$1" ]; then . "$1" >/dev/null 2>&1 || exit 98; fi; shift; exec "$@"' \
        _ "$cwd" "$ENV_SCRIPT_ABS" "$@" > "$log" 2>&1
}

now_ns() { date +%s%N; }

# ROI seconds of one run directory: slowest process, sum(E-B) - excluded (x lines). "-" if none.
roi_seconds() {
    local f best=""
    for f in "$1"/roi.*; do
        [ -f "$f" ] || continue
        local s
        s=$(awk '/^B /{b=$2} /^E /{w+=$2-b} /^x /{ex+=$2} END{printf "%.6f", (w-ex)/1e9}' "$f")
        if [ -z "$best" ] || awk -v a="$s" -v b="$best" 'BEGIN{exit !(a>b)}'; then best="$s"; fi
    done
    echo "${best:--}"
}

roi_where() {           # source locations of the ROI markers for this app (relative paths)
    local dir
    if [ "$LEVEL" = 1 ]; then dir="level1/$1"; else dir="level2/$1"; fi
    ( cd "$REPO" && /usr/bin/grep -rnE 'HPCPERF_ROI_(BEGIN|END)(_SYNC)?\(|hpcperf_roi_(begin|end)(_sync)?\(' \
        --include='*.c' --include='*.cc' --include='*.cpp' --include='*.cu' --include='*.hip' \
        --include='*.h' --include='*.hpp' --include='*.cuh' --include='*.f90' --include='*.F90' \
        --include='*.f' --include='*.F' "$dir" 2>/dev/null | cut -d: -f1,2 | tr '\n' ' ' )
}

# ---------------------------------------------------------------- one case
# fields: level app case backend gpus cwd timeout_s env argv fom_name fom_unit fom_better
#         fom_source fom_regex roi_excludes verify_vs_roi notes
measure_case() {
    local level="$1" app="$2" case="$3" backend="$4" gpus="$5" cwd="$6" tmo="$7" env="$8" argv="$9"
    local fom_name="${10}" fom_unit="${11}" fom_better="${12}" fom_source="${13}" fom_regex="${14}"
    local roi_excl="${15}" verify="${16}" notes="${17}"
    local -a cmd case_env=() run_env=()
    local kv
    eval "cmd=( $argv )"                       # argv was shlex-quoted by cases.py
    if [ "$env" != "-" ]; then
        IFS=';' read -r -a case_env <<< "$env"
    fi
    for kv in "${case_env[@]}"; do
        printf '%s' "${kv%%=*}" | /usr/bin/grep -qE "$ENV_DENY" && { echo "  $app/$case: refusing variable ${kv%%=*}" >&2; return 1; }
    done
    run_env=("${case_env[@]}")
    [ "$level" = 2 ] && run_env+=("HPCPERF_GPUS=$gpus")
    [ "$SKIP_VERIFY" = 1 ] && run_env+=("HPCPERF_SKIP_VERIFY=1")

    local out="$RAW_ROOT/level$level/$app/$case/$RUN_ID"
    local label; label=$(printf 'level%s %-34s' "$level" "$app/$case")

    if [ "$DRY_RUN" = 1 ]; then
        echo "  $label"
        echo "    cwd      $cwd"
        echo "    env      env -i $(_allowed_pairs | cut -d= -f1 | tr '\n' ' ')${run_env[*]:+${run_env[*]} }HPCPERF_ROI_LOG=<run dir>/roi"
        [ -n "$ENV_SCRIPT_ABS" ] && echo "    source   ${ENV_SCRIPT_ABS#$REPO/} (inside the clean environment)"
        echo "    runs     warmup=$WARMUP_RUNS clean=$CLEAN_RUNS profiled=$PROFILED_RUNS collector=$COLLECTOR"
        echo "    command  timeout $tmo ${cmd[*]}"
        return 0
    fi

    mkdir -p "$out" || return 1
    {
        echo "schema=hpcperf-timing-raw-2"
        echo "run_id=$RUN_ID"
        echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "level=$level"
        echo "app=$app"
        echo "case=$case"
        echo "backend=$backend"
        echo "gpus=$gpus"
        echo "cwd=$cwd"
        echo "timeout_s=$tmo"
        echo "case_env=$env"
        echo "argv=$argv"
        echo "fom_name=$fom_name"
        echo "fom_unit=$fom_unit"
        echo "fom_better=$fom_better"
        echo "fom_source=$fom_source"
        echo "fom_regex=$fom_regex"
        echo "roi_excludes=$roi_excl"
        echo "verify_vs_roi=$verify"
        echo "notes=$notes"
        echo "roi_where=$(roi_where "$app")"
        echo "warmup_runs=$WARMUP_RUNS"
        echo "clean_runs=$CLEAN_RUNS"
        echo "profiled_runs=$PROFILED_RUNS"
        echo "skip_verify=$SKIP_VERIFY"
        echo "collector=$COLLECTOR"
        echo "collector_version=$COLLECTOR_VERSION"
        echo "env_script=${ENV_SCRIPT:--}"
        echo "env_allow=$(_env_allow | tr -s ' \n' '  ')"
        echo "env_deny_regex=$ENV_DENY"
        echo "platform_id=$PLATFORM_ID"
        echo "device_json=$DEVICE_JSON"
        echo "git_commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)"
        echo "git_dirty=$(test -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" && echo 1 || echo 0)"
        [ "$level" = 1 ] && [ -f "${cmd[0]}" ] && echo "exe_sha256=$(sha256sum "${cmd[0]}" | cut -d' ' -f1)"
    } > "$out/run_meta.txt"

    local i t0 t1 rc d status=ok roi_s="-"
    i=0
    while [ "$i" -lt "$WARMUP_RUNS" ]; do
        mkdir -p "$out/warmup.$i"
        run_clean "$out/warmup.$i/run.log" "$cwd" "${run_env[@]}" -- timeout "$tmo" "${cmd[@]}"
        i=$((i + 1))
    done

    i=0
    while [ "$i" -lt "$CLEAN_RUNS" ]; do
        d="$out/clean.$i"; mkdir -p "$d"
        t0=$(now_ns)
        run_clean "$d/run.log" "$cwd" "${run_env[@]}" "HPCPERF_ROI_LOG=$d/roi" -- timeout "$tmo" "${cmd[@]}"
        rc=$?; t1=$(now_ns)
        echo "start_ns=$t0 end_ns=$t1 rc=$rc" > "$d/run.txt"
        if [ "$rc" -ne 0 ]; then
            status="clean_failed"
            [ "$rc" -eq 124 ] && status="clean_timeout"
            echo "  $label FAIL (clean run $i: exit $rc, see ${d#$REPO/}/run.log)"
            break
        fi
        if ! ls "$d"/roi.* >/dev/null 2>&1; then
            status="roi_missing"
            echo "  $label FAIL (clean run $i wrote no ROI record: markers missing or not reached)"
            break
        fi
        [ "$i" -eq 0 ] && roi_s=$(roi_seconds "$d")
        i=$((i + 1))
    done

    local prof_note=""
    if [ "$status" = ok ] && [ "$PROFILED_RUNS" -gt 0 ] && [ "$COLLECTOR" != none ]; then
        d="$out/prof"; mkdir -p "$d"
        collector_wrap "$COLLECTOR" "$d/trace" || return 2
        t0=$(now_ns)
        run_clean "$d/run.log" "$cwd" "${run_env[@]}" "HPCPERF_ROI_LOG=$d/roi" -- \
            timeout "$((tmo * PROFILE_TIMEOUT_FACTOR))" "${COLLECTOR_ARGV[@]}" "${cmd[@]}"
        rc=$?; t1=$(now_ns)
        echo "start_ns=$t0 end_ns=$t1 rc=$rc" > "$d/run.txt"
        if [ "$rc" -ne 0 ]; then
            status="prof_failed"
            echo "  $label FAIL (profiled run: exit $rc, see ${d#$REPO/}/run.log)"
        elif ! collector_export "$COLLECTOR" "$d/trace" "$d"; then
            status="export_failed"
            echo "  $label FAIL (collector export, see ${d#$REPO/}/export.log)"
        else
            prof_note=" prof_roi=$(roi_seconds "$d")s"
        fi
    fi
    echo "status=$status" >> "$out/run_meta.txt"
    [ "$status" = ok ] && echo "  $label ok   roi=${roi_s}s${prof_note}  raw=${out#$REPO/}"
    [ "$status" = ok ]
}

# ---------------------------------------------------------------- main
# engine_setup: environment script, platform probe (inside the clean environment the runs
# use), collector choice, run id. Separate from engine_main so the conformance runner can
# reuse it without the case tables.
engine_setup() {
    command -v python3 >/dev/null 2>&1 || engine_die "python3 not on PATH"
    ENV_SCRIPT_ABS=""
    if [ -n "$ENV_SCRIPT" ]; then
        case "$ENV_SCRIPT" in /*) ENV_SCRIPT_ABS="$ENV_SCRIPT" ;; *) ENV_SCRIPT_ABS="$REPO/$ENV_SCRIPT" ;; esac
        [ -f "$ENV_SCRIPT_ABS" ] || engine_die "environment script not found: $ENV_SCRIPT (pass --env-script)"
    fi
    local probe_log; probe_log="$(mktemp)"
    run_clean "$probe_log" "$REPO" -- python3 "$TOOLS/probes/device.py"
    DEVICE_JSON="$(tail -1 "$probe_log")"; rm -f "$probe_log"
    PLATFORM_ID="$(printf '%s' "$DEVICE_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["device"]["platform_id"])' 2>/dev/null)"
    [ -n "$PLATFORM_ID" ] || engine_die "device probe failed: $DEVICE_JSON"
    if [ "$COLLECTOR" = auto ]; then
        case "$PLATFORM_ID" in
            nvidia-*) if collector_available nvidia_nsys; then COLLECTOR=nvidia_nsys; else COLLECTOR=none; fi ;;
            *)        COLLECTOR=none ;;
        esac
    fi
    [ "$PROFILED_RUNS" -gt 0 ] || COLLECTOR=none
    if [ "$COLLECTOR" != none ] && [ "$DRY_RUN" != 1 ]; then
        collector_available "$COLLECTOR" || engine_die "collector $COLLECTOR is not available here"
    fi
    COLLECTOR_VERSION="$(collector_version "$COLLECTOR")"
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
}

engine_main() {
    local -a resolve=(python3 "$TOOLS/cases.py" resolve --level "$LEVEL" --backend "$BACKEND")
    [ -n "$BUILD_ROOT" ] && resolve+=(--build-root "$BUILD_ROOT")
    local rows
    rows="$("${resolve[@]}" "$@")" || exit 2
    [ -n "$rows" ] || engine_die "no case selected"
    engine_setup

    local n; n=$(printf '%s\n' "$rows" | wc -l)
    echo "measure_level${LEVEL}: run_id=$RUN_ID platform=$PLATFORM_ID collector=$COLLECTOR cases=$n" \
         "protocol=warmup:$WARMUP_RUNS,clean:$CLEAN_RUNS,profiled:$PROFILED_RUNS skip_verify=$SKIP_VERIFY"
    if [ "$COLLECTOR" = none ] && [ "$PROFILED_RUNS" -gt 0 ]; then
        echo "measure_level${LEVEL}: no profiler for $PLATFORM_ID -- ROI time and FOM only, device columns will be null"
    fi

    local rc_all=0
    local -a f
    while IFS=$'\t' read -r -a f; do
        [ "${#f[@]}" -eq 17 ] || engine_die "malformed case row (${#f[@]} fields)"
        measure_case "${f[@]}" < /dev/null || rc_all=1
    done <<< "$rows"
    return $rc_all
}
