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
# After the cases, the engine runs tools/timing/summarize.py on this run (JSON per case,
# the per-level CSVs and the web page results/timing/report/) unless SUMMARIZE=0.

ENGINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$(cd "$ENGINE_DIR/.." && pwd)"
REPO="$(cd "$TOOLS/../.." && pwd)"
# shellcheck source=collectors.sh
. "$ENGINE_DIR/collectors.sh"

: "${LEVEL:?}" "${CLEAN_RUNS:=1}" "${WARMUP_RUNS:=0}" "${PROFILED_RUNS:=1}"
: "${RAW_ROOT:=$REPO/build/timing}" "${COLLECTOR:=auto}" "${ENV_SCRIPT:=}" "${BUILD_ROOT:=}"
: "${BACKEND:=CUDA}" "${SKIP_VERIFY:=0}" "${DRY_RUN:=0}" "${PROFILE_TIMEOUT_FACTOR:=3}"
: "${SUMMARIZE:=1}" "${RESULTS_ROOT:=$REPO/results/timing}" "${REGISTRY:=0}"
# HPCPERF_ROI_LOG is derived from RAW_ROOT and read by processes that run in another cwd
# (run.sh changes into its run directory): a relative root would silently lose every log.
case "$RAW_ROOT" in /*) ;; *) RAW_ROOT="$PWD/$RAW_ROOT" ;; esac
case "$RESULTS_ROOT" in /*) ;; *) RESULTS_ROOT="$PWD/$RESULTS_ROOT" ;; esac

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
#         fom_source fom_regex roi_excludes verify_vs_roi notes input_id
measure_case() {
    local level="$1" app="$2" case="$3" backend="$4" gpus="$5" cwd="$6" tmo="$7" env="$8" argv="$9"
    local fom_name="${10}" fom_unit="${11}" fom_better="${12}" fom_source="${13}" fom_regex="${14}"
    local roi_excl="${15}" verify="${16}" notes="${17}" input_id="${18:--}"
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

    # A registry case measures one registered input: its workload identity (tools/inputs) is stored
    # with the raw runs, and a case whose identity cannot be established -- or whose executable
    # (a compile-time input's own build) does not exist -- is not run at all.
    local ident_tmp="" ident_status="-"
    if [ "$input_id" != "-" ]; then
        ident_tmp="$(mktemp)"
        python3 "$REPO/tools/inputs/hpcperf_inputs.py" identity "$REPO/level$level/$app" "$input_id" > "$ident_tmp" 2> "$ident_tmp.err"
        case $? in
            0) ident_status=ok ;;
            3) ident_status=incomplete ;;
            *) ident_status=failed ;;
        esac
    fi

    if [ "$DRY_RUN" = 1 ]; then
        echo "  $label"
        [ "$input_id" != "-" ] && echo "    input    $input_id (registry level$level/$app/inputs.yaml; workload identity: $ident_status)"
        echo "    cwd      $cwd"
        echo "    env      env -i $(_allowed_pairs | cut -d= -f1 | tr '\n' ' ')${run_env[*]:+${run_env[*]} }HPCPERF_ROI_LOG=<run dir>/roi"
        [ -n "$ENV_SCRIPT_ABS" ] && echo "    source   ${ENV_SCRIPT_ABS#$REPO/} (inside the clean environment)"
        echo "    runs     warmup=$WARMUP_RUNS clean=$CLEAN_RUNS profiled=$PROFILED_RUNS collector=$COLLECTOR"
        echo "    command  timeout $tmo ${cmd[*]}"
        [ -n "$ident_tmp" ] && rm -f "$ident_tmp" "$ident_tmp.err"
        return 0
    fi

    mkdir -p "$out" || return 1
    local pre_status=""
    if [ -n "$ident_tmp" ]; then
        mv "$ident_tmp" "$out/workload_identity.json"
        [ -s "$ident_tmp.err" ] && mv "$ident_tmp.err" "$out/workload_identity.err" || rm -f "$ident_tmp.err"
        case "$ident_status" in
            incomplete) pre_status="identity_incomplete" ;;
            failed)     pre_status="identity_failed" ;;
        esac
        if [ -z "$pre_status" ] && [ "$level" = 1 ] && [ ! -x "${cmd[0]}" ]; then
            pre_status="build_not_materialized"
        fi
    fi
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
        echo "input_id=$input_id"
        [ -f "$out/workload_identity.json" ] && echo "workload_identity_sha256=$(sha256sum "$out/workload_identity.json" | cut -d' ' -f1)"
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
    if [ -n "$pre_status" ]; then
        echo "status=$pre_status" >> "$out/run_meta.txt"
        echo "  $label FAIL ($pre_status: input $input_id not run; see ${out#$REPO/}/workload_identity.*)"
        return 1
    fi
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
    [ "$REGISTRY" = 1 ] && resolve+=(--registry)
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
        [ "${#f[@]}" -eq 18 ] || engine_die "malformed case row (${#f[@]} fields)"
        measure_case "${f[@]}" < /dev/null || rc_all=1
    done <<< "$rows"
    if [ "$DRY_RUN" != 1 ] && [ "$SUMMARIZE" = 1 ]; then
        echo "measure_level${LEVEL}: summarizing run $RUN_ID -> ${RESULTS_ROOT#$REPO/} (JSON, CSV, report/index.html)"
        python3 "$TOOLS/summarize.py" --raw-root "$RAW_ROOT" --out-root "$RESULTS_ROOT" --run-id "$RUN_ID" \
            || { echo "measure_level${LEVEL}: summarize failed (raw data kept in ${RAW_ROOT#$REPO/})"; rc_all=1; }
    fi
    return $rc_all
}
