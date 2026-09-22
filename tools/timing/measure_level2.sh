#!/usr/bin/env bash
# measure_level2.sh -- measure the runtime of Level 2 mini-applications.
#
#   tools/timing/measure_level2.sh xsbench
#   tools/timing/measure_level2.sh all
#   tools/timing/measure_level2.sh --dry-run all
#
# One profiled run per application, from the case table tools/timing/cases_l2.tsv:
#
#   nsys profile ... -- bash level2/<app>/run.sh <backend>
#
# and nothing else. No warm-up, no repeats, no validate.sh. That is deliberate and
# it is what the measurement protocol asks for; the consequences are recorded in
# every JSON rather than hidden:
#
#   * The wall clock INCLUDES profiler overhead. Measured on quicksilver: 9.29 s
#     clean vs 15.4-17.7 s profiled (1.66x-1.91x). There is no clean baseline to
#     subtract it from, so wall_s is an upper bound, not the application's time.
#   * The application's own FOM is depressed by the profiler (quicksilver: 5.631e6
#     clean vs 5.27-5.33e6 profiled, -5.4%..-6.4%). Every JSON therefore carries
#     fom_from_profiled_run=true.
#   * GPU-side numbers are NOT affected: CUPTI timestamps kernels on the device.
#     Across -s none / -s process-tree / +cpuctxsw the kernel total moved 1.2%
#     (3.6839 / 3.6745 / 3.6406 s) with an identical launch count (411).
#
# Why nsys wraps run.sh from the OUTSIDE: 20 of 24 run.sh end in `exec`, so there
# is no post-run hook to attach to, and a profiler placed inside the launcher's
# wrapper breaks its GPU-binding audit (mpi_gpu_bind.sh logs pid=$$ and
# hpcperf_mpi_launch.sh joins that pid against nvidia-smi compute-apps; nsys would
# own that pid and the app would get another, making every rank "unverified").
# Wrapping from outside keeps the audit intact -- verified on quicksilver:
# "1 verified, 0 mismatch, 0 unverified" with nsys as an ancestor process.
#
# Verification is not run at all: Level 2 keeps it in validate.sh, which this tool
# never calls. Unlike Level 1 no source change was needed for that.
#
# The run happens under a minimal `env -i` allow-list, then the repo's environment
# script is sourced inside that clean environment to rebuild what the applications
# need. That is not cosmetic: nsys stores the whole process environment in the
# report (TARGET_INFO_SYSTEM_ENV / DeviceEnvironment). Measured on this node:
# inheriting the login environment gives 349 variables / 13187 chars including
# CLAUDE_CODE* (9 hits), SSH_CONNECTION and SSH_CLIENT; the allow-list gives 101
# variables / 8390 chars with none of them.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CASES="$HERE/cases_l2.tsv"

ENV_SCRIPT="${HPCPERF_TIMING_ENV_SCRIPT:-hpcperf_env.sh}"
PROFILE=1
DRY_RUN=0
RAW_ROOT="$REPO/build/timing-l2"
SELECT=""
GPUS_OVERRIDE=""

die() { echo "measure_level2: $*" >&2; exit 2; }

usage() {
    sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --env-script) ENV_SCRIPT="${2:?}"; shift 2 ;;
        --raw-root)   RAW_ROOT="${2:?}"; shift 2 ;;
        --gpus)       GPUS_OVERRIDE="${2:?}"; shift 2 ;;
        --no-profile) PROFILE=0; shift ;;
        --dry-run)    DRY_RUN=1; shift ;;
        -h|--help)    usage ;;
        -*)           die "unknown option '$1' (try --help)" ;;
        *)            SELECT="$1"; shift ;;
    esac
done

[ -n "$SELECT" ] || die "give an application name or 'all' (try --help)"
[ -f "$CASES" ] || die "case table missing: $CASES"
[ -f "$REPO/$ENV_SCRIPT" ] || [ -f "$ENV_SCRIPT" ] \
    || die "environment script not found: $ENV_SCRIPT (pass --env-script)"

# ---------------------------------------------------------------- clean environment
# Only these names reach nsys and the application. The SLURM_* ones are required:
# hpcperf_mpi_launch.sh reads the allocation through them (and through `scontrol -d
# $SLURM_JOB_ID`), and they cannot be rebuilt by sourcing the environment script.
ENV_ALLOW="PATH HOME USER LOGNAME SHELL TERM LANG LC_ALL TMPDIR TZ \
           LD_LIBRARY_PATH CUDA_HOME CUDA_PATH CUDA_VISIBLE_DEVICES CUDA_CACHE_PATH \
           CUDA_DEVICE_ORDER CUDA_MODULE_LOADING GPU_DEVICE_ORDINAL \
           NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES \
           SLURM_JOB_ID SLURM_JOB_NODELIST SLURM_JOB_NUM_NODES SLURM_JOB_GPUS \
           SLURM_GPUS_ON_NODE SLURM_GPUS_PER_TASK SLURM_CPUS_ON_NODE \
           SLURM_JOB_CPUS_PER_NODE SLURM_TASKS_PER_NODE SLURM_LOCALID SLURM_PROCID"

# A name matching this never leaves the parent, even if it is on the allow-list
# above or gets added to it later. The deny rule beats the allow-list on purpose.
ENV_DENY='(TOKEN|SECRET|PASSWD|PASSWORD|CREDENTIAL|PRIVATE_KEY|API_KEY|SESSION)'

DENIED=""

clean_env_args() {   # prints NAME=VALUE for every allow-listed name that is set
    local n v
    DENIED=""
    for n in $ENV_ALLOW; do
        if printf '%s' "$n" | /usr/bin/grep -qE "$ENV_DENY"; then
            DENIED="$DENIED $n"
            continue
        fi
        eval "v=\${$n+set}"
        [ "${v:-}" = set ] && eval "printf '%s=%s\n' \"$n\" \"\$$n\""
    done
}

# Runs <cmd...> with only the allow-listed variables, after sourcing the repo's
# environment script inside that clean environment. The script's own chatter goes
# to the log; a failure to source is fatal for the run.
run_clean() {        # run_clean <logfile> <extra NAME=VALUE ...> -- <cmd...>
    local log="$1"; shift
    local -a envargs=() extra=()
    while IFS= read -r line; do envargs+=("$line"); done < <(clean_env_args)
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do extra+=("$1"); shift; done
    [ $# -gt 0 ] || die "run_clean: missing the -- separator before the command"
    shift
    env -i "${envargs[@]}" "${extra[@]}" \
        bash -c 'cd "$1" || exit 2; shift; . "./$1" >/dev/null 2>&1 || exit 3; shift; exec "$@"' \
        _ "$REPO" "$ENV_SCRIPT" "$@" > "$log" 2>&1
}

now_ns() { date +%s%N; }

# ---------------------------------------------------------------- provenance
gpu_props() {
    nvidia-smi --query-gpu=index,name,uuid,driver_version,compute_cap,memory.total,\
clocks.max.sm,clocks.max.mem,clocks.sm,clocks.mem,power.limit,temperature.gpu,persistence_mode \
        --format=csv,noheader,nounits 2>/dev/null | head -1
}

write_meta() {       # write_meta <dir> <app> <backend> <gpus> <timeout> <fom fields...>
    local d="$1" app="$2" be="$3" gpus="$4" tmo="$5"
    local fname="$6" funit="$7" fbetter="$8" fsrc="$9" frx="${10}" fnote="${11}"
    {
        echo "schema=hpcperf-timing-raw-1"
        echo "level=2"
        echo "run_id=$RUN_ID"
        echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "benchmark=$app"
        echo "backend=$be"
        echo "runner=level2/$app/run.sh"
        echo "gpus=$gpus"
        echo "timeout_s=$tmo"
        echo "skip_verify=0"
        echo "verify_run=0"
        echo "repeats=1"
        echo "warmup=0"
        echo "profiled=$PROFILE"
        echo "profiler_in_wall=$PROFILE"
        echo "fom_name=$fname"
        echo "fom_unit=$funit"
        echo "fom_better=$fbetter"
        echo "fom_source=$fsrc"
        echo "fom_regex=$frx"
        echo "fom_note=$fnote"
        echo "env_script=$ENV_SCRIPT"
        echo "hostname=$(hostname -s)"
        echo "kernel=$(uname -r)"
        echo "cpu_model=$(sed -n 's/^model name[ \t]*: //p' /proc/cpuinfo | head -1)"
        echo "cpu_allowed=$(nproc)"
        echo "loadavg_1m=$(cut -d' ' -f1 /proc/loadavg)"
        echo "mem_available_kb=$(sed -n 's/^MemAvailable:[ \t]*\([0-9]*\).*/\1/p' /proc/meminfo)"
        echo "nsys_version=$(nsys --version 2>/dev/null | sed -n 's/.*version //p' | head -1)"
        echo "nvcc_version=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)"
        echo "gpu_csv=$(gpu_props)"
        echo "git_commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null)"
        echo "git_dirty=$(test -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" && echo 1 || echo 0)"
        echo "env_allow=$(echo $ENV_ALLOW | tr -s ' ')"
        echo "env_deny_regex=$ENV_DENY"
        [ -n "$DENIED" ] && echo "env_denied=$(echo $DENIED | tr -s ' ')"
    } > "$d/run_meta.txt"
}

# ---------------------------------------------------------------- one application
measure_one() {
    local app="$1" be="$2" gpus="$3" tmo="$4"
    local fname="$5" funit="$6" fbetter="$7" fsrc="$8" frx="$9" fnote="${10}"
    [ -n "$GPUS_OVERRIDE" ] && gpus="$GPUS_OVERRIDE"

    local runner="$REPO/level2/$app/run.sh"
    if [ ! -f "$runner" ]; then
        printf '  %-16s SKIP (no run.sh)\n' "$app"
        return 0
    fi

    local out="$RAW_ROOT/$app/$RUN_ID"
    local -a inner=(bash "level2/$app/run.sh" "$be")
    # No --cuda-memory-usage here, unlike measure_level1.sh. That option segfaults
    # MiniEM inside cudaFreeAsync/cuMemFreeAsync (exit 139 at 203 s, no FOM printed),
    # while the same run without it completes: 236 s, FOM 18277.2, and plain run.sh
    # with no profiler at all takes 110 s. It also contributes nothing here: with and
    # without it, all four reports this tool reads (cuda_gpu_kern_sum,
    # cuda_gpu_mem_time_sum, cuda_gpu_mem_size_sum, cuda_gpu_trace) have identical
    # shape and equivalent values -- it only adds memory-pool events we never read.
    local -a profiled=(nsys profile -o "$out/nsys/$app" -f true -x true
                       -t cuda -s process-tree --cpuctxsw=process-tree
                       --stats=false --)

    if [ "$DRY_RUN" = 1 ]; then
        printf '  %-16s gpus=%s timeout=%ss fom=%s\n' "$app" "$gpus" "$tmo" "$fname"
        printf '    env     env -i %s HPCPERF_GPUS=%s\n' \
               "$(clean_env_args | cut -d= -f1 | tr '\n' ' ')" "$gpus"
        printf '    source  %s (inside the clean environment)\n' "$ENV_SCRIPT"
        if [ "$PROFILE" = 1 ]; then
            printf '    run     timeout %s %s %s\n' "$tmo" "${profiled[*]}" "${inner[*]}"
        else
            printf '    run     timeout %s %s\n' "$tmo" "${inner[*]}"
        fi
        return 0
    fi

    mkdir -p "$out/nsys" || return 1
    write_meta "$out" "$app" "$be" "$gpus" "$tmo" \
               "$fname" "$funit" "$fbetter" "$fsrc" "$frx" "$fnote"

    local t0 t1 rc
    t0="$(now_ns)"
    if [ "$PROFILE" = 1 ]; then
        run_clean "$out/run.log" "HPCPERF_GPUS=$gpus" -- \
            timeout "$tmo" "${profiled[@]}" "${inner[@]}"
    else
        run_clean "$out/run.log" "HPCPERF_GPUS=$gpus" -- \
            timeout "$tmo" "${inner[@]}"
    fi
    rc=$?
    t1="$(now_ns)"
    echo "$((t1 - t0))" > "$out/wall_ns.txt"
    echo "$rc" > "$out/exit_codes.txt"

    if [ "$rc" -eq 124 ]; then
        printf '  %-16s FAIL (timeout after %ss, see %s)\n' "$app" "$tmo" "${out#$REPO/}/run.log"
        echo "run_status=timeout" >> "$out/run_meta.txt"
        return 1
    fi
    if [ "$rc" -ne 0 ]; then
        printf '  %-16s FAIL (exit %s, see %s)\n' "$app" "$rc" "${out#$REPO/}/run.log"
        echo "run_status=failed" >> "$out/run_meta.txt"
        return 1
    fi
    echo "run_status=ok" >> "$out/run_meta.txt"

    # The launcher's GPU-binding audit is required evidence that the ranks used the
    # GPUs we asked for, and that nsys did not break the join. Record it verbatim.
    local audit
    audit="$(/usr/bin/grep -oE 'audit summary: .*' "$out/run.log" | head -1)"
    [ -n "$audit" ] && echo "gpu_audit=$audit" >> "$out/run_meta.txt"

    if [ "$PROFILE" = 1 ]; then
        if [ ! -f "$out/nsys/$app.nsys-rep" ]; then
            printf '  %-16s WARN nsys produced no report, wall clock kept\n' "$app"
            echo "nsys_status=no_report" >> "$out/run_meta.txt"
        elif nsys stats --format csv --output "$out/nsys/rep" \
                --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
                --report cuda_gpu_mem_size_sum --report cuda_api_sum \
                --report cuda_gpu_trace "$out/nsys/$app.nsys-rep" \
                > "$out/nsys/stats.log" 2>&1; then
            echo "nsys_status=ok" >> "$out/run_meta.txt"
        else
            printf '  %-16s WARN nsys stats failed, see %s\n' "$app" "${out#$REPO/}/nsys/stats.log"
            echo "nsys_status=stats_failed" >> "$out/run_meta.txt"
        fi
    fi

    local wall
    wall="$(awk '{printf "%.2f", $1/1e9}' "$out/wall_ns.txt")"
    printf '  %-16s ok   wall=%ss  raw=%s\n' "$app" "$wall" "${out#$REPO/}"
    return 0
}

# ---------------------------------------------------------------- main
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
if [ "$PROFILE" = 1 ] && [ "$DRY_RUN" = 0 ]; then
    command -v nsys >/dev/null 2>&1 || die "nsys not on PATH (or pass --no-profile)"
fi
echo "measure_level2: run_id=$RUN_ID env_script=$ENV_SCRIPT profile=$PROFILE repeats=1 verify=none"

rc_all=0
found=0
while IFS=$'\t' read -r app be gpus tmo fname funit fbetter fsrc frx fnote; do
    case "$app" in ''|\#*) continue ;; esac
    if [ "$SELECT" != "all" ] && [ "$SELECT" != "$app" ]; then continue; fi
    found=1
    measure_one "$app" "$be" "$gpus" "$tmo" \
                "$fname" "$funit" "$fbetter" "$fsrc" "$frx" "$fnote" || rc_all=1
done < "$CASES"

[ "$found" = 1 ] || die "no case named '$SELECT' in $CASES"
exit "$rc_all"
