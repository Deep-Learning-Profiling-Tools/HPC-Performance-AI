#!/usr/bin/env bash
# measure_level1.sh -- measure the runtime of Level 1 benchmarks.
#
#   tools/timing/measure_level1.sh --build-root build/all daxpy
#   tools/timing/measure_level1.sh --build-root build/all --repeats 3 all
#
# What it does, per benchmark, from the case table tools/timing/cases.tsv:
#   1. one warm-up run (discarded)
#   2. N un-instrumented runs, wall clock each   -> the honest total time
#   3. one nsys run + nsys stats CSV reports     -> the GPU/host decomposition
# Raw artifacts go to <raw-root>/<benchmark>/<run-id>/; tools/timing/summarize.py
# turns each of those directories into one JSON and the aggregate CSV.
#
# Why two kinds of run: most Level 1 cases finish in under two seconds while nsys
# attach/flush costs about a second, so a single profiled run would distort the
# headline wall clock. GPU-side durations are timestamped on the device by CUPTI
# and are far less sensitive, so ratios come from the profiled run and absolutes
# from the clean ones. Both wall clocks end up in the JSON.
#
# Verification is skipped by default (HPCPERF_SKIP_VERIFY=1): these benchmarks
# recompute the GPU result on one CPU core afterwards, which is not part of what
# we want to time. --keep-verify measures the default path instead; the two modes
# must agree on GPU time and differ only in wall clock.
#
# The benchmark and nsys run under a minimal `env -i` allow-list. That is not
# cosmetic: nsys stores the complete process environment in the report
# (TARGET_INFO_SYSTEM_ENV / DeviceEnvironment -- measured: 368 variables, 16 kB,
# including session tokens and SSH_* on this node).
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CASES="$HERE/cases.tsv"

BUILD_ROOT=""
REPEATS=5
WARMUP=1
PROFILE=1
KEEP_VERIFY=0
DRY_RUN=0
RAW_ROOT="$REPO/build/timing"
SELECT=""

die() { echo "measure_level1: $*" >&2; exit 2; }

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --build-root) BUILD_ROOT="${2:?}"; shift 2 ;;
        --repeats)    REPEATS="${2:?}"; shift 2 ;;
        --warmup)     WARMUP="${2:?}"; shift 2 ;;
        --raw-root)   RAW_ROOT="${2:?}"; shift 2 ;;
        --no-profile) PROFILE=0; shift ;;
        --keep-verify) KEEP_VERIFY=1; shift ;;
        --dry-run)    DRY_RUN=1; shift ;;
        -h|--help)    usage ;;
        -*)           die "unknown option '$1' (try --help)" ;;
        *)            SELECT="$1"; shift ;;
    esac
done

[ -n "$SELECT" ] || die "give a benchmark name or 'all' (try --help)"
[ -f "$CASES" ] || die "case table missing: $CASES (run gen_cases.py)"
case "$REPEATS" in ''|*[!0-9]*) die "--repeats must be a number" ;; esac
case "$WARMUP"  in ''|*[!0-9]*) die "--warmup must be a number" ;; esac

# Build root: explicit, else the first candidate that carries the case table's layout.
if [ -z "$BUILD_ROOT" ]; then
    for cand in "$REPO/build/all" "$REPO/build/gcc13"; do
        [ -d "$cand/level1" ] && { BUILD_ROOT="$cand"; break; }
    done
fi
[ -n "$BUILD_ROOT" ] || die "no build tree found; pass --build-root"
BUILD_ROOT="$(cd "$BUILD_ROOT" && pwd)" || die "bad --build-root"

command -v nsys >/dev/null 2>&1 || { [ "$PROFILE" = 1 ] && die "nsys not on PATH (or pass --no-profile)"; }

# ---------------------------------------------------------------- clean environment
# Only these names reach the benchmark and nsys. Everything else -- including the
# login shell's tokens -- is dropped, so it cannot land in the .nsys-rep.
ENV_ALLOW="PATH HOME USER LOGNAME SHELL TERM LANG LC_ALL TMPDIR TZ \
           LD_LIBRARY_PATH CUDA_HOME CUDA_PATH CUDA_VISIBLE_DEVICES CUDA_CACHE_PATH \
           CUDA_DEVICE_ORDER CUDA_MODULE_LOADING GPU_DEVICE_ORDINAL \
           NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES"

clean_env_args() {   # prints NAME=VALUE for every allow-listed name that is set
    local n v
    for n in $ENV_ALLOW; do
        eval "v=\${$n+set}"
        [ "${v:-}" = set ] && eval "printf '%s=%s\n' \"$n\" \"\$$n\""
    done
    [ "$KEEP_VERIFY" = 1 ] || printf 'HPCPERF_SKIP_VERIFY=%s\n' 1
}

run_clean() {        # run_clean <logfile> <cmd...>; returns the command's exit code
    local log="$1"; shift
    local -a envargs=()
    while IFS= read -r line; do envargs+=("$line"); done < <(clean_env_args)
    env -i "${envargs[@]}" "$@" > "$log" 2>&1
}

now_ns() { date +%s%N; }

subst() {            # expand the {REPO}/{BUILD} placeholders of the case table ("-" = empty)
    local s="$1"
    [ "$s" = "-" ] && { printf ''; return; }
    s="${s//\{REPO\}/$REPO}"
    s="${s//\{BUILD\}/$BUILD_ROOT}"
    printf '%s' "$s"
}

# ---------------------------------------------------------------- provenance
gpu_props() {
    nvidia-smi --query-gpu=index,name,uuid,driver_version,compute_cap,memory.total,\
clocks.max.sm,clocks.max.mem,clocks.sm,clocks.mem,power.limit,temperature.gpu,persistence_mode \
        --format=csv,noheader,nounits 2>/dev/null | head -1
}

write_meta() {       # write_meta <dir> <benchmark> <exe> <args> <cwd> <timeout> <wrapper>
    local d="$1" bm="$2" exe="$3" args="$4" cwd="$5" tmo="$6" wrap="$7"
    {
        echo "schema=hpcperf-timing-raw-1"
        echo "run_id=$RUN_ID"
        echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "benchmark=$bm"
        echo "backend=CUDA"
        echo "exe=$exe"
        echo "args=$args"
        echo "cwd=$cwd"
        echo "ctest_timeout_s=$tmo"
        echo "ctest_wrapper=$wrap"
        echo "skip_verify=$([ "$KEEP_VERIFY" = 1 ] && echo 0 || echo 1)"
        echo "repeats=$REPEATS"
        echo "warmup=$WARMUP"
        echo "profiled=$PROFILE"
        echo "exe_sha256=$(sha256sum "$exe" 2>/dev/null | cut -d' ' -f1)"
        echo "exe_bytes=$(stat -c %s "$exe" 2>/dev/null)"
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
    } > "$d/run_meta.txt"
}

# ---------------------------------------------------------------- one benchmark
measure_one() {
    local bm="$1" exe_t="$2" args_t="$3" cwd_t="$4" tmo="$5" wrap="$6"
    local exe args cwd
    exe="$(subst "$exe_t")"; args="$(subst "$args_t")"; cwd="$(subst "$cwd_t")"

    if [ ! -x "$exe" ]; then
        printf '  %-26s SKIP (not built: %s)\n' "$bm" "${exe#$BUILD_ROOT/}"
        return 0
    fi

    local out="$RAW_ROOT/$bm/$RUN_ID"
    # shellcheck disable=SC2206
    local -a argv=("$exe" $args)

    if [ "$DRY_RUN" = 1 ]; then
        printf '  %-26s\n' "$bm"
        printf '    cwd     %s\n' "$cwd"
        printf '    env     env -i %s%s\n' "$(clean_env_args | cut -d= -f1 | tr '\n' ' ')" ""
        printf '    timed   %s\n' "${argv[*]}"
        [ "$PROFILE" = 1 ] && printf '    nsys    nsys profile -o %s/%s ... -- %s\n' "$out" "$bm" "${argv[*]}"
        return 0
    fi

    mkdir -p "$out" || return 1
    write_meta "$out" "$bm" "$exe" "$args" "$cwd" "$tmo" "$wrap"
    ( cd "$cwd" ) || { echo "  $bm: cwd missing: $cwd" >&2; return 1; }

    local i t0 t1 rc
    # warm-up (discarded: first touch pays CUDA context creation and page faults)
    i=0
    while [ "$i" -lt "$WARMUP" ]; do
        ( cd "$cwd" && run_clean "$out/warmup.$i.log" "${argv[@]}" )
        i=$((i + 1))
    done

    : > "$out/wall_ns.txt"
    : > "$out/exit_codes.txt"
    i=0
    while [ "$i" -lt "$REPEATS" ]; do
        t0="$(now_ns)"
        ( cd "$cwd" && run_clean "$out/run.$i.log" "${argv[@]}" ); rc=$?
        t1="$(now_ns)"
        echo "$((t1 - t0))" >> "$out/wall_ns.txt"
        echo "$rc" >> "$out/exit_codes.txt"
        if [ "$rc" -ne 0 ]; then
            printf '  %-26s FAIL (exit %s on repeat %s, see %s)\n' "$bm" "$rc" "$i" "$out/run.$i.log"
            echo "failed_repeat=$i" >> "$out/run_meta.txt"
            return 1
        fi
        i=$((i + 1))
    done

    if [ "$PROFILE" = 1 ]; then
        mkdir -p "$out/nsys"
        t0="$(now_ns)"
        ( cd "$cwd" && run_clean "$out/nsys/profile.log" \
            nsys profile -o "$out/nsys/$bm" -f true -x true \
                 -t cuda -s process-tree --cpuctxsw=process-tree \
                 --cuda-memory-usage=true --stats=false -- "${argv[@]}" ); rc=$?
        t1="$(now_ns)"
        echo "$((t1 - t0))" > "$out/wall_ns_profiled.txt"
        if [ "$rc" -ne 0 ] || [ ! -f "$out/nsys/$bm.nsys-rep" ]; then
            printf '  %-26s WARN nsys run failed (exit %s), wall clock kept\n' "$bm" "$rc"
            echo "nsys_status=failed" >> "$out/run_meta.txt"
        else
            nsys stats --format csv --output "$out/nsys/rep" \
                --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
                --report cuda_gpu_mem_size_sum --report cuda_api_sum \
                --report cuda_gpu_trace "$out/nsys/$bm.nsys-rep" > "$out/nsys/stats.log" 2>&1
            if [ $? -ne 0 ]; then
                printf '  %-26s WARN nsys stats failed, see %s\n' "$bm" "$out/nsys/stats.log"
                echo "nsys_status=stats_failed" >> "$out/run_meta.txt"
            else
                echo "nsys_status=ok" >> "$out/run_meta.txt"
            fi
        fi
    fi

    local median
    median="$(sort -n "$out/wall_ns.txt" | awk '{a[NR]=$1} END {printf "%.3f", (NR%2 ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2)/1e9}')"
    printf '  %-26s ok   wall_median=%ss  raw=%s\n' "$bm" "$median" "${out#$REPO/}"
    return 0
}

# ---------------------------------------------------------------- main
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
echo "measure_level1: build_root=$BUILD_ROOT run_id=$RUN_ID repeats=$REPEATS warmup=$WARMUP profile=$PROFILE skip_verify=$([ "$KEEP_VERIFY" = 1 ] && echo 0 || echo 1)"

rc_all=0
found=0
while IFS=$'\t' read -r bm exe args cwd tmo wrap pass_re; do
    case "$bm" in ''|\#*) continue ;; esac
    if [ "$SELECT" != "all" ] && [ "$SELECT" != "$bm" ]; then continue; fi
    found=1
    measure_one "$bm" "$exe" "$args" "$cwd" "$tmo" "$wrap" || rc_all=1
done < "$CASES"

[ "$found" = 1 ] || die "no case named '$SELECT' in $CASES"
exit "$rc_all"
