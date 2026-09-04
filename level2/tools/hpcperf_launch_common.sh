#!/bin/bash
# hpcperf_launch_common.sh -- shared helpers for the Level 2 run.sh scripts.
# Source it (functions only). Every run.sh resolves its rank count through
# hpcperf_ranks so that the common parameters behave identically everywhere:
# a request the script cannot honour is an ERROR, never a silent 1-rank run.

HPCPERF_LAUNCHER_BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hpcperf_mpi_launch.sh"

# hpcperf_ranks <app> <scale_modes: yes|no>
#   Prints the rank count from HPCPERF_GPUS (or the legacy alias HPCPERF_NP;
#   both set and different -> error). 'all' is resolved against the current
#   allocation. If the app does not implement HPCPERF_SCALE_MODE (arg 2 = no),
#   a set HPCPERF_SCALE_MODE is an ERROR -- it is never silently ignored.
hpcperf_ranks() {
    local app=$1 modes=$2 n gpn
    if [ -n "${HPCPERF_GPUS:-}" ] && [ -n "${HPCPERF_NP:-}" ] \
       && [ "$HPCPERF_GPUS" != all ] && [ "$HPCPERF_GPUS" != "$HPCPERF_NP" ]; then
        echo "$app/run.sh: HPCPERF_GPUS=$HPCPERF_GPUS and HPCPERF_NP=$HPCPERF_NP disagree (one rank per GPU: they must match)" >&2
        return 2
    fi
    if [ "$modes" = no ] && [ -n "${HPCPERF_SCALE_MODE:-}" ]; then
        echo "$app/run.sh: HPCPERF_SCALE_MODE=$HPCPERF_SCALE_MODE is not implemented for $app (only HPCPERF_GPUS rank selection is); unset it -- see the per-app interface table in level2/README.md" >&2
        return 2
    fi
    n="${HPCPERF_GPUS:-${HPCPERF_NP:-1}}"
    if [ "$n" = all ]; then
        gpn="${SLURM_GPUS_ON_NODE:-}"
        [ -n "$gpn" ] || gpn="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
        n=$(( ${SLURM_JOB_NUM_NODES:-1} * ${gpn:-0} ))
        if [ "$n" -lt 1 ]; then echo "$app/run.sh: HPCPERF_GPUS=all but no GPUs detected" >&2; return 2; fi
    fi
    case "$n" in (''|*[!0-9]*) echo "$app/run.sh: HPCPERF_GPUS must be a positive integer or 'all' (got '$n')" >&2; return 2;; esac
    if [ "$n" -lt 1 ]; then echo "$app/run.sh: rank count must be >= 1" >&2; return 2; fi
    echo "$n"
}

# hpcperf_forbid_args <app> <flag>... -- <args>...
#   Extra command-line arguments must not override parameters this script has
#   already validated (topology, sizes, decks, device selection): the app
#   parsers take the last occurrence, which would bypass the checks. Matches
#   the exact flag and the flag=value form.
hpcperf_forbid_args() {
    local app=$1; shift
    local -a forbid=()
    while [ $# -gt 0 ] && [ "$1" != -- ]; do forbid+=("$1"); shift; done
    [ "${1:-}" = -- ] && shift
    local a f
    for a in "$@"; do
        for f in "${forbid[@]}"; do
            if [ "$a" = "$f" ] || [ "${a%%=*}" = "$f" ]; then
                echo "$app/run.sh: extra argument '$a' would override a validated parameter; use the HPCPERF_* variables instead (not allowed as extra args here: ${forbid[*]})" >&2
                return 2
            fi
        done
    done
    return 0
}

# hpcperf_topology <app> <N> [helper args...]
#   Prints "PX PY PZ" from hpcperf_topology.py, propagating its failure (a
#   plain `read <<< "$(cmd)"` would swallow the helper's exit status).
hpcperf_topology() {
    local app=$1 n=$2; shift 2
    local out
    out="$("$(dirname "$HPCPERF_LAUNCHER_BIN")/hpcperf_topology.py" "$n" "$@")" || {
        echo "$app/run.sh: no feasible process topology for $n ranks under the constraints above" >&2
        return 2
    }
    echo "$out"
}
