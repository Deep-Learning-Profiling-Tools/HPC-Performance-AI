#!/usr/bin/env bash
# Registered-input selector for run.sh scripts (sourced by level2/tools/hpcperf_launch_common.sh and
# level3/tools/l3_common.sh; a run.sh may also source it directly).
# hpcperf_apply_input <bench_dir> <SELECTOR_VAR>
#   Registered-input selector shared by the run.sh scripts (tools/inputs/README.md). When the
#   selector variable is set, the input's registry entry supplies environment knobs (exported
#   here) and extra command-line arguments (left in HPCPERF_INPUT_ARGS for the caller to prepend
#   with: set -- "${HPCPERF_INPUT_ARGS[@]}" "$@"). A knob the registry sets that is ALREADY in the
#   environment with another value is a conflict (nothing is silently overridden -> exit 2), an
#   unknown id fails (exit 2), and without the selector nothing changes.
HPCPERF_INPUT_ARGS=()
hpcperf_apply_input() {
    local bench_dir="$1" var="$2" id="${!2:-}" tool root
    [ -n "$id" ] || return 0
    root="$(cd "$bench_dir/../.." && pwd)"; tool="$root/tools/inputs/hpcperf_inputs.py"
    # the registry reader needs the repository's python3 + pyyaml: source the (idempotent) environment
    # script when the calling shell has not done so yet (some run.sh only source it later or never).
    if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$root" ] && [ -f "$root/hpcperf_env.sh" ]; then
        local _sel_opts; _sel_opts="$(set +o)"; set +eu
        # shellcheck disable=SC1091
        source "$root/hpcperf_env.sh" 2>/dev/null; eval "$_sel_opts"
    fi
    local lines; lines="$(python3 "$tool" shell-env "$bench_dir" "$id")" || { echo "run.sh: $var=$id is not a registered input of $(basename "$bench_dir")" >&2; return 2; }
    local kind k v
    while IFS=$'\t' read -r kind k v; do
        case "$kind" in
            E) if [ -n "${!k+x}" ] && [ "${!k}" != "$v" ]; then
                   echo "run.sh: $var=$id sets $k=$v but $k=${!k} is already set -- a registered input is mutually exclusive with the knobs it defines" >&2; return 2
               fi
               export "$k=$v" ;;
            A) HPCPERF_INPUT_ARGS+=("$k") ;;
        esac
    done <<< "$lines"
    export HPCPERF_INPUT_ID="$id"
    return 0
}
