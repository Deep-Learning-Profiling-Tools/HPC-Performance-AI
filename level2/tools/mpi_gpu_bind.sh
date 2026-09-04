#!/bin/bash
# mpi_gpu_bind.sh -- scheduler-safe one-GPU-per-rank binding wrapper.
#
# Usage:  mpirun -np N .../level2/tools/mpi_gpu_bind.sh <exe> [args...]
#         (normally injected by level2/tools/hpcperf_mpi_launch.sh)
#
# Needed for programs that do NOT map ranks to GPUs themselves (MFEM apps such
# as Laghos/Remhos; flag-selected backends like CloverLeaf/TeaLeaf). hypre,
# Kokkos/Cabana apps and Branson bind by local rank on their own; for those use
# HPCPERF_BIND_REPORT_ONLY=1 (audit line, no CUDA_VISIBLE_DEVICES change).
#
# Policy (one rank = one GPU, never silently shared):
#   1. Local rank: OMPI_COMM_WORLD_LOCAL_RANK, else SLURM_LOCALID. Without
#      either (not an MPI child) the program is exec'd unchanged.
#   2. If the scheduler already did per-task GPU binding (SLURM_GPUS_PER_TASK
#      set, or srun --gpus-per-task), the scheduler's result is respected
#      unchanged.
#   3. If every rank inherits the full CUDA_VISIBLE_DEVICES list (K entries),
#      the rank takes the (local rank)-th entry. local rank >= K is a FAILURE
#      (exit 12) -- that would silently share GPUs between ranks.
#   4. If CUDA_VISIBLE_DEVICES is unset, the physical index = local rank is
#      used; local rank >= device count is likewise a FAILURE.
#   5. Only HPCPERF_ALLOW_OVERSUBSCRIBE=1 downgrades the failure to modulo
#      reuse, with a loud warning (debug only).
#   6. An explicitly empty CUDA_VISIBLE_DEVICES (scheduler hid all GPUs) is
#      respected: no binding, program exec'd unchanged.
#
# Audit: unless HPCPERF_BIND_QUIET=1, one line per rank goes to stderr (and is
# appended to $HPCPERF_BIND_LOG if set):
#   hpcperf-bind: host=<h> grank=<g> lrank=<l> gpu=<ordinal> uuid=<GPU-...>

lr="${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-}}"
gr="${OMPI_COMM_WORLD_RANK:-${SLURM_PROCID:-$lr}}"
if [ -z "$lr" ]; then
    exec "$@"
fi

oversub_warn() {
    echo "mpi_gpu_bind.sh: WARNING: DEBUG ONLY: GPU/rank oversubscription is enabled ($1)." >&2
}
fail_shared() {
    echo "mpi_gpu_bind.sh: ERROR: local rank $lr has no dedicated GPU ($1)." >&2
    echo "mpi_gpu_bind.sh: refusing to share GPUs between ranks; reduce the rank count," >&2
    echo "mpi_gpu_bind.sh: enlarge the allocation, or set HPCPERF_ALLOW_OVERSUBSCRIBE=1 (debug only)." >&2
    exit 12
}

_sel=""      # selected GPU ordinal (in the pre-narrowing enumeration), for the audit line
_narrow=1    # whether to export the narrowed CUDA_VISIBLE_DEVICES
[ "${HPCPERF_BIND_REPORT_ONLY:-0}" = "1" ] && _narrow=0  # app binds by itself

if [ -n "${SLURM_GPUS_PER_TASK:-}" ]; then
    # scheduler bound the GPU(s) per task already -- respect its result
    _sel="${CUDA_VISIBLE_DEVICES:-scheduler}"; _narrow=0
elif [ "${CUDA_VISIBLE_DEVICES+set}" = "set" ]; then
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        _sel="<none: scheduler hid all GPUs>"; _narrow=0
    else
        IFS=',' read -r -a _devs <<< "$CUDA_VISIBLE_DEVICES"
        _k=${#_devs[@]}
        if [ "$lr" -ge "$_k" ]; then
            if [ "${HPCPERF_ALLOW_OVERSUBSCRIBE:-0}" = "1" ]; then
                oversub_warn "local rank $lr reuses entry $((lr % _k)) of $_k visible GPUs"
            else
                fail_shared "only $_k GPUs in CUDA_VISIBLE_DEVICES for this node"
            fi
        fi
        _sel="${_devs[$(( lr % _k ))]}"
        [ "$_narrow" = 1 ] && export CUDA_VISIBLE_DEVICES="$_sel"
    fi
else
    _n="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
    if [ "${_n:-0}" -gt 0 ]; then
        if [ "$lr" -ge "$_n" ]; then
            if [ "${HPCPERF_ALLOW_OVERSUBSCRIBE:-0}" = "1" ]; then
                oversub_warn "local rank $lr reuses physical GPU $((lr % _n)) of $_n"
            else
                fail_shared "only $_n physical GPUs on this node"
            fi
        fi
        _sel=$(( lr % _n ))
    else
        _sel="$lr"
    fi
    [ "$_narrow" = 1 ] && export CUDA_VISIBLE_DEVICES="$_sel"
fi

if [ "${HPCPERF_BIND_QUIET:-0}" != "1" ]; then
    case "$_sel" in
        (*[!0-9]*|"") _uuid="n/a" ;;  # not a single numeric ordinal
        (*) _uuid="$(nvidia-smi -i "$_sel" --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1)" ;;
    esac
    _mode="bound"; [ "$_narrow" = 0 ] && _mode="app-managed"
    _line="hpcperf-bind: host=$(hostname -s) grank=$gr lrank=$lr gpu=$_sel ($_mode) uuid=${_uuid:-unknown}"
    echo "$_line" >&2
    [ -n "${HPCPERF_BIND_LOG:-}" ] && echo "$_line" >> "$HPCPERF_BIND_LOG"
fi
exec "$@"
