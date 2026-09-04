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
#      set, i.e. srun --gpus-per-task), the scheduler's result is respected.
#   3. If every rank inherits the full CUDA_VISIBLE_DEVICES list (K entries),
#      the rank takes the (local rank)-th entry. local rank >= K is a FAILURE
#      (exit 12) -- that would silently share GPUs between ranks.
#   4. If CUDA_VISIBLE_DEVICES is unset, the ordinal = local rank is used;
#      local rank >= device count is likewise a FAILURE.
#   5. Only HPCPERF_ALLOW_OVERSUBSCRIBE=1 downgrades the failure to modulo
#      reuse, with a loud warning (debug only).
#   6. GPU mode with no usable GPU (no devices, or CUDA_VISIBLE_DEVICES
#      explicitly empty) is a FAILURE (exit 13) unless HPCPERF_ALLOW_NO_GPU=1.
#   7. CUDA_DEVICE_ORDER=PCI_BUS_ID is exported (unless already set) so that a
#      numeric CUDA_VISIBLE_DEVICES ordinal means the same device to CUDA and
#      to nvidia-smi (both then enumerate in PCI bus order); the expected UUID
#      is looked up through nvidia-smi's own index/uuid table, never by
#      treating a CUDA ordinal as a physical index blindly.
#
# Audit (unless HPCPERF_BIND_QUIET=1): one line per rank to stderr, and
# appended to $HPCPERF_BIND_LOG if set:
#   hpcperf-bind: host= grank= lrank= pid= mode=<bound|app-managed|scheduler>
#                 expected_gpu=<entry> expected_uuid=<GPU-...|unknown>
#                 observed=unverified cpus=<Cpus_allowed_list>
# "expected" is what this wrapper arranged from the local rank; "observed" is
# NOT known here -- hpcperf_mpi_launch.sh fills it in from nvidia-smi
# compute-apps sampling and reports verified / MISMATCH / unverified.

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
fail_nogpu() {
    if [ "${HPCPERF_ALLOW_NO_GPU:-0}" = "1" ]; then
        echo "mpi_gpu_bind.sh: WARNING: no usable GPU for local rank $lr ($1); continuing because HPCPERF_ALLOW_NO_GPU=1." >&2
        return 0
    fi
    echo "mpi_gpu_bind.sh: ERROR: GPU mode but no usable GPU for local rank $lr ($1)." >&2
    exit 13
}

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

_sel=""      # selected entry (CUDA ordinal or UUID) in the pre-narrowing view
_mode="bound"
[ "${HPCPERF_BIND_REPORT_ONLY:-0}" = "1" ] && _mode="app-managed"

if [ -n "${SLURM_GPUS_PER_TASK:-}" ]; then
    _sel="${CUDA_VISIBLE_DEVICES:-}"; _mode="scheduler"
    [ -n "$_sel" ] || fail_nogpu "scheduler per-task binding left CUDA_VISIBLE_DEVICES empty"
elif [ "${CUDA_VISIBLE_DEVICES+set}" = "set" ]; then
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        fail_nogpu "CUDA_VISIBLE_DEVICES is explicitly empty (scheduler hid all GPUs)"
        _sel="<none>"
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
        [ "$_mode" = bound ] && export CUDA_VISIBLE_DEVICES="$_sel"
    fi
else
    _n="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
    if [ "${_n:-0}" -le 0 ]; then
        fail_nogpu "nvidia-smi lists no GPUs and CUDA_VISIBLE_DEVICES is unset"
        _sel="<none>"
    else
        if [ "$lr" -ge "$_n" ]; then
            if [ "${HPCPERF_ALLOW_OVERSUBSCRIBE:-0}" = "1" ]; then
                oversub_warn "local rank $lr reuses GPU ordinal $((lr % _n)) of $_n"
            else
                fail_shared "only $_n GPUs on this node"
            fi
        fi
        _sel=$(( lr % _n ))
        [ "$_mode" = bound ] && export CUDA_VISIBLE_DEVICES="$_sel"
    fi
fi

if [ "${HPCPERF_BIND_QUIET:-0}" != "1" ]; then
    # Expected UUID: a UUID entry is itself; a numeric entry is an ordinal in
    # PCI-bus order (CUDA_DEVICE_ORDER=PCI_BUS_ID above) -> nvidia-smi's own
    # index column, which is also PCI-ordered over the same visible set.
    case "$_sel" in
        GPU-*|MIG-*) _uuid="$_sel" ;;
        ''|*[!0-9]*) _uuid="unknown" ;;
        *) _uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null \
                    | tr -d ' ' | awk -F, -v i="$_sel" '$1==i {print $2; exit}')"
           _uuid="${_uuid:-unknown}" ;;
    esac
    _cpus="$(sed -n 's/^Cpus_allowed_list:[[:space:]]*//p' /proc/self/status 2>/dev/null)"
    _line="hpcperf-bind: host=$(hostname -s) grank=$gr lrank=$lr pid=$$ mode=$_mode expected_gpu=$_sel expected_uuid=$_uuid observed=unverified cpus=${_cpus:-unknown}"
    echo "$_line" >&2
    [ -n "${HPCPERF_BIND_LOG:-}" ] && echo "$_line" >> "$HPCPERF_BIND_LOG"
fi
exec "$@"
