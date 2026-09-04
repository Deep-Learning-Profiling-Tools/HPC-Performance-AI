#!/bin/bash
# hpcperf_mpi_launch.sh -- unified GPU-count-aware MPI launcher for Level 2.
#
#   hpcperf_mpi_launch.sh [options] -- <exe> [args...]
#
# Options (environment variable defaults in parentheses):
#   --gpus N|all          ranks to launch, one rank per GPU   (HPCPERF_GPUS, def 1)
#   --launcher auto|mpirun|srun                               (HPCPERF_LAUNCHER)
#   --cpus-per-rank C     CPUs per rank (srun --cpus-per-task) (HPCPERF_CPUS_PER_RANK)
#   --bind wrapper|app|none  GPU binding mode (default wrapper):
#                           wrapper = narrow CUDA_VISIBLE_DEVICES per rank
#                           app     = app binds itself; wrapper only audits
#                           none    = no wrapper at all
#   --dry-run             print the resource plan and command, do not execute
#
# Policy:
#   * One MPI rank per GPU. Requested ranks > allocated GPUs fails fast unless
#     HPCPERF_ALLOW_OVERSUBSCRIBE=1 (then a loud DEBUG-ONLY warning).
#   * --gpus N uses exactly N GPUs even if the allocation has more; --gpus all
#     uses every allocated GPU.
#   * HPCPERF_NP is a backwards-compatibility alias: if HPCPERF_GPUS/--gpus is
#     also set and differs, that is an error.
#   * The transport and the mpirun/srun choice come from the site profile
#     (HPCPERF_SITE_PROFILE -> level2/tools/site/<profile>.sh), never from the
#     individual benchmark. Multi-node launches are refused while the site
#     profile marks multi-node BLOCKED.
#   * Slurm task slots smaller than the rank count (allocation made with -n 1)
#     are relaxed per-launch via --map-by ...:OVERSUBSCRIBE ONLY after the
#     GPU check passed -- this is Slurm slot bookkeeping, not GPU sharing; it
#     is logged. GPU oversubscription itself stays forbidden by default.
#   * Every rank logs "hpcperf-bind: host= grank= lrank= gpu= uuid=" (from
#     mpi_gpu_bind.sh) for an auditable rank->GPU mapping.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPUS="${HPCPERF_GPUS:-}"
LAUNCHER="${HPCPERF_LAUNCHER:-auto}"
CPUS_PER_RANK="${HPCPERF_CPUS_PER_RANK:-}"
BIND="wrapper"
DRYRUN="${HPCPERF_DRY_RUN:-0}"
while [ $# -gt 0 ]; do
    case "$1" in
        --gpus)          GPUS="$2"; shift 2 ;;
        --launcher)      LAUNCHER="$2"; shift 2 ;;
        --cpus-per-rank) CPUS_PER_RANK="$2"; shift 2 ;;
        --bind)          BIND="$2"; shift 2 ;;
        --dry-run)       DRYRUN=1; shift ;;
        --)              shift; break ;;
        *) echo "hpcperf_mpi_launch: unknown option '$1' (use -- before the program)" >&2; exit 2 ;;
    esac
done
[ $# -ge 1 ] || { echo "hpcperf_mpi_launch: no program given (usage: ... -- <exe> [args])" >&2; exit 2; }

say()  { echo "hpcperf-launch: $*" >&2; }
die()  { echo "hpcperf-launch: ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------- allocation
if [ -n "${SLURM_JOB_ID:-}" ]; then
    NNODES="${HPCPERF_NODES:-${SLURM_JOB_NUM_NODES:-1}}"
    NODELIST="$(scontrol show hostnames "${SLURM_JOB_NODELIST:-}" 2>/dev/null | tr '\n' ' ')"
    GPN="${HPCPERF_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-0}}"
    if [ "$GPN" = 0 ] && [ -n "${SLURM_JOB_GPUS:-}" ]; then
        GPN=$(( $(echo "$SLURM_JOB_GPUS" | tr -cd , | wc -c) + 1 ))
    fi
    # task slots per node: SLURM_TASKS_PER_NODE like "1" or "4(x2)"
    SLOTS="$(echo "${SLURM_TASKS_PER_NODE:-1}" | sed 's/(.*//')"
else
    NNODES="${HPCPERF_NODES:-1}"
    NODELIST="$(hostname -s)"
    GPN="${HPCPERF_GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')}"
    SLOTS=""   # not slot-limited outside Slurm
fi
[ "${GPN:-0}" -ge 1 ] || die "no GPUs detected in this allocation (SLURM_GPUS_ON_NODE/nvidia-smi)"
TOTAL_GPUS=$(( NNODES * GPN ))

# ---------------------------------------------------------------- rank count
if [ -n "${HPCPERF_NP:-}" ]; then
    if [ -n "$GPUS" ] && [ "$GPUS" != "all" ] && [ "$HPCPERF_NP" != "$GPUS" ]; then
        die "HPCPERF_NP=$HPCPERF_NP conflicts with HPCPERF_GPUS/--gpus=$GPUS (one rank per GPU: they must match)"
    fi
    [ -z "$GPUS" ] && GPUS="$HPCPERF_NP"
fi
GPUS="${GPUS:-1}"
if [ "$GPUS" = "all" ]; then
    N=$TOTAL_GPUS
else
    case "$GPUS" in (*[!0-9]*|"") die "--gpus must be a positive integer or 'all' (got '$GPUS')";; esac
    N="$GPUS"
fi
[ "$N" -ge 1 ] || die "--gpus must be >= 1"

if [ "$N" -gt "$TOTAL_GPUS" ]; then
    if [ "${HPCPERF_ALLOW_OVERSUBSCRIBE:-0}" = "1" ]; then
        say "WARNING: DEBUG ONLY: GPU/rank oversubscription is enabled ($N ranks > $TOTAL_GPUS GPUs)."
    else
        die "$N ranks requested but only $TOTAL_GPUS GPUs allocated ($NNODES node(s) x $GPN); refusing (set HPCPERF_ALLOW_OVERSUBSCRIBE=1 for debug only)"
    fi
fi

RPN=$(( (N + NNODES - 1) / NNODES ))
if [ "$RPN" -gt "$GPN" ] && [ "${HPCPERF_ALLOW_OVERSUBSCRIBE:-0}" != "1" ]; then
    die "$RPN ranks per node needed but only $GPN GPUs per node"
fi

# --------------------------------------------------------------- site profile
PROFILE="${HPCPERF_SITE_PROFILE:-}"
if [ -z "$PROFILE" ]; then
    case "$(hostname -s)" in dgx003|hopper*|gpu0*) PROFILE=gmu-hopper ;; *) PROFILE=generic ;; esac
fi
PROFILE_FILE="$HERE/site/$PROFILE.sh"
[ -f "$PROFILE_FILE" ] || die "site profile '$PROFILE' not found ($PROFILE_FILE)"
# shellcheck disable=SC1090
source "$PROFILE_FILE"

if [ "$NNODES" -gt 1 ]; then
    case "${SITE_MULTINODE_STATUS:-unverified}" in
        blocked)
            if [ "$DRYRUN" = 1 ]; then say "NOTE: ${SITE_MULTINODE_MSG:-multi-node blocked} (dry-run only)";
            else die "${SITE_MULTINODE_MSG:-multi-node blocked on this site}"; fi ;;
        unverified) say "WARNING: ${SITE_MULTINODE_MSG:-multi-node unverified on this site}" ;;
    esac
fi

[ "$LAUNCHER" = auto ] && LAUNCHER="${SITE_LAUNCHER_DEFAULT:-auto}"
if [ "$LAUNCHER" = auto ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then LAUNCHER=srun; else LAUNCHER=mpirun; fi
fi

# ------------------------------------------------------------------- binding
WRAP=()
case "$BIND" in
    wrapper) WRAP=("$HERE/mpi_gpu_bind.sh") ;;
    app)     WRAP=(env HPCPERF_BIND_REPORT_ONLY=1 "$HERE/mpi_gpu_bind.sh") ;;
    none)    WRAP=() ;;
    *) die "--bind must be wrapper|app|none" ;;
esac

# ------------------------------------------------------------------- command
CMD=()
if [ "$LAUNCHER" = mpirun ]; then
    MAP="ppr:${RPN}:node"
    if { [ -n "$SLOTS" ] && [ "$SLOTS" -lt "$RPN" ]; } || [ "$N" -gt "$TOTAL_GPUS" ]; then
        MAP="$MAP:OVERSUBSCRIBE"
        if [ "$N" -le "$TOTAL_GPUS" ]; then
            say "relaxing Slurm task-slot limit (slots/node=$SLOTS < ranks/node=$RPN); GPU:rank stays 1:1 ($N ranks <= $TOTAL_GPUS GPUs)"
        fi
    fi
    # shellcheck disable=SC2207
    SITE_ARGS=($(site_mpi_args "$NNODES"))
    CMD=(mpirun -np "$N" --map-by "$MAP" "${SITE_ARGS[@]}")
elif [ "$LAUNCHER" = srun ]; then
    CMD=(srun --ntasks="$N" --ntasks-per-node="$RPN" --gpus-per-task=1)
    [ -n "$CPUS_PER_RANK" ] && CMD+=(--cpus-per-task="$CPUS_PER_RANK")
else
    die "unknown launcher '$LAUNCHER'"
fi
CMD+=("${WRAP[@]}" "$@")

say "site=$PROFILE launcher=$LAUNCHER nodes=$NNODES ($NODELIST) gpus/node=$GPN slots/node=${SLOTS:-n/a}"
say "ranks=$N (one per GPU), ranks/node=$RPN, bind=$BIND"
say "command: ${CMD[*]}"
if [ "$DRYRUN" = 1 ]; then
    say "dry-run: not executing"
    exit 0
fi
exec "${CMD[@]}"
