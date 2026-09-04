#!/bin/bash
# hpcperf_mpi_launch.sh -- unified GPU-count-aware MPI launcher for Level 2.
#
#   hpcperf_mpi_launch.sh [options] -- <exe> [args...]
#
# Options (environment variable defaults in parentheses):
#   --gpus N|all          ranks to launch, one rank per GPU   (HPCPERF_GPUS, def 1)
#   --launcher auto|mpirun|srun                               (HPCPERF_LAUNCHER)
#   --cpus-per-rank C     CPUs bound per rank                 (HPCPERF_CPUS_PER_RANK)
#   --bind wrapper|app|none  GPU binding mode (default wrapper):
#                           wrapper = narrow CUDA_VISIBLE_DEVICES per rank
#                           app     = app binds itself; wrapper only audits
#                           none    = no wrapper at all (no audit possible)
#   --dry-run             print the resource plan and command, do not execute
#                                                             (HPCPERF_DRY_RUN=1)
#
# Resource model -- three distinct things, never mixed up:
#   ACTUAL      what the scheduler really allocated (Slurm), or the local node.
#   REQUESTED   what this launch uses: --gpus, and optionally a SUBSET of the
#               allocation via HPCPERF_NODES / HPCPERF_GPUS_PER_NODE. Outside
#               dry-run these can only shrink the allocation, never enlarge it;
#               a node subset is enforced in the launcher placement
#               (mpirun --host / srun --nodelist), not just printed.
#   HYPOTHETICAL dry-run only: HPCPERF_NODES / HPCPERF_GPUS_PER_NODE larger than
#               the allocation are accepted to PLAN a run elsewhere, and the plan
#               is labelled HYPOTHETICAL.
#
# Policy:
#   * One MPI rank per GPU. ranks > GPUs fails fast unless
#     HPCPERF_ALLOW_OVERSUBSCRIBE=1 (then a loud DEBUG-ONLY warning).
#   * --gpus N uses exactly N GPUs; --gpus all uses every REQUESTED GPU.
#   * HPCPERF_NP is a compatibility alias; it is compared AFTER 'all' has been
#     resolved, and a mismatch with --gpus/HPCPERF_GPUS is an error.
#   * Heterogeneous allocations (different task slots, CPUs or GPUs per node)
#     are refused explicitly -- nothing here assumes uniform nodes unproven.
#   * Slurm task slots < ranks/node (allocation made with -n 1) are relaxed
#     per launch (--map-by ...:OVERSUBSCRIBE) only after BOTH the GPU check
#     (ranks <= GPUs) and the CPU check (ranks/node * cpus/rank <= CPUs/node)
#     passed; the relaxation is logged.
#   * CPU binding: with --cpus-per-rank C, mpirun maps PE=C and binds to cores
#     (srun: --cpus-per-task=C); capacity is checked here first. Without it,
#     the launcher's default binding is left to the MPI runtime and reported.
#   * Transport and mpirun/srun come from the site profile
#     (HPCPERF_SITE_PROFILE -> level2/tools/site/<profile>.sh). Multi-node
#     launches are refused while the profile marks multi-node BLOCKED.
#   * GPU binding audit: every rank logs its EXPECTED GPU (from the local
#     rank) via mpi_gpu_bind.sh; the launcher then samples
#     nvidia-smi --query-compute-apps during the run and reports OBSERVED per
#     rank -- verified / MISMATCH / unverified (never "verified" without an
#     observation). Set HPCPERF_BIND_OBSERVE=0 to skip sampling.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPUS="${HPCPERF_GPUS:-}"; GPUS_GIVEN=0; [ -n "$GPUS" ] && GPUS_GIVEN=1
LAUNCHER="${HPCPERF_LAUNCHER:-auto}"
CPUS_PER_RANK="${HPCPERF_CPUS_PER_RANK:-}"
BIND="wrapper"
DRYRUN="${HPCPERF_DRY_RUN:-0}"
OBSERVE="${HPCPERF_BIND_OBSERVE:-1}"
while [ $# -gt 0 ]; do
    case "$1" in
        --gpus)          GPUS="$2"; GPUS_GIVEN=1; shift 2 ;;
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
is_int() { case "$1" in (''|*[!0-9]*) return 1;; (*) return 0;; esac; }

# ------------------------------------------------------------ ACTUAL allocation
# Slurm list expressions like "64" / "4(x2)" / "64,32": one uniform group is
# "<n>" or "<n>(x<k>)"; anything with a comma is heterogeneous.
uniform_or_die() { # $1 = label, $2 = SLURM_* list expression -> echoes n
    local label=$1 expr=$2
    case "$expr" in
        *,*) die "heterogeneous allocation ($label='$expr'): nodes differ; per-node resources are not assumed uniform -- unsupported, request a homogeneous allocation" ;;
    esac
    echo "${expr%%(*}"
}
if [ -n "${SLURM_JOB_ID:-}" ]; then
    A_NODES="${SLURM_JOB_NUM_NODES:-1}"
    mapfile -t A_NODELIST < <(scontrol show hostnames "${SLURM_JOB_NODELIST:-}" 2>/dev/null)
    [ "${#A_NODELIST[@]}" -ge 1 ] || A_NODELIST=("$(hostname -s)")
    A_SLOTS="$(uniform_or_die SLURM_TASKS_PER_NODE "${SLURM_TASKS_PER_NODE:-1}")"
    A_CPUS="$(uniform_or_die SLURM_JOB_CPUS_PER_NODE "${SLURM_JOB_CPUS_PER_NODE:-${SLURM_CPUS_ON_NODE:-1}}")"
    # GPUs per node: verified per node group from scontrol -d (GRES=gpu[:type]:N(IDX...)),
    # never assumed from the current node alone when several nodes are involved.
    _gres_counts="$(scontrol show job -d "$SLURM_JOB_ID" 2>/dev/null \
        | sed -n 's/.*[[:space:]]Nodes=[^[:space:]]*.*GRES=gpu\(:[^:()]*\)\?:\([0-9]*\).*/\2/p' | sort -u)"
    if [ -n "$_gres_counts" ]; then
        [ "$(echo "$_gres_counts" | wc -l)" -eq 1 ] \
            || die "heterogeneous allocation: GPUs per node differ across node groups ($(echo "$_gres_counts" | tr '\n' ' ')) -- unsupported"
        A_GPN="$_gres_counts"
        _gpn_src="scontrol -d"
    else
        [ "$A_NODES" -eq 1 ] || die "cannot verify GPU count per node for a $A_NODES-node allocation (scontrol show job -d gave no GRES); refusing to assume uniform nodes"
        A_GPN="${SLURM_GPUS_ON_NODE:-0}"
        if [ "$A_GPN" = 0 ] && [ -n "${SLURM_JOB_GPUS:-}" ]; then
            A_GPN=$(( $(echo "$SLURM_JOB_GPUS" | tr -cd , | wc -c) + 1 ))
        fi
        _gpn_src="SLURM_GPUS_ON_NODE"
    fi
    ALLOC_KIND="slurm job $SLURM_JOB_ID"
else
    A_NODES=1
    A_NODELIST=("$(hostname -s)")
    A_SLOTS=""   # not slot-limited outside Slurm
    A_CPUS="$(nproc 2>/dev/null || echo 1)"
    A_GPN="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
    _gpn_src="nvidia-smi"
    ALLOC_KIND="local node (no scheduler)"
fi
is_int "$A_GPN" || A_GPN=0
[ "$A_GPN" -ge 1 ] || die "no GPUs in the actual allocation ($_gpn_src reports $A_GPN); GPU mode needs at least one"

# --------------------------------------------- REQUESTED subset / HYPOTHETICAL
HYPO=0
R_NODES="${HPCPERF_NODES:-$A_NODES}"; is_int "$R_NODES" && [ "$R_NODES" -ge 1 ] || die "HPCPERF_NODES must be a positive integer"
R_GPN="${HPCPERF_GPUS_PER_NODE:-$A_GPN}"; is_int "$R_GPN" && [ "$R_GPN" -ge 1 ] || die "HPCPERF_GPUS_PER_NODE must be a positive integer"
if [ "$R_NODES" -gt "$A_NODES" ] || [ "$R_GPN" -gt "$A_GPN" ]; then
    if [ "$DRYRUN" = 1 ]; then
        HYPO=1
    else
        die "requested ${R_NODES} node(s) x ${R_GPN} GPUs exceeds the actual allocation (${A_NODES} x ${A_GPN}); HPCPERF_NODES/HPCPERF_GPUS_PER_NODE may only select a subset -- larger values are allowed in dry-run (HPCPERF_DRY_RUN=1) as a hypothetical plan"
    fi
fi
TOTAL_GPUS=$(( R_NODES * R_GPN ))

# ----------------------------------------------------------------- rank count
if [ "$GPUS_GIVEN" = 0 ] && [ -n "${HPCPERF_NP:-}" ]; then
    GPUS="$HPCPERF_NP"      # legacy alias, only when --gpus/HPCPERF_GPUS is absent
fi
GPUS="${GPUS:-1}"
if [ "$GPUS" = "all" ]; then
    N=$TOTAL_GPUS
else
    is_int "$GPUS" || die "--gpus must be a positive integer or 'all' (got '$GPUS')"
    N="$GPUS"
fi
[ "$N" -ge 1 ] || die "--gpus must be >= 1"
# HPCPERF_NP conflict check AFTER 'all' resolution
if [ -n "${HPCPERF_NP:-}" ] && [ "${HPCPERF_NP}" != "$N" ]; then
    die "HPCPERF_NP=${HPCPERF_NP} conflicts with the resolved rank count $N (HPCPERF_GPUS/--gpus=$GPUS); one rank per GPU means they must match"
fi

OVERSUB=0
if [ "$N" -gt "$TOTAL_GPUS" ]; then
    if [ "${HPCPERF_ALLOW_OVERSUBSCRIBE:-0}" = "1" ]; then
        say "WARNING: DEBUG ONLY: GPU/rank oversubscription is enabled ($N ranks > $TOTAL_GPUS GPUs)."
        OVERSUB=1
    else
        die "$N ranks requested but only $TOTAL_GPUS GPUs available to this launch (${R_NODES} node(s) x ${R_GPN}); refusing (HPCPERF_ALLOW_OVERSUBSCRIBE=1 is debug only)"
    fi
fi
RPN=$(( (N + R_NODES - 1) / R_NODES ))          # ranks per node (balanced)
[ "$RPN" -le "$R_GPN" ] || [ "$OVERSUB" = 1 ] || die "$RPN ranks per node needed but only $R_GPN GPUs per node"
NODES_USED=$(( (N + RPN - 1) / RPN ))          # nodes actually placed on

# ---------------------------------------------------------------- CPU capacity
if [ -n "$CPUS_PER_RANK" ]; then
    is_int "$CPUS_PER_RANK" && [ "$CPUS_PER_RANK" -ge 1 ] || die "--cpus-per-rank must be a positive integer"
    [ $(( RPN * CPUS_PER_RANK )) -le "$A_CPUS" ] \
        || die "CPU capacity: $RPN ranks/node x $CPUS_PER_RANK cpus/rank = $((RPN*CPUS_PER_RANK)) exceeds the $A_CPUS CPUs allocated per node"
    CPU_NOTE="bind $CPUS_PER_RANK core(s)/rank"
else
    [ "$RPN" -le "$A_CPUS" ] || die "CPU capacity: $RPN ranks/node but only $A_CPUS CPUs allocated per node"
    CPU_NOTE="runtime default binding (<= $(( A_CPUS / RPN )) cpus/rank available; set HPCPERF_CPUS_PER_RANK to pin)"
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

if [ "$NODES_USED" -gt 1 ]; then
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

# ------------------------------------------------------------------ placement
# Hosts actually used: the first NODES_USED of the (possibly subset) node list.
HOSTS=()
for (( i=0; i<NODES_USED; i++ )); do
    if [ "$i" -lt "${#A_NODELIST[@]}" ]; then HOSTS+=("${A_NODELIST[$i]}")
    else HOSTS+=("hypothetical-node$((i+1))"); fi
done
HOSTLIST="$(IFS=,; echo "${HOSTS[*]}")"

# -------------------------------------------------------------------- binding
BIND_LOG="${HPCPERF_BIND_LOG:-}"
if [ -z "$BIND_LOG" ] && [ "$BIND" != none ]; then
    if [ "$DRYRUN" = 1 ]; then BIND_LOG="<tmpfile>"
    else BIND_LOG="$(mktemp "${HPCPERF_RUN_TMPDIR:-${TMPDIR:-/tmp}}/hpcperf-bind.XXXXXX")"; fi
fi
WRAP=()
case "$BIND" in
    wrapper) WRAP=(env "HPCPERF_BIND_LOG=$BIND_LOG" "$HERE/mpi_gpu_bind.sh") ;;
    app)     WRAP=(env "HPCPERF_BIND_LOG=$BIND_LOG" HPCPERF_BIND_REPORT_ONLY=1 "$HERE/mpi_gpu_bind.sh") ;;
    none)    WRAP=() ;;
    *) die "--bind must be wrapper|app|none" ;;
esac

# -------------------------------------------------------------------- command
CMD=()
if [ "$LAUNCHER" = mpirun ]; then
    MAP="ppr:${RPN}:node"
    [ -n "$CPUS_PER_RANK" ] && MAP="$MAP:PE=$CPUS_PER_RANK"
    if { [ -n "$A_SLOTS" ] && [ "$A_SLOTS" -lt "$RPN" ]; } || [ "$OVERSUB" = 1 ]; then
        MAP="$MAP:OVERSUBSCRIBE"
        [ "$OVERSUB" = 1 ] || say "relaxing Slurm task-slot accounting (slots/node=$A_SLOTS < ranks/node=$RPN) -- checked first: GPUs $N<=$TOTAL_GPUS, CPUs $RPN x ${CPUS_PER_RANK:-1} <= $A_CPUS per node"
    fi
    HOSTSPEC=()
    for h in "${HOSTS[@]}"; do HOSTSPEC+=("$h:$RPN"); done
    # shellcheck disable=SC2207
    SITE_ARGS=($(site_mpi_args "$NODES_USED"))
    CMD=(mpirun -np "$N" --host "$(IFS=,; echo "${HOSTSPEC[*]}")" --map-by "$MAP")
    [ -n "$CPUS_PER_RANK" ] && CMD+=(--bind-to core)
    CMD+=("${SITE_ARGS[@]}")
elif [ "$LAUNCHER" = srun ]; then
    CMD=(srun --nodes="$NODES_USED" --nodelist="$HOSTLIST" --ntasks="$N" --ntasks-per-node="$RPN" --gpus-per-task=1)
    [ -n "$CPUS_PER_RANK" ] && CMD+=(--cpus-per-task="$CPUS_PER_RANK")
else
    die "unknown launcher '$LAUNCHER'"
fi
CMD+=("${WRAP[@]}" "$@")

# ----------------------------------------------------------------------- plan
say "actual:    $ALLOC_KIND: nodes=$A_NODES (${A_NODELIST[*]}) gpus/node=$A_GPN ($_gpn_src) cpus/node=$A_CPUS slots/node=${A_SLOTS:-n/a}"
if [ "$HYPO" = 1 ]; then
    say "HYPOTHETICAL (dry-run only): planning for nodes=$R_NODES gpus/node=$R_GPN -- NOT what is allocated"
else
    say "requested: nodes=$R_NODES gpus/node=$R_GPN (subset of the allocation)"
fi
say "launch:    site=$PROFILE launcher=$LAUNCHER ranks=$N (one per GPU) on $NODES_USED node(s) [$HOSTLIST], ranks/node=$RPN, gpu-bind=$BIND, cpu: $CPU_NOTE"
say "command:   ${CMD[*]}"
if [ "$DRYRUN" = 1 ]; then
    say "dry-run: not executing"
    exit 0
fi

# ------------------------------------------------- run + observe GPU binding
if [ "$BIND" = none ] || [ "$OBSERVE" != 1 ] || ! command -v nvidia-smi >/dev/null 2>&1; then
    exec "${CMD[@]}"
fi
OBS_LOG="$(mktemp "${HPCPERF_RUN_TMPDIR:-${TMPDIR:-/tmp}}/hpcperf-obs.XXXXXX")"
"${CMD[@]}" &
CHILD=$!
trap 'kill -TERM "$CHILD" 2>/dev/null' INT TERM
( while kill -0 "$CHILD" 2>/dev/null; do
      nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | tr -d ' ' >> "$OBS_LOG"
      sleep 1
  done ) &
SAMPLER=$!
wait "$CHILD"; RC=$?
wait "$SAMPLER" 2>/dev/null
trap - INT TERM

# Join expected (from the wrapper's audit lines) with observed (sampler; this
# node only -- ranks on other hosts stay 'unverified').
LOCAL="$(hostname -s)"; NV=0; NM=0; NU=0
if [ -s "$BIND_LOG" ]; then
    while IFS= read -r line; do
        host="$(sed -n 's/.* host=\([^ ]*\).*/\1/p' <<< "$line")"
        gr="$(sed -n 's/.* grank=\([^ ]*\).*/\1/p' <<< "$line")"
        pid="$(sed -n 's/.* pid=\([^ ]*\).*/\1/p' <<< "$line")"
        exp="$(sed -n 's/.* expected_uuid=\([^ ]*\).*/\1/p' <<< "$line")"
        obs=""
        if [ "$host" = "$LOCAL" ] && [ -n "$pid" ]; then
            obs="$(grep "^$pid," "$OBS_LOG" 2>/dev/null | cut -d, -f2 | sort -u | tr '\n' '+' | sed 's/+$//')"
        fi
        if [ -z "$obs" ]; then status="unverified"; NU=$((NU+1))
        elif [ "$obs" = "$exp" ]; then status="verified"; NV=$((NV+1))
        else status="MISMATCH"; NM=$((NM+1)); fi
        say "audit: grank=$gr host=$host expected=$exp observed=${obs:-none} -> $status"
    done < "$BIND_LOG"
    say "audit summary: $NV verified, $NM mismatch, $NU unverified (of $N ranks; observation = nvidia-smi compute-apps sampled on $LOCAL only)"
    [ "$NM" -eq 0 ] || say "WARNING: rank->GPU MISMATCH detected -- the application did not run on the GPU the launcher expected"
else
    say "audit: no binding records (wrapper produced no audit lines) -> GPU placement unverified"
fi
rm -f "$OBS_LOG" "${HPCPERF_BIND_LOG:+}"; [ -z "${HPCPERF_BIND_LOG:-}" ] && rm -f "$BIND_LOG"
exit "$RC"
