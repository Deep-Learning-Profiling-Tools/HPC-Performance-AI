#!/bin/bash
# Dry-run tests of hpcperf_mpi_launch.sh resource parsing. Nothing is launched.
# Needs a GPU node (or a Slurm GPU allocation) so that the ACTUAL allocation has
# >= 1 GPU; otherwise the test is skipped (exit 0 with SKIP).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
L="$HERE/../hpcperf_mpi_launch.sh"
pass=0; failn=0
ok()  { echo "ok   $*"; pass=$((pass+1)); }
bad() { echo "FAIL $*"; failn=$((failn+1)); }
noise() { grep -viE 'lua|posix|no file|stack traceback|\[C\]|addto|no field' | grep -vE '^\s*$'; }
unset PRTE_MCA_rmaps_default_mapping_policy HPCPERF_GPUS HPCPERF_NP HPCPERF_NODES HPCPERF_GPUS_PER_NODE HPCPERF_CPUS_PER_RANK HPCPERF_DRY_RUN HPCPERF_ALLOW_OVERSUBSCRIBE

ngpu="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
if [ "${ngpu:-0}" -lt 1 ] && [ -z "${SLURM_GPUS_ON_NODE:-}" ]; then
    echo "test_launcher_dryrun: SKIP (no GPU visible)"; exit 0
fi
out="$("$L" --gpus 1 --dry-run -- ./x 2>&1 | noise)"
a_nodes="$(sed -n 's/.*actual:.*nodes=\([0-9]*\).*/\1/p' <<<"$out")"
a_gpn="$(sed -n 's/.*actual:.*gpus\/node=\([0-9]*\).*/\1/p' <<<"$out")"
[ -n "$a_nodes" ] && [ -n "$a_gpn" ] || { echo "cannot parse actual allocation from: $out"; exit 1; }
total=$((a_nodes * a_gpn))

# 1. 'all' resolves to the whole (requested) allocation
out="$("$L" --gpus all --dry-run -- ./x 2>&1 | noise)"
grep -q "ranks=$total (one per GPU)" <<<"$out" && ok "1: --gpus all -> ranks=$total" || bad "1: all -> $out"

# 2. non-dry-run may not enlarge the allocation
out="$(HPCPERF_NODES=$((a_nodes+4)) "$L" --gpus 1 -- true 2>&1 | noise)"; rc=${PIPESTATUS[0]}
grep -q 'exceeds the actual allocation' <<<"$out" && ok "2: HPCPERF_NODES > actual refused outside dry-run" || bad "2: $out"

# 3. dry-run hypothetical is labelled and not called 'requested'
out="$(HPCPERF_NODES=$((a_nodes+4)) HPCPERF_GPUS_PER_NODE=8 "$L" --gpus 40 --dry-run -- ./x 2>&1 | noise)"
if grep -q 'HYPOTHETICAL (dry-run only)' <<<"$out" && ! grep -q '^hpcperf-launch: requested:' <<<"$out" \
   && grep -q 'hypothetical-node' <<<"$out"; then ok "3: hypothetical dry-run plan labelled HYPOTHETICAL"; else bad "3: $out"; fi

# 4. HPCPERF_NP conflict is checked AFTER 'all' resolution
out="$(HPCPERF_NP=$((total+1)) "$L" --gpus all -- true 2>&1 | noise)"
grep -q "HPCPERF_NP=$((total+1)) conflicts with the resolved rank count $total" <<<"$out" && ok "4: NP vs resolved 'all' conflict detected" || bad "4: $out"
out="$(HPCPERF_NP=$total "$L" --gpus all --dry-run -- ./x 2>&1 | noise)"
grep -q "ranks=$total" <<<"$out" && ok "4b: NP == resolved 'all' accepted" || bad "4b: $out"

# 5. GPU subset per node is enforced in placement (host:slots), not just printed
if [ "$a_gpn" -ge 2 ]; then
    out="$(HPCPERF_GPUS_PER_NODE=1 "$L" --gpus 1 --dry-run -- ./x 2>&1 | noise)"
    grep -q 'requested: nodes=.* gpus/node=1 (subset' <<<"$out" && grep -qE -- '--host [^ ]*:1 ' <<<"$out" \
        && ok "5: gpus/node subset reflected in --host <node>:1" || bad "5: $out"
    out="$(HPCPERF_GPUS_PER_NODE=1 "$L" --gpus 2 --dry-run -- ./x 2>&1 | noise)"
    grep -q 'only 1 GPUs available to this launch (1 node(s) x 1); refusing' <<<"$out" && ok "5b: 2 ranks on a 1-GPU/node subset refused" || bad "5b: $out"
fi

# 6. CPU capacity is checked; PE binding emitted when cpus/rank given
a_cpus="$(sed -n 's/.*cpus\/node=\([0-9]*\).*/\1/p' <<<"$(HPCPERF_DRY_RUN=1 "$L" --gpus 1 -- ./x 2>&1 | noise | grep actual:)")"
out="$("$L" --gpus 1 --cpus-per-rank $((a_cpus+1)) -- true 2>&1 | noise)"
grep -q 'CPU capacity' <<<"$out" && ok "6: cpus/rank beyond allocation refused" || bad "6: $out"
out="$("$L" --gpus 1 --cpus-per-rank 2 --dry-run -- ./x 2>&1 | noise)"
grep -qE 'PE=2' <<<"$out" && grep -q -- '--bind-to core' <<<"$out" && ok "6b: PE=2 + --bind-to core emitted" || bad "6b: $out"

# 7. heterogeneous allocations are refused explicitly
out="$(SLURM_JOB_ID=1 SLURM_TASKS_PER_NODE='4,2' SLURM_JOB_NUM_NODES=2 "$L" --gpus 1 --dry-run -- ./x 2>&1 | noise)"
grep -q 'heterogeneous allocation' <<<"$out" && ok "7: heterogeneous SLURM_TASKS_PER_NODE refused" || bad "7: $out"

# 8. ranks > GPUs fails fast; debug override warns loudly
out="$("$L" --gpus $((total+1)) -- true 2>&1 | noise)"
grep -q 'refusing' <<<"$out" && ok "8: ranks > GPUs refused" || bad "8: $out"
out="$(HPCPERF_ALLOW_OVERSUBSCRIBE=1 "$L" --gpus $((total+1)) --dry-run -- ./x 2>&1 | noise)"
grep -q 'DEBUG ONLY: GPU/rank oversubscription' <<<"$out" && ok "8b: oversubscription only with the debug flag + warning" || bad "8b: $out"

echo "test_launcher_dryrun: $pass passed, $failn failed"
[ "$failn" -eq 0 ]
