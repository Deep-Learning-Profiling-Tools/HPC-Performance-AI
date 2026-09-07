#!/usr/bin/env bash
# Correctness check for the LAMMPS Kokkos build (upstream mechanism: compare
# the thermodynamic output of bench/in.lj with the reference log LAMMPS ships,
# bench/log.15Jul25.lj.fixed.g++.1, a CPU run of the same 32,000-atom, 100-step
# problem; `velocity ... loop geom` makes the initial state machine- and
# rank-count-independent).
#
#   ./validate.sh [CUDA|HIP]        HPCPERF_GPUS=N (default 1) selects the rank count
#
# What is compared (Step 0 and Step 100 rows: Temp, E_pair, TotEng, Press):
#   Step 0  : relative tolerance 1e-8 -- deterministic initial state.
#   Step 100: relative tolerance 1e-5 -- reduction-order divergence only.
# With HPCPERF_GPUS>1 the N-rank result is also compared with this build's 1-GPU
# result (rank-count independence). Adapted subset of upstream's regression
# check: upstream's tools/regression-tests/run_tests.py compares every thermo
# column of every logged step with per-quantity tolerances; here the four
# state variables at the first and last step are checked (the quantities that
# move if the physics or force summation is wrong). No tolerance is loosened.
#
# Reproducibility rules enforced here:
#   * the run's real exit code is captured; a nonzero exit, a timeout, a missing
#     log or a non-finite value is a FAIL (never a PASS on a stale log);
#   * run.sh removes its target log before launching, so only THIS run's output
#     is validated;
#   * NaN/Inf in any compared quantity is rejected explicitly (l3_check).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
N="${HPCPERF_GPUS:-1}"
REF="$R/_upstream/level3/lammps/bench/log.15Jul25.lj.fixed.g++.1"
RUN_DIR="$R/build/level3/lammps/$MODEL/$L3_RUN_SUBDIR"
TIMEOUT="${HPCPERF_VALIDATE_TIMEOUT:-900}"
[ -f "$REF" ] || { echo "validate.sh: reference log $REF missing (run fetch.sh)" >&2; exit 1; }

unset HPCPERF_SCALE_MODE
export HPCPERF_GPUS="$N"
echo "validate.sh: LAMMPS $BACKEND smoke (bench/in.lj, 32000 atoms, 100 steps) on $N GPU(s)"

# run once, on N GPUs, capturing the real exit code (no pipe swallows it)
run_once() { # <ngpu> <stdout-file>
    local ng=$1 out=$2 rc=0
    HPCPERF_GPUS="$ng" HPCPERF_SCALE_MODE=smoke timeout "$TIMEOUT" "$HERE/run.sh" "$BACKEND" > "$out" 2>&1 || rc=$?
    return $rc
}
VOUT="$RUN_DIR/validate.smoke.np$N.stdout"; mkdir -p "$RUN_DIR"
rc=0; run_once "$N" "$VOUT" || rc=$?
grep -aE '^#|hpcperf-launch: audit summary|Loop time|ERROR|abort' "$VOUT" || true
if [ "$rc" -eq 124 ]; then echo "validate.sh: FAIL -- run timed out after ${TIMEOUT}s"; exit 1; fi
[ "$rc" -eq 0 ] || { echo "validate.sh: FAIL -- run.sh exited $rc (see $VOUT)"; exit 1; }
LOG="$RUN_DIR/log.smoke.np$N.lammps"
[ -f "$LOG" ] || { echo "validate.sh: FAIL -- no log produced ($LOG)"; exit 1; }
if [ "$N" -gt 1 ] && [ ! -f "$RUN_DIR/log.smoke.np1.lammps" ]; then
    echo "validate.sh: producing the 1-GPU reference run for rank-count comparison"
    r1=0; run_once 1 "$RUN_DIR/validate.smoke.np1.stdout" || r1=$?
    [ "$r1" -eq 0 ] && [ -f "$RUN_DIR/log.smoke.np1.lammps" ] || { echo "validate.sh: FAIL -- 1-GPU reference run failed (rc=$r1)"; exit 1; }
fi

python3 - "$LOG" "$REF" "$N" "$RUN_DIR/log.smoke.np1.lammps" <<'PY'
import re, sys, os
sys.path.insert(0, os.environ["L3_TOOLS"])
from l3_check import require_finite, ValidationError
def thermo(path):
    rows = {}
    lines = open(path).read().splitlines()
    for i, ln in enumerate(lines):
        if ln.split()[:2] == ["Step", "Temp"]:
            cols = ln.split()
            for row in lines[i+1:]:
                p = row.split()
                if not p or not re.match(r'^\d+$', p[0]): break
                rows[int(p[0])] = dict(zip(cols[1:], p[1:]))
    return rows
log, ref, n, log1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
keys = ["Temp", "E_pair", "TotEng", "Press"]; tol = {0: 1e-8, 100: 1e-5}
try:
    got, want = thermo(log), thermo(ref)
    for step in tol:
        if step not in got:  raise ValidationError(f"run log has no Step {step} row (got {sorted(got)}); run did not complete")
        if step not in want: raise ValidationError(f"reference log has no Step {step} row")
        for k in keys:
            if k not in got[step]:  raise ValidationError(f"field '{k}' missing at step {step} in run log")
    ok = True
    def cmp(a, b, label):
        global ok
        for step, t in tol.items():
            for k in keys:
                x = require_finite(f"{label} step{step} {k} (got)", a[step][k])
                y = require_finite(f"{label} step{step} {k} (ref)", b[step][k])
                rel = abs(x - y) / max(abs(y), 1e-30)
                if rel > t: ok = False
                print(f"  {label}: step {step:>3} {k:<7} got {x: .10g} ref {y: .10g} rel {rel:.2e} (tol {t:.0e}) {'ok ' if rel <= t else 'BAD'}")
    print(f"[1] {n}-GPU run vs upstream CPU reference log (adapted subset: 4 state vars at step 0 and 100):")
    cmp(got, want, "vs-ref")
    if n > 1:
        g1 = thermo(log1)
        for step in tol:
            if step not in g1: raise ValidationError(f"1-GPU log has no Step {step} row")
        print(f"[2] {n}-GPU run vs this build's 1-GPU run (rank-count independence):")
        cmp(got, g1, "vs-1gpu")
except ValidationError as ex:
    print(f"  VALIDATION ERROR: {ex}")
    print(f"LAMMPS {sys.argv and ''}validation ({n} GPU): FAIL"); sys.exit(1)
print(f"LAMMPS CUDA validation ({n} GPU, bench/in.lj vs log.15Jul25.lj.fixed.g++.1): {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
