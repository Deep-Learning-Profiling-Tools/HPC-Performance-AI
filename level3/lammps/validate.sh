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
#   Step 0  : relative tolerance 1e-8 -- the initial energies are deterministic
#             (same lattice, same geometric velocities) and must agree to
#             double-precision reduction-order noise.
#   Step 100: relative tolerance 1e-5 -- after 100 NVE steps the trajectory has
#             accumulated floating-point differences from the different
#             force-summation order (GPU vs CPU, N ranks vs 1), but the
#             thermodynamic averages of a 32k-atom LJ liquid are insensitive to
#             that at the 1e-6 level; 1e-5 is ten times the largest difference
#             observed on this node and far below any physics change.
# Additionally, with HPCPERF_GPUS>1 the N-rank result is compared against the
# 1-rank GPU result of the same build with the same tolerances (rank-count
# independence). Prints PASS/FAIL; exit 0/1. No tolerance is loosened to pass.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
N="${HPCPERF_GPUS:-1}"
REF="$R/_upstream/level3/lammps/bench/log.15Jul25.lj.fixed.g++.1"
RUN_DIR="$R/build/level3/lammps/$MODEL/run"
[ -f "$REF" ] || { echo "validate.sh: reference log $REF missing (run fetch.sh)" >&2; exit 1; }

unset HPCPERF_SCALE_MODE
export HPCPERF_GPUS="$N"
echo "validate.sh: LAMMPS $BACKEND smoke (bench/in.lj, 32000 atoms, 100 steps) on $N GPU(s)"
HPCPERF_SCALE_MODE=smoke "$HERE/run.sh" "$BACKEND" 2>&1 | grep -E '^#|hpcperf-launch: audit|Loop time|ERROR' || true
LOG="$RUN_DIR/log.smoke.np$N.lammps"
[ -f "$LOG" ] || { echo "validate.sh: FAIL -- no log produced ($LOG)" ; exit 1; }
if [ "$N" -gt 1 ] && [ ! -f "$RUN_DIR/log.smoke.np1.lammps" ]; then
    echo "validate.sh: producing the 1-GPU reference run for rank-count comparison"
    HPCPERF_GPUS=1 HPCPERF_SCALE_MODE=smoke "$HERE/run.sh" "$BACKEND" > /dev/null 2>&1 || true
fi

python3 - "$LOG" "$REF" "$N" "$RUN_DIR/log.smoke.np1.lammps" <<'PY'
import re, sys
def thermo(path):
    rows = {}
    with open(path) as f:
        lines = f.read().splitlines()
    for i, ln in enumerate(lines):
        if ln.split()[:2] == ["Step", "Temp"]:
            cols = ln.split()
            for row in lines[i+1:]:
                p = row.split()
                if not p or not re.match(r'^\d+$', p[0]): break
                rows[int(p[0])] = dict(zip(cols[1:], map(float, p[1:])))
    return rows
log, ref, n, log1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
got, want = thermo(log), thermo(ref)
tol = {0: 1e-8, 100: 1e-5}
keys = ["Temp", "E_pair", "TotEng", "Press"]
ok = True
def cmp(a, b, label):
    global ok
    for step, t in tol.items():
        if step not in a or step not in b:
            print(f"  {label}: step {step} missing (got {sorted(a)} vs {sorted(b)})"); ok = False; continue
        for k in keys:
            x, y = a[step][k], b[step][k]
            rel = abs(x - y) / max(abs(y), 1e-30)
            flag = "ok " if rel <= t else "BAD"
            if rel > t: ok = False
            print(f"  {label}: step {step:>3} {k:<7} got {x: .10g} ref {y: .10g} rel {rel:.2e} (tol {t:.0e}) {flag}")
print(f"[1] {n}-GPU run vs upstream CPU reference log:")
cmp(got, want, "vs-ref")
if n > 1:
    try:
        g1 = thermo(log1)
        print(f"[2] {n}-GPU run vs this build's 1-GPU run (rank-count independence):")
        cmp(got, g1, "vs-1gpu")
    except FileNotFoundError:
        print("[2] 1-GPU log unavailable; rank-count comparison skipped"); ok = False
print(f"LAMMPS CUDA validation ({n} GPU, bench/in.lj vs log.15Jul25.lj.fixed.g++.1): {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
