#!/usr/bin/env bash
# Correctness check for the SPARTA Kokkos build, using upstream's benchmark
# deck bench/in.collide as shipped (10,000 particles) and the reference log
# SPARTA ships for it, bench/log.7Jul14.collide.icc.10K.1.
#
#   ./validate.sh [CUDA|HIP]        HPCPERF_GPUS=N (default 1) selects the rank count
#
# DSMC is a stochastic method (random seed, random collision partners), and
# the particle distribution over ranks changes the random stream, so per-step
# collision counts cannot be compared exactly. What IS exact and what has a
# physically justified tolerance:
#   * particle count Np at every stats row == 10 * cells (10,000 with the
#     10x10x10 deck: exact conservation -- no chemistry, reflecting walls);
#   * gas temperature (compute temp): the equilibrated argon stays at the
#     initial 273.15 K; the reference log shows 273.28 K. Tolerance 2 % on the
#     mean over the benchmark steps (>= step 40): the statistical temperature
#     noise of 10^4 particles is ~sqrt(2/3N) ~ 0.8 %, so 2 % is ~2.5 sigma of
#     the sampling noise and far below any physics or unit error;
#   * mean collision attempts per step (Natt) within 15 % of the reference
#     mean: it is set by density/temperature/cross-section, so a wrong
#     collision model or density would move it by far more; run-to-run
#     statistical scatter is a few %.
# With HPCPERF_GPUS>1 the same three criteria are applied between the N-rank
# run and this build's 1-rank run (rank-count independence). Prints PASS/FAIL,
# exit 0/1. Nothing is loosened to pass; the deck is upstream's.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
N="${HPCPERF_GPUS:-1}"
REF="$R/_upstream/level3/sparta/bench/log.7Jul14.collide.icc.10K.1"
RUN_DIR="$R/build/level3/sparta/$MODEL/run"
[ -f "$REF" ] || { echo "validate.sh: reference log $REF missing (run fetch.sh)" >&2; exit 1; }

export HPCPERF_GPUS="$N"
echo "validate.sh: SPARTA $BACKEND smoke (bench/in.collide 10x10x10, 10,000 particles) on $N GPU(s)"
HPCPERF_SCALE_MODE=smoke "$HERE/run.sh" "$BACKEND" 2>&1 | grep -E '^#|hpcperf-launch: audit|Loop time|ERROR' || true
LOG="$RUN_DIR/log.smoke.np$N.sparta"
[ -f "$LOG" ] || { echo "validate.sh: FAIL -- no log produced ($LOG)"; exit 1; }
if [ "$N" -gt 1 ] && [ ! -f "$RUN_DIR/log.smoke.np1.sparta" ]; then
    echo "validate.sh: producing the 1-GPU run for rank-count comparison"
    HPCPERF_GPUS=1 HPCPERF_SCALE_MODE=smoke "$HERE/run.sh" "$BACKEND" > /dev/null 2>&1 || true
fi

python3 - "$LOG" "$REF" "$N" "$RUN_DIR/log.smoke.np1.sparta" <<'PY'
import re, sys
def stats(path):
    """rows of the LAST stats block (the 100-step benchmark run) as dicts"""
    blocks, cur, cols = [], [], None
    for ln in open(path).read().splitlines():
        p = ln.split()
        if p[:2] == ["Step", "CPU"]:
            # the 2014 reference log labels the compute column "temp", current SPARTA prints "c_temp"
            cols = ["temp" if c == "c_temp" else c for c in p]; cur = []; blocks.append(cur); continue
        if cols and p and re.match(r'^\d+$', p[0]):
            cur.append(dict(zip(cols, map(float, p))))
        elif cols and cur and not p:
            cols = None
    return blocks[-1] if blocks else []
def mean(rows, k, minstep=40):
    v = [r[k] for r in rows if r["Step"] >= minstep]
    return sum(v) / len(v) if v else float("nan")
log, ref, n, log1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
ok = True
def check(rows, base, label, npart):
    global ok
    if not rows: print(f"  {label}: no stats rows"); ok = False; return
    bad_np = [r["Step"] for r in rows if r["Np"] != npart]
    print(f"  {label}: Np == {npart} at every row: {'ok' if not bad_np else 'BAD at steps ' + str(bad_np)}"); ok &= not bad_np
    for k, tol in (("temp", 0.02), ("Natt", 0.15)):
        a, b = mean(rows, k), mean(base, k)
        rel = abs(a - b) / abs(b)
        print(f"  {label}: mean {k:<5} {a:12.4f} vs {b:12.4f} rel {rel:.3e} (tol {tol}) {'ok' if rel <= tol else 'BAD'}")
        ok &= rel <= tol
got, want = stats(log), stats(ref)
print(f"[1] {n}-GPU run vs upstream reference log (icc, 1 proc, 2014):")
NPART = 10.0 * 10 * 10 * 10   # deck: n = 10 * x*y*z particles = 10,000 for the 10x10x10 grid
check(got, want, "vs-ref", NPART)
if n > 1:
    try:
        one = stats(log1)
        print(f"[2] {n}-GPU run vs this build's 1-GPU run:")
        check(got, one, "vs-1gpu", NPART)
    except FileNotFoundError:
        print("[2] 1-GPU log unavailable; rank-count comparison skipped"); ok = False
print(f"SPARTA CUDA validation ({n} GPU, bench/in.collide vs log.7Jul14.collide.icc.10K.1): {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
