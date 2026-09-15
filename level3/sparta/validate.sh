#!/usr/bin/env bash
# Correctness check for the SPARTA Kokkos build, using upstream's benchmark
# deck bench/in.collide as shipped (10,000 particles) and the reference log
# SPARTA ships for it, bench/log.7Jul14.collide.icc.10K.1.
#
#   ./validate.sh [CUDA|HIP]        HPCPERF_GPUS=N (default 1) selects the rank count
#
# DSMC is stochastic, so per-step collision counts cannot match exactly. What
# is checked (adapted subset of upstream's tolerance-based regression):
#   * the benchmark stats block is COMPLETE -- it must span the equilibration
#     boundary (step 30) through the final step (130); a truncated run FAILs;
#   * particle count Np == 10,000 at every stats row (exact conservation);
#   * mean gas temperature over steps >= 40 within 2 % of the reference
#     (statistical noise ~0.8 % for 10^4 particles);
#   * mean collision attempts (Natt) within 15 %.
# With HPCPERF_GPUS>1 the same criteria compare the N-rank run with this build's
# 1-rank run. Reproducibility: the run's real exit code is captured (nonzero /
# timeout / missing log / non-finite -> FAIL), run.sh removes its target log
# first so only this run's output is used, and NaN/Inf is rejected explicitly.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
N="${HPCPERF_GPUS:-1}"
l3_require_materialized "$HERE" || exit 3
REF="$HERE/src/bench/log.7Jul14.collide.icc.10K.1"     # upstream reference log, part of the frozen source bundle
PROFILE="$(l3_backend_profile SPARTA "$MODEL")"
l3_paths_profile sparta "$PROFILE" "$MODEL" || exit 2     # the run tree of the SAME profile build.sh/run.sh use
RUN_DIR="$L3_BUILD/$L3_RUN_SUBDIR"
TIMEOUT="${HPCPERF_VALIDATE_TIMEOUT:-900}"
[ -f "$REF" ] || { echo "validate.sh: reference log $REF missing (run tools/prepare_benchmark.sh level3 sparta)" >&2; exit 1; }

export HPCPERF_GPUS="$N"
echo "validate.sh: SPARTA $BACKEND smoke (bench/in.collide 10x10x10, 10,000 particles) on $N GPU(s) [profile $PROFILE]"
run_once() { local ng=$1 out=$2 rc=0; HPCPERF_GPUS="$ng" HPCPERF_SCALE_MODE=smoke timeout "$TIMEOUT" "$HERE/run.sh" "$BACKEND" > "$out" 2>&1 || rc=$?; return $rc; }
mkdir -p "$RUN_DIR"
VOUT="$RUN_DIR/validate.smoke.np$N.stdout"
rc=0; run_once "$N" "$VOUT" || rc=$?
grep -aE '^#|hpcperf-launch: audit summary|Loop time|ERROR|abort' "$VOUT" || true
if [ "$rc" -eq 124 ]; then echo "validate.sh: FAIL -- run timed out after ${TIMEOUT}s"; exit 1; fi
[ "$rc" -eq 0 ] || { echo "validate.sh: FAIL -- run.sh exited $rc (see $VOUT)"; exit 1; }
LOG="$RUN_DIR/log.smoke.np$N.sparta"
[ -f "$LOG" ] || { echo "validate.sh: FAIL -- no log produced ($LOG)"; exit 1; }
if [ "$N" -gt 1 ] && [ ! -f "$RUN_DIR/log.smoke.np1.sparta" ]; then
    echo "validate.sh: producing the 1-GPU run for rank-count comparison"
    r1=0; run_once 1 "$RUN_DIR/validate.smoke.np1.stdout" || r1=$?
    [ "$r1" -eq 0 ] && [ -f "$RUN_DIR/log.smoke.np1.sparta" ] || { echo "validate.sh: FAIL -- 1-GPU run failed (rc=$r1)"; exit 1; }
fi

python3 - "$LOG" "$REF" "$N" "$RUN_DIR/log.smoke.np1.sparta" <<'PY'
import re, sys, os
sys.path.insert(0, os.environ["L3_TOOLS"])
from l3_check import require_finite, ValidationError
def stats(path):
    blocks, cur, cols = [], [], None
    for ln in open(path).read().splitlines():
        p = ln.split()
        if p[:2] == ["Step", "CPU"]:
            cols = ["temp" if c == "c_temp" else c for c in p]; cur = []; blocks.append(cur); continue
        if cols and p and re.match(r'^\d+$', p[0]):
            cur.append(dict(zip(cols, p))); continue
        if cols and cur and not p: cols = None
    return blocks[-1] if blocks else []
EXPECT_FIRST, EXPECT_LAST = 30, 130     # benchmark block: run 30 (equilibrate) then run 100
FIELDS = ["Np", "temp", "Natt"]
def mean(rows, k, minstep=40):
    v = [require_finite(f"{k}@{int(r['Step'])}", r[k]) for r in rows if int(r["Step"]) >= minstep]
    if not v: raise ValidationError(f"no rows with Step>={minstep} for {k}")
    return sum(v) / len(v)
log, ref, n, log1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
NPART = 10.0 * 10 * 10 * 10
ok = True
try:
    got, want = stats(log), stats(ref)
    for label, rows in (("run", got), ("reference", want)):
        if not rows: raise ValidationError(f"{label} log has no benchmark stats block")
        for f in FIELDS:
            if f not in rows[0]: raise ValidationError(f"{label} log missing field '{f}' (have {list(rows[0])})")
    steps = [int(r["Step"]) for r in got]
    if steps[0] != EXPECT_FIRST or steps[-1] != EXPECT_LAST:
        raise ValidationError(f"run stats block spans steps {steps[0]}..{steps[-1]}, expected {EXPECT_FIRST}..{EXPECT_LAST} (truncated/incomplete run)")
    def check(rows, base, label):
        global ok
        bad_np = [int(r["Step"]) for r in rows if require_finite("Np", r["Np"]) != NPART]
        print(f"  {label}: Np == {NPART:.0f} at every row: {'ok' if not bad_np else 'BAD at ' + str(bad_np)}"); ok &= not bad_np
        for k, tol in (("temp", 0.02), ("Natt", 0.15)):
            a, b = mean(rows, k), mean(base, k)
            rel = abs(a - b) / abs(b)
            print(f"  {label}: mean {k:<5} {a:12.4f} vs {b:12.4f} rel {rel:.3e} (tol {tol}) {'ok' if rel <= tol else 'BAD'}")
            ok &= rel <= tol
    print(f"[1] {n}-GPU run vs upstream reference log (icc, 1 proc, 2014):")
    check(got, want, "vs-ref")
    if n > 1:
        one = stats(log1)
        if not one: raise ValidationError("1-GPU log has no benchmark stats block")
        print(f"[2] {n}-GPU run vs this build's 1-GPU run:")
        check(got, one, "vs-1gpu")
except ValidationError as ex:
    print(f"  VALIDATION ERROR: {ex}")
    print(f"SPARTA CUDA validation ({n} GPU): FAIL"); sys.exit(1)
print(f"SPARTA CUDA validation ({n} GPU, bench/in.collide vs log.7Jul14.collide.icc.10K.1): {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
