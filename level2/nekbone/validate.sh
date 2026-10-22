#!/usr/bin/env bash
# Validate the Nekbone CUDA build the way upstream's test harness does (test/test_nekbone.py): the GPU variant must
# reproduce the serial (CPU) variant's `cg:` lines on test/nek_gpu1 (256 elements of order 16, upstream's example).
#
#   ./validate.sh [CUDA]
#
# Builds test/nek_gpu1 with build.sh CUDA and build.sh CPU when the executables are missing, runs both through the
# common launcher (one rank; the CPU binary ignores the GPU) with the registered data.rea, and compares:
#   1. exact equality of every `cg:` line, as upstream's assertOutputEqual asserts (reported; not the verdict --
#      upstream's assertion was made with the 2018 PGI toolchain and no claim about today's device reductions);
#   2. the rule of check_cg.py (the correctness criterion of the registered inputs, reference/<id>.json) between the
#      two runs: r0 within 1e-4 relative (one unit of the 5th printed digit), solution norm and functional within
#      1e-10 relative, the residual reduction of the GPU run within 10 x the CPU run's; nelt / nx1 / iteration count
#      equal. Basis (README): on test/nek_gpu1 and every registered input the two builds agree in every printed cg:
#      quantity exactly and in the solution functionals to <= 6e-16 relative (2026-10-21).
# PASS when 2. holds for both solves' r0 and for the timed solve's solution / reduction, every run exits 0 and reaches
# 'Avg MFlops', the CUDA run's build banner shows the OpenACC/CUDA Fortran path (the 'USE_GPU_DIRECT' line printed only
# by the ACC build) and the launcher audit is clean.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$R" ] && [ -f "$R/hpcperf_env.sh" ]; then
    set +u; source "$R/hpcperf_env.sh" 2>/dev/null; set -u
fi
case "$(printf '%s' "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')" in
    CUDA) ;;
    *) echo "usage: $0 [CUDA]" >&2; exit 2 ;;
esac
OUT="$R/build/level2/nekbone/validate"; mkdir -p "$OUT"
fail=0
for v in cuda cpu; do
    EXE="$R/build/level2/nekbone/$v/nek_gpu1/nekbone"
    if [ ! -x "$EXE" ]; then
        echo "== validate: building test/nek_gpu1 ($v)"
        bash "$HERE/build.sh" "$(printf '%s' "$v" | tr '[:lower:]' '[:upper:]')" nek_gpu1 > "$OUT/build.$v.log" 2>&1 || { echo "FAIL: build.sh ${v^^} nek_gpu1 (see $OUT/build.$v.log)"; exit 1; }
    fi
done
# shellcheck disable=SC1091
source "$R/level2/tools/hpcperf_launch_common.sh"
for v in cuda cpu; do
    EXE="$R/build/level2/nekbone/$v/nek_gpu1/nekbone"
    echo "== validate: running test/nek_gpu1 with the $v build"
    ( cd "$HERE/test/nek_gpu1" && "$HPCPERF_LAUNCHER_BIN" --gpus 1 --bind wrapper -- "$EXE" data ) > "$OUT/run.$v.log" 2>&1
    rc=$?
    grep -q '^Avg MFlops' "$OUT/run.$v.log" || { echo "FAIL: the $v run did not reach 'Avg MFlops' (exit $rc, $OUT/run.$v.log)"; fail=1; }
    [ $rc -eq 0 ] || { echo "FAIL: the $v run exited $rc"; fail=1; }
    grep -q 'audit summary: 1 verified, 0 mismatch' "$OUT/run.$v.log" || grep -q 'audit summary:.*0 mismatch' "$OUT/run.$v.log" || { echo "FAIL: launcher audit not clean in the $v run"; fail=1; }
done
grep -q 'USE_GPU_DIRECT' "$OUT/run.cuda.log" || { echo "FAIL: the CUDA run shows no OpenACC gather-scatter set-up line (USE_GPU_DIRECT): not the GPU code path"; fail=1; }
grep -q 'USE_GPU_DIRECT' "$OUT/run.cpu.log" && { echo "FAIL: the CPU build printed the OpenACC set-up line"; fail=1; }
echo "== cg: lines (cuda | cpu)"
paste -d'|' <(grep '^cg:' "$OUT/run.cuda.log") <(grep '^cg:' "$OUT/run.cpu.log")
if cmp -s <(grep '^cg:' "$OUT/run.cuda.log") <(grep '^cg:' "$OUT/run.cpu.log"); then
    echo "upstream exactness: every cg: line of the CUDA run equals the CPU run's (test_nekbone.py assertOutputEqual holds)"
else
    echo "upstream exactness: the cg: lines differ at the printed 5 digits (test_nekbone.py's exact assertion does not hold with this toolchain; the rule below is the verdict)"
fi
python3 - "$OUT/run.cuda.log" "$OUT/run.cpu.log" "$HERE/check_cg.py" <<'PY' || fail=1
import importlib.util, math, sys
spec = importlib.util.spec_from_file_location("check_cg", sys.argv[3]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
try:
    g = m.parse(open(sys.argv[1], errors="replace").read()); c = m.parse(open(sys.argv[2], errors="replace").read())
except ValueError as ex:
    print(f"FAIL: cannot parse the runs: {ex}"); sys.exit(1)
ok = True
def chk(label, a, b, tol):
    global ok
    d = m.rel(a, b); good = d <= tol; ok &= good
    print(f"  {label}: cuda {a:.10e} cpu {b:.10e} rel {d:.2e} (tol {tol:.0e}) {'ok' if good else 'BAD'}")
for k in ("nelt", "nx1"):
    good = g[k] == c[k]; ok &= good; print(f"  {k}: cuda {g[k]} cpu {c[k]} {'ok' if good else 'BAD'}")
for i, (sg, sc) in enumerate(zip(g["solves"], c["solves"])):
    chk(f"solve {i} r0", sg["r0"], sc["r0"], 1e-4)
    good = sg["iter"] == sc["iter"]; ok &= good; print(f"  solve {i} iterations: cuda {sg['iter']} cpu {sc['iter']} {'ok' if good else 'BAD'}")
chk("timed solve xnorm", g["xnorm"], c["xnorm"], 1e-10); chk("timed solve xf", g["xf"], c["xf"], 1e-10)
tg, tc = g["solves"][-1], c["solves"][-1]; rg, rc = tg["rN"] / tg["r0"], tc["rN"] / tc["r0"]
good = rg <= 10 * rc; ok &= good
print(f"  timed solve residual reduction: cuda {rg:.3e} cpu {rc:.3e} (limit 10 x cpu) {'ok' if good else 'BAD'}")
sys.exit(0 if ok else 1)
PY
if [ $fail -eq 0 ]; then echo "PASS: Nekbone validate (test/nek_gpu1: CUDA build vs CPU build under the check_cg.py rule)"; else echo "FAIL: Nekbone validate"; exit 1; fi
