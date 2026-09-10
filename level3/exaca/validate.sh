#!/usr/bin/env bash
# Correctness check for the ExaCA build: directional-solidification smoke case (128^3 cells, the upstream
# Inp_DirSolidification.json physics with the decomposition-independent substrate) on N GPUs.
#
#   ./validate.sh [CUDA|HIP]        HPCPERF_GPUS=N (default 1) selects the rank count
#
# ExaCA has no shipped reference output; its cell captures use atomic compare-exchange, so the final GrainID
# field is NOT bitwise reproducible (measured: identical 1-GPU runs differ bitwise while every statistic below
# agrees). The criteria are therefore statistical invariants of the final microstructure, computed from the
# GrainID field the run writes (exaca_check.py) with tolerances derived from the measured spread (see README):
#   [1] completeness: run exits 0, the field and the log exist, DIMENSIONS == the deck, every cell solidified
#       (GrainID != 0), the log reports N ranks and ONE global decomposition in Y (subdomain sizes sum to
#       Ny + 2*(N-1): 1-cell halos at each internal boundary) -- never N independent copies
#   [2] self-consistency: ExaCA's own VolFractionNucleated equals the value recomputed from the field (1e-3)
#   [3] vs the frozen reference statistics of the validated 1-GPU baseline run
#       (references/dirsolid_smoke.reference.json): grain count, nucleated-grain count and volume fraction,
#       top-layer grain count, mean <001>-to-z misorientation (bulk and top layer), mean grain volume
#   [4] with HPCPERF_GPUS>1: the N-rank statistics vs this build's 1-GPU run (rank-count independence)
# Rules enforced here: the run's real exit code is captured; a nonzero exit, a timeout, a missing field/log
# or a non-finite statistic is a FAIL (never a PASS on a stale result); run.sh recreates its run directory,
# so only THIS run's output is validated.
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
REF="$HERE/references/dirsolid_smoke.reference.json"; TOL="$HERE/references/dirsolid_smoke.tolerances.json"
ORIENT="$HERE/src/examples/Substrate/GrainOrientationVectors.csv"
[ -f "$REF" ] && [ -f "$TOL" ] || { echo "validate.sh: reference statistics/tolerances missing under references/" >&2; exit 1; }
[ -f "$ORIENT" ] || { echo "validate.sh: $ORIENT missing (run tools/prepare_benchmark.sh level3 exaca)" >&2; exit 1; }
RUN_ROOT="$R/build/level3/exaca/$MODEL/$L3_RUN_SUBDIR"
TIMEOUT="${HPCPERF_VALIDATE_TIMEOUT:-900}"
PY="$(l3_python_yaml)"
unset HPCPERF_SCALE_MODE
export HPCPERF_GPUS="$N"
echo "validate.sh: ExaCA $BACKEND smoke (dirsolid 128^3 cells, Inconel625, G=5e5 K/m, R=3e5 K/s) on $N GPU(s)"

run_once() { # <ngpu> <stdout-file>
    local ng=$1 out=$2 rc=0
    HPCPERF_GPUS="$ng" HPCPERF_SCALE_MODE=smoke timeout "$TIMEOUT" "$HERE/run.sh" "$BACKEND" > "$out" 2>&1 || rc=$?
    return $rc
}
mkdir -p "$RUN_ROOT"
VOUT="$RUN_ROOT/validate.smoke.np$N.stdout"
rc=0; run_once "$N" "$VOUT" || rc=$?
grep -aE '^#|hpcperf-launch: audit summary|ExaCA version|Kokkos version|Number of MPI ranks|Time spent performing CA|Error|error' "$VOUT" | head -12 || true
if [ "$rc" -eq 124 ]; then echo "validate.sh: FAIL -- run timed out after ${TIMEOUT}s"; exit 1; fi
[ "$rc" -eq 0 ] || { echo "validate.sh: FAIL -- run.sh exited $rc (see $VOUT)"; exit 1; }
D="$RUN_ROOT/dirsolid.smoke.np$N"; VTK="$D/dirsolid_smoke_np$N.vtk"; LOG="$D/dirsolid_smoke_np$N.json"
[ -f "$VTK" ] && [ -f "$LOG" ] || { echo "validate.sh: FAIL -- output field/log missing ($VTK, $LOG)"; exit 1; }
if [ "$N" -gt 1 ] && [ ! -f "$RUN_ROOT/dirsolid.smoke.np1/stats.json" ]; then
    echo "validate.sh: producing the 1-GPU run for the rank-count comparison"
    r1=0; run_once 1 "$RUN_ROOT/validate.smoke.np1.stdout" || r1=$?
    [ "$r1" -eq 0 ] || { echo "validate.sh: FAIL -- 1-GPU run failed (rc=$r1)"; exit 1; }
    "$PY" "$HERE/exaca_check.py" stats "$RUN_ROOT/dirsolid.smoke.np1/dirsolid_smoke_np1.vtk" "$RUN_ROOT/dirsolid.smoke.np1/dirsolid_smoke_np1.json" "$ORIENT" --json "$RUN_ROOT/dirsolid.smoke.np1/stats.json" > /dev/null || { echo "validate.sh: FAIL -- 1-GPU statistics"; exit 1; }
fi
"$PY" "$HERE/exaca_check.py" stats "$VTK" "$LOG" "$ORIENT" --json "$D/stats.json" > /dev/null || { echo "validate.sh: FAIL -- statistics of the $N-GPU field could not be computed"; exit 1; }

"$PY" - "$D/stats.json" "$REF" "$TOL" "$N" "$RUN_ROOT/dirsolid.smoke.np1/stats.json" "$HERE" <<'PY'
import json, os, sys
sys.path.insert(0, sys.argv[6]); sys.path.insert(0, os.environ["L3_TOOLS"])
import exaca_check as ec
from l3_check import ValidationError
st, ref, tol, n, st1 = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), json.load(open(sys.argv[3])), int(sys.argv[4]), sys.argv[5]
tol = {k: v for k, v in tol.items() if not k.startswith("_")}
ok = True
def crit(label, good, detail):
    global ok
    ok = ok and good
    print(f"  {label}: {detail} {'ok' if good else 'BAD'}")
try:
    print("[1] completeness / one global decomposition:")
    crit("dimensions", (st["nx"], st["ny"], st["nz"]) == (128, 128, 128), f"{st['nx']}x{st['ny']}x{st['nz']}")
    crit("all cells solidified", st["unsolidified_cells"] == 0, f"unsolidified {st['unsolidified_cells']} of {st['cells']}")
    lg = st.get("log", {})
    crit("ranks in the log", lg.get("ranks") == n, f"{lg.get('ranks')} (requested {n})")
    ys = lg.get("decomposition", {}).get("SubdomainYSize", [])
    crit("Y decomposition covers the box once", len(ys) == n and sum(ys) == st["ny"] + 2 * (n - 1), f"subdomain sizes {ys} sum {sum(ys)} == Ny + 2*(N-1) = {st['ny'] + 2 * (n - 1)}")
    print("[2] self-consistency (ExaCA log vs field):")
    vfc = lg.get("vol_fraction_nucleated_code")
    crit("VolFractionNucleated", vfc is not None and abs(float(vfc) - st["vol_fraction_nucleated"]) <= 1e-3, f"log {vfc} field {st['vol_fraction_nucleated']:.6f}")
    print(f"[3] {n}-GPU statistics vs the frozen reference (validated 1-GPU baseline, ExaCA {ref.get('provenance', {}).get('exaca_version')}):")
    good, lines = ec.compare(st, ref["stats"], tol, "vs-ref"); print("\n".join(lines)); ok = ok and good
    if n > 1:
        s1 = json.load(open(st1))
        print(f"[4] {n}-GPU statistics vs this build's 1-GPU run (rank-count independence):")
        good, lines = ec.compare(st, s1, tol, "vs-1gpu"); print("\n".join(lines)); ok = ok and good
except (ValidationError, KeyError, TypeError, ValueError) as ex:
    print(f"  VALIDATION ERROR: {ex!r}"); ok = False
print(f"ExaCA CUDA validation ({n} GPU, dirsolid smoke 128^3 vs dirsolid_smoke.reference.json): {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
