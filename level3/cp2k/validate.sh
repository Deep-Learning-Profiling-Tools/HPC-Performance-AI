#!/usr/bin/env bash
# Correctness check for CP2K (cp2k.psmp, CUDA) on N GPUs.
#
#   ./validate.sh [CUDA]        HPCPERF_GPUS=N (default 1), HPCPERF_CPUS_PER_RANK=T (default 8)
#
# [1] Upstream regression tests (adapted subset of tests/do_regtest.py): selected inputs
#     are run through run.sh on N ranks/GPUs and the quantity named by the directory's
#     TEST_FILES.toml matcher (E_total = last "Total energy:" column 3; M011 = last
#     "ENERGY| Total FORCE_EVAL" column 9 -- tests/matchers.py) must match the upstream
#     reference within the upstream tolerance. References/tolerances are upstream's; the
#     GPU CI runs the same tests. Selection (HPCPERF_CP2K_REGTESTS): GPW/OT ground states and
#     linear-scaling DBCSR-heavy cases.
# [2] Science case benchmarks/QS/H2O-64.inp (GPW-DFT NVE MD, 10 steps) on N GPUs, checked
#     by cp2k_md_summary.py --check: completeness (exit 0, PROGRAM ENDED, 10 MD steps), every
#     MD-step SCF cycle converged (the initial ATOMIC-guess SCF of upstream's deck does not
#     converge within MAX_SCF=50 -- the deck declares IGNORE_CONVERGENCE_FAILURE for exactly
#     that; it is reported, and would FAIL without that declaration), finite energies, GPU
#     evidence from CP2K's own output (cp2kflags offload_cuda+dbcsr_acc, DBCSR ACC devices
#     >= 1, GRID tasks executed on GPU; PW GPU timers reported or unverified), and --
#     pre-fixed -- the per-step "ENERGY| Total FORCE_EVAL" energies of MD steps 1..10 must
#     agree with the 1-GPU run of the same binary/input within 1e-8 Ha (upstream's
#     check-release-comparison.py demands 1e-10 across MPI x OMP layouts for single-point
#     energies of these benchmarks on CPU; the OT convergence (EPS_SCF 1e-5) and GPU
#     reduction order motivate the looser 1e-8 chosen here BEFORE the runs; the 1e-10
#     result is printed too). NVE conserved-quantity drift is reported.
# Every exit code is captured (timeout/nonzero -> FAIL); nothing is read from old runs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
N="${HPCPERF_GPUS:-1}"; T="${HPCPERF_CPUS_PER_RANK:-8}"
TIMEOUT="${HPCPERF_VALIDATE_TIMEOUT:-3600}"
E_TOL=1e-8       # pre-fixed cross-rank-count tolerance on MD potential energies (Ha)
REGTESTS="${HPCPERF_CP2K_REGTESTS:-QS/regtest-gpw-1/Ar.inp QS/regtest-gpw-1/H2O-geoopt.inp QS/regtest-gpw-1/pyridine.inp QS/regtest-dm-ls-scf-1/H2-big-1.inp QS/regtest-dm-ls-scf-1/H2-big-5.inp}"
GCC_MM="$(l3_version_mm "$(/usr/bin/gcc -dumpfullversion)")"; OMPI_V="$(mpirun --version | head -1 | /usr/bin/grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
PROFILE="${HPCPERF_CP2K_PROFILE:-cuda$(l3_version_mm "$(l3_cuda_version)")-gcc${GCC_MM}-ompi$(echo "$OMPI_V" | tr -d .)}"
SRC="$R/_upstream/level3/cp2k"; RUNS="$R/build/level3/cp2k/$PROFILE/$L3_RUN_SUBDIR"
export HPCPERF_GPUS="$N" HPCPERF_CPUS_PER_RANK="$T" HPCPERF_SCALE_MODE=smoke
mkdir -p "$RUNS"; ok=1
fail() { echo "validate.sh: FAIL -- $*"; ok=0; }
manifest_val() { /usr/bin/grep -m1 "^$2=" "$1/run_manifest.txt" 2>/dev/null | cut -d= -f2- || true; }

echo "validate.sh: [1] CP2K $BACKEND upstream regression tests (adapted subset) on $N GPU(s) x $T threads [profile $PROFILE]"
for rt in $REGTESTS; do
    dir="$(dirname "$rt")"; file="$(basename "$rt")"; toml="$SRC/tests/$dir/TEST_FILES.toml"
    [ -f "$toml" ] || { fail "$rt: $toml missing"; continue; }
    spec="$(python3 - "$toml" "$file" <<'PY'
import sys, re, tomllib
d = tomllib.load(open(sys.argv[1], "rb")); e = d.get(sys.argv[2])
if not e: sys.exit(1)
for m in e: print(m["matcher"], repr(m["tol"]), repr(m["ref"]))
PY
)" || { fail "$rt: not listed in $toml"; continue; }
    label="regtest.$(echo "$rt" | tr '/' '_' | sed 's/\.inp$//')"; D="$RUNS/$label.smoke.np$N.t$T"
    rc=0; HPCPERF_CP2K_CASE=regtest HPCPERF_CP2K_REGTEST="$rt" timeout "$TIMEOUT" "$HERE/run.sh" "$BACKEND" > "$RUNS/validate.$label.np$N.stdout" 2>&1 || rc=$?
    if [ "$rc" -eq 124 ]; then fail "$rt timed out"; continue; fi
    [ "$rc" -eq 0 ] || { fail "$rt: run.sh exited $rc (see $RUNS/validate.$label.np$N.stdout)"; continue; }
    [ -f "$D/cp2k.out" ] || { fail "$rt: no cp2k.out"; continue; }
    while read -r matcher tol ref; do
        python3 - "$D/cp2k.out" "$matcher" "$tol" "$ref" "$rt" <<'PY' || ok=0
import sys, re, os
sys.path.insert(0, os.environ["L3_TOOLS"])
from l3_check import require_finite, ValidationError
out, matcher, tol, ref, name = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
patterns = {"E_total": ("Total energy:", 3), "M011": ("ENERGY| Total FORCE_EVAL", 9), "M002": ("MD| Potential energy", 5), "M007": ("OPT| Total energy [hartree]", 5)}
if matcher not in patterns:
    print(f"  VALIDATION ERROR: matcher {matcher} not implemented in this adapted subset"); sys.exit(1)
pat, col = patterns[matcher]
val = None
for line in reversed(open(out, errors="replace").read().split("\n")):
    if pat in line:
        val = float(line.split()[col - 1]); break
try:
    if val is None: raise ValidationError(f"{name}: pattern {pat!r} not found in output")
    require_finite(f"{name} {matcher}", val)
    err = abs(val - ref)
    print(f"    {name}: {matcher} = {val:.14f} ref {ref:.14f} |diff| = {err:.2e} tol {tol:.0e} {'ok' if err <= tol else 'BAD'}")
    sys.exit(0 if err <= tol else 1)
except ValidationError as ex:
    print(f"  VALIDATION ERROR: {ex}"); sys.exit(1)
PY
    done <<< "$spec"
    /usr/bin/grep -aq 'PROGRAM ENDED AT' "$D/cp2k.out" || fail "$rt: output lacks 'PROGRAM ENDED AT' (incomplete run)"
done

echo "validate.sh: [2] CP2K $BACKEND H2O-64 GPW-DFT NVE MD, 10 steps, on $N GPU(s) x $T threads"
D="$RUNS/h2o64.smoke.np$N.t$T"
rc=0; HPCPERF_CP2K_CASE=h2o HPCPERF_CP2K_SYSTEM=64 timeout "$TIMEOUT" "$HERE/run.sh" "$BACKEND" > "$RUNS/validate.h2o64.np$N.stdout" 2>&1 || rc=$?
/usr/bin/grep -aE '^# CP2K|hpcperf-launch: audit summary' "$RUNS/validate.h2o64.np$N.stdout" || true
if [ "$rc" -eq 124 ]; then fail "h2o64 timed out after ${TIMEOUT}s"; elif [ "$rc" -ne 0 ]; then fail "h2o64 run.sh exited $rc"; fi
if [ -f "$D/cp2k.out" ]; then
    /usr/bin/grep -aE 'DBCSR\| ACC: (Number of devices|GPU backend)' "$D/cp2k.out" | head -2 | sed 's/^/    gpu: /' || true
    python3 "$HERE/cp2k_md_summary.py" "$D/cp2k.out" "$D/input.inp" --check > "$D/md_summary.txt" || { fail "h2o64 np$N: MD completeness/convergence/finiteness/GPU-evidence check failed"; }
    sed 's/^/    /' "$D/md_summary.txt"
    if [ "$N" -gt 1 ]; then
        REF="$RUNS/h2o64.smoke.np1.t$T"
        if [ ! -f "$REF/md_summary.txt" ] || [ "$(manifest_val "$REF" binary_sha256)" != "$(manifest_val "$D" binary_sha256)" ]; then
            echo "    (1-GPU reference missing/stale -- running H2O-64 on 1 GPU now)"
            rc=0; HPCPERF_GPUS=1 HPCPERF_CP2K_CASE=h2o HPCPERF_CP2K_SYSTEM=64 timeout "$TIMEOUT" "$HERE/run.sh" "$BACKEND" > "$RUNS/validate.h2o64.np1.stdout" 2>&1 || rc=$?
            [ "$rc" -eq 0 ] || fail "h2o64 1-GPU reference run exited $rc"
            if [ -f "$REF/cp2k.out" ]; then python3 "$HERE/cp2k_md_summary.py" "$REF/cp2k.out" "$REF/input.inp" --check > "$REF/md_summary.txt" || fail "h2o64 1-GPU reference run failed its own checks"; fi
        fi
        python3 - "$REF/md_summary.txt" "$D/md_summary.txt" "$E_TOL" <<'PY' || fail "h2o64 np$N: MD energies differ from the 1-GPU run beyond $E_TOL Ha"
import sys, re
def load(p):
    d = {}
    for l in open(p):
        for k, v in re.findall(r"(\w+)=(-?[0-9.]+(?:e[-+]?\d+)?)", l): d[k] = float(v)
    return d
a, b, tol = load(sys.argv[1]), load(sys.argv[2]), float(sys.argv[3])
keys = [f"efe_step{i}" for i in range(1, 11)]
missing = [k for k in keys if k not in a or k not in b]
if missing:
    print(f"    missing energies in summaries: {missing}"); sys.exit(1)
diffs = {k: abs(a[k] - b[k]) for k in keys}
worst = max(diffs, key=diffs.get)
for k in ("efe_step1", "efe_step10", worst) if worst not in ("efe_step1", "efe_step10") else ("efe_step1", "efe_step10"):
    d = diffs[k]
    print(f"    {k}: 1-GPU {a[k]:.12f}  N-GPU {b[k]:.12f}  |diff| = {d:.3e} Ha (tol {tol:.0e} {'ok' if d <= tol else 'BAD'}; upstream CPU cross-layout 1e-10: {'ok' if d <= 1e-10 else 'exceeded'})")
print(f"    max |diff| over MD steps 1..10: {diffs[worst]:.3e} Ha ({worst}); initial-SCF energy |diff| = {abs(a.get('efe_step0', 0) - b.get('efe_step0', 0)):.3e} Ha (informational)")
sys.exit(0 if all(d <= tol for d in diffs.values()) else 1)
PY
    fi
fi

if [ "$ok" -eq 1 ]; then echo "CP2K $BACKEND validation ($N GPU x $T threads; upstream regtests [$REGTESTS] within upstream tolerances, H2O-64 MD complete/all MD-step SCFs converged/finite, GPU backends active$( [ "$N" -gt 1 ] && echo ", MD-step FORCE_EVAL energies within $E_TOL Ha of 1-GPU" || true)): PASS"; exit 0; fi
echo "CP2K $BACKEND validation ($N GPU): FAIL"; exit 1
