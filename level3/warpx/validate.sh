#!/usr/bin/env bash
# Correctness check for WarpX on N GPUs, using upstream's analytic Langmuir-wave
# regression test plus an invariant of the performance case.
#
#   ./validate.sh [CUDA|HIP]        HPCPERF_GPUS=N (default 1)
#
# [1] test_3d_langmuir_multi: an e-/e+ plasma wave with an analytic field
#     solution. Adapted subset of upstream analysis_3d.py (which needs
#     yt/openPMD-viewer, not installed): the final plotfile's Ex/Ey/Ez are
#     compared with the exact solution -- require max|E_sim-E_th|/max|E_th|<5e-2
#     each -- and (Esirkepov) charge conservation max|divE-rho/eps0|/
#     max|rho/eps0|<1e-11, with WarpX's own CODATA-2022 constants. Reader is
#     hardened: it FAILs on a missing/short plotfile, a box set that does not
#     cover the domain (truncation), or any non-finite field value.
# [2] uniform_plasma smoke: macroparticle count constant AND finite at every
#     step, and the run must reach the final step; the energy series is recorded
#     for information only (not a pass/fail criterion, see header of run.sh).
# Reproducibility: the run's real exit code is captured (nonzero/timeout ->
# FAIL); run.sh writes a fresh per-run directory (dry-run never touches it).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
N="${HPCPERF_GPUS:-1}"
PROFILE="$(l3_backend_profile WARPX "$MODEL")"
l3_paths_profile warpx "$PROFILE" "$MODEL" || exit 2     # the run tree of the SAME profile build.sh/run.sh use
RUNS="$L3_BUILD/$L3_RUN_SUBDIR"
TIMEOUT="${HPCPERF_VALIDATE_TIMEOUT:-1800}"
python3 -c 'import numpy' 2>/dev/null || { echo "validate.sh: python3 with numpy required for the plotfile analysis" >&2; exit 1; }
export HPCPERF_GPUS="$N"
ok=1

run_case() { local case=$1 mode=$2 out=$3 rc=0; HPCPERF_WARPX_CASE="$case" HPCPERF_SCALE_MODE="$mode" timeout "$TIMEOUT" "$HERE/run.sh" "$BACKEND" > "$out" 2>&1 || rc=$?; return $rc; }

echo "validate.sh: [1] WarpX $BACKEND langmuir_multi (64^3, 40 steps, analytic solution) on $N GPU(s) [profile $PROFILE]"
mkdir -p "$RUNS"; L1="$RUNS/validate.langmuir.np$N.stdout"
rc=0; run_case langmuir validate "$L1" || rc=$?
grep -aE '^#|hpcperf-launch: audit summary|Total Time|ERROR|abort' "$L1" || true
if [ "$rc" -eq 124 ]; then echo "validate.sh: FAIL -- langmuir run timed out after ${TIMEOUT}s"; exit 1; fi
[ "$rc" -eq 0 ] || { echo "validate.sh: FAIL -- langmuir run.sh exited $rc (see $L1)"; exit 1; }
PLT="$RUNS/langmuir.validate.np$N/diags/diag1000040"
[ -f "$PLT/Header" ] || { echo "validate.sh: FAIL -- plotfile $PLT not produced (run did not reach step 40)"; exit 1; }
python3 - "$PLT" <<'PY' || ok=0
import sys, re, os
import numpy as np
sys.path.insert(0, os.environ["L3_TOOLS"])
from l3_check import require_finite, ValidationError
c, e, epsilon_0, m_e = 299792458.0, 1.602176634e-19, 8.8541878188e-12, 9.1093837139e-31
plt = sys.argv[1]
try:
    hdr = open(f"{plt}/Header").read().split("\n")
    ncomp = int(hdr[1]); names = hdr[2:2 + ncomp]; i = 2 + ncomp
    time = float(hdr[i + 1]); i += 3
    lo = [float(v) for v in hdr[i].split()]; hi = [float(v) for v in hdr[i + 1].split()]
    dom = re.search(r"\(\((\d+),(\d+),(\d+)\) \((\d+),(\d+),(\d+)\)", hdr[i + 3])
    n = [int(dom.group(k + 4)) - int(dom.group(k + 1)) + 1 for k in range(3)]
    for fld in ("Ex", "Ey", "Ez", "rho", "divE"):
        if fld not in names: raise ValidationError(f"plotfile missing field '{fld}' (have {names})")
    ch = open(f"{plt}/Level_0/Cell_H").read().split("\n")
    boxes = [tuple(int(v) for v in m.groups()) for m in re.finditer(r"\(\((-?\d+),(-?\d+),(-?\d+)\) \((-?\d+),(-?\d+),(-?\d+)\) \(", "\n".join(ch))]
    fabs = [(m.group(1), int(m.group(2))) for m in re.finditer(r"FabOnDisk: (\S+) (\d+)", "\n".join(ch))]
    if not (len(boxes) == len(fabs) > 0): raise ValidationError(f"plotfile box/fab mismatch ({len(boxes)} boxes, {len(fabs)} fabs)")
    data = np.full((ncomp, n[0], n[1], n[2]), np.nan)   # nan-init: any uncovered cell trips the finite check
    covered = np.zeros((n[0], n[1], n[2]), bool)
    for (lx, ly, lz, hx, hy, hz), (fname, off) in zip(boxes, fabs):
        with open(f"{plt}/Level_0/{fname}", "rb") as f:
            f.seek(off); line = b""
            while not line.endswith(b"\n"): line += f.read(1)
            h = line.decode()
            order = re.search(r"\(\d+, \((\d)(?: \d){7}\)\)\)", h).group(1)
            dt = "<f8" if order == "8" else ">f8"
            nc = int(h.strip().split()[-1])
            shape = (hx - lx + 1, hy - ly + 1, hz - lz + 1)
            raw = np.frombuffer(f.read(8 * nc * int(np.prod(shape))), dtype=dt)
            if raw.size != nc * int(np.prod(shape)): raise ValidationError(f"FAB {fname} truncated: {raw.size} of {nc*int(np.prod(shape))} reals")
            arr = raw.reshape((nc, shape[2], shape[1], shape[0])).transpose(0, 3, 2, 1)
            data[:, lx:hx + 1, ly:hy + 1, lz:hz + 1] = arr
            covered[lx:hx + 1, ly:hy + 1, lz:hz + 1] = True
    if not covered.all(): raise ValidationError(f"plotfile boxes cover only {covered.mean()*100:.1f}% of the {n} domain (missing boxes)")
    comp = {nm: data[k] for k, nm in enumerate(names)}
    epsilon, nden = 0.01, 4.0e24
    kx, ky, kz = [2.0 * np.pi * 2 / (hi[d] - lo[d]) for d in range(3)]
    wp = np.sqrt(nden * e**2 / (m_e * epsilon_0))
    kmap = {"Ex": kx, "Ey": ky, "Ez": kz}; cosf = {"Ex": (0, 1, 1), "Ey": (1, 0, 1), "Ez": (1, 1, 0)}
    def contrib(is_cos, kk, d):
        du = (hi[d] - lo[d]) / n[d]; u = lo[d] + du * (0.5 + np.arange(n[d]))
        return np.cos(kk * u) if is_cos else np.sin(kk * u)
    def theory(field, t):
        amp = epsilon * (m_e * c**2 * kmap[field]) / e * np.sin(wp * t); cf = cosf[field]
        return amp * contrib(cf[0], kx, 0)[:, None, None] * contrib(cf[1], ky, 1)[None, :, None] * contrib(cf[2], kz, 2)[None, None, :]
    require_finite("plotfile time", time)
    print(f"    plotfile time t = {time:.6e} s (wp t = {wp*time:.4f}), grid {n}, {len(boxes)} box(es), full coverage")
    err = 0.0
    for fld in ("Ex", "Ey", "Ez"):
        th = theory(fld, time)
        m = require_finite(f"max|{fld}_sim-{fld}_th|/max|{fld}_th|", abs(comp[fld] - th).max() / abs(th).max())
        err = max(err, m); print(f"    {fld}: max|E_sim-E_th|/max|E_th| = {m:.3e}")
    print(f"    error_rel = {err:.3e} (upstream tolerance_rel 5e-2) {'ok' if err < 5e-2 else 'BAD'}")
    ce = require_finite("charge-conservation residual", np.amax(np.abs(comp["divE"] - comp["rho"]/epsilon_0)) / np.amax(np.abs(comp["rho"]/epsilon_0)))
    print(f"    charge conservation max|divE-rho/eps0|/max|rho/eps0| = {ce:.3e} (upstream tolerance 1e-11) {'ok' if ce < 1e-11 else 'BAD'}")
    sys.exit(0 if (err < 5e-2 and ce < 1e-11) else 1)
except ValidationError as ex:
    print(f"  VALIDATION ERROR: {ex}"); sys.exit(1)
PY

echo "validate.sh: [2] WarpX $BACKEND uniform_plasma smoke (64x32x32, 131,072 particles) on $N GPU(s)"
L2="$RUNS/validate.uniform_plasma.np$N.stdout"
rc=0; run_case uniform_plasma smoke "$L2" || rc=$?
grep -aE '^#|hpcperf-launch: audit summary|Total Time|ERROR|abort' "$L2" || true
if [ "$rc" -eq 124 ]; then echo "validate.sh: FAIL -- uniform_plasma run timed out"; exit 1; fi
[ "$rc" -eq 0 ] || { echo "validate.sh: FAIL -- uniform_plasma run.sh exited $rc (see $L2)"; ok=0; }
D="$RUNS/uniform_plasma.smoke.np$N/diags/reducedfiles"
[ -f "$D/NP.txt" ] || { echo "validate.sh: FAIL -- reduced diagnostics not produced under $D"; exit 1; }
python3 - "$D" <<'PY' || ok=0
import sys, os
sys.path.insert(0, os.environ["L3_TOOLS"])
from l3_check import require_finite, ValidationError
d = sys.argv[1]
def load(p): return [[float(x) for x in l.split()] for l in open(p) if l.strip() and not l.startswith('#')]
try:
    npart, ep, ef = load(f"{d}/NP.txt"), load(f"{d}/EP.txt"), load(f"{d}/EF.txt")
    if not npart: raise ValidationError("NP.txt has no rows")
    steps = [int(r[0]) for r in npart]
    if steps[-1] != 10: raise ValidationError(f"uniform_plasma reached step {steps[-1]}, expected final step 10 (incomplete run)")
    vals = sorted(set(require_finite(f"Np@{int(r[0])}", r[2]) for r in npart))
    print(f"    ParticleNumber over steps {steps[0]}..{steps[-1]}: {vals} -> {'ok (exact, finite)' if len(vals) == 1 else 'BAD'}")
    e0 = require_finite("E0", ep[0][2] + ef[0][2]); e1 = require_finite("E1", ep[-1][2] + ef[-1][2])
    print(f"    for the record: E_particles+E_fields = {e0:.6e} J at step {int(ep[0][0])}, {e1:.6e} J at step {int(ep[-1][0])} (rel change {(e1-e0)/e0:+.3e}; informational, see header)")
    sys.exit(0 if len(vals) == 1 else 1)
except ValidationError as ex:
    print(f"  VALIDATION ERROR: {ex}"); sys.exit(1)
PY

if [ "$ok" -eq 1 ]; then echo "WarpX $BACKEND validation ($N GPU, langmuir_multi analytic + charge conservation, particle conservation): PASS"; exit 0; fi
echo "WarpX $BACKEND validation ($N GPU): FAIL"; exit 1
