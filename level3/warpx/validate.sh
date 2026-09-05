#!/usr/bin/env bash
# Correctness check for WarpX on N GPUs, using upstream's analytic Langmuir-wave
# regression test plus an invariant of the performance case.
#
#   ./validate.sh [CUDA|HIP]        HPCPERF_GPUS=N (default 1)
#
# [1] test_3d_langmuir_multi (Examples/Tests/langmuir): an electron/positron
#     plasma wave whose fields are known analytically,
#       Ex = eps m_e c^2 kx/e sin(kx x) cos(ky y) cos(kz z) sin(wp t)  (and cyclic),
#     64^3 cells, 40 steps. Upstream's analysis_3d.py compares the cell-centred
#     Ex/Ey/Ez of the final plotfile with this solution and requires
#     max|E_sim - E_th| / max|E_th| < 5e-2 for each component, and (Esirkepov
#     deposition) charge conservation max|divE - rho/eps0| / max|rho/eps0| <
#     1e-11. The same checks are re-implemented here (upstream's script needs
#     yt/openPMD-viewer, not available in this environment): the plotfile is
#     read directly (AMReX native format), the formulas, grid positions and
#     tolerances are upstream's. This is architecture-independent, unlike
#     upstream's checksum baselines (documented as platform-dependent).
# [2] uniform_plasma smoke run (the performance case): the macroparticle count
#     must be constant at every step (periodic box, no ionisation) -- exact; the
#     particle+field energy time series is recorded for information (the
#     shipped 2-particles-per-cell thermal plasma with E=0 initial fields is
#     not an energy-conservation test: a few % change over the first plasma
#     periods is expected for the momentum-conserving Yee scheme).
# PASS = [1] both criteria on N GPUs and [2] exact particle conservation.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u     # conda python3 + numpy for the analysis
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
N="${HPCPERF_GPUS:-1}"
RUNS="$R/build/level3/warpx/$MODEL/run"
python3 -c 'import numpy' 2>/dev/null || { echo "validate.sh: python3 with numpy required for the plotfile analysis" >&2; exit 1; }
export HPCPERF_GPUS="$N"
ok=1

echo "validate.sh: [1] WarpX $BACKEND langmuir_multi (64^3, 40 steps, analytic solution) on $N GPU(s)"
HPCPERF_WARPX_CASE=langmuir "$HERE/run.sh" "$BACKEND" 2>&1 | grep -E '^#|hpcperf-launch: audit summary|Total Time|ERROR|abort' || true
PLT="$RUNS/langmuir.validate.np$N/diags/diag1000040"
[ -f "$PLT/Header" ] || { echo "validate.sh: FAIL -- plotfile $PLT not produced"; exit 1; }
python3 - "$PLT" <<'PY' || ok=0
import sys, re, numpy as np
# WarpX's own constants (Source/ablastr/constant.H, CODATA 2022; scipy >= 1.15 as used by upstream's
# analysis_3d.py carries the same values). With CODATA 2018 eps0 the divE - rho/eps0 residual would show a
# spurious uniform 6.8e-10 offset (= the eps0 revision), 68x upstream's 1e-11 tolerance.
c, e, epsilon_0, m_e = 299792458.0, 1.602176634e-19, 8.8541878188e-12, 9.1093837139e-31
plt = sys.argv[1]
# ---- AMReX plotfile reader (single level, cell-centred data) ----
hdr = open(f"{plt}/Header").read().split("\n")
ncomp = int(hdr[1]); names = hdr[2:2 + ncomp]; i = 2 + ncomp
dim = int(hdr[i]); time = float(hdr[i + 1]); i += 3
lo = [float(v) for v in hdr[i].split()]; hi = [float(v) for v in hdr[i + 1].split()]
dom = re.search(r"\(\((\d+),(\d+),(\d+)\) \((\d+),(\d+),(\d+)\)", hdr[i + 3])
n = [int(dom.group(k + 4)) - int(dom.group(k + 1)) + 1 for k in range(3)]
ch = open(f"{plt}/Level_0/Cell_H").read().split("\n")
boxes = [tuple(int(v) for v in m.groups()) for m in re.finditer(r"\(\((-?\d+),(-?\d+),(-?\d+)\) \((-?\d+),(-?\d+),(-?\d+)\) \(", "\n".join(ch))]
fabs = [(m.group(1), int(m.group(2))) for m in re.finditer(r"FabOnDisk: (\S+) (\d+)", "\n".join(ch))]
assert len(boxes) == len(fabs) > 0, (len(boxes), len(fabs))
data = np.zeros((ncomp, n[0], n[1], n[2]))
for (lx, ly, lz, hx, hy, hz), (fname, off) in zip(boxes, fabs):
    with open(f"{plt}/Level_0/{fname}", "rb") as f:
        f.seek(off); line = b""
        while not line.endswith(b"\n"): line += f.read(1)
        h = line.decode()
        # "FAB ((8, (64 11 52 0 1 12 0 1023)),(8, (8 7 6 5 4 3 2 1)))((lo) (hi) (0,0,0)) ncomp": the second
        # descriptor is the byte order of the 8-byte reals (8 7 ... 1 = little endian)
        order = re.search(r"\(\d+, \((\d)(?: \d){7}\)\)\)", h).group(1)
        dt = "<f8" if order == "8" else ">f8"
        nc = int(h.strip().split()[-1])
        shape = (hx - lx + 1, hy - ly + 1, hz - lz + 1)
        arr = np.frombuffer(f.read(8 * nc * np.prod(shape)), dtype=dt).reshape((nc, shape[2], shape[1], shape[0])).transpose(0, 3, 2, 1)
        data[:, lx:hx + 1, ly:hy + 1, lz:hz + 1] = arr
comp = {nm: data[k] for k, nm in enumerate(names)}
# ---- upstream analysis_3d.py, verbatim parameters ----
epsilon, nden = 0.01, 4.0e24
Ncell = n
kx, ky, kz = [2.0 * np.pi * 2 / (hi[d] - lo[d]) for d in range(3)]
wp = np.sqrt(nden * e**2 / (m_e * epsilon_0))
k = {"Ex": kx, "Ey": ky, "Ez": kz}; cos = {"Ex": (0, 1, 1), "Ey": (1, 0, 1), "Ez": (1, 1, 0)}
def contrib(is_cos, kk, d):
    du = (hi[d] - lo[d]) / Ncell[d]; u = lo[d] + du * (0.5 + np.arange(Ncell[d]))
    return np.cos(kk * u) if is_cos else np.sin(kk * u)
def theory(field, t):
    amp = epsilon * (m_e * c**2 * k[field]) / e * np.sin(wp * t)
    cf = cos[field]
    return amp * contrib(cf[0], kx, 0)[:, None, None] * contrib(cf[1], ky, 1)[None, :, None] * contrib(cf[2], kz, 2)[None, None, :]
print(f"    plotfile time t = {time:.6e} s (wp t = {wp*time:.4f}), grid {n}, {len(boxes)} box(es), fields {names[:6]}...")
ok = True; err = 0.0
for fld in ("Ex", "Ey", "Ez"):
    th = theory(fld, time); m = abs(comp[fld] - th).max() / abs(th).max(); err = max(err, m)
    print(f"    {fld}: max|E_sim-E_th|/max|E_th| = {m:.3e}")
print(f"    error_rel = {err:.3e} (upstream tolerance_rel 5e-2) {'ok' if err < 5e-2 else 'BAD'}"); ok &= err < 5e-2
rho, divE = comp["rho"], comp["divE"]
ce = np.amax(np.abs(divE - rho / epsilon_0)) / np.amax(np.abs(rho / epsilon_0))
print(f"    charge conservation max|divE-rho/eps0|/max|rho/eps0| = {ce:.3e} (upstream tolerance 1e-11) {'ok' if ce < 1e-11 else 'BAD'}"); ok &= ce < 1e-11
sys.exit(0 if ok else 1)
PY

echo "validate.sh: [2] WarpX $BACKEND uniform_plasma smoke (64x32x32, 131,072 particles) on $N GPU(s)"
HPCPERF_SCALE_MODE=smoke "$HERE/run.sh" "$BACKEND" 2>&1 | grep -E '^#|hpcperf-launch: audit summary|Total Time|ERROR|abort' || true
D="$RUNS/uniform_plasma.smoke.np$N/diags/reducedfiles"
[ -f "$D/NP.txt" ] || { echo "validate.sh: FAIL -- reduced diagnostics not produced under $D"; exit 1; }
python3 - "$D" <<'PY' || ok=0
import sys
d = sys.argv[1]
def load(p): return [[float(x) for x in l.split()] for l in open(p) if l.strip() and not l.startswith('#')]
npart = load(f"{d}/NP.txt"); ep = load(f"{d}/EP.txt"); ef = load(f"{d}/EF.txt")
vals = sorted(set(r[2] for r in npart))
print(f"    ParticleNumber over steps {int(npart[0][0])}..{int(npart[-1][0])}: {vals} -> {'ok (exact)' if len(vals) == 1 else 'BAD'}")
e0 = ep[0][2] + ef[0][2]; e1 = ep[-1][2] + ef[-1][2]
print(f"    for the record: E_particles+E_fields = {e0:.6e} J at step {int(ep[0][0])}, {e1:.6e} J at step {int(ep[-1][0])} (rel change {(e1-e0)/e0:+.3e}; not a pass/fail criterion, see header)")
sys.exit(0 if len(vals) == 1 else 1)
PY

if [ "$ok" -eq 1 ]; then echo "WarpX $BACKEND validation ($N GPU, langmuir_multi analytic + charge conservation, particle conservation): PASS"; exit 0; fi
echo "WarpX $BACKEND validation ($N GPU): FAIL"; exit 1
