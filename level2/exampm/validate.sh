#!/usr/bin/env bash
# Validate the ExaMPM CUDA or HIP build.
#
#   ./validate.sh [CUDA|HIP]
#
# ExaMPM has no built-in correctness check and upstream ships no reference
# output, so this script checks the simulation against an analytic solution:
#
# 1. Free fall (primary, always run). Upstream's second example, FreeFall, is
#    a sphere of fluid (r = 0.25 m, rho = 1000 kg/m^3) at rest at the origin
#    of a fully periodic 1 m^3 box with gravity g = 9.81 m/s^2 in -z and no
#    walls. Nothing acts on the fluid but gravity, so its centre of mass is
#    in free fall:  v_cm,z(t) = -g t,  z_cm(t) = z0 - g t^2 / 2,  v_cm,x = v_cm,y = 0.
#    ExaMPM's APIC transfers (P2G, G2P) conserve linear momentum exactly and
#    gravity is applied to the grid velocity as g*dt, so the mean particle
#    velocity must match -g t to round-off. Positions are updated with
#    symplectic Euler, x_{n+1} = x_n + v_{n+1} dt_n, so the discrete centre
#    of mass is z0 - (g/2) sum_n dt_n (t_n + dt_n) = z0 - g t^2/2 - (g/2) sum dt_n^2,
#    i.e. it lags the continuous solution by at most (g/2) t dt_0 (dt_0 is
#    the initial dt; the CFL controller can only shrink it).
#    Every HDF5 dump particles_<step>.h5 (position Nx3, velocity Nx3,
#    attribute Time) written by the run is checked:
#      - particle count N equals the initial count            (exact)
#      - |mean(v_z) + g t|  <= 1e-9 * max(1, g t)             (round-off)
#      - |mean(v_x)|, |mean(v_y)| <= 1e-9                     (round-off)
#      - |z_cm - (z0 - g t^2/2)| <= g t dt_0 + 1e-12          (2x the
#        symplectic-Euler bound; z is unwrapped through the periodic
#        boundary particle by particle before averaging)
#      - the last dump is at t >= 0.9 t_final (the run completed)
#    The sphere falls 0.31 m by t = 0.25 s and crosses the periodic z
#    boundary, so the check exercises the grid halo exchange and the particle
#    migration through MPI (device buffers, see run.sh) as well as the
#    kernels. Deck: FreeFall 0.05 2 0 0.001 0.25 50 <exec_space>
#    (20^3 cells, 4,224 particles, 400 steps, 9 dumps, ~6 s on a B200).
#
# 2. Dam break (secondary, only if run.sh output exists in
#    $R/build/level2/exampm/<cuda|hip>/run). The benchmark deck is a
#    weakly compressible water column collapsing in a closed 1 m^3 box
#    (FREE_SLIP walls), so across all its dumps:
#      - particle count N is constant, all positions finite    (exact)
#      - all positions inside [-0.02, 1.02]^3 (particles may overshoot the
#        wall by a fraction of a cell before the boundary velocity and
#        the position correction push them back)
#      - total volume sum(J_p) (J = det F, volume ratio per particle) stays
#        within 1 % of its initial value N (near-incompressible fluid)
#
# The HDF5 files are read with h5dump (binary export) + numpy, since h5py is
# not in the conda environment. Prints PASS/FAIL, exits 0/1.
#
# Environment overrides:
#   HPCPERF_EXAMPM_FF_ARGS   the six FreeFall arguments (default
#                            "0.05 2 0 0.001 0.25 50")

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true

set -uo pipefail

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level2/exampm/$MODEL"
VAL_DIR="$BUILD_DIR/validate"
RUN_DIR="$BUILD_DIR/run"
mkdir -p "$VAL_DIR"
LOG="$VAL_DIR/validate.log"

read -r -a FF_ARGS <<< "${HPCPERF_EXAMPM_FF_ARGS:-0.05 2 0 0.001 0.25 50}"
if [ "${#FF_ARGS[@]}" -ne 6 ]; then
    echo "validate.sh: HPCPERF_EXAMPM_FF_ARGS must hold six values" >&2
    exit 2
fi

HPCPERF_EXAMPM_EXAMPLE=FreeFall HPCPERF_EXAMPM_RUN_DIR="$VAL_DIR" \
    "$HERE/run.sh" "$BACKEND" "${FF_ARGS[@]}" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
    echo "ExaMPM $BACKEND validation: FAIL (FreeFall exited with code $rc)" | tee -a "$LOG"
    exit 1
fi

echo "----------------------------------------------------------------" | tee -a "$LOG"
python3 - "$VAL_DIR" "$RUN_DIR" "${FF_ARGS[3]}" "${FF_ARGS[4]}" <<'PYEOF' 2>&1 | tee -a "$LOG"
import glob, os, re, subprocess, sys
import numpy as np

val_dir, run_dir, dt0, t_final = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
G = 9.81       # examples/free_fall.cpp: gravity
L = 1.0        # examples/free_fall.cpp: global_box = [-0.5, 0.5]^3
ok = True

def fail(msg):
    global ok
    ok = False
    print("  FAIL:", msg)

def h5time(f):
    out = subprocess.run(["h5dump", "-m", "%.17g", "-a", "/Time", f],
                         capture_output=True, text=True, check=True).stdout
    return float(re.search(r"\(0\):\s*([-+0-9.eE]+)", out).group(1))

def h5arr(f, ds):
    tmp = f + "." + ds + ".bin"
    subprocess.run(["h5dump", "-d", "/" + ds, "-b", "LE", "-o", tmp, f],
                   capture_output=True, check=True)
    a = np.fromfile(tmp, dtype="<f8")
    os.remove(tmp)
    return a

def dumps(d):
    fs = glob.glob(os.path.join(d, "particles_*.h5"))
    return sorted(fs, key=lambda s: int(re.search(r"_(\d+)\.h5$", s).group(1)))

# ---- 1. free fall vs analytic solution -------------------------------------
files = dumps(val_dir)
print(f"FreeFall: {len(files)} dumps in {val_dir}; g={G}, dt0={dt0}, t_final={t_final}")
if len(files) < 2:
    fail("expected at least two particles_*.h5 dumps (initial + one more)")
n0 = z0 = None
t = 0.0
print(f"  {'file':18s} {'t':>9s} {'N':>7s} {'mean vz':>12s} {'-g t':>12s} {'|dvz|':>9s} {'z_cm-z_exact':>13s} {'tol':>9s}")
for f in files:
    t = h5time(f)
    pos = h5arr(f, "position").reshape(-1, 3)
    vel = h5arr(f, "velocity").reshape(-1, 3)
    n = len(pos)
    if n0 is None:
        n0, z0 = n, pos[:, 2].mean()
    vz = vel[:, 2].mean()
    vx, vy = vel[:, 0].mean(), vel[:, 1].mean()
    z_exact = z0 - 0.5 * G * t * t
    dz = ((pos[:, 2] - z_exact + 0.5 * L) % L) - 0.5 * L   # periodic unwrap
    z_err = dz.mean()
    z_tol = G * t * dt0 + 1e-12
    v_tol = 1e-9 * max(1.0, G * t)
    print(f"  {os.path.basename(f):18s} {t:9.6f} {n:7d} {vz:+12.8f} {-G*t:+12.8f} {abs(vz+G*t):9.2e} {z_err:+13.3e} {z_tol:9.2e}")
    if not (np.isfinite(pos).all() and np.isfinite(vel).all()):
        fail(f"{f}: non-finite position/velocity")
    if n != n0:
        fail(f"{f}: particle count {n} != initial {n0}")
    if abs(vz + G * t) > v_tol:
        fail(f"{f}: mean v_z {vz:.10f} vs -g t {-G*t:.10f} (tol {v_tol:.1e})")
    if abs(vx) > 1e-9 or abs(vy) > 1e-9:
        fail(f"{f}: mean v_x, v_y = {vx:.2e}, {vy:.2e} (tol 1e-9)")
    if abs(z_err) > z_tol:
        fail(f"{f}: centre of mass off by {z_err:.3e} (tol {z_tol:.1e})")
if files and t < 0.9 * t_final:
    fail(f"last dump at t={t} < 0.9 * t_final={t_final}: run incomplete")
print("  FreeFall centre-of-mass check:", "ok" if ok else "FAILED")

# ---- 2. dam break conservation (optional) ----------------------------------
files = dumps(run_dir)
if not files:
    print(f"DamBreak: no dumps in {run_dir} (run ./run.sh first) -- skipped")
else:
    ok_db = True
    n0 = None
    print(f"DamBreak: {len(files)} dumps in {run_dir}")
    print(f"  {'file':18s} {'t':>9s} {'N':>8s} {'min pos':>10s} {'max pos':>10s} {'sum J / N':>10s}")
    for f in files:
        t = h5time(f)
        pos = h5arr(f, "position").reshape(-1, 3)
        J = h5arr(f, "J")
        n = len(pos)
        if n0 is None:
            n0 = n
        volr = J.sum() / n0
        print(f"  {os.path.basename(f):18s} {t:9.6f} {n:8d} {pos.min():+10.5f} {pos.max():+10.5f} {volr:10.6f}")
        if n != n0:
            ok_db = False; fail(f"{f}: particle count {n} != initial {n0}")
        if not np.isfinite(pos).all():
            ok_db = False; fail(f"{f}: non-finite positions")
        if pos.min() < -0.02 or pos.max() > 1.02:
            ok_db = False; fail(f"{f}: particles outside [-0.02, 1.02]^3")
        if abs(volr - 1.0) > 0.01:
            ok_db = False; fail(f"{f}: total volume sum(J)/N = {volr:.5f} off by more than 1 %")
    print("  DamBreak conservation check:", "ok" if ok_db else "FAILED")

sys.exit(0 if ok else 1)
PYEOF
rc=${PIPESTATUS[0]}

echo "----------------------------------------------------------------" | tee -a "$LOG"
if [ "$rc" -eq 0 ]; then
    echo "ExaMPM $BACKEND validation (FreeFall ${FF_ARGS[*]} vs analytic free fall): PASS" | tee -a "$LOG"
    exit 0
else
    echo "ExaMPM $BACKEND validation (FreeFall ${FF_ARGS[*]} vs analytic free fall): FAIL (exit code $rc)" | tee -a "$LOG"
    exit 1
fi
