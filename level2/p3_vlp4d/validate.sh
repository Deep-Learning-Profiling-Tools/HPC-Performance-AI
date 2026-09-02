#!/usr/bin/env bash
# Validate the P3-miniapps vlp4d build with the 2D Landau damping test case.
#
#   ./validate.sh [CUDA|HIP]
#
# Runs wk/SLD10_large.dat (test case 10: 2D Landau damping, k_x = k_y = 0.5,
# alpha = 0.05, 128^4 points, dt = 0.125, 128 steps, t_end = 16), which is the
# standard problem of the upstream docs and of Crouseilles, Latu, Sonnendruecker,
# JCP 228 (2009) sect. 5.3.1. vlp4d has no built-in reference check; its only
# diagnostic is nrj.out with columns  t  log(||E||_2)  sum(phi)  (the third
# column is ~1e-15 by construction and is NOT a mass-conservation measure).
#
# What is checked: the electric-field energy history must follow linear Landau
# theory. For a Maxwellian with v_th = 1 the perturbed (0.5, 0.5) mode has
# |k| = 0.5*sqrt(2), and the root of the dispersion relation
#   1 - Z'(omega/(sqrt(2)|k|)) / (2|k|^2) = 0     (Z = plasma dispersion function)
# is omega = 1.6829 - 0.4021 i. Hence log||E||_2 must decay with envelope slope
# gamma = -0.4021 and oscillate with angular frequency 1.6829 (period of the
# maxima of ||E|| = pi/omega_r = 1.867). The script
#   1. checks the run exits 0 and nrj.out has 129 finite lines,
#   2. finds the local maxima of log||E||_2 (needs >= 5),
#   3. least-squares fits a line through them: slope must match gamma to 5 %,
#   4. checks pi*(n_max-1)/(t_last-t_first) matches omega_r to 5 %.
# Observed here: slope = -0.4018 (0.1 % off), omega_r = 1.6755 (0.4 % off; the
# maxima are quantised to the dt = 0.125 output cadence). A wrong advection,
# interpolation, or Poisson solve would change the damping rate by O(1) or
# make the field grow, so 5 % is a robust threshold that still tolerates
# GPU-to-GPU reduction/FFT round-off differences (~1e-12).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$R" ] && [ -f "$R/hpcperf_env.sh" ]; then
    set +eu
    # shellcheck disable=SC1091
    source "$R/hpcperf_env.sh" 2>/dev/null
    set -eu
fi

BACKEND="$(printf '%s' "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
case "$BACKEND" in
    CUDA|HIP) ;;
    *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac

BUILD_DIR="$R/build/level2/p3_vlp4d/$(printf '%s' "$BACKEND" | tr '[:upper:]' '[:lower:]')"
EXE="$BUILD_DIR/miniapps/vlp4d/thrust/vlp4d"
if [ ! -x "$EXE" ]; then
    echo "FAIL: $EXE not found; run ./build.sh $BACKEND first"
    exit 1
fi

DECK="$HERE/wk/SLD10_large.dat"
RUN_DIR="$BUILD_DIR/validate"
mkdir -p "$RUN_DIR/data/vlp4d"
cd "$RUN_DIR"
rm -f nrj.out
LOG="$RUN_DIR/vlp4d.log"

GAMMA_REF=-0.40208      # linear Landau damping rate for |k| = 0.5*sqrt(2), v_th = 1
OMEGA_REF=1.68289       # real frequency of the same root
TOL=0.05                # relative tolerance on both
NBITER=128

echo "== vlp4d ($BACKEND): $EXE $DECK   (cwd: $RUN_DIR, log: $LOG)"
rc=0
"$EXE" "$DECK" >"$LOG" 2>&1 || rc=$?
grep -E '^(total|MainLoop) ' "$LOG" | sed 's/^/   /' || true
if [ "$rc" -ne 0 ]; then
    echo "   run failed (exit code $rc), tail of log:"; tail -5 "$LOG"
    echo "FAIL: p3_vlp4d ($BACKEND) run failed"
    exit 1
fi
if [ ! -s nrj.out ]; then
    echo "FAIL: p3_vlp4d ($BACKEND) produced no nrj.out"
    exit 1
fi

if awk -v gref="$GAMMA_REF" -v oref="$OMEGA_REF" -v tol="$TOL" -v nb="$NBITER" '
    function abs(x) { return x < 0 ? -x : x }
    {
        n = NR; t[n] = $1 + 0; e[n] = $2 + 0
        if ($2 !~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) bad++
    }
    END {
        pi = 3.14159265358979
        status = 0
        printf("   nrj.out: %d lines (expected %d), non-finite log||E|| values: %d\n", n, nb + 1, bad + 0)
        if (n != nb + 1 || bad > 0) status = 1
        m = 0
        for (i = 2; i < n; i++) if (e[i] > e[i-1] && e[i] >= e[i+1]) { m++; tm[m] = t[i]; em[m] = e[i] }
        printf("   local maxima of log||E||_2: %d (t = %g .. %g)\n", m, (m ? tm[1] : 0), (m ? tm[m] : 0))
        if (m < 5) { print "   too few maxima to fit a damping rate"; exit 1 }
        sx = sy = sxx = sxy = 0
        for (j = 1; j <= m; j++) { sx += tm[j]; sy += em[j]; sxx += tm[j]*tm[j]; sxy += tm[j]*em[j] }
        slope = (m*sxy - sx*sy) / (m*sxx - sx*sx)
        omega = pi * (m - 1) / (tm[m] - tm[1])
        rg = abs(slope - gref) / abs(gref); ro = abs(omega - oref) / oref
        printf("   damping rate  gamma = %.4f   linear theory %.4f   rel. diff %.3f   (tol %g)\n", slope, gref, rg, tol)
        printf("   frequency   omega_r = %.4f   linear theory %.4f   rel. diff %.3f   (tol %g)\n", omega, oref, ro, tol)
        printf("   log||E||_2: t=0 -> %.4f, t=%g -> %.4f\n", e[1], t[n], e[n])
        if (rg > tol || ro > tol) status = 1
        exit status
    }' nrj.out; then
    echo "PASS: p3_vlp4d ($BACKEND) Landau damping rate and frequency of ||E||_2 match linear theory (gamma=$GAMMA_REF, omega=$OMEGA_REF, rel tol $TOL) for SLD10_large.dat"
    exit 0
else
    echo "FAIL: p3_vlp4d ($BACKEND) Landau damping check failed (see $RUN_DIR/nrj.out and $LOG)"
    exit 1
fi
