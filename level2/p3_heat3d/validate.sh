#!/usr/bin/env bash
# Validate the P3-miniapps heat3d build against the analytical solution.
#
#   ./validate.sh [CUDA|HIP]
#
# heat3d integrates the 3D heat equation with a periodic single-mode initial
# condition u0 = cos(2*pi*(x+y+z)/L), for which the exact solution is
#   u(t) = u0 * exp(-3*kappa*(2*pi/L)^2 * t),
# and prints at the end the L2 norm of (numerical - analytical):
#   L2_norm: <value>
# Because u0 is an eigenmode of the discrete 7-point Laplacian, the FTCS
# scheme also has a closed form, u_n = A^n * u0 with
#   A = 1 + coef*(6*cos(2*pi*dx) - 6),   coef = kappa*dt/dx^2 = 0.1,
# so the printed value can be predicted exactly:
#   L2_norm = |A^n - exp(-3*kappa*(2*pi)^2*n*dt)| * sqrt(N^3/2).
# Two cases are run and compared against these closed-form values (which for
# 512^3/1000 coincide with the upstream-documented output "L2_norm: 0.00355178"
# in docs/heat3d.md):
#   128^3, 1000 steps  ->  0.0577255
#   512^3, 1000 steps  ->  0.00355178      (standard benchmark problem)
# Relative tolerance 1e-3 (the printed value has 6 significant digits; the GPU
# reduction order only perturbs the ~1e-12 level). A wrong stencil, wrong
# time step or wrong boundary treatment changes the value at the 1e-1 level.
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

BUILD_DIR="$R/build/level2/p3_heat3d/$(printf '%s' "$BACKEND" | tr '[:upper:]' '[:lower:]')"
EXE="$BUILD_DIR/miniapps/heat3d/thrust/heat3d"
if [ ! -x "$EXE" ]; then
    echo "FAIL: $EXE not found; run ./build.sh $BACKEND first"
    exit 1
fi

RUN_DIR="$BUILD_DIR/validate"
mkdir -p "$RUN_DIR/data/heat3d"
cd "$RUN_DIR"

TOL=1e-3
status=0
# name  nx  nbiter  expected L2_norm (closed form / upstream reference)
while read -r name n nbiter expected; do
    log="$RUN_DIR/$name.log"
    echo "== heat3d ($BACKEND) $name: --nx $n --ny $n --nz $n --nbiter $nbiter --freq_diag 0  (log: $log)"
    rc=0
    "$EXE" --nx "$n" --ny "$n" --nz "$n" --nbiter "$nbiter" --freq_diag 0 >"$log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "   run failed (exit code $rc), tail of log:"; tail -5 "$log"
        status=1; continue
    fi
    got="$(grep -E '^L2_norm:' "$log" | tail -1 | awk '{print $2}')"
    if [ -z "$got" ]; then
        echo "   no 'L2_norm:' line in output"; status=1; continue
    fi
    if awk -v g="$got" -v e="$expected" -v tol="$TOL" 'BEGIN {
            d = g - e; if (d < 0) d = -d; rel = d / e;
            printf("   L2_norm = %s   expected %s   rel. diff %.2e   (tol %s)\n", g, e, rel, tol);
            exit (rel <= tol) ? 0 : 1 }'; then
        :
    else
        status=1
    fi
    grep -E '^(Elapsed time|Bandwidth|Flops):' "$log" | sed 's/^/   /' || true
done <<'CASES'
n128_it1000 128 1000 0.0577255
n512_it1000 512 1000 0.00355178
CASES

if [ "$status" -eq 0 ]; then
    echo "PASS: p3_heat3d ($BACKEND) L2_norm vs. analytical solution matches the closed-form/upstream reference (rel tol $TOL) for 128^3 and 512^3, 1000 steps"
else
    echo "FAIL: p3_heat3d ($BACKEND) L2_norm check failed (see logs in $RUN_DIR)"
fi
exit "$status"
