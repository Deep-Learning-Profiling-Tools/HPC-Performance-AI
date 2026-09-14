#!/usr/bin/env bash
# Correctness check for GEOS (beam bending, hypre AMG on the GPU) on N GPUs.
#
#   ./validate.sh [CUDA]        HPCPERF_GPUS=N (default 1)
#
# Criteria (fixed before the runs; [2]-[4] are geos-ats's own checks for this test, re-implemented
# in geos_beam_check.py with the metrics of geosPythonPackages curve_check.py / restart_check.py and
# verified against upstream's published baseline before any GEOS run here):
#  [1] completeness: run.sh exit 0 (timeout -> FAIL), displacement_history.hdf5 written with
#      the 10 output times, all values finite, launcher audit "N verified, 0 mismatch".
#  [2] analytic reference (beamBending.ats CurveCheckParameters, script beamBending_curve.py =
#      Euler-Bernoulli with upstream's 1.043 discretisation factor): geos-ats metric
#      ||u - u_script||_2 / N <= HPCPERF_GEOS_ANALYTIC_TOL (default 0.0002, the .ats tolerance;
#      upstream's own 80x8x4 baseline scores 1.376e-4 -- the mesh's discretisation error, which is
#      why a per-time relative L-inf reading of the same number would fail even upstream's baseline;
#      that reading is printed as information only). Every run, every N.
#  [3] rank-count independence (AMG deck): N-GPU history vs the 1-GPU history of the same binary/deck,
#      relative L-inf <= 1e-4 (gate; 100x the GMRES krylovTol 1e-6 -- the two runs converge to
#      different iterates of the same tolerance through different AMG hierarchies -- and below the
#      official discretisation-level tolerance) and geos-ats baseline metric ||u - u_1GPU||_2 / N
#      <= 2e-4; whether the tighter 1e-5 is also met is printed (upstream's own direct-solver
#      baselines for 1 and 8 ranks agree to 1.5e-12).
#  [4] official baselines (1 rank only; the public integrated-test baseline beamBending_smoke_01
#      is for 1 rank with the shipped smoke deck): (a) every dataset/attribute of the GEOS restart
#      file at cycle 10 vs the baseline with atol 1e-3 / rtol 1e-7 (RestartcheckParameters) and
#      the geos-ats default exclusions; (b) the history vs the baseline's history, geos-ats baseline
#      curve metric <= 0.0002 and relative L-inf <= 1e-4. Requires the baseline tarball under
#      .deps/level3/geos/downloads (recorded with its sha256).
# The GPU is exercised through RAJA/CHAI kernels (assembly) and hypre's device AMG/GMRES for
# the amg solver runs; the launcher audits one GPU per rank.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
N="${HPCPERF_GPUS:-1}"; TIMEOUT="${HPCPERF_VALIDATE_TIMEOUT:-1800}"
ATOL=1e-3; RTOL=1e-7                  # restart baseline (beamBending.ats)
CURVE_TOL="${HPCPERF_GEOS_ANALYTIC_TOL:-2e-4}"  # geos-ats ||.||_2/N metric (script and baseline curve checks)
XRANK_TOL=1e-4; XRANK_INFO=1e-5                 # relative L-inf between histories
GCC_MM="$(l3_version_mm "$(/usr/bin/gcc -dumpfullversion)")"; OMPI_V="$(mpirun --version | head -1 | /usr/bin/grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
PROFILE="${HPCPERF_GEOS_PROFILE:-cuda$(l3_version_mm "$(l3_cuda_version)")-gcc${GCC_MM}-ompi$(echo "$OMPI_V" | tr -d .)}"
l3_require_materialized "$HERE" || exit 3
RUNS="$R/build/level3/geos/$PROFILE/$L3_RUN_SUBDIR"; SRC="$HERE/src"   # analytic reference script comes from the frozen bundle
PY="$R/.deps/level3/geos/$PROFILE/install/venv/bin/python"; [ -x "$PY" ] || { echo "validate.sh: FAIL -- venv python with h5py missing ($PY)"; exit 1; }
CURVE="$SRC/inputFiles/solidMechanics/beamBending_curve.py"
CHECK="$HERE/geos_beam_check.py"
export HPCPERF_GPUS="$N" HPCPERF_SCALE_MODE=smoke
mkdir -p "$RUNS"; ok=1
fail() { echo "validate.sh: FAIL -- $*"; ok=0; }
manifest_val() { /usr/bin/grep -m1 "^$2=" "$1/run_manifest.txt" 2>/dev/null | cut -d= -f2- || true; }
run_beam() { # run_beam <solver> <n> <out>
    local rc=0; HPCPERF_GEOS_SOLVER="$1" HPCPERF_GPUS="$2" timeout "$TIMEOUT" "$HERE/run.sh" "$BACKEND" > "$3" 2>&1 || rc=$?; return $rc
}

echo "validate.sh: === GEOS $BACKEND beam bending (80x8x4), hypre GMRES+AMG on device, $N GPU(s) [profile $PROFILE] ==="
D="$RUNS/beam80.amg.smoke.np$N"
rc=0; run_beam amg "$N" "$RUNS/validate.beam80.amg.np$N.stdout" || rc=$?
/usr/bin/grep -aE '^# GEOS|hpcperf-launch: audit summary|Error|error' "$RUNS/validate.beam80.amg.np$N.stdout" | head -6 || true
if [ "$rc" -eq 124 ]; then fail "beam80.amg np$N timed out"; elif [ "$rc" -ne 0 ]; then fail "beam80.amg np$N run.sh exited $rc"; else
    [ -f "$D/displacement_history.hdf5" ] || fail "beam80.amg np$N: displacement_history.hdf5 missing"
    if [ -f "$D/displacement_history.hdf5" ]; then
        "$PY" "$CHECK" analytic "$D" --tol "$CURVE_TOL" --curve "$CURVE" | tee "$D/analytic_check.txt" | tail -3 || fail "beam80.amg np$N: analytic beam check failed"
        if [ "$N" -gt 1 ]; then
            REF="$RUNS/beam80.amg.smoke.np1"
            if [ ! -f "$REF/displacement_history.hdf5" ] || [ "$(manifest_val "$REF" binary_sha256)" != "$(manifest_val "$D" binary_sha256)" ]; then
                echo "    (1-GPU reference missing/stale -- running it now)"; rc=0; run_beam amg 1 "$RUNS/validate.beam80.amg.np1.stdout" || fail "1-GPU reference run exited $?"
            fi
            "$PY" "$CHECK" compare "$REF" "$D" --tol "$XRANK_TOL" --info-tol "$XRANK_INFO" --l2n-tol "$CURVE_TOL" | tee "$D/cross_rank_check.txt" | tail -3 || fail "beam80.amg np$N: history differs from the 1-GPU run (rel L-inf > $XRANK_TOL or ||.||_2/N > $CURVE_TOL)"
        fi
    fi
fi

if [ "$N" -eq 1 ]; then
    echo "validate.sh: === GEOS $BACKEND shipped smoke deck (serial direct solver) on 1 rank: analytic check + official restart baseline ==="
    DD="$RUNS/beam80.direct.smoke.np1"
    rc=0; run_beam direct 1 "$RUNS/validate.beam80.direct.np1.stdout" || rc=$?
    if [ "$rc" -ne 0 ]; then fail "beam80.direct np1 run.sh exited $rc"; else
        "$PY" "$CHECK" analytic "$DD" --tol "$CURVE_TOL" --curve "$CURVE" | tee "$DD/analytic_check.txt" | tail -1 || fail "beam80.direct np1: analytic beam check failed"
        BASE_TGZ="$(ls "$R"/.deps/level3/geos/downloads/baseline_integratedTests-*.tar.gz 2>/dev/null | head -1)"
        if [ -n "$BASE_TGZ" ]; then
            BD="$R/.deps/level3/geos/downloads/baseline_extracted/solidMechanics/beamBending_smoke_01"
            [ -d "$BD" ] || { mkdir -p "$R/.deps/level3/geos/downloads/baseline_extracted"; tar -xzf "$BASE_TGZ" -C "$R/.deps/level3/geos/downloads/baseline_extracted" ./solidMechanics/beamBending_smoke_01 2>/dev/null || tar -xzf "$BASE_TGZ" -C "$R/.deps/level3/geos/downloads/baseline_extracted" solidMechanics/beamBending_smoke_01; }
            [ -d "$BD" ] || fail "baseline beamBending_smoke_01 not found in $BASE_TGZ"
            [ -d "$BD" ] && { "$PY" "$CHECK" restart "$DD" "$BD" --rtol "$RTOL" --atol "$ATOL" | tee "$DD/restart_baseline_check.txt" | tail -2 || fail "beam80.direct np1: restart file disagrees with the upstream baseline (atol $ATOL rtol $RTOL)"; }
            [ -d "$BD" ] && { "$PY" "$CHECK" compare "$BD" "$DD" --tol "$XRANK_TOL" --info-tol "$XRANK_INFO" --l2n-tol "$CURVE_TOL" | tee "$DD/history_baseline_check.txt" | tail -3 || fail "beam80.direct np1: history disagrees with the upstream baseline history"; }
            echo "    baseline: $(basename "$BASE_TGZ") sha256=$(l3_sha_file "$BASE_TGZ")"
        else
            fail "no integrated-test baseline tarball under .deps/level3/geos/downloads (official restart comparison NOT_RUN)"
        fi
    fi
fi

if [ "$ok" -eq 1 ]; then echo "GEOS $BACKEND validation ($N GPU; geos-ats analytic curve metric <= $CURVE_TOL$( [ "$N" -gt 1 ] && echo ", history within rel $XRANK_TOL / metric $CURVE_TOL of 1-GPU" || echo ", shipped smoke deck vs upstream restart baseline atol $ATOL rtol $RTOL + baseline history"), finite, complete): PASS"; exit 0; fi
echo "GEOS $BACKEND validation ($N GPU): FAIL"; exit 1
