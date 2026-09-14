#!/usr/bin/env bash
# Independent check of the ELPA NVIDIA-GPU kernels built for DFT-FE (before any DFT-FE run):
# ELPA's own test programs (test/Fortran/test.F90 compiled as validate_*_gpu_analytic programs,
# which call e%set("nvidia-gpu", 1)) diagonalise the analytic test matrix of
# test/shared/test_analytic_template.F90 (known eigenpairs) and verify themselves
#   max |lambda_i - lambda_i^exact|   <= 5e-14   (ELPA's tol_eigenvalues, real double)
#   max |z_i - z_i^exact|             <= 6e-10   (ELPA's tol_eigenvectors, real double)
# (nonzero exit -- `stop 1` -- on violation); this script re-parses the printed
# "Maximum error in eigenvalues/eigenvectors" values and applies the same limits, runs the
# 1-stage and 2-stage GPU solvers on 1, 2 and 4 GPUs (one MPI rank per GPU through the common
# launcher, ELPA's process grid np_rows x np_cols) and the CPU versions of the same programs on
# 1 rank as a cross-check when they exist. A GPU run whose errors are not within the limits,
# that does not exit 0, or whose launcher audit shows a GPU mismatch -> FAIL; a run whose output
# lacks the values -> FAIL (never an abort of this script).
#
#   ./elpa_probe.sh            HPCPERF_ELPA_NA=2000 HPCPERF_ELPA_NEV=1000 HPCPERF_ELPA_NBLK=32
# Record: <install>/elpa/ELPA_GPU_PROBE.txt (PASS/FAIL + values); validate.sh requires PASS.
# History: the first version looked for the "%Error Residual/Orthogonality" lines of ELPA's
# *random-matrix* validate programs, which the analytic programs do not print (they print the
# eigenvalue/eigenvector errors above) -- under `set -e -o pipefail` the empty grep aborted the
# script after the first (successful) GPU run; fixed 2026-09-06, no run was ever mis-judged.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
GCC_MM="$(l3_version_mm "$(/usr/bin/gcc -dumpfullversion)")"; OMPI_V="$(mpirun --version | head -1 | /usr/bin/grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
PROFILE="${HPCPERF_DFTFE_PROFILE:-cuda$(l3_version_mm "$(l3_cuda_version)")-gcc${GCC_MM}-ompi$(echo "$OMPI_V" | tr -d .)}"
l3_paths_profile dftfe "$PROFILE"
EB="$L3_BUILD_DEPS/elpa"; INST="$L3_INSTALL"
[ -f "$INST/elpa/.hpcperf-stage-done" ] || { echo "elpa_probe.sh: ELPA stage not built for profile $PROFILE" >&2; exit 1; }
NA="${HPCPERF_ELPA_NA:-2000}"; NEV="${HPCPERF_ELPA_NEV:-1000}"; NBLK="${HPCPERF_ELPA_NBLK:-32}"
TOL_EV=5e-14; TOL_Z=6e-10     # test_analytic_template.F90: tol_eigenvalues / tol_eigenvectors (real double)
export LD_LIBRARY_PATH="$EB/.libs:$INST/elpa/lib:$INST/scalapack/lib:$INST/openblas/lib:${LD_LIBRARY_PATH:-}"
OUT="$L3_BUILD/elpa_probe"; mkdir -p "$OUT"
REC="$INST/elpa/ELPA_GPU_PROBE.txt"; : > "$REC.tmp"
ok=1
note() { echo "$*" | tee -a "$REC.tmp"; }
run_case() { # run_case <program> <ranks> <gpus:yes|no>
    # the top-level names in the build tree are libtool wrapper scripts (they would try to relink through mpicc);
    # the real, already linked programs live in .libs/
    local prog=$1 n=$2 gpu=$3 exe="$EB/.libs/$prog" log="$OUT/$prog.np$n.log" rc=0 res orth audit
    if [ ! -x "$exe" ]; then
        if [ "$gpu" = yes ]; then note "MISSING $prog (GPU test program not built)"; ok=0; else note "SKIPPED $prog (CPU variant not built with the GPU-enabled configuration; cross-check unavailable)"; fi
        return
    fi
    if [ "$gpu" = yes ]; then
        "$L3_LAUNCHER" --gpus "$n" --cpus-per-rank 4 --bind wrapper -- "$exe" "$NA" "$NEV" "$NBLK" > "$log" 2>&1 || rc=$?
        audit="$( { /usr/bin/grep -a 'audit summary' "$log" || true; } | tail -1 | sed 's/.*audit summary: //')"
    else
        # CPU cross-check: one rank through the same launcher (no GPU binding; the CPU program ignores the device)
        "$L3_LAUNCHER" --gpus 1 --bind none -- "$exe" "$NA" "$NEV" "$NBLK" > "$log" 2>&1 || rc=$?
        audit="cpu"
    fi
    # the analytic test programs print these two lines (test_analytic_template.F90); missing -> FAIL below
    ev="$( { /usr/bin/grep -a 'Maximum error in eigenvalues' "$log" || true; } | tail -1 | awk -F: '{print $2}' | tr -d ' ')"
    zv="$( { /usr/bin/grep -a 'Maximum error in eigenvectors' "$log" || true; } | tail -1 | awk -F: '{print $2}' | tr -d ' ')"
    gpu_evidence="$( { /usr/bin/grep -ac '_gpu\|gpublas_\|gpu_copy' "$log" || true; } )"
    local verdict
    verdict="$(python3 - "$rc" "$ev" "$zv" "$TOL_EV" "$TOL_Z" "$gpu" "$audit" "$gpu_evidence" <<'PY'
import sys, math
rc, ev, zv, tol_ev, tol_z, gpu, audit, gpu_evidence = sys.argv[1:9]
try:
    e, z = float(ev), float(zv)
    fin = math.isfinite(e) and math.isfinite(z)
except ValueError:
    fin = False; e = z = float("nan")
good = rc == "0" and fin and 0 <= e <= float(tol_ev) and 0 <= z <= float(tol_z)
if gpu == "yes":
    # launcher audit "N verified, 0 mismatch, ..." and ELPA's own GPU timers (trans_ev_*_gpu, gpublas_*) in the output
    good = good and "mismatch" in audit and audit.split(",")[1].strip().startswith("0") and int(gpu_evidence or 0) > 0
print("PASS" if good else "FAIL")
PY
)"
    [ "$verdict" = PASS ] || ok=0
    note "$verdict $prog ranks=$n gpu=$gpu na=$NA nev=$NEV nblk=$NBLK exit=$rc max_err_eigenvalues=${ev:-NA} (tol $TOL_EV) max_err_eigenvectors=${zv:-NA} (tol $TOL_Z) gpu_timer_lines=$gpu_evidence audit=[${audit:-none}] log=$log"
}
note "# ELPA 2026.02.001 GPU kernel probe, profile $PROFILE, $(date -u +%FT%TZ); limits: max eigenvalue error <= $TOL_EV, max eigenvector error <= $TOL_Z (ELPA's own real-double analytic-test tolerances), exit 0, GPU timers present, launcher audit 0 mismatch"
for prog in validate_real_double_eigenvectors_1stage_gpu_analytic validate_real_double_eigenvectors_2stage_default_kernel_gpu_analytic; do   # (the *_default names are ELPA's .sh wrappers; these are the programs)
    for n in 1 2 4; do run_case "$prog" "$n" yes; done
done
for prog in validate_real_double_eigenvectors_1stage_analytic validate_real_double_eigenvectors_2stage_default_kernel_analytic; do
    run_case "$prog" 1 no
done
if [ "$ok" -eq 1 ]; then note "RESULT PASS"; else note "RESULT FAIL"; fi
mv -f "$REC.tmp" "$REC"; echo "# record: $REC"; [ "$ok" -eq 1 ]
