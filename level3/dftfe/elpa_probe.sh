#!/usr/bin/env bash
# Independent check of the ELPA NVIDIA-GPU kernels built for DFT-FE (before any DFT-FE run):
# ELPA's own test programs (test/Fortran/test.F90 compiled as validate_*_gpu_* programs, which call
# e%set("nvidia-gpu", 1)) diagonalise an analytic test matrix and verify
#   residual      max || A z_i - lambda_i z_i ||  <= 9e-10   (ELPA's tol_res_real_double)
#   orthogonality max | Z^T Z - I |               <= 9e-10   (ELPA's tol_orth_real_double)
# themselves (nonzero exit on violation); this script re-parses the printed values and applies
# the same limits, runs the 1-stage and 2-stage GPU solvers on 1, 2 and 4 GPUs (one MPI rank per
# GPU through the common launcher, ELPA's process grid np_rows x np_cols) and the CPU versions
# of the same programs on 1 rank as a cross-check. A GPU run whose residual is not smaller than
# 9e-10, that does not exit 0, or whose launcher audit shows a GPU mismatch -> FAIL.
#
#   ./elpa_probe.sh            HPCPERF_ELPA_NA=2000 HPCPERF_ELPA_NEV=1000 HPCPERF_ELPA_NBLK=32
# Record: <install>/elpa/ELPA_GPU_PROBE.txt (PASS/FAIL + values); validate.sh requires PASS.
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
TOL=9e-10
export LD_LIBRARY_PATH="$INST/elpa/lib:$INST/scalapack/lib:$INST/openblas/lib:${LD_LIBRARY_PATH:-}"
OUT="$L3_BUILD/elpa_probe"; mkdir -p "$OUT"
REC="$INST/elpa/ELPA_GPU_PROBE.txt"; : > "$REC.tmp"
ok=1
note() { echo "$*" | tee -a "$REC.tmp"; }
run_case() { # run_case <program> <ranks> <gpus:yes|no>
    local prog=$1 n=$2 gpu=$3 exe="$EB/$prog" log="$OUT/$prog.np$n.log" rc=0 res orth audit
    [ -x "$exe" ] || { note "MISSING $prog"; ok=0; return; }
    if [ "$gpu" = yes ]; then
        "$L3_LAUNCHER" --gpus "$n" --cpus-per-rank 4 --bind wrapper -- "$exe" "$NA" "$NEV" "$NBLK" > "$log" 2>&1 || rc=$?
        audit="$(/usr/bin/grep -a 'audit summary' "$log" | tail -1 | sed 's/.*audit summary: //')"
    else
        # CPU cross-check: one rank through the same launcher (no GPU binding; the CPU program ignores the device)
        "$L3_LAUNCHER" --gpus 1 --bind none -- "$exe" "$NA" "$NEV" "$NBLK" > "$log" 2>&1 || rc=$?
        audit="cpu"
    fi
    res="$(/usr/bin/grep -a '%Error Residual' "$log" | tail -1 | awk -F: '{print $2}' | tr -d ' ')"
    orth="$(/usr/bin/grep -a '%Error Orthogonality' "$log" | tail -1 | awk -F: '{print $2}' | tr -d ' ')"
    local verdict
    verdict="$(python3 - "$rc" "$res" "$orth" "$TOL" "$gpu" "$audit" <<'PY'
import sys, math
rc, res, orth, tol, gpu, audit = sys.argv[1:7]
try:
    r, o = float(res), float(orth)
    fin = math.isfinite(r) and math.isfinite(o)
except ValueError:
    fin = False; r = o = float("nan")
good = rc == "0" and fin and 0 < r <= float(tol) and o <= float(tol) and (gpu == "no" or ("mismatch" in audit and audit.split(",")[1].strip().startswith("0")))
print("PASS" if good else "FAIL")
PY
)"
    [ "$verdict" = PASS ] || ok=0
    note "$verdict $prog ranks=$n gpu=$gpu na=$NA nev=$NEV nblk=$NBLK exit=$rc residual=${res:-NA} orthogonality=${orth:-NA} tol=$TOL audit=[${audit:-none}] log=$log"
}
note "# ELPA 2026.02.001 GPU kernel probe, profile $PROFILE, $(date -u +%FT%TZ); limits: residual<=$TOL orthogonality<=$TOL (ELPA's own real-double tolerances)"
for prog in validate_real_double_eigenvectors_1stage_gpu_analytic validate_real_double_eigenvectors_2stage_default_kernel_gpu_analytic; do   # (the *_default names are ELPA's .sh wrappers; these are the programs)
    for n in 1 2 4; do run_case "$prog" "$n" yes; done
done
for prog in validate_real_double_eigenvectors_1stage_analytic validate_real_double_eigenvectors_2stage_default_kernel_analytic; do
    run_case "$prog" 1 no
done
if [ "$ok" -eq 1 ]; then note "RESULT PASS"; else note "RESULT FAIL"; fi
mv -f "$REC.tmp" "$REC"; echo "# record: $REC"; [ "$ok" -eq 1 ]
