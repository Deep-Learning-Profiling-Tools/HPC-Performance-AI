#!/usr/bin/env bash
# Correctness check for SPECFEM3D Cartesian on N GPUs, using upstream's own
# reference seismograms and comparison tool.
#
#   ./validate.sh [CUDA|HIP]        HPCPERF_GPUS=N (default 1)
#
# Case: EXAMPLES/applications/homogeneous_halfspace as shipped (36x36x16 = 20,736
# HEX8 elements, CMT source at 30 km depth, 4 stations, NSTEP 5000, DT 0.05 s),
# run in GPU mode with NPROC = N. Its README (step 7) says to "check with 6
# reference seismograms in REF_SEIS/"; upstream's BuildBot uses
# utils/scripts/compare_seismogram_correlations.py, which reports per trace the
# correlation coefficient, the L2 misfit normalised by the reference energy,
# and the cross-correlation time shift, with upstream's thresholds
# TOL_CORR = 0.8, TOL_ERR = 0.01 (1 %), TOL_SHIFT = 0.01 s. The reference
# traces were produced on CPUs (double precision, 4 ranks); the GPU solver is
# single precision, so bitwise equality is not expected -- upstream's
# tolerance-based comparison is the appropriate criterion and is used
# unchanged. PASS = every trace within all three thresholds.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
N="${HPCPERF_GPUS:-1}"
UP="$R/_upstream/level3/specfem3d"
REF="$UP/EXAMPLES/applications/homogeneous_halfspace/REF_SEIS"
CMP="$UP/utils/scripts/compare_seismogram_correlations.py"
RUN_DIR="$R/build/level3/specfem3d/$MODEL/run/smoke.np$N"
[ -d "$REF" ] && [ -f "$CMP" ] || { echo "validate.sh: $REF or $CMP missing (run fetch.sh)" >&2; exit 1; }

export HPCPERF_GPUS="$N"
echo "validate.sh: SPECFEM3D $BACKEND homogeneous_halfspace (20,736 elements, 5000 steps) on $N GPU(s)"
HPCPERF_SCALE_MODE=smoke "$HERE/run.sh" "$BACKEND" 2>&1 | grep -E '^#|hpcperf-launch: audit summary|Time loop|Elapsed time|End of the simulation|Error|ERROR' || true
OUT="$RUN_DIR/OUTPUT_FILES"
ls "$OUT"/*.semd >/dev/null 2>&1 || { echo "validate.sh: FAIL -- no seismograms under $OUT"; exit 1; }

echo "validate.sh: comparing with upstream REF_SEIS (utils/scripts/compare_seismogram_correlations.py)"
CMP_OUT="$RUN_DIR/compare_ref_seis.log"
python3 "$CMP" "$OUT/" "$REF/" > "$CMP_OUT" 2>&1 || true
grep -E '^\|' "$CMP_OUT" | sed 's/^/  /'
grep -E 'seismograms compared|poor correlation|poor match|significant time shift|no poor|no significant' "$CMP_OUT" | sed 's/^/  /'
ok=1
grep -q 'no poor correlations found' "$CMP_OUT" || ok=0
grep -q 'no poor matches found' "$CMP_OUT" || ok=0
grep -q 'no significant time shifts found' "$CMP_OUT" || ok=0
NCMP="$(grep -oE '^[0-9]+ seismograms compared' "$CMP_OUT" | awk '{print $1}')"
[ "${NCMP:-0}" -gt 0 ] || ok=0
if [ "$ok" -eq 1 ]; then
    echo "SPECFEM3D $BACKEND validation ($N GPU, homogeneous_halfspace vs REF_SEIS, corr>=0.8 err<=1% shift<=0.01s): PASS"; exit 0
fi
echo "SPECFEM3D $BACKEND validation ($N GPU, homogeneous_halfspace vs REF_SEIS): FAIL (see $CMP_OUT)"; exit 1
