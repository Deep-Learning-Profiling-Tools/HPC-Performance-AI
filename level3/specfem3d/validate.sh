#!/usr/bin/env bash
# Correctness check for SPECFEM3D Cartesian on N GPUs, using upstream's own
# reference seismograms and comparison tool.
#
#   ./validate.sh [CUDA|HIP]        HPCPERF_GPUS=N (default 1)
#
# Case: EXAMPLES/applications/homogeneous_halfspace as shipped (20,736 HEX8
# elements, NSTEP 5000, DT 0.05 s), GPU_MODE, NPROC = N. Upstream's
# utils/scripts/compare_seismogram_correlations.py reports per trace the
# correlation, the L2 misfit (normalised by the reference energy) and the
# cross-correlation time shift; thresholds corr>=0.8, err<=1%, shift<=0.01 s.
# The GPU solver is single precision, so tolerance comparison (not bitwise) is
# the right criterion and is used unchanged. PASS requires: run.sh exited 0,
# EVERY reference trace was compared (count == number of REF_SEIS traces), all
# produced seismograms are finite, and the comparison reports no poor
# correlation / no poor match / no significant time shift.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
N="${HPCPERF_GPUS:-1}"
UP="$R/_upstream/level3/specfem3d"
REF="$UP/EXAMPLES/applications/homogeneous_halfspace/REF_SEIS"
CMP="$UP/utils/scripts/compare_seismogram_correlations.py"
RUN_DIR="$R/build/level3/specfem3d/$MODEL/$L3_RUN_SUBDIR/smoke.np$N"
TIMEOUT="${HPCPERF_VALIDATE_TIMEOUT:-1800}"
[ -d "$REF" ] && [ -f "$CMP" ] || { echo "validate.sh: $REF or $CMP missing (run fetch.sh)" >&2; exit 1; }
NREF="$(ls "$REF"/*.semd 2>/dev/null | wc -l)"
[ "$NREF" -ge 1 ] || { echo "validate.sh: no reference traces in $REF" >&2; exit 1; }

export HPCPERF_GPUS="$N"
echo "validate.sh: SPECFEM3D $BACKEND homogeneous_halfspace (20,736 elements, 5000 steps) on $N GPU(s); $NREF reference traces"
VOUT="$R/build/level3/specfem3d/$MODEL/$L3_RUN_SUBDIR/validate.smoke.np$N.stdout"; mkdir -p "$(dirname "$VOUT")"
rc=0; HPCPERF_SCALE_MODE=smoke timeout "$TIMEOUT" "$HERE/run.sh" "$BACKEND" > "$VOUT" 2>&1 || rc=$?
grep -aE '^#|hpcperf-launch: audit summary|Time loop|Elapsed time|End of the simulation|Error|ERROR' "$VOUT" || true
if [ "$rc" -eq 124 ]; then echo "validate.sh: FAIL -- run timed out after ${TIMEOUT}s"; exit 1; fi
[ "$rc" -eq 0 ] || { echo "validate.sh: FAIL -- run.sh exited $rc (see $VOUT)"; exit 1; }
OUT="$RUN_DIR/OUTPUT_FILES"
NGOT="$(ls "$OUT"/*.semd 2>/dev/null | wc -l)"
[ "$NGOT" -ge "$NREF" ] || { echo "validate.sh: FAIL -- produced $NGOT seismograms, need >= $NREF"; exit 1; }

# reject non-finite samples in the produced traces before trusting the correlation
python3 - "$OUT" <<'PY' || { echo "validate.sh: FAIL -- non-finite sample in a produced seismogram"; exit 1; }
import sys, os, glob, math
sys.path.insert(0, os.environ["L3_TOOLS"])
from l3_check import ValidationError
try:
    for f in sorted(glob.glob(sys.argv[1] + "/*.semd")):
        nrow = 0
        for ln in open(f):
            parts = ln.split()
            if len(parts) < 2: continue
            for v in parts[:2]:
                if not math.isfinite(float(v)): raise ValidationError(f"{os.path.basename(f)}: non-finite sample {v}")
            nrow += 1
        if nrow == 0: raise ValidationError(f"{os.path.basename(f)}: no samples")
except ValidationError as ex:
    print(f"  VALIDATION ERROR: {ex}"); sys.exit(1)
PY

echo "validate.sh: comparing with upstream REF_SEIS (utils/scripts/compare_seismogram_correlations.py)"
CMP_OUT="$RUN_DIR/compare_ref_seis.log"
python3 "$CMP" "$OUT/" "$REF/" > "$CMP_OUT" 2>&1 || true
grep -E '^\|' "$CMP_OUT" | sed 's/^/  /'
grep -E 'seismograms compared|poor correlation|poor match|significant time shift|no poor|no significant' "$CMP_OUT" | sed 's/^/  /'
NCMP="$(grep -oE '^[0-9]+ seismograms compared' "$CMP_OUT" | awk '{print $1}')"
ok=1
[ "${NCMP:-0}" -eq "$NREF" ] || { echo "  only ${NCMP:-0} of $NREF reference traces were compared"; ok=0; }
grep -q 'no poor correlations found' "$CMP_OUT" || ok=0
grep -q 'no poor matches found' "$CMP_OUT" || ok=0
grep -q 'no significant time shifts found' "$CMP_OUT" || ok=0
if [ "$ok" -eq 1 ]; then
    echo "SPECFEM3D $BACKEND validation ($N GPU, homogeneous_halfspace vs REF_SEIS, $NREF/$NREF traces, corr>=0.8 err<=1% shift<=0.01s): PASS"; exit 0
fi
echo "SPECFEM3D $BACKEND validation ($N GPU, homogeneous_halfspace vs REF_SEIS): FAIL (see $CMP_OUT)"; exit 1
