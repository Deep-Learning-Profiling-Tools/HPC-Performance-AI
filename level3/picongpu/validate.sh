#!/usr/bin/env bash
# Validate the PIConGPU CUDA build with upstream's own KHI growth-rate test (share/picongpu/tests/KHI_growthRate).
#
#   ./validate.sh [CUDA]
#
# Upstream's bin/ci.sh builds the test's input set, runs `picongpu -d 1 1 1 -g 192 512 12 --periodic 1 1 1 -s 3000
# --fields_energy.period 10` on one device and judges fields_energy.dat with lib/python/test/setups/ESKHI (the
# growth rate of the Bx field energy within 20 % of 1/(sqrt(8) gamma); it needs numpy/matplotlib, absent here).
# validate.sh runs the same mechanism with the repository's tools: the registered `khi-192x512` input (= that
# command line on that input set) through run.sh, then check_khi_growth.py (the ESKHI criterion re-implemented with
# the Python standard library, parameters from the input set's param files). Also required: exit 0, the
# 'calculation  simulation time' line, 3000 steps done and a clean launcher audit.
# Exit 0 PASS, 1 FAIL, 3 when the source bundle or the build is missing.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
PROFILE="$(l3_backend_profile PICONGPU "$MODEL")"
l3_paths_profile picongpu "$PROFILE" "$MODEL" || exit 3
l3_require_materialized "$HERE" || exit 3
[ -x "$L3_INSTALL/khi/bin/picongpu" ] || { echo "validate.sh: $L3_INSTALL/khi/bin/picongpu missing -- run ./build.sh $BACKEND first" >&2; exit 3; }
OUT="$L3_BUILD/validate"; mkdir -p "$OUT"
echo "== validate: running the registered khi-192x512 input (upstream tests/KHI_growthRate, bin/ci.sh command line) on 1 GPU"
HPCPERF_PICONGPU_INPUT=khi-192x512 HPCPERF_GPUS=1 HPCPERF_L3_RUN_SUBDIR=run.validate bash "$HERE/run.sh" "$BACKEND" > "$OUT/khi.log" 2>&1
rc=$?
fail=0
[ $rc -eq 0 ] || { echo "FAIL: run.sh exited $rc ($OUT/khi.log)"; fail=1; }
grep -q '^calculation  simulation time:' "$OUT/khi.log" || { echo "FAIL: no 'calculation  simulation time' line"; fail=1; }
grep -qE '^\s*100 % =\s+3000 \|' "$OUT/khi.log" || { echo "FAIL: the run did not report 3000 steps"; fail=1; }
grep -q 'audit summary:.*0 mismatch' "$OUT/khi.log" || { echo "FAIL: launcher audit not clean"; fail=1; }
RUN_DIR="$L3_BUILD/run.validate/khi-192x512.np1"
python3 "$HERE/check_khi_growth.py" --energy "$RUN_DIR/fields_energy.dat" --gamma 1.021 --density 1.0e25 --dt 1.5394e-16 --acceptance 0.2 --label validate-khi | tee "$OUT/check.log"
grep -q '^PASS: PIConGPU KHI growth-rate check' "$OUT/check.log" || fail=1
if [ $fail -eq 0 ]; then echo "PASS: PIConGPU validate (upstream KHI_growthRate test on the CUDA build)"; else echo "FAIL: PIConGPU validate"; exit 1; fi
