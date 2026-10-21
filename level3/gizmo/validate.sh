#!/usr/bin/env bash
# Validate the GIZMO CUDA build with upstream's own test acceptance on upstream's Evrard collapse problem.
#
#   ./validate.sh [CUDA]
#
# Upstream's test harness (src/test/harness, pytest) builds every test problem, runs it and judges the final
# snapshot (src/test/<name>/test_<name>.py); it needs h5py, scipy, matplotlib and meshoid, which this
# environment does not provide. validate.sh runs the same mechanism with the repository's tools: the registered
# `evrard` input (upstream's test/evrard Config.sh, evrard.params and evrard_ics.hdf5, the problem the GPU path
# was brought up with) through run.sh on one GPU, then check_profile.py = test_evrard.py's criterion re-implemented
# on h5dump output (final snapshot time = TimeMax; mean |log10 rho_sim - log10 rho_exact| over 30 log-spaced radial
# bins against src/test/evrard/evrard_exact.txt < 0.3). Also required: exit 0, the 'Final time=... reached' line,
# the Kokkos CUDA device line (the GPU code path) and a clean launcher audit.
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
PROFILE="$(l3_backend_profile GIZMO "$MODEL")"
l3_paths_profile gizmo "$PROFILE" "$MODEL" || exit 3
l3_require_materialized "$HERE" || exit 3
[ -x "$L3_INSTALL/bin/GIZMO-evrard" ] || { echo "validate.sh: $L3_INSTALL/bin/GIZMO-evrard missing -- run ./build.sh $BACKEND first" >&2; exit 3; }
OUT="$L3_BUILD/validate"; mkdir -p "$OUT"
echo "== validate: running the registered evrard input (upstream test/evrard) on 1 GPU"
HPCPERF_GIZMO_INPUT=evrard HPCPERF_GPUS=1 HPCPERF_L3_RUN_SUBDIR=run.validate bash "$HERE/run.sh" "$BACKEND" > "$OUT/evrard.log" 2>&1
rc=$?
fail=0
[ $rc -eq 0 ] || { echo "FAIL: run.sh exited $rc ($OUT/evrard.log)"; fail=1; }
grep -q '^Final time=0.8 reached\. Simulation ends\.' "$OUT/evrard.log" || { echo "FAIL: the run did not reach TimeMax 0.8"; fail=1; }
grep -q -i 'Kokkos::Cuda\|KOKKOS_ENABLE_CUDA\|Cuda' "$OUT/evrard.log" || { echo "FAIL: no Kokkos CUDA device line in the output (not the GPU code path?)"; fail=1; }
grep -q 'audit summary:.*0 mismatch' "$OUT/evrard.log" || { echo "FAIL: launcher audit not clean"; fail=1; }
RUN_DIR="$L3_BUILD/run.validate/evrard.np1"
python3 "$HERE/check_profile.py" evrard --run-dir "$RUN_DIR" --exact "$HERE/src/test/evrard/evrard_exact.txt" --label validate-evrard | tee "$OUT/check.log"
grep -q '^PASS: GIZMO profile check' "$OUT/check.log" || fail=1
if [ $fail -eq 0 ]; then echo "PASS: GIZMO validate (upstream test/evrard acceptance on the CUDA build)"; else echo "FAIL: GIZMO validate"; exit 1; fi
