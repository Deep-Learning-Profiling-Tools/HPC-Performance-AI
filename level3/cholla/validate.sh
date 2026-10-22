#!/usr/bin/env bash
# Correctness check of the Cholla build: upstream's own analytic system test tHYDROtMHDSYSTEMSoundWave3D
# (src/system_tests/hydro_system_tests.cpp:99-140) reproduced without the gtest harness: its input file
# (src/system_tests/input_files/tHYDROtMHDSYSTEMSoundWave3D_CorrectInputExpectCorrectOutput.txt: a 64^3 linear sound
# wave, A = 1e-5, run to t = 0.05) is run on the GPU and the final output is compared cell by cell with the analytic
# sine (density = 1 + A sin(2 pi (x - c_s t)), momentum_x = A sin(2 pi (x - c_s t))) at upstream's tolerance 1e-7
# = 1 % of the amplitude (check_cholla.py soundwave --sine-tol 0.01: L1 and maximum error, which upstream's cell-by-cell
# assertion implies). The harness adds exit code, the final-time assertion and the launcher audit.
#
#   ./validate.sh [CUDA]        HPCPERF_GPUS=N (default 1): N must be 1
# Prints the criterion and the verdict; exit 0 PASS, 1 FAIL.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
N="${HPCPERF_GPUS:-1}"
[ "$N" -eq 1 ] || { echo "validate.sh: the validation run is a one-rank run; HPCPERF_GPUS=$N refused" >&2; exit 2; }
PROFILE="$(l3_backend_profile CHOLLA "$MODEL")"
l3_paths_profile cholla "$PROFILE" "$MODEL" || exit 2
l3_fingerprint_expect_backend "$L3_INSTALL" "$MODEL" || exit 1
l3_require_materialized "$HERE" || exit 3
EXE="$L3_INSTALL/bin/cholla"
[ -x "$EXE" ] || { echo "validate.sh: $EXE missing -- run ./build.sh $BACKEND first" >&2; exit 3; }
IN="$HERE/src/src/system_tests/input_files/tHYDROtMHDSYSTEMSoundWave3D_CorrectInputExpectCorrectOutput.txt"
[ -f "$IN" ] || { echo "validate.sh: upstream test input $IN missing" >&2; exit 3; }
RUN_DIR="$(l3_rundir "$L3_BUILD/validate/soundwave3d.np1")" || exit 2
rm -rf "$RUN_DIR"; mkdir -p "$RUN_DIR/out"
# the test input sets outdir=./ ; upstream's command-line override (the only one) points it at the run directory
cp -f "$IN" "$RUN_DIR/soundwave3d.txt"
echo "validate.sh: Cholla $BACKEND [profile $PROFILE] -- upstream system test tHYDROtMHDSYSTEMSoundWave3D on 1 GPU (64^3 linear sound wave, A = 1e-5, t = 0.05; analytic sine within 1e-7 = 1 % of A, upstream's tolerance)"
cd "$RUN_DIR"
rc=0
"$L3_LAUNCHER" --gpus 1 --bind wrapper -- "$EXE" "$RUN_DIR/soundwave3d.txt" "outdir=$RUN_DIR/out/" > run.log 2>&1 || rc=$?
grep -E 'hpcperf-launch: audit summary' run.log | tail -1 | sed 's/^/  /'
[ "$rc" -eq 0 ] || { tail -5 run.log; echo "FAIL: Cholla validate (the run exited $rc)"; exit 1; }
python3 "$HERE/check_cholla.py" soundwave --run-dir "$RUN_DIR" --params "$RUN_DIR/soundwave3d.txt" --sine-tol 0.01 --label validate-soundwave3d | sed 's/^/  /' || { echo "FAIL: Cholla validate (upstream SoundWave3D criterion)"; exit 1; }
echo "PASS: Cholla validate (upstream tHYDROtMHDSYSTEMSoundWave3D analytic test on the $BACKEND build)"
