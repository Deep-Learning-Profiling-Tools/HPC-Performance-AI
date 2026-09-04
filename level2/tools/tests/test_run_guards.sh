#!/bin/bash
# Interface guards of the Level 2 run.sh scripts (no GPU run is performed):
#   * an app without HPCPERF_SCALE_MODE support must ERROR when it is set;
#   * HPCPERF_GPUS vs HPCPERF_NP disagreement must ERROR;
#   * extra arguments that would override validated parameters must ERROR.
# The scripts check for their built binary first, so this needs built apps
# (each case is skipped if the binary is missing).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../../.." && pwd)"
pass=0; failn=0; skip=0
ok()  { echo "ok   $*"; pass=$((pass+1)); }
bad() { echo "FAIL $*"; failn=$((failn+1)); }
noise() { grep -viE 'lua|posix|no file|stack traceback|\[C\]|addto|no field' | grep -vE '^\s*$'; }
unset HPCPERF_GPUS HPCPERF_NP HPCPERF_SCALE_MODE HPCPERF_DRY_RUN
export HPCPERF_DRY_RUN=1   # never launch anything from here

built() { [ -x "$1" ]; }

# legacy-interface apps: HPCPERF_SCALE_MODE must be rejected
for app in remhos exampm cloverleaf tealeaf kripke branson hipbone miniweather haccabanapm; do
    exe=""
    case $app in
        remhos) exe="$R/build/level2/remhos/cuda/remhos";;         exampm) exe="$R/build/level2/exampm/cuda/examples/DamBreak";;
        cloverleaf) exe="$R/build/level2/cloverleaf/cuda/cuda-cloverleaf";; tealeaf) exe="$R/build/level2/tealeaf/cuda/cuda-tealeaf";;
        kripke) exe="$R/build/level2/kripke/cuda/kripke.exe";;      branson) exe="$R/build/level2/branson/cuda/BRANSON";;
        hipbone) exe="$R/build/level2/hipbone/cuda/hipBone";;       miniweather) exe="$R/build/level2/miniweather/cuda/parallelfor";;
        haccabanapm) exe="$R/build/level2/haccabanapm/cuda/pm_ic";;
    esac
    if ! built "$exe"; then echo "skip $app (not built)"; skip=$((skip+1)); continue; fi
    out="$(HPCPERF_SCALE_MODE=weak "$R/level2/$app/run.sh" CUDA 2>&1)"; rc=$?; out="$(noise <<<"$out")"
    if [ "$rc" -eq 2 ] && grep -q 'HPCPERF_SCALE_MODE.*not implemented' <<<"$out"; then ok "$app: HPCPERF_SCALE_MODE rejected (rc=2)"; else bad "$app: SCALE_MODE not rejected (rc=$rc): $(head -2 <<<"$out")"; fi
    out="$(HPCPERF_GPUS=2 HPCPERF_NP=1 "$R/level2/$app/run.sh" CUDA 2>&1)"; rc=$?; out="$(noise <<<"$out")"
    if [ "$rc" -eq 2 ] && grep -q 'disagree' <<<"$out"; then ok "$app: HPCPERF_GPUS/HPCPERF_NP disagreement rejected"; else bad "$app: GPUS/NP conflict not rejected (rc=$rc): $(head -2 <<<"$out")"; fi
done

# refactored apps: forbidden extra args
if built "$R/build/level2/amg2023/cuda/amg"; then
    out="$("$R/level2/amg2023/run.sh" CUDA -P 1 1 1 2>&1)"; rc=$?; out="$(noise <<<"$out")"
    [ "$rc" -eq 2 ] && grep -q 'would override a validated parameter' <<<"$out" && ok "amg2023: extra -P rejected" || bad "amg2023: -P not rejected (rc=$rc)"
    out="$(HPCPERF_AMG_P='2 1 1' HPCPERF_GPUS=4 "$R/level2/amg2023/run.sh" CUDA 2>&1)"; rc=$?; out="$(noise <<<"$out")"
    [ "$rc" -eq 2 ] && grep -q 'does not match' <<<"$out" && ok "amg2023: HPCPERF_AMG_P product != HPCPERF_GPUS rejected" || bad "amg2023: P/GPUS mismatch not rejected (rc=$rc)"
else skip=$((skip+1)); fi
if built "$R/build/level2/examinimd/cuda/src/ExaMiniMD"; then
    out="$("$R/level2/examinimd/run.sh" CUDA -il /dev/null 2>&1)"; rc=$?; out="$(noise <<<"$out")"
    [ "$rc" -eq 2 ] && grep -q 'would override' <<<"$out" && ok "examinimd: extra -il rejected" || bad "examinimd: -il not rejected (rc=$rc)"
else skip=$((skip+1)); fi
if built "$R/build/level2/laghos/cuda/laghos"; then
    out="$("$R/level2/laghos/run.sh" CUDA -rs 1 2>&1)"; rc=$?; out="$(noise <<<"$out")"
    [ "$rc" -eq 2 ] && grep -q 'would override' <<<"$out" && ok "laghos: extra -rs rejected" || bad "laghos: -rs not rejected (rc=$rc)"
    out="$(HPCPERF_GPUS=4096 HPCPERF_LAGHOS_RS=2 "$R/level2/laghos/run.sh" CUDA 2>&1)"; rc=$?; out="$(noise <<<"$out")"
    [ "$rc" -eq 2 ] && grep -q 'only 512 elements' <<<"$out" && ok "laghos: elements < ranks rejected before launch" || bad "laghos: element check (rc=$rc): $(head -2 <<<"$out")"
else skip=$((skip+1)); fi

echo "test_run_guards: $pass passed, $failn failed, $skip skipped"
[ "$failn" -eq 0 ]
