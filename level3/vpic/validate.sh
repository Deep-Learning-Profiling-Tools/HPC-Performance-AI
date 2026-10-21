#!/usr/bin/env bash
# Correctness check of the VPIC (vpic-kokkos) build: upstream's own analytic single-particle and
# boundary tests (test/integrated/legacy, built by build.sh with ENABLE_INTEGRATED_TESTS=ON) on the GPU,
# plus upstream's to_completion/simple deck (a small Harris-sheet run) as the completion check.
#
#   ./validate.sh [CUDA|HIP]        HPCPERF_GPUS=N (default 1): N must be 1 (one-rank decks; refused otherwise)
#
# The legacy tests are upstream's regression decks with analytic expectations checked INSIDE the deck:
#   accel    a particle in a uniform E field: after each of 10 steps every particle's ux must equal the
#            exact multiple of the field impulse (exact, "ux should be exact multiples")
#   cyclo    a particle gyrating in a uniform B field: |ux - ux_exact|, |uy - uy_exact| <= 0.6e-6 after
#            every step (the deck's own tolerance)
#   inbndj   current deposition of a particle entering the domain against the exact charge-conserving value
#   interpe  field interpolation to a particle position against the exact linear interpolation
# Each prints "pass" (sim_log) and exits 0, or "FAIL" and aborts (non-zero exit). The deck's criteria are
# upstream's, unchanged; the harness adds exit-code, timeout, log-completeness and GPU-backend checks:
# every run must report ENABLE_KOKKOS_CUDA ON in its build banner (the binary's backend) and the launcher
# audit must show the rank on its GPU. pcomm (8 ranks) and outbndj are not built by upstream's CMake
# (test/integrated/legacy/CMakeLists.txt) and are not run. Prints the criteria and the verdict;
# exit 0 PASS, 1 FAIL.
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
[ "$N" = 1 ] || { echo "validate.sh: the integrated test decks are one-rank decks; HPCPERF_GPUS=$N refused"; echo "VPIC $BACKEND validation ($N GPU): FAIL"; exit 1; }
l3_require_materialized "$HERE" || exit 3
PROFILE="$(l3_backend_profile VPIC "$MODEL")"
l3_paths_profile vpic "$PROFILE" "$MODEL" || exit 2
l3_fingerprint_expect_backend "$L3_INSTALL" "$MODEL" || exit 1
RUN_DIR="$L3_BUILD/$L3_RUN_SUBDIR/validate.np$N"
TIMEOUT="${HPCPERF_VALIDATE_TIMEOUT:-900}"
rm -rf "$RUN_DIR"; mkdir -p "$RUN_DIR"
export HPCPERF_GPUS="$N"
case "$MODEL" in cuda) WANT_BANNER="ENABLE_KOKKOS_CUDA: ON";; hip) WANT_BANNER="ENABLE_KOKKOS_HIP: ON";; *) WANT_BANNER="";; esac
echo "validate.sh: VPIC $BACKEND [profile $PROFILE] -- upstream integrated tests (accel, cyclo, inbndj, interpe) + to_completion/simple on $N GPU"
ok=1
run_test() {   # name exe expect_regex
    local name=$1 exe=$2 expect=$3 rc=0
    local d="$RUN_DIR/$name"
    mkdir -p "$d"
    [ -x "$exe" ] || { echo "  [$name] FAIL -- $exe not built (ENABLE_INTEGRATED_TESTS)"; ok=0; return; }
    ( cd "$d" && timeout "$TIMEOUT" "$L3_LAUNCHER" --gpus 1 --bind wrapper -- "$exe" > run.log 2>&1 ) || rc=$?
    local audit; audit="$(grep -a 'hpcperf-launch: audit summary' "$d/run.log" | tail -1 || true)"
    if [ "$rc" -eq 124 ]; then echo "  [$name] FAIL -- timed out after ${TIMEOUT}s"; ok=0; return; fi
    if [ "$rc" -ne 0 ]; then echo "  [$name] FAIL -- exit $rc (see $d/run.log)"; ok=0; return; fi
    if [ -n "$WANT_BANNER" ] && ! grep -aq "$WANT_BANNER" "$d/run.log"; then echo "  [$name] FAIL -- build banner '$WANT_BANNER' missing: not the $BACKEND backend"; ok=0; return; fi
    if grep -aqE '(^|[^a-zA-Z])FAIL([^a-zA-Z]|$)' "$d/run.log"; then echo "  [$name] FAIL -- the deck reported FAIL"; ok=0; return; fi
    if ! grep -aqE "$expect" "$d/run.log"; then echo "  [$name] FAIL -- expected '$expect' not found in run.log (incomplete run)"; ok=0; return; fi
    echo "  [$name] ok -- exit 0, '$expect' found; ${audit:-no launcher audit line}"
}
T="$L3_BUILD/test/integrated"
run_test accel   "$T/legacy/accel"   '(^|[^a-z])pass([^a-z]|$)'
run_test cyclo   "$T/legacy/cyclo"   '(^|[^a-z])pass([^a-z]|$)'
run_test inbndj  "$T/legacy/inbndj"  '(^|[^a-z])pass([^a-z]|$)'
run_test interpe "$T/legacy/interpe" '(^|[^a-z])pass([^a-z]|$)'
run_test simple  "$T/to_completion/simple" '\*\*\* Done \('
if [ "$ok" -eq 1 ]; then echo "VPIC $BACKEND validation ($N GPU, upstream integrated tests): PASS"; exit 0; fi
echo "VPIC $BACKEND validation ($N GPU, upstream integrated tests): FAIL"; exit 1
