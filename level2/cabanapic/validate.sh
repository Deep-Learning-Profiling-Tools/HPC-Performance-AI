#!/usr/bin/env bash
# Validate the CabanaPIC CUDA or HIP build with upstream's own tests.
#
#   ./validate.sh [CUDA|HIP]
#
# Runs the two executables of upstream tests/ (built by build.sh, ENABLE_TESTS=ON;
# `ctest` in the build tree runs exactly the same two programs):
#
# 1. tests/energy_comparison/2stream-em -- the upstream regression test. It is
#    example.cpp compiled with the deck tests/energy_comparison/2stream-em.cxx
#    (the Weibel problem: nx=1, ny=32, nz=1, nppc=100, 6000 steps, v0=0.0866,
#    EM solver) plus a Custom_Finalizer that, after the run, reads the
#    per-step "energies.txt" the app wrote (columns: step, time, E-field
#    energy, B-field energy) and compares it with
#    tests/energy_comparison/energies_gold.2stream-em.<REAL_TYPE> (upstream's
#    CPU reference, double or float): the first 3581 lines are skipped and the
#    next 1300 (steps 3581..4880, t = 70..95 / omega_pe, the growth-to-
#    saturation phase of the instability) are compared line by line -- the E
#    energy and, separately, the B energy must agree with the gold value to
#    within 10 % relative error |a-b|/min(a,b) on every line (absolute error
#    when a value is ~0; tests/energy_comparison/compare_energies.h,
#    error_margin = 0.10). It prints "E Test Pass: 1"/"B Test Pass: 1" and
#    exits 1 if either fails.
# 2. tests/decks/custom_init -- a 30-step smoke run of decks/custom_init.cxx
#    (1-D two-stream along x with a custom particle initialiser); only its exit
#    status is checked, as in upstream's ctest.
#
# PASS requires: both programs exit 0 and the energy test prints both
# "E Test Pass: 1" and "B Test Pass: 1". The comparison is against upstream's
# CPU gold data, so this checks the GPU port's physics, not just that it runs.
# With the build.sh default REAL_TYPE=double the maximum deviations seen on the
# B200 are ~5e-4 % (E) and ~5e-4 % (B). REAL_TYPE=float FAILS on the B200
# (E energy 12.8 % off the float gold data at line 4778; see README).
#
# The tests run in a fresh directory under ${TMPDIR:-/tmp} (CabanaPIC appends to
# energies.txt in the working directory on every step; on NFS that alone takes
# ~20 s). Logs and the compared energies.txt/e.out/b.out are copied to
# build/level2/cabanapic/<model>/validate/.
#
# Environment overrides:
#   HPCPERF_CABANAPIC_RUNDIR  directory to run in instead (kept afterwards)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true

set -euo pipefail

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level2/cabanapic/$MODEL"
ENERGY_TEST="$BUILD_DIR/tests/energy_comparison/2stream-em"
SMOKE_TEST="$BUILD_DIR/tests/decks/custom_init"

for exe in "$ENERGY_TEST" "$SMOKE_TEST"; do
    if [ ! -x "$exe" ]; then
        echo "validate.sh: $exe not found -- run ./build.sh $BACKEND (with HPCPERF_CABANAPIC_TESTS=ON, the default) first" >&2
        echo "CabanaPIC $BACKEND validation: FAIL"
        exit 1
    fi
done
REAL_TYPE="$(sed -n 's/^REAL_TYPE:[A-Z]*=//p' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null | head -1)"

OUT_DIR="$BUILD_DIR/validate"
mkdir -p "$OUT_DIR"

if [ -n "${HPCPERF_CABANAPIC_RUNDIR:-}" ]; then
    RUN_DIR="$HPCPERF_CABANAPIC_RUNDIR"; mkdir -p "$RUN_DIR"; KEEP=1
else
    RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hpcperf-cabanapic-validate.XXXXXX")"; KEEP=0
fi
cleanup() { [ "$KEEP" -eq 0 ] && rm -rf "$RUN_DIR"; return 0; }
trap cleanup EXIT
cd "$RUN_DIR"

# --- 1. smoke test -----------------------------------------------------------
rm -f energies.txt partloc ex1d
echo "# CabanaPIC $BACKEND: $SMOKE_TEST (30 steps, exit status only)"
set +e
"$SMOKE_TEST" > "$OUT_DIR/custom_init.log" 2>&1
rc_smoke=$?
set -e
echo "#   exit code $rc_smoke; last energies line: $(tail -1 energies.txt 2>/dev/null || echo none)"

# --- 2. energy comparison ----------------------------------------------------
rm -f energies.txt partloc ex1d e.out b.out
echo "# CabanaPIC $BACKEND: $ENERGY_TEST (6000 steps, energies vs energies_gold.2stream-em.${REAL_TYPE:-?})"
set +e
"$ENERGY_TEST" > "$OUT_DIR/2stream-em.log" 2>&1
rc_energy=$?
set -e
cp -f energies.txt e.out b.out "$OUT_DIR/" 2>/dev/null || true
echo "----------------------------------------------------------------"
grep -E 'Max found err|Test Pass' "$OUT_DIR/2stream-em.log" || true
echo "#   exit code $rc_energy; logs in $OUT_DIR/"

if [ "$rc_smoke" -eq 0 ] && [ "$rc_energy" -eq 0 ] \
   && grep -q '^E Test Pass: 1' "$OUT_DIR/2stream-em.log" \
   && grep -q '^B Test Pass: 1' "$OUT_DIR/2stream-em.log"; then
    echo "CabanaPIC $BACKEND validation (2stream-em energies vs upstream gold, REAL_TYPE=${REAL_TYPE:-?}; custom_init smoke): PASS"
    exit 0
else
    echo "CabanaPIC $BACKEND validation (2stream-em energies vs upstream gold, REAL_TYPE=${REAL_TYPE:-?}; custom_init smoke): FAIL"
    exit 1
fi
