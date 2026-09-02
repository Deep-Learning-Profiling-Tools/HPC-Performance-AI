#!/usr/bin/env bash
# Validate the ExaMiniMD CUDA or HIP build on the Lennard-Jones melt.
#
#   ./validate.sh [CUDA|HIP]
#
# Upstream ships no reference outputs (its --dumpbinary/--correctness options
# only compare a run against a dump of an earlier run of itself), so the check
# is built from independent references -- see validate_lj.py for the details:
#
#   Deck A: upstream input/in.lj unchanged (40^3 fcc cells = 256,000 atoms,
#           rho = 0.8442, T0 = 1.4, rc = 2.5, 100 NVE steps, thermo every 10)
#           * T(0) = 1.4                                  (abs 1e-5)
#           * PE(0)/atom = analytic fcc lattice sum with ExaMiniMD's
#             cutoff shift = -6.3328120                   (rel 1e-5)
#           * ETot(0)/atom = PE + 1.5*T0*(N-1)/N = -4.2328202 (rel 1e-5)
#           * energy conservation: max |ETot(t)-ETot(0)|/|ETot(0)| over all
#             thermo rows < 1e-3 (observed 0.9e-4 here)
#   Deck B: the LAMMPS bench/in.lj problem (in.lj with region 20^3 =
#           32,000 atoms, mass 1.0, T0 = 1.44; everything else identical),
#           compared against LAMMPS' published log bench/log.15Jul25.lj.fixed
#           .g++.1 (LAMMPS 12 Jun 2025): step 0 Temp 1.44, E_pair -6.7733681,
#           TotEng -4.6134356; step 100 Temp 0.7574531.
#           * same four checks as deck A (T0 = 1.44, ETot(0) = -4.1728795,
#             drift observed 2.1e-4)
#           * PE(0) - shift = LAMMPS E_pair(0), ETot(0) - shift = LAMMPS
#             TotEng(0), shift = 0.4405561 per atom      (rel 1e-5)
#           * T(100) = LAMMPS Temp(100)                   (rel 1e-3)
#
# Both decks run through run.sh (so the same launcher/GPU settings as the
# benchmark). Total ~10 s. Prints a final PASS/FAIL line, exit code 0/1.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"

LOG_DIR="$R/build/level2/examinimd/$MODEL/run"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/validate.log"
: > "$LOG"

status=0

# --- Deck A: upstream in.lj as shipped -------------------------------------
echo "=== Deck A: input/in.lj (256,000 atoms, T0 = 1.4, 100 steps) ===" | tee -a "$LOG"
HPCPERF_EXAMINIMD_DECK="$HERE/input/in.lj" "$HERE/run.sh" "$BACKEND" 2>&1 | tee "$LOG_DIR/validate_A.out" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
    echo "run.sh exited with $rc" | tee -a "$LOG"
    status=1
fi
python3 "$HERE/validate_lj.py" --log "$LOG_DIR/validate_A.out" --lattice 40 --temp 1.4 \
    2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || status=1

# --- Deck B: LAMMPS bench/in.lj (20^3, mass 1, T0 1.44) ---------------------
echo | tee -a "$LOG"
echo "=== Deck B: LAMMPS bench in.lj (32,000 atoms, T0 = 1.44, 100 steps) ===" | tee -a "$LOG"
DECK_B="$LOG_DIR/in.lj.lammps"
sed -e 's/^region[[:space:]].*/region\t\tbox block 0 20 0 20 0 20/' \
    -e 's/^mass[[:space:]].*/mass\t\t1 1.0/' \
    -e 's/^velocity[[:space:]].*/velocity\tall create 1.44 87287 loop geom/' \
    "$HERE/input/in.lj" > "$DECK_B"
HPCPERF_EXAMINIMD_DECK="$DECK_B" "$HERE/run.sh" "$BACKEND" 2>&1 | tee "$LOG_DIR/validate_B.out" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
    echo "run.sh exited with $rc" | tee -a "$LOG"
    status=1
fi
python3 "$HERE/validate_lj.py" --log "$LOG_DIR/validate_B.out" --lattice 20 --temp 1.44 \
    --lammps-epair0 -6.7733681 --lammps-etot0 -4.6134356 --lammps-temp 100 0.7574531 \
    2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || status=1

echo "----------------------------------------------------------------" | tee -a "$LOG"
if [ "$status" -eq 0 ]; then
    echo "ExaMiniMD $BACKEND validation (in.lj energy + LAMMPS reference): PASS" | tee -a "$LOG"
    exit 0
else
    echo "ExaMiniMD $BACKEND validation (in.lj energy + LAMMPS reference): FAIL" | tee -a "$LOG"
    exit 1
fi
