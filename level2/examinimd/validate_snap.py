#!/usr/bin/env python3
"""Check an ExaMiniMD SNAP (Ta06A) log: the initial state and the run's completeness.

The SNAP deck (input/snap/in.snap.Ta06A, LAMMPS' examples/snap case ported by upstream) prints
PotE = 0.000000 in every thermo row -- ExaMiniMD's SNAP force kernel computes forces only, no
energy -- so the energy-conservation bound of validate_lj.py has nothing to work on, and no
analytic lattice sum exists for the SNAP potential. What CAN be checked without inventing a
tolerance:

  1. step-0 temperature == T0 requested by `velocity all create T0 ...` (300.0; the deck's
     velocities are scaled to exactly T0 with the 3N-3 convention, as in validate_lj.py)
  2. every thermo row is finite and the temperature stays positive
  3. the last row is the requested step count (the run completed)

The final temperature is reported as a diagnostic only: the trajectory of a 64-atom
microcanonical system is chaotic, and upstream provides no reference value for it (its
--correctness option compares against a binary dump of an earlier run of itself; no dump ships).
Exit status 0 if all checks pass, 1 otherwise.
"""
import argparse
import math
import re
import sys

ROW = re.compile(r"^(\d+) ([0-9.eE+-]+) (-?[0-9.eE+-]+) (-?[0-9.eE+-]+) ([0-9.eE+-]+) ([0-9.eE+-]+)$")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, help="ExaMiniMD stdout of the SNAP run")
    ap.add_argument("--temp", type=float, default=300.0, help="T0 of the deck's `velocity all create` (default 300.0)")
    ap.add_argument("--steps", type=int, default=100, help="the deck's `run` step count (default 100)")
    ap.add_argument("--temp-rtol", type=float, default=1e-5, help="relative tolerance on T(0) (default 1e-5, as validate_lj.py's abs 1e-5 on T0 = 1.4)")
    a = ap.parse_args()
    rows = []
    for line in open(a.log, errors="replace"):
        m = ROW.match(line.strip())
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    fails, notes = [], []
    if not rows:
        fails.append("no thermo rows found")
    else:
        step0, t0, pe0, e0 = rows[0]
        if step0 != 0:
            fails.append(f"first thermo row is step {step0}, not 0")
        if not math.isfinite(t0) or abs(t0 - a.temp) > a.temp_rtol * a.temp:
            fails.append(f"T(0) = {t0} differs from the deck's velocity-create target {a.temp} (rel {abs(t0 - a.temp) / a.temp:.2e} > {a.temp_rtol:.0e})")
        else:
            notes.append(f"T(0) = {t0:.6f} = T0 {a.temp} (rel {abs(t0 - a.temp) / a.temp:.1e} <= {a.temp_rtol:.0e})")
        bad = [r for r in rows if not all(math.isfinite(v) for v in r[1:]) or r[1] <= 0]
        if bad:
            fails.append(f"{len(bad)} thermo row(s) non-finite or with T <= 0 (first at step {bad[0][0]})")
        else:
            notes.append(f"{len(rows)} thermo rows finite, T > 0")
        if rows[-1][0] != a.steps:
            fails.append(f"last thermo row is step {rows[-1][0]}, the deck runs {a.steps} steps (run incomplete?)")
        else:
            notes.append(f"last row = step {a.steps} (run complete)")
        if any(r[2] != 0.0 for r in rows):
            notes.append("PotE is not 0 in every row: the build computes a SNAP energy (this check does not use it)")
        notes.append(f"diagnostic: T({rows[-1][0]}) = {rows[-1][1]:.6f} (no reference; not a criterion)")
    for n in notes:
        print(f"   {n}")
    if fails:
        print("ExaMiniMD SNAP check: FAIL -- " + "; ".join(fails))
        return 1
    print("ExaMiniMD SNAP check: PASS (initial state T(0) = T0, finite trajectory, run complete; final T diagnostic only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
