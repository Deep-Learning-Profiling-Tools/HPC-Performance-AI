#!/usr/bin/env python3
"""Particle-conservation check of a WarpX uniform_plasma run (registered inputs other than validate.sh's smoke case).

The criterion is validate.sh [2], unchanged, with the final step taken from the registered input instead of the
hard-coded 10: the run's reduced diagnostics (diags/reducedfiles/NP.txt, written every step by run.sh) must reach the
registered final step, and the total macroparticle count must be finite and take a single value at every step. The deck
(Examples/Physics_applications/uniform_plasma/inputs_base_3d) is periodic in every direction with one species and no
injection, ionization or resampling, so the count is conserved exactly at any n_cell / step count. The total particle
weight (NP.txt column [4]) is conserved for the same reason and is checked the same way (a single value at every step).
E_particles + E_fields is printed for the record only: the 2-particles-per-cell thermal plasma is not an
energy-conservation test (level3/warpx/README.md), so no energy tolerance exists.

usage: check_particle_conservation.py <reducedfiles dir> <final step>
Prints 'PASS: ...' or 'VALIDATION ERROR: ...'; exit 0 only on PASS."""
import math, os, sys

def load(p):
    return [[float(x) for x in ln.split()] for ln in open(p) if ln.strip() and not ln.startswith("#")]

def main():
    d, want = sys.argv[1], int(sys.argv[2])
    try:
        rows = load(os.path.join(d, "NP.txt"))
        if not rows:
            raise ValueError("NP.txt has no rows")
        steps = [int(r[0]) for r in rows]
        if steps[-1] != want:
            raise ValueError(f"run reached step {steps[-1]}, registered final step {want} (incomplete run)")
        if steps != list(range(steps[0], want + 1)):
            raise ValueError(f"NP.txt does not hold every step {steps[0]}..{want}")
        np_vals, w_vals = set(), set()
        for r in rows:
            if not (math.isfinite(r[2]) and math.isfinite(r[4])):
                raise ValueError(f"non-finite particle number / weight at step {int(r[0])}")
            np_vals.add(r[2]); w_vals.add(r[4])
        print(f"    ParticleNumber over steps {steps[0]}..{steps[-1]}: {sorted(np_vals)}; total weight: {sorted(w_vals)}")
        try:
            ep, ef = load(os.path.join(d, "EP.txt")), load(os.path.join(d, "EF.txt"))
            e0, e1 = ep[0][2] + ef[0][2], ep[-1][2] + ef[-1][2]
            print(f"    for the record: E_particles+E_fields {e0:.6e} J -> {e1:.6e} J (rel change {(e1 - e0) / e0:+.3e}; informational)")
        except (OSError, IndexError):
            print("    for the record: EP.txt / EF.txt not readable (informational only)")
        if len(np_vals) != 1 or len(w_vals) != 1:
            raise ValueError(f"particle number / weight not conserved: {len(np_vals)} / {len(w_vals)} distinct values")
        print(f"PASS: WarpX uniform_plasma particle conservation ({int(next(iter(np_vals))):,} macroparticles, constant and finite at every step 0..{want})")
        return 0
    except (OSError, ValueError) as ex:
        print(f"VALIDATION ERROR: {ex}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
