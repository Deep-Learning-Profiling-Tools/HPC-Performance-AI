#!/usr/bin/env python3
"""Dam-break conservation check on ExaMPM's own HDF5 particle dumps (validate.sh's second part,
usable on the dumps of any registered dam-break run).

    check_dambreak.py <dir with particles_*.h5> [--min-dumps N]

For every dump (sorted by step): the particle count must equal the initial count, positions
finite and inside [-0.02, 1.02]^3 (the unit-cube domain with a small tolerance), and the total
volume sum(J)/N_0 within 1 % of 1 (J is the per-particle deformation-gradient determinant, so
sum(J)/N_0 is the relative volume of the incompressible-ish fluid). These are the criteria of
level2/exampm/validate.sh (dam-break section); they need h5dump on PATH. At least --min-dumps
dumps (default 2: the initial state and one evolved dump) must exist, otherwise the run wrote no
evolved state and nothing is verified. Prints "PASS: exampm dam-break check ..." /
"FAIL: exampm dam-break check ..."; exit 0 / 1.
"""
import glob, os, re, subprocess, sys
import numpy as np


def h5time(f):
    out = subprocess.run(["h5dump", "-m", "%.17g", "-a", "/Time", f], capture_output=True, text=True, check=True).stdout
    return float(re.search(r"\(0\):\s*([-+0-9.eE]+)", out).group(1))


def h5arr(f, ds):
    tmp = f + "." + ds + ".bin"
    subprocess.run(["h5dump", "-d", "/" + ds, "-b", "LE", "-o", tmp, f], capture_output=True, check=True)
    a = np.fromfile(tmp, dtype="<f8")
    os.remove(tmp)
    return a


def main(argv):
    if len(argv) < 2:
        print(__doc__); return 2
    d = argv[1]
    min_dumps = int(argv[argv.index("--min-dumps") + 1]) if "--min-dumps" in argv else 2
    files = sorted(glob.glob(os.path.join(d, "particles_*.h5")), key=lambda s: int(re.search(r"_(\d+)\.h5$", s).group(1)))
    errors = []
    if len(files) < min_dumps:
        errors.append(f"{len(files)} dump(s) in {d}, at least {min_dumps} expected (initial state + an evolved dump)")
    n0 = None
    print(f"DamBreak: {len(files)} dumps in {d}")
    print(f"  {'file':18s} {'t':>9s} {'N':>9s} {'min pos':>10s} {'max pos':>10s} {'sum J / N0':>11s}")
    for f in files:
        try:
            t = h5time(f); pos = h5arr(f, "position").reshape(-1, 3); J = h5arr(f, "J")
        except (subprocess.CalledProcessError, OSError, AttributeError) as ex:
            errors.append(f"{os.path.basename(f)}: unreadable ({ex})"); continue
        n = len(pos)
        if n0 is None:
            n0 = n
        volr = J.sum() / n0
        print(f"  {os.path.basename(f):18s} {t:9.6f} {n:9d} {pos.min():+10.5f} {pos.max():+10.5f} {volr:11.6f}")
        if n != n0:
            errors.append(f"{os.path.basename(f)}: particle count {n} != initial {n0}")
        if not np.isfinite(pos).all():
            errors.append(f"{os.path.basename(f)}: non-finite positions")
        elif pos.min() < -0.02 or pos.max() > 1.02:
            errors.append(f"{os.path.basename(f)}: particles outside [-0.02, 1.02]^3")
        if not np.isfinite(volr) or abs(volr - 1.0) > 0.01:
            errors.append(f"{os.path.basename(f)}: total volume sum(J)/N0 = {volr:.5f} off by more than 1 %")
    for e in errors:
        print("   ERROR:", e)
    what = f"{len(files)} dumps: particle count conserved, positions finite inside [-0.02, 1.02]^3, volume sum(J)/N0 within 1 %"
    print(("FAIL" if errors else "PASS") + f": exampm dam-break check ({what})")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
