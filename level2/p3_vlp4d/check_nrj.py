#!/usr/bin/env python3
"""Structural check of a vlp4d diagnostics file (nrj.out: t, log||E||_2, sum(phi) per output step).

    check_nrj.py <nrj.out> --lines N

The file must have exactly N rows (one per output step of the deck plus the initial state), three
finite numbers each, strictly increasing t, and the mass column bounded by 1e-9 in absolute value
(sum(phi) is a discrete conservation residual; vlp4d prints it at round-off, 1e-15 .. 1e-16 here).
It establishes that the run completed the whole deck; the science comparison of the rows is done
by the registry's file-sourced quantities (exact rule against the working baseline). Prints
"PASS: vlp4d nrj check ..." / "FAIL: vlp4d nrj check ..."; exit 0 / 1.
"""
import math, sys


def main(argv):
    if len(argv) < 2:
        print(__doc__); return 2
    path = argv[1]
    want = int(argv[argv.index("--lines") + 1]) if "--lines" in argv else None
    errors, rows = [], []
    try:
        for ln in open(path, errors="replace"):
            if ln.strip():
                rows.append([float(x) for x in ln.split()])
    except (OSError, ValueError) as ex:
        print(f"FAIL: vlp4d nrj check: {path}: {ex}"); return 1
    if want is not None and len(rows) != want:
        errors.append(f"{len(rows)} rows, expected {want}")
    for i, r in enumerate(rows):
        if len(r) != 3 or not all(math.isfinite(x) for x in r):
            errors.append(f"row {i}: not three finite numbers: {r}")
        elif i and r[0] <= rows[i - 1][0]:
            errors.append(f"row {i}: t {r[0]} not increasing")
        elif abs(r[2]) > 1e-9:
            errors.append(f"row {i}: |sum(phi)| = {abs(r[2]):.3e} > 1e-9")
    if rows:
        print(f"   nrj.out: {len(rows)} rows, t = {rows[0][0]:g} .. {rows[-1][0]:g}, final log||E||_2 = {rows[-1][1]:.13e}, |sum(phi)| max = {max(abs(r[2]) for r in rows if len(r) == 3):.3e}")
    for e in errors:
        print("   ERROR:", e)
    print(("FAIL" if errors else "PASS") + f": vlp4d nrj check ({len(rows)} rows, three finite columns, t increasing, |sum(phi)| <= 1e-9)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
