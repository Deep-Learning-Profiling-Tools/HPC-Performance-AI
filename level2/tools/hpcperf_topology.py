#!/usr/bin/env python3
"""hpcperf_topology.py -- balanced process-grid helper for Level 2 mini-apps.

Given a total rank count N and a decomposition dimensionality, print the most
balanced process grid PX PY PZ (PX >= PY >= PZ, PX*PY*PZ == N, minimizing the
max/min aspect ratio). Optional application constraints are checked; if N is
infeasible under them the tool FAILS (exit 1) and lists nearby feasible rank
counts instead of silently changing N.

Usage:
  hpcperf_topology.py N [--dims {1,2,3}] [--divides GX,GY,GZ]
                        [--power-of-two] [--max-aspect R] [--near K]

  --dims D          decomposition dimensionality (default 3; D<3 pads with 1)
  --divides GX,GY,GZ  each grid factor must divide the given global extent
                      (use 0 for "no constraint in this dimension")
  --power-of-two    every factor must be a power of two
  --max-aspect R    reject grids with max(P)/min(P) > R
  --near K          how many nearby feasible N to suggest on failure (default 4)

Output on success (stdout):  PX PY PZ      (single line, space separated)
Examples: 4 -> 2 2 1;  8 -> 2 2 2;  40 -> 5 4 2;  64 -> 4 4 4;  80 -> 5 4 4.
"""
import argparse, sys
from itertools import count

def factor_grids(n, dims):
    """All (a,b,c) with a*b*c==n, a>=b>=c, and exactly `dims` factors > 1 allowed
    in the first `dims` slots (trailing slots forced to 1)."""
    grids = []
    if dims == 1:
        grids.append((n, 1, 1))
    elif dims == 2:
        for b in range(1, int(n ** 0.5) + 1):
            if n % b == 0:
                grids.append((n // b, b, 1))
    else:
        for c in range(1, int(round(n ** (1 / 3))) + 2):
            if n % c:
                continue
            m = n // c
            for b in range(c, int(m ** 0.5) + 1):
                if m % b == 0 and m // b >= b:
                    grids.append((m // b, b, c))
    return grids

def is_pow2(x):
    return x & (x - 1) == 0

def feasible(n, args):
    best = None
    for g in factor_grids(n, args.dims):
        if args.power_of_two and not all(is_pow2(f) for f in g):
            continue
        if args.divides:
            ok = True
            for f, ext in zip(g, args.divides):
                if ext and ext % f != 0:
                    ok = False
                    break
            if not ok:
                continue
        aspect = max(g) / min(g)
        if args.max_aspect and aspect > args.max_aspect:
            continue
        key = (aspect, max(g))
        if best is None or key < best[0]:
            best = (key, g)
    return best[1] if best else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--dims", type=int, choices=(1, 2, 3), default=3)
    ap.add_argument("--divides", type=lambda s: tuple(int(x) for x in s.split(",")))
    ap.add_argument("--power-of-two", action="store_true")
    ap.add_argument("--max-aspect", type=float)
    ap.add_argument("--near", type=int, default=4)
    args = ap.parse_args()
    if args.n < 1:
        sys.exit("hpcperf_topology: N must be >= 1")
    if args.divides and len(args.divides) != 3:
        sys.exit("hpcperf_topology: --divides needs GX,GY,GZ (0 = unconstrained)")
    g = feasible(args.n, args)
    if g:
        print(*g)
        return
    # infeasible: report why and suggest nearby feasible N
    sys.stderr.write(
        f"hpcperf_topology: N={args.n} has no feasible {args.dims}D grid under the given "
        f"constraints (divides={args.divides}, power_of_two={args.power_of_two}, "
        f"max_aspect={args.max_aspect}).\n")
    found = []
    for delta in count(1):
        for cand in (args.n - delta, args.n + delta):
            if cand >= 1 and feasible(cand, args):
                found.append(cand)
        if len(found) >= args.near or delta > max(64, args.n):
            break
    sys.stderr.write("hpcperf_topology: nearby feasible rank counts: "
                     + ", ".join(str(x) for x in sorted(set(found))[:args.near]) + "\n")
    sys.exit(1)

if __name__ == "__main__":
    main()
