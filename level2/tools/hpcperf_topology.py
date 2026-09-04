#!/usr/bin/env python3
"""hpcperf_topology.py -- balanced process-grid helper for Level 2 mini-apps.

Given a total rank count N and a decomposition dimensionality, print the most
balanced process grid PX PY PZ (PX*PY*PZ == N, minimizing the max/min aspect
ratio). Optional application constraints are checked; if N is infeasible
under them the tool FAILS (exit 1) and lists nearby feasible rank counts
instead of silently changing N.

With --divides the global grid may be non-cubic, so every AXIS PERMUTATION of
each factorization is tried (e.g. N=6 on an 8x6x4 grid: 3x2x1 does not divide
but its permutation 2x3x1 does). Without constraints the canonical descending
order PX >= PY >= PZ is printed.

Usage:
  hpcperf_topology.py N [--dims {1,2,3}] [--divides GX,GY,GZ]
                        [--power-of-two] [--max-aspect R] [--near K]
  hpcperf_topology.py --self-test

  --dims D          decomposition dimensionality (default 3; D<3 pads with 1)
  --divides GX,GY,GZ  each grid factor must divide the given global extent
                      (use 0 for "no constraint in this dimension")
  --power-of-two    every factor must be a power of two
  --max-aspect R    reject grids with max(P)/min(P) > R
  --near K          how many nearby feasible N to suggest on failure (default 4)

Output on success (stdout):  PX PY PZ      (single line, space separated)
Examples: 4 -> 2 2 1;  8 -> 2 2 2;  40 -> 5 4 2;  64 -> 4 4 4;  80 -> 5 4 4;
          6 --divides 8,6,4 -> 2 3 1.
"""
import argparse, sys
from itertools import count, permutations


def factor_grids(n, dims):
    """All canonical (a,b,c), a>=b>=c, a*b*c==n, with at most `dims` factors > 1."""
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


def candidates(grid, dims, divides):
    """Axis arrangements of one factorization worth considering: the canonical
    order alone when no divisibility constraint is set, otherwise every distinct
    permutation (a factor > 1 may not be placed in a dimension beyond `dims`)."""
    if not divides:
        return [grid]
    out = []
    for p in set(permutations(grid)):
        if any(p[i] > 1 for i in range(dims, 3)):
            continue
        out.append(p)
    return sorted(out, reverse=True)  # deterministic tie-break: lexicographically largest first


def feasible(n, dims, divides=None, power_of_two=False, max_aspect=None):
    best = None
    for g in factor_grids(n, dims):
        if power_of_two and not all(is_pow2(f) for f in g):
            continue
        aspect = max(g) / min(g)
        if max_aspect and aspect > max_aspect:
            continue
        for p in candidates(g, dims, divides):
            if divides and any(ext and ext % f != 0 for f, ext in zip(p, divides)):
                continue
            key = (aspect, max(p), tuple(-x for x in p))  # balanced first, then canonical
            if best is None or key < best[0]:
                best = (key, p)
    return best[1] if best else None


def self_test():
    cases = [
        ((4, 3, None), (2, 2, 1)),
        ((8, 3, None), (2, 2, 2)),
        ((40, 3, None), (5, 4, 2)),
        ((64, 3, None), (4, 4, 4)),
        ((80, 3, None), (5, 4, 4)),
        ((40, 2, None), (8, 5, 1)),
        ((6, 3, (8, 6, 4)), (2, 3, 1)),        # permutation needed: 3x2x1 does not divide
        ((6, 3, (6, 8, 4)), (3, 2, 1)),        # canonical order happens to divide
        ((80, 3, (1280, 1280, 1280)), (5, 4, 4)),
        ((80, 3, (256, 256, 256)), None),      # 5 divides nothing -> infeasible
        ((12, 3, (8, 6, 4)), (2, 3, 2)),
    ]
    bad = 0
    for (n, dims, div), want in cases:
        got = feasible(n, dims, div)
        ok = got == want
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} N={n} dims={dims} divides={div}: got {got}, want {want}")
    # invariants over a range: product matches, divisibility honoured
    for n in range(1, 200):
        for div in (None, (8, 6, 4), (240, 120, 60)):
            g = feasible(n, 3, div)
            if g is None:
                continue
            if g[0] * g[1] * g[2] != n:
                print(f"FAIL product N={n} divides={div}: {g}"); bad += 1
            if div and any(e % f for f, e in zip(g, div)):
                print(f"FAIL divisibility N={n} divides={div}: {g}"); bad += 1
    print("self-test:", "PASS" if bad == 0 else f"FAIL ({bad})")
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?")
    ap.add_argument("--dims", type=int, choices=(1, 2, 3), default=3)
    ap.add_argument("--divides", type=lambda s: tuple(int(x) for x in s.split(",")))
    ap.add_argument("--power-of-two", action="store_true")
    ap.add_argument("--max-aspect", type=float)
    ap.add_argument("--near", type=int, default=4)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    if args.n is None:
        sys.exit("hpcperf_topology: N is required")
    if args.n < 1:
        sys.exit("hpcperf_topology: N must be >= 1")
    if args.divides and len(args.divides) != 3:
        sys.exit("hpcperf_topology: --divides needs GX,GY,GZ (0 = unconstrained)")
    g = feasible(args.n, args.dims, args.divides, args.power_of_two, args.max_aspect)
    if g:
        print(*g)
        return
    sys.stderr.write(
        f"hpcperf_topology: N={args.n} has no feasible {args.dims}D grid under the given "
        f"constraints (divides={args.divides}, power_of_two={args.power_of_two}, "
        f"max_aspect={args.max_aspect}).\n")
    found = []
    for delta in count(1):
        for cand in (args.n - delta, args.n + delta):
            if cand >= 1 and feasible(cand, args.dims, args.divides, args.power_of_two, args.max_aspect):
                found.append(cand)
        if len(found) >= args.near or delta > max(64, args.n):
            break
    sys.stderr.write("hpcperf_topology: nearby feasible rank counts: "
                     + ", ".join(str(x) for x in sorted(set(found))[:args.near]) + "\n")
    sys.exit(1)


if __name__ == "__main__":
    main()
