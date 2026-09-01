#!/usr/bin/env python3
"""Standalone correctness check for Rodinia pathfinder (upstream provides none).

The benchmark generates its input wall with glibc srand(9)/rand()%10 and
prints (BENCH_PRINT) the first wall row plus the final DP result row. This
script reproduces the wall with a glibc rand() implementation, recomputes the
dynamic-programming minimum path on the CPU, and compares.

Usage: verify.py <exe> <cols> <rows> <pyramid_height>
"""
import subprocess, sys

def glibc_rand(seed, n):
    """glibc rand() (TYPE_3 additive feedback generator)."""
    r = [0] * (344 + n)
    r[0] = seed
    for i in range(1, 31):
        r[i] = (16807 * r[i-1]) % 2147483647
    for i in range(31, 34):
        r[i] = r[i-31]
    for i in range(34, 344 + n):
        r[i] = (r[i-3] + r[i-31]) % (1 << 32)
    return [r[i] >> 1 for i in range(344, 344 + n)]

exe, cols, rows, pyr = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
r = subprocess.run([exe, str(cols), str(rows), pyr], capture_output=True, text=True)
if r.returncode != 0:
    print(f"FAIL: benchmark exited with {r.returncode}\n{r.stderr}")
    sys.exit(1)

lines = [l for l in r.stdout.splitlines() if l.strip()]
first_row = list(map(int, lines[-2].split()))
result = list(map(int, lines[-1].split()))
if len(first_row) != cols or len(result) != cols:
    print("FAIL: could not parse BENCH_PRINT output rows")
    sys.exit(1)

rnd = glibc_rand(9, rows * cols)  # M_SEED = 9
wall = [[rnd[i*cols + j] % 10 for j in range(cols)] for i in range(rows)]

if wall[0] != first_row:
    print("FAIL: generated wall row 0 does not match program output "
          "(glibc rand reproduction mismatch)")
    sys.exit(1)

dp = wall[0][:]
for t in range(rows - 1):
    ndp = [0]*cols
    for j in range(cols):
        best = dp[j]
        if j > 0: best = min(best, dp[j-1])
        if j < cols-1: best = min(best, dp[j+1])
        ndp[j] = best + wall[t+1][j]
    dp = ndp

if dp != result:
    bad = sum(1 for a, b in zip(dp, result) if a != b)
    print(f"FAIL: {bad}/{cols} result entries differ from CPU DP")
    sys.exit(1)
print(f"PASS: pathfinder result row matches CPU DP over {rows}x{cols} wall")
