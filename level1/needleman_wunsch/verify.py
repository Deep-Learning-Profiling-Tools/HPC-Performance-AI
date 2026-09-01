#!/usr/bin/env python3
"""Standalone correctness check for Rodinia needleman_wunsch.

The benchmark generates its sequences with glibc srand(7)/rand()%10 and (with
TRACEBACK enabled, an upstream-provided code path) writes the traceback values
to result.txt. This script reproduces the input with a glibc rand()
implementation, recomputes the NW dynamic-programming matrix on the CPU
(anti-diagonal vectorized), walks the same traceback, and compares.

Usage: verify.py <exe> <dim> <penalty>
"""
import subprocess, sys
import numpy as np

def glibc_rand(seed, n):
    r = [0] * (344 + n)
    r[0] = seed
    for i in range(1, 31):
        r[i] = (16807 * r[i-1]) % 2147483647
    for i in range(31, 34):
        r[i] = r[i-31]
    for i in range(34, 344 + n):
        r[i] = (r[i-3] + r[i-31]) % (1 << 32)
    return [r[i] >> 1 for i in range(344, 344 + n)]

BLOSUM62 = [
[ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-2,-1, 0,-4],
[-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1, 0,-1,-4],
[-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3, 3, 0,-1,-4],
[-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3, 4, 1,-1,-4],
[ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,-4],
[-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2, 0, 3,-1,-4],
[-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4],
[ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1,-2,-1,-4],
[-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3, 0, 0,-1,-4],
[-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-3,-3,-1,-4],
[-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-4,-3,-1,-4],
[-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2, 0, 1,-1,-4],
[-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-3,-1,-1,-4],
[-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-3,-3,-1,-4],
[-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-2,-1,-2,-4],
[ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2, 0, 0, 0,-4],
[ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-1,-1, 0,-4],
[-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-4,-3,-2,-4],
[-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-3,-2,-1,-4],
[ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-3,-2,-1,-4],
[-2,-1, 3, 4,-3, 0, 1,-1, 0,-3,-4, 0,-3,-3,-2, 0,-1,-4,-3,-3, 4, 1,-1,-4],
[-1, 0, 0, 1,-3, 3, 4,-2, 0,-3,-3, 1,-1,-3,-1, 0,-1,-3,-2,-2, 1, 4,-1,-4],
[ 0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2, 0, 0,-2,-1,-1,-1,-1,-1,-4],
[-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 1],
]
LIMIT = -999

exe, dim, penalty = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
r = subprocess.run([exe, str(dim), str(penalty)], capture_output=True, text=True)
if r.returncode != 0:
    print(f"FAIL: benchmark exited with {r.returncode}\n{r.stderr}")
    sys.exit(1)

max_rows = max_cols = dim + 1
# reproduce input generation (needle.cu, srand(7); row seq then col seq)
rnd = glibc_rand(7, 2*(dim))
seq_r = [rnd[i-1] % 10 + 1 for i in range(1, max_rows)]       # input_itemsets[i*max_cols]
seq_c = [rnd[dim + j-1] % 10 + 1 for j in range(1, max_cols)] # input_itemsets[j]

B = np.array(BLOSUM62, dtype=np.int64)
ref = np.zeros((max_rows, max_cols), dtype=np.int64)
ref[1:, 1:] = B[np.array(seq_r)][:, np.array(seq_c)]

# DP (anti-diagonal vectorization), same recurrence as needle kernels
M = np.zeros((max_rows, max_cols), dtype=np.int64)
M[0, 1:] = -penalty * np.arange(1, max_cols)
M[1:, 0] = -penalty * np.arange(1, max_rows)
for d in range(2, max_rows + max_cols - 1):
    lo = max(1, d - (max_cols - 1)); hi = min(max_rows - 1, d - 1)
    if lo > hi: continue
    i = np.arange(lo, hi + 1); j = d - i
    M[i, j] = np.maximum.reduce([
        M[i-1, j-1] + ref[i, j], M[i, j-1] - penalty, M[i-1, j] - penalty])

# traceback exactly as the TRACEBACK block in needle.cu
vals = []
i = j = max_rows - 2
first = True
while True:
    if first:
        vals.append(int(M[i, j])); first = False
    if i == 0 and j == 0: break
    if i > 0 and j > 0:
        nw, w, n = M[i-1, j-1], M[i, j-1], M[i-1, j]
    elif i == 0:
        nw = n = LIMIT; w = M[i, j-1]
    else:
        nw = w = LIMIT; n = M[i-1, j]
    # exact port of the upstream traceback quirks (chained ifs, then movement)
    new_nw = int(nw) + int(ref[i, j]); new_w = int(w) - penalty; new_n = int(n) - penalty
    tb = max(new_nw, new_w, new_n)
    if tb == new_nw: tb = nw
    if tb == new_w: tb = w
    if tb == new_n: tb = n
    tb = int(tb)
    vals.append(tb)
    if tb == nw:
        i -= 1; j -= 1
    elif tb == w:
        j -= 1
    elif tb == n:
        i -= 1
    else:
        print("FAIL: traceback made no progress (unexpected)"); sys.exit(1)

got = []
with open("result.txt") as f:
    for line in f:
        if line.startswith("print"): continue
        got.extend(int(t) for t in line.split())
if got != vals:
    n_bad = sum(1 for a, b in zip(got, vals) if a != b) + abs(len(got)-len(vals))
    print(f"FAIL: traceback differs ({n_bad} entries; gpu len {len(got)}, cpu len {len(vals)})")
    sys.exit(1)
print(f"PASS: NW traceback ({len(vals)} values) matches CPU DP reference")
