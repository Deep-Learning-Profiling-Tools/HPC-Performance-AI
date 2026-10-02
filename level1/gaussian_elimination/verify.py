#!/usr/bin/env python3
"""Standalone correctness check for Rodinia gaussian (upstream provides none).

Runs the benchmark, parses the printed solution vector and checks the residual
||Ax - b||_inf / max(1, ||b||_inf) on the CPU in float64.

Usage: verify.py <exe> <matrix_file>          (the ctest form)
       verify.py <exe> -f <matrix_file>       (the benchmark's own arguments)
       verify.py <exe> -s <n>                 (the benchmark's generated system)

With -s the benchmark builds the system itself (cuda/gaussian.cu create_matrix): the
symmetric Toeplitz matrix a[i][j] = 10 exp(-0.01 |i - j|) and b = 1. The same system is
rebuilt here in float64 from the same formula (the program keeps the coefficients in
float32, a ~1e-7 relative difference, far below the residual tolerance).

Tolerance: scaled residual <= 1e-2 for the float32 elimination without pivoting; it holds
for the ctest matrix (n = 208). Whether it holds at a larger n is established by running
this check, never by adjusting the tolerance.
"""
import subprocess, sys
import numpy as np

TOL = 1e-2


def usage():
    print("usage: verify.py <exe> <matrix_file> | verify.py <exe> -f <matrix_file> | verify.py <exe> -s <n>")
    sys.exit(2)


if len(sys.argv) == 3:
    exe, mode, arg = sys.argv[1], "-f", sys.argv[2]
elif len(sys.argv) == 4 and sys.argv[2] in ("-f", "-s"):
    exe, mode, arg = sys.argv[1], sys.argv[2], sys.argv[3]
else:
    usage()

if mode == "-f":
    tok = open(arg).read().split()
    n = int(tok[0])
    a = np.array(tok[1:1 + n * n], dtype=np.float64).reshape(n, n)
    b = np.array(tok[1 + n * n:1 + n * n + n], dtype=np.float64)
    system = f"file {arg}"
else:
    n = int(arg)
    if n <= 0:
        usage()
    idx = np.arange(n)
    a = 10.0 * np.exp(-0.01 * np.abs(idx[:, None] - idx[None, :]))   # create_matrix: lamda = -0.01
    b = np.ones(n, dtype=np.float64)                                 # b[j] = 1.0
    system = f"generated Toeplitz system, n = {n}"

r = subprocess.run([exe, mode, arg], capture_output=True, text=True)
if r.returncode != 0:
    print(f"FAIL: benchmark exited with {r.returncode}\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    sys.exit(1)

lines = r.stdout.splitlines()
x = None
for i, line in enumerate(lines):
    if "final solution" in line:
        x = np.array(lines[i + 1].split(), dtype=np.float64)
        break
if x is None or len(x) != n:
    print("FAIL: could not parse solution vector from output")
    sys.exit(1)
if not np.isfinite(x).all():
    print("FAIL: solution vector contains non-finite values")
    sys.exit(1)

res = np.abs(a @ x - b).max() / max(1.0, np.abs(b).max())
if res > TOL:
    print(f"FAIL: residual ||Ax-b||_inf = {res:.3e} > {TOL} ({system})")
    sys.exit(1)
print(f"PASS: residual ||Ax-b||_inf = {res:.3e} (n={n}, float32 solve, tol {TOL}, {system})")
