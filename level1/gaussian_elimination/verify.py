#!/usr/bin/env python3
"""Standalone correctness check for Rodinia gaussian (upstream provides none).

Runs the benchmark on a matrix file, parses the printed solution vector, and
checks the residual ||Ax - b||_inf on the CPU.

Usage: verify.py <exe> <matrix_file>
"""
import subprocess, sys
import numpy as np

exe, mat = sys.argv[1], sys.argv[2]
r = subprocess.run([exe, "-f", mat], capture_output=True, text=True)
if r.returncode != 0:
    print(f"FAIL: benchmark exited with {r.returncode}\n{r.stdout}{r.stderr}")
    sys.exit(1)

tok = open(mat).read().split()
n = int(tok[0])
a = np.array(tok[1:1+n*n], dtype=np.float64).reshape(n, n)
b = np.array(tok[1+n*n:1+n*n+n], dtype=np.float64)

lines = r.stdout.splitlines()
x = None
for i, line in enumerate(lines):
    if "final solution" in line:
        x = np.array(lines[i+1].split(), dtype=np.float64)
        break
if x is None or len(x) != n:
    print(f"FAIL: could not parse solution vector from output")
    sys.exit(1)

res = np.abs(a @ x - b).max() / max(1.0, np.abs(b).max())
if res > 1e-2:
    print(f"FAIL: residual ||Ax-b||_inf = {res:.3e} > 1e-2")
    sys.exit(1)
print(f"PASS: residual ||Ax-b||_inf = {res:.3e} (n={n}, float32 solve, tol 1e-2)")
