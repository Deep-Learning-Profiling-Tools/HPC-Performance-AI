#!/usr/bin/env python3
"""Correctness check for Rodinia hotspot3D using its built-in CPU reference.

The benchmark computes the same simulation on the CPU (computeTempCPU) and
prints 'Accuracy: <rmse>'. This wrapper runs it and enforces a threshold.

Usage: verify.py <exe> <grid> <layers> <iters> <power_file> <temp_file>
"""
import subprocess, sys

exe, grid, layers, iters, pf, tf = sys.argv[1:7]
r = subprocess.run([exe, grid, layers, iters, pf, tf, "output.out"],
                   capture_output=True, text=True)
if r.returncode != 0:
    print(f"FAIL: benchmark exited with {r.returncode}\n{r.stdout}{r.stderr}")
    sys.exit(1)
acc = None
for line in r.stdout.splitlines():
    if line.startswith("Accuracy:"):
        acc = float(line.split(":")[1])
if acc is None:
    print(f"FAIL: no 'Accuracy:' line in output\n{r.stdout}")
    sys.exit(1)
if acc > 1e-2:
    print(f"FAIL: accuracy (RMSE vs built-in CPU reference) {acc:.3e} > 1e-2")
    sys.exit(1)
print(f"PASS: accuracy vs built-in CPU reference = {acc:.3e} (threshold 1e-2)")
