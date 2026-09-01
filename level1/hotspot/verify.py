#!/usr/bin/env python3
"""Standalone correctness check for Rodinia hotspot (upstream provides none).

Runs the benchmark, then recomputes the transient thermal simulation in
numpy (float32, same constants as hotspot.cu) and compares output.out.

Usage: verify.py <exe> <grid> <pyramid_height> <sim_time> <temp_file> <power_file>
"""
import subprocess, sys
import numpy as np

exe, grid, pyr, sim = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
tf, pf = sys.argv[5], sys.argv[6]
r = subprocess.run([exe, str(grid), pyr, str(sim), tf, pf, "output.out"],
                   capture_output=True, text=True)
if r.returncode != 0:
    print(f"FAIL: benchmark exited with {r.returncode}\n{r.stdout}{r.stderr}")
    sys.exit(1)

temp = np.loadtxt(tf, dtype=np.float32).reshape(grid, grid)
power = np.loadtxt(pf, dtype=np.float32).reshape(grid, grid)

# constants from hotspot.cu
MAX_PD = np.float32(3.0e6); PRECISION = np.float32(0.001)
SPEC_HEAT_SI = np.float32(1.75e6); K_SI = np.float32(100.0)
FACTOR_CHIP = np.float32(0.5); t_chip = np.float32(0.0005)
chip_height = np.float32(0.016); chip_width = np.float32(0.016)
amb_temp = np.float32(80.0)
gh = chip_height / np.float32(grid); gw = chip_width / np.float32(grid)
Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * gw * gh
Rx = gw / (np.float32(2.0) * K_SI * t_chip * gh)
Ry = gh / (np.float32(2.0) * K_SI * t_chip * gw)
Rz = t_chip / (K_SI * gh * gw)
step = PRECISION / (MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI))
sdc = np.float32(step / Cap)
Rx1, Ry1, Rz1 = np.float32(1/Rx), np.float32(1/Ry), np.float32(1/Rz)

t = temp.copy()
for _ in range(sim):
    N = np.vstack([t[0:1, :], t[:-1, :]])   # north neighbor (clamped)
    S = np.vstack([t[1:, :], t[-1:, :]])
    W = np.hstack([t[:, 0:1], t[:, :-1]])
    E = np.hstack([t[:, 1:], t[:, -1:]])
    t = (t + sdc * (power + (S + N - 2*t) * Ry1 +
                    (E + W - 2*t) * Rx1 + (amb_temp - t) * Rz1)).astype(np.float32)

out = np.loadtxt("output.out", usecols=1, dtype=np.float32).reshape(grid, grid)
err = np.abs(out - t) / (1.0 + np.abs(t))
if err.max() > 1e-3:
    print(f"FAIL: max rel err {err.max():.3e} (tol 1e-3)")
    sys.exit(1)
print(f"PASS: hotspot output matches CPU reference (max rel err {err.max():.3e})")
