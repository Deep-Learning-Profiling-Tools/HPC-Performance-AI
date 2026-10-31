#!/usr/bin/env python3
"""Correctness-probe input of hotspot (inputs.yaml g512-p2-t21-pnoise9, purpose: correctness-probe; never timed).

Writes a power map: the vendored power_512 plus a spatially random heat source of 0..9 W per cell,
    P(k) = power_512(k) + 9 * u(k),   u(k) = lowbias32(k) / 2^32   (k = row-major cell index),
one "%.6f" value per line like the vendored files. lowbias32 is a fixed 32-bit integer hash, so every run on every
platform writes the same bytes (the workload identity hashes this script and power_512).

Why: with the vendored power map every cell changes by only 0.016-0.018 K per step, almost all of it the leakage
term (amb_temp - T) * Rz_1 of cuda/hotspot.cu:197-200; the power term is at most 0.00096 K per step because the time
step is chosen so that the maximum power density MAX_PD heats a cell by PRECISION = 0.001 K per step
(cuda/hotspot.cu:237-238). That is 5.7e-5 of the 1 + |T| ~ 324 K denominator of max_rel1p, so a no-op, skipped
iterations, a stale block or a dropped power term stay inside the tolerance. With this map every cell is heated by
0..3.1 K per step (step / Cap = 0.341 K/W at grid 512), a different amount in every cell, so every term of the update
(power, the four neighbours, the leakage) moves every cell of every block by far more than f32 rounding in every step.
The temperature stays within 333..394 K over the probe's 21 steps.

Usage: gen_probe.py <power_512> <output file>
"""
import os
import sys

N = 512
AMPLITUDE_W = 9.0
M32 = 0xFFFFFFFF


def lowbias32(x):
    """Chris Wellons' lowbias32 integer hash (public domain)."""
    x &= M32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & M32
    x ^= x >> 15
    x = (x * 0x846CA68B) & M32
    x ^= x >> 16
    return x


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as f:
        p = [float(line) for line in f if line.strip()]
    if len(p) != N * N:
        sys.exit(f"gen_probe.py: {src} has {len(p)} values, expected {N * N}")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = dst + ".tmp"
    with open(tmp, "w") as f:
        f.writelines("%.6f\n" % (p[k] + AMPLITUDE_W * (lowbias32(k) / 4294967296.0)) for k in range(N * N))
    os.replace(tmp, dst)


if __name__ == "__main__":
    main()
