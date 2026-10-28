#!/usr/bin/env python3
"""Generator of data/temp_64x8_amp10, the initial temperature of the correctness probe g64-l8-i50-amp10.

The probe's initial field is upstream's own data/temp_64x8 (Rodinia data/hotspot3D, 64x64x8, uniform random
320-340 K) with its fluctuations about 330 K amplified tenfold, T' = 330 + 10 (T - 330) (230-430 K), computed in
exact decimal arithmetic and written in the same one-value-per-line format, so the output is byte-reproducible.
The power map stays upstream's data/power_64x8; the kernels and the CPU reference are unchanged. Why: see
inputs.yaml (`g64-l8-i50-amp10`, probe_basis) and README.md ("Correctness probe").

usage: gen_temp_amp10.py [--check]   (--check: regenerate in memory and compare with the committed file)
"""
import hashlib, sys
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC, DST = HERE / "temp_64x8", HERE / "temp_64x8_amp10"
SRC_SHA256 = "6f1d1f8adf0b47ac1b86195bc8644fa51bfcd8550490d39b4b8d93af70eeb403"   # upstream data/hotspot3D/temp_64x8


def generate() -> bytes:
    raw = SRC.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SRC_SHA256:
        sys.exit(f"{SRC}: not the vendored upstream file (sha256 mismatch)")
    vals = raw.decode().split()
    if len(vals) != 64 * 64 * 8:
        sys.exit(f"{SRC}: {len(vals)} values, expected {64 * 64 * 8}")
    mean, gain = Decimal(330), Decimal(10)
    return "".join(f"{mean + gain * (Decimal(v) - mean):.6f}\n" for v in vals).encode()


if __name__ == "__main__":
    out = generate()
    if "--check" in sys.argv[1:]:
        ok = DST.is_file() and DST.read_bytes() == out
        print(f"{DST.name}: {'matches' if ok else 'DIFFERS from'} the generator output (sha256 {hashlib.sha256(out).hexdigest()})")
        sys.exit(0 if ok else 1)
    DST.write_bytes(out)
    print(f"wrote {DST} ({len(out)} bytes, sha256 {hashlib.sha256(out).hexdigest()})")
