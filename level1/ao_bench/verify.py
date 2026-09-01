#!/usr/bin/env python3
"""Sanity check for aobench (upstream HeCBench provides no verification).

aobench renders a fixed scene with per-pixel deterministic RNG seeds and
writes ao.ppm. This script runs the benchmark twice and checks that:
  1. the PPM output is well-formed with the expected dimensions,
  2. the image is non-degenerate (not constant),
  3. two runs produce byte-identical output (determinism; catches data races).
This is a sanity check, not a full numerical validation -- upstream ships no
CPU reference for this benchmark.

Usage: verify.py <exe> <iterations>
"""
import subprocess, sys, hashlib

exe, iters = sys.argv[1], sys.argv[2]

def run_once():
    r = subprocess.run([exe, iters], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAIL: benchmark exited with {r.returncode}\n{r.stdout}{r.stderr}")
        sys.exit(1)
    return open("ao.ppm", "rb").read()

img1 = run_once()
img2 = run_once()

if not img1.startswith(b"P6"):
    print("FAIL: ao.ppm is not a P6 PPM")
    sys.exit(1)
parts = img1.split(b"\n", 3)
w, h = map(int, parts[1].split())
pixels = parts[3]
if len(pixels) != w * h * 3:
    print(f"FAIL: pixel payload {len(pixels)} != {w}*{h}*3")
    sys.exit(1)
vals = set(pixels[:200000])
if len(vals) < 8:
    print("FAIL: image is (near-)constant; render likely broken")
    sys.exit(1)
if hashlib.sha256(img1).digest() != hashlib.sha256(img2).digest():
    print("FAIL: two runs produced different images (non-deterministic render)")
    sys.exit(1)
print(f"PASS: ao.ppm well-formed ({w}x{h}), non-degenerate, deterministic "
      f"across runs (sha256 {hashlib.sha256(img1).hexdigest()[:16]}...)")
