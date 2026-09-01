#!/usr/bin/env python3
"""Standalone correctness check for Rodinia nn (upstream provides none).

Runs the benchmark, recomputes all Euclidean distances from the hurricane db
records on the CPU, and checks the reported top-k nearest neighbors.

Usage: verify.py <exe> <filelist> <k> <lat> <lng>
"""
import subprocess, sys, math
import numpy as np

exe, filelist, k, lat, lng = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
r = subprocess.run([exe, filelist, "-r", str(k), "-lat", str(lat), "-lng", str(lng)],
                   capture_output=True, text=True)
if r.returncode != 0:
    print(f"FAIL: benchmark exited with {r.returncode}\n{r.stdout}{r.stderr}")
    sys.exit(1)

# parse db records the same way nn_cuda.cu does (lat at char 28, lng at 33)
dists = []
for db in open(filelist).read().split():
    for line in open(db):
        rec = line.rstrip("\n")
        if len(rec) < 38: continue
        la = np.float32(rec[28:33]); lo = np.float32(rec[33:38])
        d = np.float32(math.sqrt(np.float32((np.float32(lat)-la)**2 + (np.float32(lng)-lo)**2)))
        dists.append(float(d))
dists.sort()
expected = dists[:k]

got = []
for line in r.stdout.splitlines():
    if "--> Distance=" in line:
        got.append(float(line.split("Distance=")[1]))
if len(got) != k:
    print(f"FAIL: expected {k} reported neighbors, got {len(got)}")
    sys.exit(1)
ok = all(abs(g - e) <= 1e-4 * (1 + abs(e)) for g, e in zip(sorted(got), expected))
if not ok:
    print(f"FAIL: top-{k} distances differ\n  gpu: {sorted(got)}\n  cpu: {expected}")
    sys.exit(1)
print(f"PASS: top-{k} nearest-neighbor distances match CPU reference")
