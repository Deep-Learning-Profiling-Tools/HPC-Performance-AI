#!/usr/bin/env python3
"""Standalone correctness check for Rodinia nn (upstream provides none).

Runs the benchmark, recomputes all Euclidean distances from the hurricane db
records on the CPU, and checks the reported top-k nearest neighbors.

Usage: verify.py <exe> <filelist> <k> <lat> <lng>
"""
import os, subprocess, sys, math
import numpy as np
# tools/golden: when HPCPERF_GOLDEN_DUMP is set (only `tools/golden/hpcperf_golden.py make`, the one-time
# validation of a frozen golden reference), the CPU reference array is captured next to the binary's output
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "golden"))
import hpcperf_golden_capture as gc  # noqa: E402

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
gc.capture("topk_distances.cpu", np.array(expected, dtype=np.float32), "f32")   # sorted ascending, as compared

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
