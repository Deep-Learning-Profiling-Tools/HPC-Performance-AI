#!/usr/bin/env python3
"""Standalone correctness check for Rodinia nn (upstream provides none).

Runs the benchmark, recomputes all Euclidean distances from the hurricane db
records on the CPU, and checks the reported k nearest neighbors: their
distances (sorted, against the k smallest CPU distances), their identities
(every reported line is a db record, reported at most as often as it occurs),
the pairing (each reported distance is its own record's distance) and that
they are a k-nearest set (every record closer than the k-th distance is
reported, none farther). The report order is not checked (Rodinia's OpenMP nn
prints its neighbours in slot order, the CUDA one sorted). A query whose k-th
and (k+1)-th distances are tied within the tolerance fails: its record set is
ambiguous, so an exact identity reference cannot be frozen for it.

Usage: verify.py <exe> <filelist> <k> <lat> <lng>
"""
import os, subprocess, sys, math
from collections import Counter
import numpy as np
# tools/golden: when HPCPERF_GOLDEN_DUMP is set (only `tools/golden/hpcperf_golden.py make`, the one-time
# validation of a frozen golden reference), the CPU reference arrays are captured next to the binary's output
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools", "golden"))
import hpcperf_golden_capture as gc  # noqa: E402

exe, filelist, k, lat, lng = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
r = subprocess.run([exe, filelist, "-r", str(k), "-lat", str(lat), "-lng", str(lng)],
                   capture_output=True, text=True)
if r.returncode != 0:
    print(f"FAIL: benchmark exited with {r.returncode}\n{r.stdout}{r.stderr}")
    sys.exit(1)

# parse db records the same way nn_cuda.cu does (lat at char 28, lng at 33; a record is its 48-character line)
recs, dists = [], []
for db in open(filelist).read().split():
    for line in open(db):
        rec = line.rstrip("\n")
        if len(rec) < 38: continue
        la = np.float32(rec[28:33]); lo = np.float32(rec[33:38])
        d = np.float32(math.sqrt(np.float32((np.float32(lat)-la)**2 + (np.float32(lng)-lo)**2)))
        recs.append(rec[:48]); dists.append(float(d))
n = len(recs)
kk = min(k, n)                      # the binary reports min(k, number of records)
srt = sorted(dists)
expected = srt[:kk]
gc.capture("topk_distances.cpu", np.array(expected, dtype=np.float32), "f32")   # sorted ascending, as compared
tol = lambda e: 1e-4 * (1 + abs(e))

got = []                            # (record, reported distance) per "<record> --> Distance=<%f>" line
for line in r.stdout.splitlines():
    if "--> Distance=" in line:
        rec, _, val = line.partition(" --> Distance=")
        got.append((rec, float(val)))
if len(got) != kk:
    print(f"FAIL: expected {kk} reported neighbors, got {len(got)}")
    sys.exit(1)
ok = all(abs(g - e) <= tol(e) for g, e in zip(sorted(g for _, g in got), expected))
if not ok:
    print(f"FAIL: top-{kk} distances differ\n  gpu: {sorted(g for _, g in got)[:20]}\n  cpu: {expected[:20]}")
    sys.exit(1)

# identities: every reported line is a db record, none reported more often than it occurs in the db
mult = Counter(recs)
cpu_of = dict(zip(recs, dists))     # identical lines have identical lat/lng, hence one distance
gcount = Counter(rec for rec, _ in got)
foreign = sorted(rec for rec, c in gcount.items() if c > mult.get(rec, 0))
if foreign:
    print(f"FAIL: {len(foreign)} reported records are not db records or are reported more often than they occur "
          f"(e.g. {foreign[0]!r})")
    sys.exit(1)
# pairing: each reported distance is the distance of the record it is reported with
mis = [(rec, g, cpu_of[rec]) for rec, g in got if abs(g - cpu_of[rec]) > tol(cpu_of[rec])]
if mis:
    print(f"FAIL: {len(mis)} records reported with another record's distance (e.g. {mis[0][0]!r}: reported "
          f"{mis[0][1]}, its distance {mis[0][2]})")
    sys.exit(1)
# a k-nearest set: every record closer than the k-th distance is reported, none farther (both beyond the tolerance)
ek = expected[-1]
missing = Counter(recs[i] for i in range(n) if dists[i] < ek - tol(ek)) - gcount
far = [rec for rec, _ in got if cpu_of[rec] > ek + tol(ek)]
if missing or far:
    print(f"FAIL: not a k-nearest set: {sum(missing.values())} closer records missing, {len(far)} reported records "
          f"farther than the k-th distance {ek}")
    sys.exit(1)
# unambiguous identities: no tie at the k-th distance within the tolerance (otherwise another correct
# implementation may report another, equally near record and an exact identity reference would reject it)
if kk < n and srt[kk] - ek <= 2 * tol(ek):
    print(f"FAIL: the k-th and (k+1)-th distances {ek} / {srt[kk]} are tied within the tolerance: the record set of "
          f"this query is ambiguous (no exact identity reference can be frozen for it)")
    sys.exit(1)
# the CPU distances of the reported records in the order of the binary's topk_records capture (bytewise sorted lines)
gc.capture("topk_record_distances.cpu", np.array([cpu_of[rec] for rec, _ in sorted(got)], dtype=np.float32), "f32")
print(f"PASS: top-{kk} nearest-neighbor distances match CPU reference (and the {kk} record identities, each record's "
      f"own distance and the k-nearest set; k-th / next distance {ek:.6g} / "
      f"{srt[kk] if kk < n else float('nan'):.6g})")
