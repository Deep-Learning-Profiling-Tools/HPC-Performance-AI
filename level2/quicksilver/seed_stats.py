#!/usr/bin/env python3
"""Mean and standard deviation of Quicksilver's final-cycle tallies over seed-varied runs.

    seed_stats.py <dir with seed-*/stdout.log> [--k 4]

Reads the last row of the cycle table of every completed run (columns as level2/quicksilver/
inputs.yaml: census = column 12, num_seg = 13, scalar_flux = 14) and prints, per tally, the mean,
the sample standard deviation over the seeds and the relative half-width k*sigma/mean of the
rule this scatter supports:  |candidate - mean| <= k * sigma  (a candidate whose result lies
outside k sigma of the reference build's own seed-to-seed distribution is wrong or biased).
k is a decision (default shown for k = 4); the numbers are printed as the `near` rule the registry
would take (value = mean, tol = k*sigma/mean), nothing is written.
"""
import glob, os, re, statistics, sys

ROW = re.compile(r'^\s+\d+(\s+\d+){10}\s+(?P<census>\d+)\s+(?P<num_seg>\d+)\s+(?P<scalar_flux>[0-9.eE+-]+)\s+[0-9.eE+-]+\s+[0-9.eE+-]+\s+[0-9.eE+-]+\s*$')


def last_row(path):
    row = None
    for ln in open(path, errors="replace"):
        m = ROW.match(ln)
        if m:
            row = m
    return row


def main(argv):
    if len(argv) < 2:
        print(__doc__); return 2
    d = argv[1]
    k = float(argv[argv.index("--k") + 1]) if "--k" in argv else 4.0
    vals = {"census": [], "num_seg": [], "scalar_flux": []}
    seeds = []
    for run in sorted(glob.glob(os.path.join(d, "seed-*"))):
        log = os.path.join(run, "stdout.log")
        done = os.path.join(run, "DONE")
        if not (os.path.isfile(log) and os.path.isfile(done) and open(done).read().startswith("rc=0")):
            print(f"   {os.path.basename(run)}: incomplete or failed, skipped"); continue
        text = open(log, errors="replace").read()
        if "FAIL::" in text or text.count("PASS::") < 4:
            print(f"   {os.path.basename(run)}: upstream CORAL checks not all PASS, skipped"); continue
        m = last_row(log)
        if not m:
            print(f"   {os.path.basename(run)}: no cycle table row, skipped"); continue
        seeds.append(os.path.basename(run)[5:])
        for key in vals:
            vals[key].append(float(m.group(key)))
    n = len(seeds)
    print(f"seed-varied runs used: {n} ({', '.join(seeds)})")
    if n < 2:
        print("FAIL: fewer than 2 usable seed runs -- no scatter"); return 1
    for key, v in vals.items():
        mean = statistics.mean(v); sd = statistics.stdev(v)
        print(f"{key:12s} mean {mean:.10g}  sigma {sd:.4g}  sigma/mean {sd / mean if mean else float('nan'):.3e}  "
              f"k={k:g}: near value {mean:.10g} tol {k * sd / mean if mean else float('nan'):.3e}   values {v}")
    print(f"PASS: seed scatter from {n} runs (rule candidate: near mean within k*sigma, k = {k:g}; k is a decision)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
