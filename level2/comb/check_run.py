#!/usr/bin/env python3
"""Read the outcome of Comb's own halo-exchange check from the files of one run.

Comb initialises every halo zone analytically and, in each cycle, compares every received
value with its expectation (upstream/include/do_cycles.hpp). A mismatch is written to the
per-process file Comb_<run>_proc<rank> as "test pre-comm ..." / "test post-comm ... expected ..."
and asserted -- but the assert is compiled out in the Release build (build.sh: NDEBUG), so the
exit code says nothing about it. This script is the check that reads what Comb wrote:

  PASS  when at least one Comb_<run>_proc* file exists, none contains a "test pre-comm" /
        "test post-comm" line, and the run's Comb_<run>_summary reports the "test-comm" phase
        (evidence that the comparison cycle ran, "test-comm:  num 1 ...")
  FAIL  otherwise (a missing file is a failure, never an implicit pass)

Usage: check_run.py <dir with the run's Comb_* files>
(tools/inputs `check` copies the files created by the run into its outputs directory.)
"""
import glob, os, re, sys

if len(sys.argv) != 2:
    print("usage: check_run.py <dir>"); sys.exit(2)
d = sys.argv[1]
procs = sorted(f for f in glob.glob(os.path.join(d, "Comb_*_proc*")))
summaries = sorted(f for f in glob.glob(os.path.join(d, "Comb_*_summary")) if not f.endswith(".csv"))
bad = []
if not procs:
    bad.append(f"no Comb_<run>_proc* file in {d}")
if len(summaries) != 1:
    bad.append(f"expected exactly one Comb_<run>_summary in {d}, found {len(summaries)}")
runs = {re.search(r"Comb_(\d+)_", os.path.basename(f)).group(1) for f in procs + summaries}
if len(runs) > 1:
    bad.append(f"files of several runs mixed: {sorted(runs)}")
mismatches = 0
for f in procs:
    n = sum(1 for ln in open(f, errors="replace") if re.match(r"^test (pre|post)-comm .* expected ", ln))
    if n:
        bad.append(f"{os.path.basename(f)}: {n} halo value mismatch line(s)")
    mismatches += n
tested = False
if summaries:
    text = open(summaries[0], errors="replace").read()
    m = re.search(r"^test-comm:\s+num\s+(\d+)", text, re.M)
    tested = bool(m and int(m.group(1)) >= 1)
    if not tested:
        bad.append(f"{os.path.basename(summaries[0])}: no 'test-comm: num >= 1' line -- the comparison cycle did not run")
if bad:
    for b in bad:
        print("  " + b)
    print(f"FAIL: Comb halo check (run {','.join(sorted(runs)) or '?'}): {mismatches} mismatch line(s), test-comm {'ran' if tested else 'missing'}")
    sys.exit(1)
print(f"PASS: Comb halo check (run {','.join(sorted(runs))}): {len(procs)} process file(s), 0 mismatch lines, test-comm phase ran")
