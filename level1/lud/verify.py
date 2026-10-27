#!/usr/bin/env python3
"""ctest entry of lud: one plain run that captures its factors, then the post-run residual check.

    verify.py [--workdir DIR] <lud_cuda> [lud arguments ...]          e.g. verify.py lud_cuda -s 256 -v

Runs the binary once with HPCPERF_GOLDEN_DUMP=<DIR>/capture (a fresh directory; the factor matrix is
captured after the timed region) and without the offline switches (HPCPERF_CPU_REFERENCE,
HPCPERF_GOLDEN_MODE, HPCPERF_SKIP_VERIFY are dropped: upstream's single-thread host recomposition does not
run), then applies lud_residual_check.py -- upstream's lud_verify criterion, max |A - L*U| <= 1e-4 over all
n*n elements -- to that capture. Prints the benchmark's output, the check and its PASS/FAIL line; exit 0
only on PASS. The registry (inputs.yaml `check:`) runs the same checker on its own run's capture.
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lud_residual_check as lrc  # noqa: E402

OFFLINE_VARS = ("HPCPERF_CPU_REFERENCE", "HPCPERF_GOLDEN_MODE", "HPCPERF_SKIP_VERIFY", "HPCPERF_GOLDEN_DUMP")


def main(argv):
    workdir = os.getcwd()
    if len(argv) >= 2 and argv[0] == "--workdir":
        workdir, argv = argv[1], argv[2:]
    if not argv:
        print(__doc__)
        return 2
    exe, args = argv[0], argv[1:]
    dump = os.path.join(os.path.abspath(workdir), "capture")
    os.makedirs(os.path.dirname(dump), exist_ok=True)
    if os.path.lexists(dump):          # never read a previous run's capture
        shutil.rmtree(dump)
    env = dict(os.environ)
    for k in OFFLINE_VARS:
        env.pop(k, None)
    env["HPCPERF_GOLDEN_DUMP"] = dump
    r = subprocess.run([exe] + args, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        print(f"FAIL: lud residual: the benchmark exited {r.returncode}; its output is not checked")
        return 1
    return lrc.main(["--dump", dump, "--"] + args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
