#!/usr/bin/env python3
"""l3_verdict -- verdict classes of a Level 3 validate.sh outcome, and a regression-campaign summary.

Exit-code contract of level3/<app>/validate.sh (documented in each script):
    0   PASS               -- the final verdict line ends with 'PASS' (e.g. '...): PASS')
    1   FAIL
    3   PENDING            -- Nyx heat/cool: 'STATE_AND_PARTICLES_PASS; I_R_CHECK_PENDING' (state and particle
                              checks pass, the full acceptance is incomplete). NOT a PASS.
    4   UNSUPPORTED_LAYOUT -- Nyx: different but legal box layouts, comparison not performed. NOT a PASS.
    124 timeout -> FAIL; anything else -> FAIL
Exit-code contract of tools/validate_workspace.sh (agent workspaces): 0/1/3/4 propagated from validate.sh
(numerical layer), 6 REFUSED (workspace-integrity layer: tampering / untrusted baseline; nothing was built or
run), 7 BUILD_FAIL (build layer). REFUSED and BUILD_FAIL are never PASS, never PENDING, and never enter a
scientific or performance summary; rc 3 is PENDING only with the Nyx I_R_CHECK_PENDING line, never for a refusal.
A queue records the exit code of every step (l3_run_recorded in l3_common.sh) and continues; this module
turns (rc, log) into exactly one of PASS / PENDING / UNSUPPORTED_LAYOUT / FAIL / MISSING. PENDING and
UNSUPPORTED_LAYOUT are counted separately, never as PASS, and never enter a performance summary. An exit
code that contradicts the log (rc 0 without a PASS line, rc 3 with a PASS line, a PASS line after a
timeout) is FAIL.

    l3_verdict.py classify --rc N --log FILE
    l3_verdict.py summary --rc-file FILE --log FILE...      rc file: '<log basename> <rc>' per line
"""
import argparse
import os
import re
import sys

CLASSES = ("PASS", "PENDING", "UNSUPPORTED_LAYOUT", "FAIL", "REFUSED", "BUILD_FAIL", "MISSING")
NOISE = re.compile(r"lua|posix|traceback|\[C\]|no file|no field")  # lmod noise on this site
VERDICT_RE = re.compile(r"validation \(.*\): |I_R_CHECK_PENDING|UNSUPPORTED_LAYOUT|: FAIL\b|\bPASS$|validate_workspace: (REFUSED|BUILD_FAIL)")
AUDIT_RE = re.compile(r"audit summary: (\d+) verified, (\d+) mismatch, (\d+) unverified")


def last_verdict_line(text):
    lines = [l for l in text.splitlines() if l.strip() and not NOISE.search(l) and VERDICT_RE.search(l)]
    return lines[-1] if lines else ""


def is_pass_line(line):
    line = line.rstrip()
    return bool(re.search(r"\bPASS$", line)) and not re.search(r"PENDING|UNSUPPORTED|FAIL|not a ", line)


def classify(rc, verdict_line):
    """rc: int or None (log/step missing). verdict_line: the last verdict line of the log ('' if none)."""
    if rc is None:
        return "MISSING"
    line = (verdict_line or "").rstrip()
    if rc == 0:
        return "PASS" if is_pass_line(line) else "FAIL"
    if rc == 3:
        return "PENDING" if "I_R_CHECK_PENDING" in line and not is_pass_line(line) else "FAIL"
    if rc == 4:
        return "UNSUPPORTED_LAYOUT" if "UNSUPPORTED_LAYOUT" in line and not is_pass_line(line) else "FAIL"
    if rc == 6:
        return "REFUSED" if "validate_workspace: REFUSED" in line else "FAIL"
    if rc == 7:
        return "BUILD_FAIL" if "validate_workspace: BUILD_FAIL" in line else "FAIL"
    return "FAIL"


def read_log(path):
    if not path or not os.path.isfile(path):
        return None
    return open(path, errors="replace").read()


def cmd_classify(a):
    text = read_log(a.log)
    if text is None:
        print("MISSING")
        return 0
    print(classify(a.rc, last_verdict_line(text)))
    return 0


def cmd_summary(a):
    rcs = {}
    if a.rc_file:
        for l in open(a.rc_file):
            f = l.split()
            if len(f) >= 2 and f[-1].lstrip("-").isdigit():
                rcs[f[0]] = int(f[-1])
    rows = []
    for path in a.log:
        name = os.path.basename(path)
        text = read_log(path)
        rc = rcs.get(name)
        if text is None:
            rows.append((name, rc, "MISSING", "(log missing)", []))
            continue
        line = last_verdict_line(text)
        cls = classify(rc, line) if rc is not None else "MISSING"
        rows.append((name, rc, cls, line.strip() or "(no verdict line)", AUDIT_RE.findall(text)))
    print("| log | validate.sh rc | class | launcher audits verified/mismatch/unverified (one entry per launched run) | verdict line |")
    print("|---|---|---|---|---|")
    for name, rc, cls, line, au in rows:
        print(f"| {name} | {'?' if rc is None else rc} | **{cls}** | {'; '.join('/'.join(x) for x in au) or 'none found'} | {line[:160]} |")
    counts = {c: sum(1 for r in rows if r[2] == c) for c in CLASSES}
    print()
    print(f"{len(rows)} logs; " + ", ".join(f"{c}={counts[c]}" for c in CLASSES)
          + "  (PASS counts only class PASS; PENDING and UNSUPPORTED_LAYOUT are not passes; REFUSED (workspace integrity) and BUILD_FAIL (build layer) are not scientific outcomes at all; none of these enter a performance summary)")
    unv = [f"{name} [{'; '.join('/'.join(x) for x in au)}]" for name, _, _, _, au in rows if any(x[2] != "0" for x in au)]
    mis = [name for name, _, _, _, au in rows if any(x[1] != "0" for x in au)]
    print("Launcher audits with unverified ranks (GPU-binding evidence gap, not a correctness signal): " + (", ".join(unv) or "none"))
    print("Launcher audits with MISMATCH (wrong GPU observed): " + (", ".join(mis) or "none"))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify"); c.add_argument("--rc", type=int, required=True); c.add_argument("--log", required=True)
    s = sub.add_parser("summary"); s.add_argument("--rc-file"); s.add_argument("--log", nargs="+", required=True)
    a = ap.parse_args()
    return cmd_classify(a) if a.cmd == "classify" else cmd_summary(a)


if __name__ == "__main__":
    sys.exit(main())
