#!/usr/bin/env python3
"""Check one measured conformance probe against expected.json; record the verdict.

    check.py <raw run dir> <sentinel value>

Writes tools/timing/platforms/<platform>.json. Exit 0 on pass, 1 on fail.
"""
import datetime
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, TOOLS)
import summarize  # noqa: E402


def main(raw, sentinel):
    exp = json.load(open(os.path.join(HERE, "expected.json")))
    rec = summarize.build_record(raw)
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    check("run status", rec and rec["status"] == "ok", rec["status"] if rec else "no record")
    roi = rec["roi"] if rec else {}
    check("clean ROI entries", roi.get("entries") == exp["roi"]["entries"], f"entries={roi.get('entries')}")
    check("exclude carved out", (roi.get("excluded_s") or 0) > 0, f"excluded_s={roi.get('excluded_s')}")

    collector = rec["measurement"]["collector"]["name"] if rec else "none"
    if collector != "none":
        dev = rec.get("device") or {}
        for k, v in exp["inside_roi"].items():
            check(f"inside ROI {k}", dev.get(k) == v, f"{dev.get(k)} (expected {v})")
        whole = rec["context"].get("whole_process") or {}
        for k, v in exp["whole_process"].items():
            check(f"whole process {k}", whole.get(k) == v, f"{whole.get(k)} (expected {v})")
        tol = exp["tolerance"]
        c, p, m = roi.get("wall_s"), roi.get("profiled_wall_s"), roi.get("profiled_marker_wall_s")
        if c and p:
            lim = max(tol["profiled_vs_clean_roi_abs_s"], tol["profiled_vs_clean_roi_rel"] * c)
            check("profiled vs clean ROI", abs(p - c) <= lim, f"profiled {p:.6f} s, clean {c:.6f} s, limit {lim:.6f}")
        else:
            check("profiled vs clean ROI", False, f"profiled={p} clean={c}")
        if p and m:
            check("trace ROI vs marker log (same run)", abs(p - m) <= tol["marker_vs_trace_roi_abs_s"],
                  f"trace {p:.6f} s, markers {m:.6f} s -- one timeline")
        leaked = []
        for f in glob.glob(os.path.join(raw, "prof", "*")):
            if os.path.isfile(f) and sentinel.encode() in open(f, "rb").read():
                leaked.append(os.path.basename(f))
        check("caller environment isolated from the profiler output", not leaked,
              f"sentinel found in {leaked}" if leaked else
              f"sentinel absent; the profiler recorded {rec['profiler'].get('recorded_env_name_count', '?')} variable names")

    ok = all(c["ok"] for c in checks)
    status = ("pass" if ok else "fail") if collector != "none" else ("pass-clean-only" if ok else "fail")
    platform = rec["platform"] if rec else "unknown"
    verdict = {
        "platform": platform,
        "conformance": {
            "status": status,
            "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "collector": collector,
            "collector_version": rec["measurement"]["collector"]["version"] if rec else None,
            "probe": "tools/timing/probes/conformance/probe.cu",
            "raw_dir": os.path.relpath(raw, os.path.dirname(os.path.dirname(TOOLS))),
            "checks": checks,
        },
        "device": rec["platform_info"]["device"] if rec else None,
    }
    os.makedirs(os.path.join(TOOLS, "platforms"), exist_ok=True)
    out = os.path.join(TOOLS, "platforms", f"{platform}.json")
    with open(out, "w") as f:
        json.dump(verdict, f, indent=2)
        f.write("\n")
    for c in checks:
        print(f"  {'ok  ' if c['ok'] else 'FAIL'} {c['check']}: {c['detail']}")
    print(f"conformance: {status.upper()} for {platform} with {collector} -> {os.path.relpath(out, os.path.dirname(os.path.dirname(TOOLS)))}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
