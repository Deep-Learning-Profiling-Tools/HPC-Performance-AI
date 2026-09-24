#!/usr/bin/env python3
"""The current result of every registered input, from timing records -- one rule set for the report
(report.py), the summaries built on it and any campaign bookkeeping.

    registry_view.py [--json OUT] RESULTS_ROOT...

Inputs are listed from the registry (level<N>/*/inputs.yaml), never from the record names, so an input
without a successful record is still listed (failed / not measured). For every record the run verifier
(verify_registry_runs.py, read-only) decides what it is:

  PASS        the run got the input as the registry defines it NOW -> may be a current result
  SUPERSEDED  the run got the workload its identity records, but the registry has since redefined the
              input -> kept as history of the old workload, never current, never compared with the new one
  INVALIDATED the raw run carries INVALIDATED.json (it ran another workload than its input) -> history only
  NOT_RUN     the program never reached the ROI (build missing, abort before the ROI) -> a failed attempt
  FAIL / INSUFFICIENT  the verifier found a contradiction / missing evidence -> never current

Records of one input are POOLED into one measurement only when they are the same measurement
configuration: same platform, same workload identity (sha256 of the stored identity), same executable
(sha256), same source commit and the same protocol apart from the number of clean runs (warm-up runs,
profiled runs, collector). This is how an adaptive extension (3 clean runs + 2 more of the same
configuration) becomes one 5-sample result; records of different campaigns, code, binaries or protocols
stay separate measurements. The current result of an input is its newest pooled PASS measurement.

Statistics of a pooled measurement: median of all clean-run ROI samples; spread = (max - min) / median
(the stability criterion: stable when <= 0.10); cv = sample standard deviation / median (reported, not
used for the decision).
"""

import argparse
import glob
import hashlib
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "tools", "inputs"))

import verify_registry_runs as V  # noqa: E402

SCHEMA = "hpcperf-timing-2"
STABLE_SPREAD = 0.10
ANNOTATIONS = "annotations.json"   # optional, per results root: correctness evidence and blockers (see load_annotations)


def workload_key(workload):
    return hashlib.sha256(json.dumps(workload or {}, sort_keys=True).encode()).hexdigest()


def load_records(roots):
    recs = []
    for root in roots:
        for p in sorted(glob.glob(os.path.join(root, "level[0-9]", "*", "*", "*.json"))):
            try:
                r = json.load(open(p))
            except (OSError, ValueError):
                continue
            if r.get("schema") == SCHEMA and r.get("level") in (1, 2):
                r["_path"] = p
                recs.append(r)
    return recs


def classify(recs, repo=REPO, rules=None):
    """Attach _verdict (PASS/SUPERSEDED/INVALIDATED/NOT_RUN/FAIL/INSUFFICIENT) and _problems to each record."""
    rules = rules or V.load_rules(V.DEFAULT_RULES)
    for r in recs:
        raw = os.path.join(repo, (r.get("provenance") or {}).get("raw_dir") or "")
        if os.path.isfile(os.path.join(raw, "INVALIDATED.json")):
            r["_verdict"], r["_problems"] = "INVALIDATED", [json.load(open(os.path.join(raw, "INVALIDATED.json"))).get("reason", "")]
            continue
        v = V.verify_record(r["_path"], repo, rules)
        r["_verdict"], r["_problems"] = v["verdict"], v["problems"] + v["gaps"]
    return recs


def _config(r):
    m = r.get("measurement") or {}
    p = m.get("protocol") or {}
    reg = r.get("registry") or {}
    return (r.get("platform") or "-", reg.get("identity_sha256") or "-", (r.get("inputs") or {}).get("exe_sha256") or "-",
            (r.get("provenance") or {}).get("git_commit") or "-", p.get("warmup_runs"), p.get("profiled_runs"),
            (m.get("collector") or {}).get("name"))


def measurements(recs):
    """Pool the ok records of one input into measurement sets (see the module docstring)."""
    sets = {}
    for r in sorted(recs, key=lambda r: r["run_id"]):
        if r["status"] != "ok" or r["_verdict"] not in ("PASS", "SUPERSEDED"):
            continue
        sets.setdefault(_config(r), []).append(r)
    out = []
    for cfg, rs in sets.items():
        s = [x for r in rs for x in (r.get("roi") or {}).get("runs_s") or []]
        if not s:
            continue
        med = statistics.median(s)
        out.append({"platform": cfg[0], "records": rs, "run_ids": [r["run_id"] for r in rs], "samples": s,
                    "median": med, "min": min(s), "max": max(s),
                    "spread": (max(s) - min(s)) / med if med else None,
                    "cv": statistics.stdev(s) / med if (med and len(s) > 1) else None,
                    "verdict": "SUPERSEDED" if any(r["_verdict"] == "SUPERSEDED" for r in rs) else "PASS",
                    "workload_key": workload_key(((rs[0].get("registry") or {}).get("identity") or {}).get("workload")),
                    "utc_first": min(r.get("utc") or "" for r in rs), "utc_last": max(r.get("utc") or "" for r in rs),
                    "git_commit": cfg[3], "exe_sha256": cfg[2], "protocol": [(r.get("measurement") or {}).get("protocol") for r in rs]})
    out.sort(key=lambda m: m["run_ids"][-1])
    for m in out:
        m["stable"] = m["spread"] is not None and m["spread"] <= STABLE_SPREAD
        m["adaptive"] = len(m["records"]) > 1
    return out


def registered_inputs(repo=REPO):
    import hpcperf_inputs as hi
    out = []
    for lvl in (1, 2, 3):
        for f in sorted(glob.glob(os.path.join(repo, f"level{lvl}", "*", "inputs.yaml"))):
            doc = hi.load(os.path.dirname(f))
            for inp in doc["inputs"]:
                out.append({"level": lvl, "benchmark": doc["benchmark"], "input_id": inp["id"], "case": inp.get("case"),
                            "variant": inp.get("variant"), "source_kind": (inp.get("source") or {}).get("kind"),
                            "input_form": inp.get("input_form", "runtime"), "params": inp.get("params") or {},
                            "args": [str(a) for a in inp.get("args") or []], "env": inp.get("env") or {},
                            "materialized": inp.get("materialized", True) is not False,
                            "workload_key": workload_key(hi.registry_identity(doc, inp)["workload"]) if lvl < 3 else None})
    return out


def load_annotations(roots):
    """{(level, benchmark, input_id): {"correctness": .., "correctness_basis": .., "blocker": ..}} merged over the
    roots, plus campaign notes. Kept next to the records because correctness evidence and blocker
    diagnoses come from outside the timing runs."""
    ann, meta = {}, {"notes": [], "campaign": {}}
    for root in roots:
        p = os.path.join(root, ANNOTATIONS)
        if not os.path.isfile(p):
            continue
        d = json.load(open(p))
        for a in d.get("inputs") or []:
            ann.setdefault((a["level"], a["benchmark"], a["input_id"]), {}).update({k: v for k, v in a.items() if k not in ("level", "benchmark", "input_id")})
        meta["notes"] += d.get("notes") or []
        meta["campaign"].update(d.get("campaign") or {})
    return ann, meta


def current_view(roots, repo=REPO):
    """One row per registered input: status, current pooled measurement, history of every attempt."""
    recs = classify(load_records(roots), repo)
    ann, meta = load_annotations(roots)
    by = {}
    for r in recs:
        by.setdefault((r["level"], r["app"], r["case"]), []).append(r)
    rows = []
    for inp in registered_inputs(repo):
        key = (inp["level"], inp["benchmark"], inp["input_id"])
        mine = sorted(by.pop(key, []), key=lambda r: r["run_id"])
        row = dict(inp, attempts=[{"run_id": r["run_id"], "utc": r.get("utc"), "status": r["status"], "verdict": r["_verdict"],
                                   "roi_s": (r.get("roi") or {}).get("wall_s"), "clean_runs": len((r.get("roi") or {}).get("runs_s") or []),
                                   "git_commit": ((r.get("provenance") or {}).get("git_commit") or "")[:10],
                                   "workload_key": workload_key(((r.get("registry") or {}).get("identity") or {}).get("workload")),
                                   "problems": r["_problems"]} for r in mine])
        sets = measurements(mine)
        cur = [m for m in sets if m["verdict"] == "PASS" and m["workload_key"] == inp["workload_key"]]
        row["history_sets"] = sets
        row["current_by_platform"] = {}
        for m in cur:                                   # sets are ordered oldest -> newest
            row["current_by_platform"][m["platform"]] = m
        row["current"] = cur[-1] if cur else None
        if inp["level"] == 3:
            row["status"] = "ROI_NOT_SUPPORTED"
        elif row["current"]:
            row["status"] = "SUCCESS"
        elif mine:
            row["status"] = "RUN_FAILED"
        else:
            row["status"] = "NOT_MEASURED"
        row["run_verification"] = "PASS" if row["current"] else (mine[-1]["_verdict"] if mine else None)
        row.update(ann.get(key, {}))
        rows.append(row)
    orphans = sorted(f"L{k[0]} {k[1]}/{k[2]}" for k in by)       # records of names that are not registered inputs
    return rows, recs, meta, orphans


def counts(rows, recs, orphans=()):
    def n(p):
        return sum(1 for r in rows if p(r))
    L = (1, 2, 3)
    return {
        "registered_inputs": {f"level{l}": n(lambda r, l=l: r["level"] == l) for l in L} | {"total": len(rows)},
        "roi_success": {f"level{l}": n(lambda r, l=l: r["level"] == l and r["status"] == "SUCCESS") for l in (1, 2)},
        "run_failed": sorted(f"{r['benchmark']}/{r['input_id']}" for r in rows if r["status"] == "RUN_FAILED"),
        "not_measured": sorted(f"{r['benchmark']}/{r['input_id']}" for r in rows if r["status"] == "NOT_MEASURED"),
        "roi_not_supported_level3": n(lambda r: r["status"] == "ROI_NOT_SUPPORTED"),
        "run_verification_pass": n(lambda r: r["run_verification"] == "PASS"),
        "unstable": sorted(f"L{r['level']} {r['benchmark']}/{r['input_id']}" for r in rows if r["current"] and not r["current"]["stable"]),
        "adaptive": sorted(f"{r['benchmark']}/{r['input_id']}" for r in rows if r["current"] and r["current"]["adaptive"]),
        "clean_samples_current": sum(len(r["current"]["samples"]) for r in rows if r["current"]),
        "records": len(recs),
        "record_verdicts": {v: sum(1 for r in recs if r["_verdict"] == v) for v in sorted({r["_verdict"] for r in recs})},
        "correctness": {f"level{l}": {str(k): n(lambda r, l=l, k=k: r["level"] == l and r.get("correctness") == k)
                                      for k in ("PASS", "INCOMPLETE", "FAIL", None)} for l in L},
        "records_of_unregistered_names": list(orphans),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--json", help="write the counts here")
    a = ap.parse_args(argv)
    rows, recs, _meta, orphans = current_view(a.roots)
    c = counts(rows, recs, orphans)
    if a.json:
        json.dump(c, open(a.json, "w"), indent=1)
    print(json.dumps(c, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
