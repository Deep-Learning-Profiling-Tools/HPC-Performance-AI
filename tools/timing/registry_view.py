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

Records of one input are POOLED into one measurement only when an explicit association says they are
one measurement: a measurement group in <results root>/measurement_groups.json (schema
hpcperf-timing-measurement-groups-1) naming a base record and its extension records (the adaptive +2
clean runs of a Level 2 input), written by whoever ran the extension, with its evidence. The group is
used only when every member is found among the input's accepted records of THAT results root and all
members are the same measurement configuration (platform, workload identity, executable sha256, source
commit, protocol apart from the number of clean runs: warm-up runs, profiled runs, collector); otherwise it
is rejected (reported) and its records stay separate. Equal configuration alone never pools records:
two independent runs of the same configuration -- in one campaign or in two -- stay two measurements. A
record provided twice (same level, application, input and run id) is counted once. The current result
of an input is its newest measurement whose verdict is PASS or INSUFFICIENT (timing valid, verification
incomplete -- reported as such).

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
GROUPS = "measurement_groups.json"  # optional, per results root: explicit measurement groups (see measurements)
CONFIG_FIELDS = ("platform", "workload identity", "executable sha256", "source commit", "warm-up runs", "profiled runs", "collector")


def workload_key(workload):
    return hashlib.sha256(json.dumps(workload or {}, sort_keys=True).encode()).hexdigest()


def load_records(roots):
    """(records, duplicates): every timing record under the roots, each (level, app, case, run_id) once."""
    recs, seen, dups = [], {}, []
    for root in roots:
        rroot = os.path.realpath(root)
        for p in sorted(glob.glob(os.path.join(root, "level[0-9]", "*", "*", "*.json"))):
            try:
                r = json.load(open(p))
            except (OSError, ValueError):
                continue
            if r.get("schema") != SCHEMA or r.get("level") not in (1, 2):
                continue
            key = (r["level"], r["app"], r["case"], r["run_id"])
            if key in seen:
                dups.append({"record": p, "first": seen[key]})
                continue
            seen[key] = p
            r["_path"], r["_root"] = p, rroot
            recs.append(r)
    return recs, dups


def load_groups(roots):
    out = []
    for root in roots:
        p = os.path.join(root, GROUPS)
        if os.path.isfile(p):
            for g in json.load(open(p)).get("groups") or []:
                out.append(dict(g, _root=os.path.realpath(root)))
    return out


def classify(recs, repo=REPO, rules=None):
    """Attach _verdict (PASS/SUPERSEDED/INVALIDATED/NOT_RUN/FAIL/INSUFFICIENT) and _problems to each record."""
    rules = rules or V.load_rules(V.DEFAULT_RULES)
    for r in recs:
        raw = os.path.join(repo, (r.get("provenance") or {}).get("raw_dir") or "")
        if os.path.isfile(os.path.join(raw, "INVALIDATED.json")):
            r["_verdict"], r["_problems"] = "INVALIDATED", [json.load(open(os.path.join(raw, "INVALIDATED.json"))).get("reason", "")]
            continue
        v = V.verify_record(r["_path"], repo, rules)
        r["_verdict"], r["_problems"], r["_file_identity"] = v["verdict"], v["problems"] + v["gaps"], v.get("file_identity")
    return recs


def _config(r):
    m = r.get("measurement") or {}
    p = m.get("protocol") or {}
    reg = r.get("registry") or {}
    return (r.get("platform") or "-", reg.get("identity_sha256") or "-", (r.get("inputs") or {}).get("exe_sha256") or "-",
            (r.get("provenance") or {}).get("git_commit") or "-", p.get("warmup_runs"), p.get("profiled_runs"),
            (m.get("collector") or {}).get("name"))


def _make_set(rs, group=None):
    s = [x for r in rs for x in (r.get("roi") or {}).get("runs_s") or []]
    med = statistics.median(s)
    verdicts = {r["_verdict"] for r in rs}
    m = {"platform": _config(rs[0])[0], "records": rs, "run_ids": [r["run_id"] for r in rs], "samples": s,
         "median": med, "min": min(s), "max": max(s),
         "spread": (max(s) - min(s)) / med if med else None,
         "cv": statistics.stdev(s) / med if (med and len(s) > 1) else None,
         "verdict": "SUPERSEDED" if "SUPERSEDED" in verdicts else "INSUFFICIENT" if "INSUFFICIENT" in verdicts else "PASS",
         "file_identity": sorted({r.get("_file_identity") or "recorded" for r in rs}),
         "recorded_workload_key": workload_key(((rs[0].get("registry") or {}).get("identity") or {}).get("workload")),
         "utc_first": min(r.get("utc") or "" for r in rs), "utc_last": max(r.get("utc") or "" for r in rs),
         "git_commit": _config(rs[0])[3], "exe_sha256": _config(rs[0])[2],
         "protocol": [(r.get("measurement") or {}).get("protocol") for r in rs],
         "group": None if group is None else {k: group.get(k) for k in ("id", "reason", "evidence")},
         "root": rs[0]["_root"]}
    m["stable"] = m["spread"] is not None and m["spread"] <= STABLE_SPREAD
    m["adaptive"] = group is not None
    return m


def group_structure_problem(g):
    """Why a measurement group's member list is malformed (None when it is well formed). A malformed group
    is rejected as a whole -- never silently repaired -- so no sample can be counted twice."""
    base, ext = g.get("base_run_id"), g.get("extension_run_ids")
    if not isinstance(base, str) or not base.strip():
        return "missing or empty base_run_id"
    if not isinstance(ext, list) or not ext:
        return "extension_run_ids missing, empty or not a list"
    if any(not isinstance(i, str) or not i.strip() for i in ext):
        return "an extension run id is empty or not a string"
    if len(set(ext)) != len(ext):
        return "an extension run id is listed more than once"
    if base in ext:
        return "the base run id is also listed as an extension"
    return None


def measurements(recs, groups=()):
    """(measurement sets, problems) of ONE input: explicit groups pooled (see the module docstring), every
    other accepted ok record its own measurement."""
    elig = [r for r in sorted(recs, key=lambda r: r["run_id"])
            if r["status"] == "ok" and r["_verdict"] in ("PASS", "INSUFFICIENT", "SUPERSEDED") and (r.get("roi") or {}).get("runs_s")]
    by_run = {(r["_root"], r["run_id"]): r for r in elig}
    sets, problems, used = [], [], set()
    for g in groups:
        bad = group_structure_problem(g)
        if bad:
            problems.append(f"group {g.get('id')}: {bad} -- not pooled")
            continue
        ids = [g["base_run_id"]] + list(g["extension_run_ids"])
        members = [by_run.get((g["_root"], i)) for i in ids]
        missing = [i for i, m in zip(ids, members) if m is None]
        if missing:
            problems.append(f"group {g.get('id')}: record(s) {', '.join(missing)} not among this input's accepted records "
                            f"in the same results directory -- not pooled")
            continue
        cfgs = [_config(m) for m in members]
        diff = [CONFIG_FIELDS[i] for i in range(len(CONFIG_FIELDS)) if len({c[i] for c in cfgs}) > 1]
        if diff:
            problems.append(f"group {g.get('id')}: linked records differ in {', '.join(diff)} -- not pooled")
            continue
        if len({m["_verdict"] == "SUPERSEDED" for m in members}) > 1:
            problems.append(f"group {g.get('id')}: linked records are of different input definitions -- not pooled")
            continue
        if any((g["_root"], i) in used for i in ids):
            problems.append(f"group {g.get('id')}: a record already belongs to another group -- not pooled")
            continue
        sets.append(_make_set(members, g))
        used.update((g["_root"], i) for i in ids)
    for r in elig:
        if (r["_root"], r["run_id"]) not in used:
            sets.append(_make_set([r]))
    sets.sort(key=lambda m: m["run_ids"][-1])
    return sets, problems


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
    recs, dups = load_records(roots)
    recs = classify(recs, repo)
    groups = load_groups(roots)
    ann, meta = load_annotations(roots)
    meta["duplicate_records_ignored"] = dups
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
        sets, link_problems = measurements(mine, [g for g in groups if (g.get("level"), g.get("app"), g.get("case")) == key])
        for m in sets:      # the definition a measurement belongs to, as the verifier decided it
            m["workload_key"] = m["recorded_workload_key"] if m["verdict"] == "SUPERSEDED" else inp["workload_key"]
        cur = [m for m in sets if m["verdict"] in ("PASS", "INSUFFICIENT")]
        row["history_sets"] = sets
        row["link_problems"] = link_problems
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
        row["run_verification"] = row["current"]["verdict"] if row["current"] else (mine[-1]["_verdict"] if mine else None)
        row["file_identity"] = row["current"]["file_identity"] if row["current"] else None
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
        "run_verification_insufficient": sorted(f"{r['benchmark']}/{r['input_id']}" for r in rows if r["current"] and r["run_verification"] == "INSUFFICIENT"),
        "file_identity_from_supplement": sorted(f"{r['benchmark']}/{r['input_id']}" for r in rows if "supplement" in (r.get("file_identity") or [])),
        "pooling_rejected": sorted(f"{r['benchmark']}/{r['input_id']}: {p}" for r in rows for p in r.get("link_problems") or []),
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
