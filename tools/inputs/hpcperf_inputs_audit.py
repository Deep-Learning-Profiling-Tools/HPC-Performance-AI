#!/usr/bin/env python3
"""Coverage audit of the inputs registries (one row per active benchmark), generated from the
inputs.yaml files and, optionally, a measurements directory (measure output:
<dir>/level<N>-<benchmark>/<input_id>/measurement.json) and a YAML of measurement blockers
({benchmark: text}). Nothing is measured or modified here.

  tools/inputs/hpcperf_inputs_audit.py [--root REPO] [--measurements DIR] [--blockers FILE]
                                       [--md audit.md] [--json audit.json]
"""
import argparse, glob, json, os, statistics, sys
import yaml

LEVELS = {"level1": 1, "level2": 2, "level3": 3}
UPSTREAM = ("upstream-file", "upstream-parameterized")


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hpcperf_inputs as hi   # noqa: E402  (same directory; re-derives the status vocabulary of older measurement files)


def _summary(d, bench_dir):
    """The summary of a measurement.json in the current vocabulary: files written by an earlier
    schema (measurement-1: run_ok, no comparison_rules) are re-summarised from their raw run records
    with the tool's own summarize(), and the baseline verdict is re-derived over the INDEPENDENT runs
    (the baseline run is never compared with itself), as make_table.py of the pilot does."""
    s = dict(d.get("summary", {}))
    reps = d.get("measured_runs", len([r for r in d["runs"] if r.get("measured")]))
    try:
        doc = hi.load(bench_dir); inp = hi.get_input(doc, d["input_id"])
    except Exception:
        return s
    if "run_completed" not in s or "comparison_rules" not in s:
        try:
            s2, _ = hi.summarize(doc, inp, d["runs"], reps); s2.update({k: v for k, v in s.items() if k.startswith("baseline_")}); s = s2
        except Exception:
            return s
    if "baseline_verdict" not in s:
        bf = s.get("baseline_file")
        if bf and os.path.exists(os.path.join(os.path.dirname(bf), "baseline.workload-migrated.json")):
            bf = os.path.join(os.path.dirname(bf), "baseline.workload-migrated.json")
        if bf and os.path.exists(bf):
            try:
                b = json.load(open(bf)); base = b.get("from_run")
                others = [r for r in d["runs"] if r.get("measured") and r.get("exit_code") == 0 and r.get("baseline_quantities") and r.get("label") != base]
                cmps = [hi.compare(doc, b["quantities"], r["baseline_quantities"], inp) for r in others]
                s["baseline_verdict"] = "NONE" if not cmps else "FAIL" if not all(c["ok"] for c in cmps) else ("PASS" if all(c["verified"] for c in cmps) else "INCOMPLETE")
            except Exception:
                s["baseline_verdict"] = None
    return s


def load_measurement(mdir, level, bench, iid, bench_dir=None):
    f = os.path.join(mdir, f"level{level}-{bench}", iid, "measurement.json")
    if not os.path.exists(f):
        return None
    inv = hi.invalidation(f)
    if inv:     # kept on disk as evidence, never counted: no timing, no correctness verdict
        return {"run_completed": False, "invalidated": True, "invalidation_reason": inv.get("reason"),
                "invalidation_marker": inv["marker"], "replacement": inv.get("replacement"),
                "timing_status": "INVALIDATED", "timing_ok": False, "comparison_rules": None,
                "baseline_verdict": None, "native_check": None, "needs_validation": [], "measured_runs": None,
                "main_compute_median_s": None, "e2e_median_s": None, "compute_ge_1s": None, "stable": None,
                "dir": os.path.dirname(f)}
    d = json.load(open(f))
    s = _summary(d, bench_dir) if bench_dir else d.get("summary", {})
    return {"run_completed": bool(s.get("run_completed")), "timing_status": s.get("timing_status") or ("NATIVE" if s.get("timing_ok") else "FAILED"),
            "timing_ok": bool(s.get("timing_ok")), "comparison_rules": s.get("comparison_rules"), "baseline_verdict": s.get("baseline_verdict"),
            "native_check": s.get("native_check"), "needs_validation": s.get("needs_validation") or [], "measured_runs": d.get("measured_runs"),
            "main_compute_median_s": (s.get("main_compute_s") or {}).get("median"), "e2e_median_s": (s.get("e2e_s") or {}).get("median"),
            "compute_ge_1s": s.get("compute_ge_1s"), "stable": s.get("stable"), "dir": os.path.dirname(f)}


def audit(root, mdir=None, blockers=None):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "level*", "*", "inputs.yaml"))):
        d = yaml.safe_load(open(f)); lvl = LEVELS[f.split(os.sep)[-3]]; b = d["benchmark"]
        cov = d.get("coverage") or {}
        ins = d["inputs"]; runnable = [i for i in ins if i.get("materialized", True) is not False]
        kinds = [i["source"]["kind"] for i in ins]
        forms = sorted({i.get("input_form", "runtime") for i in ins})
        timing_status = "NEEDS_TIMING_SUPPORT" if d["timing"].get("kind") == "none" else "NATIVE"
        req = [q for q in (d.get("baseline") or {}).get("quantities") or [] if q.get("role", "required") == "required"]
        rec = [q["name"] for q in req if (q.get("compare") or {}).get("rule") == "record"]
        checker = "none" if not (d.get("baseline") or {}).get("quantities") else ("record-only" if req and len(rec) == len(req) else ("rules" if req else "diagnostic-only"))
        meas = {}
        if mdir:
            for i in ins:
                m = load_measurement(mdir, lvl, b, i["id"], os.path.dirname(f))
                if m:
                    meas[i["id"]] = m
        completed = [k for k, m in meas.items() if m["run_completed"]]
        native_measured = [k for k in completed if meas[k]["timing_ok"]]
        cr = {}
        for k in completed:
            cr[meas[k]["comparison_rules"] or "NONE"] = cr.get(meas[k]["comparison_rules"] or "NONE", 0) + 1
        verd = {}
        for k in completed:
            v = meas[k]["baseline_verdict"] or "NONE"; verd[v] = verd.get(v, 0) + 1
        if not mdir:
            corr = f"checker: {checker}" + (f" (record-only: {', '.join(rec)})" if rec else "")
        elif not completed:
            corr = "NOT_MEASURED; checker: " + checker
        else:
            corr = "rules " + " ".join(f"{k}:{v}" for k, v in sorted(cr.items())) + "; verdict " + " ".join(f"{k}:{v}" for k, v in sorted(verd.items()))
            nv = sorted({q for k in completed for q in meas[k]["needs_validation"]})
            if nv:
                corr += "; NEEDS_VALIDATION: " + ", ".join(nv)
        blocker = cov.get("blocker")
        mb = (blockers or {}).get(b)
        rows.append({
            "level": lvl, "benchmark": b, "status": cov.get("status", "UNSET"), "reason": cov.get("reason"),
            "n_inputs": len(ins), "n_runnable": len(runnable), "n_cases": len({i["case"] for i in ins}),
            "n_size_variants": sum(1 for i in ins if i.get("variant") == "size"),
            "n_upstream": sum(1 for k in kinds if k in UPSTREAM), "n_derived": kinds.count("derived"), "n_custom": kinds.count("custom"),
            "input_forms": forms, "selector": d.get("selector"), "timing_status": timing_status,
            "timing_reason": d["timing"].get("reason") if timing_status != "NATIVE" else None,
            "n_measured": len(completed), "n_native_measured": len(native_measured),
            "n_failed_measurements": len([k for k, m in meas.items() if not m["run_completed"] and not m.get("invalidated")]),
            "invalidated_inputs": sorted(k for k, m in meas.items() if m.get("invalidated")),
            "unmeasured_inputs": [i["id"] for i in runnable if i["id"] not in completed] if mdir else [i["id"] for i in runnable],
            "unmaterialized_inputs": [i["id"] for i in ins if i.get("materialized", True) is False],
            "checker": checker, "record_only": rec, "correctness": corr,
            "coverage_blocker": blocker, "measurement_blocker": mb,
            "upstream_inputs_not_added": cov.get("upstream_inputs_not_added") or [],
            "derived_custom": [{"id": i["id"], "kind": i["source"]["kind"], "derivation": (i["source"].get("derivation") or "").strip()} for i in ins if i["source"]["kind"] in ("derived", "custom")],
            "measurements": meas,
        })
    return rows


def totals(rows):
    out = {}
    for lvl in (1, 2, 3):
        rs = [r for r in rows if r["level"] == lvl]
        n = [r["n_inputs"] for r in rs]
        out[f"level{lvl}"] = {"benchmarks": len(rs), "inputs": sum(n), "runnable_inputs": sum(r["n_runnable"] for r in rs),
                              "mean_inputs": round(statistics.mean(n), 2) if n else 0, "median_inputs": statistics.median(n) if n else 0,
                              "status": {s: sum(1 for r in rs if r["status"] == s) for s in ("MULTI_INPUT", "SINGLE_INPUT", "BLOCKED", "UNSET")},
                              "measured_inputs": sum(r["n_measured"] for r in rs), "native_timed_inputs": sum(r["n_native_measured"] for r in rs),
                              "needs_timing_support_benchmarks": sum(1 for r in rs if r["timing_status"] != "NATIVE")}
    n = [r["n_inputs"] for r in rows]
    up = sum(r["n_upstream"] for r in rows); de = sum(r["n_derived"] for r in rows); cu = sum(r["n_custom"] for r in rows)
    cr = {}
    for r in rows:
        for m in r["measurements"].values():
            if m["run_completed"]:
                k = m["comparison_rules"] or "NONE"; cr[k] = cr.get(k, 0) + 1
    out["all"] = {"benchmarks": len(rows), "inputs": sum(n), "runnable_inputs": sum(r["n_runnable"] for r in rows),
                  "mean_inputs": round(statistics.mean(n), 2), "median_inputs": statistics.median(n),
                  "status": {s: sum(1 for r in rows if r["status"] == s) for s in ("MULTI_INPUT", "SINGLE_INPUT", "BLOCKED", "UNSET")},
                  "source_share": {"upstream": up, "derived": de, "custom": cu, "upstream_pct": round(100 * up / sum(n), 1), "derived_pct": round(100 * de / sum(n), 1), "custom_pct": round(100 * cu / sum(n), 1)},
                  "measured_inputs": sum(r["n_measured"] for r in rows), "invalidated_measurements": sum(len(r.get("invalidated_inputs") or []) for r in rows), "not_measured_runnable_inputs": sum(len(r["unmeasured_inputs"]) for r in rows),
                  "unmaterialized_inputs": sum(len(r["unmaterialized_inputs"]) for r in rows),
                  "native_timed_inputs": sum(r["n_native_measured"] for r in rows),
                  "needs_timing_support_benchmarks": sum(1 for r in rows if r["timing_status"] != "NATIVE"),
                  "comparison_rules_of_measured_inputs": cr,
                  "benchmarks_not_measured": [r["benchmark"] for r in rows if r["n_measured"] == 0]}
    return out


def markdown(rows, tot, mdir):
    L = ["# Inputs registry audit (generated by tools/inputs/hpcperf_inputs_audit.py)", ""]
    L.append(f"Measurements: {mdir or 'none given'}. Columns: inputs = registered (runnable), cases = distinct `case`, size = `variant: size` inputs, "
             "sources = upstream / derived / custom, form = runtime|file|compile-time, timing = native timer status of the registry, measured = inputs with a "
             "completed measure run (native-timed), correctness = comparison-rule / verdict counts of the measured inputs (or the registry's checker), "
             "blocker = coverage blocker | measurement blocker.")
    L += ["", "| # | L | benchmark | status | inputs (runnable) | cases | size | up/der/cus | form | selector | timing | measured | correctness | blocker |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in enumerate(rows, 1):
        bl = " | ".join(x for x in (r["coverage_blocker"], r["measurement_blocker"]) if x) or "-"
        L.append(f"| {n} | {r['level']} | {r['benchmark']} | {r['status']} | {r['n_inputs']} ({r['n_runnable']}) | {r['n_cases']} | {r['n_size_variants']} | "
                 f"{r['n_upstream']}/{r['n_derived']}/{r['n_custom']} | {','.join(r['input_forms'])} | {r['selector'] or '-'} | {r['timing_status']} | "
                 f"{r['n_measured']} ({r['n_native_measured']}) | {r['correctness']} | {bl} |")
    L += ["", "## Totals", "", "| scope | benchmarks | inputs (runnable) | mean / median inputs | MULTI / SINGLE / BLOCKED | measured inputs (native-timed) | NEEDS_TIMING_SUPPORT benchmarks |", "|---|---|---|---|---|---|---|"]
    for k in ("level1", "level2", "level3", "all"):
        t = tot[k]; st = t["status"]
        L.append(f"| {k} | {t['benchmarks']} | {t['inputs']} ({t['runnable_inputs']}) | {t['mean_inputs']} / {t['median_inputs']} | {st['MULTI_INPUT']} / {st['SINGLE_INPUT']} / {st['BLOCKED']} | {t['measured_inputs']} ({t['native_timed_inputs']}) | {t['needs_timing_support_benchmarks']} |")
    a = tot["all"]
    L += ["", f"Source share of all {a['inputs']} inputs: upstream {a['source_share']['upstream']} ({a['source_share']['upstream_pct']} %), derived {a['source_share']['derived']} ({a['source_share']['derived_pct']} %), custom {a['source_share']['custom']} ({a['source_share']['custom_pct']} %). "
          f"Runnable inputs not measured: {a['not_measured_runnable_inputs']} (incl. {a['invalidated_measurements']} whose measurement is invalidated -- kept on disk, not counted); registered but unmaterialized: {a['unmaterialized_inputs']}. Comparison rules of the measured inputs: {a['comparison_rules_of_measured_inputs']}. "
          f"Benchmarks without any measured input: {', '.join(a['benchmarks_not_measured']) or 'none'}."]
    L += ["", "## SINGLE_INPUT benchmarks (why)", ""]
    for r in rows:
        if r["status"] == "SINGLE_INPUT":
            L.append(f"- **{r['benchmark']}** (L{r['level']}): {r['reason']}" + (f" -- unmaterialized: {', '.join(r['unmaterialized_inputs'])}" if r["unmaterialized_inputs"] else ""))
    L += ["", "## BLOCKED benchmarks (upstream input, why not, what is needed)", ""]
    for r in rows:
        if r["status"] == "BLOCKED":
            L.append(f"- **{r['benchmark']}** (L{r['level']}): {r['reason']} -- blocker: {r['coverage_blocker']} -- not added: {'; '.join(r['upstream_inputs_not_added']) or '-'}")
    L += ["", "## MULTI_INPUT benchmarks with upstream inputs not added (informational)", ""]
    for r in rows:
        if r["status"] == "MULTI_INPUT" and r["upstream_inputs_not_added"]:
            L.append(f"- **{r['benchmark']}** (L{r['level']}): {'; '.join(r['upstream_inputs_not_added'])}" + (f" -- why: {r['coverage_blocker']}" if r["coverage_blocker"] else ""))
    L += ["", "## Derived / custom inputs (why upstream was insufficient, what changed)", ""]
    for r in rows:
        for dc in r["derived_custom"]:
            L.append(f"- {r['benchmark']} `{dc['id']}` ({dc['kind']}): {dc['derivation']}")
    L += ["", "## Timing blockers (NEEDS_TIMING_SUPPORT)", ""]
    for r in rows:
        if r["timing_status"] != "NATIVE":
            L.append(f"- {r['benchmark']} (L{r['level']}): {r['timing_reason']}")
    L += ["", "## Runnable inputs without a completed measurement", ""]
    for r in rows:
        if r["unmeasured_inputs"]:
            inv = set(r.get("invalidated_inputs") or [])
            L.append(f"- {r['benchmark']} (L{r['level']}): {', '.join(i + (' (INVALIDATED)' if i in inv else '') for i in r['unmeasured_inputs'])}" + (f" -- {r['measurement_blocker']}" if r["measurement_blocker"] else ""))
    L += ["", "## Correctness blockers (record-only required quantities = NEEDS_VALIDATION, no checker)", ""]
    for r in rows:
        if r["checker"] in ("none", "record-only", "diagnostic-only"):
            L.append(f"- {r['benchmark']} (L{r['level']}): checker {r['checker']}" + (f" ({', '.join(r['record_only'])})" if r["record_only"] else ""))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    ap.add_argument("--measurements"); ap.add_argument("--blockers"); ap.add_argument("--md"); ap.add_argument("--json")
    a = ap.parse_args()
    blockers = yaml.safe_load(open(a.blockers)) if a.blockers else {}
    rows = audit(a.root, a.measurements, blockers); tot = totals(rows)
    md = markdown(rows, tot, a.measurements)
    if a.md:
        open(a.md, "w").write(md)
    if a.json:
        json.dump({"totals": tot, "rows": rows}, open(a.json, "w"), indent=1)
    if not a.md and not a.json:
        sys.stdout.write(md)
    t = tot["all"]
    print(f"{t['benchmarks']} benchmarks, {t['inputs']} inputs ({t['runnable_inputs']} runnable), status {t['status']}, measured {t['measured_inputs']}", file=sys.stderr)


if __name__ == "__main__":
    main()
