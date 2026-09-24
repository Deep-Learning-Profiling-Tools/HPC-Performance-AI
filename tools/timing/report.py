#!/usr/bin/env python3
"""Render the Level 1 / Level 2 timing results as an interactive web page (+ Markdown twin).

    tools/timing/report.py                               # results/timing  ->  results/timing/report/
    tools/timing/report.py --publish                     # results/timing  ->  docs/timing/ (tracked)
    tools/timing/report.py --results-root DIR [--results-root DIR2 ...] [--history-page OLD.html] [--out DIR | --publish]

Two kinds of results are rendered, both from the same records:

* Registered inputs (records made with measure_level<N>.sh --registry): the inputs come from the
  registry (level<N>/*/inputs.yaml), so an input that failed or was never measured is still listed.
  The current result of each input and its history are chosen by tools/timing/registry_view.py -- the
  same rules as every other summary: only records the run verifier accepts for the input's CURRENT
  definition count; INVALIDATED records (the run got another workload) and SUPERSEDED records (an
  older definition of the input) are history; an adaptive extension (+2 clean runs of the same
  configuration) is pooled with its 3 runs into one 5-sample result; "vs previous" is only computed
  between measurements of the same workload. Scientific correctness and blocker notes come from
  annotations.json next to the records (evidence from outside the timing runs), never from a ROI run.
* Case-table results (measure_level<N>.sh without --registry): the cases of tools/timing/cases/ against
  the platforms, the latest run of each (case, platform) -- the original view.

Several --results-root directories are read as ONE campaign (e.g. the phases of one campaign kept in
separate directories). --history-page embeds an earlier published index.html as a separate, clearly
labelled historical campaign, verbatim: its numbers are never mixed into the current view.

Fields that need a profiler (device busy, host gap, kernel and operation counts, runtime-API calls,
profiler inflation) are null when the run had no collector; they are never filled from another run.
Spread is (max - min) / median of the clean-run samples (the stability criterion, stable <= 10 %); the
coefficient of variation (sample stddev / median) is shown separately and labelled CV.

summarize.py calls write() for its own results root after every measurement. Publishing to docs/timing/
is the deliberate step: raw evidence and the JSON/CSV records stay out of git; the rendered page and its
Markdown twin are the snapshot that may be committed. Standard library only. The output depends only on
the records, the registry and the annotations (no wall-clock time in it); absolute paths of this checkout
are written as {REPO}, the results directories as {RESULTS}.
"""

import argparse
import glob
import html
import json
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cases as case_tables  # noqa: E402
import registry_view as RV  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
HOME = os.path.expanduser("~")
SCHEMA = "hpcperf-timing-2"
DEFAULT_RESULTS = os.path.join(REPO, "results", "timing")
PUBLISH_DIR = os.path.join(REPO, "docs", "timing")
ASSETS = os.path.join(HERE, "report_assets")
PLATFORMS = os.path.join(HERE, "platforms")
TOP_OPS = 15
LEVEL_NAMES = {1: "Level 1", 2: "Level 2"}


# ----------------------------------------------------------------- inputs

SCRUB_ROOTS = []     # results directories being rendered (set by write()): written as {RESULTS}
SCRUB_HOSTS = []     # host names found in the records (set by build_bundle()): written as {HOST}


def scrub(obj):
    """Absolute paths of this checkout, of the results and of the home directory never leave the machine."""
    if isinstance(obj, str):
        s = obj
        for root in SCRUB_ROOTS:
            s = s.replace(root, "{RESULTS}")
        s = s.replace(REPO, "{REPO}")
        for h in SCRUB_HOSTS:
            s = s.replace(h, "{HOST}")
        return s.replace(HOME, "~") if HOME and HOME != "/" else s
    if isinstance(obj, list):
        return [scrub(x) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    return obj


def load(results_root):
    recs = []
    roots = results_root if isinstance(results_root, (list, tuple)) else [results_root]
    for p in sorted(q for root in roots for q in glob.glob(os.path.join(root, "level[0-9]", "*", "*", "*.json"))):
        try:
            with open(p) as f:
                r = json.load(f)
        except (OSError, ValueError):
            continue
        if r.get("schema") == SCHEMA and r.get("level") in LEVEL_NAMES:
            recs.append(r)
    return recs


def defined_inputs():
    """{level: {app: {case: {"env", "args", "gpus"}}}} from the case tables, and {app: suite}."""
    out, suites = {1: {}, 2: {}}, {}
    try:
        rows, apps = case_tables.level1_rows(None, "CUDA")
        suites = {a: r["suite"] for a, r in apps.items()}
        for a in apps:
            out[1].setdefault(a, {})
        for r in rows:
            argv = shlex.split(r["argv"])
            args = [os.path.basename(argv[0])] + argv[1:] if argv else []
            out[1][r["app"]][r["case"]] = {"env": r["env"], "args": " ".join(shlex.quote(a) for a in args),
                                           "gpus": ""}
        rows, apps = case_tables.level2_rows("CUDA")
        for a in apps:
            out[2].setdefault(a, {})
        for r in rows:
            argv = shlex.split(r["argv"])
            out[2][r["app"]][r["case"]] = {"env": r["env"], "args": " ".join(shlex.quote(a) for a in argv[3:]),
                                           "gpus": r["gpus"]}
    except case_tables.CaseError:
        pass
    return scrub(out), suites


def platform_info(recs):
    """Every platform with a measurement or a conformance record, with a short description."""
    ids = {r.get("platform") for r in recs if r.get("platform")}
    for p in glob.glob(os.path.join(PLATFORMS, "*.json")):
        ids.add(os.path.basename(p)[:-5])
    out = []
    for pid in sorted(ids):
        conf = None
        try:
            with open(os.path.join(PLATFORMS, f"{pid}.json")) as f:
                conf = (json.load(f) or {}).get("conformance")
        except (OSError, ValueError):
            pass
        dev = next((r["platform_info"]["device"] for r in sorted(recs, key=lambda r: r["run_id"], reverse=True)
                    if r.get("platform") == pid and r.get("platform_info")), {}) or {}
        rt = dev.get("runtime") or {}
        out.append({
            "id": pid,
            "label": " ".join(x for x in (dev.get("vendor", "").upper() if dev.get("vendor") else "",
                                          dev.get("product") or "") if x) or pid,
            "runtime": " ".join(x for x in (rt.get("name") or "", rt.get("version") or "") if x),
            "device": {k: dev.get(k) for k in ("vendor", "product", "arch", "count_visible", "memory_total_mib",
                                              "core_clock_max_mhz", "mem_clock_max_mhz", "power_limit_w",
                                              "driver_version")},
            "conformance": None if not conf else {
                "status": conf.get("status"), "date": (conf.get("date_utc") or "")[:10],
                "collector": conf.get("collector"),
                "checks": f"{sum(1 for c in conf.get('checks') or [] if c.get('ok'))}/{len(conf.get('checks') or [])}"},
        })
    return out


def compact(rec):
    """What the detail view shows of one run -- no environment, host names or absolute paths."""
    R = rec.get("roi") or {}
    ctx = rec.get("context") or {}
    meas = rec.get("measurement") or {}
    inp = rec.get("inputs") or {}
    procs = inp.get("processes") or []
    dev = (rec.get("platform_info") or {}).get("device") or {}
    host = (rec.get("platform_info") or {}).get("host") or {}
    return scrub({
        "run_id": rec["run_id"], "utc": rec.get("utc"), "status": rec["status"],
        "roi": {k: R.get(k) for k in ("wall_s", "runs_s", "wall_s_min", "wall_s_max", "wall_s_stddev", "entries",
                                      "excluded_s", "processes", "imbalance_s", "profiled_wall_s",
                                      "profiler_inflation")},
        "device": rec.get("device"),
        "runtime_api": rec.get("runtime_api"),
        "ops": (rec.get("ops") or [])[:TOP_OPS], "ops_total": len(rec.get("ops") or []),
        "context": {"process_wall_s": ctx.get("process_wall_s"), "pre_roi_s": ctx.get("pre_roi_s"),
                    "post_roi_s": ctx.get("post_roi_s"),
                    "whole": {k: (ctx.get("whole_process") or {}).get(k) for k in ("busy_s", "compute_ops")}},
        "fom": rec.get("fom"), "app_timer": rec.get("app_timer"),
        "audit_ok": (rec.get("launcher") or {}).get("audit_ok"),     # the raw audit line names the node: not copied
        "measurement": {"protocol": meas.get("protocol"), "collector": meas.get("collector"),
                        "skip_verify": meas.get("skip_verify"), "verify_vs_roi": meas.get("verify_vs_roi"),
                        "roi_excludes": meas.get("roi_excludes"), "backend": meas.get("backend"),
                        "gpus": meas.get("gpus")},
        "inputs": {"declared_env": inp.get("declared_env"),
                   "argv": (procs[0].get("argv") if procs else None) or inp.get("declared_argv"),
                   "processes": len(procs)},
        "device_info": {k: dev.get(k) for k in ("product", "arch", "count_visible", "driver_version",
                                                "core_clock_mhz", "mem_clock_mhz", "temperature_c")},
        "host_cpu": host.get("cpu_model"),
        "git_commit": ((rec.get("provenance") or {}).get("git_commit") or "")[:10],
        "caveats": rec.get("caveats") or [],
    })


def build_data(recs):
    """The interactive page's data: levels -> apps -> inputs x platforms -> measurement or null."""
    inputs, suites = defined_inputs()
    plats = platform_info(recs)
    groups = {}
    for r in recs:
        groups.setdefault((r["level"], r["app"], r["case"], r.get("platform") or "-"), []).append(r)
    for (lvl, app, case, _plat) in groups:
        inputs[lvl].setdefault(app, {}).setdefault(case, {"env": "", "args": "", "gpus": "", "undeclared": True})
    levels = {}
    for lvl in sorted(inputs):
        apps = []
        for app in sorted(inputs[lvl]):
            cases = []
            for case in sorted(inputs[lvl][app], key=lambda c: (c != "default", c)):
                cells = {}
                for p in plats:
                    runs = sorted(groups.get((lvl, app, case, p["id"]), []), key=lambda r: r["run_id"])
                    if not runs:
                        cells[p["id"]] = None
                        continue
                    ok = [r for r in runs if r["status"] == "ok"]
                    shown = ok[-1] if ok else runs[-1]
                    prev = ok[-2] if len(ok) > 1 else None
                    cells[p["id"]] = {
                        "history": [{"run_id": r["run_id"], "utc": r.get("utc"), "status": r["status"],
                                     "roi_s": (r.get("roi") or {}).get("wall_s")} for r in runs],
                        "latest_status": runs[-1]["status"],
                        "prev": None if not prev else {"run_id": prev["run_id"], "roi_s": prev["roi"]["wall_s"]},
                        "run": compact(shown),
                    }
                cases.append({"case": case, "input": inputs[lvl][app][case], "cells": cells})
            apps.append({"app": app, "suite": suites.get(app, "") if lvl == 1 else "", "cases": cases})
        levels[str(lvl)] = apps
    latest = max((r.get("utc") or "" for r in recs), default="")
    return {"kind": "cases", "generated_from": latest, "records": len(recs), "platforms": plats, "levels": levels}


# ----------------------------------------------------------------- registered inputs (registry_view)

def _set_summary(m, input_key):
    return {"run_ids": m["run_ids"], "verdict": m["verdict"], "n": len(m["samples"]), "median": m["median"],
            "spread": m["spread"], "stable": m["stable"], "git_commit": (m["git_commit"] or "")[:10],
            "utc_last": m["utc_last"], "platform": m["platform"], "current_definition": m["workload_key"] == input_key}


def _measurement(m):
    """One pooled measurement as the detail view shows it: the newest record for context, the pooled samples
    for the ROI statistics."""
    run = compact(m["records"][-1])
    s = m["samples"]
    run["roi"].update({"wall_s": m["median"], "runs_s": s, "wall_s_min": m["min"], "wall_s_max": m["max"],
                       "wall_s_stddev": m["cv"] * m["median"] if m["cv"] is not None else None})
    run["set"] = scrub({"run_ids": m["run_ids"], "n": len(s), "spread": m["spread"], "cv": m["cv"], "stable": m["stable"],
                        "adaptive": m["adaptive"], "utc_first": m["utc_first"], "utc_last": m["utc_last"],
                        "git_commit": (m["git_commit"] or "")[:10], "exe_sha256": (m["exe_sha256"] or "")[:16],
                        "per_record": [{"run_id": r["run_id"], "runs_s": (r.get("roi") or {}).get("runs_s") or [],
                                        "protocol": (r.get("measurement") or {}).get("protocol")} for r in m["records"]]})
    return run


def build_registry(roots):
    """The registered-input campaign: every registry input of Level 1/2, its current measurement per platform,
    its status, verification, correctness evidence and history (tools/timing/registry_view.py)."""
    rows, recs, meta, orphans = RV.current_view(roots, REPO)
    plats = platform_info(recs)
    _, suites = defined_inputs()
    levels = {}
    for lvl in (1, 2):
        apps = {}
        for row in (r for r in rows if r["level"] == lvl):
            sets = row["history_sets"]
            hist, last = [], {}
            for m in sets:
                h = _set_summary(m, row["workload_key"])
                prev = last.get((m["platform"], m["workload_key"]))
                h["vs_previous"] = None if prev is None else (m["median"] - prev) / prev
                last[(m["platform"], m["workload_key"])] = m["median"]
                hist.append(h)
            cells = {}
            for p in plats:
                m = row["current_by_platform"].get(p["id"])
                cells[p["id"]] = _measurement(m) if m else None
            apps.setdefault(row["benchmark"], []).append(scrub({
                "input_id": row["input_id"], "case": row["case"], "variant": row["variant"], "source_kind": row["source_kind"],
                "input_form": row["input_form"], "params": row["params"], "args": row["args"], "env": row["env"],
                "status": row["status"], "run_verification": row["run_verification"],
                "correctness": row.get("correctness"), "correctness_basis": row.get("correctness_basis"),
                "blocker": row.get("blocker"), "cells": cells, "sets": hist,
                "attempts": [{k: a[k] for k in ("run_id", "utc", "status", "verdict", "roi_s", "clean_runs", "git_commit")}
                             | {"current_definition": a["workload_key"] == row["workload_key"], "problems": a["problems"][:3]}
                             for a in row["attempts"]]}))
        levels[str(lvl)] = [{"app": a, "suite": suites.get(a, "") if lvl == 1 else "", "inputs": apps[a]} for a in sorted(apps)]
    utcs = sorted(r.get("utc") or "" for r in recs if r.get("utc"))
    c = RV.counts(rows, recs, orphans)
    return scrub({"kind": "registry", "campaign": meta["campaign"], "notes": meta["notes"],
                  "measured_from": utcs[0] if utcs else "", "generated_from": utcs[-1] if utcs else "",
                  "records": len(recs), "platforms": plats, "levels": levels, "counts": c,
                  "level3_inputs": sum(1 for r in rows if r["level"] == 3)})


def load_history_page(path):
    """An earlier published index.html, embedded verbatim as a historical campaign."""
    import re
    text = open(path).read()
    m = re.search(r'<script type="application/json" id="timing-data">(.*?)</script>', text, re.S)
    data = json.loads(m.group(1).replace("<\\/", "</"))
    if "campaigns" in data:              # a page of this version: take its current campaign
        data = data["campaigns"][0]
    data.setdefault("kind", "cases")
    data["historical"] = True
    return data


# ----------------------------------------------------------------- HTML

def campaign_label(c):
    if c.get("kind") == "registry":
        return (c.get("campaign") or {}).get("title") or "Registered inputs"
    return ("Earlier snapshot" if c.get("historical") else "Case tables") + \
        f" · {c.get('records', 0)} records as of {c.get('generated_from') or '-'}"


def render_html(bundle):
    css = open(os.path.join(ASSETS, "report.css")).read()
    js = open(os.path.join(ASSETS, "report.js")).read()
    # JSON inside <script>: "</" would end the element, so it is written as "<\/" (still valid JSON)
    payload = json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    data = bundle["campaigns"][0]
    n_apps = {k: len(v) for k, v in data["levels"].items()}
    as_of = html.escape(data["generated_from"] or "no measurements yet")
    ctabs = "".join(
        f'<button type="button" role="tab" data-campaign="{i}" aria-selected="{str(i == 0).lower()}">'
        f'{html.escape(campaign_label(c))}</button>' for i, c in enumerate(bundle["campaigns"]))
    campaign_nav = (f'<nav class="tabs campaigns" role="tablist" aria-label="Campaign">{ctabs}</nav>\n'
                    if len(bundle["campaigns"]) > 1 else "")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HPC-Performance-AI Timing</title>
<meta name="description" content="Region-of-interest timing of the HPC-Performance-AI Level 1 and Level 2 suites">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
{css}</style>
</head>
<body>
<div class="wrap">
<header class="top">
  <div class="eyebrow">tools/timing &middot; latest measurement {as_of}</div>
  <h1>Timing results</h1>
  <p class="sub">Region-of-interest (ROI) timing of the Level&nbsp;1 benchmarks and the Level&nbsp;2 mini-applications:
  the computation between the markers in each source, without start-up, set-up, warm-up and verification. Choose an
  application, then an input and a platform.</p>
</header>
<noscript><p class="lede">This page needs JavaScript. README.md next to it has the same results as plain tables.</p></noscript>
{campaign_nav}<nav class="tabs levels" role="tablist" aria-label="Level">
  <button type="button" role="tab" id="tab-1" data-level="1" aria-selected="true">Level 1 <span>{n_apps.get("1", 0)} benchmarks</span></button>
  <button type="button" role="tab" id="tab-2" data-level="2" aria-selected="false">Level 2 <span>{n_apps.get("2", 0)} applications</span></button>
</nav>
<div class="board">
  <aside class="apps">
    <label class="find" for="app-filter">Find</label>
    <input id="app-filter" type="search" placeholder="application name" autocomplete="off">
    <ul id="app-list" class="applist" aria-label="Applications"></ul>
  </aside>
  <main id="main" class="main" aria-live="polite"></main>
</div>
</div>
<script type="application/json" id="timing-data">{payload}</script>
<script>
{js}</script>
</body>
</html>
"""


# ----------------------------------------------------------------- Markdown twin

def _time_parts(s):
    a = abs(s)
    if a < 1e-3:
        return f"{s * 1e6:.0f}", "us"
    if a < 1:
        v = s * 1e3
        return (f"{v:.3g}" if abs(v) < 100 else f"{v:.0f}"), "ms"
    return (f"{s:.3g}" if a < 100 else f"{s:.1f}"), "s"


def fmt_plain(s):
    if s is None:
        return "null"
    n, u = _time_parts(s)
    return f"{n} {u}"


def md_cell(s):
    return str(s).replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;").replace("\n", " ")


def md_pct(x, digits=0, cap=False):
    return "null" if x is None else f"{100 * (min(x, 1.0) if cap else x):.{digits}f}%"


def fom_plain(fom):
    fom = fom or {}
    v = fom.get("value")
    if v is None:
        return "none printed" if fom.get("status") == "none" else str(fom.get("status") or "null")
    s = f"{v:.4g}" if (abs(v) >= 1e5 or abs(v) < 1e-2) else f"{v:,.1f}"
    return f"{s} {fom.get('unit') or ''}".strip()


def md_status_line(c):
    k = c["counts"]
    ok = k["roi_success"]
    reg = k["registered_inputs"]
    return (f"{reg['level1']} + {reg['level2']} registered Level 1 / Level 2 inputs: ROI timing SUCCESS for "
            f"{ok['level1']} + {ok['level2']}, run failed for {len(k['run_failed'])}, not measured {len(k['not_measured'])}; "
            f"run verification PASS for {k['run_verification_pass']}; UNSTABLE {len(k['unstable'])}. "
            f"Level 3: {reg['level3']} registered inputs without ROI support (earlier native timing only).")


def render_md_registry(c):
    camp = c.get("campaign") or {}
    k = c["counts"]
    out = [f"## {camp.get('title') or 'Registered inputs'}", "",
           f"Measured {c['measured_from'] or '-'} .. {c['generated_from'] or '-'} ({c['records']} records). " + md_status_line(c), ""]
    for lvl in ("1", "2"):
        if camp.get("protocol", {}).get(f"level{lvl}"):
            out.append(f"- Level {lvl} protocol: {camp['protocol'][f'level{lvl}']}")
    for key in ("platform_note", "not_collected", "level3"):
        if camp.get(key):
            out.append(f"- {camp[key]}")
    out += ["- Spread = (max - min) / median of all clean-run samples of the measurement (stable when <= 10 %); "
            "CV = sample stddev / median. An adaptive extension (+2 runs of the same configuration) is pooled "
            "with its 3 runs.",
            "- Three separate results per input: ROI timing (SUCCESS / RUN_FAILED / NOT_MEASURED), run verification "
            "(did the run get the registered input: tools/timing/verify_registry_runs.py), scientific correctness "
            "(evidence from outside the timing runs; its basis is given per input in index.html).", ""]
    corr = k["correctness"]
    out += ["| level | correctness PASS | INCOMPLETE | FAIL | none |", "|---|--:|--:|--:|--:|"]
    for lvl in ("1", "2", "3"):
        cc = corr[f"level{lvl}"]
        out.append(f"| {lvl} | {cc['PASS']} | {cc['INCOMPLETE']} | {cc['FAIL']} | {cc['None']} |")
    out.append("")
    for note in c.get("notes") or []:
        out.append(f"- {md_cell(note)}")
    out.append("")
    for lvl in ("1", "2"):
        out += [f"### {LEVEL_NAMES[int(lvl)]}", ""]
        head = ["application", "input", "status", "platform", "ROI median", "samples", "spread", "CV", "stable",
                "run verification", "correctness", "source"]
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "|".join("---" if i < 4 else "--:" if i < 8 else "---" for i in range(len(head))) + "|")
        for a in c["levels"].get(lvl, []):
            for i in a["inputs"]:
                cells = [(pid, run) for pid, run in sorted(i["cells"].items()) if run]
                if not cells:
                    cells = [("-", None)]
                for pid, run in cells:
                    st = run["set"] if run else None
                    out.append("| " + " | ".join([
                        md_cell(a["app"]), md_cell(i["input_id"]), i["status"], md_cell(pid),
                        fmt_plain(run["roi"]["wall_s"]) if run else "-",
                        str(st["n"]) + (" (3+2 adaptive)" if st["adaptive"] else "") if st else "-",
                        md_pct(st["spread"], 1) if st else "-", md_pct(st["cv"], 1) if st else "-",
                        ("yes" if st["stable"] else "UNSTABLE") if st else "-",
                        str(i["run_verification"] or "-"), str(i["correctness"] or "none"),
                        st["git_commit"] if st else "-"]) + " |")
        out.append("")
    fails = [(a["app"], i) for lvl in ("1", "2") for a in c["levels"].get(lvl, []) for i in a["inputs"] if i["status"] != "SUCCESS"]
    if fails:
        out += ["### Not measured successfully", ""]
        for app, i in fails:
            last = i["attempts"][-1] if i["attempts"] else None
            out.append(f"- **{md_cell(app)} / {md_cell(i['input_id'])}**: {i['status']}" +
                       (f", last attempt {last['run_id']} status {last['status']} ({last['verdict']})" if last else "") +
                       (f". {md_cell(i['blocker'])}" if i.get("blocker") else ""))
        out.append("")
    hist = [(a["app"], i, s) for lvl in ("1", "2") for a in c["levels"].get(lvl, []) for i in a["inputs"]
            for s in i["sets"] if s["verdict"] == "SUPERSEDED"]
    inval = [(a["app"], i, t) for lvl in ("1", "2") for a in c["levels"].get(lvl, []) for i in a["inputs"]
             for t in i["attempts"] if t["verdict"] == "INVALIDATED"]
    if hist or inval:
        out += ["### History kept, never current", ""]
        for app, i, s in hist:
            out.append(f"- SUPERSEDED {md_cell(app)} / {md_cell(i['input_id'])}: {s['n']} samples, median "
                       f"{fmt_plain(s['median'])} ({s['git_commit']}) -- an earlier definition of the input, not compared with the current one")
        for app, i, t in inval:
            out.append(f"- INVALIDATED {md_cell(app)} / {md_cell(i['input_id'])}: run {t['run_id']} ({t['git_commit']}) "
                       f"ran another workload than the input")
        out.append("")
    return out


def render_md(bundle, out_dir):
    out = ["# Timing results", "",
           "Region-of-interest (ROI) timing of the Level 1 benchmarks and Level 2 mini-applications, generated by "
           "tools/timing/report.py from the timing records (the same data as [index.html](index.html), which adds the "
           "per-input detail and history). `null`: not observable or not collected.", ""]
    for i, c in enumerate(bundle["campaigns"]):
        if c.get("kind") == "registry":
            out += render_md_registry(c)
        else:
            out += [f"## {campaign_label(c)}", ""] + (
                ["Kept verbatim from the earlier published page; its measurements are not part of the current "
                 "results and are not compared with them. Spread column here: CV = stddev / median.", ""]
                if c.get("historical") else []) + render_md_cases(c)
    return "\n".join(out).rstrip("\n") + "\n"


def render_md_cases(data):
    out = [f"Latest record {data['generated_from'] or '-'}, {data['records']} records: the latest successful run of every "
           f"measured (case, platform).", ""]
    for lvl in ("1", "2"):
        apps = data["levels"].get(lvl, [])
        out += [f"### {LEVEL_NAMES[int(lvl)]}", ""]
        head = ["application", "input", "platform", "ROI", "CV (stddev/median)", "device busy", "host gap", "kernels in ROI",
                "ROI share of process", "profiler x"] + (["FOM", "vs own timer"] if lvl == "2" else []) + \
               ["runs", "vs previous"]
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "|".join("---" if i < 3 else "--:" for i in range(len(head))) + "|")
        for a in apps:
            for c in a["cases"]:
                for pid, cell in sorted(c["cells"].items()):
                    if cell is None:
                        continue
                    run = cell["run"]
                    R, d, ctx = run["roi"], run.get("device") or {}, run["context"]
                    wall, proc = R.get("wall_s"), ctx.get("process_wall_s")
                    runs = R.get("runs_s") or []
                    cv = (R["wall_s_stddev"] / wall) if (wall and R.get("wall_s_stddev") is not None
                                                        and len(runs) > 1) else None
                    prev = cell["prev"]
                    dl = "first run" if not prev or not wall else f"{100 * (wall - prev['roi_s']) / prev['roi_s']:+.1f}%"
                    status = "" if cell["latest_status"] == "ok" else f" (latest: {cell['latest_status']})"
                    row = [md_cell(a["app"]) + status, md_cell(c["case"]), md_cell(pid),
                           fmt_plain(wall) if run["status"] == "ok" else run["status"], md_pct(cv, 1),
                           md_pct(d.get("busy_frac_of_roi"), cap=True),
                           fmt_plain(max(d["host_gap_s"], 0) if d.get("host_gap_s") is not None else None),
                           "null" if d.get("compute_ops") is None else f"{d['compute_ops']:,}",
                           md_pct(wall / proc if (wall and proc) else None, 1),
                           "null" if R.get("profiler_inflation") is None else f"{R['profiler_inflation']:.2f}"]
                    if lvl == "2":
                        t = run.get("app_timer") or {}
                        row += [md_cell(fom_plain(run.get("fom"))),
                                "-" if t.get("roi_diff_frac") is None else f"{100 * t['roi_diff_frac']:+.3f}%"]
                    row += [str(len(cell["history"])), dl]
                    out.append("| " + " | ".join(row) + " |")
        out.append("")
    return out


# ----------------------------------------------------------------- entry points

def build_bundle(results_roots, history_pages=()):
    roots = [os.path.realpath(r) for r in (results_roots if isinstance(results_roots, (list, tuple)) else [results_roots])]
    SCRUB_ROOTS[:] = sorted({os.path.dirname(r) for r in roots} | set(roots), key=len, reverse=True)
    recs = load(roots)
    hosts = set()
    for r in recs:
        for proc in (r.get("inputs") or {}).get("processes") or []:
            if proc.get("host"):
                hosts |= {proc["host"], proc["host"].split(".")[0]}
        h = ((r.get("platform_info") or {}).get("host") or {}).get("hostname")
        if h:
            hosts |= {h, h.split(".")[0]}
    SCRUB_HOSTS[:] = sorted((h for h in hosts if len(h) >= 4), key=len, reverse=True)
    registry = any(((r.get("registry") or {}).get("input_id")) for r in recs)
    campaigns = [build_registry(roots) if registry else build_data(recs)]
    campaigns += [load_history_page(p) for p in history_pages]
    return {"campaigns": campaigns}


def write(results_root, out_dir, history_pages=()):
    """Render the results root(s) into out_dir/index.html and out_dir/README.md; returns the two paths."""
    bundle = build_bundle(results_root, history_pages)
    os.makedirs(out_dir, exist_ok=True)
    page = os.path.join(out_dir, "index.html")
    md = os.path.join(out_dir, "README.md")
    with open(page, "w") as f:
        f.write(render_html(bundle))
    with open(md, "w") as f:
        f.write(render_md(bundle, out_dir))
    return [page, md]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", action="append", default=None,
                    help="results directory (repeatable: several directories of one campaign); default results/timing")
    ap.add_argument("--history-page", action="append", default=[],
                    help="an earlier published index.html to embed as a separate historical campaign (repeatable)")
    ap.add_argument("--out", default=None, help="output directory (default <first results-root>/report)")
    ap.add_argument("--publish", action="store_true", help=f"write to {os.path.relpath(PUBLISH_DIR, REPO)}/")
    a = ap.parse_args(argv)
    if a.publish and a.out:
        ap.error("--publish and --out are exclusive")
    roots = a.results_root or [DEFAULT_RESULTS]
    out = PUBLISH_DIR if a.publish else (a.out or os.path.join(roots[0], "report"))
    for r in roots:
        if not os.path.isdir(r):
            print(f"report: no results directory {r} (run summarize.py first)", file=sys.stderr)
            return 1
    for p in write(roots, out, a.history_page):
        print(f"report: {os.path.relpath(p, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
