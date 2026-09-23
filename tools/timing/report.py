#!/usr/bin/env python3
"""Render the Level 1 / Level 2 timing results as an interactive web page (+ Markdown twin).

    tools/timing/report.py                   # results/timing  ->  results/timing/report/
    tools/timing/report.py --publish         # results/timing  ->  docs/timing/ (tracked: commit it to show it)
    tools/timing/report.py --results-root DIR --out DIR

The page (index.html) lists the applications of each level. Choosing one shows its
inputs (the cases of tools/timing/cases/ plus anything measured) against the platforms
(every platform with a measurement or a conformance record); a combination that was
never measured is shown as null. Only after an application, an input and a platform
are chosen does it show that measurement: ROI time and runs, the process breakdown,
device activity inside the ROI per category, the top operations, runtime API calls,
the FOM, the application's own timer, the launcher audit, the caveats, the input as
run, and the run history of that combination.

README.md next to it is the plain-text twin the repository browser displays: the
latest successful run of every measured (case, platform) as a Level 1 and a Level 2
table.

summarize.py calls write() every time it runs, and the measurement front-ends run
summarize after every measurement, so results/timing/report/ always shows the newest
data. Publishing to docs/timing/ is the deliberate step: raw evidence and the JSON/CSV
records stay out of git; the rendered page is the one snapshot that may be committed.
Standard library only. The output depends only on the records and the case tables (no
wall-clock time in it), and absolute paths of this checkout are written as {REPO}.
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

def scrub(obj):
    """Absolute paths of this checkout (and the home directory) never leave the machine."""
    if isinstance(obj, str):
        s = obj.replace(REPO, "{REPO}")
        return s.replace(HOME, "~") if HOME and HOME != "/" else s
    if isinstance(obj, list):
        return [scrub(x) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    return obj


def load(results_root):
    recs = []
    for p in sorted(glob.glob(os.path.join(results_root, "level[0-9]", "*", "*", "*.json"))):
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
    return {"generated_from": latest, "records": len(recs), "platforms": plats, "levels": levels}


# ----------------------------------------------------------------- HTML

def render_html(data):
    css = open(os.path.join(ASSETS, "report.css")).read()
    js = open(os.path.join(ASSETS, "report.js")).read()
    # JSON inside <script>: "</" would end the element, so it is written as "<\/" (still valid JSON)
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    n_apps = {k: len(v) for k, v in data["levels"].items()}
    as_of = html.escape(data["generated_from"] or "no measurements yet")
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
  <div class="eyebrow">tools/timing &middot; results as of {as_of}</div>
  <h1>Timing results</h1>
  <p class="sub">Region-of-interest (ROI) timing of the Level&nbsp;1 benchmarks and the Level&nbsp;2 mini-applications:
  the computation between the markers in each source, without start-up, set-up, warm-up and verification. Choose an
  application, then an input and a platform.</p>
</header>
<noscript><p class="lede">This page needs JavaScript. README.md next to it has the same results as plain tables.</p></noscript>
<nav class="tabs" role="tablist" aria-label="Level">
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


def render_md(data, out_dir):
    out = ["# Timing results", "",
           f"Region-of-interest timing of the Level 1 benchmarks and Level 2 mini-applications (latest record "
           f"{data['generated_from'] or '-'}, {data['records']} records). Open [index.html](index.html) to choose an "
           f"application, an input and a platform; this file lists the latest successful run of every measured "
           f"combination. `null`: not observable on that platform.", ""]
    for lvl in ("1", "2"):
        apps = data["levels"].get(lvl, [])
        out += [f"## {LEVEL_NAMES[int(lvl)]}", ""]
        head = ["application", "input", "platform", "ROI", "spread", "device busy", "host gap", "kernels in ROI",
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
    return "\n".join(out)


# ----------------------------------------------------------------- entry points

def write(results_root, out_dir):
    """Render results_root into out_dir/index.html and out_dir/README.md; returns the two paths."""
    data = build_data(load(results_root))
    os.makedirs(out_dir, exist_ok=True)
    page = os.path.join(out_dir, "index.html")
    md = os.path.join(out_dir, "README.md")
    with open(page, "w") as f:
        f.write(render_html(data))
    with open(md, "w") as f:
        f.write(render_md(data, out_dir))
    return [page, md]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=DEFAULT_RESULTS)
    ap.add_argument("--out", default=None, help="output directory (default <results-root>/report)")
    ap.add_argument("--publish", action="store_true", help=f"write to {os.path.relpath(PUBLISH_DIR, REPO)}/")
    a = ap.parse_args(argv)
    if a.publish and a.out:
        ap.error("--publish and --out are exclusive")
    out = PUBLISH_DIR if a.publish else (a.out or os.path.join(a.results_root, "report"))
    if not os.path.isdir(a.results_root):
        print(f"report: no results directory {a.results_root} (run summarize.py first)", file=sys.stderr)
        return 1
    for p in write(a.results_root, out):
        print(f"report: {os.path.relpath(p, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
