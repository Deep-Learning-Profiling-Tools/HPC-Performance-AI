#!/usr/bin/env python3
"""Render the timing results as a web page, plus a Markdown twin that GitHub displays.

    tools/timing/report.py                   # results/timing  ->  results/timing/report/
    tools/timing/report.py --publish         # results/timing  ->  docs/timing/ (tracked: commit it to show it)
    tools/timing/report.py --results-root DIR --out DIR

summarize.py calls write() every time it runs, and the measurement front-ends run
summarize after every measurement, so results/timing/report/ always shows the newest
data. Publishing to docs/timing/ is the deliberate step: raw evidence and the JSON/CSV
records stay out of git; the rendered page is the one snapshot that may be committed.

Per (level, app, case, platform) the page shows the latest SUCCESSFUL run, its change
against the previous successful run, and -- as "attention" items -- a latest run that
failed, a ROI that disagrees with the application's own timer, spread between clean
runs, and the other caveat conditions. Standard library only; the output depends only
on the records (no wall-clock time in it), so the same data gives the same bytes.
"""

import argparse
import glob
import html
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cases as case_tables  # noqa: E402
import collectors  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SCHEMA = "hpcperf-timing-2"
DEFAULT_RESULTS = os.path.join(REPO, "results", "timing")
PUBLISH_DIR = os.path.join(REPO, "docs", "timing")
ASSETS = os.path.join(HERE, "report_assets")
PLATFORMS = os.path.join(HERE, "platforms")

SPREAD_NOTE = 0.05      # clean-run coefficient of variation
INFLATION_NOTE = 1.2    # profiled ROI / clean ROI
APP_TIMER_NOTE = 0.02   # ROI vs the application's own timer
SETUP_NOTE = 0.10       # Level 2: ROI share of the process below this = set-up dominated input
DELTA_NOTE = 0.05       # change against the previous successful run


# ----------------------------------------------------------------- data

def load(results_root):
    recs = []
    for p in sorted(glob.glob(os.path.join(results_root, "level[0-9]", "*", "*", "*.json"))):
        try:
            with open(p) as f:
                r = json.load(f)
        except (OSError, ValueError):
            continue
        if r.get("schema") == SCHEMA:
            recs.append(r)
    return recs


def select(recs):
    """One row per (level, app, case, platform): latest successful run, latest run, previous success."""
    groups = {}
    for r in recs:
        if r["level"] < 1:            # level 0 is the conformance probe: shown under platforms
            continue
        groups.setdefault((r["level"], r["app"], r["case"], r.get("platform") or "-"), []).append(r)
    rows = []
    for key in sorted(groups):
        runs = sorted(groups[key], key=lambda r: r["run_id"])
        ok = [r for r in runs if r["status"] == "ok"]
        rows.append({"key": key, "level": key[0], "app": key[1], "case": key[2], "platform": key[3],
                     "rec": ok[-1] if ok else runs[-1], "latest": runs[-1],
                     "prev": ok[-2] if len(ok) > 1 else None, "runs": len(runs)})
    return rows


def metrics(rec):
    R = rec.get("roi") or {}
    d = rec.get("device") or {}
    ctx = rec.get("context") or {}
    whole = ctx.get("whole_process") or {}
    wall, proc = R.get("wall_s"), ctx.get("process_wall_s")
    runs = R.get("runs_s") or []
    top = (rec.get("ops") or [{}])[0] or {}
    timer = rec.get("app_timer") or {}
    sd = R.get("wall_s_stddev")
    return {
        "ok": rec["status"] == "ok", "roi": wall, "n_clean": len(runs),
        "cv": sd / wall if (wall and sd is not None and len(runs) > 1) else None,
        "entries": R.get("entries"), "excl": R.get("excluded_s") or 0.0,
        "proc": proc, "pre": ctx.get("pre_roi_s"), "post": ctx.get("post_roi_s"),
        "share": wall / proc if (wall and proc) else None,
        "busy": d.get("busy_frac_of_roi"), "gap": d.get("host_gap_s"),
        "ops": d.get("compute_ops"), "ops_all": whole.get("compute_ops"),
        "infl": R.get("profiler_inflation"), "top": top.get("name"), "top_share": top.get("share"),
        "fom": rec.get("fom") or {}, "timer": timer.get("value_s"), "timer_diff": timer.get("roi_diff_frac"),
        "audit": (rec.get("launcher") or {}).get("audit_ok"),
        "collector": rec["measurement"]["collector"]["name"],
        "verify": rec["measurement"].get("verify_vs_roi"),
    }


def delta(row):
    if not row["prev"] or row["rec"]["status"] != "ok":
        return None
    a, b = row["rec"]["roi"]["wall_s"], row["prev"]["roi"]["wall_s"]
    return (a - b) / b if (a and b) else None


def suites():
    try:
        return {r["app"]: r["suite"] for r in case_tables.read_table("level1_apps.tsv", case_tables.L1_APP_COLS)}
    except case_tables.CaseError:
        return {}


def platform_records(ids):
    out = {}
    for pid in sorted(ids):
        p = os.path.join(PLATFORMS, f"{pid}.json")
        try:
            with open(p) as f:
                out[pid] = json.load(f)
        except (OSError, ValueError):
            out[pid] = None
    return out


def label(row, multi_platform):
    s = f"{row['app']}" + ("" if row["case"] == "default" else f"/{row['case']}")
    return s + (f" @ {row['platform']}" if multi_platform else "")


def attention(rows, plats, multi):
    """(severity, level label, text) for everything a reader should not miss."""
    rank = {"fail": 0, "warn": 1, "info": 2}
    items = []
    for row in rows:
        rec, lat, m = row["rec"], row["latest"], metrics(row["rec"])
        name = f"L{row['level']} {label(row, multi)}"
        if lat["status"] != "ok":
            tail = (f"; the table shows the earlier successful run {rec['run_id']}" if rec["status"] == "ok"
                    else "; there is no successful run of this case")
            items.append(("fail", name, f"latest run {lat['run_id']} ended with status {lat['status']}{tail}"))
        if rec["status"] != "ok":
            continue
        if m["audit"] is False:
            items.append(("fail", name, "the launcher's GPU-binding audit is not clean"))
        if m["timer_diff"] is not None and abs(m["timer_diff"]) > APP_TIMER_NOTE:
            items.append(("warn", name, f"ROI differs by {100 * m['timer_diff']:+.2f}% from the application's own "
                                        f"timer for the same region -- check the markers"))
        if m["cv"] is not None and m["cv"] > SPREAD_NOTE:
            items.append(("warn", name, f"clean runs spread {100 * m['cv']:.1f}% (coefficient of variation, "
                                        f"{m['n_clean']} runs)"))
        if m["fom"].get("status") in ("not_matched", "log_missing"):
            items.append(("warn", name, f"the FOM pattern found nothing ({m['fom']['status']}); left empty"))
        dl = delta(row)
        if dl is not None and abs(dl) > DELTA_NOTE:
            items.append(("info", name, f"ROI changed {100 * dl:+.1f}% against the previous successful run "
                                        f"({row['prev']['run_id']})"))
        if m["infl"] is not None and m["infl"] > INFLATION_NOTE:
            items.append(("info", name, f"the profiler stretched the ROI {m['infl']:.2f}x (device times unaffected; "
                                        f"the headline is the clean run)"))
        if row["level"] == 2 and m["share"] is not None and m["share"] < SETUP_NOTE:
            items.append(("info", name, f"set-up dominated input: the ROI is {100 * m['share']:.1f}% of the process "
                                        f"({fmt_plain(m['roi'])} of {fmt_plain(m['proc'])})"))
    for pid, prec in plats.items():
        used = {metrics(r["rec"])["collector"] for r in rows if r["platform"] == pid}
        if used - {"none"} and not (prec and (prec.get("conformance") or {}).get("status") == "pass"):
            items.append(("warn", f"platform {pid}", "no passing conformance record for its collector"))
    items.sort(key=lambda it: (rank[it[0]], it[1], it[2]))
    return items


# ----------------------------------------------------------------- formatting

def esc(s):
    return html.escape("" if s is None else str(s), quote=True)


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
        return "-"
    n, u = _time_parts(s)
    return f"{n} {u}"


def fmt_time(s):
    if s is None:
        return '<span class="z">&ndash;</span>'
    n, u = _time_parts(s)
    return f"{n}<small>&thinsp;{'&micro;s' if u == 'us' else u}</small>"


def fmt_pct(x, digits=0, cap=False):
    if x is None:
        return '<span class="z">&ndash;</span>'
    return f"{100 * (min(x, 1.0) if cap else x):.{digits}f}%"


def minibar(x):
    x = 0.0 if x is None else max(0.0, min(1.0, x))
    return f'<span class="minibar"><i style="width:{100 * x:.1f}%"></i></span>'


def short_op(name):
    if not name:
        return ""
    if name.startswith("["):
        return name.strip("[]").replace("memcpy ", "copy ")
    n = name[5:] if name.startswith("void ") else name
    n = n.split("(")[0]
    depth, out = 0, ""
    for c in n:
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
        elif depth == 0:
            out += c
    return out.split("::")[-1].strip() or n[:40]


def fom_text(fom, markup=True):
    v = fom.get("value")
    if v is None:
        if fom.get("status") == "none":
            return '<span class="blank">none printed</span>' if markup else "none printed"
        return esc(fom.get("status")) if markup else str(fom.get("status"))
    s = f"{v:.4g}" if (abs(v) >= 1e5 or abs(v) < 1e-2) else f"{v:,.1f}"
    unit = fom.get("unit") or ""
    return f'{s} <span class="unit">{esc(unit)}</span>' if markup else f"{s} {unit}".strip()


def dv(v):
    if v is None:
        return ""
    return f"{v:.9g}" if isinstance(v, float) else str(v)


def td(content, v=None, cls=""):
    c = f' class="{cls}"' if cls else ""
    return f'<td{c} data-v="{esc(dv(v))}">{content}</td>'


def delta_html(dl):
    if dl is None:
        return '<span class="z">first run</span>'
    cls = "dlt big" if abs(dl) > DELTA_NOTE else "dlt"
    return f'<span class="{cls}">{100 * dl:+.1f}%</span>'


# ----------------------------------------------------------------- HTML pieces

def anatomy(rows, multi):
    rows = sorted(rows, key=lambda r: (-(metrics(r["rec"])["share"] or 0), r["key"]))
    out = ['<div class="anat">']
    for row in rows:
        m = metrics(row["rec"])
        key = esc(f"{row['app']} {row['case']} {row['platform']}".lower())
        name = esc(row["app"]) + ("" if row["case"] == "default" else f" <small>{esc(row['case'])}</small>")
        if multi:
            name += f" <small>{esc(row['platform'])}</small>"
        if not m["ok"] or not m["proc"]:
            out.append(f'<div class="arow" data-key="{key}"><div class="lbl">{name}</div>'
                       f'<div class="pbar" aria-hidden="true"></div>'
                       f'<div class="num"><span class="pill bad">{esc(row["rec"]["status"])}</span></div></div>')
            continue
        pre, post, roi = max(m["pre"] or 0, 0), max(m["post"] or 0, 0), m["roi"]
        between = max(m["proc"] - pre - post - roi, 0)
        if m["busy"] is None:
            segs = [("s-pre", pre), ("s-roi", roi), ("s-excl", between), ("s-post", post)]
        else:
            busy = min(roi, m["busy"] * roi)
            segs = [("s-pre", pre), ("s-busy", busy), ("s-idle", roi - busy), ("s-excl", between), ("s-post", post)]
        tot = sum(v for _, v in segs) or 1.0
        bar = "".join(f'<i class="{c}" style="width:{100 * v / tot:.2f}%"></i>' for c, v in segs if v > 0)
        out.append(f'<div class="arow" data-key="{key}"><div class="lbl">{name}</div>'
                   f'<div class="pbar" aria-hidden="true">{bar}</div>'
                   f'<div class="num">{fmt_time(roi)} / {fmt_time(m["proc"])}</div></div>')
    out.append("</div>")
    return "\n".join(out)


LEGEND = ('<div class="legend">'
          '<span><b class="s-pre"></b>before the ROI (start-up, set-up, warm-up)</span>'
          '<span><b class="s-busy"></b>ROI, device busy</span>'
          '<span><b class="s-idle"></b>ROI, device idle (host gap)</span>'
          '<span><b class="s-roi"></b>ROI, device not observed</span>'
          '<span><b class="s-excl"></b>excluded, or between ROI entries</span>'
          '<span><b class="s-post"></b>after the ROI (checks, output, teardown)</span></div>')


def name_cell(row, m, multi):
    tags = ""
    if row["latest"]["status"] != "ok":
        tags += f'<span class="pill bad">latest: {esc(row["latest"]["status"])}</span> '
    if m["verify"] == "inside":
        tags += '<span class="tag">check inside ROI</span>'
    elif m["excl"]:
        tags += f'<span class="tag">{fmt_time(m["excl"])} excluded</span>'
    name = esc(row["app"]) + ("" if row["case"] == "default" else f' <span class="unit">{esc(row["case"])}</span>')
    return f'<th data-v="{esc(row["app"] + "/" + row["case"])}">{name}{" " + tags if tags else ""}</th>'


def table_level1(rows, multi, suite):
    head = ['<th>benchmark</th>', '<th>suite</th>'] + (['<th>platform</th>'] if multi else []) + [
        '<th class="n" data-type="num">ROI</th>', '<th class="n" data-type="num">spread</th>',
        '<th class="n" data-type="num">entries</th>', '<th data-type="num">device busy in ROI</th>',
        '<th class="n" data-type="num">host gap</th>', '<th class="n" data-type="num">kernels in ROI / process</th>',
        '<th data-type="num">ROI share of process</th>', '<th class="n" data-type="num">profiler &times;</th>',
        '<th class="n" data-type="num">runs</th>', '<th class="n" data-type="num">vs previous</th>']
    body = []
    for row in rows:
        m = metrics(row["rec"])
        key = esc(f"{row['app']} {row['case']} {row['platform']}".lower())
        if m["ops"] is None:
            ops = ""
        elif m["ops_all"] is not None:
            ops = f"{m['ops']:,} / {m['ops_all']:,}"
        else:
            ops = f"{m['ops']:,}"
        cells = [name_cell(row, m, multi), td(esc(suite.get(row["app"], "")), suite.get(row["app"], ""))]
        if multi:
            cells.append(td(esc(row["platform"]), row["platform"]))
        cells += [
            td(fmt_time(m["roi"]), m["roi"], "n em"),
            td(fmt_pct(m["cv"], 1), m["cv"], "n"),
            td(esc(m["entries"]) if m["entries"] is not None else "", m["entries"], "n"),
            td(minibar(m["busy"]) + fmt_pct(m["busy"], cap=True), m["busy"]),
            td(fmt_time(max(m["gap"], 0) if m["gap"] is not None else None), m["gap"], "n"),
            td(ops or '<span class="z">&ndash;</span>', m["ops"], "n"),
            td(minibar(m["share"]) + fmt_pct(m["share"], 1), m["share"]),
            td(f"{m['infl']:.2f}" if m["infl"] is not None else '<span class="z">&ndash;</span>', m["infl"],
               "n em" if (m["infl"] or 0) > INFLATION_NOTE else "n"),
            td(str(row["runs"]), row["runs"], "n"),
            td(delta_html(delta(row)), delta(row), "n"),
        ]
        body.append(f'<tr data-key="{key}">' + "".join(cells) + "</tr>")
    return ('<div class="tscroll"><table class="sortable"><caption>Latest successful run per case. ROI: median of the '
            'clean runs; device columns: the profiled run, clipped to the same markers (device busy capped at 100%: '
            'the profiled busy time can exceed the clean ROI by timing noise). Click a header to sort.</caption>'
            f'<thead><tr>{"".join(head)}</tr></thead><tbody>\n' + "\n".join(body) + "\n</tbody></table></div>")


def table_level2(rows, multi):
    head = ['<th>application / case</th>'] + (['<th>platform</th>'] if multi else []) + [
        '<th class="n" data-type="num">ROI</th>', '<th data-type="num">ROI share of process</th>',
        '<th data-type="num">device busy in ROI</th>', '<th class="n" data-type="num">host gap</th>',
        '<th class="n" data-type="num">kernels in ROI</th>', '<th>top operation (share)</th>',
        '<th data-type="num">FOM (clean run)</th>', '<th class="n" data-type="num">vs own timer</th>',
        '<th class="n" data-type="num">profiler &times;</th>', '<th>audit</th>',
        '<th class="n" data-type="num">runs</th>', '<th class="n" data-type="num">vs previous</th>']
    body = []
    for row in rows:
        m = metrics(row["rec"])
        key = esc(f"{row['app']} {row['case']} {row['platform']}".lower())
        if m["timer_diff"] is None:
            own = '<span class="z">&ndash;</span>'
        else:
            cls = "chk" if abs(m["timer_diff"]) <= APP_TIMER_NOTE else "off"
            own = f'<span class="{cls}">{100 * m["timer_diff"]:+.3f}%</span>'
        audit = ('<span class="pill ok">clean</span>' if m["audit"] else
                 '<span class="pill na">no launcher</span>' if m["audit"] is None else
                 '<span class="pill bad">not clean</span>')
        cells = [name_cell(row, m, multi)]
        if multi:
            cells.append(td(esc(row["platform"]), row["platform"]))
        cells += [
            td(fmt_time(m["roi"]), m["roi"], "n em"),
            td(minibar(m["share"]) + fmt_pct(m["share"]), m["share"]),
            td(minibar(m["busy"]) + fmt_pct(m["busy"], cap=True), m["busy"]),
            td(fmt_time(max(m["gap"], 0) if m["gap"] is not None else None), m["gap"], "n"),
            td(f"{m['ops']:,}" if m["ops"] is not None else '<span class="z">&ndash;</span>', m["ops"], "n"),
            f'<td class="op" title="{esc(m["top"])}" data-v="{esc(short_op(m["top"]))}">{esc(short_op(m["top"]))} '
            f'<span class="unit">{fmt_pct(m["top_share"]) if m["top"] else ""}</span></td>',
            td(fom_text(m["fom"]), m["fom"].get("value")),
            td(own, m["timer_diff"], "n"),
            td(f"{m['infl']:.2f}" if m["infl"] is not None else '<span class="z">&ndash;</span>', m["infl"],
               "n em" if (m["infl"] or 0) > INFLATION_NOTE else "n"),
            td(audit, {True: "clean", False: "not clean", None: "no launcher"}[m["audit"]]),
            td(str(row["runs"]), row["runs"], "n"),
            td(delta_html(delta(row)), delta(row), "n"),
        ]
        body.append(f'<tr data-key="{key}">' + "".join(cells) + "</tr>")
    return ('<div class="tscroll"><table class="sortable"><caption>Latest successful run per case. ROI from the clean '
            'run(s); device columns from the profiled run; FOM from the clean run; "vs own timer" compares the ROI '
            'with the timer the application prints for the same region. Click a header to sort.</caption>'
            f'<thead><tr>{"".join(head)}</tr></thead><tbody>\n' + "\n".join(body) + "\n</tbody></table></div>")


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def render_html(rows, recs, out_dir):
    multi = len({r["platform"] for r in rows}) > 1
    plats = platform_records({r["platform"] for r in rows if r["platform"] != "-"})
    att = attention(rows, plats, multi)
    suite = suites()
    rel = os.path.relpath(HERE, out_dir)
    css = open(os.path.join(ASSETS, "report.css")).read()
    js = open(os.path.join(ASSETS, "report.js")).read()
    by_level = {L: [r for r in rows if r["level"] == L] for L in sorted({r["level"] for r in rows})}
    latest_utc = max((r.get("utc") or "" for r in recs), default="") or "no measurements"
    commits = sorted({(r["rec"].get("provenance") or {}).get("git_commit", "")[:10] for r in rows} - {""})

    def ok_count(L):
        rs = by_level.get(L, [])
        return sum(1 for r in rs if r["rec"]["status"] == "ok"), len(rs)

    figs = []
    for L in sorted(by_level):
        k, n = ok_count(L)
        figs.append(f'<div class="fig"><div class="v">{k}<small>/{n}</small></div><div class="k">Level {L} cases '
                    f'with a successful run</div></div>')
    for L in sorted(by_level):
        ms = [metrics(r["rec"]) for r in by_level[L] if r["rec"]["status"] == "ok"]
        b = median(m["busy"] for m in ms)
        s = median(m["share"] for m in ms)
        if s is not None:
            figs.append(f'<div class="fig"><div class="v">{100 * s:.1f}<small>%</small></div><div class="k">Level {L}: '
                        f'ROI share of the process (median)</div></div>')
        if b is not None:
            figs.append(f'<div class="fig"><div class="v">{100 * min(b, 1):.0f}<small>%</small></div><div class="k">'
                        f'Level {L}: device busy inside the ROI (median)</div></div>')
    diffs = [abs(metrics(r["rec"])["timer_diff"]) for r in rows if metrics(r["rec"])["timer_diff"] is not None]
    if diffs:
        figs.append(f'<div class="fig"><div class="v">&le;{100 * max(diffs):.2f}<small>%</small></div><div class="k">'
                    f'ROI vs the apps\' own timers ({len(diffs)} cases)</div></div>')
    nfail = sum(1 for a in att if a[0] == "fail")
    figs.append(f'<div class="fig"><div class="v">{nfail}</div><div class="k">failures needing attention '
                f'({len(att)} items in all)</div></div>')

    env = []
    for pid, prec in plats.items():
        dev = next((r["rec"]["platform_info"]["device"] for r in rows if r["platform"] == pid), {}) or {}
        env.append(f'<span><b>platform</b> {esc(pid)}</span>')
        if dev.get("product"):
            env.append(f'<span><b>device</b> {esc(dev.get("count_visible"))}&times; {esc(dev.get("product"))} '
                       f'{esc(dev.get("arch") or "")}, driver {esc(dev.get("driver_version") or "?")}</span>')
    collectors_used = sorted({metrics(r["rec"])["collector"] for r in rows})
    env.append(f'<span><b>collectors</b> {esc(", ".join(collectors_used) or "-")}</span>')
    env.append(f'<span><b>runs</b> {len(recs)} records</span>')
    env.append(f'<span><b>source commits</b> {esc(", ".join(commits) or "-")}</span>')

    if att:
        sevpill = {"fail": "bad", "warn": "warn", "info": "na"}
        att_html = '<ul class="att">' + "".join(
            f'<li data-key="{esc(n.lower())}"><span class="pill {sevpill[s]}">{s}</span><span><b>{esc(n)}</b> '
            f'&mdash; {esc(t)}</span></li>' for s, n, t in att) + "</ul>"
    else:
        att_html = '<p class="none">Nothing needs attention.</p>'

    sections = []
    for L, rs in by_level.items():
        ms = [metrics(r["rec"]) for r in rs if r["rec"]["status"] == "ok"]
        s, b = median(m["share"] for m in ms), median(m["busy"] for m in ms)
        lede = (f"{len(ms)} of {len(rs)} cases have a successful run. The ROI is a median "
                f"{'-' if s is None else f'{100 * s:.1f}%'} of the process, and the device is busy a median "
                f"{'-' if b is None else f'{100 * min(b, 1):.0f}%'} of the ROI.")
        what = ("50 standalone GPU kernels (ctest cases): 1 warm-up, 5 clean and 1 profiled run each."
                if L == 1 else "24 mini-applications through their run.sh: 1 clean and 1 profiled run each."
                if L == 2 else "")
        table = table_level1(rs, multi, suite) if L == 1 else table_level2(rs, multi)
        sections.append(
            f'<section id="level{L}"><div class="shead"><h2>Level {L}</h2><span class="badge">{len(rs)} cases</span>'
            f'</div><p class="lede">{esc(what)} {esc(lede)}</p>{LEGEND}{anatomy(rs, multi)}'
            f'<p class="cap">Each bar is one clean run scaled to its own process wall clock; right: ROI / process. '
            f'The device-busy share comes from the profiled run and is applied to the clean ROI.</p>{table}</section>')

    timer_rows = [r for r in rows if metrics(r["rec"])["timer"] is not None]
    timer_html = ""
    if timer_rows:
        trs = []
        for r in sorted(timer_rows, key=lambda r: -(metrics(r["rec"])["roi"] or 0)):
            m = metrics(r["rec"])
            cls = "chk" if abs(m["timer_diff"]) <= APP_TIMER_NOTE else "off"
            trs.append(f'<tr data-key="{esc((r["app"] + " " + r["case"] + " " + r["platform"]).lower())}">'
                       f'<th>{esc(label(r, multi))}</th><td class="n em">{m["roi"]:.6f}</td>'
                       f'<td class="n">{m["timer"]:.6f}</td>'
                       f'<td class="n"><span class="{cls}">{100 * m["timer_diff"]:+.3f}%</span></td></tr>')
        timer_html = (
            '<section id="check"><div class="shead"><h2>Is the ROI the right region?</h2><span class="badge">check'
            '</span></div><p class="lede">These applications print their own timer for the region their markers '
            'enclose. The two are measured independently; a difference above 2% means the markers and the '
            'application disagree about what the computation is.</p><div class="tscroll"><table><caption>seconds; '
            'the timer pattern of each application is in tools/timing/cases/level2_apps.tsv</caption><thead><tr>'
            '<th>case</th><th class="n">ROI</th><th class="n">own timer</th><th class="n">difference</th></tr>'
            '</thead><tbody>' + "".join(trs) + "</tbody></table></div></section>")

    prow = []
    for pid, prec in plats.items():
        conf = (prec or {}).get("conformance") or {}
        checks = conf.get("checks") or []
        good = sum(1 for c in checks if c.get("ok"))
        pill = (f'<span class="pill ok">pass</span> {good}/{len(checks)} checks, {esc(conf.get("date_utc", "")[:10])}'
                if conf.get("status") == "pass" else '<span class="pill warn">no passing record</span>')
        dev = next((r["rec"]["platform_info"] for r in rows if r["platform"] == pid), {}) or {}
        d, h = dev.get("device") or {}, dev.get("host") or {}
        rt = d.get("runtime") or {}
        prow.append(f'<tr><th>{esc(pid)}</th><td>{esc(d.get("product") or d.get("vendor") or "-")} '
                    f'{esc(d.get("arch") or "")}</td><td>{esc(d.get("driver_version") or "-")}</td>'
                    f'<td>{esc((rt.get("name") or "-") + " " + (rt.get("version") or ""))}</td>'
                    f'<td>{esc(h.get("cpu_model") or "-")}</td><td>{esc(conf.get("collector") or "-")}</td>'
                    f'<td>{pill}</td></tr>')
    crow = []
    for name in collectors.names():
        mod = collectors.get(name)
        state = ('<span class="pill ok">verified</span>' if mod.VERIFIED else
                 '<span class="pill na">interface only</span>')
        caps = ", ".join(sorted(mod.CAPABILITIES)) or "none (ROI time and FOM only)"
        crow.append(f'<tr><th>{esc(name)}</th><td>{state}</td><td>{esc(caps)}</td></tr>')

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
  <div class="eyebrow">tools/timing &middot; results as of {esc(latest_utc)}</div>
  <h1>Timing results</h1>
  <p class="sub">Region-of-interest (ROI) timing: the computation between the markers in each benchmark's source,
  without start-up, set-up, warm-up, verification and bulk output. Each row is the latest successful run of a case;
  the page is regenerated after every measurement.</p>
  <div class="env">{"".join(env)}</div>
  <div class="figs">{"".join(figs)}</div>
  <div class="tools"><label for="filter">Filter</label><input id="filter" type="search"
    placeholder="application, case or platform" autocomplete="off"></div>
</header>

<section id="attention"><div class="shead"><h2>Needs attention</h2><span class="badge">{len(att)} items</span></div>
{att_html}</section>

{"".join(sections)}
{timer_html}
<section id="platforms"><div class="shead"><h2>Platforms and collectors</h2><span class="badge">hardware</span></div>
<p class="lede">A platform's device columns are trusted only after its collector reproduced the conformance probe
exactly. On a platform without a collector the ROI time and the FOM are still measured; device columns stay empty,
never 0.</p>
<div class="tscroll"><table><thead><tr><th>platform</th><th>device</th><th>driver</th><th>runtime</th>
<th>host CPU</th><th>collector</th><th>conformance</th></tr></thead><tbody>{"".join(prow)}</tbody></table></div>
<div class="tscroll"><table><thead><tr><th>collector</th><th>status</th><th>observes</th></tr></thead>
<tbody>{"".join(crow)}</tbody></table></div></section>

<section id="method"><div class="shead"><h2>How to read and regenerate this page</h2><span class="badge">method</span></div>
<div class="twocol">
<p class="lede"><b>ROI</b>: median of the clean runs (no profiler), timed by the markers themselves.
<b>Device busy</b>: union of all device activity inside the ROI in the profiled run. <b>Host gap</b>: ROI minus
device busy -- time in the ROI when the device was idle. <b>Profiler &times;</b>: profiled ROI / clean ROI.
<b>Entries</b>: how often the ROI was entered (their times add up). Method and placement rule:
<a href="{esc(rel)}/README.md">tools/timing/README.md</a>, <a href="{esc(rel)}/roi/README.md">roi/README.md</a>,
formats <a href="{esc(rel)}/SCHEMA.md">SCHEMA.md</a>.</p>
<p class="lede">Every <code>measure_level1.sh</code> / <code>measure_level2.sh</code> run summarizes its data and
rewrites <code>results/timing/report/</code>. To update the copy in the repository:
<code>python3 tools/timing/report.py --publish</code> (writes <code>docs/timing/</code>), then commit it.</p>
</div></section>

<footer>generated by tools/timing/report.py from {len(recs)} records (schema {SCHEMA}) &middot; latest record
{esc(latest_utc)} &middot; raw evidence stays in build/timing/, records in results/timing/ (not in git)</footer>
</div>
<script>
{js}</script>
</body>
</html>
"""


# ----------------------------------------------------------------- Markdown twin

def md_cell(s):
    return str(s).replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;").replace("\n", " ")


def md_pct(x, digits=0, cap=False):
    return "-" if x is None else f"{100 * (min(x, 1.0) if cap else x):.{digits}f}%"


def render_md(rows, recs, out_dir):
    multi = len({r["platform"] for r in rows}) > 1
    plats = platform_records({r["platform"] for r in rows if r["platform"] != "-"})
    att = attention(rows, plats, multi)
    suite = suites()
    latest_utc = max((r.get("utc") or "" for r in recs), default="") or "no measurements"
    out = ["# Timing results", "",
           f"Region-of-interest timing of the Level 1 benchmarks and Level 2 mini-applications, generated by "
           f"`tools/timing/report.py` from {len(recs)} records; latest record {latest_utc}. "
           f"Open [index.html](index.html) for the charts and sortable tables; this file is the plain-text twin "
           f"that the repository browser shows.", "",
           "Platforms: " + (", ".join(f"`{p}`" for p in plats) or "-"), ""]
    out += ["## Needs attention", ""]
    if att:
        out += [f"- **{s}** {md_cell(n)} -- {md_cell(t)}" for s, n, t in att]
    else:
        out.append("Nothing needs attention.")
    for L in sorted({r["level"] for r in rows}):
        rs = [r for r in rows if r["level"] == L]
        out += ["", f"## Level {L}", ""]
        if L == 1:
            out.append("| benchmark | suite | ROI | spread | entries | device busy | host gap | kernels ROI / process "
                       "| ROI share | profiler x | runs | vs previous |")
            out.append("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        else:
            out.append("| application / case | ROI | ROI share | device busy | host gap | kernels in ROI | FOM "
                       "| vs own timer | profiler x | audit | runs | vs previous |")
            out.append("|---|--:|--:|--:|--:|--:|---|--:|--:|---|--:|--:|")
        for r in rs:
            m = metrics(r["rec"])
            dl = delta(r)
            name = md_cell(label(r, multi)) + ("" if r["latest"]["status"] == "ok"
                                               else f" (latest: {r['latest']['status']})")
            common_tail = [md_cell(f"{m['infl']:.2f}" if m["infl"] is not None else "-")]
            dlt = "first run" if dl is None else f"{100 * dl:+.1f}%"
            if L == 1:
                ops = "-" if m["ops"] is None else f"{m['ops']:,} / {m['ops_all']:,}" if m["ops_all"] is not None \
                    else f"{m['ops']:,}"
                cells = [name, md_cell(suite.get(r["app"], "")), fmt_plain(m["roi"]), md_pct(m["cv"], 1),
                         str(m["entries"] if m["entries"] is not None else "-"), md_pct(m["busy"], cap=True),
                         fmt_plain(max(m["gap"], 0) if m["gap"] is not None else None), ops,
                         md_pct(m["share"], 1)] + common_tail + [str(r["runs"]), dlt]
            else:
                own = "-" if m["timer_diff"] is None else f"{100 * m['timer_diff']:+.3f}%"
                audit = {True: "clean", False: "NOT CLEAN", None: "no launcher"}[m["audit"]]
                cells = [name, fmt_plain(m["roi"]), md_pct(m["share"]), md_pct(m["busy"], cap=True),
                         fmt_plain(max(m["gap"], 0) if m["gap"] is not None else None),
                         "-" if m["ops"] is None else f"{m['ops']:,}", md_cell(fom_text(m["fom"], markup=False)),
                         own] + common_tail + [audit, str(r["runs"]), dlt]
            out.append("| " + " | ".join(cells) + " |")
    out += ["", "## Columns", "",
            "- **ROI**: median of the clean runs (no profiler), timed by the markers; **spread**: coefficient of "
            "variation of the clean runs.",
            "- **device busy**: union of all device activity inside the ROI (profiled run); **host gap**: ROI minus "
            "device busy.",
            "- **profiler x**: profiled ROI / clean ROI; **vs own timer**: ROI against the timer the application "
            "prints for the same region; **vs previous**: ROI against the previous successful run of the case.",
            f"- Method: [tools/timing/README.md]({os.path.relpath(HERE, out_dir)}/README.md). Regenerate: "
            "`python3 tools/timing/report.py --publish`.", ""]
    return "\n".join(out)


# ----------------------------------------------------------------- entry points

def write(results_root, out_dir):
    """Render results_root into out_dir/index.html and out_dir/README.md; returns the two paths."""
    recs = load(results_root)
    rows = select(recs)
    os.makedirs(out_dir, exist_ok=True)
    page = os.path.join(out_dir, "index.html")
    md = os.path.join(out_dir, "README.md")
    with open(page, "w") as f:
        f.write(render_html(rows, recs, out_dir))
    with open(md, "w") as f:
        f.write(render_md(rows, recs, out_dir))
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
