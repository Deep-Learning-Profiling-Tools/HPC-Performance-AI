#!/usr/bin/env python3
"""Turn raw measurement directories into one JSON per run plus per-level CSVs.

    tools/timing/summarize.py                   # scan build/timing, write results/timing
    tools/timing/summarize.py --raw-root DIR --out-root DIR
    tools/timing/summarize.py --csv-only        # rebuild the CSVs from existing JSONs
    tools/timing/summarize.py --run-id ID       # only that run's raw data (the engine does this)

Input : <raw-root>/level<L>/<app>/<case>/<run_id>/   (tools/timing/lib/engine.sh)
Output: <out-root>/level<L>/<app>/<case>/<run_id>.json   schema hpcperf-timing-2
        <out-root>/summary_level<L>.csv  one row per run      (regenerated, never appended)
        <out-root>/ops_level<L>.csv      one row per (run, device op inside the ROI)
        <out-root>/report/               the web page + Markdown twin (report.py), unless --no-report

Levels are kept in separate files so runs measured under different protocols are never
averaged by accident. Level 0 is the conformance probe (probes/conformance/).

What the numbers mean -- everything is about the region of interest (ROI): the
computation between the markers, without start-up, set-up, warm-up and verification.

  roi_wall_s              clean runs (no profiler), timed by the markers; median
  roi_profiled_wall_s     the same region in the profiled run; the ratio is
                          roi_profiler_inflation and makes the profiler's cost visible
  device_busy_s           union of all device activity clipped to the ROI -- concurrent
                          operations count once (device_op_time_sum_s is the naive sum)
  host_gap_s              roi_wall_s - device_busy_s: time inside the ROI in which the
                          device was idle, i.e. the host was the bottleneck
  device_<category>_s     time per activity category inside the ROI (compute, copy_h2d,
                          copy_d2h, copy_d2d, copy_other, fill, collective, other)
  process_wall_s, pre_roi_s, post_roi_s, whole_*   context only, never the headline

Device columns are empty (null) when the platform's collector cannot observe them --
never 0. Device-side durations come from the profiled run: they are timestamped on the
device and are insensitive to profiler overhead (measured: 1.2% spread in kernel time
across sampling settings), unlike host time.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analysis  # noqa: E402
import cases as case_tables  # noqa: E402
import collectors  # noqa: E402
import report  # noqa: E402
from collectors import CATEGORIES, COPY_CATEGORIES  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SCHEMA = "hpcperf-timing-2"
RAW_SCHEMA = "hpcperf-timing-raw-2"
PLATFORMS = os.path.join(HERE, "platforms")
INFLATION_NOTE = 1.2      # profiled ROI / clean ROI above this gets a caveat
APP_TIMER_NOTE = 0.02     # ROI vs the application's own timer for the same region
HOST_GAP_TOLERANCE = 0.01  # a negative host gap within 1% of the ROI is timing noise

COLUMNS = [
    "level", "app", "case", "platform", "run_id", "utc", "status",
    "roi_wall_s", "roi_wall_s_min", "roi_wall_s_max", "roi_wall_s_stddev", "roi_runs",
    "roi_entries", "roi_excluded_s", "roi_processes",
    "roi_profiled_wall_s", "roi_profiler_inflation",
    "device_busy_s", "device_busy_frac_of_roi", "host_gap_s", "host_gap_frac_of_roi",
    "device_compute_s", "device_copy_h2d_s", "device_copy_d2h_s", "device_copy_d2d_s",
    "device_copy_other_s", "device_fill_s", "device_collective_s", "device_other_s",
    "device_compute_ops", "device_copy_ops", "device_fill_ops",
    "device_copy_h2d_bytes", "device_copy_d2h_bytes", "device_copy_d2d_bytes",
    "device_op_time_sum_s", "device_overlap_s",
    "runtime", "runtime_api_calls", "runtime_api_time_s", "runtime_api_sync_calls", "runtime_api_sync_s",
    "ops_n", "top_op_name", "top_op_category", "top_op_total_s", "top_op_share",
    "process_wall_s", "pre_roi_s", "post_roi_s", "whole_device_busy_s", "whole_compute_ops",
    "fom_name", "fom_value", "fom_unit", "fom_better", "fom_status",
    "launcher_audit_ok", "launcher_audit",
    "verify_vs_roi", "skip_verify", "inputs_env", "inputs_argv",
    "device_vendor", "device_product", "device_arch", "device_count", "device_memory_mib",
    "device_core_clock_mhz", "device_core_clock_max_mhz", "device_mem_clock_mhz",
    "device_power_limit_w", "device_temp_c", "driver_version", "runtime_version",
    "collector", "collector_version", "conformance",
    "host_cpu_model", "host_cpus_allowed", "host_loadavg_1m",
    "exe_sha256", "git_commit", "git_dirty", "raw_dir",
    "app_timer_s", "roi_vs_app_timer",
]
OPS_COLUMNS = ["level", "app", "case", "platform", "run_id", "op", "category", "count",
               "total_s", "avg_s", "min_s", "max_s", "share"]

finite = analysis.finite
APP_TIMERS = case_tables.app_timers()


# ----------------------------------------------------------------- small helpers

def dash(v):
    return "" if v in (None, "-") else v


def read_meta(path):
    meta = {}
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k] = v
    return meta


def read_run_txt(d):
    p = os.path.join(d, "run.txt")
    if not os.path.isfile(p):
        return None
    out = {}
    for tok in open(p).read().split():
        k, _, v = tok.partition("=")
        out[k] = int(v)
    return out


def sec(ns):
    return None if ns is None else finite("seconds", ns / 1e9)


def platform_conformance(platform_id):
    p = os.path.join(PLATFORMS, f"{platform_id}.json")
    if not os.path.isfile(p):
        return None
    try:
        return json.load(open(p)).get("conformance")
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------- FOM (the app's own metric)

def extract_fom(log_path, meta):
    name = dash(meta.get("fom_name"))
    out = {"name": name or None, "value": None, "unit": dash(meta.get("fom_unit")) or None,
           "better": dash(meta.get("fom_better")) or None, "source": dash(meta.get("fom_source")) or None,
           "regex": dash(meta.get("fom_regex")) or None, "from_clean_run": True, "status": "none"}
    if not name:
        return out
    if out["source"] != "stdout":
        raise RuntimeError(f"fom_source={out['source']!r} is not implemented")
    pat = re.compile(out["regex"], re.M)
    if pat.groups != 1:
        raise RuntimeError(f"fom_regex needs exactly one capture group: {out['regex']!r}")
    if not log_path or not os.path.isfile(log_path):
        out["status"] = "log_missing"
        return out
    with open(log_path, errors="replace") as f:
        matches = pat.findall(f.read())
    if not matches:
        out["status"] = "not_matched"
        return out
    out["value"] = finite("fom", float(matches[-1].replace(",", "")))   # xsbench prints 1,234,567
    out["status"] = "ok"
    return out


def extract_app_timer(log_path, spec):
    """The application's own timer for the region its ROI marks (cases/level2_apps.tsv).
    A cross-check of the marker placement, not a metric: None when the app has none."""
    if not spec:
        return None
    regex, scale = spec
    out = {"regex": regex, "value_s": None, "roi_diff_frac": None, "status": "not_matched"}
    if not log_path or not os.path.isfile(log_path):
        out["status"] = "log_missing"
        return out
    with open(log_path, errors="replace") as f:
        matches = re.findall(regex, f.read(), re.M)
    if matches:
        out["value_s"] = finite("app timer", float(matches[-1].replace(",", "")) * scale)
        out["status"] = "ok"
    return out


def launcher_audit(log_path):
    if not log_path or not os.path.isfile(log_path):
        return None, None
    with open(log_path, errors="replace") as f:
        m = re.search(r"audit summary: .*", f.read())
    if not m:
        return None, None
    line = m.group(0)
    return line, bool(re.search(r"\b0 mismatch, 0 unverified\b", line))


# ----------------------------------------------------------------- one run

def build_record(raw):
    meta_path = os.path.join(raw, "run_meta.txt")
    if not os.path.isfile(meta_path):
        return None
    meta = read_meta(meta_path)
    if meta.get("schema") != RAW_SCHEMA:
        return None
    level = int(meta["level"])
    status = meta.get("status", "incomplete")
    caveats = []

    dev_desc = {}
    if meta.get("device_json"):
        try:
            dev_desc = json.loads(meta["device_json"])
        except ValueError:
            caveats.append("device descriptor in run_meta.txt is not valid JSON")
    device_info = dev_desc.get("device", {})
    host_info = dev_desc.get("host", {})
    platform_id = meta.get("platform_id") or device_info.get("platform_id")

    # ---- clean runs: the ROI time
    runs = []
    for d in sorted(glob.glob(os.path.join(raw, "clean.*")), key=lambda p: int(p.rsplit(".", 1)[1])):
        rt = read_run_txt(d)
        logs = [analysis.parse_roi_log(p) for p in sorted(glob.glob(os.path.join(d, "roi.*")))]
        roi = analysis.clean_roi(logs)
        runs.append({"dir": d, "run": rt, "logs": logs, "roi": roi})
    good = [r for r in runs if r["run"] and r["run"].get("rc") == 0 and r["roi"] and r["roi"]["entries"] > 0]
    walls = [sec(r["roi"]["wall_ns"]) for r in good]

    roi = {"wall_s": None, "runs_s": walls, "wall_s_min": None, "wall_s_max": None, "wall_s_stddev": None,
           "entries": None, "excluded_s": None, "processes": None, "imbalance_s": None,
           "profiled_wall_s": None, "profiled_marker_wall_s": None, "profiler_inflation": None}
    context = {"process_wall_s": None, "pre_roi_s": None, "post_roi_s": None}
    if walls:
        r0 = good[0]["roi"]
        roi.update({
            "wall_s": finite("roi median", statistics.median(walls)),
            "wall_s_min": min(walls), "wall_s_max": max(walls),
            "wall_s_stddev": finite("roi stddev", statistics.stdev(walls)) if len(walls) > 1 else 0.0,
            "entries": r0["entries"], "excluded_s": sec(r0["excluded_ns"]),
            "processes": r0["processes"], "imbalance_s": sec(r0["imbalance_ns"]),
        })
        pw, pre, post = [], [], []
        for r in good:
            pw.append((r["run"]["end_ns"] - r["run"]["start_ns"]) / 1e9)
            if r["roi"]["first_begin_real"] is not None:
                pre.append((r["roi"]["first_begin_real"] - r["run"]["start_ns"]) / 1e9)
            if r["roi"]["last_end_real"] is not None:
                post.append((r["run"]["end_ns"] - r["roi"]["last_end_real"]) / 1e9)
        context.update({"process_wall_s": finite("wall", statistics.median(pw)),
                        "pre_roi_s": finite("pre", statistics.median(pre)) if pre else None,
                        "post_roi_s": finite("post", statistics.median(post)) if post else None})
        if any(r["roi"]["unterminated"] for r in good):
            caveats.append("An ROI was begun but never ended in at least one clean run (the program "
                           "exited inside it); its time is not counted.")
        if any(r["roi"]["overflow"] for r in good):
            caveats.append("The ROI event buffer overflowed: the markers are inside a loop that runs "
                           "more than 8192 times. Move them outward.")
        if any(r["roi"]["unmatched_end"] for r in good):
            caveats.append("HPCPERF_ROI_END was called without a matching BEGIN.")
        if len({r["roi"]["entries"] for r in good}) > 1:
            caveats.append("The number of ROI entries differs between clean runs.")
        if r0["processes"] > 1:
            caveats.append("Multi-process ROI: the job's ROI is the slowest process's. This path is "
                           "UNVERIFIED on the node the tool was built on (one GPU).")
    elif status == "ok":
        status = "roi_missing"

    # ---- inputs: what was declared, and what the processes actually ran with
    declared_env = {}
    for kv in dash(meta.get("case_env")).split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            declared_env[k] = v
    procs = []
    if good:
        for log in good[0]["logs"]:
            procs.append({"rank": log.get("rank"), "argv": log.get("argv"), "exe": log.get("exe"),
                          "cwd": log.get("cwd"), "host": log.get("host")})
    exe_sha = meta.get("exe_sha256")
    if not exe_sha and procs and procs[0]["exe"] and os.path.isfile(procs[0]["exe"]):
        import hashlib
        h = hashlib.sha256()
        with open(procs[0]["exe"], "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        exe_sha = h.hexdigest()
    inputs = {"declared_env": declared_env, "declared_argv": dash(meta.get("argv")),
              "processes": procs, "exe_sha256": exe_sha}

    # ---- FOM and launcher audit, from the first clean run
    clean0_log = os.path.join(runs[0]["dir"], "run.log") if runs else None
    fom = extract_fom(clean0_log, meta)
    audit, audit_ok = launcher_audit(clean0_log)
    app_timer = extract_app_timer(clean0_log, APP_TIMERS.get(meta["app"])) if level == 2 else None
    if app_timer and app_timer["value_s"] and roi["wall_s"]:
        diff = finite("roi vs app timer", (roi["wall_s"] - app_timer["value_s"]) / app_timer["value_s"])
        app_timer["roi_diff_frac"] = diff
        if abs(diff) > APP_TIMER_NOTE:
            caveats.append(f"The ROI ({roi['wall_s']:.6f} s) differs by {100 * diff:+.2f}% from the application's "
                           f"own timer for the same region ({app_timer['value_s']:.6f} s): check the markers.")
    if fom["status"] == "not_matched":
        caveats.append(f"The case expects a FOM named {fom['name']!r} but its pattern did not match the "
                       f"clean run's output; left empty rather than guessed.")
    if audit_ok is False:
        caveats.append(f"The launcher's GPU-binding audit is not clean: {audit!r}.")

    # ---- profiled run: the device picture inside the ROI
    name = meta.get("collector", "none")
    mod = collectors.get(name)
    caps = mod.CAPABILITIES
    device = runtime = whole = None
    ops, prof_info = [], {}
    prof_dir = os.path.join(raw, "prof")
    if name != "none" and os.path.isdir(prof_dir) and status == "ok":
        trace = mod.open(prof_dir)
        try:
            res = analysis.analyze_trace(trace, caps)
            prof_info = trace.info()
        finally:
            trace.close()
        pr = res["profiled_roi"]
        if not pr["found"]:
            status = "roi_missing_in_trace"
            caveats.append("The profiled run's trace contains no ROI range although the clean runs "
                           "recorded one: the collector did not see the markers.")
        else:
            roi["profiled_wall_s"] = pr["wall_s"]
            plogs = [analysis.parse_roi_log(p) for p in sorted(glob.glob(os.path.join(prof_dir, "roi.*")))]
            proi = analysis.clean_roi(plogs)
            if proi:
                roi["profiled_marker_wall_s"] = sec(proi["wall_ns"])
            if roi["wall_s"]:
                roi["profiler_inflation"] = finite("inflation", pr["wall_s"] / roi["wall_s"])
            if roi["entries"] is not None and pr["entries"] != roi["entries"]:
                caveats.append(f"ROI entries differ between the clean ({roi['entries']}) and the "
                               f"profiled run ({pr['entries']}).")
            device = res["device"]
            whole = res["whole_process"]
            runtime = {"name": mod.RUNTIME, "roi": res["runtime_api_roi"], "whole": res["runtime_api_whole"]}
            ops = res["ops"]
            if device and roi["wall_s"]:
                gap = roi["wall_s"] - device["busy_s"]
                device["host_gap_s"] = finite("host gap", gap)
                device["host_gap_frac_of_roi"] = finite("host gap frac", gap / roi["wall_s"])
                device["busy_frac_of_roi"] = finite("busy frac", device["busy_s"] / roi["wall_s"])
                if gap < -HOST_GAP_TOLERANCE * roi["wall_s"]:
                    caveats.append(f"Device busy time inside the ROI ({device['busy_s']:.6f} s, profiled run) "
                                   f"exceeds the clean ROI ({roi['wall_s']:.6f} s): the two runs differ by "
                                   f"more than timing noise, so host_gap_s is not meaningful.")
            infl = roi["profiler_inflation"]
            if infl is not None and infl > INFLATION_NOTE:
                caveats.append(f"The profiler stretched the ROI {infl:.2f}x (profiled {pr['wall_s']:.4f} s vs "
                               f"clean {roi['wall_s']:.4f} s). Device-side durations are unaffected; "
                               f"roi_wall_s is the clean value.")
    elif name == "none":
        caveats.append("No collector for this platform: device columns are null (not observable), "
                       "only the ROI time and the FOM were measured.")
    if name != "none" and not mod.VERIFIED:
        caveats.append(f"Collector {name} is interface-only and UNVERIFIED.")

    conformance = platform_conformance(platform_id) if platform_id else None
    if level > 0 and name != "none" and not (conformance and conformance.get("status") == "pass"):
        caveats.append(f"Platform {platform_id} has no passing conformance record for collector "
                       f"{name} (tools/timing/probes/conformance/run_conformance.sh).")
    if meta.get("verify_vs_roi") == "inside":
        caveats.append("This benchmark's correctness check runs inside a timed kernel and cannot be "
                       "separated from the ROI; it is included in the measured time.")

    recorded = prof_info.get("recorded_env_names")
    if recorded is not None:
        bad = [n for n in recorded if re.search(meta.get("env_deny_regex", "$^"), n)]
        if bad:
            caveats.append(f"The profiler recorded variable names matching the credential deny rule: {bad}.")

    return {
        "schema": SCHEMA, "level": level, "app": meta["app"], "case": meta["case"],
        "platform": platform_id, "run_id": meta["run_id"], "utc": meta.get("utc"), "status": status,
        "measurement": {
            "tool": f"tools/timing/measure_level{level}.sh" if level else "tools/timing/probes/conformance",
            "protocol": {"warmup_runs": int(meta.get("warmup_runs", 0)), "clean_runs": int(meta.get("clean_runs", 0)),
                         "profiled_runs": int(meta.get("profiled_runs", 0))},
            "collector": {"name": name, "version": dash(meta.get("collector_version")) or None,
                          "verified": bool(mod.VERIFIED), "capabilities": sorted(caps)},
            "skip_verify": meta.get("skip_verify") == "1",
            "verify_vs_roi": dash(meta.get("verify_vs_roi")) or None,
            "roi_where": dash(meta.get("roi_where")).split(),
            "roi_excludes": dash(meta.get("roi_excludes")) or None,
            "backend": meta.get("backend"), "gpus": dash(meta.get("gpus")) or None,
            "timeout_s": meta.get("timeout_s"),
            "env_script": dash(meta.get("env_script")) or None,
            "env_allow": meta.get("env_allow", "").split(),
            "env_deny_regex": meta.get("env_deny_regex"),
            "notes": dash(meta.get("notes")) or None,
        },
        "inputs": inputs,
        "roi": roi,
        "device": device,
        "runtime_api": runtime,
        "ops": ops,
        "context": dict(context, whole_process=whole),
        "fom": fom,
        "app_timer": app_timer,
        "launcher": {"audit": audit, "audit_ok": audit_ok},
        "platform_info": {"device": device_info, "host": host_info, "conformance": conformance},
        "profiler": {k: v for k, v in prof_info.items() if k != "recorded_env_names"} |
                    ({"recorded_env_name_count": len(recorded)} if recorded is not None else {}),
        "provenance": {"git_commit": meta.get("git_commit"), "git_dirty": meta.get("git_dirty") == "1",
                       "raw_dir": os.path.relpath(raw, REPO)},
        "caveats": caveats,
    }


# ----------------------------------------------------------------- flattening

def flatten(rec):
    row = {c: "" for c in COLUMNS}
    roi, dev, ctx = rec["roi"], rec.get("device") or {}, rec["context"]
    whole = ctx.get("whole_process") or {}
    rt = rec.get("runtime_api") or {}
    rt_roi = rt.get("roi") or {}
    pdev, host = rec["platform_info"]["device"], rec["platform_info"]["host"]
    ops = rec.get("ops") or []
    top = ops[0] if ops else None

    def v(x):
        return "" if x is None else x

    def copy_ops():
        vals = [dev.get(f"{c}_ops") for c in COPY_CATEGORIES]
        return "" if all(x is None for x in vals) else sum(x for x in vals if x is not None)

    conf = rec["platform_info"].get("conformance") or {}
    row.update({
        "level": rec["level"], "app": rec["app"], "case": rec["case"], "platform": v(rec["platform"]),
        "run_id": rec["run_id"], "utc": v(rec["utc"]), "status": rec["status"],
        "roi_wall_s": v(roi["wall_s"]), "roi_wall_s_min": v(roi["wall_s_min"]),
        "roi_wall_s_max": v(roi["wall_s_max"]), "roi_wall_s_stddev": v(roi["wall_s_stddev"]),
        "roi_runs": len(roi["runs_s"]), "roi_entries": v(roi["entries"]),
        "roi_excluded_s": v(roi["excluded_s"]), "roi_processes": v(roi["processes"]),
        "roi_profiled_wall_s": v(roi["profiled_wall_s"]), "roi_profiler_inflation": v(roi["profiler_inflation"]),
        "device_busy_s": v(dev.get("busy_s")), "device_busy_frac_of_roi": v(dev.get("busy_frac_of_roi")),
        "host_gap_s": v(dev.get("host_gap_s")), "host_gap_frac_of_roi": v(dev.get("host_gap_frac_of_roi")),
        "device_compute_ops": v(dev.get("compute_ops")), "device_copy_ops": copy_ops() if dev else "",
        "device_fill_ops": v(dev.get("fill_ops")),
        "device_op_time_sum_s": v(dev.get("op_time_sum_s")), "device_overlap_s": v(dev.get("overlap_s")),
        "runtime": v(rt.get("name")), "runtime_api_calls": v(rt_roi.get("calls")),
        "runtime_api_time_s": v(rt_roi.get("time_s")), "runtime_api_sync_calls": v(rt_roi.get("sync_calls")),
        "runtime_api_sync_s": v(rt_roi.get("sync_s")),
        "ops_n": len(ops) if dev else "", "top_op_name": top["name"] if top else "",
        "top_op_category": top["category"] if top else "", "top_op_total_s": top["total_s"] if top else "",
        "top_op_share": top["share"] if top else "",
        "process_wall_s": v(ctx.get("process_wall_s")), "pre_roi_s": v(ctx.get("pre_roi_s")),
        "post_roi_s": v(ctx.get("post_roi_s")), "whole_device_busy_s": v(whole.get("busy_s")),
        "whole_compute_ops": v(whole.get("compute_ops")),
        "fom_name": v(rec["fom"]["name"]), "fom_value": v(rec["fom"]["value"]), "fom_unit": v(rec["fom"]["unit"]),
        "fom_better": v(rec["fom"]["better"]), "fom_status": rec["fom"]["status"],
        "launcher_audit_ok": "" if rec["launcher"]["audit_ok"] is None else int(rec["launcher"]["audit_ok"]),
        "launcher_audit": v(rec["launcher"]["audit"]),
        "verify_vs_roi": v(rec["measurement"]["verify_vs_roi"]), "skip_verify": int(rec["measurement"]["skip_verify"]),
        "inputs_env": ";".join(f"{k}={x}" for k, x in rec["inputs"]["declared_env"].items()),
        "inputs_argv": json.dumps(rec["inputs"]["processes"][0]["argv"]) if rec["inputs"]["processes"] else "",
        "device_vendor": v(pdev.get("vendor")), "device_product": v(pdev.get("product")),
        "device_arch": v(pdev.get("arch")), "device_count": v(pdev.get("count_visible")),
        "device_memory_mib": v(pdev.get("memory_total_mib")), "device_core_clock_mhz": v(pdev.get("core_clock_mhz")),
        "device_core_clock_max_mhz": v(pdev.get("core_clock_max_mhz")),
        "device_mem_clock_mhz": v(pdev.get("mem_clock_mhz")), "device_power_limit_w": v(pdev.get("power_limit_w")),
        "device_temp_c": v(pdev.get("temperature_c")), "driver_version": v(pdev.get("driver_version")),
        "runtime_version": v((pdev.get("runtime") or {}).get("version")),
        "collector": rec["measurement"]["collector"]["name"],
        "collector_version": v(rec["measurement"]["collector"]["version"]),
        "conformance": v(conf.get("status")),
        "host_cpu_model": v(host.get("cpu_model")), "host_cpus_allowed": v(host.get("cpus_allowed")),
        "host_loadavg_1m": v(host.get("loadavg_1m")),
        "exe_sha256": v(rec["inputs"]["exe_sha256"]), "git_commit": v(rec["provenance"]["git_commit"]),
        "git_dirty": int(rec["provenance"]["git_dirty"]), "raw_dir": rec["provenance"]["raw_dir"],
        "app_timer_s": v((rec.get("app_timer") or {}).get("value_s")),
        "roi_vs_app_timer": v((rec.get("app_timer") or {}).get("roi_diff_frac")),
    })
    for c in CATEGORIES:
        row[f"device_{c}_s"] = v(dev.get(f"{c}_s")) if dev else ""
    for c in ("copy_h2d", "copy_d2h", "copy_d2d"):
        row[f"device_{c}_bytes"] = v(dev.get(f"{c}_bytes")) if dev else ""
    return row


def op_rows(rec):
    for o in rec.get("ops") or []:
        yield {"level": rec["level"], "app": rec["app"], "case": rec["case"], "platform": rec["platform"],
               "run_id": rec["run_id"], "op": o["name"], "category": o["category"], "count": o["count"],
               "total_s": o["total_s"], "avg_s": o["avg_s"], "min_s": o["min_s"], "max_s": o["max_s"],
               "share": o["share"]}


# ----------------------------------------------------------------- driver

def raw_dirs(raw_root):
    for level_dir in sorted(glob.glob(os.path.join(raw_root, "level[0-9]"))):
        for run in sorted(glob.glob(os.path.join(level_dir, "*", "*", "*"))):
            if os.path.isfile(os.path.join(run, "run_meta.txt")):
                yield run


def json_path(out_root, rec):
    return os.path.join(out_root, f"level{rec['level']}", rec["app"], rec["case"], f"{rec['run_id']}.json")


def load_records(out_root):
    records, skipped = [], 0
    for path in sorted(glob.glob(os.path.join(out_root, "level[0-9]", "*", "*", "*.json"))):
        rec = json.load(open(path))
        if rec.get("schema") != SCHEMA:
            skipped += 1
            continue
        records.append(rec)
    records.sort(key=lambda r: (r["level"], r["app"], r["case"], r["run_id"]))
    return records, skipped


def write_csvs(out_root, records):
    written = []
    for level in sorted({r["level"] for r in records} | {1, 2}):
        recs = [r for r in records if r["level"] == level]
        if not recs and level not in (1, 2):
            continue
        p = os.path.join(out_root, f"summary_level{level}.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            for r in recs:
                w.writerow(flatten(r))
        q = os.path.join(out_root, f"ops_level{level}.csv")
        with open(q, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=OPS_COLUMNS)
            w.writeheader()
            for r in recs:
                for row in op_rows(r):
                    w.writerow(row)
        written += [p, q]
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default=os.path.join(REPO, "build", "timing"))
    ap.add_argument("--out-root", default=os.path.join(REPO, "results", "timing"))
    ap.add_argument("--csv-only", action="store_true", help="rebuild the CSVs from existing JSONs")
    ap.add_argument("--run-id", action="append", default=[], help="only these runs' raw data (repeatable)")
    ap.add_argument("--no-report", action="store_true", help="do not regenerate <out-root>/report/")
    a = ap.parse_args(argv)

    written = failed = 0
    if not a.csv_only:
        if not os.path.isdir(a.raw_root):
            print(f"summarize: no raw directory {a.raw_root}", file=sys.stderr)
            return 1
        for raw in raw_dirs(a.raw_root):
            if a.run_id and os.path.basename(raw) not in a.run_id:
                continue
            try:
                rec = build_record(raw)
            except Exception as exc:  # noqa: BLE001 -- report and keep going
                print(f"summarize: FAILED {os.path.relpath(raw, REPO)}: {exc}", file=sys.stderr)
                failed += 1
                continue
            if rec is None:
                continue
            p = json_path(a.out_root, rec)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                json.dump(rec, f, indent=2)
                f.write("\n")
            written += 1

    os.makedirs(a.out_root, exist_ok=True)
    records, skipped = load_records(a.out_root)
    outs = write_csvs(a.out_root, records)
    by = {}
    for r in records:
        by.setdefault(r["level"], []).append(r)
    print(f"summarize: json_written={written} failed={failed} records={len(records)} skipped_old_schema={skipped}")
    for level, recs in sorted(by.items()):
        ok = sum(1 for r in recs if r["status"] == "ok")
        fom = sum(1 for r in recs if r["fom"]["status"] == "ok")
        print(f"           level{level}: {len(recs)} runs, {ok} ok, {len(recs) - ok} not ok, fom_ok={fom}")
    if not a.no_report:
        outs += report.write(a.out_root, os.path.join(a.out_root, "report"))
    for p in outs:
        print(f"           {os.path.relpath(p, REPO)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
