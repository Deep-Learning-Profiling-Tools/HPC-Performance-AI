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
Level 3 has no markers: its region is the application's own timer for the time-step loop
(apptimers.py, roi.source = "app_timer"), and without markers the profiled run's device
picture is the whole process (context) unless the application emits an NVTX range for
its loop itself.

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
import apptimers  # noqa: E402
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
    "input_id",
    "placement_cpus_allowed", "placement_mems_allowed", "placement_gpus", "placement_consistent", "placement_policy",
    "roi_source", "roi_steps", "roi_setup_s", "ops_scope",
    "whole_compute_s", "whole_copy_h2d_s", "whole_copy_d2h_s", "whole_copy_d2d_s", "whole_fill_s",
    "whole_runtime_api_calls",
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


def read_bind_logs(d):
    """The placement records (bind.<pid>, probes/bindprobe.c) of one run directory."""
    procs = []
    for p in sorted(glob.glob(os.path.join(d, "bind.*"))):
        rec = {"tasks": [], "gpus": [], "env": {}}
        with open(p, errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                k, _, v = line.partition(" ")
                t = v.split()
                if k == "task" and len(t) >= 6:          # tid cpus <set> last_cpu <n> name <comm>
                    rec["tasks"].append({"tid": t[0], "cpus": t[2], "last_cpu": int(t[4]), "name": " ".join(t[6:])})
                elif k == "gpu" and len(t) >= 4:         # minor <n> bus <pci>
                    rec["gpus"].append({"minor": int(t[1]), "pci_bus_id": t[3]})
                elif k == "env":
                    n, _, val = v.partition("=")
                    rec["env"][n] = val
                else:
                    rec[k] = v
        procs.append(rec)
    return procs


def placement_of(procs):
    """Per measured process: the CPU set, NUMA memory set and GPU it ran with (diagnostic context)."""
    def num(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    out = []
    for r in procs:
        masks = {}
        for t in r["tasks"]:
            masks[t["cpus"]] = masks.get(t["cpus"], 0) + 1
        out.append({"pid": num(r.get("pid")), "exe": r.get("exe"), "roi_log": r.get("roi_log") == "1",
                    "cpus_allowed": r.get("cpus_allowed_end"), "mems_allowed": r.get("mems_allowed_end"),
                    "cpus_allowed_at_start": r.get("cpus_allowed_start"),
                    "threads": num(r.get("threads")), "thread_cpusets": masks,
                    "last_cpus": sorted({t["last_cpu"] for t in r["tasks"]}),
                    "cpu_start": num(r.get("cpu_start")), "cpu_end": num(r.get("cpu_end")),
                    "voluntary_ctxt_switches": num(r.get("voluntary_ctxt_switches")),
                    "nonvoluntary_ctxt_switches": num(r.get("nonvoluntary_ctxt_switches")),
                    "gpus": [g["pci_bus_id"] for g in r["gpus"]], "env": r["env"]})
    return out


def bind_policy_of(meta):
    """The CPU-binding policy the engine applied (run_meta.txt bind_*): explicit (cases/level2_binding.tsv
    through the launcher's HPCPERF_CPUS_PER_RANK interface, with the pairs it added to the run's
    environment), runtime (the MPI runtime's default binding: the campaigns before 2026-09-30), application
    (Level 3: no engine policy -- each run.sh applies its own application-specific CPU/thread policy, and the
    placement block shows the actual placement), none (Level 1: no policy, the process stays unbound inside
    the allocation) or not recorded."""
    out = {"kind": meta.get("bind_policy") or "not recorded"}
    for k in ("launcher", "host_threads", "cpus_per_rank", "omp_pin"):
        v = meta.get(f"bind_{k}")
        if v not in (None, "", "-"):
            out[k] = int(v) if v.isdigit() else v
    env = meta.get("bind_env")
    out["env"] = [e for e in env.split(";") if e] if env else []
    return out


def placement_block(meta, runs, prof_dir):
    """The record's `placement`: what the probe saw in every run, and whether it was the same
    CPU set / memory set / GPU in all of them, plus the binding policy the engine applied. Never a metric."""
    clean = [placement_of(read_bind_logs(r["dir"])) for r in runs]
    prof = placement_of(read_bind_logs(prof_dir)) if os.path.isdir(prof_dir) else []
    # the measured processes are the ones that wrote an ROI log; helpers that merely opened a
    # GPU (a launcher's nvidia-smi audit, a profiler) count only when no process wrote one
    measured = [p for ps in clean for p in ps if p["roi_log"]] or [p for ps in clean for p in ps if p["gpus"]]
    cpus = sorted({p["cpus_allowed"] for p in measured if p["cpus_allowed"]})
    mems = sorted({p["mems_allowed"] for p in measured if p["mems_allowed"]})
    gpus = sorted({g for p in measured for g in p["gpus"]})
    return {"probe": dash(meta.get("bind_probe")) or None, "policy": bind_policy_of(meta), "clean_runs": clean, "profiled": prof,
            "summary": {"processes_recorded": len(measured), "cpus_allowed": cpus, "mems_allowed": mems,
                        "gpus": gpus, "consistent": bool(measured) and len(cpus) <= 1 and len(mems) <= 1 and len(gpus) <= 1}}


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


def l3_timer(app, d):
    """The application's own timer of one Level 3 run directory, or (None, reason)."""
    try:
        return apptimers.extract(app, d), None
    except (apptimers.TimerMissing, OSError, ValueError, IndexError) as exc:
        return None, str(exc)


def l3_manifest(d):
    """run_manifest.txt of the (first) run directory harvested into <d>/app/, as a dict."""
    for p in sorted(glob.glob(os.path.join(d, "app", "*", "run_manifest.txt"))):
        return read_meta(p)
    return {}


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

    # ---- clean runs: the ROI time (Level 3: the application's own timer)
    app = meta["app"]
    runs = []
    for d in sorted(glob.glob(os.path.join(raw, "clean.*")), key=lambda p: int(p.rsplit(".", 1)[1])):
        rt = read_run_txt(d)
        if level == 3:
            timer, err = l3_timer(app, d)
            runs.append({"dir": d, "run": rt, "logs": [], "roi": None, "timer": timer, "timer_err": err})
            continue
        logs = [analysis.parse_roi_log(p) for p in sorted(glob.glob(os.path.join(d, "roi.*")))]
        roi = analysis.clean_roi(logs)
        runs.append({"dir": d, "run": rt, "logs": logs, "roi": roi})
    if level == 3:
        good = [r for r in runs if r["run"] and r["run"].get("rc") == 0 and r["timer"]]
        walls = [finite("app timer", r["timer"]["wall_s"]) for r in good]
    else:
        good = [r for r in runs if r["run"] and r["run"].get("rc") == 0 and r["roi"] and r["roi"]["entries"] > 0]
        walls = [sec(r["roi"]["wall_ns"]) for r in good]

    # warm-up runs: discarded from every statistic, recorded (when the engine logged their ROI)
    # so the cost of the first contact with the input is visible next to the clean runs
    warm = []
    for d in sorted(glob.glob(os.path.join(raw, "warmup.*")), key=lambda p: int(p.rsplit(".", 1)[1])):
        wl = [analysis.parse_roi_log(p) for p in sorted(glob.glob(os.path.join(d, "roi.*")))]
        wr = analysis.clean_roi(wl) if wl else None
        rt = read_run_txt(d)
        warm.append(sec(wr["wall_ns"]) if wr and wr["entries"] > 0 and (rt is None or rt.get("rc") == 0) else None)

    roi = {"wall_s": None, "runs_s": walls, "warmup_runs_s": warm, "wall_s_min": None, "wall_s_max": None,
           "wall_s_stddev": None, "entries": None, "excluded_s": None, "processes": None, "imbalance_s": None,
           "profiled_wall_s": None, "profiled_marker_wall_s": None, "profiler_inflation": None,
           "source": "app_timer" if level == 3 else "markers"}
    context = {"process_wall_s": None, "pre_roi_s": None, "post_roi_s": None}
    if level == 3:
        desc = apptimers.describe(app)
        roi.update({"definition": desc["definition"], "where": desc["where"], "reduction": desc["reduction"],
                    "device_sync": desc["device_sync"], "steps": None, "setup_s": None, "parts": {},
                    "evidence": []})
        if desc.get("caveat"):
            caveats.append(desc["caveat"])
    if walls and level == 3:
        t0 = good[0]["timer"]
        excl = [r["timer"]["excluded_s"] for r in good if r["timer"]["excluded_s"] is not None]
        roi.update({
            "wall_s": finite("timer median", statistics.median(walls)),
            "wall_s_min": min(walls), "wall_s_max": max(walls),
            "wall_s_stddev": finite("timer stddev", statistics.stdev(walls)) if len(walls) > 1 else 0.0,
            "excluded_s": finite("excluded", statistics.median(excl)) if excl else None,
            "steps": t0["steps"], "setup_s": t0["setup_s"], "parts": t0["parts"], "evidence": t0["files"],
        })
        pw = [(r["run"]["end_ns"] - r["run"]["start_ns"]) / 1e9 for r in good]
        context["process_wall_s"] = finite("wall", statistics.median(pw))
        context["outside_region_s"] = finite("outside", context["process_wall_s"] - roi["wall_s"])
        if len({r["timer"]["steps"] for r in good}) > 1:
            caveats.append("The number of steps inside the timed region differs between clean runs.")
    elif level == 3:
        if status == "ok":
            status = "app_timer_missing"
        errs = [r["timer_err"] for r in runs if r.get("timer_err")]
        if errs:
            caveats.append(f"The application's timer was not found in the clean run: {errs[0]}")
    elif walls:
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
    manifest = l3_manifest(runs[0]["dir"]) if (level == 3 and runs) else {}
    if manifest:
        exe_sha = exe_sha or manifest.get("binary_sha256")
        procs = [{"rank": None, "argv": None, "exe": manifest.get("binary"), "cwd": None, "host": None}]
    if not exe_sha and procs and procs[0]["exe"] and os.path.isfile(procs[0]["exe"]):
        import hashlib
        h = hashlib.sha256()
        with open(procs[0]["exe"], "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        exe_sha = h.hexdigest()
    inputs = {"declared_env": declared_env, "declared_argv": dash(meta.get("argv")),
              "processes": procs, "exe_sha256": exe_sha,
              "exe_sha256_when": "measurement" if meta.get("exe_sha256") else ("summarize" if exe_sha else None)}
    if manifest:
        inputs["run_manifest"] = {k: manifest[k] for k in sorted(manifest)
                                  if k in ("app", "case", "mode", "ranks", "profile", "backend", "input_sha256",
                                           "binary_sha256", "fingerprint_sha256", "source_tree_sha256",
                                           "threads", "steps")}
    placement = placement_block(meta, runs, os.path.join(raw, "prof"))

    # ---- registry input (a --registry case): the workload identity stored by the engine
    registry = None
    if dash(meta.get("input_id")):
        ident, ident_path = None, os.path.join(raw, "workload_identity.json")
        try:
            with open(ident_path) as f:
                ident = json.load(f)
        except (OSError, ValueError):
            pass
        registry = {"input_id": meta["input_id"], "identity": ident,
                    "identity_sha256": dash(meta.get("workload_identity_sha256")) or None,
                    "identity_complete": bool(ident and ident.get("complete")
                                              and ident.get("input_id") == meta["input_id"]
                                              and ident.get("benchmark") == meta["app"])}
        if not registry["identity_complete"]:
            caveats.append("The registry input's workload identity could not be established: this record "
                           "is not a result for that input.")
            if status == "ok":
                status = "identity_failed"

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
    ops, prof_info, ops_scope = [], {}, None
    prof_dir = os.path.join(raw, "prof")
    nvtx_roi = dash(meta.get("nvtx_roi"))
    if level == 3 and name != "none" and os.path.isdir(prof_dir) and status == "ok":
        trace = mod.open(prof_dir)
        try:
            marks = trace.named_ranges([nvtx_roi]) if (nvtx_roi and hasattr(trace, "named_ranges")) else []
            res = analysis.analyze_trace(trace, caps, markers=marks, whole_ops=True)
            prof_info = trace.info()
        finally:
            trace.close()
        whole = res["whole_process"]
        runtime = {"name": mod.RUNTIME, "roi": res["runtime_api_roi"], "whole": res["runtime_api_whole"]}
        ops, ops_scope = res["ops"], res["ops_scope"]
        prt = read_run_txt(prof_dir)
        if prt:
            context["profiled_process_wall_s"] = finite("profiled wall", (prt["end_ns"] - prt["start_ns"]) / 1e9)
            if whole and whole.get("busy_s") is not None:
                whole["busy_frac_of_process"] = finite(
                    "busy/process", whole["busy_s"] / (context["profiled_process_wall_s"] * max(whole["processes"], 1)))
        ptimer, _ = l3_timer(app, prof_dir)
        if ptimer and roi["wall_s"]:
            roi["profiled_wall_s"] = ptimer["wall_s"]
            roi["profiler_inflation"] = finite("inflation", ptimer["wall_s"] / roi["wall_s"])
        if res["profiled_roi"]["found"]:
            device = res["device"]
            if device and roi["wall_s"]:
                per = device["busy_s"] / max(device["processes"], 1)       # per process (GPU), on average
                gap = roi["wall_s"] - per
                device["host_gap_s"] = finite("host gap", gap)
                device["host_gap_frac_of_roi"] = finite("host gap frac", gap / roi["wall_s"])
                device["busy_frac_of_roi"] = finite("busy frac", per / roi["wall_s"])
            caveats.append(f"Device activity is clipped to the application's own NVTX range {nvtx_roi!r} "
                           f"({res['profiled_roi']['entries']} entries in the profiled run), which may differ "
                           f"slightly from the printed timer's region.")
        else:
            if nvtx_roi:
                caveats.append(f"The profiled run contains no NVTX range {nvtx_roi!r}: device activity is "
                               f"whole-process only.")
            caveats.append("Level 3 has no markers: the device activity and the operations table cover the "
                           "WHOLE process (start-up, set-up and output included), not only the timed region.")
        infl = roi["profiler_inflation"]
        if infl is not None and infl > INFLATION_NOTE:
            caveats.append(f"The profiler stretched the timed region {infl:.2f}x (profiled {roi['profiled_wall_s']:.4f} s "
                           f"vs clean {roi['wall_s']:.4f} s). Device-side durations are unaffected; the headline "
                           f"is the clean value.")
    elif name != "none" and os.path.isdir(prof_dir) and status == "ok":
        trace = mod.open(prof_dir)
        try:
            res = analysis.analyze_trace(trace, caps)
            prof_info = trace.info()
        finally:
            trace.close()
        ops_scope = "roi"
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
                per = device["busy_s"] / max(device.get("processes") or 1, 1)   # per process (GPU), on average
                gap = roi["wall_s"] - per
                device["host_gap_s"] = finite("host gap", gap)
                device["host_gap_frac_of_roi"] = finite("host gap frac", gap / roi["wall_s"])
                device["busy_frac_of_roi"] = finite("busy frac", per / roi["wall_s"])
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
        skipped = dash(meta.get("profile_skipped"))
        if skipped:
            caveats.append(f"Not profiled by default (tools/timing/cases/level{level}_apps.tsv, "
                           f"measure_level{level}.sh --profile-all overrides): {skipped}. Device columns are null "
                           f"(not observed), only the application's timer and the FOM were measured.")
        else:
            caveats.append("No collector for this platform: device columns are null (not observable), "
                           "only the ROI time and the FOM were measured." if level != 3 else
                           "No profiled run: device columns are null (not observable), only the application's "
                           "timer and the FOM were measured.")
    if name != "none" and not mod.VERIFIED:
        caveats.append(f"Collector {name} is interface-only and UNVERIFIED.")

    conformance = platform_conformance(platform_id) if platform_id else None
    if level > 0 and name != "none" and not (conformance and conformance.get("status") == "pass"):
        caveats.append(f"Platform {platform_id} has no passing conformance record for collector "
                       f"{name} (tools/timing/probes/conformance/run_conformance.sh).")
    if meta.get("verify_vs_roi") == "none":
        caveats.append("No numerical acceptance for this input: nothing checks its results against a reference "
                       "(the application's validate.sh checks a smaller input). The run completed with exit 0; "
                       "the time is a completeness record of a run whose results were not verified.")
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
            "region": meta.get("region") or "markers",
            "profile_skipped": dash(meta.get("profile_skipped")) or None,
            "nvtx_roi": nvtx_roi or None,
        },
        "inputs": inputs,
        "registry": registry,
        "roi": roi,
        "device": device,
        "runtime_api": runtime,
        "ops": ops,
        "ops_scope": ops_scope,
        "context": dict(context, whole_process=whole),
        "fom": fom,
        "app_timer": app_timer,
        "launcher": {"audit": audit, "audit_ok": audit_ok},
        "placement": placement,
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
    plc = (rec.get("placement") or {}).get("summary") or {}
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
        "input_id": v((rec.get("registry") or {}).get("input_id")),
        "placement_cpus_allowed": ";".join(plc.get("cpus_allowed", [])),
        "placement_mems_allowed": ";".join(plc.get("mems_allowed", [])),
        "placement_gpus": ";".join(plc.get("gpus", [])),
        "placement_consistent": "" if not plc.get("processes_recorded") else int(plc["consistent"]),
        "placement_policy": (lambda p: p.get("kind", "") + (f":PE={p['cpus_per_rank']}" if p.get("cpus_per_rank") else "")
                             + (":omp_pin" if p.get("omp_pin") == "yes" else ""))((rec.get("placement") or {}).get("policy") or {}),
        "roi_source": v(roi.get("source") or "markers"), "roi_steps": v(roi.get("steps")),
        "roi_setup_s": v(roi.get("setup_s")), "ops_scope": v(rec.get("ops_scope")),
        "whole_compute_s": v(whole.get("compute_s")), "whole_copy_h2d_s": v(whole.get("copy_h2d_s")),
        "whole_copy_d2h_s": v(whole.get("copy_d2h_s")), "whole_copy_d2d_s": v(whole.get("copy_d2d_s")),
        "whole_fill_s": v(whole.get("fill_s")),
        "whole_runtime_api_calls": v((rt.get("whole") or {}).get("calls")),
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

# A raw run shown to have measured another workload than its case/input (see
# tools/inputs/hpcperf_inputs.py `invalidation`) keeps its evidence on disk but carries an
# INVALIDATED.json: no record is built from it and an existing record of it is not loaded, so it
# reaches no CSV, summary, report or baseline selection.
INVALIDATION_FILE = "INVALIDATED.json"


def invalidated(raw):
    return os.path.isfile(os.path.join(raw, INVALIDATION_FILE))


def raw_dirs(raw_root):
    for level_dir in sorted(glob.glob(os.path.join(raw_root, "level[0-9]"))):
        for run in sorted(glob.glob(os.path.join(level_dir, "*", "*", "*"))):
            if os.path.isfile(os.path.join(run, "run_meta.txt")):
                yield run


def json_path(out_root, rec):
    return os.path.join(out_root, f"level{rec['level']}", rec["app"], rec["case"], f"{rec['run_id']}.json")


def load_records(out_root, invalid=None):
    records, skipped = [], 0
    for path in sorted(glob.glob(os.path.join(out_root, "level[0-9]", "*", "*", "*.json"))):
        rec = json.load(open(path))
        if rec.get("schema") != SCHEMA:
            skipped += 1
            continue
        raw = (rec.get("provenance") or {}).get("raw_dir")
        if raw and invalidated(os.path.join(REPO, raw)):
            if invalid is not None:
                invalid.append(path)
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


REGISTRY_CURRENT_COLS = ["level", "benchmark", "input_id", "status", "run_verification", "platform", "samples", "roi_median_s",
                         "spread", "cv", "stable", "adaptive", "run_ids", "git_commit", "correctness", "correctness_basis"]


def write_registry_current(out_root):
    """registry_current.csv: the current result of every registered input -- the same rules as the report
    (tools/timing/registry_view.py). The per-record CSVs above stay a log of every record."""
    import csv
    import registry_view
    rows, _recs, _meta, _orph = registry_view.current_view([out_root], REPO)
    path = os.path.join(out_root, "registry_current.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(REGISTRY_CURRENT_COLS)
        for r in rows:
            m = r["current"] or {}
            w.writerow([r["level"], r["benchmark"], r["input_id"], r["status"], r["run_verification"] or "",
                        m.get("platform", ""), len(m.get("samples") or []), m.get("median", ""),
                        "" if m.get("spread") is None else m["spread"], "" if m.get("cv") is None else m["cv"],
                        m.get("stable", ""), m.get("adaptive", ""), " ".join(m.get("run_ids") or []),
                        (m.get("git_commit") or "")[:10], r.get("correctness") or "", r.get("correctness_basis") or ""])
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default=os.path.join(REPO, "build", "timing"))
    ap.add_argument("--out-root", default=os.path.join(REPO, "results", "timing"))
    ap.add_argument("--csv-only", action="store_true", help="rebuild the CSVs from existing JSONs")
    ap.add_argument("--run-id", action="append", default=[], help="only these runs' raw data (repeatable)")
    ap.add_argument("--no-report", action="store_true", help="do not regenerate <out-root>/report/")
    a = ap.parse_args(argv)

    written = failed = n_invalid_raw = 0
    if not a.csv_only:
        if not os.path.isdir(a.raw_root):
            print(f"summarize: no raw directory {a.raw_root}", file=sys.stderr)
            return 1
        for raw in raw_dirs(a.raw_root):
            if a.run_id and os.path.basename(raw) not in a.run_id:
                continue
            if invalidated(raw):
                n_invalid_raw += 1
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
    invalid = []
    records, skipped = load_records(a.out_root, invalid)
    outs = write_csvs(a.out_root, records)
    by = {}
    for r in records:
        by.setdefault(r["level"], []).append(r)
    print(f"summarize: json_written={written} failed={failed} records={len(records)} skipped_old_schema={skipped}"
          f" invalidated_raw={n_invalid_raw} invalidated_records={len(invalid)}")
    for level, recs in sorted(by.items()):
        ok = sum(1 for r in recs if r["status"] == "ok")
        fom = sum(1 for r in recs if r["fom"]["status"] == "ok")
        print(f"           level{level}: {len(recs)} runs, {ok} ok, {len(recs) - ok} not ok, fom_ok={fom}")
    if any((r.get("registry") or {}).get("input_id") for r in records):
        outs.append(write_registry_current(a.out_root))
    if not a.no_report:
        outs += report.write(a.out_root, os.path.join(a.out_root, "report"))
    for p in outs:
        print(f"           {os.path.relpath(p, REPO)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
