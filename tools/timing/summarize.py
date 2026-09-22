#!/usr/bin/env python3
"""Turn measure_level1.sh / measure_level2.sh raw directories into JSON plus CSVs.

    tools/timing/summarize.py                       # scan both raw roots, write results/timing
    tools/timing/summarize.py --raw-root DIR --raw-root-l2 DIR --out-root DIR
    tools/timing/summarize.py --csv-only            # rebuild the CSVs from existing JSONs

Per run it emits results/timing/level<N>/<benchmark>/<run_id>.json (schema
hpcperf-timing-1) and then regenerates, never appends:

    results/timing/summary.csv          one row per Level 1 run, fixed column list
    results/timing/kernels.csv          one row per (Level 1 run, kernel)
    results/timing/summary_level2.csv   one row per Level 2 run
    results/timing/kernels_level2.csv   one row per (Level 2 run, kernel)

The two levels are measured differently and are kept in separate files so a reader
cannot average them by accident:

    Level 1   N clean runs for the wall clock + 1 profiled run for the breakdown,
              host-side verification skipped (HPCPERF_SKIP_VERIFY)
    Level 2   ONE profiled run of level2/<app>/run.sh, no repeats, no verification
              -> wall_s includes profiler overhead (~1.7x measured) and the
              application's own FOM is read from that same profiled run

What the numbers mean
---------------------
wall_s_*            un-instrumented runs, so no profiler overhead
gpu_kernel_time_s   sum of kernel durations (CUPTI device timestamps)
gpu_busy_s          UNION of all GPU activity intervals -- concurrent kernels are
                    counted once, unlike the naive sum (gpu_op_time_sum_s)
gpu_active_span_s   first GPU operation start .. last GPU operation end
gpu_idle_in_span_s  span - busy: the GPU was idle inside the active window,
                    i.e. the host was the bottleneck between launches
host_outside_gpu_s  wall - span: process start-up, allocation, data generation and
                    teardown, which happen before/after any GPU work

The report is produced with verification skipped by default, so these numbers do
not include the CPU reference recomputation those benchmarks perform. The JSON
records skip_verify so a reader can never confuse the two modes.
"""

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SCHEMA = "hpcperf-timing-1"

SUMMARY_COLUMNS = [
    "run_id", "utc", "benchmark", "backend", "hostname", "skip_verify", "profiled",
    "exit_code", "repeats", "verify_kind", "verify_skip_effect",
    "wall_s_median", "wall_s_min", "wall_s_max", "wall_s_stddev", "wall_s_profiled",
    "profiling_overhead_ratio",
    "gpu_kernel_time_s", "gpu_kernel_launches", "gpu_busy_s", "gpu_op_time_sum_s",
    "gpu_overlap_s", "gpu_active_span_s", "gpu_idle_in_span_s", "host_outside_gpu_s",
    "gpu_busy_frac_of_span", "gpu_busy_frac_of_wall", "host_frac_of_wall",
    "memcpy_h2d_s", "memcpy_d2h_s", "memcpy_dtod_s", "memset_s",
    "memcpy_h2d_mb", "memcpy_d2h_mb", "memcpy_dtod_mb",
    "cuda_api_time_s", "cuda_api_sync_s", "cuda_api_calls",
    "kernels_n", "top_kernel_name", "top_kernel_total_s", "top_kernel_share",
    "gpu_name", "gpu_uuid", "driver_version", "compute_cap", "sm_clock_mhz",
    "mem_clock_mhz", "power_limit_w", "gpu_temp_c",
    "cpu_model", "cpu_allowed", "loadavg_1m",
    "nsys_version", "nvcc_version", "exe_sha256", "git_commit", "git_dirty",
    "args", "raw_dir",
]

KERNEL_COLUMNS = ["run_id", "benchmark", "kernel", "count", "total_s", "avg_s", "min_s", "max_s", "share"]

# Level 2 is a different measurement, so it gets its own column list rather than
# blank cells in the Level 1 one: there are no repeats to take a median of, the wall
# clock carries profiler overhead, and there is a FOM block.
SUMMARY_L2_COLUMNS = [
    "run_id", "utc", "app", "backend", "gpus", "runner",
    "exit_code", "run_status", "gpu_audit_ok", "gpu_audit",
    "wall_s", "wall_includes_profiler", "timeout_s",
    "fom_name", "fom_value", "fom_unit", "fom_better", "fom_status",
    "fom_from_profiled_run",
    "gpu_kernel_time_s", "gpu_kernel_launches", "gpu_busy_s", "gpu_op_time_sum_s",
    "gpu_overlap_s", "gpu_active_span_s", "gpu_idle_in_span_s", "host_outside_gpu_s",
    "gpu_busy_frac_of_span", "gpu_busy_frac_of_wall", "host_frac_of_wall",
    "memcpy_h2d_s", "memcpy_d2h_s", "memcpy_dtod_s", "memset_s",
    "memcpy_h2d_mb", "memcpy_d2h_mb", "memcpy_dtod_mb",
    "cuda_api_time_s", "cuda_api_sync_s", "cuda_api_calls",
    "kernels_n", "top_kernel_name", "top_kernel_total_s", "top_kernel_share",
    "gpu_name", "gpu_uuid", "driver_version", "compute_cap", "sm_clock_mhz",
    "mem_clock_mhz", "power_limit_w", "gpu_temp_c",
    "cpu_model", "cpu_allowed", "loadavg_1m",
    "nsys_version", "nvcc_version", "git_commit", "git_dirty", "raw_dir",
]

# Measured on this node (quicksilver, HPCPERF_GPUS=1), used only in the caveat text
# so a reader of a single JSON knows the size of the effect they are looking at.
L2_PROFILER_WALL_RATIO = "1.66x-1.91x (9.29 s clean vs 15.4-17.7 s profiled)"
L2_PROFILER_FOM_EFFECT = "-5.4%..-6.4% (5.631e6 clean vs 5.271/5.327/5.329e6 profiled)"

# What verification each benchmark performs, and whether HPCPERF_SKIP_VERIFY actually
# removes it. Without this a reader would take skip_verify=true to mean "no host-side
# verification ran", which is false for the benchmarks listed as not_skippable.
VERIFY_KIND = {
    # host recomputation of the GPU workload; the switch removes it entirely
    **{b: ("cpu-recompute", "skipped", "")
       for b in ("daxpy", "del_dot_vec_2d", "energy", "fdtd_2d", "floyd_warshall",
                 "jacobi_2d", "ltimes", "mat_mat_shared", "matvec_3d_stencil", "pressure",
                 "aes", "black_scholes", "color_histogram", "fir", "pagerank",
                 "adam", "background_subtraction", "bezier_surface", "bitonic_sort",
                 "burgers_equation", "burrows_wheeler_transform", "channel_shuffle",
                 "histogram", "nbody", "backprop", "hotspot_3d", "lud", "spgemm",
                 "bilateral_filter", "all_pairs_distance")},
    # host recomputation, but part of it is fused with work the GPU needs
    "spmv": ("cpu-recompute", "partially_skipped",
             "the gold loop also writes the matrix values uploaded to the GPU and the "
             "extra matvec is GPU work, so both still run; only check_errors is skipped"),
    "murmurhash3": ("cpu-recompute", "partially_skipped",
                    "the reference hash sits in the key-generation loop the GPU consumes; "
                    "only that call is skipped"),
    "block_scan": ("cpu-reference-cheap", "partially_skipped",
                   "Initialize() fills both the GPU input and the reference; only the two "
                   "comparisons are skipped"),
    "spadd": ("cpu-recompute", "skipped", ""),
    "graph_coloring": ("cpu-reference-cheap", "skipped", ""),
    "atomic_reduction": ("cpu-reference-cheap", "skipped", ""),
    # O(1) comparisons against hardcoded reference constants: nothing to remove
    **{b: ("cpu-reference-cheap", "not_skippable",
           "verification is an O(1) comparison against hardcoded reference constants; it "
           "recomputes nothing, so it still runs and costs no measurable time")
       for b in ("cg", "ep", "ft", "mg")},
    "is": ("cpu-reference-cheap", "not_skippable",
           "verification runs inside the timed ranking kernels (rank_gpu_kernel_7) and in "
           "three further CUDA kernels; it still runs and is included in the GPU time"),
    "binary_search": ("none", "not_applicable",
                      "its check is behind #ifdef DEBUG, which is never defined"),
    # verification lives in a repo-authored verify.py the harness does not run
    **{b: ("external-python", "not_executed",
           "verification lives in verify.py, which the harness does not run: the binary is "
           "measured directly")
       for b in ("ao_bench", "bfs", "gaussian_elimination", "hotspot", "nearest_neighbor",
                 "needleman_wunsch", "pathfinder", "srad_v1")},
}

SYNC_API = re.compile(r"^cuda(Device|Stream|Event)Synchronize$|^cudaMemcpy$|^cudaMemset$")


# ----------------------------------------------------------------- helpers

def finite(name, value):
    """Every number that reaches the JSON goes through this."""
    if value is None:
        return None
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name}: not a finite number ({value!r})")
    return v


def read_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            meta[k] = v
    return meta


def read_csv_rows(path):
    """nsys stats CSV -> list of dicts; missing file -> None (not an empty list)."""
    if not os.path.isfile(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def need(row, column, path):
    if column not in row:
        raise KeyError(f"{path}: expected column {column!r}, got {list(row)}")
    return row[column]


def merge_intervals(intervals):
    """Union length and span of [start, end) intervals, in the input's units.

    Sorting then sweeping is the whole point: concurrent GPU operations overlap,
    so summing durations double-counts them. Returns (union, span, naive_sum).
    """
    if not intervals:
        return 0.0, 0.0, 0.0
    ordered = sorted(intervals)
    naive = sum(e - s for s, e in ordered)
    union = 0.0
    cur_s, cur_e = ordered[0]
    for s, e in ordered[1:]:
        if s > cur_e:                 # disjoint: close the open interval
            union += cur_e - cur_s
            cur_s, cur_e = s, e
        elif e > cur_e:               # overlapping or adjacent: extend
            cur_e = e
    union += cur_e - cur_s
    span = ordered[-1][1] - ordered[0][0]
    span = max(span, max(e for _, e in ordered) - ordered[0][0])
    return union, span, naive


# ----------------------------------------------------------------- nsys parsing

def parse_kernels(nsys_dir):
    rows = read_csv_rows(os.path.join(nsys_dir, "rep_cuda_gpu_kern_sum.csv"))
    if rows is None:
        return None
    out = []
    p = os.path.join(nsys_dir, "rep_cuda_gpu_kern_sum.csv")
    for r in rows:
        out.append({
            "name": need(r, "Name", p),
            "count": int(float(need(r, "Instances", p))),
            "total_s": finite("kernel total", float(need(r, "Total Time (ns)", p)) / 1e9),
            "avg_s": finite("kernel avg", float(need(r, "Avg (ns)", p)) / 1e9),
            "min_s": finite("kernel min", float(need(r, "Min (ns)", p)) / 1e9),
            "max_s": finite("kernel max", float(need(r, "Max (ns)", p)) / 1e9),
        })
    out.sort(key=lambda k: k["total_s"], reverse=True)
    return out


def parse_memops(nsys_dir):
    """memcpy/memset time and size, keyed by direction taken from the Operation text."""
    res = {k: 0.0 for k in ("h2d_s", "d2h_s", "dtod_s", "memset_s", "other_s")}
    res.update({k: 0.0 for k in ("h2d_mb", "d2h_mb", "dtod_mb")})
    res["counts"] = {}

    def direction(op):
        o = op.lower()
        if "host-to-device" in o:
            return "h2d"
        if "device-to-host" in o:
            return "d2h"
        if "device-to-device" in o:
            return "dtod"
        if "memset" in o:
            return "memset"
        return "other"

    p = os.path.join(nsys_dir, "rep_cuda_gpu_mem_time_sum.csv")
    rows = read_csv_rows(p)
    if rows is not None:
        for r in rows:
            d = direction(need(r, "Operation", p))
            secs = float(need(r, "Total Time (ns)", p)) / 1e9
            key = "memset_s" if d == "memset" else f"{d}_s"
            res[key] = finite(key, res.get(key, 0.0) + secs)
            res["counts"][d] = res["counts"].get(d, 0) + int(float(need(r, "Count", p)))

    p = os.path.join(nsys_dir, "rep_cuda_gpu_mem_size_sum.csv")
    rows = read_csv_rows(p)
    if rows is not None:
        for r in rows:
            d = direction(need(r, "Operation", p))
            if d in ("h2d", "d2h", "dtod"):
                res[f"{d}_mb"] = finite("mem size", res[f"{d}_mb"] + float(need(r, "Total (MB)", p)))
    return res


def parse_api(nsys_dir):
    p = os.path.join(nsys_dir, "rep_cuda_api_sum.csv")
    rows = read_csv_rows(p)
    if rows is None:
        return None
    total = sync = 0.0
    calls = 0
    for r in rows:
        secs = float(need(r, "Total Time (ns)", p)) / 1e9
        total += secs
        calls += int(float(need(r, "Num Calls", p)))
        if SYNC_API.match(need(r, "Name", p).strip()):
            sync += secs
    return {"total_s": finite("api total", total), "sync_s": finite("api sync", sync), "calls": calls}


def parse_trace(nsys_dir):
    """GPU busy (union), span and naive sum from the per-operation trace."""
    p = os.path.join(nsys_dir, "rep_cuda_gpu_trace.csv")
    rows = read_csv_rows(p)
    if rows is None:
        return None
    intervals = []
    for r in rows:
        start = float(need(r, "Start (ns)", p))
        dur = float(need(r, "Duration (ns)", p))
        intervals.append((start, start + dur))
    union, span, naive = merge_intervals(intervals)
    return {
        "busy_s": finite("gpu busy", union / 1e9),
        "span_s": finite("gpu span", span / 1e9),
        "op_sum_s": finite("gpu op sum", naive / 1e9),
        "ops": len(intervals),
    }


# ----------------------------------------------------------------- FOM (Level 2)

def extract_fom(raw_dir, meta):
    """Read the application's own figure of merit out of the run's stdout.

    16 of the 24 Level 2 applications print one; the other 8 do not and get
    value=None with status="none" -- a blank cell, never a fabricated number. The
    wording, unit and position differ per application, so the pattern comes from
    the case table (one capture group, last match wins) and never from a guess.
    """
    name = (meta.get("fom_name") or "-").strip()
    unit = (meta.get("fom_unit") or "-").strip()
    better = (meta.get("fom_better") or "-").strip()
    source = (meta.get("fom_source") or "-").strip()
    regex = meta.get("fom_regex") or ""
    note = (meta.get("fom_note") or "").strip()
    out = {
        "name": None if name == "-" else name,
        "value": None,
        "unit": None if unit == "-" else unit,
        "better": None if better == "-" else better,
        "source": None if source == "-" else source,
        "regex": regex or None,
        "note": "" if note == "-" else note,
        "from_profiled_run": meta.get("profiled") == "1",
        "status": "none",
    }
    if name == "-":
        return out
    if source != "stdout":
        raise RuntimeError(
            f"{raw_dir}: fom_source={source!r} is not implemented; no case needs a "
            f"file-sourced FOM yet, so add it deliberately rather than by accident")

    pat = re.compile(regex, re.M)
    if pat.groups != 1:
        raise RuntimeError(f"{raw_dir}: fom_regex must have exactly one capture group, "
                           f"got {pat.groups}: {regex!r}")

    log = os.path.join(raw_dir, "run.log")
    if not os.path.isfile(log):
        out["status"] = "log_missing"
        return out
    with open(log, errors="replace") as f:
        matches = pat.findall(f.read())
    if not matches:
        out["status"] = "not_matched"
        return out
    # Thousands separators appear in xsbench's Lookups/s.
    out["value"] = finite("fom", float(matches[-1].replace(",", "")))
    out["status"] = "ok"
    return out


# ----------------------------------------------------------------- shared blocks

def env_block(meta):
    gpu = [g.strip() for g in meta.get("gpu_csv", "").split(",")]

    def g(i):
        return gpu[i] if len(gpu) > i and gpu[i] not in ("", "[N/A]") else None

    return {
        "hostname": meta.get("hostname"),
        "kernel": meta.get("kernel"),
        "cpu_model": meta.get("cpu_model"),
        "cpu_allowed": int(meta["cpu_allowed"]) if meta.get("cpu_allowed") else None,
        "loadavg_1m": finite("loadavg", meta["loadavg_1m"]) if meta.get("loadavg_1m") else None,
        "mem_available_kb": int(meta["mem_available_kb"]) if meta.get("mem_available_kb") else None,
        "nsys_version": meta.get("nsys_version"),
        "nvcc_version": meta.get("nvcc_version"),
        "gpu": {
            "index": g(0), "name": g(1), "uuid": g(2), "driver_version": g(3),
            "compute_cap": g(4), "memory_total_mib": g(5),
            "clocks_max_sm_mhz": g(6), "clocks_max_mem_mhz": g(7),
            "clocks_sm_mhz": g(8), "clocks_mem_mhz": g(9),
            "power_limit_w": g(10), "temperature_c": g(11), "persistence_mode": g(12),
        },
    }


def gpu_blocks(raw_dir, meta, wall_s):
    """The nsys-derived blocks, shared by both levels. Returns (dict, caveats)."""
    nsys_dir = os.path.join(raw_dir, "nsys")
    if meta.get("nsys_status") != "ok" or not os.path.isdir(nsys_dir):
        caveats = []
        if meta.get("profiled") == "1":
            caveats.append(f"nsys did not produce reports (nsys_status={meta.get('nsys_status')})")
        return None, caveats

    kernels = parse_kernels(nsys_dir)
    memops = parse_memops(nsys_dir)
    api = parse_api(nsys_dir)
    trace = parse_trace(nsys_dir)
    if kernels is None or trace is None:
        raise RuntimeError(f"{raw_dir}: nsys_status=ok but reports are missing")

    kern_total = sum(k["total_s"] for k in kernels)
    launches = sum(k["count"] for k in kernels)
    span, busy = trace["span_s"], trace["busy_s"]
    return {
        "gpu": {
            "kernel_time_s": finite("kernel time", kern_total),
            "kernel_launches": launches,
            "busy_s": busy,
            "op_time_sum_s": trace["op_sum_s"],
            "overlap_s": finite("overlap", trace["op_sum_s"] - busy),
            "active_span_s": span,
            "idle_in_span_s": finite("gpu idle", span - busy),
            "operations": trace["ops"],
            "busy_frac_of_span": finite("busy/span", busy / span) if span > 0 else None,
            "busy_frac_of_wall": finite("busy/wall", busy / wall_s) if wall_s > 0 else None,
        },
        "host": {
            "outside_gpu_s": finite("host outside", wall_s - span),
            "frac_of_wall": finite("host/wall", (wall_s - span) / wall_s) if wall_s > 0 else None,
        },
        "memops": memops,
        "cuda_api": api,
        "kernels": kernels,
        "profile": {
            "tool": "nsys",
            "version": meta.get("nsys_version"),
            "trace": "cuda",
            "sample": "process-tree",
            "report": os.path.relpath(
                os.path.join(nsys_dir, f"{meta.get('benchmark')}.nsys-rep"), REPO),
        },
    }, []


# ----------------------------------------------------------------- one Level 2 run

def build_record_l2(raw_dir):
    meta_path = os.path.join(raw_dir, "run_meta.txt")
    if not os.path.isfile(meta_path):
        return None
    meta = read_meta(meta_path)
    if meta.get("level") != "2":
        return None

    wall_path = os.path.join(raw_dir, "wall_ns.txt")
    if not os.path.isfile(wall_path):
        return None
    txt = open(wall_path).read().strip()
    if not txt:
        return None
    wall = int(txt.split()[0]) / 1e9

    code = None
    cpath = os.path.join(raw_dir, "exit_codes.txt")
    if os.path.isfile(cpath):
        got = open(cpath).read().split()
        if got:
            code = int(got[0])

    audit = meta.get("gpu_audit", "")
    profiled = meta.get("profiled") == "1"

    rec = {
        "schema": SCHEMA,
        "level": 2,
        "run_id": meta.get("run_id"),
        "utc": meta.get("utc"),
        "benchmark": meta.get("benchmark"),
        "backend": meta.get("backend", "CUDA"),
        "measurement": {
            "tool": "tools/timing/measure_level2.sh",
            "runner": meta.get("runner"),
            "gpus": meta.get("gpus"),
            "repeats": 1,
            "warmup": 0,
            "profiled": profiled,
            "profiler_in_wall": meta.get("profiler_in_wall") == "1",
            "exit_code": code,
            "run_status": meta.get("run_status"),
            "timeout_s": meta.get("timeout_s"),
            "verify": "not_run",
            "verify_kind": "external-script",
            "verify_note": "Level 2 keeps correctness checks in validate.sh, which this "
                           "tool never calls; run.sh contains none, so no source change "
                           "was needed and nothing was skipped inside the application.",
            "env_script": meta.get("env_script"),
            "env_allow": meta.get("env_allow", "").split(),
            "env_deny_regex": meta.get("env_deny_regex"),
            "env_denied": meta.get("env_denied", "").split(),
        },
        "launcher": {
            "gpu_audit": audit or None,
            "audit_ok": bool(re.search(r"\b0 mismatch, 0 unverified\b", audit)) if audit else None,
        },
        "timing": {
            "wall_s": finite("wall", wall),
            "wall_includes_profiler": profiled,
        },
        "fom": extract_fom(raw_dir, meta),
        "env": env_block(meta),
        "provenance": {
            "git_commit": meta.get("git_commit"),
            "git_dirty": meta.get("git_dirty") == "1",
            "raw_dir": os.path.relpath(raw_dir, REPO),
        },
        "caveats": [],
    }

    blocks, caveats = gpu_blocks(raw_dir, meta, wall)
    rec["caveats"].extend(caveats)
    if blocks:
        rec.update(blocks)
        rec["host"]["note"] = (
            "wall clock minus the GPU active span: process start-up, MPI_Init, input "
            "setup and teardown -- and, because this is a single profiled run, the "
            "profiler's own overhead. It is an upper bound on host time.")
        if rec["host"]["outside_gpu_s"] < 0:
            rec["caveats"].append(
                "GPU active span exceeds the wall clock; host_outside_gpu_s is not "
                "meaningful for this run.")
    else:
        rec["gpu"] = None

    if profiled:
        rec["caveats"].append(
            f"Single profiled run, no clean baseline: wall_s includes nsys overhead, "
            f"measured at {L2_PROFILER_WALL_RATIO} on this node. GPU-side numbers are "
            f"unaffected (CUPTI timestamps on the device; kernel total moved 1.2% across "
            f"sampling settings), host and wall numbers are upper bounds.")
        if rec["fom"]["status"] == "ok":
            rec["caveats"].append(
                f"fom_value comes from the profiled run, where the profiler depresses it: "
                f"{L2_PROFILER_FOM_EFFECT}. Do not compare it against a published "
                f"unprofiled FOM without that correction.")
    if rec["fom"]["status"] == "not_matched":
        rec["caveats"].append(
            f"The case table expects a FOM named {rec['fom']['name']!r} but its pattern "
            f"did not match this run's stdout; the application's output format may have "
            f"changed. fom_value is left empty rather than guessed.")
    if rec["launcher"]["audit_ok"] is False:
        rec["caveats"].append(
            f"The launcher's GPU-binding audit is not clean: {audit!r}. The run may not "
            f"have used the GPUs it asked for.")
    return rec


# ----------------------------------------------------------------- one run

def build_record(raw_dir):
    meta_path = os.path.join(raw_dir, "run_meta.txt")
    if not os.path.isfile(meta_path):
        return None
    meta = read_meta(meta_path)
    if meta.get("level") == "2":
        return None                      # measure_level2.sh output, see build_record_l2

    wall_path = os.path.join(raw_dir, "wall_ns.txt")
    if not os.path.isfile(wall_path):
        return None
    walls = [int(x) / 1e9 for x in open(wall_path).read().split() if x.strip()]
    if not walls:
        return None
    codes = []
    if os.path.isfile(os.path.join(raw_dir, "exit_codes.txt")):
        codes = [int(x) for x in open(os.path.join(raw_dir, "exit_codes.txt")).read().split()]

    wall_prof = None
    wp = os.path.join(raw_dir, "wall_ns_profiled.txt")
    if os.path.isfile(wp):
        txt = open(wp).read().strip()
        if txt:
            wall_prof = int(txt) / 1e9

    rec = {
        "schema": SCHEMA,
        "run_id": meta.get("run_id"),
        "utc": meta.get("utc"),
        "benchmark": meta.get("benchmark"),
        "backend": meta.get("backend", "CUDA"),
        "measurement": {
            "skip_verify": meta.get("skip_verify") == "1",
            "profiled": meta.get("profiled") == "1",
            "repeats": int(meta.get("repeats", len(walls))),
            "warmup": int(meta.get("warmup", 0)),
            "exit_code": max(codes) if codes else None,
            "env_allow": meta.get("env_allow", "").split(),
            "tool": "tools/timing/measure_level1.sh",
        },
        "command": {
            "exe": meta.get("exe"),
            "args": meta.get("args", ""),
            "cwd": meta.get("cwd"),
            "ctest_wrapper": meta.get("ctest_wrapper"),
            "ctest_timeout_s": meta.get("ctest_timeout_s"),
            "exe_sha256": meta.get("exe_sha256"),
            "exe_bytes": int(meta["exe_bytes"]) if meta.get("exe_bytes") else None,
        },
        "timing": {
            "wall_s": [finite("wall", w) for w in walls],
            "wall_s_median": finite("wall median", statistics.median(walls)),
            "wall_s_min": finite("wall min", min(walls)),
            "wall_s_max": finite("wall max", max(walls)),
            "wall_s_stddev": finite("wall stddev", statistics.stdev(walls) if len(walls) > 1 else 0.0),
            "wall_s_profiled": finite("wall profiled", wall_prof) if wall_prof else None,
        },
        "env": env_block(meta),
        "provenance": {
            "git_commit": meta.get("git_commit"),
            "git_dirty": meta.get("git_dirty") == "1",
            "raw_dir": os.path.relpath(raw_dir, REPO),
        },
        "caveats": [],
    }

    nsys_dir = os.path.join(raw_dir, "nsys")
    if meta.get("nsys_status") == "ok" and os.path.isdir(nsys_dir):
        kernels = parse_kernels(nsys_dir)
        memops = parse_memops(nsys_dir)
        api = parse_api(nsys_dir)
        trace = parse_trace(nsys_dir)
        if kernels is None or trace is None:
            raise RuntimeError(f"{raw_dir}: nsys_status=ok but reports are missing")

        kern_total = sum(k["total_s"] for k in kernels)
        launches = sum(k["count"] for k in kernels)
        wall_med = rec["timing"]["wall_s_median"]
        span, busy = trace["span_s"], trace["busy_s"]

        rec["gpu"] = {
            "kernel_time_s": finite("kernel time", kern_total),
            "kernel_launches": launches,
            "busy_s": busy,
            "op_time_sum_s": trace["op_sum_s"],
            "overlap_s": finite("overlap", trace["op_sum_s"] - busy),
            "active_span_s": span,
            "idle_in_span_s": finite("gpu idle", span - busy),
            "operations": trace["ops"],
            "busy_frac_of_span": finite("busy/span", busy / span) if span > 0 else None,
            "busy_frac_of_wall": finite("busy/wall", busy / wall_med) if wall_med > 0 else None,
        }
        rec["host"] = {
            "outside_gpu_s": finite("host outside", wall_med - span),
            "frac_of_wall": finite("host/wall", (wall_med - span) / wall_med) if wall_med > 0 else None,
            "note": "wall clock minus the GPU active span: start-up, allocation, data "
                    "generation and teardown. Measured without the profiler attached.",
        }
        rec["memops"] = memops
        rec["cuda_api"] = api
        rec["kernels"] = kernels
        rec["profile"] = {
            "tool": "nsys",
            "version": meta.get("nsys_version"),
            "trace": "cuda",
            "sample": "process-tree",
            "report": os.path.relpath(os.path.join(nsys_dir, f"{meta.get('benchmark')}.nsys-rep"), REPO),
        }
        if wall_prof and wall_med > 0:
            rec["timing"]["profiling_overhead_ratio"] = finite("overhead", wall_prof / wall_med)
        if rec["host"]["outside_gpu_s"] < 0:
            rec["caveats"].append(
                "GPU active span exceeds the median clean wall clock; the profiled run "
                "was slower than the clean runs, so host_outside_gpu_s is not meaningful.")
    else:
        rec["gpu"] = None
        if meta.get("profiled") == "1":
            rec["caveats"].append(f"nsys did not produce reports (nsys_status={meta.get('nsys_status')})")

    kind, effect, note = VERIFY_KIND.get(rec["benchmark"], ("unknown", "unknown", ""))
    rec["measurement"]["verify_kind"] = kind
    rec["measurement"]["verify_skip_effect"] = effect if rec["measurement"]["skip_verify"] else "not_requested"
    rec["measurement"]["verify_note"] = note

    if rec["measurement"]["skip_verify"] and effect in ("not_skippable", "partially_skipped"):
        rec["caveats"].append(
            f"HPCPERF_SKIP_VERIFY was set but verification is {effect.replace('_', ' ')} here: "
            f"{note}. Do not read host_outside_gpu_s as verification-free.")
    if kind == "unknown":
        rec["caveats"].append(
            "This benchmark is not in VERIFY_KIND (tools/timing/summarize.py); what the "
            "measured time includes on the host side is undocumented.")
    if not rec["measurement"]["skip_verify"]:
        rec["caveats"].append(
            "Verification was NOT skipped: the wall clock includes the benchmark's CPU "
            "reference recomputation and is not comparable to a skip-verify run.")
    return rec


# ----------------------------------------------------------------- aggregation

def flatten(rec):
    t, e, p = rec["timing"], rec["env"], rec["provenance"]
    gpu, host = rec.get("gpu"), rec.get("host")
    mem, api = rec.get("memops") or {}, rec.get("cuda_api") or {}
    kernels = rec.get("kernels") or []
    top = kernels[0] if kernels else None
    kern_total = gpu["kernel_time_s"] if gpu else 0.0
    row = {c: "" for c in SUMMARY_COLUMNS}
    row.update({
        "run_id": rec["run_id"], "utc": rec["utc"], "benchmark": rec["benchmark"],
        "backend": rec["backend"], "hostname": e["hostname"],
        "skip_verify": int(rec["measurement"]["skip_verify"]),
        "profiled": int(rec["measurement"]["profiled"]),
        "verify_kind": rec["measurement"].get("verify_kind", ""),
        "verify_skip_effect": rec["measurement"].get("verify_skip_effect", ""),
        "exit_code": rec["measurement"]["exit_code"], "repeats": rec["measurement"]["repeats"],
        "wall_s_median": t["wall_s_median"], "wall_s_min": t["wall_s_min"],
        "wall_s_max": t["wall_s_max"], "wall_s_stddev": t["wall_s_stddev"],
        "wall_s_profiled": t.get("wall_s_profiled") or "",
        "profiling_overhead_ratio": t.get("profiling_overhead_ratio") or "",
        "cpu_model": e["cpu_model"], "cpu_allowed": e["cpu_allowed"], "loadavg_1m": e["loadavg_1m"],
        "nsys_version": e["nsys_version"], "nvcc_version": e["nvcc_version"],
        "gpu_name": e["gpu"]["name"], "gpu_uuid": e["gpu"]["uuid"],
        "driver_version": e["gpu"]["driver_version"], "compute_cap": e["gpu"]["compute_cap"],
        "sm_clock_mhz": e["gpu"]["clocks_sm_mhz"], "mem_clock_mhz": e["gpu"]["clocks_mem_mhz"],
        "power_limit_w": e["gpu"]["power_limit_w"], "gpu_temp_c": e["gpu"]["temperature_c"],
        "exe_sha256": rec["command"]["exe_sha256"], "args": rec["command"]["args"],
        "git_commit": p["git_commit"], "git_dirty": int(p["git_dirty"]), "raw_dir": p["raw_dir"],
    })
    if gpu:
        row.update({
            "gpu_kernel_time_s": gpu["kernel_time_s"], "gpu_kernel_launches": gpu["kernel_launches"],
            "gpu_busy_s": gpu["busy_s"], "gpu_op_time_sum_s": gpu["op_time_sum_s"],
            "gpu_overlap_s": gpu["overlap_s"], "gpu_active_span_s": gpu["active_span_s"],
            "gpu_idle_in_span_s": gpu["idle_in_span_s"],
            "gpu_busy_frac_of_span": gpu["busy_frac_of_span"],
            "gpu_busy_frac_of_wall": gpu["busy_frac_of_wall"],
            "host_outside_gpu_s": host["outside_gpu_s"], "host_frac_of_wall": host["frac_of_wall"],
            "memcpy_h2d_s": mem.get("h2d_s"), "memcpy_d2h_s": mem.get("d2h_s"),
            "memcpy_dtod_s": mem.get("dtod_s"), "memset_s": mem.get("memset_s"),
            "memcpy_h2d_mb": mem.get("h2d_mb"), "memcpy_d2h_mb": mem.get("d2h_mb"),
            "memcpy_dtod_mb": mem.get("dtod_mb"),
            "cuda_api_time_s": api.get("total_s"), "cuda_api_sync_s": api.get("sync_s"),
            "cuda_api_calls": api.get("calls"),
            "kernels_n": len(kernels),
            "top_kernel_name": top["name"] if top else "",
            "top_kernel_total_s": top["total_s"] if top else "",
            "top_kernel_share": (top["total_s"] / kern_total) if (top and kern_total > 0) else "",
        })
    return row


def flatten_l2(rec):
    t, e, p = rec["timing"], rec["env"], rec["provenance"]
    m, fom = rec["measurement"], rec["fom"]
    gpu, host = rec.get("gpu"), rec.get("host")
    mem, api = rec.get("memops") or {}, rec.get("cuda_api") or {}
    kernels = rec.get("kernels") or []
    top = kernels[0] if kernels else None
    kern_total = gpu["kernel_time_s"] if gpu else 0.0
    audit_ok = rec["launcher"]["audit_ok"]
    row = {c: "" for c in SUMMARY_L2_COLUMNS}
    row.update({
        "run_id": rec["run_id"], "utc": rec["utc"], "app": rec["benchmark"],
        "backend": rec["backend"], "gpus": m.get("gpus"), "runner": m.get("runner"),
        "exit_code": m["exit_code"], "run_status": m.get("run_status") or "",
        "gpu_audit_ok": "" if audit_ok is None else int(audit_ok),
        "gpu_audit": rec["launcher"]["gpu_audit"] or "",
        "wall_s": t["wall_s"], "wall_includes_profiler": int(t["wall_includes_profiler"]),
        "timeout_s": m.get("timeout_s") or "",
        # A blank fom_value is the whole point for the 8 applications without one.
        "fom_name": fom["name"] or "", "fom_value": "" if fom["value"] is None else fom["value"],
        "fom_unit": fom["unit"] or "", "fom_better": fom["better"] or "",
        "fom_status": fom["status"], "fom_from_profiled_run": int(fom["from_profiled_run"]),
        "cpu_model": e["cpu_model"], "cpu_allowed": e["cpu_allowed"], "loadavg_1m": e["loadavg_1m"],
        "nsys_version": e["nsys_version"], "nvcc_version": e["nvcc_version"],
        "gpu_name": e["gpu"]["name"], "gpu_uuid": e["gpu"]["uuid"],
        "driver_version": e["gpu"]["driver_version"], "compute_cap": e["gpu"]["compute_cap"],
        "sm_clock_mhz": e["gpu"]["clocks_sm_mhz"], "mem_clock_mhz": e["gpu"]["clocks_mem_mhz"],
        "power_limit_w": e["gpu"]["power_limit_w"], "gpu_temp_c": e["gpu"]["temperature_c"],
        "git_commit": p["git_commit"], "git_dirty": int(p["git_dirty"]), "raw_dir": p["raw_dir"],
    })
    if gpu:
        row.update({
            "gpu_kernel_time_s": gpu["kernel_time_s"], "gpu_kernel_launches": gpu["kernel_launches"],
            "gpu_busy_s": gpu["busy_s"], "gpu_op_time_sum_s": gpu["op_time_sum_s"],
            "gpu_overlap_s": gpu["overlap_s"], "gpu_active_span_s": gpu["active_span_s"],
            "gpu_idle_in_span_s": gpu["idle_in_span_s"],
            "gpu_busy_frac_of_span": gpu["busy_frac_of_span"],
            "gpu_busy_frac_of_wall": gpu["busy_frac_of_wall"],
            "host_outside_gpu_s": host["outside_gpu_s"], "host_frac_of_wall": host["frac_of_wall"],
            "memcpy_h2d_s": mem.get("h2d_s"), "memcpy_d2h_s": mem.get("d2h_s"),
            "memcpy_dtod_s": mem.get("dtod_s"), "memset_s": mem.get("memset_s"),
            "memcpy_h2d_mb": mem.get("h2d_mb"), "memcpy_d2h_mb": mem.get("d2h_mb"),
            "memcpy_dtod_mb": mem.get("dtod_mb"),
            "cuda_api_time_s": api.get("total_s"), "cuda_api_sync_s": api.get("sync_s"),
            "cuda_api_calls": api.get("calls"),
            "kernels_n": len(kernels),
            "top_kernel_name": top["name"] if top else "",
            "top_kernel_total_s": top["total_s"] if top else "",
            "top_kernel_share": (top["total_s"] / kern_total) if (top and kern_total > 0) else "",
        })
    return row


def scan_raw(raw_root, builder, json_root):
    """Build one JSON per raw run directory. Returns (written, failed)."""
    written = failed = 0
    if not os.path.isdir(raw_root):
        return 0, 0
    for bm in sorted(os.listdir(raw_root)):
        bm_dir = os.path.join(raw_root, bm)
        if not os.path.isdir(bm_dir):
            continue
        for run in sorted(os.listdir(bm_dir)):
            raw = os.path.join(bm_dir, run)
            if not os.path.isdir(raw):
                continue
            try:
                rec = builder(raw)
            except Exception as exc:  # noqa: BLE001 - report and keep going
                print(f"summarize: FAILED {raw}: {exc}", file=sys.stderr)
                failed += 1
                continue
            if rec is None:
                continue
            dest_dir = os.path.join(json_root, rec["benchmark"])
            os.makedirs(dest_dir, exist_ok=True)
            with open(os.path.join(dest_dir, f"{rec['run_id']}.json"), "w") as f:
                json.dump(rec, f, indent=2, sort_keys=False)
                f.write("\n")
            written += 1
    return written, failed


def load_records(json_root):
    records, skipped = [], 0
    for dirpath, _dirs, files in os.walk(json_root):
        for name in sorted(files):
            if not name.endswith(".json"):
                continue
            rec = json.load(open(os.path.join(dirpath, name)))
            if rec.get("schema") != SCHEMA:
                print(f"summarize: refusing {name}: schema {rec.get('schema')!r}", file=sys.stderr)
                skipped += 1
                continue
            records.append(rec)
    records.sort(key=lambda r: (r["benchmark"], r["run_id"]))
    return records, skipped


def write_kernels_csv(path, records):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KERNEL_COLUMNS)
        w.writeheader()
        for rec in records:
            kernels = rec.get("kernels") or []
            total = sum(k["total_s"] for k in kernels) or 1.0
            for k in kernels:
                w.writerow({"run_id": rec["run_id"], "benchmark": rec["benchmark"],
                            "kernel": k["name"], "count": k["count"], "total_s": k["total_s"],
                            "avg_s": k["avg_s"], "min_s": k["min_s"], "max_s": k["max_s"],
                            "share": k["total_s"] / total})


def write_summary_csv(path, columns, records, flattener):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            w.writerow(flattener(rec))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default=os.path.join(REPO, "build", "timing"),
                    help="Level 1 raw directories (measure_level1.sh)")
    ap.add_argument("--raw-root-l2", default=os.path.join(REPO, "build", "timing-l2"),
                    help="Level 2 raw directories (measure_level2.sh)")
    ap.add_argument("--out-root", default=os.path.join(REPO, "results", "timing"))
    ap.add_argument("--csv-only", action="store_true", help="do not re-read raw dirs; rebuild CSVs from JSONs")
    args = ap.parse_args()

    json_root = os.path.join(args.out_root, "level1")
    json_root_l2 = os.path.join(args.out_root, "level2")
    written = failed = written2 = failed2 = 0

    if not args.csv_only:
        if not os.path.isdir(args.raw_root) and not os.path.isdir(args.raw_root_l2):
            print(f"summarize: no raw directory {args.raw_root} or {args.raw_root_l2}", file=sys.stderr)
            return 1
        written, failed = scan_raw(args.raw_root, build_record, json_root)
        written2, failed2 = scan_raw(args.raw_root_l2, build_record_l2, json_root_l2)

    os.makedirs(args.out_root, exist_ok=True)
    records, skipped = load_records(json_root)
    summary = os.path.join(args.out_root, "summary.csv")
    kern_csv = os.path.join(args.out_root, "kernels.csv")
    write_summary_csv(summary, SUMMARY_COLUMNS, records, flatten)
    write_kernels_csv(kern_csv, records)

    records2, skipped2 = load_records(json_root_l2)
    summary2 = os.path.join(args.out_root, "summary_level2.csv")
    kern_csv2 = os.path.join(args.out_root, "kernels_level2.csv")
    write_summary_csv(summary2, SUMMARY_L2_COLUMNS, records2, flatten_l2)
    write_kernels_csv(kern_csv2, records2)

    with_fom = sum(1 for r in records2 if r["fom"]["status"] == "ok")
    print(f"summarize: level1 json_written={written} failed={failed} "
          f"rows={len(records)} skipped={skipped}")
    print(f"           {os.path.relpath(summary, REPO)}  {os.path.relpath(kern_csv, REPO)}")
    print(f"summarize: level2 json_written={written2} failed={failed2} "
          f"rows={len(records2)} skipped={skipped2} fom_ok={with_fom} "
          f"fom_blank={len(records2) - with_fom}")
    print(f"           {os.path.relpath(summary2, REPO)}  {os.path.relpath(kern_csv2, REPO)}")
    return 1 if (failed or failed2) else 0


if __name__ == "__main__":
    sys.exit(main())
