"""Vendor-neutral analysis for tools/timing.

Everything here works on the canonical model (collectors/__init__.py) and on the ROI
logs written by roi/hpcperf_roi.h and roi/hpcperf_roi.py. Nothing here knows about a
vendor, a profiler or a file format of one.

Two inputs, two questions:
  clean runs      ROI logs only, no profiler  -> how long the computation takes
  profiled run    a Trace from a collector    -> what the device did inside the ROI
"""

import bisect
import json
import math

from collectors import CATEGORIES, COPY_CATEGORIES

ROI_LOG_VERSION = 2


# ----------------------------------------------------------------- helpers

def finite(name, value):
    """Every float that reaches a record goes through this."""
    if value is None:
        return None
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name}: not a finite number ({value!r})")
    return v


def merge(ranges):
    """Union of [start, end) ranges -> sorted, disjoint list."""
    out = []
    for s, e in sorted(ranges):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def subtract(ranges, holes):
    """Sorted disjoint `ranges` minus sorted disjoint `holes` -> sorted disjoint list."""
    out = []
    j = 0
    for s, e in ranges:
        cur = s
        while j < len(holes) and holes[j][1] <= cur:
            j += 1
        k = j
        while k < len(holes) and holes[k][0] < e:
            hs, he = holes[k]
            if hs > cur:
                out.append((cur, hs))
            cur = max(cur, he)
            k += 1
        if cur < e:
            out.append((cur, e))
    return out


def overlap_len(ranges, holes):
    """Total length of the intersection of two sorted disjoint lists."""
    total, j = 0, 0
    for s, e in ranges:
        while j < len(holes) and holes[j][1] <= s:
            j += 1
        k = j
        while k < len(holes) and holes[k][0] < e:
            total += max(0, min(e, holes[k][1]) - max(s, holes[k][0]))
            k += 1
    return total


class _Union:
    """Streaming union length of intervals fed in non-decreasing start order."""
    __slots__ = ("cs", "ce", "total")

    def __init__(self):
        self.cs = self.ce = None
        self.total = 0

    def add(self, s, e):
        if self.ce is None:
            self.cs, self.ce = s, e
        elif s > self.ce:
            self.total += self.ce - self.cs
            self.cs, self.ce = s, e
        elif e > self.ce:
            self.ce = e

    def length(self):
        return self.total + ((self.ce - self.cs) if self.ce is not None else 0)


# ----------------------------------------------------------------- ROI logs (clean runs)

def parse_roi_log(path):
    """One process's ROI log -> dict. Raises on an unknown log version."""
    log = {"path": path, "events": [], "overflow": 0, "unmatched_end": 0, "argv": None,
           "pid": None, "rank": None, "host": None, "exe": None, "cwd": None}
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("# hpcperf-roi-log"):
                version = int(line.split()[-1])
                if version != ROI_LOG_VERSION:
                    raise ValueError(f"{path}: ROI log version {version}, expected {ROI_LOG_VERSION}")
                continue
            head, _, rest = line.partition(" ")
            if head in ("B", "E", "U", "x"):
                a, b = rest.split()
                log["events"].append((head, int(a), int(b)))
            elif head == "argv":
                log["argv"] = json.loads(rest)
            elif head in ("overflow", "unmatched_end"):
                log[head] += int(rest)
            elif head in ("pid", "rank", "host", "exe", "cwd", "clock"):
                log[head] = rest
    return log


def roi_from_log(log):
    """Outermost ROI entries of one process: wall = sum(E - B) - sum of excluded time.

    Events: B/E/U carry (monotonic ns, realtime ns); x, written right after the E of an
    entry that had excludes, carries (excluded ns summed over that entry, their count)."""
    entries, wall, excluded, n_excl = 0, 0, 0, 0
    first_begin = last_end = None
    b = None
    unterminated = False
    for kind, a, c in log["events"]:
        if kind == "B":
            b = (a, c)
            entries += 1
            if first_begin is None:
                first_begin = c
        elif kind == "E" and b is not None:
            wall += a - b[0]
            last_end = c
            b = None
        elif kind == "x":
            excluded += a
            n_excl += c
        elif kind == "U":
            unterminated = True
    if b is not None:
        unterminated = True
    return {"entries": entries, "gross_ns": wall, "excluded_ns": excluded, "excludes": n_excl,
            "wall_ns": wall - excluded,
            "first_begin_real": first_begin, "last_end_real": last_end,
            "unterminated": unterminated, "overflow": log["overflow"], "unmatched_end": log["unmatched_end"]}


def clean_roi(logs):
    """All processes of one clean run -> job-level ROI.

    The job's ROI wall is the slowest process's (it determines completion); the spread
    across processes is reported as imbalance. With one process these are exact. The
    multi-process path is UNVERIFIED on the node this was built on (one GPU).
    """
    if not logs:
        return None
    per = [roi_from_log(l) for l in logs]
    per = [p for p in per if p["entries"] > 0 or p["unterminated"]]
    if not per:
        return None
    walls = [p["wall_ns"] for p in per]
    firsts = [p["first_begin_real"] for p in per if p["first_begin_real"] is not None]
    lasts = [p["last_end_real"] for p in per if p["last_end_real"] is not None]
    return {
        "processes": len(per),
        "entries": max(p["entries"] for p in per),
        "wall_ns": max(walls),
        "excluded_ns": max(p["excluded_ns"] for p in per),
        "imbalance_ns": max(walls) - min(walls),
        "first_begin_real": min(firsts) if firsts else None,
        "last_end_real": max(lasts) if lasts else None,
        "unterminated": any(p["unterminated"] for p in per),
        "overflow": sum(p["overflow"] for p in per),
        "unmatched_end": sum(p["unmatched_end"] for p in per),
    }


# ----------------------------------------------------------------- profiled run

def roi_windows(markers):
    """Markers -> ({proc: sub-windows (ROI minus excludes)}, {proc: stats})."""
    rois, excls = {}, {}
    for m in markers:
        (rois if m.kind == "roi" else excls).setdefault(m.proc, []).append((m.start, m.end))
    windows, stats = {}, {}
    for proc, rs in rois.items():
        roi = merge(rs)
        ex = merge(excls.get(proc, []))
        windows[proc] = subtract(roi, ex)
        gross = sum(e - s for s, e in roi)
        excl = overlap_len(roi, ex)
        stats[proc] = {"entries": len(roi), "gross_ns": gross, "excluded_ns": excl, "wall_ns": gross - excl}
    return windows, stats


def analyze_trace(trace, capabilities, top_ops=None):
    """Clip device activity to the ROI windows the trace itself recorded.

    Returns the ROI-scoped device picture, the whole-process one (context), the ops
    table and runtime API calls. A category outside `capabilities` is None.
    """
    windows, wstats = roi_windows(trace.markers())
    ends = {p: [w[1] for w in ws] for p, ws in windows.items()}

    roi_cat = {c: 0 for c in CATEGORIES}
    roi_ops = {c: 0 for c in CATEGORIES}
    roi_bytes = {c: 0 for c in CATEGORIES}
    whole_cat = {c: 0 for c in CATEGORIES}
    whole_ops = {c: 0 for c in CATEGORIES}
    roi_union = {p: [_Union() for _ in ws] for p, ws in windows.items()}
    whole_union = {}
    roi_op_sum = whole_op_sum = 0
    table = {}

    for iv in trace.intervals():
        s, e, cat, proc = iv.start, iv.end, iv.category, iv.proc
        if e <= s:
            continue
        u = whole_union.get(proc)
        if u is None:
            u = whole_union[proc] = _Union()
        u.add(s, e)
        whole_cat[cat] += e - s
        whole_ops[cat] += 1
        whole_op_sum += e - s

        wins = windows.get(proc)
        if not wins:
            continue
        i = bisect.bisect_right(ends[proc], s)
        clipped = 0
        while i < len(wins) and wins[i][0] < e:
            cs, ce = max(s, wins[i][0]), min(e, wins[i][1])
            if ce > cs:
                roi_union[proc][i].add(cs, ce)
                clipped += ce - cs
            i += 1
        if clipped <= 0:
            continue
        roi_cat[cat] += clipped
        roi_ops[cat] += 1
        roi_op_sum += clipped
        if iv.nbytes:
            roi_bytes[cat] += iv.nbytes * clipped / (e - s)
        t = table.get(iv.key)
        if t is None:
            table[iv.key] = [cat, 1, clipped, clipped, clipped]
        else:
            t[1] += 1
            t[2] += clipped
            t[3] = min(t[3], clipped)
            t[4] = max(t[4], clipped)

    busy = sum(u.length() for us in roi_union.values() for u in us)
    whole_busy = sum(u.length() for u in whole_union.values())
    prof_wall = max((st["wall_ns"] for st in wstats.values()), default=0)

    def cap(cat, v):
        return v if cat in capabilities else None

    def sec(ns):
        return None if ns is None else finite("seconds", ns / 1e9)

    observable = bool(capabilities)
    device = None
    if observable and windows:
        device = {
            "busy_s": sec(busy),
            "busy_frac_of_profiled_roi": finite("busy/roi", busy / prof_wall) if prof_wall > 0 else None,
            "op_time_sum_s": sec(roi_op_sum),
            "overlap_s": sec(roi_op_sum - busy),
        }
        for c in CATEGORIES:
            device[f"{c}_s"] = sec(cap(c, roi_cat[c]))
            device[f"{c}_ops"] = cap(c, roi_ops[c])
        for c in COPY_CATEGORIES + ("fill",):
            b = cap(c, roi_bytes[c])
            device[f"{c}_bytes"] = None if b is None else int(round(b))

    whole = None
    if observable:
        whole = {"busy_s": sec(whole_busy), "op_time_sum_s": sec(whole_op_sum)}
        for c in CATEGORIES:
            whole[f"{c}_s"] = sec(cap(c, whole_cat[c]))
            whole[f"{c}_ops"] = cap(c, whole_ops[c])

    ops = []
    if table:
        names = trace.op_names(list(table))
        total = sum(t[2] for t in table.values()) or 1
        for key, (cat, count, tot, mn, mx) in sorted(table.items(), key=lambda kv: -kv[1][2]):
            ops.append({"name": names.get(key, str(key)), "category": cat, "count": count,
                        "total_s": sec(tot), "avg_s": sec(tot / count), "min_s": sec(mn),
                        "max_s": sec(mx), "share": finite("share", tot / total)})
        if top_ops:
            ops = ops[:top_ops]

    rt_roi = trace.runtime_calls(windows) if windows else None
    rt_whole = trace.runtime_calls(None)

    def rt(d):
        if d is None:
            return None
        return {"calls": d["calls"], "time_s": sec(d["time_ns"]),
                "sync_calls": d["sync_calls"], "sync_s": sec(d["sync_ns"])}

    return {
        "profiled_roi": {
            "found": bool(windows),
            "processes": len(windows),
            "entries": max((st["entries"] for st in wstats.values()), default=0),
            "wall_s": sec(prof_wall) if windows else None,
            "excluded_s": sec(max((st["excluded_ns"] for st in wstats.values()), default=0)) if windows else None,
        },
        "device": device,
        "whole_process": whole,
        "runtime_api_roi": rt(rt_roi),
        "runtime_api_whole": rt(rt_whole),
        "ops": ops,
    }
