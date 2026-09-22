"""Collector adapters: vendor trace -> the canonical activity model.

This package is the contract between "how a platform is profiled" (per vendor) and
"what is measured" (tools/timing/analysis.py, vendor-neutral). Analysis never reads
a vendor format; it only calls the methods below. Adding a platform means adding
one module here, a wrapper in tools/timing/lib/collectors.sh, and passing the
conformance probe (tools/timing/probes/conformance/).

A collector module exposes

    NAME            registry key, e.g. "nvidia_nsys"
    RUNTIME         the device runtime whose API calls it counts ("cuda", "hip", "xla")
    CAPABILITIES    frozenset of CATEGORIES it can observe; a category outside it is
                    reported as null (not observable), never as 0
    VERIFIED        False for interface-only modules (their open() raises)
    open(raw_dir)   -> Trace for one profiled run's raw directory

and a Trace provides

    info()                     dict: collector version, recorded-environment variable
                               NAMES (never values), anything the adapter wants to log
    markers()                  list[Marker]  -- ROI / exclude ranges as recorded by the tool
    intervals()                iterator of Interval sorted by start (ties arbitrary)
    op_names(keys)             dict key -> human-readable op name, for the ops table
    runtime_calls(windows)     {"calls","time_ns","sync_calls","sync_ns"} restricted to
                               windows = {proc: [(start_ns, end_ns), ...]}; None = whole run
    close()

All timestamps of one Trace are nanoseconds on ONE timeline (the adapter's job; the
conformance probe checks it). `proc` is an opaque process key that is identical in
markers and intervals of the same process.
"""

import collections
import importlib

# Device activity categories. Everything the device does falls in exactly one.
CATEGORIES = ("compute", "copy_h2d", "copy_d2h", "copy_d2d", "copy_other", "fill", "collective", "other")
COPY_CATEGORIES = ("copy_h2d", "copy_d2h", "copy_d2d", "copy_other")

Marker = collections.namedtuple("Marker", "proc kind start end")          # kind: "roi" | "exclude"
Interval = collections.namedtuple("Interval", "start end category key proc nbytes")

ROI_RANGE = "hpcperf:roi"
EXCLUDE_RANGE = "hpcperf:exclude"

_MODULES = {
    "nvidia_nsys": "collectors.nvidia_nsys",
    "none": "collectors.none",
    "amd_rocprofv3": "collectors.amd_rocprofv3",
    "tpu_xprof": "collectors.tpu_xprof",
}


def names():
    return sorted(_MODULES)


def get(name):
    if name not in _MODULES:
        raise KeyError(f"unknown collector {name!r}; known: {', '.join(names())}")
    return importlib.import_module(_MODULES[name])
