#!/usr/bin/env python3
"""Describe the accelerator and host a measurement runs on -- vendor-neutral fields.

    python3 tools/timing/probes/device.py            # JSON on stdout
    python3 tools/timing/probes/device.py --platform # just the platform id

The descriptor is the join key for cross-hardware data: the same (level, app, case)
measured on two platforms differs only in this block. Field names are neutral; what a
vendor calls differently (SMs, CUs, TPU cores) goes into `vendor_extras`.

Probes: NVIDIA (nvidia-smi) is implemented and verified on dgx003. AMD and TPU are
INTERFACE ONLY -- they state the fields they must fill and refuse to guess. On a host
where no probe applies the descriptor says vendor "unknown": the `none` collector can
still time ROIs there, and nothing is claimed about the device.
"""

import json
import os
import platform as _platform
import re
import shutil
import socket
import subprocess
import sys

SCHEMA = "hpcperf-device-1"


def _run(argv, timeout=20):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout if p.returncode == 0 else None


def _num(s):
    try:
        v = float(s)
        return int(v) if v.is_integer() else v
    except (TypeError, ValueError):
        return None


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def host_info():
    cpu = None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    mem_kb = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    mem_kb = int(line.split()[1])
    except OSError:
        pass
    try:
        allowed = len(os.sched_getaffinity(0))
    except AttributeError:
        allowed = os.cpu_count()
    try:
        load1 = os.getloadavg()[0]
    except OSError:
        load1 = None
    return {"hostname": socket.gethostname().split(".")[0], "cpu_model": cpu, "cpus_allowed": allowed,
            "kernel": _platform.release(), "arch": _platform.machine(),
            "loadavg_1m": load1, "mem_available_kb": mem_kb}


def probe_nvidia():
    fields = ["index", "name", "uuid", "driver_version", "compute_cap", "memory.total",
              "clocks.max.sm", "clocks.max.mem", "clocks.sm", "clocks.mem",
              "power.limit", "temperature.gpu", "persistence_mode", "pci.bus_id"]
    out = _run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    if not out:
        return None
    rows = [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]
    if not rows:
        return None
    g = dict(zip(fields, rows[0]))
    toolkit = None
    nv = _run(["nvcc", "--version"])
    if nv:
        m = re.search(r"release (\d+\.\d+)", nv)
        toolkit = m.group(1) if m else None
    product = re.sub(r"^NVIDIA\s+", "", g["name"])
    cc = g["compute_cap"]
    dev = {
        "vendor": "nvidia",
        "product": product,
        "arch": f"sm_{cc.replace('.', '')}" if cc else None,
        "uuid": g["uuid"],
        "count_visible": len(rows),
        "memory_total_mib": _num(g["memory.total"]),
        "core_clock_mhz": _num(g["clocks.sm"]),
        "core_clock_max_mhz": _num(g["clocks.max.sm"]),
        "mem_clock_mhz": _num(g["clocks.mem"]),
        "mem_clock_max_mhz": _num(g["clocks.max.mem"]),
        "power_limit_w": _num(g["power.limit"]),
        "temperature_c": _num(g["temperature.gpu"]),
        "driver_version": g["driver_version"],
        "runtime": {"name": "cuda", "version": toolkit},
        "vendor_extras": {"compute_capability": cc, "persistence_mode": g["persistence_mode"],
                          "pci_bus_id": g["pci.bus_id"]},
    }
    dev["platform_id"] = f"nvidia-{_slug(product)}.cuda{toolkit or 'unknown'}"
    return dev


def probe_amd():
    """INTERFACE ONLY. Must fill the same fields as probe_nvidia() from amd-smi/rocminfo:
    product (e.g. "MI300X"), arch (gfx target, e.g. "gfx942"), memory_total_mib, clocks,
    power_limit_w, driver_version, runtime {"name": "hip", "version": ROCm version},
    vendor_extras {compute_units, xcds, ...}, platform_id "amd-<product>.rocm<ver>".
    Unimplemented because it could not be checked here (no ROCm on dgx003)."""
    if shutil.which("amd-smi") or shutil.which("rocm-smi") or os.path.isdir("/opt/rocm"):
        raise NotImplementedError("AMD device probe is interface-only and UNVERIFIED; implement probe_amd()")
    return None


def probe_tpu():
    """INTERFACE ONLY (project decision). Must fill: product ("v5p", ...), arch, count_visible
    (chips), memory_total_mib (HBM per chip), runtime {"name": "xla", "version": jaxlib/libtpu},
    vendor_extras {cores_per_chip, topology}, platform_id "google-tpu-<product>.xla<ver>"."""
    if os.path.exists("/dev/accel0") or os.environ.get("TPU_NAME"):
        raise NotImplementedError("TPU device probe is interface-only (project decision); implement probe_tpu()")
    return None


def describe():
    dev = probe_nvidia() or probe_amd() or probe_tpu()
    if dev is None:
        dev = {"vendor": "unknown", "product": None, "arch": None, "count_visible": 0,
               "runtime": {"name": None, "version": None}, "vendor_extras": {},
               "platform_id": f"unknown-{_slug(_platform.machine())}"}
    return {"schema": SCHEMA, "device": dev, "host": host_info()}


def main(argv):
    d = describe()
    if "--platform" in argv:
        print(d["device"]["platform_id"])
    else:
        print(json.dumps(d, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
