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
    query = ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]
    out = _run(query)
    if not out:
        return None
    rows = [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]
    if not rows:
        return None
    # The device described is the one CUDA device 0 will be: nvidia-smi ignores
    # CUDA_VISIBLE_DEVICES, so its first entry is resolved here. A UUID names the GPU
    # exactly; an index is taken in nvidia-smi's enumeration order (PCI bus order, the
    # order CUDA also uses for identical GPUs unless CUDA_DEVICE_ORDER says otherwise).
    selected, first = "first enumerated", (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    if first:
        pick = _run(query + ["-i", first])
        prow = [[c.strip() for c in line.split(",")] for line in (pick or "").strip().splitlines() if line.strip()]
        if prow and len(prow[0]) == len(fields):
            rows, selected = prow + [r for r in rows if r != prow[0]], f"CUDA_VISIBLE_DEVICES={first}"
        else:
            selected = f"first enumerated (CUDA_VISIBLE_DEVICES={first} not resolved by nvidia-smi)"
    g = dict(zip(fields, rows[0]))
    toolkit = None
    # the toolkit version is part of the platform id: a slow first nvcc start (cold file
    # cache) must not turn it into "unknown", so give it time and one more try
    nv = _run(["nvcc", "--version"], timeout=60) or _run(["nvcc", "--version"], timeout=60)
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
                          "pci_bus_id": g["pci.bus_id"], "selected_by": selected,
                          "visible_pci_bus_ids": [dict(zip(fields, r))["pci.bus_id"] for r in rows]},
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
