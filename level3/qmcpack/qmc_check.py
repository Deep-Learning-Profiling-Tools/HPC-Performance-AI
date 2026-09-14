#!/usr/bin/env python3
"""Completeness / GPU-evidence / statistical check of one QMCPACK run for validate.sh.

usage: qmc_check.py <run_dir> <prefix> "<ref_mean> <ref_sigma>" <nsigma> <equil_blocks> <check_scalars.py>

Prints key=value summary lines; exit 1 (after printing VALIDATION ERROR) on any violation:
  * qmc.out contains "QMCPACK execution completed successfully" and no "QMCPACK ERROR"; the manifest
    records exit_code=0;
  * banner lines "OpenMP target offload to accelerators build option is enabled" and
    "CUDA acceleration build option is enabled" are present (the binary's GPU paths are compiled in and
    initialised; the run had OMP_TARGET_OFFLOAD=MANDATORY, so a host fallback would have aborted);
  * <prefix>.s001.scalar.dat (DMC) exists with 25 blocks, every column finite; s000 (VMC) exists;
  * upstream's check_scalars.py (--ns nsigma --series 1 -e equil --le "<ref>") passes; its computed
    mean/error bar are exported as dmc_mean/dmc_err for the cross-rank consistency check.
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.environ["L3_TOOLS"])
from l3_check import ValidationError, require_finite  # noqa: E402

run_dir, prefix, ref, nsigma, equil, checker = sys.argv[1:7]
run_dir, checker = os.path.abspath(run_dir), os.path.abspath(checker)   # check_scalars.py runs with cwd = run_dir
out = os.path.join(run_dir, "qmc.out")
txt = open(out, errors="replace").read() if os.path.exists(out) else ""
manifest = open(os.path.join(run_dir, "run_manifest.txt"), errors="replace").read() if os.path.exists(os.path.join(run_dir, "run_manifest.txt")) else ""


def scalar_table(path):
    rows, cols = [], None
    for line in open(path):
        if line.startswith("#"):
            cols = line[1:].split()
            continue
        if line.strip():
            rows.append([require_finite(f"{os.path.basename(path)} value", x) for x in line.split()])
    return cols, rows


try:
    if not txt:
        raise ValidationError("qmc.out missing or empty")
    m = re.search(r"^exit_code=(\d+)$", manifest, re.M)
    if not m or int(m.group(1)) != 0:
        raise ValidationError(f"manifest exit_code = {m.group(1) if m else 'missing'}")
    if "QMCPACK ERROR" in txt:
        raise ValidationError("'QMCPACK ERROR' in output")
    if "QMCPACK execution completed successfully" not in txt:
        raise ValidationError("no 'QMCPACK execution completed successfully' -- run incomplete")
    offload = "OpenMP target offload to accelerators build option is enabled" in txt
    cuda = "CUDA acceleration build option is enabled" in txt
    if not (offload and cuda):
        raise ValidationError(f"GPU banner missing (offload={offload}, cuda={cuda})")
    # the drivers/wavefunction components must report the device code paths, and device memory must
    # actually have been allocated through the offload runtime during the run
    n_offload_paths = len(re.findall(r"Running OpenMP offload code path", txt))
    n_cuda_paths = len(re.findall(r"Running on a GPU via CUDA/HIP acceleration", txt))
    dev_mib = [int(x) for x in re.findall(r"Device memory allocated via OpenMP offload\s*:\s*(\d+) MiB", txt)]
    if n_offload_paths == 0 or n_cuda_paths == 0:
        raise ValidationError(f"no device code paths reported (offload paths={n_offload_paths}, cuda paths={n_cuda_paths})")
    if not dev_mib or max(dev_mib) <= 0:
        raise ValidationError(f"no device memory was allocated through the offload runtime (records: {dev_mib})")
    m_wpr = re.search(r"walkers_per_rank\s*=\s*\[([^\]]*)\]", txt)
    walkers_per_rank = m_wpr.group(1).strip() if m_wpr else "?"
    s0, s1 = (os.path.join(run_dir, f"{prefix}.s00{i}.scalar.dat") for i in (0, 1))
    for p in (s0, s1):
        if not os.path.exists(p):
            raise ValidationError(f"{os.path.basename(p)} missing")
    cols, rows = scalar_table(s1)
    if len(rows) != 25:
        raise ValidationError(f"DMC scalar file has {len(rows)} blocks, expected 25")
    if "LocalEnergy" not in cols:
        raise ValidationError("DMC scalar file lacks a LocalEnergy column")
    vcols, vrows = scalar_table(s0)
    pop = None   # the batched drivers print no per-block walker count; the population is taken from walkers_per_rank in qmc.out
    res = subprocess.run([sys.executable, checker, "--ns", nsigma, "--series", "1", "-p", prefix, "-e", equil, "--le", ref],
                         cwd=run_dir, capture_output=True, text=True)
    rep = res.stdout + res.stderr
    mean = re.search(r"computed\s+mean value\s*:\s*(-?[0-9.]+)", rep)
    err = re.search(r"computed\s+error bar\s*:\s*(-?[0-9.]+)", rep)
    print(f"completed=True offload_banner={offload} cuda_banner={cuda} offload_code_paths={n_offload_paths} cuda_code_paths={n_cuda_paths} "
          f"max_device_mib_via_offload={max(dev_mib)} walkers_per_rank=[{walkers_per_rank}] vmc_blocks={len(vrows)} dmc_blocks={len(rows)}")
    if mean and err:
        print(f"dmc_mean={float(mean.group(1))!r} dmc_err={float(err.group(1))!r} reference=\"{ref}\" nsigma={nsigma} equilibration_blocks={equil}")
    for line in rep.splitlines():
        if re.search(r"reference|computed|tolerance|deviation|status of this test|Testing quantity", line):
            print("check_scalars: " + line.strip())
    if res.returncode != 0 or "status of this test      :   pass" not in rep:
        raise ValidationError(f"check_scalars.py exit {res.returncode}: DMC total energy outside the upstream reference window")
    if not (mean and err):
        raise ValidationError("could not parse check_scalars.py computed mean/error")
except ValidationError as ex:
    print(f"  VALIDATION ERROR: {ex}")
    sys.exit(1)
