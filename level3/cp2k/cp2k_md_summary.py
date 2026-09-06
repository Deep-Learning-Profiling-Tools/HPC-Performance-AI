#!/usr/bin/env python3
"""Summarise (and with --check, validate) a CP2K Quickstep MD run for validate.sh.

usage: cp2k_md_summary.py <cp2k.out> <input.inp> [--check]

Prints key=value lines (md_summary.txt).  With --check the run must satisfy:
  * "PROGRAM ENDED AT" present and MD step 10 reached;
  * exactly one SCF cycle before MD_INI (the initial SCF from SCF_GUESS) and one per MD
    step afterwards; EVERY MD-step SCF cycle converged.  The initial cycle may end with
    "SCF run NOT converged" ONLY if the deck itself declares IGNORE_CONVERGENCE_FAILURE
    (upstream's benchmarks/QS/H2O-*.inp do: the ATOMIC-guess start does not converge
    within the default MAX_SCF=50 OT iterations by design) -- its iteration count and
    status are reported, never hidden;
  * all FORCE_EVAL / MD energies finite;
  * GPU evidence from CP2K's own output: cp2kflags contain offload_cuda and dbcsr_acc,
    DBCSR reports >= 1 accelerator device per node, and the GRID task statistics show
    collocate/integrate tasks executed on the GPU.  PW (FFT) GPU use is reported from the
    pw_gpu_* timers when they reach the timing report, otherwise marked unverified.
The energies compared across rank counts are the per-cycle "ENERGY| Total FORCE_EVAL"
values (18 decimals), efe_step0 = initial SCF, efe_step1..10 = MD steps 1..10.
"""
import os
import re
import sys

sys.path.insert(0, os.environ["L3_TOOLS"])
from l3_check import ValidationError, require_finite  # noqa: E402

out, inp = sys.argv[1], sys.argv[2]
check = "--check" in sys.argv[3:]
txt = open(out, errors="replace").read()
deck = open(inp, errors="replace").read()
ignore_conv = re.search(r"^\s*IGNORE_CONVERGENCE_FAILURE\b", deck, re.M) is not None
header = txt[: txt.find(" GLOBAL|")] if " GLOBAL|" in txt else txt[:20000]

md_ini = txt.find("MD_INI| MD initialization")
events = [(m.start(), m.group(1) is None) for m in re.finditer(r"SCF run (NOT )?converged", txt)]
init = [c for p, c in events if md_ini < 0 or p < md_ini]
md = [c for p, c in events if md_ini >= 0 and p > md_ini]
m_iter = re.search(r"Leaving inner SCF loop after reaching\s+(\d+) steps", txt)
m_conv = re.search(r"SCF run converged in\s+(\d+) steps", txt)
init_iters = int(m_iter.group(1)) if (m_iter and init and not init[0]) else (int(m_conv.group(1)) if m_conv else -1)
steps = [int(m) for m in re.findall(r"MD\| Step number\s+(\d+)", txt)]
efe = [float(x) for x in re.findall(r"ENERGY\| Total FORCE_EVAL \( QS \) energy \[hartree\]\s+(\S+)", txt)]
epot = [float(x) for x in re.findall(r"MD\| Potential energy \[hartree\]\s+(\S+)", txt)]
cons = [float(x) for x in re.findall(r"MD\| Conserved quantity \[hartree\]\s+(\S+)", txt)]
m = re.search(r"DBCSR\| ACC: Number of devices/node\s+(\d+)", txt)
ndev = int(m.group(1)) if m else -1
flags_ok = ("offload_cuda" in header) and ("dbcsr_acc" in header)
grid = {"GPU": 0, "CPU": 0}
for backend, cnt in re.findall(r"^\s*\d+\s+(?:collocate|integrate) (?:ortho|general)\s+(GPU|CPU)\s+(\d+)", txt, re.M):
    grid[backend] += int(cnt)
pw_gpu = sorted(set(re.findall(r"^\s*(pw_gpu_\w+)", txt, re.M)))

try:
    if check:
        if "PROGRAM ENDED AT" not in txt:
            raise ValidationError("no 'PROGRAM ENDED AT' -- run incomplete")
        if not steps or steps[-1] != 10:
            raise ValidationError(f"MD reached step {steps[-1] if steps else None}, expected 10")
        if md_ini < 0:
            raise ValidationError("no MD_INI block")
        if len(init) != 1:
            raise ValidationError(f"expected exactly one SCF cycle before MD_INI, found {len(init)}")
        if not init[0] and not ignore_conv:
            raise ValidationError("initial SCF did not converge and the deck does not declare IGNORE_CONVERGENCE_FAILURE")
        if len(md) != 10:
            raise ValidationError(f"expected 10 MD-step SCF cycles, found {len(md)}")
        if not all(md):
            raise ValidationError(f"{sum(1 for c in md if not c)} of 10 MD-step SCF cycles did not converge")
        if len(efe) != 11:
            raise ValidationError(f"expected 11 FORCE_EVAL energies (initial + 10 steps), found {len(efe)}")
        if len(epot) < 10 or len(cons) < 10:
            raise ValidationError(f"expected 10 MD energy records, got {len(epot)}/{len(cons)}")
        for i, e in enumerate(efe):
            require_finite(f"FORCE_EVAL energy #{i}", e)
        for e in epot + cons:
            require_finite("MD energy", e)
        if ndev < 1:
            raise ValidationError(f"DBCSR reports {ndev} accelerator devices -- GPU backend not active")
        if not flags_ok:
            raise ValidationError("cp2kflags lack offload_cuda/dbcsr_acc -- not a CUDA-offload binary")
        if grid["GPU"] <= 0:
            raise ValidationError("GRID statistics show no collocate/integrate tasks executed on the GPU")
    print(f"steps={steps[-1] if steps else -1} initial_scf_converged={init[0] if init else None} initial_scf_iterations={init_iters} "
          f"deck_ignore_convergence_failure={ignore_conv} md_scf_converged={sum(1 for c in md if c)}/{len(md)}")
    print(f"dbcsr_acc_devices_per_node={ndev} cp2kflags_offload_cuda_dbcsr_acc={flags_ok} grid_gpu_tasks={grid['GPU']} grid_cpu_tasks={grid['CPU']} "
          f"pw_gpu_timers={','.join(pw_gpu) if pw_gpu else 'absent-from-timing-report(unverified)'}")
    print(" ".join(f"efe_step{i}={e!r}" for i, e in enumerate(efe)))
    if epot and cons:
        print("epot_step1=%.12f epot_step10=%.12f cons_step1=%.12f cons_step10=%.12f cons_drift=%.3e"
              % (epot[0], epot[-1], cons[0], cons[-1], cons[-1] - cons[0]))
except ValidationError as ex:
    print(f"  VALIDATION ERROR: {ex}")
    sys.exit(1)
