#!/usr/bin/env python3
"""Compare a DFT-FE run (REPRODUCIBLE OUTPUT = true) with a reference output for validate.sh.

usage: dftfe_check.py <dftfe.out> <reference.output> [--check] [--label NAME]
                      [--tol-e0 1e-5] [--tol-emd 2e-5] [--tol-temp 0.1] [--tol-force 2e-5]

Quantities (all parsed from DFT-FE's own output, both files):
  e0        first ground-state "Total energy:" (Ha, printed with 8 decimals)
  emd[k]    "Total Energy in Ha at timeIndex" of MD step k (5 decimals)        [MD decks]
  temp[k]   "Temperature from velocities" of MD step k (K, 2 decimals)          [MD decks]
  forces    "Absolute values of ion forces" per atom (Ha/Bohr, 6 decimals)      [when printed]
  steps     number of MD steps; MD decks must end with "MD run completed successfully"
Pre-fixed tolerances (absolute): |d e0| <= 1e-5 Ha, |d emd| <= 2e-5 Ha, |d temp| <= 0.1 K,
max |d force| <= 2e-5 Ha/Bohr -- chosen BEFORE the runs from the deck's own convergence settings
(SCF TOLERANCE 1e-5 on the density residual, energies quadratically converged) and the print
resolution of the reference (5 decimals for the MD energies). Upstream's own GPU check is a plain
`diff` against accuracyBenchmarks/, i.e. an even stricter expectation.
With --check any violation, NaN/Inf, missing quantity, incomplete MD or step-count mismatch exits 1.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.environ["L3_TOOLS"])
from l3_check import ValidationError, require_finite  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("ref"); ap.add_argument("--check", action="store_true"); ap.add_argument("--label", default="")
ap.add_argument("--tol-e0", type=float, default=1e-5); ap.add_argument("--tol-emd", type=float, default=2e-5)
ap.add_argument("--tol-temp", type=float, default=0.1); ap.add_argument("--tol-force", type=float, default=2e-5)
a = ap.parse_args()


def parse(path, name):
    txt = open(path, errors="replace").read()
    d = {"complete": "MD run completed successfully" in txt, "is_md": "MD STEP" in txt or "Molecular Dynamics" in txt}
    e0 = re.findall(r"Total energy:\s+(\S+)", txt)
    d["e0"] = require_finite(f"{name} e0", e0[0]) if e0 else None
    d["scf_iters"] = [int(x) for x in re.findall(r"converged to the specified tolerance after:\s+(\d+) iterations", txt)]
    d["emd"] = [require_finite(f"{name} emd", x) for x in re.findall(r"Total Energy in Ha at timeIndex\s+(\S+)", txt)]
    d["temp"] = [require_finite(f"{name} temp", x) for x in re.findall(r"Temperature from velocities:\s+(\S+)", txt)]
    d["steps"] = [int(x) for x in re.findall(r"MD STEP\s+(\d+)", txt)]
    forces = []
    for block in re.findall(r"Absolute values of ion forces[^\n]*\n(.*?)\n-{20,}", txt, re.S):
        forces.append([tuple(require_finite(f"{name} force", v) for v in m.split(",")) for m in re.findall(r"AtomId\s+\d+:\s+([-0-9.eE,]+)", block)])
    d["forces"] = forces
    d["not_converged"] = "not converged" in txt.lower() or "NOT converged" in txt
    return d


try:
    o, r = parse(a.out, "run"), parse(a.ref, "reference")
    if a.check:
        if o["e0"] is None:
            raise ValidationError("run has no 'Total energy:' line (no completed SCF)")
        if o["not_converged"]:
            raise ValidationError("run reports a non-converged solve")
        if r["is_md"]:
            if not o["complete"]:
                raise ValidationError("MD run did not end with 'MD run completed successfully'")
            if len(o["emd"]) != len(r["emd"]) or len(o["steps"]) != len(r["steps"]):
                raise ValidationError(f"MD step count {len(o['steps'])}/{len(o['emd'])} energies vs reference {len(r['steps'])}/{len(r['emd'])}")
    de0 = abs(o["e0"] - r["e0"]) if (o["e0"] is not None and r["e0"] is not None) else None
    demd = [abs(x - y) for x, y in zip(o["emd"], r["emd"])]
    dtemp = [abs(x - y) for x, y in zip(o["temp"], r["temp"])]
    dforce = None
    if o["forces"] and r["forces"] and len(o["forces"][0]) == len(r["forces"][0]):
        dforce = max(abs(x - y) for fo, fr in zip(o["forces"], r["forces"]) for po, pr in zip(fo, fr) for x, y in zip(po, pr))
    tag = f"[{a.label}] " if a.label else ""
    print(f"{tag}e0_run={o['e0']!r} e0_ref={r['e0']!r} d_e0={de0!r} tol_e0={a.tol_e0} scf_iters_run={o['scf_iters']} scf_iters_ref={r['scf_iters']}")
    if r["is_md"] or o["is_md"]:
        print(f"{tag}md_steps={len(o['steps'])} complete={o['complete']} emd_run={o['emd']} emd_ref={r['emd']} max_d_emd={max(demd) if demd else None!r} tol_emd={a.tol_emd}")
        print(f"{tag}temp_run={o['temp']} temp_ref={r['temp']} max_d_temp={max(dtemp) if dtemp else None!r} tol_temp={a.tol_temp}")
    print(f"{tag}force_blocks_run={len(o['forces'])} force_blocks_ref={len(r['forces'])} max_d_force={dforce!r} tol_force={a.tol_force}")
    if a.check:
        bad = []
        if de0 is None or de0 > a.tol_e0: bad.append(f"e0 |diff| {de0} > {a.tol_e0}")
        if any(x > a.tol_emd for x in demd): bad.append(f"MD energy |diff| {max(demd)} > {a.tol_emd}")
        if any(x > a.tol_temp for x in dtemp): bad.append(f"temperature |diff| {max(dtemp)} > {a.tol_temp}")
        if dforce is not None and dforce > a.tol_force: bad.append(f"force |diff| {dforce} > {a.tol_force}")
        if bad:
            raise ValidationError("; ".join(bad))
        print(f"{tag}RESULT within tolerances")
except ValidationError as ex:
    print(f"  VALIDATION ERROR: {ex}")
    sys.exit(1)
