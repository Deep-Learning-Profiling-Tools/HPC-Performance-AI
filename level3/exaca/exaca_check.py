#!/usr/bin/env python3
"""exaca_check -- statistics of an ExaCA GrainID field (legacy VTK STRUCTURED_POINTS, ASCII or big-endian
binary, as written by ExaCA's Interlayer printing) and their comparison against a reference / another run.

    exaca_check.py stats <run.vtk> <run.json> <GrainOrientationVectors.csv> [--json OUT]
    exaca_check.py compare <a.stats.json> <b.stats.json> [--tol-file tolerances.json] [--label TEXT]

Statistics (all computed from the final GrainID field; ExaCA's own log adds VolFractionNucleated):
    cells, nx, ny, nz                       domain (must equal the deck)
    unsolidified_cells                      cells with GrainID == 0 (liquid / never assigned) -- must be 0
    n_grains, n_epitaxial, n_nucleated      distinct nonzero IDs; > 0 = substrate (epitaxial), < 0 = nucleated
    vol_fraction_nucleated                  cells with GrainID < 0 / cells (the code's log value is cross-checked)
    mean_grain_volume_cells                 cells / n_grains
    top_layer_grains                        distinct IDs in the top z layer (grain selection during growth)
    mean_misorientation_z_deg               cell-weighted mean over all solidified cells of the angle between the
                                            grain's closest <001> axis and the build (+z) direction (texture)
    mean_misorientation_z_top_deg           the same over the top z layer
The orientation of a grain is row (|GrainID| - 1) mod N of the orientation file (ExaCA's mapping), each row
= three <001> unit vectors (9 components). Everything is deterministic given the field; NaN/Inf never pass.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("L3_TOOLS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")))
from l3_check import ValidationError  # noqa: E402

DEFAULT_TOL = {   # absolute unless the key ends with _rel; justified in README.md ("Correctness criteria")
    "unsolidified_cells": 0,
    "vol_fraction_nucleated": 0.02,
    "n_grains_rel": 0.05,
    "n_nucleated_rel": 0.05,
    "top_layer_grains_rel": 0.10,
    "mean_misorientation_z_deg": 1.0,
    "mean_misorientation_z_top_deg": 1.5,
    "mean_grain_volume_cells_rel": 0.05,
}


def read_vtk_grainid(path):
    with open(path, "rb") as f:
        data = f.read()
    # header: text lines up to and including "LOOKUP_TABLE default"
    pos = 0
    header = {}
    binary = False
    dims = None
    n = None
    while True:
        nl = data.index(b"\n", pos)
        line = data[pos:nl].decode("ascii", "replace").strip()
        pos = nl + 1
        if line == "BINARY":
            binary = True
        elif line.startswith("DIMENSIONS"):
            dims = tuple(int(x) for x in line.split()[1:4])
        elif line.startswith("POINT_DATA"):
            n = int(line.split()[1])
        elif line.startswith("SCALARS"):
            header["scalars"] = line.split()[1:]
        elif line.startswith("LOOKUP_TABLE"):
            break
        if pos >= len(data):
            raise ValidationError(f"{path}: VTK header incomplete")
    if dims is None or n is None or header.get("scalars", [""])[0] != "GrainID":
        raise ValidationError(f"{path}: expected a GrainID STRUCTURED_POINTS field (dims {dims}, scalars {header.get('scalars')})")
    if n != dims[0] * dims[1] * dims[2]:
        raise ValidationError(f"{path}: POINT_DATA {n} != nx*ny*nz {dims}")
    if binary:
        arr = np.frombuffer(data, dtype=">i4", count=n, offset=pos)
        if arr.size != n:
            raise ValidationError(f"{path}: binary GrainID block truncated ({arr.size} of {n})")
    else:
        arr = np.array(data[pos:].split()[:n], dtype=np.int64)
        if arr.size != n:
            raise ValidationError(f"{path}: ASCII GrainID block truncated ({arr.size} of {n})")
    return dims, arr.astype(np.int64)


def read_orientations(path):
    with open(path) as f:
        n = int(f.readline().strip())
        rows = np.loadtxt(f, delimiter=",", max_rows=n)
    if rows.shape != (n, 9):
        raise ValidationError(f"{path}: expected {n} x 9 orientation components, got {rows.shape}")
    # misorientation of the closest <001> axis with +z: min over the three axes of arccos(|z-component|)
    z = np.abs(rows[:, [2, 5, 8]])
    z = np.clip(z, 0.0, 1.0)
    return n, np.degrees(np.arccos(z.max(axis=1)))


def stats(vtk, log_json, orient_csv):
    dims, g = read_vtk_grainid(vtk)
    nx, ny, nz = dims
    n_or, mis = read_orientations(orient_csv)
    cells = int(g.size)
    solid = g != 0
    unsolid = int(cells - solid.sum())
    ids = np.unique(g[solid])
    n_grains = int(ids.size)
    n_epi = int((ids > 0).sum()); n_nuc = int((ids < 0).sum())
    vf_nuc = float((g < 0).sum()) / cells
    # orientation index of every solidified cell: (|id| - 1) mod n_or
    oidx = (np.abs(g[solid]) - 1) % n_or
    mis_all = float(mis[oidx].mean()) if solid.any() else float("nan")
    top = g.reshape(nz, ny, nx)[-1]     # ExaCA writes x fastest, then y, then z
    top_solid = top[top != 0]
    top_grains = int(np.unique(top_solid).size)
    mis_top = float(mis[(np.abs(top_solid) - 1) % n_or].mean()) if top_solid.size else float("nan")
    out = {"vtk": os.path.basename(vtk), "nx": nx, "ny": ny, "nz": nz, "cells": cells, "unsolidified_cells": unsolid,
           "n_grains": n_grains, "n_epitaxial": n_epi, "n_nucleated": n_nuc, "vol_fraction_nucleated": vf_nuc,
           "mean_grain_volume_cells": cells / n_grains if n_grains else float("nan"), "top_layer_grains": top_grains,
           "mean_misorientation_z_deg": mis_all, "mean_misorientation_z_top_deg": mis_top, "min_grain_id": int(g.min()), "max_grain_id": int(g.max())}
    if log_json and os.path.isfile(log_json):
        lg = json.load(open(log_json))
        out["log"] = {"ranks": lg.get("NumberMPIRanks"), "time_step_of_output": lg.get("TimeStepOfOutput"),
                      "vol_fraction_nucleated_code": lg.get("Nucleation", {}).get("VolFractionNucleated"),
                      "domain": lg.get("Domain", {}), "decomposition": lg.get("Decomposition", {}),
                      "exaca_version": lg.get("ExaCAVersion"), "kokkos_version": lg.get("KokkosVersion")}
    for k, v in out.items():
        if isinstance(v, float) and not math.isfinite(v):
            raise ValidationError(f"statistic {k} is not finite")
    return out


def compare(a, b, tol=None, label="run vs reference"):
    """Return (ok, lines). Absolute tolerances, or relative when the tolerance key ends with _rel."""
    tol = {**DEFAULT_TOL, **(tol or {})}
    ok = True
    lines = []
    for key in ("nx", "ny", "nz", "cells"):
        if a.get(key) != b.get(key):
            ok = False; lines.append(f"  {label}: {key} {a.get(key)} != {b.get(key)}  BAD (different problem)")
    for key, t in tol.items():
        rel = key.endswith("_rel"); k = key[:-4] if rel else key
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            ok = False; lines.append(f"  {label}: {k} missing"); continue
        if not (math.isfinite(x) and math.isfinite(y)):
            ok = False; lines.append(f"  {label}: {k} not finite  BAD"); continue
        d = abs(x - y)
        if rel:
            ref = max(abs(y), 1e-30); good = d / ref <= t
            lines.append(f"  {label}: {k:<32} got {x:>12.6g} ref {y:>12.6g} rel {d / ref:.3e} (tol {t:.0e} rel) {'ok ' if good else 'BAD'}")
        else:
            good = d <= t
            lines.append(f"  {label}: {k:<32} got {x:>12.6g} ref {y:>12.6g} abs {d:.3e} (tol {t:g}) {'ok ' if good else 'BAD'}")
        ok = ok and good
    return ok, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats"); s.add_argument("vtk"); s.add_argument("log_json"); s.add_argument("orient"); s.add_argument("--json")
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b"); c.add_argument("--tol-file"); c.add_argument("--label", default="run vs reference")
    a = ap.parse_args()
    try:
        if a.cmd == "stats":
            st = stats(a.vtk, a.log_json, a.orient)
            if a.json:
                with open(a.json, "w") as f:
                    json.dump(st, f, indent=1)
            for k, v in st.items():
                if k != "log":
                    print(f"  {k}: {v}")
            return 0
        A, B = json.load(open(a.a)), json.load(open(a.b))
        tol = json.load(open(a.tol_file)) if a.tol_file else None
        ok, lines = compare(A, B, tol, a.label)
        print("\n".join(lines))
        return 0 if ok else 1
    except ValidationError as ex:
        print(f"  VALIDATION ERROR: {ex}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
