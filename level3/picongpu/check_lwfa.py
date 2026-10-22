#!/usr/bin/env python3
"""PIConGPU laser-wakefield check: the run's plugin histories against a frozen self-baseline of the unmodified build.

    check_lwfa.py --run-dir DIR --reference reference/<id>.json [--label TEXT]      check (PASS / FAIL, exit 0 / 1)
    check_lwfa.py --make-reference <input id> --run-dir DIR --out reference/<id>.json --binary TEXT
                  --charge-factor X --energy-rel X [--basis TEXT] [--note TEXT]           freeze (never overwrites)

The example has no analytic reference upstream (README: a demonstration setup). What a run leaves behind that a
wrong push, deposition or field solve changes: chargeConservation.dat (the charge-conservation plugin,
'#timestep max-charge-deviation unit[As]': the largest |div E - rho| on the grid in units of one macro-particle's
charge at every sampled step -- exact charge conservation is a property of the Esirkepov current deposition, so the
deviation stays at the discretization/roundoff level), fields_energy.dat (total and per-component field energy)
and e_energy_all.dat (kinetic and total electron energy). Checked per sampled step of the reference:
  * max-charge-deviation <= charge_factor x the reference's value at that step (and the sampled steps identical);
  * total field energy and electron kinetic energy within energy_rel relative of the reference
    (the plugins' sums are documented as naive reductions, 'not precise to more than a few percent' and not
    deterministic; energy_rel is recorded from the observed run-to-run spread of the unmodified build).
The reference is this build's own first run (self-baseline, stated as such in provenance); the rule is recorded in
the reference at freeze time from the stated basis and the check never applies a looser one.
"""
import argparse, hashlib, json, os, sys, time

SCHEMA = "hpcperf-picongpu-lwfa-reference-1"


def read_dat(path, ncols):
    rows = []
    with open(path, errors="replace") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            rows.append([float(x) for x in p[:ncols]])
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_run(d):
    return {"charge": read_dat(os.path.join(d, "chargeConservation.dat"), 3),
            "fields": read_dat(os.path.join(d, "fields_energy.dat"), 2),
            "electrons": read_dat(os.path.join(d, "e_energy_all.dat"), 3)}


def rel(a, b):
    return abs(a - b) / abs(b) if b != 0 else (0.0 if a == 0 else float("inf"))


def compare(run, ref):
    rule = ref["rule"]; cf, er = float(rule["charge_factor"]), float(rule["energy_rel"])
    lines, ok = [], True
    for key, label in (("charge", "chargeConservation.dat"), ("fields", "fields_energy.dat"), ("electrons", "e_energy_all.dat")):
        a, b = run[key], ref["run"][key]
        if [r[0] for r in a] != [r[0] for r in b]:
            lines.append(f"  {label}: sampled steps differ from the reference ({len(a)} vs {len(b)} rows) BAD"); ok = False
    if not ok:
        return ok, lines
    worst = 0.0; bad = 0; first = None
    for a, b in zip(run["charge"], ref["run"]["charge"]):
        limit = cf * b[1]
        if a[1] > limit and a[1] > 0:
            bad += 1; first = first or (int(a[0]), a[1], b[1])
        worst = max(worst, a[1] / b[1] if b[1] > 0 else (0.0 if a[1] == 0 else float("inf")))
    good = bad == 0; ok &= good
    lines.append(f"  charge conservation: max-deviation / reference per sampled step, worst {worst:.3f} (limit {cf:g}) violations={bad}" + (f" first at step {first[0]}: run {first[1]:.4e} ref {first[2]:.4e}" if first else "") + (" ok" if good else " BAD"))
    for key, col, label in (("fields", 1, "total field energy"), ("electrons", 1, "electron kinetic energy")):
        w = 0.0; bad = 0; first = None
        for a, b in zip(run[key], ref["run"][key]):
            if b[col] == 0 and a[col] == 0:
                continue
            d = rel(a[col], b[col]); w = max(w, d)
            if d > er:
                bad += 1; first = first or (int(a[0]), a[col], b[col])
        good = bad == 0; ok &= good
        lines.append(f"  {label}: worst relative deviation {w:.3e} (tol {er:.1e}) violations={bad}" + (f" first at step {first[0]}: run {first[1]:.6e} ref {first[2]:.6e}" if first else "") + (" ok" if good else " BAD"))
    return ok, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True); ap.add_argument("--reference"); ap.add_argument("--label", default="")
    ap.add_argument("--make-reference", metavar="INPUT_ID"); ap.add_argument("--out"); ap.add_argument("--binary", default="")
    ap.add_argument("--charge-factor", type=float); ap.add_argument("--energy-rel", type=float)
    ap.add_argument("--basis", default=""); ap.add_argument("--note", default="")
    a = ap.parse_args()
    label = a.label or os.path.basename(a.run_dir.rstrip("/"))
    try:
        run = load_run(a.run_dir)
    except (OSError, ValueError) as ex:
        print(f"FAIL: PIConGPU LWFA check ({label}): cannot read the run's plugin files: {ex}"); return 1
    if a.make_reference:
        if not (a.out and a.charge_factor is not None and a.energy_rel is not None):
            print("usage: --make-reference needs --out --charge-factor --energy-rel", file=sys.stderr); return 2
        if os.path.exists(a.out):
            print(f"refusing to overwrite the existing reference {a.out}", file=sys.stderr); return 2
        prov = {"made_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "binary": a.binary, "host": os.uname().nodename, "note": a.note,
                "files_sha256": {f: sha256(os.path.join(a.run_dir, f)) for f in ("chargeConservation.dat", "fields_energy.dat", "e_energy_all.dat")},
                "self_baseline": "this unmodified build's own run; no independent realisation exists for this example"}
        ref = {"schema": SCHEMA, "input_id": a.make_reference, "provenance": prov,
               "rule": {"charge_factor": a.charge_factor, "energy_rel": a.energy_rel, "basis": a.basis,
                        "text": "per sampled step: max-charge-deviation <= charge_factor x reference; total field energy and electron kinetic energy within energy_rel relative"},
               "run": run}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(ref, f, indent=1)
        print(f"reference written: {a.out} ({len(run['charge'])} charge rows, {len(run['fields'])} field rows, {len(run['electrons'])} electron rows; charge_factor {a.charge_factor:g}, energy_rel {a.energy_rel:.1e})")
        return 0
    if not a.reference:
        print("usage: --reference is required for a check", file=sys.stderr); return 2
    try:
        with open(a.reference) as f:
            ref = json.load(f)
        if ref.get("schema") != SCHEMA:
            raise ValueError(f"schema {ref.get('schema')!r} is not {SCHEMA}")
    except (OSError, ValueError) as ex:
        print(f"FAIL: PIConGPU LWFA check ({label}): cannot read the reference {a.reference}: {ex}"); return 1
    ok, lines = compare(run, ref)
    print(f"PIConGPU LWFA check ({label}): {a.run_dir} vs {os.path.basename(a.reference)} (input {ref.get('input_id')}; {ref['provenance'].get('binary', '')})")
    print("\n".join(lines))
    print(f"{'PASS' if ok else 'FAIL'}: PIConGPU LWFA check ({label})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
