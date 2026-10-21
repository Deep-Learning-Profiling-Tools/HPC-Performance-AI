#!/usr/bin/env python3
"""VPIC correctness check: a run's energies history against a frozen reference, plus total-energy conservation.

    check_energies.py --run <energies file> --reference reference/<id>.json [--label TEXT]      check (PASS / FAIL, exit 0 / 1)
    check_energies.py --make-reference <input id> --run <energies file> --out reference/<id>.json --deck <deck.cxx>
                      --log <run.log> --binary TEXT --rel-tol X --floor X --drift-tol X [--basis TEXT] [--note TEXT]   freeze (never overwrites)

VPIC's energies diagnostic (src/vpic/dump.cc dump_energies, every energies_interval steps of the registered decks)
writes one row per sampled step: the electric and magnetic field energies per component (ex ey ez bx by bz,
field_array->kernel->energy_f_kokkos) and the kinetic energy of every species (energy_p_kokkos), 6 significant
digits (%e). Their sum is the total energy of the periodic / conducting-wall system, which the explicit
leapfrog / Yee / Esirkepov scheme conserves only approximately (no exact invariant), so two checks:

  * reference comparison: the sampled step list must be identical (same num_step and energies_interval), and at
    every sampled step every column must satisfy  |run - ref| <= max(rel_tol * |ref|, floor * E_total(ref row)) :
    rel_tol bounds the dominant columns (the background field and the species kinetic energies), floor x total
    bounds the small field components (zero by symmetry, or the 1e-7..1e-4-of-the-total perturbation fields,
    whose relative deviation between two correct realisations is large while their share of the energy is tiny);
  * conservation: |E_total(step) - E_total(0)| / E_total(0) <= drift_tol at every sampled step -- for a closed
    system; a reference frozen with `--drift-tol none` (an open system such as the laser-driven short pulse, where
    the boundary injects energy) records the observed drift and skips this term: its energy history is then
    bounded only by the column-wise comparison above.

A statistically reproducible input (the laser-driven short pulse: two correct realisations differ by ~10 % per
column) is frozen with --groups-rel instead: the sums total / particles / fields at every sampled step within
that relative tolerance (rule mode 'groups'), no per-column or conservation bound.

The tolerances are recorded in the reference at freeze time from the stated basis (the input's README: the
deviation between two correct realisations of the same deck -- the Kokkos OpenMP build vs this CUDA build --
or, where no independent run of that size exists, the print precision and the measured drift of the unmodified
build); the check applies the recorded rule and never a looser one from the command line.
"""
import argparse, hashlib, json, math, os, sys, time

SCHEMA = "hpcperf-vpic-energies-reference-1"


def read_energies(path):
    """(columns, rows): columns = ['step', 'ex', ..., species...], rows = [[step, values...], ...]"""
    cols, rows = None, []
    with open(path, errors="replace") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            if s.startswith("%"):
                if s.startswith("% step"):
                    cols = s[2:].replace('"', "").split()
                continue
            p = s.split()
            rows.append([int(p[0])] + [float(x) for x in p[1:]])
    if cols is None or not rows:
        raise ValueError(f"{path}: no energies layout / rows")
    if any(len(r) != len(cols) for r in rows):
        raise ValueError(f"{path}: a row does not match the layout {cols}")
    return cols, rows


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def drift_of(rows):
    e0 = sum(rows[0][1:])
    return max(abs(sum(r[1:]) - e0) / abs(e0) for r in rows) if e0 != 0 else float("inf")


def compare_groups(cols, rows, ref):
    """Statistical inputs (rule mode 'groups'): the sums of column groups -- total, particles, fields -- at every
    sampled step within group_rel relative of the reference. Used where single columns of two correct realisations
    differ at the 10 % level (the laser-driven short pulse) and a per-column rule has no basis."""
    rule = ref["rule"]; grel = float(rule["group_rel"]); groups = rule["groups"]
    lines, ok = [], True
    if cols != ref["columns"]:
        return False, [f"  layout {cols} != reference {ref['columns']}: BAD"]
    rrows = {int(r[0]): r for r in ref["rows"]}
    if [r[0] for r in rows] != [int(r[0]) for r in ref["rows"]]:
        return False, [f"  sampled steps ({len(rows)}) != reference ({len(rrows)}): BAD"]
    idx = {c: i for i, c in enumerate(cols)}
    for name, members in groups.items():
        cols_g = cols[1:] if members == "all" else members
        worst = 0.0; bad = 0; first = None
        for r in rows:
            rr = rrows[r[0]]
            a = sum(r[idx[c]] for c in cols_g); b = sum(rr[idx[c]] for c in cols_g)
            d = abs(a - b) / abs(b) if b != 0 else (0.0 if a == 0 else float("inf"))
            worst = max(worst, d)
            if d > grel:
                bad += 1; first = first or (r[0], a, b)
        good = bad == 0; ok &= good
        lines.append(f"  group {name} ({'all columns' if members == 'all' else '+'.join(members)}): worst relative deviation {worst:.3e} (tol {grel:.2f}) violations={bad}"
                     + (f" first at step {first[0]}: run {first[1]:.6e} ref {first[2]:.6e}" if first else "") + (" ok" if good else " BAD"))
    e0 = sum(rows[0][1:]); drift = drift_of(rows)
    lines.append(f"  conservation: not applied (rule mode groups, open system); E_total(0)={e0:.6e}, max |E_total(step) - E_total(0)| / E_total(0) = {drift:.3e} (reference run: {ref['provenance'].get('observed_drift', float('nan')):.3e})")
    return ok, lines


def compare(cols, rows, ref):
    rule = ref["rule"]
    if rule.get("mode") == "groups":
        return compare_groups(cols, rows, ref)
    rel_tol, floor = float(rule["rel_tol"]), float(rule["floor"])
    drift_tol = None if rule["drift_tol"] is None else float(rule["drift_tol"])
    lines, ok = [], True
    if cols != ref["columns"]:
        return False, [f"  layout {cols} != reference {ref['columns']}: BAD"]
    rrows = {int(r[0]): r for r in ref["rows"]}
    steps = [r[0] for r in rows]
    if steps != [int(r[0]) for r in ref["rows"]]:
        return False, [f"  sampled steps {steps[:3]}..{steps[-1:]} ({len(steps)}) != reference ({len(rrows)}): BAD"]
    used = {c: 0.0 for c in cols[1:]}      # worst |run - ref| / tolerance per column (1.0 = at the limit)
    nbad = 0; first = None
    for r in rows:
        rr = rrows[r[0]]; total = sum(abs(v) for v in rr[1:])
        for i, c in enumerate(cols[1:], 1):
            a, b = r[i], rr[i]
            if not (math.isfinite(a) and math.isfinite(b)):
                nbad += 1; first = first or (r[0], c, a, b); continue
            tol = max(rel_tol * abs(b), floor * total)
            frac = abs(a - b) / tol if tol > 0 else (0.0 if a == b else float("inf"))
            used[c] = max(used[c], frac)
            if frac > 1.0:
                nbad += 1; first = first or (r[0], c, a, b)
    ok &= nbad == 0
    lines.append("  reference: worst |run-ref| / tol per column: " + ", ".join(f"{c} {used[c]:.3f}" for c in cols[1:])
                 + f" (tol = max({rel_tol:.0e} rel, {floor:.0e} of the row total)) violations={nbad}"
                 + (f" first at step {first[0]} {first[1]}: run {first[2]:.6e} ref {first[3]:.6e}" if first else "") + (" ok" if nbad == 0 else " BAD"))
    drift = drift_of(rows); e0 = sum(rows[0][1:])
    if drift_tol is None:
        lines.append(f"  conservation: not applied (open system, rule drift_tol none); E_total(0)={e0:.6e}, max |E_total(step) - E_total(0)| / E_total(0) = {drift:.3e} (reference run: {ref['provenance'].get('observed_drift', float('nan')):.3e})")
    else:
        good = drift <= drift_tol; ok &= good
        lines.append(f"  conservation: E_total(0)={e0:.6e}, max |E_total(step) - E_total(0)| / E_total(0) = {drift:.3e} (tol {drift_tol:.1e}) {'ok' if good else 'BAD'}")
    return ok, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--reference"); ap.add_argument("--label", default="")
    ap.add_argument("--make-reference", metavar="INPUT_ID"); ap.add_argument("--out"); ap.add_argument("--deck"); ap.add_argument("--log")
    ap.add_argument("--binary", default=""); ap.add_argument("--note", default="")
    ap.add_argument("--rel-tol", type=float); ap.add_argument("--floor", type=float)
    ap.add_argument("--drift-tol", type=lambda v: None if v == "none" else float(v), help="closed-system bound, or 'none' (open system: recorded, not checked)")
    ap.add_argument("--basis", default="")
    ap.add_argument("--groups-rel", type=float, help="rule mode 'groups' instead of per-column: total / particles / fields sums within this relative tolerance")
    a = ap.parse_args()
    label = a.label or os.path.basename(a.run)
    try:
        cols, rows = read_energies(a.run)
    except (OSError, ValueError) as ex:
        print(f"FAIL: VPIC energies check ({label}): cannot read the run's energies file: {ex}"); return 1
    if a.make_reference:
        if a.groups_rel is not None and not (a.out and a.deck and a.log):
            print("usage: --make-reference --groups-rel needs --out --deck --log", file=sys.stderr); return 2
        if a.groups_rel is None and not (a.out and a.deck and a.log and a.rel_tol is not None and "--drift-tol" in sys.argv and a.floor is not None):
            print("usage: --make-reference needs --out --deck --log --rel-tol --floor --drift-tol", file=sys.stderr); return 2
        if os.path.exists(a.out):
            print(f"refusing to overwrite the existing reference {a.out}", file=sys.stderr); return 2
        drift = drift_of(rows)
        ref = {"schema": SCHEMA, "input_id": a.make_reference,
               "provenance": {"made_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "deck": os.path.basename(a.deck), "deck_sha256": sha256(a.deck),
                              "energies_sha256": sha256(a.run), "log": os.path.basename(a.log), "log_sha256": sha256(a.log), "binary": a.binary,
                              "host": os.uname().nodename, "note": a.note, "observed_drift": drift},
               "rule": ({"mode": "groups", "group_rel": a.groups_rel, "basis": a.basis,
                         "groups": {"total": "all", "particles": [c for c in cols[1:] if c not in ("ex", "ey", "ez", "bx", "by", "bz")], "fields": [c for c in cols[1:] if c in ("ex", "ey", "ez", "bx", "by", "bz")]},
                         "text": "at every sampled step the sums total / particles / fields within group_rel relative of the reference; no per-column or conservation bound (statistical input)"}
                        if a.groups_rel is not None else
                        {"rel_tol": a.rel_tol, "floor": a.floor, "drift_tol": a.drift_tol, "basis": a.basis,
                        "text": "at every sampled step every column: |run - ref| <= max(rel_tol * |ref|, floor * E_total(ref row)); "
                                + ("|E_total(step) - E_total(0)| / E_total(0) <= drift_tol" if a.drift_tol is not None else "open system: no conservation bound (observed drift recorded in provenance)")}),
               "columns": cols, "rows": rows}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(ref, f, indent=1)
        print(f"reference written: {a.out} ({len(rows)} rows x {len(cols) - 1} columns, " + (f"groups rule rel {a.groups_rel:.2f}" if a.groups_rel is not None else f"rel_tol {a.rel_tol:.1e}, floor {a.floor:.1e}, drift_tol {a.drift_tol}") + f", observed drift {drift:.3e})")
        return 0
    if not a.reference:
        print("usage: --reference is required for a check", file=sys.stderr); return 2
    try:
        with open(a.reference) as f:
            ref = json.load(f)
        if ref.get("schema") != SCHEMA:
            raise ValueError(f"schema {ref.get('schema')!r} is not {SCHEMA}")
    except (OSError, ValueError) as ex:
        print(f"FAIL: VPIC energies check ({label}): cannot read the reference {a.reference}: {ex}"); return 1
    ok, lines = compare(cols, rows, ref)
    print(f"VPIC energies check ({label}): {os.path.basename(a.run)} ({len(rows)} rows) vs {os.path.basename(a.reference)} (input {ref.get('input_id')}; {ref['provenance'].get('binary', '')})")
    print("\n".join(lines))
    print(f"{'PASS' if ok else 'FAIL'}: VPIC energies check ({label})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
