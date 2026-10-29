#!/usr/bin/env python3
"""PENNANT correctness check: a run's per-zone profile (<name>.xy) against a reference.

    check_xy.py --run <name>.xy --reference <ref> [--label TEXT]             check (PASS / FAIL, exit 0 / 1)
    check_xy.py --make-reference <input id> --run <cpu run>.xy --out reference/<input id>.json
                --deck <deck.pnt> --log <cpu run.log> --binary TEXT [--omp-threads N] [--note TEXT]
                [--samples N] [--rel-tol X] [--zero-floor X]                 freeze a reference (never overwrites)

PENNANT writes, after the last cycle, the per-zone density zr, specific internal energy ze and pressure
zp of every zone with 8 significant digits (src/WriteXY.cc: scientific, setprecision(8)); upstream ships
such a profile as the gold standard <test>.xy.std for its five small decks (doc/pennantdoc.tex: "gold
standard file ... provided for reference"; the doc states no tolerance).

Two reference forms:
  * a profile file (upstream's <test>.xy.std, or any .xy): every zone of every column is compared;
  * a frozen JSON reference (schema hpcperf-pennant-reference-1) made ONCE from a CPU run of the SAME
    deck with upstream's own CPU implementation (lanl/PENNANT master v0.6, the version the cuda branch
    is a port of; provenance inside): per column its zone count, scale (max |value|), sum, min, max and
    a fixed stride sample of zones (every k-th zone, value as printed) -- the large decks' full
    profiles are 5-22 MB each and are not kept in git.

One comparison rule for a column, with the zone-independent reference scale S = max |ref|:
  * populated zones, |ref| >= zero_floor * S:   |run - ref| <= rel_tol * |ref|
  * all other zones (zero to the field's print precision): |run - ref| <= zero_floor * S
  * every value finite; the zone count equal; for a JSON reference also sum / min / max under the
    populated-zone rule (sum: rel_tol; min and max: the rule of their magnitude).
Default rel_tol = 1e-6 and zero_floor = 1e-8 (the gold-file rule): the profile carries 8 significant
digits, so two correct realisations of the same arithmetic agree to ~1e-7 relative in every populated
zone (1e-6 leaves a factor 10 for a last-digit rounding flip on both sides), and a value below 1e-8 of
the column's maximum is zero at the print precision of the field (the undisturbed near-vacuum zones
of the Sedov decks hold values of 1e-10 ... 1e-50 that are roundoff residues, and that differ between
every compiler and GPU: noise, never physics). Observed on 2026-10-20 with the unmodified-algorithm
CUDA build (B200, CUDA 13.2) and the CPU v0.6 build (conda GCC 13.3) against the five upstream gold
files: every zone at or above the floor identical to all 8 digits in every column, below the floor
|diff| <= 1.5e-18 * S -- both builds sit 1e6 and 1e10 inside the two bounds. A frozen JSON reference
records the rule it was made with (rel_tol, zero_floor); the check applies the recorded rule and never
a looser one from the command line.
"""
import argparse, hashlib, json, math, os, sys, time

SCHEMA = "hpcperf-pennant-reference-1"
COLS = ("zr", "ze", "zp")
DEFAULT_REL_TOL = 1e-6
DEFAULT_ZERO_FLOOR = 1e-8


def read_xy(path):
    """{column: [values in zone order]} from a PENNANT .xy profile; zone numbers must be 1..n contiguous."""
    cols, cur = {}, None
    with open(path, errors="replace") as f:
        for ln in f:
            p = ln.split()
            if not p:
                continue
            if p[0] == "#":
                cur = p[1]
                if cur not in COLS:
                    raise ValueError(f"{path}: unknown column '{cur}'")
                cols[cur] = []
                continue
            if cur is None or len(p) != 2:
                raise ValueError(f"{path}: malformed line {ln.rstrip()!r}")
            z, v = int(p[0]), float(p[1])
            if z != len(cols[cur]) + 1:
                raise ValueError(f"{path}: column {cur}: zone {z} out of order (expected {len(cols[cur]) + 1})")
            cols[cur].append(v)
    for c in COLS:
        if c not in cols:
            raise ValueError(f"{path}: column {c} missing")
    if len({len(cols[c]) for c in COLS}) != 1:
        raise ValueError(f"{path}: columns have different zone counts")
    return cols


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def within(run, ref, scale, rel_tol, zero_floor):
    """(ok, kind, deviation): the rule for one value; deviation is relative for populated zones, absolute/scale otherwise."""
    if not (math.isfinite(run) and math.isfinite(ref)):
        return False, "nonfinite", float("inf")
    if scale == 0.0:                      # the reference column is identically zero: so must the run's be
        return run == 0.0, "floor", abs(run)
    if abs(ref) >= zero_floor * scale:
        d = abs(run - ref) / abs(ref)
        return d <= rel_tol, "populated", d
    d = abs(run - ref) / scale if scale > 0 else abs(run - ref)
    return d <= zero_floor, "floor", d


def compare_full(run, ref, rel_tol, zero_floor):
    lines, ok = [], True
    for c in COLS:
        x, y = run[c], ref[c]
        if len(x) != len(y):
            lines.append(f"  {c}: zone count {len(x)} != reference {len(y)}: BAD"); ok = False; continue
        scale = max(abs(v) for v in y)
        worst = {"populated": 0.0, "floor": 0.0}; nbad = 0; npop = 0; first_bad = None
        for z, (u, v) in enumerate(zip(x, y), 1):
            good, kind, d = within(u, v, scale, rel_tol, zero_floor)
            if kind == "populated":
                npop += 1
            if kind in worst:
                worst[kind] = max(worst[kind], d)
            if not good:
                nbad += 1
                if first_bad is None:
                    first_bad = (z, u, v, kind)
        ok &= nbad == 0
        lines.append(f"  {c}: zones={len(y)} populated={npop} scale={scale:.6e} max rel dev (populated)={worst['populated']:.3e}"
                     f" (tol {rel_tol:.1e}) max |dev|/scale (floor zones)={worst['floor']:.3e} (tol {zero_floor:.1e}) violations={nbad}"
                     + (f" first at zone {first_bad[0]}: run {first_bad[1]:.8e} ref {first_bad[2]:.8e} [{first_bad[3]}]" if first_bad else "")
                     + (" ok" if nbad == 0 else " BAD"))
    return ok, lines


def column_summary(vals, samples):
    n = len(vals); stride = max(1, -(-n // samples))
    idx = list(range(0, n, stride))
    imax = max(range(n), key=lambda i: abs(vals[i]))
    if imax not in idx:
        idx.append(imax)
    idx.sort()
    return {"n": n, "scale": max(abs(v) for v in vals), "sum": math.fsum(vals), "min": min(vals), "max": max(vals),
            "sample_stride": stride, "samples": [[i + 1, vals[i]] for i in idx]}


def compare_json(run, ref):
    rule = ref["rule"]; rel_tol, zero_floor = float(rule["rel_tol"]), float(rule["zero_floor"])
    lines, ok = [], True
    for c in COLS:
        x, rc = run[c], ref["columns"][c]
        if len(x) != rc["n"]:
            lines.append(f"  {c}: zone count {len(x)} != reference {rc['n']}: BAD"); ok = False; continue
        scale = float(rc["scale"])
        worst = {"populated": 0.0, "floor": 0.0}; nbad = 0; npop = 0; first_bad = None
        for z, v in rc["samples"]:
            u = x[int(z) - 1]
            good, kind, d = within(u, float(v), scale, rel_tol, zero_floor)
            if kind == "populated":
                npop += 1
            if kind in worst:
                worst[kind] = max(worst[kind], d)
            if not good:
                nbad += 1
                if first_bad is None:
                    first_bad = (int(z), u, float(v), kind)
        agg = []
        for key, val in (("sum", math.fsum(x)), ("min", min(x)), ("max", max(x))):
            r = float(rc[key])
            good, kind, d = within(val, r, scale if key != "sum" else abs(r), rel_tol, zero_floor) if key != "sum" \
                else ((abs(val - r) <= rel_tol * abs(r)) if r != 0 else (abs(val) <= zero_floor * scale), "sum", (abs(val - r) / abs(r) if r != 0 else abs(val)))
            agg.append(f"{key} dev={d:.3e}{'' if good else ' BAD'}")
            if not good:
                nbad += 1
        ok &= nbad == 0
        lines.append(f"  {c}: zones={rc['n']} sampled={len(rc['samples'])} populated={npop} scale={scale:.6e}"
                     f" max rel dev (populated)={worst['populated']:.3e} (tol {rel_tol:.1e}) max |dev|/scale (floor)={worst['floor']:.3e}"
                     f" (tol {zero_floor:.1e}) {' '.join(agg)} violations={nbad}"
                     + (f" first at zone {first_bad[0]}: run {first_bad[1]:.8e} ref {first_bad[2]:.8e} [{first_bad[3]}]" if first_bad else "")
                     + (" ok" if nbad == 0 else " BAD"))
    return ok, lines, rel_tol, zero_floor


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the run's <name>.xy profile")
    ap.add_argument("--reference", help="reference: <test>.xy.std / .xy profile, or a frozen reference/<id>.json")
    ap.add_argument("--label", default="")
    ap.add_argument("--make-reference", metavar="INPUT_ID", help="freeze --run (a CPU reference run) into --out")
    ap.add_argument("--out"); ap.add_argument("--deck"); ap.add_argument("--log"); ap.add_argument("--binary", default="")
    ap.add_argument("--omp-threads", type=int); ap.add_argument("--note", default="")
    ap.add_argument("--samples", type=int, default=2048, help="sampled zones per column in a frozen reference")
    ap.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL, help="populated-zone relative tolerance (frozen into a JSON reference; a .xy reference always uses the default)")
    ap.add_argument("--zero-floor", type=float, default=DEFAULT_ZERO_FLOOR, help="zero floor as a fraction of the column scale (same remark)")
    a = ap.parse_args()
    try:
        run = read_xy(a.run)
    except (OSError, ValueError) as ex:
        print(f"FAIL: PENNANT xy check ({a.label}): cannot read the run profile: {ex}"); return 1

    if a.make_reference:
        if not (a.out and a.deck and a.log):
            print("usage: --make-reference needs --out, --deck and --log", file=sys.stderr); return 2
        if os.path.exists(a.out):
            print(f"refusing to overwrite the existing reference {a.out}", file=sys.stderr); return 2
        ref = {"schema": SCHEMA, "input_id": a.make_reference,
               "provenance": {"made_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "deck": os.path.basename(a.deck),
                              "deck_sha256": sha256(a.deck), "profile": os.path.basename(a.run), "profile_sha256": sha256(a.run),
                              "log": os.path.basename(a.log), "log_sha256": sha256(a.log), "binary": a.binary,
                              "omp_threads": a.omp_threads, "note": a.note},   # no host name (machine identity)
               "rule": {"rel_tol": a.rel_tol, "zero_floor": a.zero_floor,
                        "text": "populated zones (|ref| >= zero_floor * max|ref|): |run - ref| <= rel_tol * |ref|; other zones: |run - ref| <= zero_floor * max|ref|; sum rel_tol; min/max by magnitude"},
               "columns": {c: column_summary(run[c], a.samples) for c in COLS}}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        import re
        text = re.sub(r'\[\s*(\d+),\s*(-?[0-9.eE+-]+)\s*\]', r'[\1, \2]', json.dumps(ref, indent=1))   # one sample per line
        with open(a.out, "w") as f:
            f.write(text + "\n")
        print(f"reference written: {a.out} ({sum(len(ref['columns'][c]['samples']) for c in COLS)} sampled values, "
              f"rel_tol {a.rel_tol:.1e}, zero_floor {a.zero_floor:.1e})")
        return 0

    if not a.reference:
        print("usage: --reference is required for a check", file=sys.stderr); return 2
    label = a.label or os.path.basename(a.run)
    if a.reference.endswith(".json"):
        try:
            with open(a.reference) as f:
                ref = json.load(f)
            if ref.get("schema") != SCHEMA:
                raise ValueError(f"schema {ref.get('schema')!r} is not {SCHEMA}")
        except (OSError, ValueError) as ex:
            print(f"FAIL: PENNANT xy check ({label}): cannot read the frozen reference {a.reference}: {ex}"); return 1
        ok, lines, rel_tol, zero_floor = compare_json(run, ref)
        kind = f"frozen reference {os.path.basename(a.reference)} (input {ref.get('input_id')}; {ref['provenance'].get('binary', '')})"
    else:
        try:
            refp = read_xy(a.reference)
        except (OSError, ValueError) as ex:
            print(f"FAIL: PENNANT xy check ({label}): cannot read the reference profile {a.reference}: {ex}"); return 1
        rel_tol, zero_floor = DEFAULT_REL_TOL, DEFAULT_ZERO_FLOOR
        ok, lines = compare_full(run, refp, rel_tol, zero_floor)
        kind = f"profile {os.path.basename(a.reference)} (every zone)"
    print(f"PENNANT xy check ({label}): {os.path.basename(a.run)} vs {kind}; rule rel_tol={rel_tol:.1e} zero_floor={zero_floor:.1e}")
    print("\n".join(lines))
    print(f"{'PASS' if ok else 'FAIL'}: PENNANT xy check ({label})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
