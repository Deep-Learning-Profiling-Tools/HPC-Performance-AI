#!/usr/bin/env python3
"""Strict plotfile comparison for the Nyx validator (wraps AMReX fcompare, never trusts it blindly).

    nyx_fcompare_check.py <ref_plotfile> <plotfile> --rel_tol R [--abs_tol_zero_ref A] [--diagnostic VAR ...]
                          --fcompare PATH --fextrema PATH [--out FILE] [--label L]

Steps (every one must succeed; a structural problem is exit 2, a tolerance violation exit 1, agreement exit 0):
 1. Header check (independent of fcompare): both plotfile Headers are parsed; the variable SET, the
    dimension, the number of levels, the simulation time (relative difference <= 1e-12), the domain box,
    the cell sizes and the box array of every level must be identical.
 2. Raw finiteness: `fextrema` on BOTH plotfiles; every variable of the header set must appear exactly once
    with finite min/max. A reference variable whose min == max == 0 is a ZERO-REFERENCE field.
 3. fcompare (`-n 0 --rel_tol R --abort_if_not_all_found`): stdout is parsed strictly -- one `level = L`
    block per level, exactly one row per variable of the header set, no duplicates, no extra rows, no
    `< ... >` message rows (missing variable, NaN), no WARNING/ERROR lines; abs and rel must parse as
    numbers (inf/nan are parsed, never dropped).
 4. Decision per variable (all levels):
      diagnostic variables : parsed and reported only (still must be finite in step 2 and parseable in 3);
      zero-reference field : abs error <= abs_tol_zero_ref (default 0 = exact) -- the relative error is
                             undefined there (fcompare prints inf or 0) and is never used;
      otherwise            : rel must be finite and <= rel_tol. A non-finite rel on a non-zero reference
                             is a violation, not a skip.
 5. Consistency with the tool: if the parsed verdict is AGREE while fcompare exited non-zero, or DISAGREE
    while it exited 0, the run is a STRUCTURAL failure (the parser and the tool must agree).
The full report (headers, extrema, fcompare output, per-variable decisions) is written to --out.
"""
import argparse, math, os, re, subprocess, sys

STRUCT, TOL, OK = 2, 1, 0


class Fail(Exception):
    def __init__(self, code, msg):
        super().__init__(msg); self.code = code


def read_header(plt):
    """Parse an AMReX plotfile Header: names, dim, time, finest level, domain, cell sizes, per-level boxes."""
    p = os.path.join(plt, "Header")
    if not os.path.isfile(p): raise Fail(STRUCT, f"{p} missing")
    L = [l.rstrip("\n") for l in open(p)]
    try:
        i = 0; version = L[i].strip(); i += 1
        nvars = int(L[i]); i += 1
        names = [L[i + k].strip() for k in range(nvars)]; i += nvars
        dim = int(L[i]); i += 1
        time = float(L[i]); i += 1
        finest = int(L[i]); i += 1
        prob_lo = [float(x) for x in L[i].split()]; i += 1
        prob_hi = [float(x) for x in L[i].split()]; i += 1
        i += 1  # refinement ratios (empty line for a single level)
        dom = re.findall(r"\(\(([-\d, ]+)\) \(([-\d, ]+)\) \(([-\d, ]+)\)\)", L[i]); i += 1
        steps = L[i].split(); i += 1
        dx = [[float(x) for x in L[i + k].split()] for k in range(finest + 1)]; i += finest + 1
        coord = int(L[i]); i += 1
        i += 1  # bwidth
        levels = []
        for lev in range(finest + 1):
            hdr = L[i].split(); i += 1
            if int(hdr[0]) != lev: raise ValueError(f"level header {hdr} for level {lev}")
            ngrids = int(hdr[1]); ltime = float(hdr[2])
            lstep = int(L[i]); i += 1
            boxes = []
            for g in range(ngrids):
                ext = []
                for d in range(dim):
                    lo, hi = [float(x) for x in L[i].split()]; i += 1
                    ext.append((lo, hi))
                boxes.append(tuple(ext))
            path = L[i].strip(); i += 1
            levels.append(dict(ngrids=ngrids, time=ltime, step=lstep, boxes=boxes, path=path))
    except (IndexError, ValueError) as ex:
        raise Fail(STRUCT, f"{p}: cannot parse AMReX plotfile Header ({ex})")
    if len(set(names)) != nvars: raise Fail(STRUCT, f"{p}: duplicate variable names in Header")
    return dict(version=version, nvars=nvars, names=names, dim=dim, time=time, finest=finest,
                prob_lo=prob_lo, prob_hi=prob_hi, domain=dom, dx=dx, coord=coord, levels=levels)


def check_headers(a, b):
    out = []
    if set(a["names"]) != set(b["names"]):
        raise Fail(STRUCT, f"variable sets differ: only-in-ref={sorted(set(a['names'])-set(b['names']))} only-in-test={sorted(set(b['names'])-set(a['names']))}")
    if a["names"] != b["names"]: out.append(f"    note: variable ORDER differs (same set); matched by name")
    for k in ("dim", "finest", "prob_lo", "prob_hi", "domain", "dx", "coord"):
        if a[k] != b[k]: raise Fail(STRUCT, f"Header field {k} differs: ref={a[k]} test={b[k]}")
    if abs(a["time"] - b["time"]) > 1e-12 * max(abs(a["time"]), 1e-300):
        raise Fail(STRUCT, f"simulation time differs: ref={a['time']!r} test={b['time']!r}")
    for lev, (la, lb) in enumerate(zip(a["levels"], b["levels"])):
        if la["ngrids"] != lb["ngrids"] or la["boxes"] != lb["boxes"]:
            raise Fail(STRUCT, f"level {lev}: box array differs ({la['ngrids']} vs {lb['ngrids']} grids or different extents)")
        if la["step"] != lb["step"]: raise Fail(STRUCT, f"level {lev}: step differs {la['step']} vs {lb['step']}")
    out.append(f"    headers: {a['nvars']} variables, dim {a['dim']}, {a['finest']+1} level(s), time {a['time']!r}, "
               f"{sum(l['ngrids'] for l in a['levels'])} grids -- identical structure")
    return out


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout


def parse_float(tok):
    try:
        return float(tok)
    except ValueError:
        raise Fail(STRUCT, f"cannot parse number {tok!r}")


def extrema(fextrema, plt, names):
    rc, out = run([fextrema, plt])
    if rc != 0: raise Fail(STRUCT, f"fextrema failed on {plt} (rc={rc}): {out.strip()[-300:]}")
    got = {}
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 3 and f[0] in names:
            if f[0] in got: raise Fail(STRUCT, f"fextrema: variable {f[0]} listed twice for {plt}")
            lo, hi = parse_float(f[-2]), parse_float(f[-1])
            if not (math.isfinite(lo) and math.isfinite(hi)): raise Fail(STRUCT, f"non-finite raw values in {plt}: {f[0]} min={f[-2]} max={f[-1]}")
            got[f[0]] = (lo, hi)
    missing = [n for n in names if n not in got]
    if missing: raise Fail(STRUCT, f"fextrema output for {plt} lacks variables {missing}")
    return got, out


def parse_fcompare(text, names, nlevels):
    """Return {(level, name): (abs, rel)}; raise Fail(STRUCT) on any structural anomaly."""
    rows, level, seen_levels = {}, None, []
    msg_re = re.compile(r"^\s*(\S+)\s+<\s*(.*?)\s*>\s*$")
    num_re = re.compile(r"^\s*(\S+)\s+(\S+)\s+(\S+)\s*$")
    for line in text.splitlines():
        s = line.strip()
        if not s: continue
        if re.search(r"\b(ERROR|WARNING)\b", s):
            raise Fail(STRUCT, f"fcompare reported: {s}")
        m = re.match(r"^level\s*=\s*(\d+)$", s)
        if m:
            level = int(m.group(1)); seen_levels.append(level); continue
        if s.startswith("variable name") or s.startswith("(||A - B||") or set(s) <= set("-"): continue
        if s.startswith("PLOTFILE AGREE"): continue
        m = msg_re.match(line)
        if m and m.group(1) in names:
            raise Fail(STRUCT, f"fcompare row for {m.group(1)} is a message, not a value: <{m.group(2)}>")
        m = num_re.match(line)
        if m and m.group(1) in names:
            if level is None: raise Fail(STRUCT, f"table row before any 'level =' line: {s}")
            key = (level, m.group(1))
            if key in rows: raise Fail(STRUCT, f"duplicate fcompare row for {m.group(1)} at level {level}")
            rows[key] = (parse_float(m.group(2)), parse_float(m.group(3)))
            continue
        # anything else that looks like a table row but is not a known variable
        if m and re.match(r"^[-+0-9.eEinfa]+$", m.group(2)) and re.match(r"^[-+0-9.eEinfa]+$", m.group(3)):
            raise Fail(STRUCT, f"fcompare row for unexpected variable {m.group(1)!r}")
    if seen_levels != list(range(nlevels)): raise Fail(STRUCT, f"fcompare levels {seen_levels} != expected {list(range(nlevels))}")
    for lev in range(nlevels):
        missing = [n for n in names if (lev, n) not in rows]
        if missing: raise Fail(STRUCT, f"fcompare table incomplete at level {lev}: missing rows for {missing}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("test")
    ap.add_argument("--rel_tol", type=float, required=True)
    ap.add_argument("--abs_tol_zero_ref", type=float, default=0.0, help="absolute tolerance for fields whose reference is identically zero (default 0 = exact)")
    ap.add_argument("--diagnostic", nargs="*", default=[], help="variables reported but not gated (must still be present and finite)")
    ap.add_argument("--fcompare", required=True); ap.add_argument("--fextrema", required=True)
    ap.add_argument("--out"); ap.add_argument("--label", default="")
    a = ap.parse_args()
    report = [f"# nyx_fcompare_check {a.label}", f"ref={a.ref}", f"test={a.test}", f"rel_tol={a.rel_tol!r} abs_tol_zero_ref={a.abs_tol_zero_ref!r} diagnostic={a.diagnostic}"]
    code, summary = OK, ""
    try:
        ha, hb = read_header(a.ref), read_header(a.test)
        report += check_headers(ha, hb)
        names = ha["names"]; nlev = ha["finest"] + 1
        unknown = [d for d in a.diagnostic if d not in names]
        if unknown: raise Fail(STRUCT, f"diagnostic variables not in the plotfile: {unknown}")
        exa, txa = extrema(a.fextrema, a.ref, names); exb, txb = extrema(a.fextrema, a.test, names)
        report += ["## fextrema ref", txa.rstrip(), "## fextrema test", txb.rstrip()]
        zero_ref = {n for n in names if exa[n] == (0.0, 0.0)}
        rc, out = run([a.fcompare, "-n", "0", "--rel_tol", repr(a.rel_tol), "--abort_if_not_all_found", a.ref, a.test])
        report += [f"## fcompare rc={rc}", out.rstrip()]
        rows = parse_fcompare(out, names, nlev)
        bad, worst, diag = [], (None, 0.0), []
        for (lev, n), (ab, rel) in sorted(rows.items()):
            if n in a.diagnostic:
                diag.append(f"{n}@L{lev}: abs={ab:.6e} rel={rel:.6e}")
                if not (math.isfinite(ab)): raise Fail(STRUCT, f"diagnostic {n} at level {lev}: non-finite absolute error {ab}")
                continue
            if n in zero_ref:
                ok = math.isfinite(ab) and ab <= a.abs_tol_zero_ref
                report.append(f"    L{lev} {n:24s} ZERO-REFERENCE abs={ab:.6e} (tol {a.abs_tol_zero_ref:g}) {'ok' if ok else 'BAD'}")
                if not ok: bad.append(f"{n}@L{lev}(abs={ab:.3e}, zero reference)")
                continue
            ok = math.isfinite(rel) and rel <= a.rel_tol
            report.append(f"    L{lev} {n:24s} abs={ab:.6e} rel={rel:.6e} {'ok' if ok else 'BAD'}")
            if not ok: bad.append(f"{n}@L{lev}(rel={rel:.3e})")
            if math.isfinite(rel) and rel > worst[1]: worst = (f"{n}@L{lev}", rel)
        verdict = OK if not bad else TOL
        if verdict == OK and rc != 0 and not (a.diagnostic and rc == 1):
            # fcompare found something we did not (its own rc=1 for an over-tolerance diagnostic variable is expected)
            raise Fail(STRUCT, f"parsed verdict AGREE but fcompare exited {rc} -- parser/tool inconsistency")
        if verdict == TOL and rc == 0:
            raise Fail(STRUCT, "parsed verdict DISAGREE but fcompare exited 0 -- parser/tool inconsistency")
        code = verdict
        summary = (f"RESULT: {'AGREE' if code == OK else 'DISAGREE'} nvars={len(names)} levels={nlev} rel_tol={a.rel_tol:g} "
                   f"worst={worst[0]} {worst[1]:.3e} zero_ref={sorted(zero_ref) or 'none'} bad={bad or 'none'} diagnostic={diag or 'none'}")
    except Fail as ex:
        code, summary = ex.code, f"RESULT: STRUCTURAL_FAIL {ex}"
    report.append(summary)
    text = "\n".join(report) + "\n"
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, "w").write(text)
    print(summary)
    return code


if __name__ == "__main__":
    sys.exit(main())
