#!/usr/bin/env python3
"""Nekbone correctness check: the timed CG solve of a run against a frozen reference of the same input.

    check_cg.py --run <stdout of run.sh> --reference reference/<id>.json [--label TEXT]      check (PASS / FAIL, exit 0 / 1)
    check_cg.py --make-reference <input id> --run <stdout> --out reference/<id>.json --size <SIZE> --rea <data.rea>
                --binary TEXT --r0-tol X --x-tol X --reduction-factor X [--basis TEXT] [--note TEXT]        freeze (never overwrites)

What a Nekbone run prints about its solves (src/driver.f, src/cg.f): for every (nelt, nx1) combination of
data.rea it runs the conjugate-gradient solve twice -- a warm-up solve and the timed one (set_timer_flop_cnt) --
and each solve prints `cg:   0  <r0>` (the weighted norm of the masked right-hand side) and after its fixed
100 iterations `cg: 101  <rN> <alpha> <beta> <pap>` (the residual norm and the last step's scalars, 5
significant digits). This benchmark's driver.f additionally prints, after the timed solve and outside the
timed region, `HPCPERF solution: xnorm=<...> xf=<...>`: the weighted norm sqrt(x^T C x) of the solution and the
functional x^T C f (16 digits), computed on the host from the solution and the right-hand side.

Checked quantities of the timed (second) solve against the frozen reference (a run of the SAME input with the
CPU build of the same source, upstream's makenek.mpi code path, or where stated the unmodified CUDA build):
  * r0      within r0_tol relative     -- the right-hand side, mask and multiplicity (printed to 5 digits)
  * xnorm   within x_tol relative       -- the solution of the SPD system is unique: a wrong operator (geometric
  * xf      within x_tol relative          factors, gather-scatter, mask) converges to a different solution
  * rN / r0 <= reduction_factor x (reference rN / r0)  -- the solve converged as the reference did (a broken
            reduction, a non-symmetric operator or a lost iteration leaves the residual orders of magnitude above)
  * both solves present, the run reached 'Avg MFlops' (the program's normal end)
The tolerances are recorded in the reference at freeze time from the stated basis (README); the check applies
the recorded rule and never a looser one from the command line.
"""
import argparse, hashlib, json, math, os, re, sys, time

SCHEMA = "hpcperf-nekbone-cg-reference-1"
RE_CG0 = re.compile(r"^cg:\s+0\s+([0-9.eE+-]+)\s*$", re.M)
RE_CGN = re.compile(r"^cg:\s+(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*$", re.M)
RE_SOL = re.compile(r"^HPCPERF solution: xnorm=\s*([0-9.eE+-]+)\s+xf=\s*([0-9.eE+-]+)\s*$", re.M)
RE_NELT = re.compile(r"^nelt =\s+(\d+), np =\s+(\d+), nx1 =\s+(\d+), elements =\s+(\d+)\s*$", re.M)


def parse(text):
    """the solves of a run: [{'r0', 'iter', 'rN', 'alpha', 'beta', 'pap'}, ...] in print order, the solution line, the size line"""
    solves = []
    for m0 in RE_CG0.finditer(text):
        mN = RE_CGN.search(text, m0.end())
        if not mN:
            raise ValueError("a 'cg:   0' line without its final 'cg: <iter>' line")
        solves.append({"r0": float(m0.group(1)), "iter": int(mN.group(1)), "rN": float(mN.group(2)),
                       "alpha": float(mN.group(3)), "beta": float(mN.group(4)), "pap": float(mN.group(5))})
    sol = RE_SOL.search(text)
    size = RE_NELT.search(text)
    if len(solves) < 2:
        raise ValueError(f"{len(solves)} solve(s) printed; the warm-up and the timed solve are expected")
    if sol is None:
        raise ValueError("no 'HPCPERF solution:' line (the timed solve did not finish)")
    if size is None:
        raise ValueError("no 'nelt = ..., np = ..., nx1 = ...' summary line")
    if "Avg MFlops" not in text:
        raise ValueError("no 'Avg MFlops' line: the program did not reach its normal end")
    return {"solves": solves, "xnorm": float(sol.group(1)), "xf": float(sol.group(2)),
            "nelt": int(size.group(1)), "np": int(size.group(2)), "nx1": int(size.group(3)), "elements": int(size.group(4))}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(a, b):
    return abs(a - b) / abs(b) if b != 0 else (0.0 if a == 0 else float("inf"))


def compare(run, ref):
    rule, rs = ref["rule"], ref["run"]
    r0_tol, x_tol, red = float(rule["r0_tol"]), float(rule["x_tol"]), float(rule["reduction_factor"])
    lines, ok = [], True
    for k in ("nelt", "np", "nx1"):
        good = run[k] == rs[k]; ok &= good
        lines.append(f"  {k}: run {run[k]} reference {rs[k]} {'ok' if good else 'BAD'}")
    if len(run["solves"]) != len(rs["solves"]):
        lines.append(f"  solves printed: run {len(run['solves'])} reference {len(rs['solves'])} BAD"); return False, lines
    t, tr = run["solves"][-1], rs["solves"][-1]          # the timed solve is the last one printed
    good = t["iter"] == tr["iter"]; ok &= good
    lines.append(f"  iterations: {t['iter']} (reference {tr['iter']}) {'ok' if good else 'BAD'}")
    d = rel(t["r0"], tr["r0"]); good = d <= r0_tol; ok &= good
    lines.append(f"  r0 (masked RHS norm): run {t['r0']:.5e} reference {tr['r0']:.5e} rel {d:.2e} (tol {r0_tol:.1e}) {'ok' if good else 'BAD'}")
    for k in ("xnorm", "xf"):
        d = rel(run[k], rs[k]); good = d <= x_tol; ok &= good
        lines.append(f"  {k}: run {run[k]:.16e} reference {rs[k]:.16e} rel {d:.2e} (tol {x_tol:.1e}) {'ok' if good else 'BAD'}")
    red_run, red_ref = t["rN"] / t["r0"], tr["rN"] / tr["r0"]
    good = math.isfinite(red_run) and red_run <= red * red_ref; ok &= good
    lines.append(f"  residual reduction rN/r0: run {red_run:.3e} reference {red_ref:.3e} (limit {red:g} x reference = {red * red_ref:.3e}) {'ok' if good else 'BAD'}")
    w, wr = run["solves"][0], rs["solves"][0]
    lines.append(f"  warm-up solve (not checked): r0 {w['r0']:.5e} (reference {wr['r0']:.5e}), rN {w['rN']:.5e} (reference {wr['rN']:.5e})")
    return ok, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True); ap.add_argument("--reference"); ap.add_argument("--label", default="")
    ap.add_argument("--make-reference", metavar="INPUT_ID"); ap.add_argument("--out"); ap.add_argument("--size"); ap.add_argument("--rea")
    ap.add_argument("--binary", default=""); ap.add_argument("--note", default=""); ap.add_argument("--basis", default="")
    ap.add_argument("--r0-tol", type=float); ap.add_argument("--x-tol", type=float); ap.add_argument("--reduction-factor", type=float)
    a = ap.parse_args()
    label = a.label or os.path.basename(a.run)
    try:
        with open(a.run, errors="replace") as f:
            run = parse(f.read())
    except (OSError, ValueError) as ex:
        print(f"FAIL: Nekbone CG check ({label}): cannot read the run: {ex}"); return 1
    if a.make_reference:
        if not (a.out and a.size and a.rea and a.r0_tol is not None and a.x_tol is not None and a.reduction_factor is not None):
            print("usage: --make-reference needs --out --size --rea --r0-tol --x-tol --reduction-factor", file=sys.stderr); return 2
        if os.path.exists(a.out):
            print(f"refusing to overwrite the existing reference {a.out}", file=sys.stderr); return 2
        ref = {"schema": SCHEMA, "input_id": a.make_reference,
               "provenance": {"made_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "SIZE_sha256": sha256(a.size), "data_rea_sha256": sha256(a.rea),
                              "log": os.path.basename(a.run), "log_sha256": sha256(a.run), "binary": a.binary, "note": a.note},   # no host name (machine identity)
               "rule": {"r0_tol": a.r0_tol, "x_tol": a.x_tol, "reduction_factor": a.reduction_factor, "basis": a.basis,
                        "text": "timed solve: r0 within r0_tol relative, xnorm and xf within x_tol relative, rN/r0 <= reduction_factor x the reference's rN/r0; nelt / np / nx1 / iteration count equal"},
               "run": run}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(ref, f, indent=1)
        t = run["solves"][-1]
        print(f"reference written: {a.out} (nelt {run['nelt']} nx1 {run['nx1']}; timed solve r0 {t['r0']:.5e} rN {t['rN']:.5e} xnorm {run['xnorm']:.16e} xf {run['xf']:.16e}; "
              f"r0_tol {a.r0_tol:.1e} x_tol {a.x_tol:.1e} reduction_factor {a.reduction_factor:g})")
        return 0
    if not a.reference:
        print("usage: --reference is required for a check", file=sys.stderr); return 2
    try:
        with open(a.reference) as f:
            ref = json.load(f)
        if ref.get("schema") != SCHEMA:
            raise ValueError(f"schema {ref.get('schema')!r} is not {SCHEMA}")
    except (OSError, ValueError) as ex:
        print(f"FAIL: Nekbone CG check ({label}): cannot read the reference {a.reference}: {ex}"); return 1
    ok, lines = compare(run, ref)
    print(f"Nekbone CG check ({label}): {os.path.basename(a.run)} vs {os.path.basename(a.reference)} (input {ref.get('input_id')}; {ref['provenance'].get('binary', '')})")
    print("\n".join(lines))
    print(f"{'PASS' if ok else 'FAIL'}: Nekbone CG check ({label})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
