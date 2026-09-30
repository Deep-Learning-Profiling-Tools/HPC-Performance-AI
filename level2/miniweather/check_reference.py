#!/usr/bin/env python3
"""miniWeather: the registered inputs' correctness check against a CPU reference run of the same grid.

    check_reference.py --log <run stdout> --nx N --nz N --sim-time T --data-spec SPEC     PASS / FAIL, exit 0 / 1
    check_reference.py --make-reference --nx N --nz N --sim-time T --data-spec SPEC <cpu log> [...]

The reference of every registered input is the same parallel_for source built for the CPU
(build.sh OPENMP: YAKL's OpenMP backend, -O3 without fast-math) for the exact grid, simulation time
and data spec (a compile-time configuration in miniWeather), run to completion on this node
(2026-09-30; provenance in reference/<key>.json). miniWeather prints two numbers after the run:
d_mass and d_te, the relative change of the domain-integrated mass and total energy.

Benchmark-wide rule (one rule for every registered grid, fixed BEFORE any candidate exists):
  |d_mass| <= 1e-9                              upstream check_output.sh's own mass criterion, unchanged
  |d_te - d_te_reference| <= D_TE_ABS_TOL       the same-grid CPU reference, absolute
D_TE_ABS_TOL = 1e-8. Observed before the rule was fixed (2026-09-30): the unoptimized CUDA build
(--use_fast_math) reproduced the CPU reference's d_te to all 7 printed digits at thermal 1024x512
(1.239597e-04 on both); the GPU's own 10 repeats (2026-09-28) reproduced d_te to all printed digits
at the three thermal / collision grids and scattered by 1.8e-11 absolute at gravity waves (d_te =
3.16e-8, atomics). The remaining grids' CPU values are recorded in reference/<key>.json as their runs
finish, with the observed difference. 1e-8 is >= 100 x the largest difference seen between two correct
realisations and about a third of the smallest registered |d_te|, while the physics the number
describes -- the energy the hyper-viscosity removes over 1000 s -- moves d_te by 1e-5 or more when it
is wrong (upstream's own validation bound is 4.5e-5). Nothing here is derived from a candidate.
"""
import argparse, hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, "reference")
D_MASS_ABS_TOL = 1e-9
D_TE_ABS_TOL = 1e-8
RX_MASS = re.compile(r"^d_mass:\s*([0-9.eE+-]+)", re.M)
RX_TE = re.compile(r"^d_te:\s*([0-9.eE+-]+)", re.M)


def key(a):
    return f"{a.data_spec.lower().replace('data_spec_', '').replace('_', '-')}-{a.nx}x{a.nz}-{a.sim_time}s"


def parse(path):
    t = open(path, errors="replace").read()
    m, e = RX_MASS.findall(t), RX_TE.findall(t)
    return {"d_mass": float(m[-1]) if m else None, "d_te": float(e[-1]) if e else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log")
    ap.add_argument("--nx", type=int, required=True); ap.add_argument("--nz", type=int, required=True)
    ap.add_argument("--sim-time", type=int, required=True); ap.add_argument("--data-spec", required=True)
    ap.add_argument("--make-reference", nargs="+", metavar="CPU_LOG")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    p = os.path.join(REF_DIR, f"{key(a)}.json")
    if a.make_reference:
        runs = [{"log": os.path.basename(lg), "sha256": hashlib.sha256(open(lg, "rb").read()).hexdigest(), **parse(lg)} for lg in a.make_reference]
        if any(r["d_te"] is None or r["d_mass"] is None for r in runs):
            sys.exit("a reference log has no d_mass / d_te line")
        tes = [r["d_te"] for r in runs]
        ref = {"schema": "hpcperf-miniweather-reference-1", "key": key(a), "nx": a.nx, "nz": a.nz, "sim_time": a.sim_time,
               "data_spec": a.data_spec, "note": a.note, "d_mass": runs[0]["d_mass"], "d_te": runs[0]["d_te"],
               "d_te_spread_abs": max(tes) - min(tes), "runs": runs,
               "rule": {"d_mass": f"abs <= {D_MASS_ABS_TOL}", "d_te": f"abs diff to reference <= {D_TE_ABS_TOL}"}}
        os.makedirs(REF_DIR, exist_ok=True)
        json.dump(ref, open(p, "w"), indent=1)
        print(f"wrote {p}: d_te {ref['d_te']:.7e} (spread over {len(runs)} run(s): {ref['d_te_spread_abs']:.2e}), d_mass {ref['d_mass']:.3e}")
        return 0
    if not a.log:
        ap.error("--log or --make-reference")
    if not os.path.isfile(p):
        print(f"FAIL: miniweather reference check -- no CPU reference for {key(a)} ({p} missing)"); return 1
    ref = json.load(open(p))
    v = parse(a.log)
    fails, notes = [], []
    if v["d_mass"] is None or v["d_te"] is None:
        fails.append("d_mass / d_te not found in the log (did the run finish?)")
    else:
        (fails if abs(v["d_mass"]) > D_MASS_ABS_TOL else notes).append(f"|d_mass| = {abs(v['d_mass']):.3e} {'>' if abs(v['d_mass']) > D_MASS_ABS_TOL else '<='} {D_MASS_ABS_TOL} (upstream criterion)")
        d = abs(v["d_te"] - ref["d_te"])
        (fails if d > D_TE_ABS_TOL else notes).append(f"d_te {v['d_te']:.7e} vs CPU reference {ref['d_te']:.7e}: |diff| {d:.2e} {'>' if d > D_TE_ABS_TOL else '<='} {D_TE_ABS_TOL}")
    for n in notes:
        print(f"   {n}")
    if fails:
        print(f"FAIL: miniweather reference check ({key(a)}) -- " + "; ".join(fails)); return 1
    print(f"PASS: miniweather reference check ({key(a)}): |d_mass| <= {D_MASS_ABS_TOL}, d_te within {D_TE_ABS_TOL} absolute of the same-grid CPU (OpenMP) reference build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
