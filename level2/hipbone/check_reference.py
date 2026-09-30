#!/usr/bin/env python3
"""hipBone: the registered inputs' correctness check against a CPU reference run of the same problem.

    check_reference.py --log <run stdout>                       check a run (PASS / FAIL, exit 0 / 1)
    check_reference.py --make-reference <id> <cpu log> [...]    write reference/<id>.json from CPU runs

The reference of every registered input is a run of the SAME hipBone build with OCCA's CPU backends
(Serial, and OpenMP with the kernels JIT-compiled for the host), the exact registered argument list
(-nx -ny -nz -p, -v), on this node (2026-09-30; provenance in reference/<id>.json). hipBone's right-hand
side is a fixed pseudo-random sequence, so CPU and GPU solve the same discrete problem; their residual
histories differ only by rounding (libm vs the CUDA math library for the RHS, reduction order in the
dot products), which accumulates over the 100 unpreconditioned CG iterations.

Benchmark-wide rule (one rule for every registered size, fixed BEFORE any candidate exists):
  cg_iterations  == reference (exact)            dofs == reference (exact)
  r_norm_initial within 1 % of the reference     (validate.sh check 4: two libm realisations of the RHS)
  |log10(r_norm_final / reference)| <= R_FINAL_LOG10_TOL
R_FINAL_LOG10_TOL = 0.1 (a factor 1.26). Observed before the rule was fixed (2026-09-30, the
unoptimized CUDA build vs the CPU references; Serial and OpenMP agreed to all 13 printed digits at
every size where both were run): sweep-nx9-p14 |log10| 0.0022, sweep-nx16-p8 0.0029, sweep-nx32-p4
0.0268 -- the remaining sizes are added to reference/<id>.json as their CPU runs finish and the
observed deviation is recorded there. 0.1 is about four times the largest of these deviations
between two correct realisations of the arithmetic, and ten times tighter than the factor 10
validate.sh allows for its 27-element case. It is a rounding-accumulation bound, not a convergence
criterion: a wrong operator changes the residual by orders of magnitude. Nothing here is derived
from a candidate.
"""
import argparse, hashlib, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, "reference")
R_INITIAL_REL_TOL = 1e-2
R_FINAL_LOG10_TOL = 0.1
RX = {"input_id": re.compile(r"input=([A-Za-z0-9._-]+)\)"),
      "dofs": re.compile(r"^hipBone: \d+, (\d+),", re.M),
      "cg_iterations": re.compile(r"^hipBone: \d+, \d+, [0-9.eE+-]+, (\d+),", re.M),
      "r_norm_initial": re.compile(r"^CG: initial res norm ([0-9.eE+-]+)", re.M),
      "r_norm_final": re.compile(r"^CG: it 100, r norm ([0-9.eE+-]+)", re.M)}


def parse(path):
    t = open(path, errors="replace").read()
    out = {}
    for k, rx in RX.items():
        m = rx.findall(t)
        if not m:
            continue
        v = m[-1]
        out[k] = v if k == "input_id" else (int(v) if k in ("dofs", "cg_iterations") else float(v))
    return out


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def make_reference(input_id, logs, note):
    runs = []
    for lg in logs:
        v = parse(lg)
        mode = re.search(r"-m (\w+)", open(lg, errors="replace").read())
        runs.append({"log": os.path.basename(lg), "sha256": sha(lg), "backend": mode.group(1) if mode else "?",
                     **{k: v.get(k) for k in ("dofs", "cg_iterations", "r_norm_initial", "r_norm_final")}})
    keys = ("dofs", "cg_iterations", "r_norm_initial", "r_norm_final")
    for k in keys:
        vals = {r[k] for r in runs}
        if k in ("dofs", "cg_iterations") and len(vals) != 1:
            sys.exit(f"reference runs disagree on {k}: {vals}")
    finals = [r["r_norm_final"] for r in runs]
    ref = {"schema": "hpcperf-hipbone-reference-1", "input_id": input_id, "note": note,
           "dofs": runs[0]["dofs"], "cg_iterations": runs[0]["cg_iterations"],
           "r_norm_initial": runs[0]["r_norm_initial"], "r_norm_final": runs[0]["r_norm_final"],
           "r_norm_final_spread_log10": (math.log10(max(finals) / min(finals)) if min(finals) > 0 else None),
           "runs": runs, "rule": {"r_norm_initial": f"rel {R_INITIAL_REL_TOL}", "r_norm_final": f"|log10 ratio| <= {R_FINAL_LOG10_TOL}",
                                  "dofs": "exact", "cg_iterations": "exact"}}
    os.makedirs(REF_DIR, exist_ok=True)
    p = os.path.join(REF_DIR, f"{input_id}.json")
    json.dump(ref, open(p, "w"), indent=1)
    print(f"wrote {p}: r_norm_final {ref['r_norm_final']:.12e} from {len(runs)} CPU run(s) "
          f"(spread log10 {ref['r_norm_final_spread_log10']})")


def check(log, input_id=None):
    v = parse(log)
    input_id = input_id or v.get("input_id")
    if not input_id:
        print("FAIL: hipBone reference check -- the log names no registered input (run.sh echoes input=<id>)"); return 1
    p = os.path.join(REF_DIR, f"{input_id}.json")
    if not os.path.isfile(p):
        print(f"FAIL: hipBone reference check -- no CPU reference for input {input_id} ({p} missing)"); return 1
    ref = json.load(open(p))
    fails, notes = [], []
    for k in ("dofs", "cg_iterations"):
        if v.get(k) != ref[k]:
            fails.append(f"{k} {v.get(k)} != reference {ref[k]}")
        else:
            notes.append(f"{k} = {v[k]} (exact)")
    if "r_norm_initial" not in v or "r_norm_final" not in v:
        fails.append("residual norms not found in the log (hipBone needs -v; did the run finish?)")
    else:
        d0 = abs(v["r_norm_initial"] - ref["r_norm_initial"]) / abs(ref["r_norm_initial"])
        (fails if d0 > R_INITIAL_REL_TOL else notes).append(
            f"r_norm_initial {v['r_norm_initial']:.12g} vs CPU reference {ref['r_norm_initial']:.12g}: rel {d0:.2e} {'>' if d0 > R_INITIAL_REL_TOL else '<='} {R_INITIAL_REL_TOL}")
        if v["r_norm_final"] <= 0 or ref["r_norm_final"] <= 0:
            fails.append(f"non-positive final residual ({v['r_norm_final']}, reference {ref['r_norm_final']})")
        else:
            l = abs(math.log10(v["r_norm_final"] / ref["r_norm_final"]))
            (fails if l > R_FINAL_LOG10_TOL else notes).append(
                f"r_norm_final {v['r_norm_final']:.12e} vs CPU reference {ref['r_norm_final']:.12e}: |log10 ratio| {l:.4f} {'>' if l > R_FINAL_LOG10_TOL else '<='} {R_FINAL_LOG10_TOL}")
    for n in notes:
        print(f"   {n}")
    if fails:
        print(f"FAIL: hipBone reference check ({input_id}) -- " + "; ".join(fails)); return 1
    print(f"PASS: hipBone reference check ({input_id}): dofs and CG iterations exact, r_norm_initial within {R_INITIAL_REL_TOL:.0%}, "
          f"r_norm_final within |log10| {R_FINAL_LOG10_TOL} of the CPU (Serial/OpenMP) reference of the same build and problem")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log")
    ap.add_argument("--input", help="registered input id (default: read from the log's run.sh echo)")
    ap.add_argument("--make-reference", nargs="+", metavar=("ID", "CPU_LOG"))
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    if a.make_reference:
        make_reference(a.make_reference[0], a.make_reference[1:], a.note); return 0
    if not a.log:
        ap.error("--log or --make-reference")
    return check(a.log, a.input)


if __name__ == "__main__":
    sys.exit(main())
