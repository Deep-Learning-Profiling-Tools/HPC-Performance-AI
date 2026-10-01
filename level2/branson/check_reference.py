#!/usr/bin/env python3
"""Branson: candidate validation of a GPU run against the FROZEN CPU-only reference of its deck.

    check_reference.py <gpu.log> --deck <deck.xml> [--reference-dir DIR]

This is the registry's correctness check (level2/branson/inputs.yaml `check:`) and the only step a
candidate evaluation runs: it reads reference/<id>.json -- written ONCE, at construction time, by
generate_reference.sh from a finished CPU-only run (README "Correctness reference") -- and compares the
candidate's final completed step with it. It builds and runs nothing: a missing, corrupted or
mismatching reference is an explicit error (exit 2), never a trigger to regenerate one.

Rule (validate.sh check C; frozen in the reference, repeated here -- a reference whose tolerances differ
from these constants is refused, so the artifact cannot loosen the rule):
  final-step Post mat E, Absorption E, Exit E         rel <= 0.05
  final-step "Total Photons transported"              rel <= 0.05   (the last such line of the log, never the first)
  final-step T_e of every cell (when the deck prints it)  abs <= 0.02
  the candidate must complete the reference's number of steps and print the FOM line.
The CPU reference is the correctness oracle only. It is not a performance baseline: speedups are
measured against the original (unoptimized) GPU implementation's timing records.
Prints the reference provenance and every compared quantity, then
"PASS: branson reference check (...)" / "FAIL: branson reference check (...)"; exit 0 / 1 / 2 (reference or usage error).
"""
import argparse, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_log as CL  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, "reference")
REQUIRED = {"schema": str, "benchmark": str, "input_id": str, "deck": str, "deck_sha256": str, "steps": int,
            "final_step": dict, "tolerances": dict, "reference_run": dict, "generated_utc": str}
FINAL_KEYS = ("post_mat_e", "absorption_e", "exit_e", "photons_transported")
PROVENANCE = {"binary_sha256": str, "mpi_ranks": int, "log_sha256": str, "finished_utc": str}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def error(msg):
    print(f"   ERROR: {msg}")
    print(f"FAIL: branson reference check -- ERROR: {msg}")
    return 2


def validate_reference(d, path, deck):
    """Every structural problem of a frozen reference is a refusal (returns the message), never repaired."""
    if not isinstance(d, dict):
        return f"{path}: not a JSON object"
    for k, t in REQUIRED.items():
        if k not in d or not isinstance(d[k], t) or (t is int and isinstance(d[k], bool)):
            return f"{path}: field {k!r} missing or not a {t.__name__} -- corrupted reference refused"
    if d["schema"] != CL.REFERENCE_SCHEMA:
        return f"{path}: schema {d['schema']!r} is not {CL.REFERENCE_SCHEMA!r}"
    if d["benchmark"] != "branson":
        return f"{path}: benchmark {d['benchmark']!r} is not branson"
    if d["steps"] < 1:
        return f"{path}: steps must be >= 1"
    fs = d["final_step"]
    for k in FINAL_KEYS:
        v = fs.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
            return f"{path}: final_step.{k} missing or not a finite number -- corrupted reference refused"
    if not isinstance(fs["photons_transported"], int) or fs["photons_transported"] < 0:
        return f"{path}: final_step.photons_transported must be a non-negative integer"
    te = fs.get("t_e")
    if te is not None and (not isinstance(te, list) or not te or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in te)):
        return f"{path}: final_step.t_e must be null or a non-empty list of numbers"
    want = {"energy_rel": CL.ENERGY_REL_TOL, "photons_rel": CL.PHOTONS_REL_TOL, "t_e_abs": CL.TE_ABS_TOL}
    if d["tolerances"] != want:
        return f"{path}: tolerances {d['tolerances']} differ from the checker's rule {want} -- refused (the artifact cannot change the rule)"
    rr = d["reference_run"]
    for k, t in PROVENANCE.items():
        if k not in rr or not isinstance(rr[k], t) or (t is int and isinstance(rr[k], bool)):
            return f"{path}: reference_run.{k} missing or not a {t.__name__} -- provenance incomplete, refused"
    if not HEX64.match(rr["binary_sha256"]) or not HEX64.match(rr["log_sha256"]) or not HEX64.match(d["deck_sha256"]):
        return f"{path}: a sha256 field is not 64 hex digits -- refused"
    if rr["mpi_ranks"] < 1:
        return f"{path}: reference_run.mpi_ranks must be >= 1"
    if os.path.basename(d["deck"]) != os.path.basename(deck):
        return f"{path}: reference deck {d['deck']!r} is not {os.path.basename(deck)!r}"
    actual = CL.sha256_file(deck)
    if actual != d["deck_sha256"]:
        return (f"deck sha256 mismatch: {deck} is {actual[:16]}..., the reference {path} was generated for "
                f"{d['deck_sha256'][:16]}... -- the reference does not belong to this deck content; refused")
    return None


def find_reference(ref_dir, deck):
    """-> (path, dict) of the reference whose deck matches, or raises LookupError / ValueError."""
    base = os.path.basename(deck)
    hits, unreadable = [], []
    for path in sorted(glob.glob(os.path.join(ref_dir, "*.json"))):
        try:
            d = json.load(open(path))
        except (OSError, ValueError) as ex:
            unreadable.append(f"{os.path.basename(path)} ({ex})")
            continue
        if isinstance(d, dict) and os.path.basename(str(d.get("deck", ""))) == base:
            hits.append((path, d))
    if len(hits) > 1:
        raise ValueError(f"{len(hits)} references in {ref_dir} claim deck {base}: " + ", ".join(os.path.basename(p) for p, _ in hits))
    if not hits:
        more = ("; unreadable reference file(s): " + "; ".join(unreadable)) if unreadable else ""
        raise LookupError(f"no frozen reference for deck {base} in {ref_dir}{more} -- generating one is a construction-time "
                          "step (level2/branson/generate_reference.sh), never done by this check")
    return hits[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gpu_log")
    ap.add_argument("--deck", required=True, help="the deck the candidate ran (its sha256 must match the reference)")
    ap.add_argument("--reference-dir", default=REF_DIR)
    a = ap.parse_args()
    if not os.path.isfile(a.gpu_log):
        return error(f"candidate log {a.gpu_log} missing")
    if not os.path.isfile(a.deck):
        return error(f"deck {a.deck} missing")
    try:
        path, ref = find_reference(a.reference_dir, a.deck)
    except (LookupError, ValueError) as ex:
        return error(str(ex))
    problem = validate_reference(ref, path, a.deck)
    if problem:
        return error(problem)
    rr = ref["reference_run"]
    print(f"== frozen reference {os.path.relpath(path, HERE)}: CPU-only Branson (USE_GPU=OFF), {rr['mpi_ranks']} MPI rank(s), "
          f"binary sha256 {rr['binary_sha256'][:12]}..., CPU log sha256 {rr['log_sha256'][:12]}... (run finished {rr['finished_utc']}), "
          f"generated {ref['generated_utc']}; deck {ref['deck']} sha256 {ref['deck_sha256'][:12]}... verified")
    steps, text = CL.parse(a.gpu_log)
    errors = []
    if not CL.finished(text):
        errors.append('the candidate run did not finish (no "Photons Per Second (FOM)" line)')
    if len(steps) != ref["steps"]:
        errors.append(f"step count differs: candidate {len(steps)} vs reference {ref['steps']}")
    if not errors:
        fs = ref["final_step"]
        c = {"PostMat": fs["post_mat_e"], "Absorption": fs["absorption_e"], "Exit": fs["exit_e"],
             "Photons": fs["photons_transported"], "Te": fs.get("t_e") or []}
        print(f"   final completed step: {len(steps)} of {ref['steps']}")
        errors += CL.compare_final(steps[-1], c, "GPU", "ref")
    for e in errors:
        print("   ERROR:", e)
    deck = os.path.basename(a.deck)
    rule = (f"final-step Post-mat/Absorption/Exit E within {CL.ENERGY_REL_TOL * 100:g} %, final-step transported photons within "
            f"{CL.PHOTONS_REL_TOL * 100:g} %, T_e within {CL.TE_ABS_TOL:g} where printed")
    if errors:
        print(f"FAIL: branson reference check ({deck}): GPU run vs the frozen CPU-only reference {os.path.basename(path)} "
              f"({rr['mpi_ranks']} rank(s)) -- {len(errors)} criterion/criteria violated: " + "; ".join(errors))
        return 1
    print(f"PASS: branson reference check ({deck}): GPU run vs the frozen CPU-only reference {os.path.basename(path)} "
          f"({rr['mpi_ranks']} rank(s), generated {ref['generated_utc'][:10]}) -- {rule}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
