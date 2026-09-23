#!/usr/bin/env python3
"""Generate the timing execution cases of the registered inputs from the inputs registry.

    gen_registry_cases.py            write cases/level1_registry.tsv and cases/level2_registry.tsv
    gen_registry_cases.py --check    exit 1 if either file differs from what the registry gives (drift)

The inputs registry (`level<N>/<benchmark>/inputs.yaml`, read by tools/inputs/hpcperf_inputs.py)
is the only definition of a benchmark's inputs; these two files are execution artifacts derived
from it, never edited by hand. One row per registered input of a Level 1 benchmark or Level 2
application (Level 3 has no ROI markers). The case name IS the registry's input id, and every row
also carries the id in its own column, so a timing record can always be joined back to the
registry entry (the engine additionally stores the input's full workload identity, see
`hpcperf_inputs.py identity`, next to the raw runs).

Level 1 row: the registry's own executable (`binary` of a materialized compile-time input, else the
entry path), its arguments (repository-relative paths written as {REPO}/..., resolved when the case
is run), the executable's directory as the working directory, and the input's env.
Level 2 row: the application's run.sh with the registry selector set to the input id -- run.sh
applies the input's knobs and arguments itself (tools/inputs/hpcperf_input_selector.sh) -- plus
the names of the knobs the selector sets, which cases.py refuses when they are set in the caller's
shell instead of through the case.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools", "inputs"))
sys.path.insert(0, HERE)

import cases as C            # noqa: E402
import hpcperf_inputs as hi  # noqa: E402

L1_FILE, L2_FILE = "level1_registry.tsv", "level2_registry.tsv"
L1_COLS = ("app", "case", "input_id", "materialized", "exe", "args", "cwd", "env", "timeout_s")
L2_COLS = ("app", "case", "input_id", "materialized", "selector", "registry_knobs", "gpus", "timeout_s")
L1_MIN_TIMEOUT = 1800        # per run; the ctest default case's timeout when larger
L2_MIN_TIMEOUT = 1800

HEADER = """# tools/timing/cases/{name} -- GENERATED from the inputs registry by tools/timing/gen_registry_cases.py.
# Do not edit: change level{level}/<benchmark>/inputs.yaml and regenerate; `gen_registry_cases.py --check`
# (run by `cases.py check` and the tests) fails when this file and the registry disagree.
# One row per registered input; case == input_id. {{REPO}} is the repository root. "-" = empty.
#{cols}
"""


def repo_arg(a):
    a = str(a)
    return "{REPO}/" + a if a.startswith(hi.REPO_PATH_PREFIXES) else a


def cell(v):
    v = "" if v is None else str(v)
    if "\t" in v or "\n" in v:
        raise C.CaseError(f"value {v!r} contains a tab or newline")
    return v if v != "" else "-"


def quote_args(args):
    import shlex
    return " ".join(shlex.quote(repo_arg(a)) for a in args)


def registry_docs(level):
    base = os.path.join(REPO, f"level{level}")
    for d in sorted(os.listdir(base)):
        f = os.path.join(base, d, "inputs.yaml")
        if os.path.isfile(f):
            doc = hi.load(os.path.join(base, d))
            errs = hi.validate(doc)
            if errs:
                raise C.CaseError(f"level{level}/{d}/inputs.yaml is invalid: {errs[0]}")
            yield d, doc


def level1_rows():
    defaults = {}
    for r in C.read_table("level1.tsv", C.L1_CASE_COLS):
        defaults.setdefault(r["app"], r)
    rows = []
    for app, doc in registry_docs(1):
        if doc["entry"]["kind"] != "binary":
            raise C.CaseError(f"level1/{app}: Level 1 registry entries must be binaries")
        d_to = int(defaults[app]["timeout_s"] or 0) if app in defaults else 0
        for inp in doc["inputs"]:
            exe = inp.get("binary") or doc["entry"]["path"]
            env = ";".join(f"{k}={v}" for k, v in (inp.get("env") or {}).items())
            rows.append({"app": app, "case": inp["id"], "input_id": inp["id"],
                         "materialized": "1" if inp.get("materialized", True) is not False else "0",
                         "exe": "{REPO}/" + exe, "args": quote_args(inp.get("args") or []),
                         "cwd": "{REPO}/" + os.path.dirname(exe), "env": env,
                         "timeout_s": str(max(L1_MIN_TIMEOUT, d_to))})
    return rows


def level2_rows():
    apps = {r["app"]: r for r in C.read_table("level2_apps.tsv", C.L2_APP_COLS)}
    rows = []
    for app, doc in registry_docs(2):
        if doc["entry"]["kind"] != "run.sh" or not doc.get("selector"):
            raise C.CaseError(f"level2/{app}: Level 2 registry entries must be run.sh with a selector")
        if app not in apps:
            raise C.CaseError(f"level2/{app} has a registry but no row in cases/level2_apps.tsv")
        a_to = int(apps[app]["timeout_s"] or 0)
        knobs = sorted({str(k) for i in doc["inputs"] for k in (i.get("env") or {})})
        for inp in doc["inputs"]:
            gpus = (inp.get("runtime_config") or {}).get("gpus", 1)
            rows.append({"app": app, "case": inp["id"], "input_id": inp["id"],
                         "materialized": "1" if inp.get("materialized", True) is not False else "0",
                         "selector": doc["selector"], "registry_knobs": ",".join(knobs),
                         "gpus": str(gpus), "timeout_s": str(max(L2_MIN_TIMEOUT, a_to))})
    return rows


def render(name, level, cols, rows):
    out = HEADER.format(name=name, level=level, cols="\t".join(cols))
    for r in rows:
        out += "\t".join(cell(r[c]) for c in cols) + "\n"
    return out


def expected():
    return {L1_FILE: render(L1_FILE, 1, L1_COLS, level1_rows()),
            L2_FILE: render(L2_FILE, 2, L2_COLS, level2_rows())}


def check():
    """Problems (empty when both generated files match the registry)."""
    problems = []
    for name, text in expected().items():
        path = os.path.join(C.CASES, name)
        if not os.path.isfile(path):
            problems.append(f"cases/{name} is missing -- run tools/timing/gen_registry_cases.py")
        elif open(path).read() != text:
            problems.append(f"cases/{name} differs from the inputs registry -- run tools/timing/gen_registry_cases.py")
    return problems


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report drift, write nothing")
    a = ap.parse_args(argv)
    try:
        if a.check:
            p = check()
            for m in p:
                print(f"gen_registry_cases: {m}", file=sys.stderr)
            if not p:
                print("gen_registry_cases: level1_registry.tsv and level2_registry.tsv match the registry")
            return 1 if p else 0
        for name, text in expected().items():
            with open(os.path.join(C.CASES, name), "w") as f:
                f.write(text)
            print(f"gen_registry_cases: wrote cases/{name} ({text.count(chr(10)) - 5} rows)")
        return 0
    except (C.CaseError, hi.InputError) as exc:
        print(f"gen_registry_cases: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
