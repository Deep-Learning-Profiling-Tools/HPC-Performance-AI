#!/usr/bin/env python3
"""Resolve the measurement case tables into the normalized rows the shell engine runs.

    cases.py resolve --level 1 --build-root build/gcc13 [SELECT ...]
    cases.py resolve --level 2 [SELECT ...]
    cases.py resolve --level 1|2 --registry [SELECT ...]   # the registered inputs (inputs.yaml)
    cases.py check                    # validate every table; CPU only, used by the tests
    cases.py allowed-env <app>        # the input variables a Level 2 case may set

SELECT is `all` (default), `<app>`, or `<app>/<case>`.

A case is identified by (level, app, case); a measurement adds the platform. Tables live
in tools/timing/cases/, tab-separated, "-" for an empty field:

  level1.tsv        generated from ctest by gen_cases.py (one row per ctest test)
  level1_extra.tsv  extra Level 1 inputs ctest does not know: app case args env timeout_s notes
  level1_apps.tsv   per-benchmark facts: suite backends roi_excludes verify_vs_roi extra_env notes
  level2_apps.tsv   per-application facts, including the application's own FOM pattern
  level2_cases.tsv  Level 2 cases: app case gpus env args timeout_s fom_regex notes
  level1_registry.tsv, level2_registry.tsv
                    GENERATED from the inputs registry (level<N>/<app>/inputs.yaml) by
                    gen_registry_cases.py: one case per registered input, case == input_id.
                    Resolved only with --registry; `check` fails when they drift from the
                    registry. The registry, not these files, defines the inputs.

Sweeps: in level2_cases.tsv one variable may take several values, `NAME=a|b|c`; the
case name must contain `{}`, which becomes each value (`n{}` -> n128, n192, n256).

Inputs are explicit or refused. Every variable a case sets must be one the
application's run.sh actually reads (or its `extra_env`), and a variable the
application reads that is set in YOUR shell but not declared by the case is an error:
the engine starts every run from `env -i`, so it would be dropped silently and the
default case measured under the wrong name.
"""

import argparse
import itertools
import os
import re
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CASES = os.path.join(HERE, "cases")

NAME_OK = re.compile(r"^[A-Za-z0-9._-]+$")
VAR_REF = re.compile(r"\$\{?(HPCPERF_[A-Z0-9_]+)")
DENY = re.compile(r"TOKEN|SECRET|PASSWD|PASSWORD|CREDENTIAL|PRIVATE_KEY|API_KEY|SESSION")
# set by the engine or by the launcher interface -- never an input of a case
RESERVED = {"HPCPERF_ROI_LOG", "HPCPERF_SKIP_VERIFY", "HPCPERF_GPUS", "HPCPERF_NP", "HPCPERF_DRY_RUN",
            # set by the registry selector helper inside run.sh (tools/inputs/hpcperf_input_selector.sh)
            "HPCPERF_INPUT_ARGS", "HPCPERF_INPUT_ID"}
PLATFORM = {"HPCPERF_SITE_PROFILE", "HPCPERF_LAUNCHER", "HPCPERF_LAUNCHER_BIN", "HPCPERF_NODES",
            "HPCPERF_GPUS_PER_NODE", "HPCPERF_GPU_BACKEND", "HPCPERF_RUN_TMPDIR", "HPCPERF_CPUS_PER_RANK"}
CASE_SETTABLE_PLATFORM = {"HPCPERF_CPUS_PER_RANK"}

ROW_FIELDS = ("level", "app", "case", "backend", "gpus", "cwd", "timeout_s", "env", "argv",
              "fom_name", "fom_unit", "fom_better", "fom_source", "fom_regex",
              "roi_excludes", "verify_vs_roi", "notes", "input_id")
# input_id: the registry input a --registry case measures ("" for the hand-written cases); the
# engine stores that input's workload identity next to the raw runs and refuses the case when the
# identity cannot be established.


class CaseError(Exception):
    pass


# ----------------------------------------------------------------- tables

def read_table(name, columns):
    path = os.path.join(CASES, name)
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cells = line.split("\t")
            if len(cells) != len(columns):
                raise CaseError(f"{name}:{lineno}: {len(cells)} fields, expected {len(columns)} ({', '.join(columns)})")
            if any(c == "" for c in cells):
                raise CaseError(f"{name}:{lineno}: empty field -- write '-' (tabs collapse in bash `read`)")
            row = {k: ("" if v == "-" else v) for k, v in zip(columns, cells)}
            row["_where"] = f"{name}:{lineno}"
            rows.append(row)
    return rows


L1_CASE_COLS = ("app", "case", "exe", "args", "cwd", "timeout_s", "wrapper", "pass_regex")
L1_EXTRA_COLS = ("app", "case", "args", "env", "timeout_s", "notes")
L1_APP_COLS = ("app", "suite", "backends", "roi_excludes", "verify_vs_roi", "extra_env", "notes")
L2_APP_COLS = ("app", "backends", "timeout_s", "fom_name", "fom_unit", "fom_better", "fom_source",
               "fom_regex", "extra_env", "roi_excludes", "app_timer_regex", "app_timer_unit", "notes")
APP_TIMER_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6}
L2_CASE_COLS = ("app", "case", "gpus", "env", "args", "timeout_s", "fom_regex", "notes")
L1_REG_COLS = ("app", "case", "input_id", "materialized", "exe", "args", "cwd", "env", "timeout_s")
L2_REG_COLS = ("app", "case", "input_id", "materialized", "selector", "registry_knobs", "gpus", "timeout_s")


def registry_l2():
    """{app: (selector, {knob names})} from the generated Level 2 registry table: the selector
    variable run.sh reads indirectly (hpcperf_apply_input) and the knobs it sets from the registry."""
    out = {}
    for r in read_table("level2_registry.tsv", L2_REG_COLS):
        out[r["app"]] = (r["selector"], set(x for x in r["registry_knobs"].split(",") if x))
    return out


def parse_env(text, where):
    """'A=1;B=x' -> [(A, '1'), (B, 'x')], names validated."""
    out = []
    if not text:
        return out
    for part in text.split(";"):
        if "=" not in part:
            raise CaseError(f"{where}: env entry {part!r} is not NAME=VALUE")
        k, v = part.split("=", 1)
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", k):
            raise CaseError(f"{where}: bad variable name {k!r}")
        if DENY.search(k):
            raise CaseError(f"{where}: {k} matches the credential deny rule and can never be passed")
        if k in RESERVED:
            raise CaseError(f"{where}: {k} is set by the engine; use the gpus column / protocol, not env")
        out.append((k, v))
    names = [k for k, _ in out]
    if len(names) != len(set(names)):
        raise CaseError(f"{where}: a variable is set twice")
    return out


def run_sh_vars(app):
    path = os.path.join(REPO, "level2", app, "run.sh")
    if not os.path.isfile(path):
        raise CaseError(f"level2/{app}/run.sh does not exist")
    with open(path) as f:
        return set(VAR_REF.findall(f.read()))


def input_vars(app, registry=None):
    """Variables that change what a Level 2 application computes: read by its run.sh (directly,
    or -- for the registry selector -- through tools/inputs/hpcperf_input_selector.sh, which the
    text scan cannot see), not part of the launcher / platform interface."""
    reg = registry_l2() if registry is None else registry
    sel = {reg[app][0]} if app in reg and reg[app][0] else set()
    return {v for v in run_sh_vars(app) | sel if v not in RESERVED and v not in PLATFORM
            and not v.startswith("HPCPERF_BIND_")}


def registry_knobs(app, registry=None):
    """Knobs the registry selector sets inside run.sh; a case never sets them directly."""
    reg = registry_l2() if registry is None else registry
    return reg.get(app, ("", set()))[1]


def allowed_env(app, extra, registry=None):
    return input_vars(app, registry) | CASE_SETTABLE_PLATFORM | set(x for x in extra.split(",") if x)


def expand_sweep(case, env, where):
    """At most one variable may carry 'a|b|c'; returns [(case_name, env), ...]."""
    swept = [(k, v.split("|")) for k, v in env if "|" in v]
    if len(swept) > 1:
        raise CaseError(f"{where}: only one variable may sweep, got {[k for k, _ in swept]}")
    if not swept:
        if "{}" in case:
            raise CaseError(f"{where}: case name {case!r} has {{}} but no variable sweeps")
        return [(case, env)]
    if "{}" not in case:
        raise CaseError(f"{where}: {swept[0][0]} sweeps; the case name needs {{}} for the value")
    name, values = swept[0]
    out = []
    for val in values:
        if not NAME_OK.match(val):
            raise CaseError(f"{where}: sweep value {val!r} is not usable in a case name")
        out.append((case.replace("{}", val), [(k, val if k == name else v) for k, v in env]))
    return out


def env_text(env):
    for k, v in env:
        if any(c in v for c in ";\t\n"):
            raise CaseError(f"{k}: value may not contain ';', tab or newline")
    return ";".join(f"{k}={v}" for k, v in env)


# ----------------------------------------------------------------- level 1

def level1_rows(build_root, backend):
    apps = {r["app"]: r for r in read_table("level1_apps.tsv", L1_APP_COLS)}
    base = read_table("level1.tsv", L1_CASE_COLS)
    extra = read_table("level1_extra.tsv", L1_EXTRA_COLS)
    if not base:
        raise CaseError("cases/level1.tsv is empty or missing -- run gen_cases.py")

    def sub(s):
        s = s.replace("{REPO}", REPO)
        return s.replace("{BUILD}", os.path.abspath(build_root)) if build_root else s

    defaults = {}
    rows = []
    for r in base:
        app = apps.get(r["app"])
        if app is None:
            raise CaseError(f"{r['_where']}: {r['app']} has no row in cases/level1_apps.tsv")
        defaults.setdefault(r["app"], r)
        rows.append((r, app, r["args"], [], r["timeout_s"], ""))
    for r in extra:
        app = apps.get(r["app"])
        if app is None or r["app"] not in defaults:
            raise CaseError(f"{r['_where']}: {r['app']} is not a Level 1 benchmark with a default case")
        env = parse_env(r["env"], r["_where"])
        bad = [k for k, _ in env if k not in set(x for x in app["extra_env"].split(",") if x)]
        if bad:
            raise CaseError(f"{r['_where']}: {bad} not in the benchmark's extra_env (Level 1 binaries read none by default)")
        d = defaults[r["app"]]
        rows.append((dict(d, case=r["case"], _where=r["_where"]), app, r["args"], env,
                     r["timeout_s"] or d["timeout_s"], r["notes"]))

    out, seen = [], set()
    for r, app, args, env, timeout, notes in rows:
        key = (r["app"], r["case"])
        if not NAME_OK.match(r["case"]):
            raise CaseError(f"{r['_where']}: case name {r['case']!r} must match [A-Za-z0-9._-]+")
        if key in seen:
            raise CaseError(f"{r['_where']}: duplicate case {r['app']}/{r['case']}")
        seen.add(key)
        if backend.lower() not in app["backends"].split(","):
            continue
        argv = [sub(r["exe"])] + [sub(a) for a in shlex.split(args)]
        out.append({
            "level": "1", "app": r["app"], "case": r["case"], "backend": backend.upper(), "gpus": "",
            "cwd": sub(r["cwd"]), "timeout_s": timeout or "600", "env": env_text(env),
            "argv": " ".join(shlex.quote(a) for a in argv),
            "fom_name": "", "fom_unit": "", "fom_better": "", "fom_source": "", "fom_regex": "",
            "roi_excludes": app["roi_excludes"], "verify_vs_roi": app["verify_vs_roi"],
            "notes": "; ".join(x for x in (app["notes"], notes) if x), "input_id": "",
        })
    return out, apps


def level1_registry_rows(backend):
    """Level 1 cases of the registered inputs: the registry's own executable (a materialized
    compile-time input's binary, e.g. another NPB class, is never replaced by the default one),
    arguments and env; repository-relative paths resolved here, recorded relative in the identity."""
    apps = {r["app"]: r for r in read_table("level1_apps.tsv", L1_APP_COLS)}
    out, seen = [], set()
    for r in read_table("level1_registry.tsv", L1_REG_COLS):
        app = apps.get(r["app"])
        if app is None:
            raise CaseError(f"{r['_where']}: {r['app']} has no row in cases/level1_apps.tsv")
        if r["case"] != r["input_id"] or not NAME_OK.match(r["case"]):
            raise CaseError(f"{r['_where']}: case must equal the input id and match [A-Za-z0-9._-]+")
        if (r["app"], r["case"]) in seen:
            raise CaseError(f"{r['_where']}: duplicate registry case {r['app']}/{r['case']}")
        seen.add((r["app"], r["case"]))
        env = parse_env(r["env"], r["_where"])
        if backend.lower() not in app["backends"].split(","):
            continue
        sub = lambda x: x.replace("{REPO}", REPO)
        argv = [sub(r["exe"])] + [sub(a) for a in shlex.split(r["args"])]
        out.append({
            "level": "1", "app": r["app"], "case": r["case"], "backend": backend.upper(), "gpus": "",
            "cwd": sub(r["cwd"]), "timeout_s": r["timeout_s"] or "1800", "env": env_text(env),
            "argv": " ".join(shlex.quote(a) for a in argv),
            "fom_name": "", "fom_unit": "", "fom_better": "", "fom_source": "", "fom_regex": "",
            "roi_excludes": app["roi_excludes"], "verify_vs_roi": app["verify_vs_roi"],
            "notes": "; ".join(x for x in (app["notes"], "registry input" + ("" if r["materialized"] == "1" else ", NOT materialized")) if x),
            "input_id": r["input_id"],
        })
    return out, apps


def level2_registry_rows(backend):
    """Level 2 cases of the registered inputs: run.sh with the registry selector = input id."""
    apps = {r["app"]: r for r in read_table("level2_apps.tsv", L2_APP_COLS)}
    reg = registry_l2()
    out, seen = [], set()
    for r in read_table("level2_registry.tsv", L2_REG_COLS):
        app = apps.get(r["app"])
        if app is None:
            raise CaseError(f"{r['_where']}: {r['app']} has no row in cases/level2_apps.tsv")
        if r["case"] != r["input_id"] or not NAME_OK.match(r["case"]):
            raise CaseError(f"{r['_where']}: case must equal the input id and match [A-Za-z0-9._-]+")
        if (r["app"], r["case"]) in seen:
            raise CaseError(f"{r['_where']}: duplicate registry case {r['app']}/{r['case']}")
        seen.add((r["app"], r["case"]))
        env = parse_env(f"{r['selector']}={r['input_id']}", r["_where"])
        allowed = allowed_env(r["app"], app["extra_env"], reg)
        if r["selector"] not in allowed:
            raise CaseError(f"{r['_where']}: selector {r['selector']} is not an input variable of {r['app']}")
        gpus = r["gpus"] or "1"
        if backend.lower() not in app["backends"].split(","):
            continue
        argv = ["bash", f"level2/{r['app']}/run.sh", backend.upper()]
        out.append({
            "level": "2", "app": r["app"], "case": r["case"], "backend": backend.upper(), "gpus": gpus,
            "cwd": REPO, "timeout_s": r["timeout_s"] or app["timeout_s"] or "1800",
            "env": env_text(env), "argv": " ".join(shlex.quote(a) for a in argv),
            "fom_name": app["fom_name"], "fom_unit": app["fom_unit"], "fom_better": app["fom_better"],
            "fom_source": app["fom_source"], "fom_regex": app["fom_regex"],
            "roi_excludes": app["roi_excludes"], "verify_vs_roi": "outside",
            "notes": "; ".join(x for x in (app["notes"], "registry input" + ("" if r["materialized"] == "1" else ", NOT materialized")) if x),
            "input_id": r["input_id"],
        })
    return out, apps


# ----------------------------------------------------------------- level 2

def check_app_timer(app):
    """The application's own timer for the region its ROI marks: one capture group, a known unit."""
    rx, unit = app["app_timer_regex"], app["app_timer_unit"]
    if not rx and not unit:
        return
    if not rx or unit not in APP_TIMER_UNITS:
        raise CaseError(f"{app['_where']}: app_timer_regex and app_timer_unit ({'|'.join(APP_TIMER_UNITS)}) "
                        f"go together")
    try:
        groups = re.compile(rx, re.M).groups
    except re.error as exc:
        raise CaseError(f"{app['_where']}: app_timer_regex does not compile: {exc}")
    if groups != 1:
        raise CaseError(f"{app['_where']}: app_timer_regex needs exactly one capture group")


def app_timers():
    """{app: (regex, seconds per unit)} for the Level 2 applications that print a timer for their ROI."""
    out = {}
    for r in read_table("level2_apps.tsv", L2_APP_COLS):
        check_app_timer(r)
        if r["app_timer_regex"]:
            out[r["app"]] = (r["app_timer_regex"], APP_TIMER_UNITS[r["app_timer_unit"]])
    return out


def level2_rows(backend):
    apps = {r["app"]: r for r in read_table("level2_apps.tsv", L2_APP_COLS)}
    for app in apps.values():
        check_app_timer(app)
    cases = read_table("level2_cases.tsv", L2_CASE_COLS)
    out, seen = [], set()
    for r in cases:
        app = apps.get(r["app"])
        if app is None:
            raise CaseError(f"{r['_where']}: {r['app']} has no row in cases/level2_apps.tsv")
        env = parse_env(r["env"], r["_where"])
        allowed = allowed_env(r["app"], app["extra_env"])
        sel = registry_l2().get(r["app"], ("", set()))[0]
        if sel and any(k == sel for k, _ in env):
            raise CaseError(f"{r['_where']}: {sel} selects a registered input -- measure it with --registry "
                            f"(cases/level2_registry.tsv), not as a hand-written case")
        bad = sorted(k for k, _ in env if k not in allowed)
        if bad:
            raise CaseError(f"{r['_where']}: {bad} are not read by level2/{r['app']}/run.sh "
                            f"(allowed: {', '.join(sorted(allowed)) or 'none'})")
        gpus = r["gpus"] or "1"
        if not gpus.isdigit() or int(gpus) < 1:
            raise CaseError(f"{r['_where']}: gpus must be a positive integer")
        for case, cenv in expand_sweep(r["case"], env, r["_where"]):
            if not NAME_OK.match(case):
                raise CaseError(f"{r['_where']}: case name {case!r} must match [A-Za-z0-9._-]+")
            if (r["app"], case) in seen:
                raise CaseError(f"{r['_where']}: duplicate case {r['app']}/{case}")
            seen.add((r["app"], case))
            if backend.lower() not in app["backends"].split(","):
                continue
            argv = ["bash", f"level2/{r['app']}/run.sh", backend.upper()] + shlex.split(r["args"])
            out.append({
                "level": "2", "app": r["app"], "case": case, "backend": backend.upper(), "gpus": gpus,
                "cwd": REPO, "timeout_s": r["timeout_s"] or app["timeout_s"] or "900",
                "env": env_text(cenv), "argv": " ".join(shlex.quote(a) for a in argv),
                "fom_name": app["fom_name"], "fom_unit": app["fom_unit"], "fom_better": app["fom_better"],
                "fom_source": app["fom_source"], "fom_regex": r["fom_regex"] or app["fom_regex"],
                "roi_excludes": app["roi_excludes"], "verify_vs_roi": "outside",
                "notes": "; ".join(x for x in (app["notes"], r["notes"]) if x), "input_id": "",
            })
    return out, apps


# ----------------------------------------------------------------- selection

def select(rows, specs):
    if not specs or specs == ["all"]:
        return rows
    out, hit = [], set()
    for r in rows:
        for s in specs:
            if s == r["app"] or s == f"{r['app']}/{r['case']}":
                out.append(r)
                hit.add(s)
                break
    missing = [s for s in specs if s not in hit and s != "all"]
    if missing:
        raise CaseError(f"no case matches: {', '.join(missing)}")
    return out


def refuse_undeclared(rows, environ):
    """A variable an application reads, set in the caller's shell but not declared by the
    case, would be dropped by `env -i` -- the run would silently measure another input."""
    problems = []
    reg = registry_l2()
    for r in rows:
        if r["level"] != "2":
            continue
        declared = {kv.split("=", 1)[0] for kv in r["env"].split(";") if kv}
        watched = input_vars(r["app"], reg) | registry_knobs(r["app"], reg)
        stray = sorted(v for v in watched if v in environ and v not in declared)
        # the registry selector set in the shell to ANOTHER input than the case's: the case would win
        # silently and the caller would believe the other input was measured (declared case variables
        # otherwise keep the case's value, as before)
        sel = reg.get(r["app"], ("", set()))[0]
        dvals = dict(kv.split("=", 1) for kv in r["env"].split(";") if kv)
        if sel and sel in dvals and sel in environ and environ[sel] != dvals[sel]:
            stray.append(f"{sel} (shell {environ[sel]!r}, case {dvals[sel]!r})")
        if stray:
            problems.append(f"{r['app']}/{r['case']}: {', '.join(stray)}")
    if problems:
        raise CaseError("set in your shell but not declared by the case (the run would ignore them):\n  "
                        + "\n  ".join(problems)
                        + "\nput them in tools/timing/cases/level2_cases.tsv as a case, or unset them")


def render(rows):
    lines = []
    for r in rows:
        cells = []
        for k in ROW_FIELDS:
            v = str(r[k])
            if "\t" in v or "\n" in v:
                raise CaseError(f"{r['app']}/{r['case']}: field {k} contains a tab or newline")
            cells.append(v if v != "" else "-")
        lines.append("\t".join(cells))
    return "\n".join(lines) + ("\n" if lines else "")


# ----------------------------------------------------------------- check

def check_all(build_root=None):
    msgs = []
    l1, l1apps = level1_rows(build_root, "CUDA")
    msgs.append(f"level1: {len(l1)} CUDA cases, {len(l1apps)} benchmarks described")
    dirs = sorted(d for d in os.listdir(os.path.join(REPO, "level1"))
                  if os.path.isfile(os.path.join(REPO, "level1", d, "CMakeLists.txt")))
    for d in dirs:
        if d not in l1apps:
            raise CaseError(f"level1/{d} has no row in cases/level1_apps.tsv")
    for a in l1apps:
        if a not in dirs:
            raise CaseError(f"cases/level1_apps.tsv lists {a}, which is not a level1 benchmark")
    l2, l2apps = level2_rows("CUDA")
    msgs.append(f"level2: {len(l2)} CUDA cases, {len(l2apps)} applications described")
    apps_with_run = sorted(d for d in os.listdir(os.path.join(REPO, "level2"))
                           if os.path.isfile(os.path.join(REPO, "level2", d, "run.sh")))
    covered = {r["app"] for r in l2}
    for a in apps_with_run:
        if a not in l2apps:
            raise CaseError(f"level2/{a}/run.sh has no row in cases/level2_apps.tsv")
        if a not in covered:
            raise CaseError(f"level2/{a} has no case in cases/level2_cases.tsv")
    for a, app in l2apps.items():
        if app["fom_name"]:
            if app["fom_better"] not in ("higher", "lower"):
                raise CaseError(f"{app['_where']}: fom_better must be higher|lower")
            if app["fom_source"] != "stdout":
                raise CaseError(f"{app['_where']}: fom_source {app['fom_source']!r} is not implemented")
            if re.compile(app["fom_regex"], re.M).groups != 1:
                raise CaseError(f"{app['_where']}: fom_regex needs exactly one capture group")
        elif any(app[k] for k in ("fom_unit", "fom_better", "fom_source", "fom_regex")):
            raise CaseError(f"{app['_where']}: no fom_name but other FOM columns are filled")
    for r in l2:
        if r["fom_regex"] and re.compile(r["fom_regex"], re.M).groups != 1:
            raise CaseError(f"{r['app']}/{r['case']}: fom_regex needs exactly one capture group")
    r1, _ = level1_registry_rows("CUDA")
    r2, _ = level2_registry_rows("CUDA")
    msgs.append(f"registry: {len(r1)} Level 1 and {len(r2)} Level 2 registered inputs (cases/level<N>_registry.tsv)")
    try:
        import gen_registry_cases
    except ImportError as exc:
        raise CaseError(f"cannot check the registry tables against inputs.yaml ({exc}); source hpcperf_env.sh")
    drift = gen_registry_cases.check()
    if drift:
        raise CaseError("; ".join(drift))
    msgs.append("registry: generated tables match the inputs registry (no drift)")
    return msgs


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("resolve")
    r.add_argument("--level", choices=("1", "2"), required=True)
    r.add_argument("--build-root", default=None)
    r.add_argument("--backend", default="CUDA")
    r.add_argument("--no-env-check", action="store_true", help="skip the undeclared-variable refusal (tests)")
    r.add_argument("--registry", action="store_true", help="the registered inputs (cases/level<N>_registry.tsv)")
    r.add_argument("select", nargs="*")
    c = sub.add_parser("check")
    c.add_argument("--build-root", default=None)
    e = sub.add_parser("allowed-env")
    e.add_argument("app")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "resolve":
            if a.registry:
                rows, _ = (level1_registry_rows if a.level == "1" else level2_registry_rows)(a.backend)
            elif a.level == "1":
                if not a.build_root:
                    raise CaseError("--build-root is required for Level 1")
                rows, _ = level1_rows(a.build_root, a.backend)
            else:
                rows, _ = level2_rows(a.backend)
            rows = select(rows, a.select)
            if not a.no_env_check:
                refuse_undeclared(rows, os.environ)
            sys.stdout.write(render(rows))
        elif a.cmd == "check":
            for m in check_all(a.build_root):
                print(f"cases: {m}")
        else:
            apps = {r["app"]: r for r in read_table("level2_apps.tsv", L2_APP_COLS)}
            extra = apps.get(a.app, {}).get("extra_env", "")
            print("\n".join(sorted(allowed_env(a.app, extra))))
    except CaseError as exc:
        print(f"cases: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
