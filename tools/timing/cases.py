#!/usr/bin/env python3
"""Resolve the measurement case tables into the normalized rows the shell engine runs.

    cases.py resolve --level 1 --build-root build/gcc13 [SELECT ...]
    cases.py resolve --level 2 [SELECT ...]
    cases.py resolve --level 1|2|3 --registry [SELECT ...] # the registered inputs (inputs.yaml)
    cases.py resolve --level 3 [SELECT ...]
    cases.py check                    # validate every table; CPU only, used by the tests
    cases.py allowed-env <app>        # the input variables a Level 2 case may set
    cases.py allowed-env --level 3 <app>

SELECT is `all` (default), `<app>`, or `<app>/<case>`.

A case is identified by (level, app, case); a measurement adds the platform. Tables live
in tools/timing/cases/, tab-separated, "-" for an empty field:

  level1.tsv        generated from ctest by gen_cases.py (one row per ctest test)
  level1_extra.tsv  extra Level 1 inputs ctest does not know: app case args env timeout_s notes
  level1_apps.tsv   per-benchmark facts: suite backends roi_excludes verify_vs_roi extra_env notes
  level2_apps.tsv   per-application facts, including the application's own FOM pattern
  level2_cases.tsv  Level 2 cases: app case gpus env args timeout_s fom_regex notes
  level1_registry.tsv, level2_registry.tsv, level3_registry.tsv
                    GENERATED from the inputs registry (level<N>/<app>/inputs.yaml) by
                    gen_registry_cases.py: one case per registered input, case == input_id.
                    Resolved only with --registry; `check` fails when they drift from the
                    registry. The registry, not these files, defines the inputs.
  level3_apps.tsv   per-application facts for Level 3 (no markers: the region is the
                    application's own timer, defined in apptimers.py)
  level3_cases.tsv  Level 3 cases: the level2_cases.tsv columns plus `acceptance` (validate.sh
                    or "-": whether anything checks that input numerically)

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
            "HPCPERF_INPUT_ARGS", "HPCPERF_INPUT_ID",
            "HPCPERF_L3_RUN_SUBDIR"}
PLATFORM = {"HPCPERF_SITE_PROFILE", "HPCPERF_LAUNCHER", "HPCPERF_LAUNCHER_BIN", "HPCPERF_NODES",
            "HPCPERF_GPUS_PER_NODE", "HPCPERF_GPU_BACKEND", "HPCPERF_RUN_TMPDIR", "HPCPERF_CPUS_PER_RANK"}
CASE_SETTABLE_PLATFORM = {"HPCPERF_CPUS_PER_RANK"}
# read for every Level 3 application by level3/tools/l3_common.sh, not by the run.sh text
L3_COMMON_INPUTS = {"HPCPERF_SCALE_MODE"}

ROW_FIELDS = ("level", "app", "case", "backend", "gpus", "cwd", "timeout_s", "env", "argv",
              "fom_name", "fom_unit", "fom_better", "fom_source", "fom_regex",
              "roi_excludes", "verify_vs_roi", "notes", "input_id", "nvtx_roi", "profile")
# input_id: the registry input a --registry case measures ("" for the hand-written cases); the
# engine stores that input's workload identity next to the raw runs and refuses the case when the
# identity cannot be established. nvtx_roi / profile: Level 3 (see level3_rows).


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
L2_BIND_COLS = ("app", "launcher", "host_threads", "cpus_per_rank", "omp_pin", "basis")
OMP_PIN_ENV = ("OMP_PLACES=cores", "OMP_PROC_BIND=close")


def binding_table():
    """cases/level2_binding.tsv: the CPU-binding policy of every Level 2 application, checked."""
    rows = {r["app"]: r for r in read_table("level2_binding.tsv", L2_BIND_COLS)}
    for r in rows.values():
        if r["launcher"] not in ("mpirun", "direct"):
            raise CaseError(f"{r['_where']}: launcher must be mpirun|direct")
        if r["omp_pin"] not in ("yes", "no"):
            raise CaseError(f"{r['_where']}: omp_pin must be yes|no")
        if r["launcher"] == "direct":
            if r["host_threads"] or r["cpus_per_rank"] or r["omp_pin"] == "yes":
                raise CaseError(f"{r['_where']}: a direct-exec application has no binding (host_threads / cpus_per_rank '-', omp_pin no)")
            continue
        if not (r["host_threads"].isdigit() and r["cpus_per_rank"].isdigit()) or int(r["host_threads"]) < 1:
            raise CaseError(f"{r['_where']}: host_threads and cpus_per_rank must be positive integers")
        if int(r["cpus_per_rank"]) != int(r["host_threads"]):
            raise CaseError(f"{r['_where']}: cpus_per_rank must equal host_threads (one core per host thread, no idle bound cores)")
        if r["omp_pin"] == "yes" and int(r["host_threads"]) < 2:
            raise CaseError(f"{r['_where']}: omp_pin yes needs more than one host thread")
    return rows


def binding_policy(app):
    """The explicit CPU-binding policy of one Level 2 application."""
    rows = binding_table()
    if app not in rows:
        raise CaseError(f"level2/{app} has no row in cases/level2_binding.tsv (its CPU-binding policy)")
    return rows[app]


def binding_env(app):
    """NAME=VALUE pairs that put the explicit policy into effect through the launcher's interface
    (HPCPERF_CPUS_PER_RANK -> mpirun --map-by ppr:N:node:PE=<cpus_per_rank> --bind-to core) and, for an
    application with several host threads, pin its OpenMP threads one per core inside that set.
    Empty for a direct-exec application (nothing to bind through)."""
    r = binding_policy(app)
    if r["launcher"] == "direct":
        return []
    out = [f"HPCPERF_CPUS_PER_RANK={r['cpus_per_rank']}"]
    if r["omp_pin"] == "yes":
        out += [f"OMP_NUM_THREADS={r['host_threads']}", *OMP_PIN_ENV]
    return out


def binding_meta(app, policy):
    """run_meta.txt lines (bind_*) that record the policy applied to a run."""
    if policy == "runtime":
        r = binding_policy(app)
        return [f"bind_policy=runtime", f"bind_launcher={r['launcher']}"]
    r = binding_policy(app)
    return ["bind_policy=explicit", f"bind_launcher={r['launcher']}", f"bind_host_threads={r['host_threads'] or '-'}",
            f"bind_cpus_per_rank={r['cpus_per_rank'] or '-'}", f"bind_omp_pin={r['omp_pin']}",
            "bind_env=" + ";".join(binding_env(app))]
L1_REG_COLS = ("app", "case", "input_id", "materialized", "exe", "args", "cwd", "env", "timeout_s")
L2_REG_COLS = ("app", "case", "input_id", "materialized", "selector", "registry_knobs", "gpus", "timeout_s")


def registry_l2():
    """{app: (selector, {knob names})} from the generated Level 2 registry table: the selector
    variable run.sh reads indirectly (hpcperf_apply_input) and the knobs it sets from the registry."""
    out = {}
    for r in read_table("level2_registry.tsv", L2_REG_COLS):
        out[r["app"]] = (r["selector"], set(x for x in r["registry_knobs"].split(",") if x))
    return out
L3_APP_COLS = ("app", "backends", "timeout_s", "fom_name", "fom_unit", "fom_better", "fom_source",
               "fom_regex", "extra_env", "nvtx_roi", "profile", "notes")
L3_CASE_COLS = ("app", "case", "gpus", "env", "args", "timeout_s", "fom_regex", "acceptance", "notes")

L3_REG_COLS = L2_REG_COLS


def registry_l3():
    """{app: (selector, {knob names})} from the generated Level 3 registry table (cases/level3_registry.tsv):
    the same shape as registry_l2; every Level 3 run.sh applies its registered input through
    hpcperf_apply_input (or its own equivalent, LAMMPS / SPARTA) when the selector is set."""
    out = {}
    for r in read_table("level3_registry.tsv", L3_REG_COLS):
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


def run_sh_vars(app, level=2):
    path = os.path.join(REPO, f"level{level}", app, "run.sh")
    if not os.path.isfile(path):
        raise CaseError(f"level{level}/{app}/run.sh does not exist")
    with open(path) as f:
        return set(VAR_REF.findall(f.read()))


def input_vars(app, registry=None, level=2):
    """Variables that change what a Level 2/3 application computes: read by its run.sh (directly,
    or -- for the Level 2 registry selector -- through tools/inputs/hpcperf_input_selector.sh, which
    the text scan cannot see; Level 3: or by the common helpers for every application), not part of
    the launcher / platform interface."""
    if level == 3:
        common = L3_COMMON_INPUTS
        reg = registry_l3() if registry is None else registry
    else:
        common = set()
        reg = registry_l2() if registry is None else registry
    sel = {reg[app][0]} if app in reg and reg[app][0] else set()
    return {v for v in run_sh_vars(app, level) | common | sel if v not in RESERVED and v not in PLATFORM
            and not v.startswith("HPCPERF_BIND_")}


def registry_knobs(app, registry=None, level=2):
    """Knobs the registry selector sets inside run.sh; a case never sets them directly."""
    reg = (registry_l3() if level == 3 else registry_l2()) if registry is None else registry
    return reg.get(app, ("", set()))[1]


def allowed_env(app, extra, registry=None, level=2):
    return input_vars(app, registry, level) | CASE_SETTABLE_PLATFORM | set(x for x in extra.split(",") if x)


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
            "notes": "; ".join(x for x in (app["notes"], notes) if x), "input_id": "", "nvtx_roi": "", "profile": "yes",
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
            "input_id": r["input_id"], "nvtx_roi": "", "profile": "yes",
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
            "input_id": r["input_id"], "nvtx_roi": "", "profile": "yes",
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
                "notes": "; ".join(x for x in (app["notes"], r["notes"]) if x), "input_id": "", "nvtx_roi": "", "profile": "yes",
            })
    return out, apps


# ----------------------------------------------------------------- level 3

def level3_apps_in_suite():
    """level3/<app> directories with a run.sh, except those benchmark.yaml marks
    `suite_status: retired` (GEOS: kept for provenance, never built or measured)."""
    out = []
    for d in sorted(os.listdir(os.path.join(REPO, "level3"))):
        if not os.path.isfile(os.path.join(REPO, "level3", d, "run.sh")):
            continue
        y = os.path.join(REPO, "level3", d, "benchmark.yaml")
        if os.path.isfile(y) and re.search(r"^suite_status:\s*retired\b", open(y).read(), re.M):
            continue
        out.append(d)
    return out


def level3_rows(backend):
    """Level 3: full applications without markers. The timed region of every application is
    its own timer (apptimers.py); the table only says how to run it and what it prints."""
    import apptimers
    apps = {r["app"]: r for r in read_table("level3_apps.tsv", L3_APP_COLS)}
    for a, app in apps.items():
        if a not in apptimers.TIMERS:
            raise CaseError(f"{app['_where']}: {a} has no timer definition in tools/timing/apptimers.py")
        if not (app["profile"] == "yes" or re.fullmatch(r"no \(.{10,}\)", app["profile"])):
            raise CaseError(f"{app['_where']}: profile must be 'yes' or 'no (<the reason>)'")
    cases = read_table("level3_cases.tsv", L3_CASE_COLS)
    out, seen = [], set()
    for r in cases:
        app = apps.get(r["app"])
        if app is None:
            raise CaseError(f"{r['_where']}: {r['app']} has no row in cases/level3_apps.tsv")
        env = parse_env(r["env"], r["_where"])
        allowed = allowed_env(r["app"], app["extra_env"], level=3)
        sel = registry_l3().get(r["app"], ("", set()))[0]
        if sel and any(k == sel for k, _ in env):
            raise CaseError(f"{r['_where']}: {sel} selects a registered input -- measure it with --registry "
                            f"(cases/level3_registry.tsv), not as a hand-written case")
        bad = sorted(k for k, _ in env if k not in allowed)
        if bad:
            raise CaseError(f"{r['_where']}: {bad} are not read by level3/{r['app']}/run.sh "
                            f"(allowed: {', '.join(sorted(allowed)) or 'none'})")
        gpus = r["gpus"] or "1"
        if not gpus.isdigit() or int(gpus) < 1:
            raise CaseError(f"{r['_where']}: gpus must be a positive integer")
        if r["acceptance"] not in ("", "validate.sh"):
            raise CaseError(f"{r['_where']}: acceptance must be validate.sh or -")
        for case, cenv in expand_sweep(r["case"], env, r["_where"]):
            if not NAME_OK.match(case):
                raise CaseError(f"{r['_where']}: case name {case!r} must match [A-Za-z0-9._-]+")
            if (r["app"], case) in seen:
                raise CaseError(f"{r['_where']}: duplicate case {r['app']}/{case}")
            seen.add((r["app"], case))
            if backend.lower() not in app["backends"].split(","):
                continue
            argv = ["bash", f"level3/{r['app']}/run.sh", backend.upper()] + shlex.split(r["args"])
            out.append({
                "level": "3", "app": r["app"], "case": case, "backend": backend.upper(), "gpus": gpus,
                "cwd": REPO, "timeout_s": r["timeout_s"] or app["timeout_s"] or "1800",
                "env": env_text(cenv), "argv": " ".join(shlex.quote(a) for a in argv),
                "fom_name": app["fom_name"], "fom_unit": app["fom_unit"], "fom_better": app["fom_better"],
                "fom_source": app["fom_source"], "fom_regex": r["fom_regex"] or app["fom_regex"],
                "roi_excludes": "", "verify_vs_roi": "outside" if r["acceptance"] else "none",
                "notes": "; ".join(x for x in (app["notes"], r["notes"]) if x), "input_id": "", "nvtx_roi": app["nvtx_roi"],
                "profile": app["profile"],
            })
    return out, apps


def level3_registry_rows(backend):
    """Level 3 cases of the registered inputs: run.sh with the registry selector = input id and
    HPCPERF_GPUS = the input's runtime_config.gpus; the region is the application's own timer
    (apptimers.py), the FOM / NVTX range / profile default come from cases/level3_apps.tsv."""
    import apptimers
    apps = {r["app"]: r for r in read_table("level3_apps.tsv", L3_APP_COLS)}
    reg = registry_l3()
    out, seen = [], set()
    for r in read_table("level3_registry.tsv", L3_REG_COLS):
        app = apps.get(r["app"])
        if app is None:
            raise CaseError(f"{r['_where']}: {r['app']} has no row in cases/level3_apps.tsv")
        if r["app"] not in apptimers.TIMERS:
            raise CaseError(f"{r['_where']}: {r['app']} has no timer definition in tools/timing/apptimers.py")
        if r["case"] != r["input_id"] or not NAME_OK.match(r["case"]):
            raise CaseError(f"{r['_where']}: case must equal the input id and match [A-Za-z0-9._-]+")
        if (r["app"], r["case"]) in seen:
            raise CaseError(f"{r['_where']}: duplicate registry case {r['app']}/{r['case']}")
        seen.add((r["app"], r["case"]))
        env = parse_env(f"{r['selector']}={r['input_id']}", r["_where"])
        allowed = allowed_env(r["app"], app["extra_env"], reg, level=3)
        if r["selector"] not in allowed:
            raise CaseError(f"{r['_where']}: selector {r['selector']} is not an input variable of {r['app']}")
        gpus = r["gpus"] or "1"
        if not gpus.isdigit() or int(gpus) < 1:
            raise CaseError(f"{r['_where']}: gpus must be a positive integer")
        if backend.lower() not in app["backends"].split(","):
            continue
        argv = ["bash", f"level3/{r['app']}/run.sh", backend.upper()]
        out.append({
            "level": "3", "app": r["app"], "case": r["case"], "backend": backend.upper(), "gpus": gpus,
            "cwd": REPO, "timeout_s": r["timeout_s"] or app["timeout_s"] or "1800",
            "env": env_text(env), "argv": " ".join(shlex.quote(a) for a in argv),
            "fom_name": app["fom_name"], "fom_unit": app["fom_unit"], "fom_better": app["fom_better"],
            "fom_source": app["fom_source"], "fom_regex": app["fom_regex"],
            "roi_excludes": "", "verify_vs_roi": "outside",
            "notes": "; ".join(x for x in (app["notes"], "registry input" + ("" if r["materialized"] == "1" else ", NOT materialized")) if x),
            "input_id": r["input_id"], "nvtx_roi": app["nvtx_roi"], "profile": app["profile"],
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
    reg2, reg3 = registry_l2(), registry_l3()
    for r in rows:
        if r["level"] not in ("2", "3"):
            continue
        level = int(r["level"])
        reg = reg3 if level == 3 else reg2
        declared = {kv.split("=", 1)[0] for kv in r["env"].split(";") if kv}
        watched = input_vars(r["app"], reg, level) | registry_knobs(r["app"], reg, level)
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
                        + "\nput them in tools/timing/cases/level<L>_cases.tsv as a case, or unset them")


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
    bind = binding_table()
    for a in l2apps:
        if a not in bind:
            raise CaseError(f"level2/{a} has no row in cases/level2_binding.tsv (its CPU-binding policy)")
    for a in bind:
        if a not in l2apps:
            raise CaseError(f"cases/level2_binding.tsv lists {a}, which is not a level2 application")
    msgs.append(f"level2 binding policy: {sum(1 for r in bind.values() if r['launcher'] == 'mpirun')} mpirun-launched applications "
                f"bound through HPCPERF_CPUS_PER_RANK, {sum(1 for r in bind.values() if r['launcher'] == 'direct')} direct-exec unbound")
    r1, _ = level1_registry_rows("CUDA")
    r2, _ = level2_registry_rows("CUDA")
    r3, _ = level3_registry_rows("CUDA")
    msgs.append(f"registry: {len(r1)} Level 1, {len(r2)} Level 2 and {len(r3)} Level 3 registered inputs (cases/level<N>_registry.tsv)")
    try:
        import gen_registry_cases
    except ImportError as exc:
        raise CaseError(f"cannot check the registry tables against inputs.yaml ({exc}); source hpcperf_env.sh")
    drift = gen_registry_cases.check()
    if drift:
        raise CaseError("; ".join(drift))
    msgs.append("registry: generated tables match the inputs registry (no drift)")
    l3, l3apps = level3_rows("CUDA")
    msgs.append(f"level3: {len(l3)} CUDA cases, {len(l3apps)} applications described")
    l3_dirs = level3_apps_in_suite()
    covered3 = {r["app"] for r in l3}
    for a in l3_dirs:
        if a not in l3apps:
            raise CaseError(f"level3/{a}/run.sh has no row in cases/level3_apps.tsv")
        if a not in covered3:
            raise CaseError(f"level3/{a} has no case in cases/level3_cases.tsv")
    for a, app in l3apps.items():
        if app["fom_name"]:
            if app["fom_better"] not in ("higher", "lower") or app["fom_source"] != "stdout":
                raise CaseError(f"{app['_where']}: fom_better must be higher|lower and fom_source stdout")
            if re.compile(app["fom_regex"], re.M).groups != 1:
                raise CaseError(f"{app['_where']}: fom_regex needs exactly one capture group")
        elif any(app[k] for k in ("fom_unit", "fom_better", "fom_source", "fom_regex")):
            raise CaseError(f"{app['_where']}: no fom_name but other FOM columns are filled")
    return msgs


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("resolve")
    r.add_argument("--level", choices=("1", "2", "3"), required=True)
    r.add_argument("--build-root", default=None)
    r.add_argument("--backend", default="CUDA")
    r.add_argument("--no-env-check", action="store_true", help="skip the undeclared-variable refusal (tests)")
    r.add_argument("--registry", action="store_true", help="the registered inputs (cases/level<N>_registry.tsv)")
    r.add_argument("select", nargs="*")
    c = sub.add_parser("check")
    c.add_argument("--build-root", default=None)
    e = sub.add_parser("allowed-env")
    e.add_argument("--level", choices=("2", "3"), default="2")
    e.add_argument("app")
    b = sub.add_parser("binding-env", help="NAME=VALUE pairs of the explicit CPU-binding policy of a Level 2 application")
    b.add_argument("app")
    bm = sub.add_parser("binding-meta", help="run_meta.txt lines recording the binding policy applied")
    bm.add_argument("--policy", choices=("explicit", "runtime"), default="explicit")
    bm.add_argument("app")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "resolve":
            if a.registry:
                rows, _ = {"1": level1_registry_rows, "2": level2_registry_rows, "3": level3_registry_rows}[a.level](a.backend)
            elif a.level == "1":
                if not a.build_root:
                    raise CaseError("--build-root is required for Level 1")
                rows, _ = level1_rows(a.build_root, a.backend)
            elif a.level == "2":
                rows, _ = level2_rows(a.backend)
            else:
                rows, _ = level3_rows(a.backend)
            rows = select(rows, a.select)
            if not a.no_env_check:
                refuse_undeclared(rows, os.environ)
            sys.stdout.write(render(rows))
        elif a.cmd == "check":
            for m in check_all(a.build_root):
                print(f"cases: {m}")
        elif a.cmd == "binding-env":
            sys.stdout.write("".join(f"{kv}\n" for kv in binding_env(a.app)))
        elif a.cmd == "binding-meta":
            sys.stdout.write("".join(f"{line}\n" for line in binding_meta(a.app, a.policy)))
        else:
            lvl = int(a.level)
            table, cols = ("level2_apps.tsv", L2_APP_COLS) if lvl == 2 else ("level3_apps.tsv", L3_APP_COLS)
            apps = {r["app"]: r for r in read_table(table, cols)}
            extra = apps.get(a.app, {}).get("extra_env", "")
            print("\n".join(sorted(allowed_env(a.app, extra, level=lvl))))
    except CaseError as exc:
        print(f"cases: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
