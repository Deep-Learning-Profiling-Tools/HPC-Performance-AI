#!/usr/bin/env python3
"""hpcperf_inputs.py -- minimal common input registry for a benchmark directory.

A benchmark that supports several inputs carries an `inputs.yaml` (schema
hpcperf-inputs-1) next to its run.sh / CMakeLists.txt. The file is the single
source of truth for:

  * the stable input ids and what each one IS (case, parameters, upstream origin,
    build-time / run-time configuration, seed, validated backends);
  * how the benchmark's own timer is read (which line, which field, which unit,
    what the timed region covers) -- the tool never substitutes end-to-end wall
    time for a missing or malformed timer value; optional secondary timers are
    reported next to it, never added to it;
  * which scientific quantities form the baseline output and how a later run is
    compared with them (rule per quantity; `record` = kept, not verified).

Sub-commands (all read-only except `measure`, which writes into --out):

  validate <bench_dir>                     schema / consistency check (exit 1 on error)
  list     <bench_dir>                     one line per input id
  show     <bench_dir> <input_id>          the resolved entry as JSON
  args     <bench_dir> <input_id>          the input's command-line arguments, one per line
                                           (run.sh reads these; unknown id -> exit 2)
  param    <bench_dir> <input_id> <key>    one parameter value (exit 2 if unknown)
  shell-env <bench_dir> <input_id>         the input's env knobs and extra args as tab-separated
                                           lines (E/A) for run.sh's hpcperf_apply_input
  parse-timing <bench_dir> <log> [--rc N] [--input ID]
                                           main-compute time from a log (JSON); a nonzero
                                           --rc or a missing/ambiguous/non-finite timer line -> exit 1
  extract  <bench_dir> <log> [--input ID]  the baseline quantities of a log (JSON)
  compare  <bench_dir> <baseline.json> <log> [--input ID] [--rc N]
                                           compare a log against a stored baseline:
                                           exit 0 verified = no rule failed AND every REQUIRED
                                           quantity was compared by a verifying rule (verdict PASS);
                                           1 a rule failed / the candidate run failed (verdict FAIL);
                                           2 refused: baseline belongs to another input or benchmark,
                                           was recorded for a different workload (params/args/files),
                                           is bound to another input, or is the same output file;
                                           3 nothing failed but the comparison is not complete: a
                                           required quantity is still `record`, or the workload
                                           identity of baseline/candidate is not established (old
                                           record without identity, incomplete identity) -- verdict
                                           INCOMPLETE, never a pass. Identity is established by a
                                           complete matching `workload` (measured or migrated with
                                           evidence) or by a `reference_binding` of an upstream
                                           reference to this input with recorded evidence.
  status   <bench_dir> <measurement.json>  the status vocabulary of a finished measurement (the
                                           self-comparison is re-derived over the runs OTHER than
                                           the baseline run)
  migrate-baseline <bench_dir> <baseline.json> --input ID --evidence <measurement.json> [--note ..] [--out F]
                                           attach a workload identity to an old baseline from the
                                           evidence of the measurement it came from (same benchmark,
                                           input id, inputs.yaml sha256, command/selector); writes a
                                           NEW file and records source/evidence/basis; never copies
                                           the candidate's identity
  measure  <bench_dir> <input_id> --out DIR [--warmup 1] [--reps 3] [--timeout S] [--gpus 1]
                                           warm-up run + N measured runs, one directory per run,
                                           timing + baseline + summary (median, min, max, MAD, spread)

Status vocabulary (summary of `measure`, also printed by `status`), kept separate on purpose:
  run_completed        every measured run exited 0
  timing_ok            every measured run yielded a main-compute value from the benchmark's own timer
  native_check         PASS / FAIL / NONE -- rules that need no baseline (present, absent, abs_lt,
                       ge, le) evaluated on every measured run; NONE when the input has no such rule
  baseline_saved       the quantities of the first successful measured run were stored
  comparison_rules     READY (every REQUIRED quantity has a verifying rule; diagnostic records allowed) /
                       PARTIAL (a required quantity is still record) / NONE (no required quantity verified)
  needs_validation     the REQUIRED quantities whose rule is `record` (tolerance not fixed yet)
  baseline_verdict     PASS / INCOMPLETE / FAIL of the measured runs against the working baseline
Quantity roles (inputs.yaml `role: required|diagnostic`, default required): only required quantities
decide acceptance; diagnostic quantities are reported and may stay `record` without a tolerance.
Rollout fields (optional, backward compatible): `timing.kind: none` + `status: NEEDS_TIMING_SUPPORT` +
`reason` for a benchmark without a usable native timer (measure records E2E only, never as main
compute); top-level `coverage: {status: MULTI_INPUT|SINGLE_INPUT|BLOCKED, reason, blocker,
upstream_inputs_not_added}` for the audit; per-input `input_form: runtime|file|compile-time`.
  compute_ge_1s        median main compute >= 1 s (a reference value, not a gate)
  stable               n >= 3 and (max-min)/median <= 0.10 over the measured runs

Exit codes: 0 ok, 1 validation/measurement/comparison failure, 2 usage / unknown input /
inconsistent baseline, 3 comparison inconclusive (record-only rules).
"""
import argparse, hashlib, json, math, os, re, shutil, socket, statistics, subprocess, sys, time
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.stderr.write("hpcperf_inputs: PyYAML is required (source hpcperf_env.sh)\n"); sys.exit(2)

SCHEMA = "hpcperf-inputs-1"
UNIT_TO_S = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
SOURCE_KINDS = ("upstream-file", "upstream-parameterized", "derived", "custom")
VARIANTS = ("size", "case", "parameter", "implementation-path", "steps", "build-config", "default")
ENTRY_KINDS = ("run.sh", "binary")
SELECT = ("first", "last", "only")
RULES = ("exact", "rel", "abs", "abs_lt", "present", "absent", "record", "ge", "le")
VERIFYING_RULES = ("exact", "rel", "abs", "abs_lt", "present", "absent", "ge", "le")
NATIVE_RULES = ("present", "absent", "abs_lt", "ge", "le")     # need no baseline
STABLE_SPREAD = 0.10


class InputError(Exception):
    pass


# ----------------------------------------------------------------------------- registry
def repo_root(bench_dir: Path) -> Path:
    p = bench_dir.resolve()
    for cand in [p] + list(p.parents):
        if (cand / "hpcperf_env.sh").is_file() and (cand / "level1").is_dir():
            return cand
    raise InputError(f"{bench_dir}: not inside the HPC-Performance-AI repository")


def load(bench_dir: Path) -> dict:
    f = Path(bench_dir) / "inputs.yaml"
    if not f.is_file():
        raise InputError(f"{f}: no inputs.yaml (this benchmark has no registered inputs)")
    with open(f) as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        raise InputError(f"{f}: top level must be a mapping")
    doc["_path"] = str(f)
    doc["_sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
    return doc


COVERAGE_STATUS = ("MULTI_INPUT", "SINGLE_INPUT", "BLOCKED")
INPUT_FORMS = ("runtime", "file", "compile-time")


def _check_timer(t, where, errs, need_work=True):
    if t.get("kind") == "none":
        # no usable native timer: the registry records why; measure() reports NEEDS_TIMING_SUPPORT
        # and records only E2E wall as auxiliary information (never as main compute)
        if not t.get("status") == "NEEDS_TIMING_SUPPORT" or not t.get("reason"):
            errs.append(f"{where}.kind none requires status: NEEDS_TIMING_SUPPORT and a reason")
        return
    for k in ("regex", "unit", "select"):
        if k not in t:
            errs.append(f"{where}.{k} missing")
    if t.get("unit") not in UNIT_TO_S:
        errs.append(f"{where}.unit must be one of {sorted(UNIT_TO_S)}")
    if t.get("select") not in SELECT:
        errs.append(f"{where}.select must be one of {SELECT}")
    if t.get("kind", "total") not in ("total", "per_iteration", "per_step", "none"):
        errs.append(f"{where}.kind must be total | per_iteration | per_step | none")
    if need_work and t.get("kind") in ("per_iteration", "per_step") and not t.get("work"):
        errs.append(f"{where}.work {{key, offset}} is required for per_iteration/per_step timers")
    try:
        rx = re.compile(t.get("regex", ""))
        if "value" not in rx.groupindex:
            errs.append(f"{where}.regex needs a named group (?P<value>...)")
    except re.error as ex:
        errs.append(f"{where}.regex does not compile: {ex}")


def validate(doc: dict) -> list:
    errs = []
    for k in ("schema", "benchmark", "level", "entry", "default_input", "timing", "baseline", "inputs"):
        if k not in doc:
            errs.append(f"missing top-level key '{k}'")
    if errs:
        return errs
    if doc["schema"] != SCHEMA:
        errs.append(f"schema must be {SCHEMA} (got {doc['schema']})")
    if doc["level"] not in (1, 2, 3):
        errs.append("level must be 1, 2 or 3")
    e = doc["entry"]
    if not isinstance(e, dict) or e.get("kind") not in ENTRY_KINDS or not e.get("path"):
        errs.append("entry must be {kind: run.sh|binary, path: <repo-relative>}")
    if isinstance(e, dict) and e.get("kind") == "run.sh" and not doc.get("selector"):
        errs.append("a run.sh entry needs 'selector' (the HPCPERF_<APP>_INPUT variable run.sh honours)")
    t = doc["timing"]
    if not isinstance(t, dict):
        errs.append("timing must be a mapping"); t = {}
    if "scope" not in t and t.get("kind") != "none":
        errs.append("timing.scope missing (where the timer starts/ends, what it includes/excludes)")
    if "kind" not in t:
        errs.append("timing.kind missing")
    _check_timer(t, "timing", errs)
    for i, sec in enumerate(t.get("secondary") or []):
        if not isinstance(sec, dict) or not sec.get("name") or not sec.get("scope"):
            errs.append(f"timing.secondary[{i}] needs name and scope")
        else:
            _check_timer(sec, f"timing.secondary[{sec['name']}]", errs, need_work=False)

    def check_quantities(qs, where):
        for q in qs:
            if not isinstance(q, dict) or not q.get("name") or not q.get("regex"):
                errs.append(f"{where}: quantity {q!r} needs name and regex"); continue
            cmp = q.get("compare", {})
            if cmp.get("rule") not in RULES:
                errs.append(f"{where}: quantity {q['name']}: compare.rule must be one of {RULES}")
            if cmp.get("rule") in ("rel", "abs", "abs_lt", "ge", "le") and "tol" not in cmp and "value" not in cmp:
                errs.append(f"{where}: quantity {q['name']}: rule {cmp.get('rule')} needs tol or value")
            if q.get("select", "last") not in SELECT:
                errs.append(f"{where}: quantity {q['name']}: select must be one of {SELECT}")
            if q.get("role", "required") not in ("required", "diagnostic"):
                errs.append(f"{where}: quantity {q['name']}: role must be required | diagnostic")
            try:
                rx = re.compile(q["regex"])
                if cmp.get("rule") not in ("present", "absent") and "value" not in rx.groupindex:
                    errs.append(f"{where}: quantity {q['name']}: regex needs (?P<value>...)")
            except re.error as ex:
                errs.append(f"{where}: quantity {q['name']}: regex does not compile: {ex}")
    b = doc["baseline"]
    if not isinstance(b, dict) or not isinstance(b.get("quantities"), list) or not b["quantities"]:
        errs.append("baseline.quantities must be a non-empty list")
    else:
        check_quantities(b["quantities"], "baseline")
    for inp in doc.get("inputs", []):
        ov = inp.get("baseline") if isinstance(inp, dict) else None
        if ov:
            if not isinstance(ov, dict) or not isinstance(ov.get("quantities"), list) or not ov["quantities"]:
                errs.append(f"input '{inp.get('id')}': baseline override needs a non-empty quantities list"); continue
            check_quantities(ov["quantities"], f"input '{inp.get('id')}' baseline override")
    ids = []
    for inp in doc["inputs"]:
        if not isinstance(inp, dict) or not inp.get("id"):
            errs.append(f"input {inp!r}: needs id"); continue
        i = inp["id"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", i):
            errs.append(f"input '{i}': id must match [a-z0-9][a-z0-9._-]*")
        if i in ids:
            errs.append(f"input '{i}': duplicate id")
        ids.append(i)
        for k in ("case", "variant", "source", "params"):
            if k not in inp:
                errs.append(f"input '{i}': missing '{k}'")
        if inp.get("variant") not in VARIANTS:
            errs.append(f"input '{i}': variant must be one of {VARIANTS}")
        src = inp.get("source") or {}
        if src.get("kind") not in SOURCE_KINDS:
            errs.append(f"input '{i}': source.kind must be one of {SOURCE_KINDS}")
        if src.get("kind") in ("derived", "custom") and not src.get("derivation"):
            errs.append(f"input '{i}': derived/custom inputs must state 'derivation'")
        if src.get("kind") in ("upstream-file", "upstream-parameterized") and not src.get("upstream"):
            errs.append(f"input '{i}': upstream inputs must state 'upstream' (repo/version/path)")
        if isinstance(e, dict) and e.get("kind") == "binary" and not isinstance(inp.get("args"), list):
            errs.append(f"input '{i}': binary entries need an 'args' list")
        if "backends_validated" not in inp:
            errs.append(f"input '{i}': missing backends_validated (may be an empty list)")
        for f in inp.get("files", []) or []:
            if not (Path(doc["_path"]).parent / f).exists():
                errs.append(f"input '{i}': referenced file '{f}' does not exist under the benchmark directory")
    if doc["default_input"] not in ids:
        errs.append(f"default_input '{doc['default_input']}' is not a registered input id")
    cov = doc.get("coverage")
    if cov is not None:
        if not isinstance(cov, dict) or cov.get("status") not in COVERAGE_STATUS:
            errs.append(f"coverage.status must be one of {COVERAGE_STATUS}")
        else:
            runnable = [i for i in doc["inputs"] if isinstance(i, dict) and i.get("materialized", True) is not False]
            if cov["status"] == "MULTI_INPUT" and len(runnable) < 2:
                errs.append("coverage MULTI_INPUT needs at least two runnable (materialized) inputs")
            if cov["status"] == "SINGLE_INPUT" and len(runnable) != 1:
                errs.append("coverage SINGLE_INPUT means exactly one runnable (materialized) input")
            if cov["status"] == "BLOCKED" and len(runnable) > 1:
                errs.append("coverage BLOCKED means at most one runnable input (with two or more it is MULTI_INPUT; list the missing upstream inputs under upstream_inputs_not_added)")
            if cov["status"] in ("SINGLE_INPUT", "BLOCKED") and not cov.get("reason"):
                errs.append(f"coverage {cov['status']} needs a reason")
            if cov["status"] == "BLOCKED" and not cov.get("blocker"):
                errs.append("coverage BLOCKED needs 'blocker' (what is missing to add the other upstream inputs)")
    for inp in doc["inputs"]:
        if isinstance(inp, dict) and inp.get("input_form", "runtime") not in INPUT_FORMS:
            errs.append(f"input '{inp.get('id')}': input_form must be one of {INPUT_FORMS}")
        # a compile-time configuration that has no built binary/deck is registered but not runnable
        if isinstance(inp, dict) and inp.get("materialized", True) is False and inp.get("input_form") != "compile-time":
            errs.append(f"input '{inp.get('id')}': materialized: false is only meaningful for input_form compile-time")
        if isinstance(inp, dict) and inp.get("input_form") == "compile-time" and not inp.get("build_config"):
            errs.append(f"input '{inp.get('id')}': a compile-time input needs build_config (what is fixed at build time)")
    return errs


def get_input(doc: dict, input_id: str) -> dict:
    for inp in doc["inputs"]:
        if inp.get("id") == input_id:
            return inp
    known = ", ".join(i.get("id", "?") for i in doc["inputs"])
    raise InputError(f"unknown input id '{input_id}' for {doc['benchmark']} (registered: {known})")


def quantities(doc: dict, inp=None):
    """The baseline quantity list: an input may override the benchmark-wide one
    (e.g. a deck with another thermo output style)."""
    if inp and isinstance(inp.get("baseline"), dict) and inp["baseline"].get("quantities"):
        return inp["baseline"]["quantities"]
    return doc["baseline"]["quantities"]


# ----------------------------------------------------------------------------- parsing
def _matches(regex: str, lines, section_start=None):
    rx = re.compile(regex)
    active = section_start is None
    srx = re.compile(section_start) if section_start else None
    out = []
    for ln in lines:
        if not active:
            if srx.search(ln):
                active = True
            continue
        m = rx.search(ln)
        if m:
            out.append(m)
    return out


def _pick(matches, select, what):
    if not matches:
        raise InputError(f"{what}: no matching line in the log")
    if select == "only" and len(matches) != 1:
        raise InputError(f"{what}: expected exactly one matching line, found {len(matches)}")
    return matches[0] if select == "first" else matches[-1]


def _finite(raw: str, what: str) -> float:
    try:
        v = float(raw)
    except ValueError:
        raise InputError(f"{what}: value '{raw}' is not a number")
    if not math.isfinite(v):
        raise InputError(f"{what}: value '{raw}' is not finite")
    return v


def _resolution(raw: str, unit: str):
    """Print resolution of a timer value from its decimal places ('0.2876' s -> 1e-4 s)."""
    m = re.match(r"^[-+]?\d*\.(\d+)$", raw.strip())
    if not m:
        return None
    return 10.0 ** (-len(m.group(1))) * UNIT_TO_S[unit]


def _read_timer(t: dict, lines, what: str, params=None) -> dict:
    m = _pick(_matches(t["regex"], lines, t.get("section_start")), t["select"], what)
    raw = m.group("value")
    v = _finite(raw, what)
    if v < 0:
        raise InputError(f"{what}: negative timer value {raw}")
    seconds = v * UNIT_TO_S[t["unit"]]
    res = {"raw_value": v, "raw_text": raw, "unit": t["unit"], "kind": t.get("kind", "total"),
           "line": m.group(0).strip(), "scope": t.get("scope"),
           "print_resolution_s": _resolution(raw, t["unit"])}
    if t.get("kind") in ("per_iteration", "per_step"):
        w = t["work"]
        if params is None or w["key"] not in params:
            raise InputError(f"{what}: work key '{w['key']}' not in the input parameters")
        n = int(params[w["key"]]) + int(w.get("offset", 0))
        if n <= 0:
            raise InputError(f"{what}: work count {n} <= 0 for key {w['key']}")
        res.update({"per_unit_s": seconds, "work_count": n, "seconds": seconds * n,
                    "print_resolution_s": (res["print_resolution_s"] * n) if res["print_resolution_s"] else None})
    else:
        res["seconds"] = seconds
    return res


def parse_timing(doc: dict, log_path: Path, rc=0, params=None) -> dict:
    """Main-compute time in seconds from the benchmark's OWN timer line, plus any
    secondary timers (reported separately, never summed).

    Never falls back to wall time: a nonzero exit code, a missing or ambiguous
    line, a non-finite value or an unknown unit raises InputError."""
    if rc != 0:
        raise InputError(f"run exited {rc}; timer output of a failed run is not used")
    if doc["timing"].get("kind") == "none":
        raise InputError("NEEDS_TIMING_SUPPORT: " + str(doc["timing"].get("reason", "no usable native timer")))
    if not Path(log_path).is_file():
        raise InputError(f"log {log_path} missing")
    lines = Path(log_path).read_text(errors="replace").splitlines()
    main = _read_timer(doc["timing"], lines, "timing", params)
    res = dict(main)
    res["main_compute_s"] = main["seconds"]
    sec = {}
    for s in doc["timing"].get("secondary") or []:
        try:
            r = _read_timer(s, lines, f"secondary timer {s['name']}")
            sec[s["name"]] = {"seconds": r["seconds"], "raw_text": r["raw_text"], "unit": r["unit"],
                              "scope": s["scope"], "line": r["line"]}
        except InputError as ex:
            sec[s["name"]] = {"seconds": None, "error": str(ex), "scope": s["scope"]}
    if sec:
        res["secondary_timers"] = sec
    return res


def extract(doc: dict, log_path: Path, inp=None) -> dict:
    if not Path(log_path).is_file():
        raise InputError(f"log {log_path} missing")
    lines = Path(log_path).read_text(errors="replace").splitlines()
    out = {}
    for q in quantities(doc, inp):
        rule = q.get("compare", {}).get("rule")
        ms = _matches(q["regex"], lines, q.get("section_start"))
        if rule in ("present", "absent"):
            out[q["name"]] = {"present": bool(ms), "count": len(ms)}
            continue
        try:
            m = _pick(ms, q.get("select", "last"), f"baseline quantity {q['name']}")
            raw = m.group("value")
            out[q["name"]] = {"value": _finite(raw, f"baseline quantity {q['name']}"), "raw": raw, "line": m.group(0).strip()}
        except InputError as ex:
            out[q["name"]] = {"value": None, "error": str(ex)}
    return out


def role_of(q: dict) -> str:
    """required (default): a quantity the acceptance of this input depends on;
    diagnostic: kept for information -- may stay `record` without blocking acceptance."""
    return q.get("role", "required")


def compare(doc: dict, baseline: dict, current: dict, inp=None) -> dict:
    """Apply each quantity's rule. Returns {ok, complete, verified, record_only, verdict, checks, ...}:
    ok           no rule failed (a `record` quantity only has to be present and finite; a
                 missing/non-finite DIAGNOSTIC record is noted, not a failure);
    complete     every REQUIRED quantity has a verifying (non-record) rule -- the required
                 science comparison is ready; `required_pending` lists the ones still `record`;
    verified     ok AND complete AND at least one required quantity was actually verified;
    record_only  every rule is `record` -> nothing was verified;
    verdict      FAIL (a rule failed) / INCOMPLETE (nothing failed, but a required comparison is
                 not ready) / PASS (verified). Only PASS means the science result was compared.
    Rules passing on configuration quantities (iterations, DOFs, step counts, markers) never
    stand in for a pending required result."""
    checks, ok, n_verifying, n_req_verified = [], True, 0, 0
    required_pending, diagnostic_recorded, diagnostic_missing = [], [], []
    for q in quantities(doc, inp):
        n = q["name"]; cmp = q.get("compare", {}); rule = cmp["rule"]; role = role_of(q)
        b = baseline.get(n, {}); c = current.get(n, {})
        rec = {"name": n, "rule": rule, "role": role}
        if rule in VERIFYING_RULES:
            n_verifying += 1
            if role == "required":
                n_req_verified += 1
        elif role == "required":
            required_pending.append(n)
        else:
            diagnostic_recorded.append(n)
        if rule == "present":
            good = bool(c.get("present")); rec.update({"present": bool(c.get("present"))})
        elif rule == "absent":
            good = not c.get("present"); rec.update({"present": bool(c.get("present"))})
        elif rule == "record":
            present = c.get("value") is not None
            rec.update({"value": c.get("value"), "baseline": b.get("value"),
                        "note": ("recorded only, no tolerance defined yet (NEEDS_VALIDATION: required result)"
                                 if role == "required" else "diagnostic, recorded only")})
            if not present:
                rec["error"] = c.get("error", "missing")
                if role != "required":
                    diagnostic_missing.append(n)
            good = present or role != "required"     # a required result must at least be there and finite
        else:
            cv = c.get("value")
            if cv is None:
                good = False; rec["error"] = c.get("error", "missing")
            elif rule in ("abs_lt", "ge", "le"):
                thr = float(cmp["value"])
                good = {"abs_lt": abs(cv) < thr, "ge": cv >= thr, "le": cv <= thr}[rule]
                rec.update({"value": cv, "threshold": thr})
            else:
                bv = b.get("value")
                if bv is None:
                    good = False; rec["error"] = "baseline has no value for this quantity"
                elif rule == "exact":
                    good = (cv == bv); rec.update({"value": cv, "baseline": bv})
                elif rule == "rel":
                    tol = float(cmp["tol"]); rel = abs(cv - bv) / max(abs(bv), 1e-300)
                    good = rel <= tol; rec.update({"value": cv, "baseline": bv, "rel_err": rel, "tol": tol})
                elif rule == "abs":
                    tol = float(cmp["tol"]); d = abs(cv - bv)
                    good = d <= tol; rec.update({"value": cv, "baseline": bv, "abs_err": d, "tol": tol})
                else:
                    good = False; rec["error"] = f"unknown rule {rule}"
        rec["ok"] = good
        ok = ok and good
        checks.append(rec)
    complete = not required_pending
    verified = ok and complete and n_req_verified > 0
    verdict = "FAIL" if not ok else ("PASS" if verified else "INCOMPLETE")
    return {"ok": ok, "complete": complete, "verified": verified, "verdict": verdict,
            "record_only": n_verifying == 0, "verifying_rules": n_verifying,
            "required_verified_rules": n_req_verified, "required_pending": required_pending,
            "diagnostic_recorded": diagnostic_recorded, "diagnostic_missing": diagnostic_missing,
            "failed": [c["name"] for c in checks if not c["ok"]], "checks": checks}


def workload_identity(doc: dict, inp: dict) -> dict:
    """What a baseline is a baseline OF: the registry entry's parameters, arguments, env and the
    sha256 of the input files it names. Deliberately excludes the binary, the git revision and
    the build fingerprint: an optimized build is compared against the baseline of the SAME
    workload, and a different code identity is exactly what such a comparison is for."""
    bdir = Path(doc["_path"]).parent
    files = {}
    for f in inp.get("files", []) or []:
        files[str(f)] = sha256_file(bdir / f)
    wl = {"input_id": inp["id"], "params": inp.get("params") or {}, "args": [str(a) for a in inp.get("args", []) or []],
          "env": {str(k): str(v) for k, v in (inp.get("env") or {}).items()}, "files_sha256": files}
    if inp.get("input_form") == "compile-time":
        wl["params"] = dict(wl["params"], _build_config=inp.get("build_config") or {})   # what was fixed at build time is part of the workload
    return wl


WORKLOAD_KEYS = ("input_id", "params", "args", "env", "files_sha256")


def workload_complete(wl) -> bool:
    """A workload identity is usable only when every key is present with the right shape and
    no input-file hash is missing (a None sha256 = the file could not be read)."""
    if not isinstance(wl, dict):
        return False
    if any(k not in wl for k in WORKLOAD_KEYS):
        return False
    if not isinstance(wl["input_id"], str) or not wl["input_id"]:
        return False
    if not isinstance(wl["params"], dict) or not isinstance(wl["args"], list) or not isinstance(wl["env"], dict):
        return False
    if not isinstance(wl["files_sha256"], dict) or any(v is None for v in wl["files_sha256"].values()):
        return False
    return True


def workload_mismatch(baseline_wl: dict, current_wl: dict) -> list:
    """The keys of the workload identity that differ (empty list = same workload)."""
    diff = []
    for k in WORKLOAD_KEYS:
        if baseline_wl.get(k) != current_wl.get(k):
            diff.append(k)
    return diff


def comparability(bdoc: dict, doc: dict, inp) -> dict:
    """Whether baseline and candidate are known to be the SAME workload. Three outcomes:
      established   -- the baseline carries a complete workload identity equal to the candidate
                       input's registry identity, or an upstream reference explicitly bound to
                       this input with recorded evidence (reference_binding);
      contradicted  -- identities exist and differ (refused, exit 2);
      not-established -- identity missing / incomplete on either side: the comparison may be
                       computed and shown, but it can never be a PASS (verdict INCOMPLETE).
    A candidate's identity is never copied into an old baseline; migration is explicit
    (`migrate-baseline`) and records its evidence."""
    if inp is None:
        return {"status": "not-established", "reason": "candidate input unknown (no --input and the baseline names none)"}
    cur = workload_identity(doc, inp)
    if not workload_complete(cur):
        return {"status": "not-established", "reason": "candidate workload identity incomplete (registry entry / input files unreadable)"}
    wl = bdoc.get("workload")
    if isinstance(wl, dict) and wl:
        if not workload_complete(wl):
            return {"status": "not-established", "reason": "baseline workload identity incomplete (missing or empty keys)"}
        diff = workload_mismatch(wl, cur)
        if diff:
            return {"status": "contradicted", "reason": "differs in: " + ", ".join(diff), "diff": diff}
        return {"status": "established", "reason": "baseline workload identity equals the candidate's registry identity" +
                (" (migrated: " + str(bdoc["workload_migration"].get("source")) + ")" if isinstance(bdoc.get("workload_migration"), dict) else "")}
    rb = bdoc.get("reference_binding")
    if isinstance(rb, dict):
        if rb.get("bound_input") != inp["id"]:
            return {"status": "contradicted", "reason": f"upstream reference is bound to input '{rb.get('bound_input')}', not '{inp['id']}'"}
        if not rb.get("evidence") or not rb.get("source") or not rb.get("adapted_by"):
            return {"status": "not-established", "reason": "reference_binding lacks evidence/source/adapted_by"}
        return {"status": "established", "reason": f"upstream reference bound to '{inp['id']}' (source: {rb['source']}; evidence: {rb['evidence']})", "kind": "upstream-reference"}
    return {"status": "not-established", "reason": "baseline carries no workload identity (recorded before round 3); read/display only -- migrate it with evidence (migrate-baseline) to compare formally"}


def _registry_entry_at(doc: dict, inp: dict, commit: str):
    """The workload identity of this input as recorded in inputs.yaml at a git commit (None if absent)."""
    root = repo_root(Path(doc["_path"]).parent); rel = str(Path(doc["_path"]).resolve().relative_to(root))
    txt = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{rel}"], capture_output=True, text=True, check=True).stdout
    then = yaml.safe_load(txt); tinp = next((i for i in then.get("inputs", []) if i.get("id") == inp["id"]), None)
    return None if tinp is None else workload_identity(doc, tinp)


def migrate_baseline(doc: dict, bdoc: dict, inp: dict, evidence_path: Path, note: str, registry_commit=None, manual_basis=None) -> dict:
    """Attach a workload identity to an old baseline from EVIDENCE, never from the candidate:
    the measurement.json the baseline was written from must name the same benchmark and
    input id, its recorded inputs.yaml sha256 must equal the current registry file's (so the
    registry entry the identity is built from is the one that produced the run), and for a
    binary entry the recorded command must end with the registry args. On success the
    identity is built from the CURRENT registry entry and the migration is recorded
    (source, evidence, note, time); the caller writes it to a NEW file."""
    ev = json.loads(Path(evidence_path).read_text())
    problems = []
    if ev.get("benchmark") != doc["benchmark"]:
        problems.append(f"evidence benchmark '{ev.get('benchmark')}' != '{doc['benchmark']}'")
    if ev.get("input_id") != inp["id"]:
        problems.append(f"evidence input_id '{ev.get('input_id')}' != '{inp['id']}'")
    if bdoc.get("input_id") and bdoc["input_id"] != inp["id"]:
        problems.append(f"baseline input_id '{bdoc['input_id']}' != '{inp['id']}'")
    registry_basis = "evidence inputs_yaml_sha256 equals the current inputs.yaml"; kind = "automatic"
    if ev.get("inputs_yaml_sha256") != doc["_sha256"]:
        # The registry file changed since the run (roles, comments, other inputs...). The entry the run
        # used is recoverable from git ONLY when the run recorded a clean commit; otherwise the operator
        # must name a registry commit to check against AND state the basis that bridges the gap between
        # the run-time (dirty) file and that commit -- both are recorded verbatim, nothing is assumed.
        g = ev.get("git") or {}
        commit = g["head"] if (g.get("head") and g.get("dirty") is False) else registry_commit
        if commit is None:
            problems.append("evidence inputs_yaml_sha256 differs from the current inputs.yaml and the run's git tree was dirty/unknown; pass --registry-commit <commit> (entry to check against) and --manual-basis <text>")
        else:
            try:
                then_wl = _registry_entry_at(doc, inp, commit); now_wl = workload_identity(doc, inp)
                if then_wl is None:
                    problems.append(f"input '{inp['id']}' did not exist in inputs.yaml at commit {commit[:12]}")
                else:
                    diff = workload_mismatch(then_wl, now_wl)
                    if diff:
                        problems.append(f"registry entry changed since commit {commit[:12]} in: {', '.join(diff)}")
                    elif g.get("dirty") is False and commit == g["head"]:
                        registry_basis = f"inputs.yaml changed since the run, but the entry '{inp['id']}' at the run's clean commit {commit[:12]} has the same params/args/env/files as now (git show)"
                    else:
                        if not manual_basis or not manual_basis.strip():
                            problems.append("the run's git tree was dirty: --manual-basis <text> stating what links the run-time registry file to the named commit is required")
                        else:
                            kind = "manual"
                            registry_basis = (f"run-time inputs.yaml sha256 {ev.get('inputs_yaml_sha256')} differs from now (tree dirty at run time); the entry at the "
                                              f"operator-named commit {commit[:12]} equals the current one (git show); the operator's basis for the run-time file: " + manual_basis.strip())
            except (subprocess.CalledProcessError, InputError, yaml.YAMLError) as ex:
                problems.append(f"could not recover the registry entry at commit {str(commit)[:12]}: {ex}")
    # what the run itself printed about its input (recorded for the reviewer; not a pass criterion)
    log_lines = []
    try:
        for ln in Path(bdoc.get("log", "")).read_text(errors="replace").splitlines()[:80]:
            if (f"input={inp['id']}" in ln) or (inp.get("args") and all(str(a) in ln for a in inp["args"])):
                log_lines.append(ln.strip()[:300])
    except OSError:
        pass
    logs = [r.get("log") for r in ev.get("runs", [])]
    if bdoc.get("log") and bdoc["log"] not in logs:
        problems.append("baseline log is not one of the evidence measurement's run logs")
    if doc["entry"]["kind"] == "binary":
        args = [str(a) for a in inp.get("args", [])]
        cmd = [str(c) for c in ev.get("command", [])]
        if args and cmd[-len(args):] != args:
            problems.append("evidence command line does not end with the registry args")
    else:
        sel = ev.get("selector") or {}
        if sel.get(doc.get("selector")) != inp["id"]:
            problems.append("evidence selector does not name this input id")
    if problems:
        raise InputError("migration refused: " + "; ".join(problems))
    out = dict(bdoc)
    out["workload"] = workload_identity(doc, inp)
    out["workload_migration"] = {"kind": kind, "source": str(evidence_path), "evidence": {"benchmark": ev.get("benchmark"), "input_id": ev.get("input_id"),
                                 "inputs_yaml_sha256": ev.get("inputs_yaml_sha256"), "command": ev.get("command"), "selector": ev.get("selector"),
                                 "started_utc": ev.get("started_utc"), "host": ev.get("host"), "git": ev.get("git"), "entry_sha256": ev.get("entry_sha256"),
                                 "registry_commit_checked": registry_commit, "run_log_lines": log_lines},
                                 "note": note, "migrated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                 "basis": "identity built from the current registry entry; the evidence measurement recorded the same benchmark, input id and command/selector; registry: " + registry_basis}
    return out


def native_check(doc, current: dict, inp=None) -> dict:
    """Only the rules that need no baseline (present/absent/abs_lt/ge/le): the
    benchmark's own pass criteria as far as they are expressed in inputs.yaml."""
    qs = [q for q in quantities(doc, inp) if q.get("compare", {}).get("rule") in NATIVE_RULES]
    if not qs:
        return {"status": "NONE", "checks": []}
    res = compare({"baseline": {"quantities": qs}}, {}, current, None)
    return {"status": "PASS" if res["ok"] else "FAIL", "checks": res["checks"]}


# ----------------------------------------------------------------------------- measure
def sha256_file(p: Path):
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except OSError:
        return None


def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        return []


def git_head(root: Path):
    try:
        h = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        d = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True).stdout
        return {"head": h, "dirty": bool(d.strip())}
    except Exception:
        return {}


def build_command(doc, inp, root: Path, bench_dir: Path, gpus: int):
    e = doc["entry"]
    env = dict(os.environ)
    if inp.get("materialized", True) is False:
        raise InputError(f"input '{inp['id']}' is a compile-time configuration that is not materialized in this worktree "
                         f"(build_config {inp.get('build_config')}); registered, NOT runnable -- nothing else is substituted")
    if inp.get("binary"):        # a materialized compile-time input carries its own binary (e.g. another NPB class)
        exe = root / inp["binary"]
        if not exe.is_file() or not os.access(exe, os.X_OK):
            raise InputError(f"binary {exe} of input '{inp['id']}' not built")
        cmd = [str(exe)] + [str(a) for a in inp.get("args", [])]
        env["CUDA_VISIBLE_DEVICES"] = env.get("HPCPERF_CUDA_VISIBLE_DEVICE", "0")
        for k, v in (inp.get("env") or {}).items():
            env[str(k)] = str(v)
        return cmd, env, exe
    if e["kind"] == "binary":
        exe = root / e["path"]
        if not exe.is_file() or not os.access(exe, os.X_OK):
            raise InputError(f"binary {exe} not built")
        cmd = [str(exe)] + [str(a) for a in inp.get("args", [])]
        env["CUDA_VISIBLE_DEVICES"] = env.get("HPCPERF_CUDA_VISIBLE_DEVICE", "0")
        for k, v in (inp.get("env") or {}).items():
            env[str(k)] = str(v)
        return cmd, env, exe
    rs = root / e["path"]
    if not rs.is_file():
        raise InputError(f"{rs} missing")
    cmd = ["bash", str(rs)] + ([e["backend_arg"]] if e.get("backend_arg") else [])
    env[doc["selector"]] = inp["id"]
    env["HPCPERF_GPUS"] = str(gpus)
    for k in ("HPCPERF_NP", "HPCPERF_SCALE_MODE"):
        env.pop(k, None)
    for k, v in (inp.get("env") or {}).items():          # documented per-input knobs only
        env[str(k)] = str(v)
    return cmd, env, rs


def run_once(cmd, env, cwd: Path, log: Path, timeout: int):
    t0 = time.monotonic()
    with open(log, "w") as fh:
        try:
            p = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = 124
    return rc, time.monotonic() - t0


def stats(v, resolution=None):
    if not v:
        return None
    med = statistics.median(v)
    spread = (max(v) - min(v)) / med if med > 0 else None
    return {"n": len(v), "median": med, "min": min(v), "max": max(v),
            "mad": statistics.median([abs(x - med) for x in v]),
            "spread_rel": spread, "spread_formula": "(max - min) / median over the measured runs",
            "stable_rule": f"n >= 3 and spread_rel <= {STABLE_SPREAD}",
            "print_resolution_s": resolution,
            "spread_below_print_resolution": (resolution is not None and (max(v) - min(v)) <= resolution),
            "values": v}


def summarize(doc, inp, runs, reps):
    measured = [r for r in runs if r["measured"]]
    good = [r for r in measured if r["exit_code"] == 0]
    mc = [r["main_compute_s"] for r in good if r.get("main_compute_s") is not None]
    e2e = [r["e2e_s"] for r in good]
    res = None
    for r in good:
        if r.get("timing") and r["timing"].get("print_resolution_s"):
            res = r["timing"]["print_resolution_s"]; break
    s = {"run_completed": len(good) == len(measured) and len(measured) == reps,
         "timing_ok": len(mc) == len(measured) and len(measured) == reps,
         "timing_status": ("NEEDS_TIMING_SUPPORT" if doc["timing"].get("kind") == "none"
                           else ("NATIVE" if len(mc) == len(measured) and len(measured) == reps else "FAILED")),
         "main_compute_s": stats(mc, res), "e2e_s": stats(e2e)}
    sec = {}
    for r in good:
        for k, v in (r.get("timing", {}).get("secondary_timers") or {}).items():
            if v.get("seconds") is not None:
                sec.setdefault(k, {"scope": v["scope"], "values": []})["values"].append(v["seconds"])
    for k in sec:
        sec[k].update({kk: vv for kk, vv in stats(sec[k]["values"]).items() if kk != "values"})
    s["secondary_timers_s"] = sec or None
    s["compute_ge_1s"] = bool(mc) and s["main_compute_s"]["median"] >= 1.0
    s["stable"] = (bool(mc) and len(mc) >= 3 and s["main_compute_s"]["spread_rel"] is not None
                   and s["main_compute_s"]["spread_rel"] <= STABLE_SPREAD)
    # native (baseline-free) checks on every good run
    nat = [native_check(doc, r["baseline_quantities"], inp) for r in good if r.get("baseline_quantities")]
    if not nat or all(n["status"] == "NONE" for n in nat):
        s["native_check"] = "NONE"
    else:
        s["native_check"] = "PASS" if all(n["status"] == "PASS" for n in nat) and len(nat) == len(measured) else "FAIL"
    s["native_checks"] = nat
    qs = quantities(doc, inp)
    rec = [q["name"] for q in qs if q.get("compare", {}).get("rule") == "record" and role_of(q) == "required"]
    diag = [q["name"] for q in qs if q.get("compare", {}).get("rule") == "record" and role_of(q) != "required"]
    ver = [q["name"] for q in qs if q.get("compare", {}).get("rule") in VERIFYING_RULES and role_of(q) == "required"]
    # READY: every required quantity has a verifying rule (diagnostic records do not count against it);
    # PARTIAL: a required quantity is still record-only; NONE: no required quantity is verified at all.
    s["comparison_rules"] = "READY" if (ver and not rec) else ("PARTIAL" if ver else "NONE")
    s["needs_validation"] = rec
    s["diagnostic_recorded"] = diag
    return s, good


def measure(doc, inp, root: Path, bench_dir: Path, out: Path, warmup: int, reps: int, timeout: int, gpus: int):
    out.mkdir(parents=True, exist_ok=True)
    cmd, env, entry_path = build_command(doc, inp, root, bench_dir, gpus)
    e2e_boundary = ("process wall of the benchmark binary (fork/exec to exit; includes CUDA context creation)"
                    if doc["entry"]["kind"] == "binary" else
                    "wrapper wall of run.sh (env sourcing, launcher, mpirun, binding audit, application)")
    meta = {"schema": "hpcperf-inputs-measurement-2", "benchmark": doc["benchmark"], "level": doc["level"],
            "input_id": inp["id"], "inputs_yaml_sha256": doc["_sha256"], "entry": doc["entry"],
            "entry_sha256": sha256_file(entry_path), "command": cmd, "gpus": gpus,
            "selector": {doc.get("selector"): inp["id"]} if doc.get("selector") else None,
            "host": socket.gethostname(), "gpu": gpu_info(), "git": git_head(root),
            "warmup_runs": warmup, "measured_runs": reps, "timeout_s": timeout,
            "e2e_boundary": e2e_boundary, "timing_scope": doc["timing"].get("scope") or doc["timing"].get("reason"),
            "secondary_timer_scopes": {s["name"]: s["scope"] for s in (doc["timing"].get("secondary") or [])},
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": []}
    for k in range(warmup + reps):
        label = f"warmup{k}" if k < warmup else f"rep{k - warmup + 1}"
        rdir = out / label
        if rdir.exists():
            shutil.rmtree(rdir)               # never read a previous run's output
        rdir.mkdir(parents=True)
        renv = dict(env)
        if doc["level"] == 3:
            renv["HPCPERF_L3_RUN_SUBDIR"] = f"run.inputs.{inp['id']}.{label}"   # isolate app-written results per run
        log = rdir / "stdout.log"
        t_start = time.time()
        rc, wall = run_once(cmd, renv, rdir, log, timeout)
        rec = {"label": label, "exit_code": rc, "e2e_s": round(wall, 4), "log": str(log), "measured": k >= warmup,
               "log_written_after_start": (log.exists() and log.stat().st_mtime >= t_start - 1)}
        try:
            rec["timing"] = parse_timing(doc, log, rc, inp.get("params"))
            rec["main_compute_s"] = rec["timing"]["main_compute_s"]
        except InputError as ex:
            rec["timing_error"] = str(ex); rec["main_compute_s"] = None
        rec["baseline_quantities"] = extract(doc, log, inp) if rc == 0 and log.exists() else None
        (rdir / "result.json").write_text(json.dumps(rec, indent=2))
        meta["runs"].append(rec)
        print(f"[{doc['benchmark']}/{inp['id']}] {label}: rc={rc} e2e={wall:.3f}s main_compute="
              f"{rec['main_compute_s'] if rec['main_compute_s'] is None else round(rec['main_compute_s'], 4)}"
              + (f" ({rec['timing_error']})" if 'timing_error' in rec else ""), flush=True)
    summary, good = summarize(doc, inp, meta["runs"], reps)
    # the working baseline comes from the first measured run that exited 0 AND passed the
    # benchmark's own baseline-free checks (a run that prints FAIL and exits 0 is never a baseline)
    base = next((r for r in good if r["baseline_quantities"]
                 and native_check(doc, r["baseline_quantities"], inp)["status"] != "FAIL"), None)
    if base:
        bfile = out / "baseline.json"
        bfile.write_text(json.dumps({"schema": "hpcperf-inputs-baseline-1", "benchmark": doc["benchmark"],
                                     "input_id": inp["id"], "from_run": base["label"], "log": base["log"],
                                     "log_sha256": sha256_file(base["log"]),
                                     "method": doc["baseline"].get("method"),
                                     "reference": (inp.get("baseline") or {}).get("reference", doc["baseline"].get("reference")),
                                     "workload": workload_identity(doc, inp),
                                     "code_identity": {"entry_sha256": meta.get("entry_sha256"), "git": meta.get("git"),
                                                       "note": "informational only -- compare never refuses on code identity"},
                                     "quantity_rules": quantities(doc, inp), "quantities": base["baseline_quantities"]}, indent=2))
        summary["baseline_saved"] = True; summary["baseline_file"] = str(bfile)
        # independent runs only: the baseline run is never compared with itself (that would be
        # evidence of nothing); `independent_runs_compared` says how many runs the verdict rests on
        others = [r for r in good if r["baseline_quantities"] and r["label"] != base["label"]]
        cmps = [dict(compare(doc, base["baseline_quantities"], r["baseline_quantities"], inp), run=r["label"]) for r in others]
        summary["baseline_from_run"] = base["label"]
        summary["independent_runs_compared"] = len(cmps)
        summary["baseline_self_consistent"] = bool(cmps) and all(c["ok"] for c in cmps)
        # the verdict of the self-comparison: PASS only when every required quantity was verified
        # in at least one independent run; NONE when no independent run exists
        summary["baseline_verdict"] = ("NONE" if not cmps else "FAIL" if not all(c["ok"] for c in cmps)
                                       else ("PASS" if all(c["verified"] for c in cmps) else "INCOMPLETE"))
        summary["baseline_checks"] = cmps
    else:
        summary["baseline_saved"] = False; summary["baseline_file"] = None
        summary["baseline_self_consistent"] = False; summary["baseline_verdict"] = "NONE"
    meta["summary"] = summary
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "measurement.json").write_text(json.dumps(meta, indent=2))
    return meta


def status_line(s: dict) -> str:
    keys = ("run_completed", "timing_ok", "timing_status", "native_check", "baseline_saved", "comparison_rules",
            "baseline_verdict", "baseline_from_run", "independent_runs_compared", "compute_ge_1s", "stable", "baseline_self_consistent")
    parts = [f"{k}={s.get(k)}" for k in keys]
    if s.get("needs_validation"):
        parts.append("NEEDS_VALIDATION=" + ",".join(s["needs_validation"]))
    if s.get("diagnostic_recorded"):
        parts.append("diagnostic_recorded=" + ",".join(s["diagnostic_recorded"]))
    return " ".join(parts)


# ----------------------------------------------------------------------------- cli
def main(argv=None):
    ap = argparse.ArgumentParser(prog="hpcperf_inputs.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("validate", "list"):
        s = sub.add_parser(c); s.add_argument("bench_dir")
    for c in ("show", "args", "shell-env"):
        s = sub.add_parser(c); s.add_argument("bench_dir"); s.add_argument("input_id")
    s = sub.add_parser("param"); s.add_argument("bench_dir"); s.add_argument("input_id"); s.add_argument("key")
    s = sub.add_parser("parse-timing"); s.add_argument("bench_dir"); s.add_argument("log"); s.add_argument("--rc", type=int, default=0); s.add_argument("--input")
    s = sub.add_parser("extract"); s.add_argument("bench_dir"); s.add_argument("log"); s.add_argument("--input")
    s = sub.add_parser("compare"); s.add_argument("bench_dir"); s.add_argument("baseline"); s.add_argument("log"); s.add_argument("--input"); s.add_argument("--rc", type=int, default=0)
    s = sub.add_parser("status"); s.add_argument("bench_dir"); s.add_argument("measurement")
    s = sub.add_parser("migrate-baseline"); s.add_argument("bench_dir"); s.add_argument("baseline"); s.add_argument("--input", required=True)
    s.add_argument("--evidence", required=True, help="the measurement.json the baseline was written from"); s.add_argument("--note", default="")
    s.add_argument("--out", help="destination (default: <baseline>.workload-migrated.json; never overwrites)")
    s.add_argument("--registry-commit", help="when the run's tree was dirty: the commit whose registry entry is checked against the current one")
    s.add_argument("--manual-basis", help="required with --registry-commit for a dirty run: the operator's stated basis linking the run-time registry file to that commit (recorded verbatim)")
    s = sub.add_parser("measure"); s.add_argument("bench_dir"); s.add_argument("input_id"); s.add_argument("--out", required=True)
    s.add_argument("--warmup", type=int, default=1); s.add_argument("--reps", type=int, default=3)
    s.add_argument("--timeout", type=int, default=1800); s.add_argument("--gpus", type=int, default=1)
    a = ap.parse_args(argv)
    bench_dir = Path(a.bench_dir)
    try:
        doc = load(bench_dir)
        errs = validate(doc)
        if a.cmd == "validate":
            for e in errs:
                print(f"inputs.yaml: {e}")
            print(f"{doc['_path']}: {'OK' if not errs else str(len(errs)) + ' error(s)'} ({len(doc['inputs'])} inputs)")
            return 1 if errs else 0
        if errs:
            raise InputError(f"{doc['_path']} is invalid: " + "; ".join(errs))
        if a.cmd == "list":
            for inp in doc["inputs"]:
                d = " (default)" if inp["id"] == doc["default_input"] else ""
                print(f"{inp['id']:32s} {inp['variant']:20s} {inp['source']['kind']:24s} {inp['case']}{d}")
            return 0
        if a.cmd == "show":
            print(json.dumps(get_input(doc, a.input_id), indent=2)); return 0
        if a.cmd == "args":
            inp = get_input(doc, a.input_id)
            for x in inp.get("args", []):
                print(x)
            return 0
        if a.cmd == "shell-env":
            # for run.sh selectors: the input's environment knobs (E<TAB>KEY<TAB>VALUE) and extra
            # command-line arguments (A<TAB>ARG), one per line, for hpcperf_apply_input
            inp = get_input(doc, a.input_id)
            for k, v in (inp.get("env") or {}).items():
                print(f"E\t{k}\t{v}")
            for x in inp.get("args", []) or []:
                print(f"A\t{x}")
            return 0
        if a.cmd == "param":
            inp = get_input(doc, a.input_id)
            if a.key not in (inp.get("params") or {}):
                raise InputError(f"input '{a.input_id}' has no parameter '{a.key}'")
            print(inp["params"][a.key]); return 0
        if a.cmd == "parse-timing":
            params = get_input(doc, a.input).get("params") if a.input else None
            print(json.dumps(parse_timing(doc, Path(a.log), a.rc, params), indent=2)); return 0
        if a.cmd == "extract":
            inp = get_input(doc, a.input) if a.input else None
            print(json.dumps(extract(doc, Path(a.log), inp), indent=2)); return 0
        if a.cmd == "compare":
            bdoc = json.loads(Path(a.baseline).read_text())
            bid = bdoc.get("input_id")
            if a.input and bid and bid != a.input:
                sys.stderr.write(f"hpcperf_inputs: baseline belongs to input '{bid}', not '{a.input}' -- refusing to compare\n"); return 2
            if bdoc.get("benchmark") and bdoc["benchmark"] != doc["benchmark"]:
                sys.stderr.write(f"hpcperf_inputs: baseline belongs to benchmark '{bdoc['benchmark']}', not '{doc['benchmark']}'\n"); return 2
            blog = bdoc.get("log")
            if blog and Path(blog).exists() and Path(a.log).exists() and os.path.realpath(blog) == os.path.realpath(a.log):
                sys.stderr.write("hpcperf_inputs: baseline and candidate are the same output file -- refusing to compare\n"); return 2
            if a.rc != 0:
                print(json.dumps({"ok": False, "complete": False, "verified": False, "verdict": "FAIL",
                                  "error": f"candidate run exited {a.rc}; its output is not compared"}, indent=2)); return 1
            inp = get_input(doc, a.input or bid) if (a.input or bid) else None
            comp = comparability(bdoc, doc, inp)
            if comp["status"] == "contradicted":
                sys.stderr.write(f"hpcperf_inputs: baseline '{bid}' is not the candidate's workload ({comp['reason']}) -- refusing to compare\n"); return 2
            res = compare(doc, bdoc["quantities"], extract(doc, Path(a.log), inp), inp)
            res["baseline_input_id"] = bid
            res["workload_status"] = comp["status"]; res["workload_reason"] = comp["reason"]
            if comp.get("kind"):
                res["comparison_kind"] = comp["kind"]
            if comp["status"] != "established":
                # the numbers are shown for reading/archiving, but nothing is verified: a comparison
                # whose two sides are not known to be the same workload can never be a PASS
                res["complete"] = False; res["verified"] = False
                if res["ok"]:
                    res["verdict"] = "INCOMPLETE"
                res["required_pending"] = sorted(set(res["required_pending"]) | {"(workload identity not established)"})
            print(json.dumps(res, indent=2))
            # acceptance: 0 only when verified (no failure, every required quantity compared AND the
            # workload identity of both sides established); 1 = a rule or the run failed (never
            # downgraded); 3 = nothing failed but the comparison is not complete (required quantity
            # still record, or identity not established); 2 = identity contradictions above.
            if not res["ok"]:
                return 1
            return 0 if res["verified"] else 3
        if a.cmd == "migrate-baseline":
            bdoc = json.loads(Path(a.baseline).read_text())
            inp = get_input(doc, a.input)
            out = migrate_baseline(doc, bdoc, inp, Path(a.evidence), a.note, a.registry_commit, a.manual_basis)
            dest = Path(a.out) if a.out else Path(a.baseline).with_name(Path(a.baseline).stem + ".workload-migrated.json")
            if dest.exists():
                raise InputError(f"{dest} exists; not overwriting a migration record")
            dest.write_text(json.dumps(out, indent=2))
            print(json.dumps({"migrated_to": str(dest), "workload": out["workload"], "workload_migration": out["workload_migration"]}, indent=2)); return 0
        if a.cmd == "status":
            m = json.loads(Path(a.measurement).read_text())
            inp = get_input(doc, m["input_id"])
            s, _ = summarize(doc, inp, m["runs"], m.get("measured_runs", len([r for r in m["runs"] if r["measured"]])))
            old = m.get("summary", {})
            for k in ("baseline_saved", "baseline_file", "baseline_self_consistent", "baseline_verdict", "baseline_from_run", "independent_runs_compared"):
                s[k] = old.get(k, (old.get("baseline_file") is not None) if k == "baseline_saved" else None)
            # older files: recompute the self-comparison over the runs OTHER than the baseline run
            bf = old.get("baseline_file")
            if bf and Path(bf).is_file():
                b = json.loads(Path(bf).read_text()); base_label = b.get("from_run")
                others = [r for r in m["runs"] if r["measured"] and r["exit_code"] == 0 and r.get("baseline_quantities") and r["label"] != base_label]
                cmps = [compare(doc, b["quantities"], r["baseline_quantities"], inp) for r in others]
                s["baseline_from_run"] = base_label; s["independent_runs_compared"] = len(cmps)
                s["baseline_self_consistent"] = bool(cmps) and all(c["ok"] for c in cmps)
                s["baseline_verdict"] = ("NONE" if not cmps else "FAIL" if not all(c["ok"] for c in cmps)
                                         else ("PASS" if all(c["verified"] for c in cmps) else "INCOMPLETE"))
            if s.get("baseline_verdict") is None and old.get("baseline_checks"):
                s["baseline_verdict"] = ("FAIL" if not all(c["ok"] for c in old["baseline_checks"])
                                         else ("PASS" if s["comparison_rules"] == "READY" and all(c.get("ok") for c in old["baseline_checks"]) else "INCOMPLETE"))
            print(status_line(s)); return 0
        if a.cmd == "measure":
            inp = get_input(doc, a.input_id)
            root = repo_root(bench_dir)
            meta = measure(doc, inp, root, bench_dir, Path(a.out), a.warmup, a.reps, a.timeout, a.gpus)
            s = meta["summary"]
            print(status_line(s))
            return 0 if s["run_completed"] and (s["timing_ok"] or s["timing_status"] == "NEEDS_TIMING_SUPPORT") else 1
    except InputError as ex:
        sys.stderr.write(f"hpcperf_inputs: {ex}\n")
        return 2 if a.cmd in ("args", "param", "show", "shell-env") else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
