#!/usr/bin/env python3
"""hpcperf_inputs.py -- minimal common input registry for a benchmark directory.

A benchmark that supports several inputs carries an `inputs.yaml` (schema
hpcperf-inputs-1) next to its run.sh / CMakeLists.txt. The file is the single
source of truth for:

  * the stable input ids and what each one IS (case, parameters, upstream origin,
    build-time / run-time configuration, seed, validated backends);
  * how the benchmark's own timer is read (which line, which field, which unit,
    what the timed region covers) -- the tool never substitutes end-to-end wall
    time for a missing or malformed timer value;
  * which scientific quantities form the baseline output and how a later run is
    compared with them.

Sub-commands (all read-only except `measure`, which writes into --out):

  validate <bench_dir>                     schema / consistency check (exit 1 on error)
  list     <bench_dir>                     one line per input id
  show     <bench_dir> <input_id>          the resolved entry as JSON
  args     <bench_dir> <input_id>          the input's command-line arguments, one per line
                                           (run.sh reads these; unknown id -> exit 2)
  param    <bench_dir> <input_id> <key>    one parameter value (exit 2 if unknown)
  parse-timing <bench_dir> <log> [--rc N]  main-compute time from a log (JSON); a nonzero
                                           --rc or a missing/ambiguous timer line -> exit 1
  extract  <bench_dir> <log>               the baseline quantities of a log (JSON)
  compare  <bench_dir> <baseline.json> <log>   compare a log against a stored baseline
  measure  <bench_dir> <input_id> --out DIR [--warmup 1] [--reps 3] [--timeout S] [--gpus 1]
                                           warm-up run + N measured runs, one directory per run,
                                           timing + baseline + summary (median, min, max, MAD)

Exit codes: 0 ok, 1 validation/measurement failure, 2 usage / unknown input.
"""
import argparse, hashlib, json, os, re, shutil, socket, statistics, subprocess, sys, time
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
NOISE = re.compile(r"lua|posix|stack traceback|no file|no field|\[C\]|addto:65")


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


def validate(doc: dict) -> list:
    errs = []
    need = lambda k: errs.append(f"missing top-level key '{k}'") if k not in doc else None
    for k in ("schema", "benchmark", "level", "entry", "default_input", "timing", "baseline", "inputs"):
        need(k)
    if errs:
        return errs
    if doc["schema"] != SCHEMA:
        errs.append(f"schema must be {SCHEMA} (got {doc['schema']})")
    if doc["level"] not in (1, 2, 3):
        errs.append("level must be 1, 2 or 3")
    e = doc["entry"]
    if not isinstance(e, dict) or e.get("kind") not in ENTRY_KINDS or not e.get("path"):
        errs.append(f"entry must be {{kind: run.sh|binary, path: <repo-relative>}}")
    if e.get("kind") == "run.sh" and not doc.get("selector"):
        errs.append("a run.sh entry needs 'selector' (the HPCPERF_<APP>_INPUT variable run.sh honours)")
    t = doc["timing"]
    for k in ("scope", "kind", "unit", "regex", "select"):
        if k not in t:
            errs.append(f"timing.{k} missing")
    if t.get("unit") not in UNIT_TO_S:
        errs.append(f"timing.unit must be one of {sorted(UNIT_TO_S)}")
    if t.get("select") not in SELECT:
        errs.append(f"timing.select must be one of {SELECT}")
    if t.get("kind") not in ("total", "per_iteration", "per_step"):
        errs.append("timing.kind must be total | per_iteration | per_step")
    if t.get("kind") in ("per_iteration", "per_step") and not t.get("work"):
        errs.append("timing.work {key, offset} is required for per_iteration/per_step timers")
    try:
        rx = re.compile(t.get("regex", ""))
        if "value" not in rx.groupindex:
            errs.append("timing.regex needs a named group (?P<value>...)")
    except re.error as ex:
        errs.append(f"timing.regex does not compile: {ex}")
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
    # per-input overrides of the quantity list are validated with the same rules
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
        if e.get("kind") == "binary" and not isinstance(inp.get("args"), list):
            errs.append(f"input '{i}': binary entries need an 'args' list")
        if "backends_validated" not in inp:
            errs.append(f"input '{i}': missing backends_validated (may be an empty list)")
        # files referenced by the input must exist (protected harness side, or frozen tree)
        for f in inp.get("files", []) or []:
            if not (Path(doc["_path"]).parent / f).exists():
                errs.append(f"input '{i}': referenced file '{f}' does not exist under the benchmark directory")
    if doc["default_input"] not in ids:
        errs.append(f"default_input '{doc['default_input']}' is not a registered input id")
    return errs


def get_input(doc: dict, input_id: str) -> dict:
    for inp in doc["inputs"]:
        if inp.get("id") == input_id:
            return inp
    known = ", ".join(i.get("id", "?") for i in doc["inputs"])
    raise InputError(f"unknown input id '{input_id}' for {doc['benchmark']} (registered: {known})")


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


def parse_timing(doc: dict, log_path: Path, rc=0, params=None) -> dict:
    """Main-compute time in seconds from the benchmark's OWN timer line.

    Never falls back to wall time: a nonzero exit code, a missing or ambiguous
    line, a non-finite value or an unknown unit raises InputError."""
    t = doc["timing"]
    if rc != 0:
        raise InputError(f"run exited {rc}; timer output of a failed run is not used")
    if not Path(log_path).is_file():
        raise InputError(f"log {log_path} missing")
    lines = Path(log_path).read_text(errors="replace").splitlines()
    m = _pick(_matches(t["regex"], lines, t.get("section_start")), t["select"], "timing")
    raw = m.group("value")
    try:
        v = float(raw)
    except ValueError:
        raise InputError(f"timing value '{raw}' is not a number")
    if not (v == v) or v in (float("inf"), float("-inf")) or v < 0:
        raise InputError(f"timing value {raw} is not finite/non-negative")
    seconds = v * UNIT_TO_S[t["unit"]]
    res = {"raw_value": v, "unit": t["unit"], "kind": t["kind"], "line": m.group(0).strip(),
           "scope": t["scope"]}
    if t["kind"] in ("per_iteration", "per_step"):
        w = t["work"]
        if params is None or w["key"] not in params:
            raise InputError(f"timing.work key '{w['key']}' not in the input parameters")
        n = int(params[w["key"]]) + int(w.get("offset", 0))
        if n <= 0:
            raise InputError(f"work count {n} <= 0 for key {w['key']}")
        res.update({"per_unit_s": seconds, "work_count": n, "main_compute_s": seconds * n})
    else:
        res["main_compute_s"] = seconds
    return res


def quantities(doc: dict, inp=None):
    """The baseline quantity list: an input may override the benchmark-wide one
    (e.g. a deck with another thermo output style)."""
    if inp and isinstance(inp.get("baseline"), dict) and inp["baseline"].get("quantities"):
        return inp["baseline"]["quantities"]
    return doc["baseline"]["quantities"]


def extract(doc: dict, log_path: Path, inp=None) -> dict:
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
            out[q["name"]] = {"value": float(raw), "raw": raw, "line": m.group(0).strip()}
        except InputError as ex:
            out[q["name"]] = {"value": None, "error": str(ex)}
    return out


def compare(doc: dict, baseline: dict, current: dict, inp=None) -> dict:
    """Apply each quantity's rule. Returns {ok, checks:[...]}."""
    checks, ok = [], True
    for q in quantities(doc, inp):
        n = q["name"]; cmp = q.get("compare", {}); rule = cmp["rule"]
        b = baseline.get(n, {}); c = current.get(n, {})
        rec = {"name": n, "rule": rule}
        if rule == "present":
            good = bool(c.get("present"))
            rec.update({"present": good})
        elif rule == "absent":
            good = not c.get("present")
            rec.update({"present": not good})
        elif rule == "record":
            good = c.get("value") is not None            # recorded for later analysis; tolerance not yet fixed
            rec.update({"value": c.get("value"), "baseline": b.get("value"), "note": "recorded only, no tolerance defined yet"})
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
                    good = False; rec["error"] = "baseline has no value"
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
    return {"ok": ok, "checks": checks}


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
    if e["kind"] == "binary":
        exe = root / e["path"]
        if not exe.is_file() or not os.access(exe, os.X_OK):
            raise InputError(f"binary {exe} not built")
        cmd = [str(exe)] + [str(a) for a in inp.get("args", [])]
        env["CUDA_VISIBLE_DEVICES"] = env.get("HPCPERF_CUDA_VISIBLE_DEVICE", "0")
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


def measure(doc, inp, root: Path, bench_dir: Path, out: Path, warmup: int, reps: int, timeout: int, gpus: int):
    out.mkdir(parents=True, exist_ok=True)
    cmd, env, entry_path = build_command(doc, inp, root, bench_dir, gpus)
    e2e_boundary = ("process wall of the benchmark binary (fork/exec to exit; includes CUDA context creation)"
                    if doc["entry"]["kind"] == "binary" else
                    "wrapper wall of run.sh (env sourcing, launcher, mpirun, binding audit, application)")
    meta = {"schema": "hpcperf-inputs-measurement-1", "benchmark": doc["benchmark"], "level": doc["level"],
            "input_id": inp["id"], "inputs_yaml_sha256": doc["_sha256"], "entry": doc["entry"],
            "entry_sha256": sha256_file(entry_path), "command": cmd, "gpus": gpus,
            "selector": {doc.get("selector"): inp["id"]} if doc.get("selector") else None,
            "host": socket.gethostname(), "gpu": gpu_info(), "git": git_head(root),
            "warmup_runs": warmup, "measured_runs": reps, "timeout_s": timeout,
            "e2e_boundary": e2e_boundary, "timing_scope": doc["timing"]["scope"],
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": []}
    for k in range(warmup + reps):
        label = f"warmup{k}" if k < warmup else f"rep{k - warmup + 1}"
        rdir = out / label
        if rdir.exists():
            shutil.rmtree(rdir)
        rdir.mkdir(parents=True)
        renv = dict(env)
        if doc["level"] == 3:
            renv["HPCPERF_L3_RUN_SUBDIR"] = f"run.inputs.{inp['id']}.{label}"   # isolate app-written results per run
        log = rdir / "stdout.log"
        rc, wall = run_once(cmd, renv, rdir, log, timeout)
        rec = {"label": label, "exit_code": rc, "e2e_s": round(wall, 4), "log": str(log), "measured": k >= warmup}
        try:
            rec["timing"] = parse_timing(doc, log, rc, inp.get("params"))
            rec["main_compute_s"] = rec["timing"]["main_compute_s"]
        except InputError as ex:
            rec["timing_error"] = str(ex); rec["main_compute_s"] = None
        rec["baseline_quantities"] = extract(doc, log, inp) if rc == 0 else None
        (rdir / "result.json").write_text(json.dumps(rec, indent=2))
        meta["runs"].append(rec)
        print(f"[{doc['benchmark']}/{inp['id']}] {label}: rc={rc} e2e={wall:.3f}s main_compute="
              f"{rec['main_compute_s'] if rec['main_compute_s'] is None else round(rec['main_compute_s'], 4)}"
              + (f" ({rec['timing_error']})" if 'timing_error' in rec else ""), flush=True)
    measured = [r for r in meta["runs"] if r["measured"]]
    good = [r for r in measured if r["exit_code"] == 0]
    mc = [r["main_compute_s"] for r in good if r["main_compute_s"] is not None]
    e2e = [r["e2e_s"] for r in good]

    def stats(v):
        if not v:
            return None
        med = statistics.median(v)
        return {"n": len(v), "median": med, "min": min(v), "max": max(v),
                "mad": statistics.median([abs(x - med) for x in v]),
                "spread_rel": (max(v) - min(v)) / med if med > 0 else None, "values": v}
    summary = {"run_ok": len(good) == len(measured) and len(measured) == reps,
               "timing_ok": len(mc) == len(measured) and len(measured) == reps,
               "main_compute_s": stats(mc), "e2e_s": stats(e2e)}
    summary["compute_ge_1s"] = bool(mc) and summary["main_compute_s"]["median"] >= 1.0
    summary["stable"] = (bool(mc) and summary["main_compute_s"]["spread_rel"] is not None
                         and summary["main_compute_s"]["spread_rel"] <= 0.10)
    # baseline = the first measured run that succeeded; every later measured run is compared with it
    base = next((r for r in good if r["baseline_quantities"]), None)
    if base:
        bfile = out / "baseline.json"
        bfile.write_text(json.dumps({"schema": "hpcperf-inputs-baseline-1", "benchmark": doc["benchmark"],
                                     "input_id": inp["id"], "from_run": base["label"], "log": base["log"],
                                     "method": doc["baseline"].get("method"), "reference": (inp.get("baseline") or {}).get("reference", doc["baseline"].get("reference")),
                                     "quantity_rules": quantities(doc, inp), "quantities": base["baseline_quantities"]}, indent=2))
        summary["baseline_file"] = str(bfile)
        cmps = [compare(doc, base["baseline_quantities"], r["baseline_quantities"], inp) for r in good if r["baseline_quantities"]]
        summary["baseline_self_consistent"] = all(c["ok"] for c in cmps)
        summary["baseline_checks"] = cmps
    else:
        summary["baseline_file"] = None
        summary["baseline_self_consistent"] = False
    meta["summary"] = summary
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "measurement.json").write_text(json.dumps(meta, indent=2))
    return meta


# ----------------------------------------------------------------------------- cli
def main(argv=None):
    ap = argparse.ArgumentParser(prog="hpcperf_inputs.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("validate", "list"):
        s = sub.add_parser(c); s.add_argument("bench_dir")
    for c in ("show", "args"):
        s = sub.add_parser(c); s.add_argument("bench_dir"); s.add_argument("input_id")
    s = sub.add_parser("param"); s.add_argument("bench_dir"); s.add_argument("input_id"); s.add_argument("key")
    s = sub.add_parser("parse-timing"); s.add_argument("bench_dir"); s.add_argument("log"); s.add_argument("--rc", type=int, default=0); s.add_argument("--input")
    s = sub.add_parser("extract"); s.add_argument("bench_dir"); s.add_argument("log"); s.add_argument("--input")
    s = sub.add_parser("compare"); s.add_argument("bench_dir"); s.add_argument("baseline"); s.add_argument("log"); s.add_argument("--input")
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
            inp = get_input(doc, a.input or bdoc.get("input_id")) if (a.input or bdoc.get("input_id")) else None
            res = compare(doc, bdoc["quantities"], extract(doc, Path(a.log), inp), inp)
            print(json.dumps(res, indent=2)); return 0 if res["ok"] else 1
        if a.cmd == "measure":
            inp = get_input(doc, a.input_id)
            root = repo_root(bench_dir)
            meta = measure(doc, inp, root, bench_dir, Path(a.out), a.warmup, a.reps, a.timeout, a.gpus)
            s = meta["summary"]
            print(json.dumps({k: s[k] for k in ("run_ok", "timing_ok", "compute_ge_1s", "stable", "baseline_self_consistent")}))
            return 0 if s["run_ok"] and s["timing_ok"] else 1
    except InputError as ex:
        sys.stderr.write(f"hpcperf_inputs: {ex}\n")
        return 2 if a.cmd in ("args", "param", "show") else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
