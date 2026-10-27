#!/usr/bin/env python3
"""Frozen golden references: the correctness oracle of every registered input whose benchmark
used to recompute its result with a CPU reference on every run.

    hpcperf_golden.py make   <bench_dir> <input_id>... | --all  [--replace --reason TEXT] [--timeout S]
    hpcperf_golden.py check  <bench_dir> <input_id> --dump DIR
    hpcperf_golden.py run    <bench_dir> [<input_id>] [--exe PATH] [--workdir DIR]
    hpcperf_golden.py status <bench_dir>...
    hpcperf_golden.py validate <bench_dir>...

make   ONE-TIME construction of the reference of an input: runs the committed, unmodified build on the
       reference GPU once with HPCPERF_CPU_REFERENCE=1 (the only place the benchmark's own CPU
       reference still executes), requires that CPU validation to pass, measures how far the GPU
       output is from the CPU reference, and freezes the GPU outputs: reference/<input_id>.json in
       the benchmark directory (rules, sha256, provenance; committed) plus, for outputs compared
       with a tolerance, the raw bytes in the content-addressed golden store
       ($HPCPERF_GOLDEN_STORE, default <repo>/.golden-store; never in git). An existing reference
       is never replaced without --replace --reason.
check  compares the outputs a run captured (HPCPERF_GOLDEN_DUMP=DIR, see hpcperf_golden.h) with
       the frozen reference. Used by the inputs.yaml `check:` block (post_run). Never runs a CPU
       reference and never writes a reference. Prints "PASS: golden check ..." / "FAIL: golden
       check ..."; exit 0 / 1 (2 = usage).
run    one plain run of the binary (no CPU reference) with capture, then check -- the ctest entry.
status which inputs have a reference and whether their blobs are in the store.

Rules (inputs.yaml `golden:`; frozen into the reference by make):
  exact       byte-identical (sha256) -- integer, permutation and other deterministic outputs
  max_abs     max |a - g|                     max_rel1p  max |a - g| / (1 + |g|)
  max_rel     max |a - g| / |g|               l1_rel     sum |a - g| / sum |g|
  max_relabs  max |a - g| / (rtol |g| + atol)  (tol 1: |a - g| <= rtol |g| + atol, e.g. ExaCMech)
  l2_rel_sq   sum (a - g)^2 / sum g^2         rmse       sqrt(mean (a - g)^2)
  trunc_int   an image written as (int) pixels: d = |trunc(a) - trunc(g)|, max d <= tol and the share of
              pixels with d >= 1 <= frac_tol (srad_v1's upstream-less criterion, made conservative)
An output is either captured by the program (hpcperf_golden.h) or, for applications that print their
result, read from the run's log: `log: {regex: <one (?P<value>) group, or several groups flattened per
line in order>}` (all matching lines, f64). `cpu:` names the CPU-reference capture or reads it from a
log written by the one-time CPU validation (`cpu: {log: {file: <template>, regex: <default: the same>}}`).
`cpu_check.after_run: true` = make runs the plain GPU command first and then cpu_check.command (templated,
{log} = that GPU run's log; `env:` templated) as the one-time CPU validation of that very output.
Each tolerance is the benchmark's own upstream tolerance for its GPU-vs-CPU check. make measures
d0 = the same metric between the B200 output and the CPU reference, and the check uses
tol_effective = tol - d0 (for l2_rel_sq: (sqrt tol - sqrt d0)^2): by the triangle inequality a
candidate within tol_effective of the golden is within the upstream tolerance of the CPU reference,
so replacing the CPU reference by the golden does not loosen the original criterion (relative metrics:
(tol - d0) / (1 + d0), see tol_effective). A non-finite value where the golden is finite always fails.
"""
import argparse, hashlib, json, os, re, shutil, socket, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "inputs"))
import hpcperf_inputs as hi  # noqa: E402  (registry loading, workload identity, templating)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

REF_SCHEMA = "hpcperf-golden-1"
METRICS = ("exact", "max_abs", "max_rel1p", "max_rel", "max_relabs", "l1_rel", "l2_rel_sq", "rmse", "trunc_int")
NORM_METRICS = ("max_abs", "max_rel1p", "max_rel", "max_relabs", "l1_rel", "rmse")
DTYPES = {"f32": "<f4", "f64": "<f8", "i8": "i1", "u8": "u1", "i16": "<i2", "u16": "<u2",
          "i32": "<i4", "u32": "<u4", "i64": "<i8", "u64": "<u8"}
MODES = ("data", "hash")
CHUNK = 1 << 23
EXIT_GOLDEN_IO = 97


class GoldenError(Exception):
    pass


# ----------------------------------------------------------------------------- registry side
def spec_of(doc, inp):
    """The golden spec of an input: its own `golden:` block or the benchmark-level one."""
    if inp is not None and "golden" in inp:
        return inp["golden"]
    return doc.get("golden")


def validate_spec(spec, where="golden"):
    errs = []
    if not isinstance(spec, dict):
        return [f"{where}: must be a mapping"]
    if spec.get("unsupported") is not None:
        if not isinstance(spec["unsupported"], str) or not spec["unsupported"].strip():
            errs.append(f"{where}.unsupported: give the reason a golden comparison is not meaningful")
        return errs
    if spec.get("mode", "data") not in MODES:
        errs.append(f"{where}.mode must be one of {MODES}")
    if spec.get("binary") is not None and not isinstance(spec["binary"], str):
        errs.append(f"{where}.binary: the repository-relative executable a run.sh entry launches")
    cc = spec.get("cpu_check")
    if not isinstance(cc, dict) or not isinstance(cc.get("pass_regex"), str):
        errs.append(f"{where}.cpu_check needs pass_regex (how make recognises the one-time CPU validation passed)")
    else:
        for k in ("pass_regex", "fail_regex"):
            if cc.get(k) is not None:
                try:
                    re.compile(str(cc[k]))
                except re.error as ex:
                    errs.append(f"{where}.cpu_check.{k}: {ex}")
        if cc.get("command") is not None and (not isinstance(cc["command"], list) or not cc["command"]):
            errs.append(f"{where}.cpu_check.command must be a list of tokens")
        if cc.get("after_run") and not cc.get("command"):
            errs.append(f"{where}.cpu_check.after_run needs the command that validates the GPU run")
        if cc.get("env") is not None and not isinstance(cc["env"], dict):
            errs.append(f"{where}.cpu_check.env must be a mapping")
    outs = spec.get("outputs")
    if not isinstance(outs, list) or not outs:
        errs.append(f"{where}.outputs must be a non-empty list"); return errs
    for i, o in enumerate(outs):
        w = f"{where}.outputs[{i}]"
        if not isinstance(o, dict):
            errs.append(f"{w}: must be a mapping"); continue
        if bool(o.get("name")) == bool(o.get("match")):
            errs.append(f"{w}: exactly one of name / match")
        if o.get("match") is not None:
            try:
                re.compile(o["match"])
            except re.error as ex:
                errs.append(f"{w}.match: {ex}")
            if not isinstance(o.get("count"), int) or o["count"] < 1:
                errs.append(f"{w}: a match needs count (how many outputs it must select)")
        if o.get("dtype") not in DTYPES:
            errs.append(f"{w}.dtype must be one of {tuple(DTYPES)}")
        if o.get("log") is not None:
            lg = o["log"]
            if not isinstance(lg, dict) or not isinstance(lg.get("regex"), str):
                errs.append(f"{w}.log needs regex")
            else:
                try:
                    re.compile(lg["regex"], re.MULTILINE)
                except re.error as ex:
                    errs.append(f"{w}.log.regex: {ex}")
            if o.get("dtype") not in ("f64", "i64"):
                errs.append(f"{w}: values read from a log are f64 (or i64)")
            if o.get("match"):
                errs.append(f"{w}: a log output needs a name, not match")
        r = o.get("rule")
        if not isinstance(r, dict) or r.get("metric") not in METRICS:
            errs.append(f"{w}.rule.metric must be one of {METRICS}")
        elif r["metric"] != "exact":
            if not isinstance(r.get("tol"), (int, float)) or isinstance(r.get("tol"), bool) or r["tol"] <= 0:
                errs.append(f"{w}.rule.tol must be a positive number")
            if (spec.get("mode", "data")) == "hash":
                errs.append(f"{w}: a tolerance rule needs mode data (the bytes are compared)")
            if o["dtype"] not in ("f32", "f64"):
                errs.append(f"{w}: tolerance rules apply to f32/f64 outputs; compare integers exactly")
            if r["metric"] == "trunc_int" and (not isinstance(r.get("frac_tol"), (int, float)) or r["frac_tol"] < 0):
                errs.append(f"{w}.rule: trunc_int needs frac_tol (the allowed share of pixels that differ)")
        if not isinstance(o.get("basis"), str) or not o["basis"].strip():
            errs.append(f"{w}.basis: what is compared and where the tolerance comes from")
        c = o.get("cpu")
        if c is not None and not isinstance(c, str) and not (isinstance(c, dict) and isinstance(c.get("log"), dict)
                                                             and isinstance(c["log"].get("file"), str)):
            errs.append(f"{w}.cpu: name of the CPU-reference capture, {{log: {{file, regex?}}}}, or null")
        if isinstance(r, dict) and r.get("metric") == "max_relabs":
            for k in ("rtol", "atol"):
                if not isinstance(r.get(k), (int, float)) or r[k] < 0:
                    errs.append(f"{w}.rule: max_relabs needs {k} >= 0")
            if r.get("tol") != 1:
                errs.append(f"{w}.rule: max_relabs is normalised: tol must be 1")
    return errs


def ref_path(bench_dir: Path, input_id: str) -> Path:
    return Path(bench_dir) / "reference" / f"{input_id}.json"


def store_root(root: Path) -> Path:
    return Path(os.environ.get("HPCPERF_GOLDEN_STORE") or (root / ".golden-store"))


def blob_path(store: Path, sha: str) -> Path:
    return store / "sha256" / sha[:2] / f"{sha}.bin"


def frozen_spec(spec):
    """The part of the spec that defines the comparison (what the reference freezes)."""
    return {"mode": spec.get("mode", "data"),
            "outputs": [{k: o.get(k) for k in ("name", "match", "count", "dtype", "rule", "cpu", "basis", "log")} for o in spec["outputs"]]}


# ----------------------------------------------------------------------------- captures
def read_manifest(dump: Path) -> dict:
    m = Path(dump) / "manifest.jsonl"
    if not m.is_file():
        raise GoldenError(f"no capture manifest {m} (the run captured nothing: HPCPERF_GOLDEN_DUMP not honoured?)")
    out = {}
    for n, line in enumerate(m.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as ex:
            raise GoldenError(f"{m}:{n}: not JSON ({ex})")
        if rec["name"] in out:
            raise GoldenError(f"{m}: output '{rec['name']}' captured twice in one run")
        rec["path"] = str(Path(dump) / f"{rec['name']}.bin") if rec.get("data") else None
        out[rec["name"]] = rec
    return out


def log_values(path, regex):
    """Every match of `regex` in the file, in order: its (?P<value>) group, or all its groups (one line with
    several numbers -> several values)."""
    text = Path(path).read_text(errors="replace")
    vals = []
    for m in re.finditer(regex, text, re.MULTILINE):
        groups = [m.group("value")] if "value" in m.re.groupindex else list(m.groups())
        for gstr in groups:
            vals.append(float(gstr))
    return vals


def capture_logs(spec, dump: Path, gpu_log, cpu_ctx=None, cpu=False) -> list:
    """Write the log-sourced outputs of the spec into the capture directory (the files and manifest format of
    hpcperf_golden.h): the GPU values from the run's log, with cpu=True also the CPU reference values from
    the files the one-time CPU validation wrote. Returns notes."""
    sys.path.insert(0, str(HERE))
    import hpcperf_golden_capture as gc
    notes = []
    for o in spec["outputs"]:
        if not o.get("log"):
            continue
        dtype = DTYPES[o["dtype"]]
        if gpu_log is None or not Path(gpu_log).is_file():
            raise GoldenError(f"output '{o['name']}' is read from the run's log, but no log was given")
        vals = log_values(gpu_log, o["log"]["regex"])
        if not vals:
            raise GoldenError(f"output '{o['name']}': no line of {gpu_log} matches {o['log']['regex']!r}")
        gc.capture_to(str(dump), o["name"], np.array(vals), o["dtype"])
        notes.append(f"{o['name']}: {len(vals)} values from {gpu_log}")
        c = o.get("cpu")
        if cpu and isinstance(c, dict):
            f = render([c["log"]["file"]], cpu_ctx)[0]
            if not Path(f).is_file():
                raise GoldenError(f"CPU-reference log {f} of output '{o['name']}' missing")
            cv = log_values(f, c["log"].get("regex") or o["log"]["regex"])
            gc.capture_to(str(dump), o["name"] + ".cpu", np.array(cv), o["dtype"])
            notes.append(f"{o['name']}.cpu: {len(cv)} values from {f}")
    return notes


def resolve_outputs(spec, manifest: dict) -> list:
    """(spec item, output name) for every output the spec selects, in spec order."""
    sel = []
    for o in spec["outputs"]:
        if o.get("name"):
            names = [o["name"]]
        else:
            rx = re.compile(o["match"])
            names = sorted(n for n in manifest if rx.fullmatch(n) and not n.endswith(".cpu"))
            if len(names) != o["count"]:
                raise GoldenError(f"match '{o['match']}' selected {len(names)} outputs, expected {o['count']}: {names}")
        sel += [(o, n) for n in names]
    return sel


def _need_numpy():
    if np is None:
        raise GoldenError("numpy is required for tolerance comparisons (source hpcperf_env.sh)")


def _arr(path, dtype, count):
    if count == 0:
        return np.zeros(0, dtype=DTYPES[dtype])
    return np.memmap(path, dtype=DTYPES[dtype], mode="r", shape=(count,))


def array_stats(path, dtype, count, sha_check=None):
    """min / max / sum / non-finite count of a capture, streaming; optionally verifies its sha256."""
    _need_numpy()
    a = _arr(path, dtype, count)
    h = hashlib.sha256() if sha_check else None
    mn = mx = None; s = 0.0; nonfinite = 0
    for i in range(0, count, CHUNK):
        c = np.asarray(a[i:i + CHUNK])
        if h:
            h.update(c)
        x = c.astype(np.float64)
        fin = np.isfinite(x)
        nonfinite += int(x.size - fin.sum())
        if fin.any():
            xf = x[fin]
            mn = float(xf.min()) if mn is None else min(mn, float(xf.min()))
            mx = float(xf.max()) if mx is None else max(mx, float(xf.max()))
            s += float(xf.sum())
    if h and h.hexdigest() != sha_check:
        raise GoldenError(f"{path}: content does not match its recorded sha256")
    return {"min": mn, "max": mx, "sum": s, "nonfinite": nonfinite}


def metric_value(metric, a_path, g_path, dtype, count, a_sha=None, g_sha=None, rule=None):
    """The rule's metric of candidate a against reference g (streaming, float64). A non-finite a
    where g is finite gives inf. Verifies both files against their sha256 when given."""
    _need_numpy()
    a = _arr(a_path, dtype, count); g = _arr(g_path, dtype, count)
    ha = hashlib.sha256() if a_sha else None
    hg = hashlib.sha256() if g_sha else None
    worst = 0.0; num = 0.0; den = 0.0
    inf = float("inf")
    for i in range(0, count, CHUNK):
        ca = np.asarray(a[i:i + CHUNK]); cg = np.asarray(g[i:i + CHUNK])
        if ha:
            ha.update(ca)
        if hg:
            hg.update(cg)
        if worst == inf:
            continue                      # already a FAIL; keep reading only for the integrity checks
        with np.errstate(all="ignore"):
            d = np.abs(np.subtract(ca, cg, dtype=np.float64))
            if not np.isfinite(d).all():
                # slow path: a non-finite value somewhere. The same non-finite value on both sides (inf == inf,
                # nan vs nan) is no difference; anything else non-finite fails.
                x = ca.astype(np.float64); y = cg.astype(np.float64)
                same = ~np.isfinite(x) & ~np.isfinite(y) & ((x == y) | (np.isnan(x) & np.isnan(y)))
                if (~np.isfinite(d) & ~same).any():
                    worst = inf; continue
                d[same] = 0.0
            if d.size == 0:
                continue
            if metric == "max_abs":
                worst = max(worst, float(d.max()))
            elif metric == "max_rel1p":
                worst = max(worst, float((d / (1.0 + np.abs(cg, dtype=np.float64))).max()))
            elif metric == "max_rel":
                ag = np.abs(cg, dtype=np.float64)
                r = np.where(ag != 0, d / ag, np.where(d == 0, 0.0, inf))
                worst = max(worst, float(r.max()))
            elif metric == "max_relabs":
                den_ = rule["rtol"] * np.abs(cg, dtype=np.float64) + rule["atol"]
                r = np.where(den_ > 0, d / den_, np.where(d == 0, 0.0, inf))
                worst = max(worst, float(r.max()))
            elif metric == "l1_rel":
                num += float(d.sum()); den += float(np.abs(cg, dtype=np.float64).sum())
            elif metric in ("l2_rel_sq", "rmse"):
                y = cg.astype(np.float64)
                num += float(np.dot(d, d)); den += float(np.dot(y, y))
    if ha and ha.hexdigest() != a_sha:
        raise GoldenError(f"{a_path}: content does not match its recorded sha256 (partial capture?)")
    if hg and hg.hexdigest() != g_sha:
        raise GoldenError(f"{g_path}: golden blob does not match its sha256 (corrupted store)")
    if worst == float("inf"):
        return float("inf")
    if metric == "l1_rel":
        return num / den if den > 0 else (0.0 if num == 0 else float("inf"))
    if metric == "l2_rel_sq":
        return num / den if den > 0 else (0.0 if num == 0 else float("inf"))
    if metric == "rmse":
        return (num / count) ** 0.5 if count else 0.0
    return worst


def trunc_int_value(a_path, g_path, dtype, count, a_sha=None, g_sha=None):
    """(max |trunc a - trunc g|, share of elements with a difference >= 1) -- C's (int) cast is a
    truncation toward zero. A non-finite a where g is finite gives (inf, 1)."""
    _need_numpy()
    a = _arr(a_path, dtype, count); g = _arr(g_path, dtype, count)
    ha = hashlib.sha256() if a_sha else None
    hg = hashlib.sha256() if g_sha else None
    worst = 0.0; ge1 = 0
    for i in range(0, count, CHUNK):
        ca = np.asarray(a[i:i + CHUNK]); cg = np.asarray(g[i:i + CHUNK])
        if ha:
            ha.update(ca)
        if hg:
            hg.update(cg)
        x = ca.astype(np.float64); y = cg.astype(np.float64)
        if (~np.isfinite(x) & np.isfinite(y)).any():
            worst = float("inf"); ge1 = count; continue
        with np.errstate(all="ignore"):
            d = np.abs(np.trunc(x) - np.trunc(y))
        d[~np.isfinite(d)] = 0.0
        worst = max(worst, float(d.max()) if d.size else 0.0)
        ge1 += int((d >= 1).sum())
    if ha and ha.hexdigest() != a_sha:
        raise GoldenError(f"{a_path}: content does not match its recorded sha256 (partial capture?)")
    if hg and hg.hexdigest() != g_sha:
        raise GoldenError(f"{g_path}: golden blob does not match its sha256 (corrupted store)")
    return worst, (ge1 / count if count else 0.0)


def tol_effective(metric, tol, d0, rule=None):
    """The tolerance against the golden g that keeps a candidate a within `tol` of the CPU reference c, given
    d0 = metric(g, c) (triangle inequality, strictly):
      max_abs, rmse                 |a-c| <= |a-g| + |g-c|                      -> tol - d0
      max_rel1p, max_rel, l1_rel    the golden's magnitude in the denominator is at most (1 + d0) times the
                                    CPU reference's                              -> (tol - d0) / (1 + d0)
      max_relabs (tol 1)            rtol |g| + atol <= (rtol |c| + atol)(1 + rtol d0) -> (1 - d0) / (1 + rtol d0)
      l2_rel_sq                     the same in the norm, s = sqrt               -> ((s tol - s d0) / (1 + s d0))^2"""
    if d0 is None:
        return tol
    if metric in ("max_abs", "rmse"):
        return max(tol - d0, 0.0)
    if metric in ("max_rel1p", "max_rel", "l1_rel"):
        return max((tol - d0) / (1.0 + d0), 0.0)
    if metric == "max_relabs":
        return max((tol - d0) / (1.0 + rule["rtol"] * d0), 0.0)
    if metric == "l2_rel_sq":
        r = (tol ** 0.5 - d0 ** 0.5) / (1.0 + d0 ** 0.5)
        return r * r if r > 0 else 0.0
    raise GoldenError(f"no tolerance budget for metric {metric}")


# ----------------------------------------------------------------------------- helpers
def git(root: Path, *a):
    return subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout


def gpu_identity():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
        dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        idx = int(dev) if dev.isdigit() and int(dev) < len(lines) else 0
        name, uuid, drv = [x.strip() for x in lines[idx].split(",")]
        return {"name": name, "uuid": uuid, "driver": drv}
    except Exception:
        return None


def cuda_version():
    for p in ("/usr/local/cuda/version.json",):
        try:
            return json.load(open(p))["cuda"]["version"]
        except Exception:
            pass
    return None


def clean_env(base=None):
    """The environment of a golden run: never a skip-verify or CPU-reference run unless asked."""
    env = dict(os.environ if base is None else base)
    for k in ("HPCPERF_SKIP_VERIFY", "HPCPERF_CPU_REFERENCE", "HPCPERF_GOLDEN_DUMP", "HPCPERF_GOLDEN_MODE",
              "HPCPERF_RANDOM_SEED"):
        env.pop(k, None)
    env.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("HPCPERF_CUDA_VISIBLE_DEVICE", "0"))
    return env


def run_logged(cmd, env, cwd: Path, log: Path, timeout: int):
    t0 = time.monotonic()
    with open(log, "w") as fh:
        try:
            rc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    return rc, time.monotonic() - t0


def context(doc, inp, root, bench_dir, run_dir, log=None):
    return hi.check_context(doc, inp, root, bench_dir, run_dir, log=log, outputs=None)


def render(tokens, ctx):
    out = []
    for t in tokens:
        r = hi.render_token(str(t), ctx)
        out += r if isinstance(r, list) else [r]
    return out


HARNESS = re.compile(r"(^|/)(reference/|tests?/|inputs\.yaml$|[^/]*\.md$|run\.sh$|validate\.sh$|build\.sh$|check_[^/]*$|"
                     r"generate_reference[^/]*$|LICENSE|COPYRIGHT|NOTICE|CHANGELOG|RELEASE$)")


def source_files(bench_dir: Path, root: Path, level=1):
    """The files a build of the benchmark depends on (for the binary-older-than-sources refusal): Level 1 its
    cuda/ common/ hip/ trees; Level 2 every vendored file except the harness (run/validate/check scripts, docs,
    references); plus the shared headers."""
    files = []
    if level == 1:
        for sub in ("cuda", "common", "hip"):
            d = bench_dir / sub
            if d.is_dir():
                files += [p for p in d.rglob("*") if p.is_file()]
    else:
        files += [p for p in bench_dir.rglob("*") if p.is_file() and not HARNESS.search(str(p.relative_to(bench_dir)))]
    files += [root / "tools/golden/hpcperf_golden.h", root / "tools/timing/roi/hpcperf_roi.h"]
    return files


# ----------------------------------------------------------------------------- make
def make_one(doc, inp, root: Path, bench_dir: Path, args) -> int:
    spec = spec_of(doc, inp)
    tag = f"{doc['benchmark']}/{inp['id']}"
    if spec is None:
        print(f"REFUSED: {tag}: no golden spec in inputs.yaml"); return 2
    errs = validate_spec(spec)
    if errs:
        print(f"REFUSED: {tag}: " + "; ".join(errs)); return 2
    if spec.get("unsupported"):
        print(f"REFUSED: {tag}: golden comparison not applicable ({spec['unsupported']})"); return 2
    rp = ref_path(bench_dir, inp["id"])
    if rp.exists() and not args.replace:
        print(f"REFUSED: {tag}: {rp} exists (frozen); replacing it needs --replace --reason"); return 2
    if args.replace and not (args.reason or "").strip():
        print(f"REFUSED: {tag}: --replace needs --reason"); return 2
    rel_bench = os.path.relpath(bench_dir, root)
    if rel_bench.startswith("workspaces") or "/workspaces/" in str(bench_dir):
        print(f"REFUSED: {tag}: a golden reference is never made from an agent workspace"); return 2
    canonical = (root / f"level{doc['level']}" / doc["benchmark"]).resolve()
    if Path(bench_dir).resolve() != canonical:
        print(f"REFUSED: {tag}: {bench_dir} is not the canonical benchmark directory {os.path.relpath(canonical, root)} "
              f"(a copy -- e.g. an optimization candidate -- never makes a golden reference)"); return 2
    dirty = git(root, "status", "--porcelain", "--", rel_bench, "tools/golden", "tools/timing/roi").strip()
    dirty = "\n".join(l for l in dirty.splitlines() if not l[3:].startswith(f"{rel_bench}/reference/"))
    if dirty and not args.allow_dirty_for_tests:
        print(f"REFUSED: {tag}: uncommitted changes under {rel_bench} / tools/golden / tools/timing/roi -- a golden "
              f"reference comes only from committed, unmodified sources:\n{dirty}"); return 2
    try:
        cmd, env0, exe = hi.build_command(doc, inp, root, bench_dir, 1)
    except hi.InputError as ex:
        print(f"REFUSED: {tag}: {ex}"); return 2
    if doc["entry"]["kind"] != "binary" and not inp.get("binary"):
        # a run.sh entry: the executable it launches is named by the golden spec
        if not spec.get("binary"):
            print(f"REFUSED: {tag}: a run.sh entry needs golden.binary (the executable run.sh launches) for provenance"); return 2
        exe = root / spec["binary"]
        if not exe.is_file():
            print(f"REFUSED: {tag}: {exe} not built"); return 2
    newer = [str(p) for p in source_files(bench_dir, root, doc["level"]) if p.exists() and p.stat().st_mtime > exe.stat().st_mtime + 1]
    if newer:
        print(f"REFUSED: {tag}: the binary {exe} is older than {len(newer)} of its sources (rebuild first): {newer[:3]}"); return 2
    work = Path(args.workdir or (root / "build" / "golden-make" / doc["benchmark"] / inp["id"]))
    if work.exists():
        shutil.rmtree(work)
    run_dir = work / "run"; dump = work / "capture"
    run_dir.mkdir(parents=True)
    env = clean_env(env0)
    env.update({"HPCPERF_CPU_REFERENCE": "1", "HPCPERF_GOLDEN_DUMP": str(dump), "HPCPERF_GOLDEN_MODE": spec.get("mode", "data")})
    log = run_dir / "stdout.log"
    cc = spec["cpu_check"]
    gpu_log = None
    cpu_ctx = None
    if cc.get("after_run"):
        # the plain GPU run first (no CPU reference), then the one-time CPU validation of that output
        genv = clean_env(env0)
        genv.update({"HPCPERF_GOLDEN_DUMP": str(dump), "HPCPERF_GOLDEN_MODE": spec.get("mode", "data")})
        gpu_log = run_dir / "gpu_run.log"
        print(f"[{tag}] GPU run: {' '.join(cmd)}", flush=True)
        grc, gwall = run_logged(cmd, genv, run_dir, gpu_log, args.timeout)
        if grc != 0:
            print(f"REFUSED: {tag}: the GPU run exited {grc}; log {gpu_log}"); return 1
        cpu_ctx = hi.check_context(doc, inp, root, bench_dir, run_dir, log=gpu_log, outputs=None)
        env.pop("HPCPERF_GOLDEN_DUMP", None); env.pop("HPCPERF_GOLDEN_MODE", None)   # the validation captures nothing
        for k, v in (cc.get("env") or {}).items():
            env[k] = render([str(v)], cpu_ctx)[0]
        gpu_cmd = cmd
        cmd = render(cc["command"], cpu_ctx)
    elif cc.get("command"):
        ctx = context(doc, inp, root, bench_dir, run_dir)
        cpu_ctx = ctx
        for k, v in (cc.get("env") or {}).items():
            env[k] = render([str(v)], ctx)[0]
        cmd = render(cc["command"], ctx)
    print(f"[{tag}] one-time CPU validation: {' '.join(cmd)}", flush=True)
    rc, wall = run_logged(cmd, env, run_dir, log, args.timeout)
    if gpu_log is None:
        gpu_log = log
    if cpu_ctx is None:
        cpu_ctx = hi.check_context(doc, inp, root, bench_dir, run_dir, log=gpu_log, outputs=None)
    text = log.read_text(errors="replace")
    passes = [l for l in text.splitlines() if re.search(cc["pass_regex"], l)]
    fails = [l for l in text.splitlines() if cc.get("fail_regex") and re.search(cc["fail_regex"], l)]
    if rc != 0 or not passes or fails:
        why = f"exit {rc}" if rc != 0 else ("a fail line: " + fails[0].strip()) if fails else "no pass line"
        print(f"REFUSED: {tag}: the CPU-reference validation of the B200 output did not pass ({why}); log {log}")
        return 1
    try:
        if any(o.get("log") for o in spec["outputs"]):
            for n in capture_logs(spec, dump, gpu_log, cpu_ctx, cpu=True):
                print(f"   {n}")
        man = read_manifest(dump)
        sel = resolve_outputs(spec, man)
    except GoldenError as ex:
        print(f"REFUSED: {tag}: {ex}"); return 1
    store = store_root(root)
    outs = []
    for o, name in sel:
        rec = man.get(name)
        if rec is None:
            print(f"REFUSED: {tag}: output '{name}' was not captured"); return 1
        if rec["dtype"] != o["dtype"]:
            print(f"REFUSED: {tag}: output '{name}' captured as {rec['dtype']}, spec says {o['dtype']}"); return 1
        rule = dict(o["rule"])
        entry = {"name": name, "dtype": rec["dtype"], "count": rec["count"], "bytes": rec["bytes"], "sha256": rec["sha256"],
                 "rule": rule, "basis": o["basis"]}
        if rec.get("data"):
            entry["stats"] = array_stats(rec["path"], rec["dtype"], rec["count"], sha_check=rec["sha256"])
            if rule["metric"] != "exact" and entry["stats"]["nonfinite"]:
                print(f"REFUSED: {tag}: output '{name}' has {entry['stats']['nonfinite']} non-finite values; a tolerance "
                      f"reference must be finite (report it)"); return 1
        if rule["metric"] != "exact":
            c = o.get("cpu")
            cpu_name = (name + ".cpu") if isinstance(c, dict) else ((c or "").replace("{name}", name) or None)
            if cpu_name:
                crec = man.get(cpu_name)
                if crec is None or not crec.get("data") or crec["count"] != rec["count"] or crec["dtype"] != rec["dtype"]:
                    print(f"REFUSED: {tag}: CPU-reference capture '{cpu_name}' missing or of another shape"); return 1
                if rule["metric"] == "trunc_int":
                    d0, f0 = trunc_int_value(rec["path"], crec["path"], rec["dtype"], rec["count"], rec["sha256"], crec["sha256"])
                    rule["d0_frac_vs_cpu_reference"] = f0
                    rule["frac_tol_effective"] = max(rule["frac_tol"] - f0, 0.0)
                else:
                    d0 = metric_value(rule["metric"], rec["path"], crec["path"], rec["dtype"], rec["count"], rec["sha256"], crec["sha256"], rule)
                rule["d0_vs_cpu_reference"] = d0
                rule["d0_source"] = cpu_name
            else:
                d0 = None
                rule["d0_vs_cpu_reference"] = None
                rule["d0_source"] = "not measured (the CPU reference keeps no comparable array); tol_effective = tol"
                if rule["metric"] == "trunc_int":
                    rule["d0_frac_vs_cpu_reference"] = None
                    rule["frac_tol_effective"] = rule["frac_tol"]
            te = tol_effective(rule["metric"], rule["tol"], d0, rule) if rule["metric"] != "trunc_int" else None
            if rule["metric"] == "trunc_int":
                te = rule["tol"] - (d0 or 0.0)          # integer pixel differences: tol - d0 may be 0 (exact pixels)
                if te < 0:
                    print(f"REFUSED: {tag}: output '{name}': the B200 image is {d0} levels from the CPU reference, "
                          f"more than the tolerance {rule['tol']} (report it)"); return 1
                rule["tol_effective"] = te
            elif te <= 0:
                print(f"REFUSED: {tag}: output '{name}': the B200 output is {d0:.3e} from the CPU reference, which uses up "
                      f"the whole tolerance {rule['tol']:.3e} (report it)"); return 1
            else:
                rule["tol_effective"] = te
            bp = blob_path(store, rec["sha256"])
            if not bp.exists():
                bp.parent.mkdir(parents=True, exist_ok=True)
                tmp = bp.with_suffix(".tmp")
                shutil.copyfile(rec["path"], tmp)
                os.replace(tmp, bp)
            entry["blob"] = f"sha256/{rec['sha256'][:2]}/{rec['sha256']}.bin"
        outs.append(entry)
    ref = {"schema": REF_SCHEMA, "benchmark": doc["benchmark"], "level": doc["level"], "input_id": inp["id"],
           "frozen": True, "workload": hi.workload_identity(doc, inp), "inputs_yaml_sha256": doc["_sha256"],
           "spec": frozen_spec(spec), "outputs": outs,
           "provenance": {
               "made_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname(),
               "gpu": gpu_identity(), "cuda_toolkit": cuda_version(),
               "git_head": git(root, "rev-parse", "HEAD").strip(),
               "source_clean": not bool(dirty),
               "binary": os.path.relpath(exe, root), "binary_sha256": hi.sha256_file(exe),
               "command": [os.path.relpath(c, root) if str(c).startswith(str(root)) else c for c in cmd],
               "gpu_run": (None if not cc.get("after_run") else
                           {"command": [os.path.relpath(c, root) if str(c).startswith(str(root)) else c for c in gpu_cmd],
                            "log": os.path.relpath(gpu_log, root), "log_sha256": hi.sha256_file(gpu_log)}),
               "env": {"HPCPERF_CPU_REFERENCE": "1", "HPCPERF_GOLDEN_MODE": spec.get("mode", "data")},
               "cpu_validation": {"exit_code": rc, "wall_s": round(wall, 3), "pass_line": passes[0].strip(),
                                  "pass_regex": cc["pass_regex"], "fail_regex": cc.get("fail_regex"),
                                  "log": os.path.relpath(log, root), "log_sha256": hi.sha256_file(log)},
               "tool_sha256": {"hpcperf_golden.py": hi.sha256_file(HERE / "hpcperf_golden.py"),
                               "hpcperf_golden.h": hi.sha256_file(HERE / "hpcperf_golden.h")},
               "store": "blobs of tolerance outputs: $HPCPERF_GOLDEN_STORE (default <repo>/.golden-store), content-addressed",
           }}
    if args.replace:
        old = json.loads(rp.read_text()) if rp.exists() else None
        ref["replaces"] = {"reason": args.reason, "previous_made_utc": (old or {}).get("provenance", {}).get("made_utc"),
                           "previous_sha256": hi.sha256_file(rp) if rp.exists() else None}
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(ref, indent=2) + "\n")
    shown = ", ".join(f"{e['name']} {e['rule']['metric']}" + (f" tol_eff {e['rule']['tol_effective']:.3g} (d0 "
                      f"{e['rule']['d0_vs_cpu_reference'] if e['rule']['d0_vs_cpu_reference'] is None else format(e['rule']['d0_vs_cpu_reference'], '.3g')})"
                      if e['rule']['metric'] != 'exact' else "") for e in outs)
    if not args.keep_capture:      # the bytes that matter are in the store now; logs and manifest stay
        for f in dump.glob("*.bin"):
            f.unlink()
    print(f"FROZEN: {tag} -> {os.path.relpath(rp, root)} ({len(outs)} outputs: {shown}; CPU validation '{passes[0].strip()}' in {wall:.1f} s)")
    return 0


# ----------------------------------------------------------------------------- check
def check(doc, inp, root: Path, bench_dir: Path, dump: Path, log=None, result=None) -> int:
    """Compare a capture with the frozen reference. `result` (a dict) receives the machine-readable verdict:
    the reference file and its sha256, every output's metric value and limit, and the verdict/reason."""
    tag = f"{doc['benchmark']}/{inp['id']}"
    res = result if result is not None else {}
    res.update({"schema": "hpcperf-golden-verdict-1", "benchmark": doc["benchmark"], "level": doc["level"],
                "input_id": inp["id"], "verdict": None, "reason": None, "outputs": []})

    def fail(msg):
        res["verdict"] = "FAIL"; res["reason"] = msg
        print(f"FAIL: golden check {tag}: {msg}"); return 1
    canonical = (root / f"level{doc['level']}" / doc["benchmark"]).resolve()
    if Path(bench_dir).resolve() != canonical:
        return fail(f"{bench_dir} is not the canonical benchmark directory {os.path.relpath(canonical, root)}: the registry "
                    f"and the frozen reference are read only from there (a copy -- e.g. an optimization candidate's -- is never trusted)")
    spec = spec_of(doc, inp)
    if spec is None or spec.get("unsupported"):
        return fail("this input has no golden spec" + (f" ({spec['unsupported']})" if spec else ""))
    rp = ref_path(bench_dir, inp["id"])
    if not rp.is_file():
        return fail(f"no frozen reference {os.path.relpath(rp, root)} (make it once from the unmodified build: hpcperf_golden.py make)")
    try:
        ref = json.loads(rp.read_text())
    except json.JSONDecodeError as ex:
        return fail(f"{rp} is not JSON ({ex})")
    res.update({"reference": os.path.relpath(rp, root), "reference_sha256": hi.sha256_file(rp),
                "reference_made_utc": (ref.get("provenance") or {}).get("made_utc")})
    if ref.get("schema") != REF_SCHEMA or ref.get("input_id") != inp["id"] or ref.get("benchmark") != doc["benchmark"]:
        return fail(f"{rp} is not the reference of this input (schema/benchmark/input_id)")
    mism = hi.workload_mismatch(ref.get("workload") or {}, hi.workload_identity(doc, inp))
    if mism:
        return fail("the reference was made for another workload: " + "; ".join(mism))
    if ref.get("spec") != frozen_spec(spec):
        return fail("inputs.yaml golden spec differs from the one frozen in the reference (a rule change needs an explicit make --replace)")
    try:
        if any(o.get("log") for o in spec["outputs"]):
            Path(dump).mkdir(parents=True, exist_ok=True)
            for n in capture_logs(spec, Path(dump), log):
                print(f"   {n}")
        man = read_manifest(dump)
    except GoldenError as ex:
        return fail(str(ex))
    store = store_root(root)
    bad = []

    def note(name, ok, metric, value=None, limit=None, why=None):
        res["outputs"].append({"name": name, "ok": ok, "metric": metric, "value": value, "limit": limit, "why": why})
    for e in ref["outputs"]:
        name, rule = e["name"], e["rule"]
        nbad = len(bad)
        rec = man.get(name)
        if rec is None:
            bad.append(f"{name}: not captured"); print(f"   FAIL {name}: not captured"); continue
        if rec["dtype"] != e["dtype"] or rec["count"] != e["count"]:
            bad.append(f"{name}: shape"); print(f"   FAIL {name}: {rec['dtype']}[{rec['count']}] != reference {e['dtype']}[{e['count']}]"); continue
        if rule["metric"] == "exact":
            if rec.get("data"):
                try:
                    hi_sha = hashlib.sha256()
                    with open(rec["path"], "rb") as fh:
                        for blk in iter(lambda: fh.read(1 << 24), b""):
                            hi_sha.update(blk)
                    if hi_sha.hexdigest() != rec["sha256"]:
                        raise GoldenError("capture file does not match its manifest sha256")
                except (OSError, GoldenError) as ex:
                    bad.append(f"{name}: {ex}"); print(f"   FAIL {name}: {ex}"); continue
            if rec["sha256"] != e["sha256"]:
                bad.append(f"{name}: differs"); print(f"   FAIL {name}: exact: sha256 {rec['sha256'][:16]} != reference {e['sha256'][:16]}")
            else:
                print(f"   ok   {name}: exact ({e['dtype']}[{e['count']}], sha256 {e['sha256'][:16]})")
            continue
        if not rec.get("data"):
            bad.append(f"{name}: no data"); print(f"   FAIL {name}: captured as hash only; a tolerance rule needs the bytes"); continue
        bp = store / e["blob"]
        if not bp.is_file():
            bad.append(f"{name}: blob missing"); print(f"   FAIL {name}: golden blob {bp} missing (the golden store is not available)"); continue
        if rule["metric"] == "trunc_int":
            try:
                v, f = trunc_int_value(rec["path"], bp, e["dtype"], e["count"], rec["sha256"], e["sha256"])
            except GoldenError as ex:
                bad.append(f"{name}: {ex}"); print(f"   FAIL {name}: {ex}"); continue
            te, tf = rule["tol_effective"], rule["frac_tol_effective"]
            line = (f"trunc_int max {v:g} <= {te:g} and share(d>=1) {f:.3e} <= {tf:.3e} (tol {rule['tol']} / {rule['frac_tol']} "
                    f"- B200/CPU {rule.get('d0_vs_cpu_reference')} / {rule.get('d0_frac_vs_cpu_reference')})")
            if v <= te and f <= tf:
                print(f"   ok   {name}: {line}")
            else:
                bad.append(f"{name}: trunc_int"); print(f"   FAIL {name}: {line}")
            continue
        try:
            v = metric_value(rule["metric"], rec["path"], bp, e["dtype"], e["count"], rec["sha256"], e["sha256"], rule)
        except GoldenError as ex:
            bad.append(f"{name}: {ex}"); print(f"   FAIL {name}: {ex}"); continue
        te = rule["tol_effective"]
        d0 = rule.get("d0_vs_cpu_reference")
        how = f"tol {rule['tol']:.3g}" + (f" - B200/CPU {d0:.3g}" if d0 is not None else " (B200/CPU deviation not measured)")
        if not (v <= te):
            bad.append(f"{name}: {rule['metric']} {v:.3e} > {te:.3e}")
            print(f"   FAIL {name}: {rule['metric']} {v:.3e} > {te:.3e} ({how})")
            note(name, False, rule["metric"], v, te)
        else:
            print(f"   ok   {name}: {rule['metric']} {v:.3e} <= {te:.3e} ({how})")
            note(name, True, rule["metric"], v, te)
    failed = {b.split(":")[0] for b in bad}
    for e in ref["outputs"]:
        if not any(o["name"] == e["name"] for o in res["outputs"]):
            note(e["name"], e["name"] not in failed, e["rule"]["metric"], limit=e["rule"].get("tol_effective"),
                 why=next((b for b in bad if b.split(":")[0] == e["name"]), None))
    extra = sorted(set(man) - {e["name"] for e in ref["outputs"]})
    if extra:
        print(f"   note: captured but not in the reference (ignored): {', '.join(extra)}")
    if bad:
        return fail(f"{len(bad)} of {len(ref['outputs'])} outputs differ from the frozen B200 reference: " + "; ".join(bad))
    res["verdict"] = "PASS"
    print(f"PASS: golden check {tag} ({len(ref['outputs'])} outputs match the frozen reference made {ref['provenance'].get('made_utc')} "
          f"on {(ref['provenance'].get('gpu') or {}).get('name')})")
    return 0


# ----------------------------------------------------------------------------- run (ctest)
def run(doc, inp, root: Path, bench_dir: Path, exe, workdir, timeout, result=None) -> int:
    tag = f"{doc['benchmark']}/{inp['id']}"
    res = result if result is not None else {}
    spec = spec_of(doc, inp)
    try:
        cmd, env0, _ = hi.build_command(doc, inp, root, bench_dir, 1)
    except hi.InputError as ex:
        if not exe:
            print(f"FAIL: golden check {tag}: {ex}"); return 1
        cmd, env0 = [str(exe)] + hi.resolve_repo_args(root, inp.get("args", [])), dict(os.environ)
    if exe:
        cmd = [str(Path(exe).resolve())] + cmd[1:]
    work = Path(workdir or (root / "build" / "golden-run" / doc["benchmark"] / inp["id"]))
    if work.exists():
        shutil.rmtree(work)
    run_dir = work / "run"; dump = work / "capture"
    run_dir.mkdir(parents=True)
    env = clean_env(env0)
    env["HPCPERF_GOLDEN_DUMP"] = str(dump)
    env["HPCPERF_GOLDEN_MODE"] = (spec or {}).get("mode", "data")
    log = run_dir / "stdout.log"
    rc, wall = run_logged(cmd, env, run_dir, log, timeout)
    print(f"   run: {' '.join(cmd)} -> exit {rc} in {wall:.1f} s (log {log})")
    run_rec = {"binary": cmd[0], "binary_sha256": hi.sha256_file(cmd[0]), "command": cmd, "exit_code": rc,
               "wall_s": round(wall, 3), "log": str(log), "env": {"HPCPERF_GOLDEN_MODE": env["HPCPERF_GOLDEN_MODE"]},
               "cpu_reference": False}
    if rc != 0:
        print(log.read_text(errors="replace")[-2000:])
        res.update({"schema": "hpcperf-golden-verdict-1", "benchmark": doc["benchmark"], "input_id": inp["id"],
                    "verdict": "FAIL", "reason": f"the benchmark run exited {rc}", "outputs": [], "run": run_rec})
        print(f"FAIL: golden check {tag}: the benchmark run exited {rc}"); return 1
    out = check(doc, inp, root, bench_dir, dump, log, res)
    res["run"] = run_rec
    return out


# ----------------------------------------------------------------------------- store helpers
def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def store_put(root: Path, path: Path) -> str:
    """Copy a file into the content-addressed store (idempotent); returns its sha256. For frozen references
    that are files rather than captured arrays (e.g. a plotfile archive)."""
    sha = file_sha256(path)
    bp = blob_path(store_root(root), sha)
    if not bp.exists():
        bp.parent.mkdir(parents=True, exist_ok=True)
        tmp = bp.with_suffix(".tmp")
        shutil.copyfile(path, tmp)
        if file_sha256(tmp) != sha:
            tmp.unlink(); raise GoldenError(f"copy of {path} into the store does not match its sha256")
        os.replace(tmp, bp)
    return sha


def store_get(root: Path, sha: str) -> Path:
    """The verified store path of a blob; a missing or corrupted blob is an error (never a fallback)."""
    if not re.fullmatch(r"[0-9a-f]{64}", sha or ""):
        raise GoldenError(f"not a sha256: {sha!r}")
    bp = blob_path(store_root(root), sha)
    if not bp.is_file():
        raise GoldenError(f"golden blob {sha} missing in the store {store_root(root)}")
    if file_sha256(bp) != sha:
        raise GoldenError(f"golden blob {bp} does not match its sha256 (corrupted store)")
    return bp


# ----------------------------------------------------------------------------- status / validate
def status(bench_dir: Path, root: Path) -> int:
    doc = hi.load(bench_dir)
    store = store_root(root)
    worst = 0
    for inp in doc["inputs"]:
        spec = spec_of(doc, inp)
        rp = ref_path(bench_dir, inp["id"])
        if spec is None:
            print(f"{doc['benchmark']}/{inp['id']}: no golden spec"); continue
        if spec.get("unsupported"):
            print(f"{doc['benchmark']}/{inp['id']}: golden not applicable: {spec['unsupported']}"); continue
        if not rp.is_file():
            print(f"{doc['benchmark']}/{inp['id']}: MISSING reference"); worst = max(worst, 1); continue
        ref = json.loads(rp.read_text())
        miss = [e["name"] for e in ref["outputs"] if e.get("blob") and not (store / e["blob"]).is_file()]
        stale = hi.workload_mismatch(ref.get("workload") or {}, hi.workload_identity(doc, inp))
        spec_changed = ref.get("spec") != frozen_spec(spec)
        st = "STALE" if stale or spec_changed else ("BLOB-MISSING" if miss else "OK")
        if st != "OK":
            worst = max(worst, 1)
        print(f"{doc['benchmark']}/{inp['id']}: {st} ({len(ref['outputs'])} outputs, made {ref['provenance']['made_utc']} on "
              f"{(ref['provenance'].get('gpu') or {}).get('name')})" + (f" missing blobs: {miss}" if miss else "")
              + (f" workload: {stale}" if stale else "") + (" spec changed since freezing" if spec_changed else ""))
    return worst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("bench_dir"); m.add_argument("input_ids", nargs="*")
    m.add_argument("--all", action="store_true"); m.add_argument("--replace", action="store_true"); m.add_argument("--reason")
    m.add_argument("--timeout", type=int, default=7200); m.add_argument("--workdir")
    m.add_argument("--keep-capture", action="store_true", help="keep the captured .bin files in the work directory")
    m.add_argument("--allow-dirty-for-tests", action="store_true", help=argparse.SUPPRESS)
    c = sub.add_parser("check"); c.add_argument("bench_dir"); c.add_argument("input_id"); c.add_argument("--dump", required=True)
    c.add_argument("--log"); c.add_argument("--json", help="write the machine-readable verdict here")
    r = sub.add_parser("run"); r.add_argument("bench_dir"); r.add_argument("input_id", nargs="?")
    r.add_argument("--exe"); r.add_argument("--workdir"); r.add_argument("--timeout", type=int, default=7200)
    r.add_argument("--json", help="write the machine-readable verdict (incl. the binary's sha256) here")
    s = sub.add_parser("status"); s.add_argument("bench_dirs", nargs="+")
    v = sub.add_parser("validate"); v.add_argument("bench_dirs", nargs="+")
    sp = sub.add_parser("store-put", help="copy a file into the content-addressed golden store; prints its sha256")
    sp.add_argument("file")
    sg = sub.add_parser("store-get", help="print the verified store path of a blob (exit 1 when missing/corrupted)")
    sg.add_argument("sha256")
    a = ap.parse_args(argv)
    if a.cmd in ("store-put", "store-get"):
        root = hi.repo_root(HERE)          # the store of the repository this tool belongs to (or $HPCPERF_GOLDEN_STORE)
        try:
            print(store_put(root, Path(a.file)) if a.cmd == "store-put" else store_get(root, a.sha256))
            return 0
        except (GoldenError, OSError) as ex:
            print(f"ERROR: {ex}"); return 1
    try:
        if a.cmd in ("status", "validate"):
            rc = 0
            for b in a.bench_dirs:
                bd = Path(b).resolve(); root = hi.repo_root(bd)
                if a.cmd == "status":
                    rc = max(rc, status(bd, root))
                else:
                    doc = hi.load(bd)
                    errs = []
                    if doc.get("golden") is not None:
                        errs += validate_spec(doc["golden"])
                    for inp in doc["inputs"]:
                        if "golden" in inp:
                            errs += validate_spec(inp["golden"], f"input '{inp['id']}' golden")
                    print(f"{doc['benchmark']}: " + ("OK" if not errs else "; ".join(errs)))
                    rc = max(rc, 1 if errs else 0)
            return rc
        bd = Path(a.bench_dir).resolve(); root = hi.repo_root(bd); doc = hi.load(bd)
        if a.cmd == "make":
            ids = [i["id"] for i in doc["inputs"]] if a.all else a.input_ids
            if not ids:
                print("make: name input ids or --all"); return 2
            rc = 0
            for iid in ids:
                rc = max(rc, make_one(doc, hi.get_input(doc, iid), root, bd, a))
            return rc
        inp = hi.get_input(doc, a.input_id or doc["default_input"])
        res = {}
        try:
            if a.cmd == "check":
                rc = check(doc, inp, root, bd, Path(a.dump), a.log, res)
            else:
                rc = run(doc, inp, root, bd, a.exe, a.workdir, a.timeout, res)
        finally:
            if getattr(a, "json", None):
                res.setdefault("verdict", "ERROR")
                res["tool_sha256"] = hi.sha256_file(HERE / "hpcperf_golden.py")
                res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                Path(a.json).write_text(json.dumps(res, indent=2) + "\n")
        return rc
    except (hi.InputError, GoldenError) as ex:
        print(f"FAIL: golden check: {ex}" if a.cmd in ("check", "run") else f"ERROR: {ex}")
        return 1 if a.cmd in ("check", "run") else 2


if __name__ == "__main__":
    sys.exit(main())
