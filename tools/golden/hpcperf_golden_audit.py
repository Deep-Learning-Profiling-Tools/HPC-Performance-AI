#!/usr/bin/env python3
"""Read-only audit of the correctness mechanisms of the suite: do ctest, the registry check (inputs.yaml `check:`),
the frozen golden references and (Level 2/3) validate.sh agree, and how strong is each comparison?

    hpcperf_golden_audit.py [--root DIR] [--level 1,2,3] [--bench NAME|level1/NAME ...] [--json FILE]
                            [--table FILE|-] [--no-ctest] [--build-root DIR] [--aggregate-build DIR] [--strict]

Nothing is built, run or written except the --json / --table files: ctest is only asked for its test list
(`ctest --test-dir <build> --show-only=json-v1`, parsed as tools/timing/gen_cases.py does), git only for history,
the golden store for the presence of blobs and (pairwise check, --no-pairwise to skip) for the golden-vs-golden
comparison of inputs whose outputs have the same shape. Every finding has a severity:

  HARD    an inconsistency that makes a correctness verdict wrong or impossible: a reference whose frozen spec /
          workload / command no longer matches the registry, a golden input without a committed reference and
          without a recorded reason, a ctest of a golden benchmark that does not run the golden check of the
          default input, a registry check of a golden input that is not the golden check, a pass regex that
          matches text proving no comparison (e.g. SKIP_VERIFY), a recorded tol_effective that does not follow
          from tol and d0, a reference made from a commit that is not an ancestor of HEAD or from a dirty tree.
          Exit 1 when there is at least one.
  WARN    a check that can pass a wrong result or that is not judged by text proving a comparison (an all-zero or
          constant output within the tolerance, another registered input's golden within this input's tolerance, an exit-code-only ctest, a generic pass word, a stale build tree,
          a missing golden blob, a timed case without a reference). Exit 1 with --strict.
  REVIEW  needs a person: sources changed since the reference was frozen, d0 using more than half of the
          tolerance, a tolerance within a few ulp of the signal or below the printed resolution, scalar-only
          references, a reason recorded only in a YAML comment.
  INFO    facts (mechanism, rebuilt binary, inputs.yaml edited since freezing, ...).

The registry, the golden spec and the frozen-spec comparison are read through tools/inputs/hpcperf_inputs.py and
tools/golden/hpcperf_golden.py (spec_of, frozen_spec, validate_spec, tol_effective, HARNESS): no rule of those tools
is re-implemented here. Exit 0 = no HARD finding (and with --strict no WARN), 1 = findings, 2 = usage / tool error.
"""
import argparse, glob, hashlib, json, math, os, re, shlex, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "inputs"))
sys.path.insert(0, str(HERE.parent / "timing"))
import hpcperf_golden as hg  # noqa: E402  (spec_of, frozen_spec, validate_spec, tol_effective, HARNESS, store_root)
import hpcperf_inputs as hi  # noqa: E402  (load, validate, check_of, workload_identity, workload_mismatch, is_probe)

try:
    import gen_cases as gc_cases  # noqa: E402  (resolve_ctest, ctest_show: how the timing tool reads ctest)
except Exception:  # pragma: no cover - tools/timing absent in a stripped checkout
    gc_cases = None

SCHEMA = "hpcperf-golden-audit-1"
SEVERITIES = ("HARD", "WARN", "REVIEW", "INFO")
EPS = {"f32": 2.0 ** -23, "f64": 2.0 ** -52}

# text a benchmark or a wrapper prints without having compared anything: a pass regex matching one of these is HARD
NONPROOF_SKIP_LINES = ("SKIP_VERIFY", "HPCPERF_SKIP_VERIFY=1: verification skipped", "verification skipped",
                       "Verification skipped", "SKIPPED", "skipping the CPU reference", "CPU reference: skipped (default)",
                       "   WARNING: the reference level1/x/reference/y.json is untracked", "   ok   y: exact",
                       "FAIL: golden check x/y: no frozen reference")
# generic success words: a pass regex matching one of these proves a comparison only if the program prints it
# exclusively after one (read the source): WARN
NONPROOF_GENERIC_LINES = ("PASS", "Passed", "PASSED", "OK", "ok", "Success", "SUCCESS", "Test passed", "Done", "done",
                          "Verification: PASS")
GOLDEN_PASS_LINE = "PASS: golden check bench/input (1 outputs match the frozen reference made 2026-01-01T00:00:00Z on NVIDIA B200)"
GOLDEN_FAIL_LINE = "FAIL: golden check bench/input: 1 of 1 outputs differ from the frozen B200 reference: y: differs"
VERIFY_FLAGS = ("-v", "--verify", "-verify", "--validate", "-validate", "--check", "-check")
SHARED_HEADERS = ("tools/golden/hpcperf_golden.h", "tools/timing/roi/hpcperf_roi.h")
NO_GOLDEN_COMMENT = re.compile(r"no golden|not frozen|no validated reference|golden.*(pending|refused)", re.I)
SHA40 = re.compile(r"^[0-9a-f]{40}$")


# ----------------------------------------------------------------------------- small helpers
def git(root, *a):
    p = subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True)
    return p.returncode, p.stdout


def rel(p, root):
    try:
        return os.path.relpath(str(p), str(root))
    except ValueError:
        return str(p)


def fnum(x, fmt=".3g"):
    if x is None:
        return "-"
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return str(x)
    return format(x, fmt)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def regex_search(rx, line):
    try:
        return re.search(rx, line) is not None
    except re.error:
        return False


def printed_quanta(strs):
    """[(text, value, one unit of the last printed digit)] of numbers printed in a log. A fixed-point format (every
    value without exponent and with the same number of decimals) has the quantum 10^-decimals; otherwise (%g / %e,
    where %g strips trailing zeros) the precision is the largest number of significant digits seen, p, and the
    quantum of v is 10^(floor(log10|v|) - p + 1)."""
    parsed = []
    for s in strs:
        mm = re.fullmatch(r"\s*[-+]?(\d*)(?:\.(\d*))?(?:[eE]([-+]?\d+))?\s*", str(s))
        if not mm or not ((mm.group(1) or "") + (mm.group(2) or "")):
            continue
        digits = ((mm.group(1) or "") + (mm.group(2) or "")).lstrip("0")
        parsed.append((str(s).strip(), float(s), len(mm.group(2) or ""), mm.group(3) is not None, max(len(digits), 1)))
    if not parsed:
        return []
    decs = {p[2] for p in parsed}
    fixed = not any(p[3] for p in parsed) and len(decs) == 1 and any((p[0].split(".")[-1].endswith("0")) for p in parsed if "." in p[0])
    if fixed:
        d = decs.pop()
        return [(s, v, 10.0 ** -d) for s, v, _, _, _ in parsed]
    prec = max(p[4] for p in parsed)
    out = []
    for s, v, _, _, _ in parsed:
        e = math.floor(math.log10(abs(v))) if v else 0
        out.append((s, v, 10.0 ** (e - prec + 1)))
    return out


def pass_regex_quality(rx, golden=False):
    """(severity, why) of a pass regex: HARD when it matches text that proves no comparison (a skip marker), WARN
    when it is a generic success word, None when it is specific. For the golden check: HARD unless it matches the
    golden PASS line and not the FAIL line."""
    if rx is None or rx == "":
        return "WARN", "no pass regex: the verdict is the exit code (and the absence of a fail line) only"
    try:
        re.compile(rx)
    except re.error as ex:
        return "HARD", f"the pass regex does not compile ({ex})"
    skip = [l for l in NONPROOF_SKIP_LINES if regex_search(rx, l)]
    if skip:
        return "HARD", f"the pass regex {rx!r} matches text that proves no comparison: {skip[0]!r}"
    if golden:
        if not regex_search(rx, GOLDEN_PASS_LINE):
            return "HARD", f"the pass regex {rx!r} does not match the golden check's PASS line (the check can never pass)"
        if regex_search(rx, GOLDEN_FAIL_LINE):
            return "HARD", f"the pass regex {rx!r} also matches the golden FAIL line"
    gen = [l for l in NONPROOF_GENERIC_LINES if regex_search(rx, l)]
    if gen:
        return "WARN", (f"the pass regex {rx!r} matches a bare success word ({gen[0]!r}): it proves a comparison only if the "
                        f"program prints it exclusively after one (read the source)")
    return None, None


# ----------------------------------------------------------------------------- the audit
class Audit:
    def __init__(self, root: Path, opts):
        self.root = Path(root).resolve()
        self.opts = opts
        self.findings = []
        self.benchmarks = []
        self.references = []
        rc, out = git(self.root, "rev-parse", "HEAD")
        self.head = out.strip() if rc == 0 else None
        rc, out = git(self.root, "ls-files", "-z")
        self.tracked = set(out.split("\0")) if rc == 0 else set()
        self.store = hg.store_root(self.root)
        self._anc = {}
        self._ctest = None
        self.shared_changes = {}
        self.pairwise_rows = []
        self.timing_rows = self._timing_rows()

    # -- findings
    def add(self, sev, code, bench, msg, input_id=None, path=None, data=None):
        assert sev in SEVERITIES
        f = {"severity": sev, "code": code, "benchmark": bench, "input_id": input_id, "path": path, "message": msg}
        if data is not None:
            f["data"] = data
        self.findings.append(f)
        return f

    # -- git helpers
    def is_ancestor(self, commit):
        if commit not in self._anc:
            if not commit or not SHA40.match(commit) or not self.head:
                self._anc[commit] = None
            else:
                p = subprocess.run(["git", "-C", str(self.root), "merge-base", "--is-ancestor", commit, "HEAD"],
                                   capture_output=True, text=True)
                self._anc[commit] = True if p.returncode == 0 else (False if p.returncode == 1 else None)
        return self._anc[commit]

    def committed_clean(self, relpath):
        if relpath not in self.tracked:
            return False, "untracked"
        p = subprocess.run(["git", "-C", str(self.root), "diff", "--quiet", "HEAD", "--", relpath], capture_output=True)
        return (p.returncode == 0), ("modified relative to HEAD" if p.returncode else "committed")

    def numstat(self, commit, paths, worktree=False):
        """{file: (added, deleted)} changed between commit and HEAD (worktree=False) or between HEAD and the working
        tree (worktree=True; plus untracked files)."""
        out = {}
        if not paths:
            return out
        args = ["diff", "--numstat", "HEAD"] if worktree else ["diff", "--numstat", commit, "HEAD"]
        rc, txt = git(self.root, *args, "--", *paths)
        if rc == 0:
            for line in txt.splitlines():
                a, d, f = line.split("\t", 2)
                out[f] = (int(a) if a.isdigit() else None, int(d) if d.isdigit() else None)
        if worktree:
            rc, txt = git(self.root, "ls-files", "--others", "--exclude-standard", "--", *paths)
            for f in txt.splitlines():
                out.setdefault(f, (None, None))
        return out

    def commits_touching(self, commit, paths):
        rc, txt = git(self.root, "log", "--format=%h %s", f"{commit}..HEAD", "--", *paths)
        return txt.splitlines() if rc == 0 else []

    # -- discovery
    def bench_dirs(self):
        levels = self.opts.levels
        out = []
        for lv in levels:
            for d in sorted(glob.glob(str(self.root / f"level{lv}" / "*"))):
                if os.path.isfile(os.path.join(d, "inputs.yaml")):
                    out.append(Path(d))
        if self.opts.bench:
            want = set()
            for b in self.opts.bench:
                want.add(b.rstrip("/").split("/")[-1])
            out = [d for d in out if d.name in want]
        return out

    # -- ctest
    def ctest_exe(self, build_dir):
        if self.opts.ctest:
            return self.opts.ctest
        if gc_cases is None:
            return None
        try:
            return gc_cases.resolve_ctest(str(build_dir))[0]
        except Exception:
            return None

    def ctest_tests(self, build_dir):
        """[(test dict), ...] of `ctest --show-only=json-v1` for one build directory, or raises RuntimeError."""
        exe = self.ctest_exe(build_dir)
        if not exe:
            raise RuntimeError("no usable ctest (build tree CMAKE_CTEST_COMMAND, $HPCPERF_CTEST, PATH)")
        if gc_cases is not None and not self.opts.ctest:
            payload = gc_cases.ctest_show(exe, str(build_dir))
        else:
            p = subprocess.run([exe, "--test-dir", str(build_dir), "--show-only=json-v1"], capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError(f"ctest failed in {build_dir}: {p.stderr.strip()[-300:]}")
            s = p.stdout.find("{")
            if s < 0:
                raise RuntimeError(f"ctest produced no JSON in {build_dir}")
            payload = json.loads(p.stdout[s:])
        tests = []
        for t in payload.get("tests", []):
            props = {}
            for pr in t.get("properties", []) or []:
                props[pr.get("name")] = pr.get("value")
            tests.append({"name": t.get("name"), "command": [str(c) for c in t.get("command", []) or []], "properties": props})
        return tests

    @staticmethod
    def _as_list(v):
        if v is None:
            return []
        return v if isinstance(v, list) else [v]

    def parse_golden_ctest(self, cmd):
        """(bench_dir, input_id or None, exe, workdir) of `<py> .../hpcperf_golden.py run <bench_dir> [<input>] --exe X`,
        or None when the command is not a golden run."""
        for i, tok in enumerate(cmd):
            if os.path.basename(tok) == "hpcperf_golden.py":
                a = cmd[i + 1:]
                if not a or a[0] != "run" or len(a) < 2:
                    return {"bench_dir": None, "input_id": None, "exe": None, "sub": a[0] if a else None}
                bench = a[1]
                inp = a[2] if len(a) > 2 and not a[2].startswith("--") else None
                exe = a[a.index("--exe") + 1] if "--exe" in a and a.index("--exe") + 1 < len(a) else None
                wd = a[a.index("--workdir") + 1] if "--workdir" in a and a.index("--workdir") + 1 < len(a) else None
                return {"bench_dir": bench, "input_id": inp, "exe": exe, "workdir": wd, "sub": "run"}
        return None

    def static_cmake_tests(self, bench_dir):
        """What CMakeLists.txt itself declares (independent of a configured build tree): the add_test blocks and
        whether they run the golden check."""
        f = Path(bench_dir) / "CMakeLists.txt"
        if not f.is_file():
            return None
        text = f.read_text(errors="replace")
        blocks = []
        for m in re.finditer(r"add_test\s*\(", text):
            depth, j = 1, m.end()
            while j < len(text) and depth:
                depth += {"(": 1, ")": -1}.get(text[j], 0); j += 1
            blk = text[m.start():j]
            nm = re.search(r"NAME\s+(\S+)", blk)
            blocks.append({"name": nm.group(1) if nm else None,
                           "golden": "hpcperf_golden.py" in blk and re.search(r"\brun\b", blk) is not None,
                           "text": " ".join(blk.split())[:300]})
        return {"file": rel(f, self.root), "mtime": f.stat().st_mtime, "tests": blocks}

    def gen_cases_skips_probes(self):
        """Whether the audited repository's tools/timing/gen_cases.py filters correctness-probe ctest tests (static)."""
        f = self.root / "tools" / "timing" / "gen_cases.py"
        return (not f.is_file()) or ("probe" in f.read_text(errors="replace"))

    def audit_ctest(self, doc, bench_dir, golden_ids, rec):
        b = doc["benchmark"]
        inputs_by_id = {i.get("id"): i for i in doc.get("inputs") or []}
        default = doc.get("default_input")
        static = self.static_cmake_tests(bench_dir)
        rec["cmake_static"] = static
        builds = []
        if self.opts.no_ctest:
            rec["ctest"] = {"skipped": True}
            return
        per = self.root / self.opts.build_root / b / "cuda"
        agg = self.root / self.opts.aggregate_build / "level1" / b if self.opts.aggregate_build else None
        for kind, bd in (("per-benchmark", per), ("aggregate", agg)):
            if bd is None:
                continue
            entry = {"kind": kind, "build_dir": rel(bd, self.root), "tests": []}
            builds.append(entry)
            ctf = bd / "CTestTestfile.cmake"
            if not ctf.is_file():
                entry["missing"] = True
                if kind == "per-benchmark":
                    self.add("WARN", "CTEST_BUILD_MISSING", b, f"{rel(bd, self.root)} has no CTestTestfile.cmake: the ctest of "
                             f"this benchmark cannot be audited (configure it)", path=rel(bd, self.root))
                continue
            try:
                tests = self.ctest_tests(bd)
            except Exception as ex:  # noqa: BLE001
                entry["error"] = str(ex)
                self.add("WARN", "CTEST_UNAVAILABLE", b, f"ctest --show-only failed in {rel(bd, self.root)}: {ex}")
                continue
            if static and static["mtime"] > ctf.stat().st_mtime + 1:
                self.add("WARN", "CTEST_BUILD_STALE", b, f"{static['file']} is newer than {rel(ctf, self.root)}: the configured "
                         f"ctest may not be what CMakeLists.txt declares (re-run cmake)", path=rel(ctf, self.root))
            n_golden = n_default = 0
            for t in tests:
                cmd = t["command"]; props = t["properties"]
                pr = self._as_list(props.get("PASS_REGULAR_EXPRESSION"))
                fr = self._as_list(props.get("FAIL_REGULAR_EXPRESSION"))
                env = self._as_list(props.get("ENVIRONMENT"))
                g = self.parse_golden_ctest(cmd)
                wrapper = "golden" if g else ("python" if len(cmd) > 1 and cmd[1].endswith(".py") else "direct")
                trow = {"name": t["name"], "wrapper": wrapper, "command": [rel(c, self.root) if str(c).startswith(str(self.root)) else c for c in cmd],
                        "pass_regex": pr, "fail_regex": fr, "environment": env, "timeout": props.get("TIMEOUT"),
                        "will_fail": props.get("WILL_FAIL"), "disabled": props.get("DISABLED")}
                entry["tests"].append(trow)
                where = f"{kind} ctest {t['name']}"
                for e in env:
                    k = str(e).split("=", 1)[0]
                    if k in ("HPCPERF_GOLDEN_ALLOW_UNCOMMITTED_REFERENCE", "HPCPERF_SKIP_VERIFY", "HPCPERF_CPU_REFERENCE",
                             "HPCPERF_GOLDEN_STORE", "HPCPERF_RANDOM_SEED"):
                        sev = "HARD" if k in ("HPCPERF_GOLDEN_ALLOW_UNCOMMITTED_REFERENCE", "HPCPERF_SKIP_VERIFY") else "WARN"
                        self.add(sev, "CTEST_ENVIRONMENT", b, f"{where} sets {e} in its ENVIRONMENT property")
                if props.get("WILL_FAIL") in (True, "ON", "TRUE", "1", 1):
                    self.add("HARD", "CTEST_WILL_FAIL", b, f"{where} has WILL_FAIL (the verdict is inverted)")
                if props.get("DISABLED") in (True, "ON", "TRUE", "1", 1):
                    self.add("HARD" if golden_ids else "WARN", "CTEST_DISABLED", b, f"{where} is DISABLED")
                if g:
                    n_golden += 1
                    trow["golden"] = g
                    canon = (self.root / f"level{doc['level']}" / b).resolve()
                    if g.get("sub") != "run" or not g.get("bench_dir"):
                        self.add("HARD", "CTEST_NOT_GOLDEN_RUN", b, f"{where} calls hpcperf_golden.py {g.get('sub')}, not `run <bench_dir>`")
                        continue
                    if Path(g["bench_dir"]).resolve() != canon:
                        self.add("HARD", "CTEST_WRONG_BENCH_DIR", b, f"{where} runs the golden check of {g['bench_dir']}, not of the "
                                 f"canonical {rel(canon, self.root)}")
                    iid = g["input_id"] or default
                    trow["input_id"] = iid
                    if iid == default:
                        n_default += 1
                    elif iid in inputs_by_id and hi.is_probe(inputs_by_id[iid]):
                        trow["probe"] = True
                        self.add("INFO", "CTEST_PROBE_TEST", b, f"{where} also checks the correctness probe '{iid}'")
                        if not self.gen_cases_skips_probes():
                            self.add("WARN", "CTEST_PROBE_WOULD_BE_TIMED", b, f"{where} checks the correctness probe '{iid}', but "
                                     f"tools/timing/gen_cases.py makes one timing case of EVERY ctest test (it has no probe filter): "
                                     f"regenerating cases/level1.tsv would time the probe, which a probe must never be")
                    else:
                        self.add("WARN", "CTEST_EXTRA_INPUT", b, f"{where} checks the performance input '{iid}', not the default "
                                 f"'{default}' (tools/timing/gen_cases.py makes it a timing case of its own)")
                    if iid not in golden_ids:
                        self.add("HARD", "CTEST_INPUT_NOT_GOLDEN", b, f"{where} checks input '{iid}', which has no golden spec")
                    if not g.get("exe"):
                        self.add("HARD", "CTEST_NO_EXE", b, f"{where} has no --exe: it would run the registry's binary, not this build's")
                    else:
                        reg_exe = self.root / doc["entry"]["path"]
                        if kind == "per-benchmark" and Path(g["exe"]).resolve() != reg_exe.resolve():
                            self.add("HARD", "CTEST_EXE_MISMATCH", b, f"{where} runs {rel(g['exe'], self.root)}, the registry entry is "
                                     f"{doc['entry']['path']}")
                        elif kind == "aggregate" and os.path.basename(g["exe"]) != os.path.basename(doc["entry"]["path"]):
                            self.add("HARD", "CTEST_EXE_MISMATCH", b, f"{where} runs {rel(g['exe'], self.root)} (another target than "
                                     f"the registry's {os.path.basename(doc['entry']['path'])})")
                    if pr != ["PASS: golden check"]:
                        sev, why = pass_regex_quality(pr[0] if pr else None, golden=True)
                        self.add(sev or "WARN", "CTEST_GOLDEN_REGEX", b, f"{where}: PASS_REGULAR_EXPRESSION {pr} is not the golden "
                                 f"check's 'PASS: golden check'" + (f" ({why})" if why else ""))
                    if not any(regex_search(x, GOLDEN_FAIL_LINE) for x in fr):
                        self.add("HARD", "CTEST_GOLDEN_REGEX", b, f"{where}: no FAIL_REGULAR_EXPRESSION matching 'FAIL: golden check' "
                                 f"(ctest ignores the exit code when a PASS regex is set)")
                    continue
                # a non-golden test
                if golden_ids:
                    self.add("WARN", "CTEST_EXTRA_NON_GOLDEN", b, f"{where} of a golden benchmark does not run the golden check: "
                             f"{' '.join(trow['command'])[:200]}")
                sev, why = pass_regex_quality(pr[0] if pr else None)
                if len(pr) > 1:
                    for x in pr[1:]:
                        s2, w2 = pass_regex_quality(x)
                        if s2 and (sev is None or SEVERITIES.index(s2) < SEVERITIES.index(sev)):
                            sev, why = s2, w2
                if sev:
                    code = "CTEST_EXIT_CODE_ONLY" if not pr else "CTEST_PASS_REGEX"
                    extra = ""
                    if not pr:
                        extra = (" -- a run that skips its verification (e.g. HPCPERF_SKIP_VERIFY=1 inherited from the caller's "
                                 "shell) and exits 0 passes")
                    self.add(sev, code, b, f"{where}: {why}{extra}")
                flags = [a for a in cmd[1:] if a in VERIFY_FLAGS]
                if flags:
                    trow["verify_flags"] = flags
                # the registry check's own checker script (non-golden): does ctest judge the default input the same way?
                dblk = hi.check_of(doc, inputs_by_id.get(default)) if default in inputs_by_id else None
                if not golden_ids and isinstance(dblk, dict) and dblk.get("kind") in ("post_run", "standalone"):
                    scripts = sorted({os.path.basename(str(x)) for x in dblk.get("command") or []
                                      if str(x).endswith((".py", ".sh")) and "{bench_dir}" in str(x)})
                    used = {os.path.basename(c) for c in cmd}
                    for c in cmd:                      # a ctest wrapper that imports / calls the registry's checker
                        if str(c).endswith((".py", ".sh")) and os.path.isfile(c):
                            txt = Path(c).read_text(errors="replace")
                            used |= {sc for sc in scripts if sc in txt or re.search(rf"\bimport\s+{re.escape(Path(sc).stem)}\b", txt)}
                    if scripts and not (set(scripts) & used):
                        self.add("WARN", "CTEST_REGISTRY_DIVERGE", b, f"{where} judges the default input with "
                                 f"`{' '.join(os.path.basename(c) for c in cmd[:3])}`, the registry check with {', '.join(scripts)} "
                                 f"(pass {dblk.get('pass_regex')!r}): the two correctness verdicts are not the same comparison")
            if golden_ids and n_golden == 0:
                self.add("HARD", "CTEST_NOT_GOLDEN", b, f"{kind} ctest ({rel(bd, self.root)}) of a golden benchmark runs no "
                         f"`hpcperf_golden.py run`: the default input is never compared with its frozen reference by ctest")
            elif golden_ids and n_default == 0:
                self.add("HARD", "CTEST_DEFAULT_NOT_CHECKED", b, f"no {kind} ctest test ({rel(bd, self.root)}) runs the golden check "
                         f"of the default input '{default}'")
        if static:
            sg = any(t["golden"] for t in static["tests"])
            for e in builds:
                if e.get("tests"):
                    bg = any(t["wrapper"] == "golden" for t in e["tests"])
                    if sg != bg:
                        self.add("WARN", "CTEST_BUILD_STALE", b, f"CMakeLists.txt {'runs' if sg else 'does not run'} the golden check "
                                 f"but the configured {e['kind']} build tree {e['build_dir']} {'does' if bg else 'does not'} (re-run cmake)")
            if golden_ids and not sg:
                self.add("HARD", "CTEST_NOT_GOLDEN", b, f"{static['file']} declares no add_test that runs hpcperf_golden.py run")
        rec["ctest"] = builds

    # -- registry check of one input
    def audit_check_block(self, doc, inp, spec, rec):
        b = doc["benchmark"]; iid = inp["id"]
        blk = hi.check_of(doc, inp)
        info = {"kind": None}
        rec["check"] = info
        if blk is None:
            info["kind"] = None
        elif isinstance(blk, dict):
            info.update({"kind": blk.get("kind"), "command": blk.get("command"), "pass_regex": blk.get("pass_regex"),
                         "fail_regex": blk.get("fail_regex"), "covers": blk.get("covers"), "run_env": blk.get("run_env"),
                         "reason": blk.get("reason"), "own": "check" in inp})
        golden = spec is not None and not spec.get("unsupported")
        if golden:
            if not isinstance(blk, dict) or blk.get("kind") in (None, "none"):
                self.add("HARD", "CHECK_MISSING", b, "a golden input whose registry check is " +
                         ("absent" if blk is None else f"kind none ({(blk or {}).get('reason')})") +
                         ": `hpcperf_inputs.py check` never compares it with its frozen reference", input_id=iid)
                return
            cmd = [str(t) for t in (blk.get("command") or [])]
            idx = next((i for i, t in enumerate(cmd) if os.path.basename(t) == "hpcperf_golden.py"), None)
            problems = []
            if blk.get("kind") != "post_run":
                problems.append(f"kind is {blk.get('kind')}, not post_run")
            if idx is None:
                problems.append("the command does not call tools/golden/hpcperf_golden.py")
            else:
                a = cmd[idx + 1:]
                if cmd[idx] != "{repo}/tools/golden/hpcperf_golden.py":
                    self.add("WARN", "CHECK_TOOL_PATH", b, f"the check calls {cmd[idx]} instead of {{repo}}/tools/golden/hpcperf_golden.py "
                             f"(the canonical tool)", input_id=iid)
                if a[:1] != ["check"]:
                    problems.append(f"it calls hpcperf_golden.py {a[:1]}, not check")
                if a[1:3] != ["{bench_dir}", "{input_id}"]:
                    problems.append(f"its bench_dir / input_id arguments are {a[1:3]} instead of {{bench_dir}} {{input_id}} "
                                    f"(another benchmark's or input's reference)")
                dump = a[a.index("--dump") + 1] if "--dump" in a and a.index("--dump") + 1 < len(a) else None
                if dump is None:
                    problems.append("no --dump")
                has_data = any(not o.get("log") for o in spec.get("outputs") or [])
                has_log = any(o.get("log") for o in spec.get("outputs") or [])
                run_env = blk.get("run_env") or {}
                if has_data and run_env.get("HPCPERF_GOLDEN_DUMP") != dump:
                    problems.append(f"run_env HPCPERF_GOLDEN_DUMP {run_env.get('HPCPERF_GOLDEN_DUMP')!r} is not the --dump "
                                    f"directory {dump!r}: the run captures nothing that the check reads")
                if has_log and ("--log" not in a or a[a.index("--log") + 1:a.index("--log") + 2] != ["{log}"]):
                    problems.append("outputs are read from the log but the check gets no --log {log}")
                mode = spec.get("mode", "data")
                tol_out = any(o.get("rule", {}).get("metric") != "exact" for o in spec.get("outputs") or [])
                if run_env.get("HPCPERF_GOLDEN_MODE") == "hash" and tol_out:
                    problems.append("run_env sets HPCPERF_GOLDEN_MODE=hash but a tolerance output needs the bytes")
                if mode == "hash" and has_data and run_env.get("HPCPERF_GOLDEN_MODE") != "hash":
                    self.add("WARN", "CHECK_MODE_NOT_HASH", b, "the spec is mode hash but the registry check's run_env does not set "
                             "HPCPERF_GOLDEN_MODE=hash: the check run writes every byte of the outputs to disk", input_id=iid)
            sev, why = pass_regex_quality(blk.get("pass_regex"), golden=True)
            if sev == "HARD":
                problems.append(why)
            elif sev:
                self.add(sev, "CHECK_PASS_REGEX", b, why, input_id=iid)
            if not blk.get("fail_regex") or not regex_search(blk["fail_regex"], GOLDEN_FAIL_LINE):
                self.add("WARN", "CHECK_FAIL_REGEX", b, f"the check's fail_regex {blk.get('fail_regex')!r} does not match the golden "
                         f"FAIL line", input_id=iid)
            if blk.get("covers") != "all":
                self.add("INFO", "CHECK_COVERS", b, f"the golden check covers {blk.get('covers')!r}, not all", input_id=iid)
            if problems:
                self.add("HARD", "CHECK_NOT_GOLDEN", b, "the registry check of a golden input is not the golden post_run check: "
                         + "; ".join(problems), input_id=iid)
            info["golden"] = not problems
            return
        # a non-golden input
        if blk is None:
            req = [q for q in hi.quantities(doc, inp) if q.get("role", "required") == "required"
                   and (q.get("compare") or {}).get("rule") in hi.VERIFYING_RULES]
            info["baseline_quantities"] = [f"{q['name']}:{q['compare']['rule']}" for q in req]
            if not req:
                self.add("WARN", "NO_CORRECTNESS_CHECK", b, "no registry check and no required verifying baseline quantity",
                         input_id=iid)
            return
        if blk.get("kind") == "none":
            return
        sev, why = pass_regex_quality(blk.get("pass_regex"))
        if sev:
            self.add(sev, "CHECK_PASS_REGEX", b, why, input_id=iid)

    # -- references
    def recorded_no_golden_reason(self, doc, iid):
        """A reason for a missing reference recorded only as a YAML comment inside the input's block."""
        try:
            lines = Path(doc["_path"]).read_text().splitlines()
        except OSError:
            return None
        start = None
        for i, l in enumerate(lines):
            if re.match(r"^\s*-\s+id:\s*['\"]?" + re.escape(iid) + r"['\"]?\s*$", l):
                start = i; break
        if start is None:
            return None
        com = []
        for l in lines[start + 1:]:
            if re.match(r"^\s*-\s+id:", l) or re.match(r"^\S", l):
                break
            s = l.strip()
            if s.startswith("#"):
                com.append(s.lstrip("# ").rstrip())
        txt = " ".join(com)
        return txt if com and NO_GOLDEN_COMMENT.search(txt) else None

    def verify_flag_in_source(self, bench_dir, flags):
        """'file:line' where the benchmark's own source turns a verification on for one of these flags (getopt
        `case 'v': ... verif...` or a string compare with the flag next to 'verif'), or None."""
        for sub in ("cuda", "common"):
            d = Path(bench_dir) / sub
            if not d.is_dir():
                continue
            for f in sorted(d.rglob("*")):
                if not f.is_file() or f.suffix not in (".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".cuh"):
                    continue
                lines = f.read_text(errors="replace").splitlines()
                for i, l in enumerate(lines):
                    for fl in flags:
                        letter = fl.lstrip("-")
                        if (len(letter) == 1 and re.search(rf"case\s*'{re.escape(letter)}'\s*:", l)) or f'"{fl}"' in l:
                            window = " ".join(lines[i:i + 3])
                            if re.search(r"verif", window, re.I):
                                # the flag's verification is itself behind the CPU-reference switch (off by default)
                                gated = "hpcperf_cpu_reference_enabled" in "\n".join(lines)
                                return f"{rel(f, self.root)}:{i + 1}", gated
        return None, False

    def source_paths(self, doc, bench_dir):
        """Repository-relative paths whose change can change the frozen output: the benchmark's sources (Level 1:
        cuda/ common/ hip/ + CMakeLists.txt; Level 2/3: every file except the harness, hpcperf_golden.HARNESS) and the
        shared capture/ROI headers."""
        rb = rel(bench_dir, self.root)
        if doc["level"] == 1:
            paths = [f"{rb}/{s}" for s in ("cuda", "common", "hip", "CMakeLists.txt") if (Path(bench_dir) / s).exists()]
        else:
            paths = [rb]
        return paths + list(SHARED_HEADERS)

    def filter_sources(self, doc, bench_dir, files):
        rb = rel(bench_dir, self.root) + "/"
        out = {}
        for f, v in files.items():
            if f.startswith(rb) and doc["level"] != 1 and hg.HARNESS.search(f[len(rb):]):
                continue
            out[f] = v
        return out

    def audit_reference(self, doc, bench_dir, rp, inputs_by_id, rec_b):
        b = doc["benchmark"]
        rrel = rel(rp, self.root)
        try:
            ref = json.loads(Path(rp).read_text())
        except (OSError, json.JSONDecodeError) as ex:
            self.add("HARD", "REFERENCE_UNREADABLE", b, f"{rrel}: {ex}", path=rrel)
            return None
        if not isinstance(ref, dict) or ref.get("schema") != hg.REF_SCHEMA:
            return self.audit_app_reference(doc, bench_dir, rp, ref, inputs_by_id)
        iid = Path(rp).stem
        row = {"benchmark": b, "level": doc["level"], "input_id": iid, "path": rrel, "schema": hg.REF_SCHEMA,
               "outputs": [], "problems": []}
        self.references.append(row)

        def hard(code, msg, data=None):
            row["problems"].append(code)
            self.add("HARD", code, b, msg, input_id=iid, path=rrel, data=data)
        ok, why = self.committed_clean(rrel)
        row["committed"] = ok
        if not ok:
            hard("REFERENCE_UNCOMMITTED", f"{rrel} is {why}: hpcperf_golden.py check FAILs it (a reference counts only "
                 f"committed and unmodified)")
        if ref.get("input_id") != iid or ref.get("benchmark") != b or ref.get("level") != doc["level"]:
            hard("REFERENCE_IDENTITY", f"{rrel} names benchmark {ref.get('benchmark')!r} level {ref.get('level')!r} input "
                 f"{ref.get('input_id')!r}")
        inp = inputs_by_id.get(iid)
        if inp is None:
            hard("REFERENCE_ORPHAN", f"{rrel}: input '{iid}' is not registered in inputs.yaml")
            return row
        row["purpose"] = inp.get("purpose", "performance")
        spec = hg.spec_of(doc, inp)
        if spec is None or spec.get("unsupported"):
            hard("REFERENCE_WITHOUT_SPEC", f"{rrel}: the input has no golden spec" + (f" ({spec['unsupported']})" if spec else ""))
            return row
        fz = hg.frozen_spec(spec)
        if ref.get("spec") != fz:
            diff = []
            rs = ref.get("spec") or {}
            if rs.get("mode") != fz["mode"]:
                diff.append(f"mode {rs.get('mode')} -> {fz['mode']}")
            ro = {(o.get("name") or o.get("match")): o for o in rs.get("outputs") or []}
            co = {(o.get("name") or o.get("match")): o for o in fz["outputs"]}
            for k in sorted(set(ro) | set(co)):
                if ro.get(k) != co.get(k):
                    keys = sorted(kk for kk in set((ro.get(k) or {})) | set((co.get(k) or {}))
                                  if (ro.get(k) or {}).get(kk) != (co.get(k) or {}).get(kk))
                    diff.append(f"output {k}: {', '.join(keys) if ro.get(k) and co.get(k) else ('added' if k in co else 'removed')}")
            hard("REFERENCE_SPEC_MISMATCH", f"{rrel}: the frozen spec differs from inputs.yaml's golden spec ({'; '.join(diff)}): "
                 f"check FAILs every run until an explicit make --replace", data=diff)
        wl = hi.workload_identity(doc, inp)
        mism = hi.workload_mismatch(ref.get("workload") or {}, wl)
        if mism:
            hard("REFERENCE_WORKLOAD_MISMATCH", f"{rrel}: the reference was made for another workload ({', '.join(mism)}): "
                 f"reference {json.dumps({k: (ref.get('workload') or {}).get(k) for k in mism})} vs registry "
                 f"{json.dumps({k: wl.get(k) for k in mism})}")
        prov = ref.get("provenance") or {}
        row["made_utc"] = prov.get("made_utc"); row["git_head"] = prov.get("git_head")
        row["gpu"] = (prov.get("gpu") or {}).get("name")
        # the command that made it: the registry's binary with the registry's arguments
        cc = spec.get("cpu_check") or {}
        entry = doc["entry"]
        cmdp = [str(c) for c in prov.get("command") or []]
        if entry["kind"] == "binary" and not inp.get("binary"):
            exp_bin = entry["path"]
            if prov.get("binary") != exp_bin:
                hard("REFERENCE_COMMAND_MISMATCH", f"{rrel}: made with binary {prov.get('binary')}, the registry entry is {exp_bin}")
            if not cc.get("command"):
                args = [str(a) for a in inp.get("args") or []]
                if cmdp[:1] != [exp_bin] or cmdp[1:] != args:
                    hard("REFERENCE_COMMAND_MISMATCH", f"{rrel}: made with `{' '.join(cmdp)}`, the registry runs "
                         f"`{' '.join([exp_bin] + args)}`")
        elif inp.get("binary"):
            if prov.get("binary") != inp["binary"]:
                hard("REFERENCE_COMMAND_MISMATCH", f"{rrel}: made with binary {prov.get('binary')}, the input's binary is {inp['binary']}")
        else:
            if spec.get("binary") and prov.get("binary") != spec["binary"]:
                hard("REFERENCE_COMMAND_MISMATCH", f"{rrel}: made with binary {prov.get('binary')}, golden.binary is {spec['binary']}")
            exp = ["bash", entry["path"]] + ([entry["backend_arg"]] if entry.get("backend_arg") else [])
            gr = (prov.get("gpu_run") or {}).get("command") if cc.get("after_run") else cmdp
            if gr is not None and [str(x) for x in gr] != exp:
                hard("REFERENCE_COMMAND_MISMATCH", f"{rrel}: the frozen GPU run was `{' '.join(map(str, gr))}`, the registry runs "
                     f"`{' '.join(exp)}` (with {doc.get('selector')}={iid})")
        # provenance
        anc = self.is_ancestor(prov.get("git_head"))
        row["git_head_ancestor"] = anc
        if anc is False:
            hard("REFERENCE_NOT_ANCESTOR", f"{rrel}: provenance git_head {prov.get('git_head')} is not an ancestor of HEAD "
                 f"(made on another branch / rewritten history)")
        elif anc is None:
            hard("REFERENCE_UNKNOWN_COMMIT", f"{rrel}: provenance git_head {prov.get('git_head')!r} is not a commit of this repository")
        if prov.get("source_clean") is not True:
            hard("REFERENCE_SOURCE_DIRTY", f"{rrel}: made from a tree with uncommitted changes (source_clean "
                 f"{prov.get('source_clean')!r})")
        if ref.get("replaces"):
            self.add("INFO", "REFERENCE_REPLACED", b, f"{rrel} replaced an earlier reference: {ref['replaces'].get('reason')}",
                     input_id=iid, path=rrel)
        if ref.get("inputs_yaml_sha256") != doc["_sha256"]:
            row["inputs_yaml_changed"] = True
        tool = (prov.get("tool_sha256") or {}).get("hpcperf_golden.py")
        row["tool_sha256"] = tool
        # sources changed since the reference's commit (the shared capture / ROI headers are reported once, in run())
        if anc:
            paths = self.source_paths(doc, bench_dir)
            ch = self.filter_sources(doc, bench_dir, self.numstat(prov["git_head"], paths))
            wt = self.filter_sources(doc, bench_dir, self.numstat(None, paths, worktree=True))
            shared = {f: v for f, v in ch.items() if f in SHARED_HEADERS}
            ch = {f: v for f, v in ch.items() if f not in SHARED_HEADERS}
            row["shared_changed"] = sorted(shared)
            for f in shared:
                self.shared_changes.setdefault(f, {"refs": [], "commits": set()})
                self.shared_changes[f]["refs"].append(rrel)
                self.shared_changes[f]["commits"].update(self.commits_touching(prov["git_head"], [f]))
            row["sources_changed"] = sorted(ch); row["sources_uncommitted"] = sorted(wt)
            if ch:
                nonhip = [f for f in ch if "/hip/" not in f]
                commits = self.commits_touching(prov["git_head"], sorted(ch))
                row["source_commits"] = commits
                stat = ", ".join(f"{f} (+{a}/-{d})" for f, (a, d) in sorted(ch.items()))
                self.add("REVIEW" if nonhip else "INFO", "REFERENCE_SOURCES_CHANGED", b,
                         f"{rrel}: {len(ch)} source file(s) changed since the reference's commit {prov['git_head'][:10]}: {stat}; "
                         f"commits: {'; '.join(commits[:6])}" + ("" if nonhip else " (HIP sources only: the CUDA golden is unaffected)"),
                         input_id=iid, path=rrel, data={"files": ch, "commits": commits})
            if wt:
                self.add("REVIEW", "REFERENCE_SOURCES_UNCOMMITTED", b, f"{rrel}: uncommitted changes to {len(wt)} source file(s) "
                         f"in the working tree: {', '.join(sorted(wt)[:6])}", input_id=iid, path=rrel)
        # the binary that made it vs the canonical binary now
        binp = self.root / (prov.get("binary") or "")
        if prov.get("binary") and binp.is_file() and self.opts.hash_binaries:
            cur = sha256_file(binp)
            row["binary_rebuilt"] = cur != prov.get("binary_sha256")
        # outputs
        self.audit_ref_outputs(doc, inp, spec, ref, row, rrel)
        return row

    def audit_ref_outputs(self, doc, inp, spec, ref, row, rrel):
        b = doc["benchmark"]; iid = inp["id"]
        outs = ref.get("outputs") or []
        names = [o.get("name") for o in outs]
        # the spec's selection must be in the reference (check compares only what the reference lists)
        for o in spec.get("outputs") or []:
            if o.get("name"):
                if o["name"] not in names:
                    row["problems"].append("REFERENCE_OUTPUTS_MISMATCH")
                    self.add("HARD", "REFERENCE_OUTPUTS_MISMATCH", b, f"{rrel}: the spec's output '{o['name']}' is not in the reference",
                             input_id=iid, path=rrel)
            else:
                n = sum(1 for x in names if x and re.fullmatch(o["match"], x))
                if n != o.get("count"):
                    row["problems"].append("REFERENCE_OUTPUTS_MISMATCH")
                    self.add("HARD", "REFERENCE_OUTPUTS_MISMATCH", b, f"{rrel}: match '{o['match']}' selects {n} reference outputs, "
                             f"the spec says {o.get('count')}", input_id=iid, path=rrel)
        # an output name may carry several rules (e.g. rmse + a per-element max_abs gate): keep every spec rule per
        # name and match a reference output to the spec rule of the same metric
        spec_rules = {}
        for o in spec.get("outputs") or []:
            for x in names:
                if x and ((o.get("name") and o["name"] == x) or (o.get("match") and re.fullmatch(o["match"], x))):
                    spec_rules.setdefault(x, []).append(o)
        scalar_only = bool(outs)
        for e in outs:
            name = e.get("name"); r = dict(e.get("rule") or {}); st = e.get("stats") or {}
            orow = {"name": name, "dtype": e.get("dtype"), "count": e.get("count"), "metric": r.get("metric"),
                    "tol": r.get("tol"), "d0": r.get("d0_vs_cpu_reference"), "tol_effective": r.get("tol_effective"),
                    "min": st.get("min"), "max": st.get("max"), "nonfinite": st.get("nonfinite"), "flags": []}
            row["outputs"].append(orow)
            if (e.get("count") or 0) > 16:
                scalar_only = False
            cands = spec_rules.get(name) or []
            so = next((c for c in cands if (c.get("rule") or {}).get("metric") == r.get("metric")), cands[0] if cands else None)
            if so is not None:
                base = {k: v for k, v in (so.get("rule") or {}).items()}
                got = {k: r.get(k) for k in base}
                if got != base:
                    row["problems"].append("REFERENCE_RULE_MISMATCH")
                    self.add("HARD", "REFERENCE_RULE_MISMATCH", b, f"{rrel}: output {name}: rule {got} != spec {base}", input_id=iid, path=rrel)
            # blob presence (tolerance outputs only)
            if e.get("blob"):
                if not (self.store / e["blob"]).is_file():
                    orow["flags"].append("BLOB_MISSING")
                    self.add("WARN", "BLOB_MISSING", b, f"{rrel}: output {name}: golden blob {e['blob']} is not in the store "
                             f"{self.store} (check FAILs on this machine)", input_id=iid, path=rrel)
            elif r.get("metric") not in (None, "exact"):
                row["problems"].append("REFERENCE_NO_BLOB")
                self.add("HARD", "REFERENCE_NO_BLOB", b, f"{rrel}: tolerance output {name} records no blob", input_id=iid, path=rrel)
            m = r.get("metric")
            if m == "exact":
                if st.get("nonfinite"):
                    orow["flags"].append("NONFINITE_GOLDEN")
                    self.add("WARN", "NONFINITE_GOLDEN", b, f"{rrel}: exact output {name} holds {st['nonfinite']} non-finite value(s)",
                             input_id=iid, path=rrel)
                if (e.get("count") or 0) > 1 and st.get("min") is not None and st.get("min") == st.get("max"):
                    orow["flags"].append("GOLDEN_CONSTANT")
                    self.add("WARN", "GOLDEN_CONSTANT", b, f"{rrel}: output {name} ({e.get('dtype')}[{e.get('count')}]) is the constant "
                             f"{st['min']}: a program that writes that constant passes", input_id=iid, path=rrel)
                continue
            self.audit_tolerance(b, iid, rrel, e, r, st, orow, so, ref)
        if scalar_only:
            tot = sum(int(o.get("count") or 0) for o in outs)
            row["scalar_only"] = True
            self.add("REVIEW", "SCALAR_SUMMARY_ONLY", b, f"{rrel}: the reference holds only {tot} value(s) in {len(outs)} output(s) "
                     f"({', '.join(str(n) for n in names[:6])}{'...' if len(names) > 6 else ''}): the check compares a summary, not "
                     f"the computed field -- is a stronger comparison practical?", input_id=iid, path=rrel)

    def audit_tolerance(self, b, iid, rrel, e, r, st, orow, so, ref):
        m = r["metric"]; tol = r.get("tol"); d0 = r.get("d0_vs_cpu_reference"); te = r.get("tol_effective")
        name = e.get("name")

        def flag(sev, code, msg, data=None):
            orow["flags"].append(code)
            if sev == "HARD":
                self.references[-1]["problems"].append(code) if self.references else None
            self.add(sev, code, b, f"{rrel}: output {name} ({m}, tol {fnum(tol)}): {msg}", input_id=iid, path=rrel, data=data)
        if d0 is None:
            flag("HARD", "REFERENCE_D0_MISSING", "no measured d0 (make refuses this today): tol_effective = tol would be looser than "
                 "the upstream check against the CPU reference")
            return
        if m == "trunc_int":
            exp_te = (tol or 0) - d0
            exp_tf = max((r.get("frac_tol") or 0) - (r.get("d0_frac_vs_cpu_reference") or 0), 0.0)
            if te is None or abs(te - exp_te) > 1e-12 or abs((r.get("frac_tol_effective") or 0) - exp_tf) > 1e-12 * max(1, exp_tf):
                flag("HARD", "REFERENCE_TOL_EFFECTIVE_MISMATCH", f"recorded tol_effective {te} / frac {r.get('frac_tol_effective')} != "
                     f"{exp_te} / {exp_tf}")
            ratio = max(d0 / tol if tol else 0.0, (r.get("d0_frac_vs_cpu_reference") or 0) / r["frac_tol"] if r.get("frac_tol") else 0.0)
            orow["budget_used"] = ratio
            if ratio > 0.5:
                flag("REVIEW", "BUDGET_CONSUMED", f"d0 uses {ratio:.0%} of the tolerance (levels {d0}/{tol}, share "
                     f"{fnum(r.get('d0_frac_vs_cpu_reference'))}/{fnum(r.get('frac_tol'))})")
            mx = max(abs(st.get("min") or 0), abs(st.get("max") or 0)) if st else None
            if mx is not None and math.floor(mx) <= (te or 0):
                flag("WARN", "ZERO_OUTPUT_PASSES", f"every |pixel| <= {mx:g}: an all-zero image is within {te} levels")
            return
        try:
            exp_te = hg.tol_effective(m, tol, d0, r)
        except hg.GoldenError as ex:
            flag("HARD", "REFERENCE_TOL_EFFECTIVE_MISMATCH", str(ex)); return
        if te is None or not math.isclose(te, exp_te, rel_tol=1e-12, abs_tol=1e-300):
            flag("HARD", "REFERENCE_TOL_EFFECTIVE_MISMATCH", f"recorded tol_effective {te!r} != tol_effective(tol, d0) = {exp_te!r}")
            te = exp_te
        # budget used by the golden's own deviation from the CPU reference
        if m == "l2_rel_sq":
            used = math.sqrt(d0) / math.sqrt(tol) if tol else 0.0
        elif m == "max_relabs":
            used = d0 / 1.0
        else:
            used = d0 / tol if tol else 0.0
        orow["budget_used"] = used
        if used > 0.5:
            flag("REVIEW", "BUDGET_CONSUMED", f"d0 {fnum(d0)} uses {used:.0%} of the tolerance (tol_effective {fnum(te)}): a "
                 f"legitimate implementation on another architecture has less room than the upstream check gave the original")
        elif d0 == 0:
            orow["flags"].append("D0_ZERO")
        if not st or st.get("min") is None:
            return
        mn, mx = float(st["min"]), float(st["max"])
        M = max(abs(mn), abs(mx))
        lo = 0.0 if mn <= 0.0 <= mx else min(abs(mn), abs(mx))
        cnt = int(e.get("count") or 0)
        if cnt > 1 and mn == mx:
            flag("WARN", "GOLDEN_CONSTANT", f"every one of the {cnt} golden values is {mn:g}: a program that writes that constant "
                 f"(without computing anything) passes")
            if so and so.get("log"):
                self.audit_log_precision(b, iid, rrel, e, r, te, so, ref, orow, flag)
            return
        half = (mx - mn) / 2.0
        eps = EPS.get(e.get("dtype"), 2.0 ** -52)
        zero = const_hi = const_lo = None; rel_sig = None; ulps = None
        if m == "max_abs":
            zero = M; const_hi = const_lo = half
            rel_sig = te / M if M else math.inf
            ulps = te / (eps * M) if M else math.inf
        elif m == "rmse":
            zero_lo = abs(float(st.get("sum") or 0.0)) / cnt if cnt else 0.0
            zero = M if M <= te else (zero_lo if zero_lo > te else None)       # exact only at the bounds
            const_hi = half
            rel_sig = te / M if M else math.inf
            ulps = te / (eps * M) if M else math.inf
        elif m == "max_rel1p":
            zero = M / (1.0 + M); const_hi = half / (1.0 + lo); const_lo = half / (1.0 + M)
            rel_sig = te * (1.0 + M) / M if M else math.inf
            ulps = te / (eps * M / (1.0 + M)) if M else math.inf
        elif m == "max_rel":
            zero = 1.0 if M > 0 else 0.0
            const_hi = (half / lo) if lo > 0 else math.inf
            rel_sig = te; ulps = te / eps
        elif m == "max_relabs":
            rt, at = r.get("rtol") or 0.0, r.get("atol") or 0.0
            zero = M / (rt * M + at) if (rt * M + at) > 0 else math.inf
            den_lo = rt * lo + at
            const_hi = half / den_lo if den_lo > 0 else math.inf
            rel_sig = te * (rt * M + at) / M if M else math.inf
            ulps = te * (rt * M + at) / (eps * M) if M else math.inf
        elif m == "l1_rel":
            zero = 1.0 if M > 0 else 0.0
            rel_sig = te; ulps = te / eps
        elif m == "l2_rel_sq":
            zero = 1.0 if M > 0 else 0.0
            rel_sig = math.sqrt(te); ulps = math.sqrt(te) / eps
        orow.update({"abs_max": M, "rel_tol_vs_signal": rel_sig, "tol_in_ulps": ulps, "zero_metric": zero,
                     "constant_metric_upper": const_hi})
        if cnt <= 1:
            const_hi = None                    # a single value is always "a constant": SCALAR_SUMMARY_ONLY covers it
        if zero is not None and zero <= te:
            flag("WARN", "ZERO_OUTPUT_PASSES", f"an all-zero output scores {fnum(zero)} <= tol_effective {fnum(te)} (largest |golden| "
                 f"{fnum(M)}): a program that computes nothing passes")
        elif const_hi is not None and const_hi <= te:
            flag("WARN", "CONSTANT_OUTPUT_PASSES", f"the constant (min+max)/2 scores <= {fnum(const_hi)} <= tol_effective {fnum(te)} "
                 f"(golden range [{fnum(mn)}, {fnum(mx)}]): a program that writes one constant passes")
        elif rel_sig is not None and rel_sig > 0.1:
            flag("WARN", "TOL_LARGE_VS_SIGNAL", f"the tolerance allows a {rel_sig:.0%} error at the largest |golden| value {fnum(M)}")
        if ulps is not None and ulps < 16 and m != "max_rel":
            flag("REVIEW", "TOL_NEAR_ROUNDING", f"tol_effective {fnum(te)} is {ulps:.2g} {e.get('dtype')} ulp of the largest |golden| "
                 f"value {fnum(M)}: a legitimate reordering may fail")
        elif m == "max_rel" and te < 16 * eps:
            flag("REVIEW", "TOL_NEAR_ROUNDING", f"tol_effective {fnum(te)} is {te / eps:.2g} {e.get('dtype')} ulp (relative)")
        # log outputs: the printed resolution vs the tolerance
        if so and so.get("log"):
            self.audit_log_precision(b, iid, rrel, e, r, te, so, ref, orow, flag)

    def audit_log_precision(self, b, iid, rrel, e, r, te, so, ref, orow, flag):
        prov = ref.get("provenance") or {}
        logp = (prov.get("gpu_run") or {}).get("log") or (prov.get("cpu_validation") or {}).get("log")
        if not logp or not (self.root / logp).is_file():
            orow["log_precision"] = "log not available"
            return
        text = (self.root / logp).read_text(errors="replace")
        strs = []
        for mt in re.finditer(so["log"]["regex"], text, re.MULTILINE):
            strs += [mt.group("value")] if "value" in mt.re.groupindex else list(mt.groups())
        q = printed_quanta(strs)
        worst = 0.0; worst_s = None
        m = r["metric"]
        for s, v, qi in q:
            v = abs(v)
            if m == "max_rel":
                qv = qi / v if v else math.inf
            elif m == "max_rel1p":
                qv = qi / (1.0 + v)
            elif m == "max_relabs":
                den = (r.get("rtol") or 0) * v + (r.get("atol") or 0)
                qv = qi / den if den > 0 else math.inf
            else:
                qv = qi
            if qv > worst:
                worst, worst_s = qv, s
        orow["log_values"] = len(strs); orow["log_printed_unit_metric"] = worst
        if q and worst > te:
            flag("REVIEW", "TOL_BELOW_PRINTED_RESOLUTION", f"one unit of the last printed digit (at {worst_s!r}) scores {fnum(worst)}, "
                 f"above tol_effective {fnum(te)}: a candidate whose printed value differs in its last digit there fails "
                 f"(effectively exact on the printed digits)")
        elif q and worst > 0.5 * te:
            flag("REVIEW", "PRINTED_RESOLUTION_NEAR_TOL", f"one unit of the last printed digit (at {worst_s!r}) scores {fnum(worst)} = "
                 f"{worst / te:.0%} of tol_effective {fnum(te)}: the comparison is close to exact on the printed digits")

    def audit_app_reference(self, doc, bench_dir, rp, ref, inputs_by_id):
        """A reference with an application-specific schema (Level 2/3): only generic facts are checked."""
        b = doc["benchmark"]; rrel = rel(rp, self.root)
        row = {"benchmark": b, "level": doc["level"], "path": rrel, "schema": (ref or {}).get("schema") if isinstance(ref, dict) else None,
               "app_specific": True, "problems": []}
        self.references.append(row)
        ok, why = self.committed_clean(rrel)
        row["committed"] = ok
        if not ok:
            self.add("WARN", "APP_REFERENCE_UNCOMMITTED", b, f"{rrel} is {why}", path=rrel)
        key = None
        if isinstance(ref, dict):
            key = ref.get("input_id") or ref.get("key")
            if key is None and ref.get("case"):
                key = f"{ref.get('case')}.{ref.get('mode')}" if ref.get("mode") else ref.get("case")
        row["key"] = key
        stem = Path(rp).stem
        row["registered"] = stem in inputs_by_id
        if key is not None and key != stem and stem in inputs_by_id:
            self.add("WARN", "APP_REFERENCE_KEY", b, f"{rrel} records key {key!r} but is named {stem!r}", path=rrel)

        def walk(d, p=""):
            if isinstance(d, dict):
                for k, v in d.items():
                    yield from walk(v, f"{p}.{k}" if p else k)
            elif isinstance(d, list):
                for i, v in enumerate(d):
                    yield from walk(v, f"{p}[{i}]")
            else:
                yield p, d
        heads = [(p, v) for p, v in walk(ref) if isinstance(v, str) and SHA40.match(v) and re.search(r"git_head|git\.head|repo_commit", p)]
        row["git_heads"] = heads
        for p, v in heads:
            anc = self.is_ancestor(v)
            if anc is False:
                self.add("WARN", "APP_REFERENCE_NOT_ANCESTOR", b, f"{rrel}: {p} {v[:10]} is not an ancestor of HEAD", path=rrel)
            elif anc is None:
                self.add("WARN", "APP_REFERENCE_UNKNOWN_COMMIT", b, f"{rrel}: {p} {v[:10]} is not a commit of this repository", path=rrel)
        return row

    # -- validate.sh (static, heuristic)
    GATE_POS = re.compile(r"(=|==|-eq)\s*['\"]?1['\"]?(\s|\]|$)|(!=|-ne)\s*['\"]?0['\"]?(\s|\]|$)")
    GATE_NEG = re.compile(r"(=|==|-eq)\s*['\"]?0['\"]?(\s|\]|$)|(!=|-ne)\s*['\"]?1['\"]?(\s|\]|$)")
    CPU_MARK = re.compile(r"(?<![A-Za-z_])(--arch[ =]+Sequential|Sequential|Serial|OpenMP|-d\s+cpu|cpu|CPU)(?![A-Za-z_])")
    CPU_EXCLUDE = re.compile(r"CPUS|cpus|HPCPERF_CPU_REFERENCE|CPU_REF\b|nproc|cpus-per|cpu-bind|--bind|CPU_SRC|cpu_count|"
                             r"taskset|numactl|OMP_NUM|cpuset|CPU_PROFILE\s*=|CPU-side|CPU time|cpu time")

    def scan_validate(self, doc, bench_dir, rec):
        f = Path(bench_dir) / "validate.sh"
        if not f.is_file():
            rec["validate_sh"] = None
            return
        lines = f.read_text(errors="replace").splitlines()
        gate_vars = {"HPCPERF_CPU_REFERENCE"}
        incase = False
        for l in lines:
            s = l.strip()
            if re.match(r'case\s+"?\$\{?HPCPERF_CPU_REFERENCE', s):
                if re.search(r"\besac\b", s):          # the whole case on one line
                    for mm in re.finditer(r"\)\s*([A-Z_][A-Z0-9_]*)=", s):
                        gate_vars.add(mm.group(1))
                else:
                    incase = True
                continue
            if incase:
                if s.startswith("esac"):
                    incase = False; continue
                mm = re.match(r"^[^)]*\)\s*([A-Z_][A-Z0-9_]*)=", s)
                if mm:
                    gate_vars.add(mm.group(1))
        for l in lines:                                   # names set only under the gate (e.g. OFFLINE=1)
            mm = re.match(r'^\s*\[\s*"?\$\{?(CPU_REF)\}?"?\s*=\s*1\s*\]\s*&&\s*([A-Z_][A-Z0-9_]*)=', l)
            if mm:
                gate_vars.add(mm.group(2))
        gated_rx = re.compile(r"\$\{?(" + "|".join(sorted(map(re.escape, gate_vars))) + r")\b")
        has_gate = any(not l.strip().startswith("#") and "HPCPERF_CPU_REFERENCE" in l for l in lines)
        stack = []          # frames: [kind, polarity(, case-on-gate)]  polarity: +1 offline-only, -1 default-only, 0 neutral
        default_cpu = []; offline_cpu = []; func_cpu = []
        func = None

        def pol(cond):
            if not gated_rx.search(cond):
                return 0
            if self.GATE_POS.search(cond) or re.search(r"-n\s+\"?\$\{?(" + "|".join(map(re.escape, gate_vars)) + r")", cond):
                return 1
            if self.GATE_NEG.search(cond) or re.search(r"-z\s+\"?\$\{?(" + "|".join(map(re.escape, gate_vars)) + r")", cond):
                return -1
            return 0
        heredoc = None
        for n, l in enumerate(lines, 1):
            s = l.strip()
            if heredoc is not None:                       # python / awk programs embedded in a heredoc are data here
                if s == heredoc:
                    heredoc = None
                continue
            if not s or s.startswith("#"):
                continue
            hd = re.search(r"<<-?\s*['\"]?([A-Za-z_]\w*)['\"]?", s)
            if hd:
                heredoc = hd.group(1)
            code = re.sub(r"\s+#.*$", "", s)
            if func is None and re.match(r"^(function\s+)?[A-Za-z_]\w*\s*\(\)\s*\{?\s*$", code):
                func = re.match(r"^(function\s+)?([A-Za-z_]\w*)", code).group(2); continue
            if func is not None and re.match(r"^\}", l):
                func = None; continue
            one_line_pol = 0
            m_if = re.match(r"^(el)?if\s+(.*?);?\s*then\b(.*)$", code)
            if m_if:
                p = pol(m_if.group(2))
                tail = m_if.group(3)
                if re.search(r"\bfi\b", tail):
                    one_line_pol = p                          # if ...; then ...; fi on one line
                elif m_if.group(1) and stack:
                    stack[-1] = ["if", p]
                else:
                    stack.append(["if", p])
                if not re.search(r"\bfi\b", tail):
                    code = tail
            elif re.match(r"^else\b", code) and stack:
                stack[-1][1] = -stack[-1][1]
                code = code[4:]
            elif re.match(r"^fi\b", code) and stack:
                stack.pop(); continue
            elif re.match(r"^case\s+(.*)\s+in\b", code):
                word = re.match(r"^case\s+(.*)\s+in\b", code).group(1)
                stack.append(["case", 0, bool(gated_rx.search(word))]); continue
            elif re.match(r"^esac\b", code) and stack:
                stack.pop(); continue
            elif stack and stack[-1][0] == "case" and re.match(r"^[^()\s]+\)", code):
                arm = code.split(")")[0].strip("'\" ")
                if stack[-1][2]:
                    stack[-1][1] = 1 if arm in ("1", "1*") else (-1 if arm in ("0", "''|0*", "0*") else 0)
            mm = re.match(r"^\[\[?\s*(.*?)\s*\]\]?\s*(&&|\|\|)\s*(.*)$", code)
            if mm:
                p = pol(mm.group(1))
                one_line_pol = p if mm.group(2) == "&&" else -p
            eff = one_line_pol or next((fr[1] for fr in reversed(stack) if fr[1]), 0)
            bare = re.sub(r'"(?:[^"\\]|\\.)*"|\'[^\']*\'', '""', code)    # text inside quotes is a message or a value
            if re.match(r"^(echo|printf|fail|warn|log|note|die|:)\b", bare) or re.search(r"\|\|\s*(fail|die|echo)\b", bare):
                bare = re.split(r"\|\||&&", bare)[0] if not re.match(r"^(echo|printf|fail|warn|log|note|die|:)\b", bare) else ""
            if bare and self.CPU_MARK.search(bare) and not self.CPU_EXCLUDE.search(bare):
                item = {"line": n, "text": s[:160]}
                if eff > 0:
                    offline_cpu.append(item)
                elif func is not None:
                    item["function"] = func; func_cpu.append(item)
                else:
                    default_cpu.append(item)
        refs = sorted(set(re.findall(r"reference/[\w./${}<>*-]+", "\n".join(lines))))
        called = sorted(set(re.findall(r"\$\{?(?:HERE|BENCH_DIR|SCRIPT_DIR)\}?/([\w.-]+\.(?:py|sh))", "\n".join(lines))))
        called_refs = {}
        for c in called:
            cf = Path(bench_dir) / c
            if cf.is_file():
                t = cf.read_text(errors="replace")
                rr = sorted(set(re.findall(r"reference/[\w./${}<>*-]+", t)))
                if rr or "REF_DIR" in t:
                    called_refs[c] = rr or ["REF_DIR"]
        vrec = {"path": rel(f, self.root), "cpu_reference_gate": has_gate, "gate_vars": sorted(gate_vars),
                "default_path_cpu_candidates": default_cpu, "offline_cpu_lines": len(offline_cpu),
                "function_cpu_candidates": func_cpu,
                "references_named": refs, "checkers_reading_references": called_refs, "heuristic": True}
        rec["validate_sh"] = vrec
        if default_cpu:
            self.add("INFO", "VALIDATE_CPU_CANDIDATES", doc["benchmark"], f"{vrec['path']}: {len(default_cpu)} line(s) mention a CPU "
                     f"backend outside a {'/'.join(sorted(gate_vars))} gate (static heuristic; read them): "
                     + "; ".join(f"{d['line']}: {d['text'][:90]}" for d in default_cpu[:4]), path=vrec["path"],
                     data=default_cpu)

    # -- timing tables (read only): is what tools/timing measures what the golden check validates?
    def _timing_rows(self):
        out = {"level1": [], "extra": []}
        d = self.root / "tools" / "timing" / "cases"
        for key, fn in (("level1", "level1.tsv"), ("extra", "level1_extra.tsv")):
            f = d / fn
            if not f.is_file():
                continue
            cols = None
            for line in f.read_text().splitlines():
                if line.startswith("#app\t"):
                    cols = line[1:].split("\t"); continue
                if not line.strip() or line.startswith("#") or cols is None:
                    continue
                vals = line.split("\t")
                out[key].append(dict(zip(cols, [("" if v == "-" else v) for v in vals])))
        return out

    @staticmethod
    def _norm_args(args):
        out = []
        for a in args:
            a = a.replace("{REPO}/", "").replace("{BUILD}/", "")
            out.append(os.path.basename(a) if "/" in a else a)
        return out

    def audit_timing(self, doc, golden_ids, rec):
        b = doc["benchmark"]
        if doc["level"] != 1:
            return
        reg = {i["id"]: self._norm_args([str(a) for a in i.get("args") or []]) for i in doc["inputs"]}
        rows = []
        for r in self.timing_rows["level1"]:
            if r.get("app") != b:
                continue
            args = self._norm_args(shlex.split(r.get("args") or ""))
            match = [k for k, v in reg.items() if v == args]
            rows.append({"table": "level1.tsv", "case": r.get("case"), "wrapper": r.get("wrapper"), "args": r.get("args"), "inputs": match})
            if golden_ids and r.get("wrapper") == "golden" and doc.get("default_input") not in match:
                self.add("WARN", "TIMING_CASE_ARGS", b, f"tools/timing/cases/level1.tsv case {r.get('case')} times `{r.get('args')}`, which "
                         f"is not the default input {doc.get('default_input')} that its golden ctest validates")
        for r in self.timing_rows["extra"]:
            if r.get("app") != b:
                continue
            args = self._norm_args(shlex.split(r.get("args") or ""))
            match = [k for k, v in reg.items() if v == args]
            rows.append({"table": "level1_extra.tsv", "case": r.get("case"), "args": r.get("args"), "inputs": match, "notes": r.get("notes")})
            if golden_ids and not any(m in golden_ids for m in match):
                self.add("WARN", "TIMING_CASE_UNREGISTERED", b, f"tools/timing/cases/level1_extra.tsv case {r.get('case')} (args "
                         f"`{r.get('args')}`) is no registered input: it is timed without any golden reference "
                         f"(notes: {r.get('notes')})")
        rec["timing_cases"] = rows

    # -- one benchmark
    def audit_bench(self, bench_dir):
        try:
            doc = hi.load(bench_dir)
        except hi.InputError as ex:
            self.add("HARD", "REGISTRY_UNREADABLE", bench_dir.name, str(ex))
            return
        b = doc.get("benchmark", bench_dir.name)
        rec = {"level": doc.get("level"), "benchmark": b, "dir": rel(bench_dir, self.root), "entry": doc.get("entry"),
               "default_input": doc.get("default_input"), "inputs": []}
        self.benchmarks.append(rec)
        errs = hi.validate(doc)
        if errs:
            self.add("HARD", "REGISTRY_INVALID", b, "; ".join(errs[:5]))
        if doc.get("golden") is not None:
            for e in hg.validate_spec(doc["golden"]):
                self.add("HARD", "GOLDEN_SPEC_INVALID", b, e)
        inputs = doc.get("inputs") or []
        by_id = {i.get("id"): i for i in inputs}
        golden_ids = set()
        for inp in inputs:
            if "golden" in inp:
                for e in hg.validate_spec(inp["golden"], f"input '{inp['id']}' golden"):
                    self.add("HARD", "GOLDEN_SPEC_INVALID", b, e, input_id=inp["id"])
            spec = hg.spec_of(doc, inp)
            if spec is not None and not spec.get("unsupported"):
                golden_ids.add(inp["id"])
        if doc.get("default_input") in by_id and hi.is_probe(by_id[doc["default_input"]]):
            self.add("HARD", "DEFAULT_IS_PROBE", b, f"the default input {doc['default_input']} is a correctness probe")
        n_ref = 0
        for inp in inputs:
            iid = inp["id"]
            spec = hg.spec_of(doc, inp)
            irec = {"id": iid, "purpose": inp.get("purpose", "performance"), "variant": inp.get("variant"), "golden": iid in golden_ids,
                    "args": inp.get("args"), "params": inp.get("params")}
            rec["inputs"].append(irec)
            if spec is not None and spec.get("unsupported"):
                irec["golden_unsupported"] = spec["unsupported"]
            self.audit_check_block(doc, inp, spec, irec)
            flags = [str(a) for a in inp.get("args") or [] if str(a) in VERIFY_FLAGS]
            if flags and doc["entry"]["kind"] == "binary":
                irec["verify_flags"] = flags
                conf, gated = self.verify_flag_in_source(bench_dir, flags)
                irec["verify_flag_source"] = conf; irec["verify_flag_gated"] = gated
                if conf and gated:
                    msg = (f", whose host-side verification ({conf}) the source puts behind hpcperf_cpu_reference_enabled() "
                           f"(off by default): read it to confirm the default path runs no host recomputation")
                elif conf:
                    msg = (f", which switches the program's host-side verification on ({conf}): it runs on every default path "
                           f"(plain run, ctest, measure, check) at params {inp.get('params')}")
                else:
                    msg = " (a verification flag by name; the source does not confirm it)"
                self.add("WARN" if conf and not gated else "INFO", "ARGS_ENABLE_CPU_VERIFY", b,
                         f"the registered arguments {inp.get('args')} pass {', '.join(flags)}" + msg, input_id=iid)
            rp = hg.ref_path(bench_dir, iid)
            irec["reference"] = rel(rp, self.root) if rp.is_file() else None
            if iid in golden_ids:
                if not rp.is_file():
                    why = self.recorded_no_golden_reason(doc, iid)
                    if why:
                        irec["missing_reason"] = why
                        self.add("REVIEW", "REFERENCE_MISSING_RECORDED", b, f"no reference/{iid}.json; reason recorded only as a YAML "
                                 f"comment (not machine-readable -- `golden: {{unsupported: ...}}` on the input would make it explicit): "
                                 f"{why[:240]}", input_id=iid)
                    else:
                        self.add("HARD", "REFERENCE_MISSING", b, f"golden input {iid} has no reference/{iid}.json and no recorded "
                                 f"reason: its check FAILs", input_id=iid)
        # every reference file
        refdir = Path(bench_dir) / "reference"
        rows = []
        if refdir.is_dir():
            for rp in sorted(refdir.rglob("*.json")):
                row = self.audit_reference(doc, bench_dir, rp, by_id, rec)
                if row is not None:
                    rows.append(row)
                    if row.get("schema") == hg.REF_SCHEMA:
                        n_ref += 1
        rec["references"] = [r["path"] for r in rows]
        # mechanism
        if golden_ids:
            mech = "golden" if len(golden_ids) == len(inputs) else "golden (partial)"
        elif any(r.get("app_specific") for r in rows):
            mech = "app-specific frozen reference"
        else:
            blks = {(hi.check_of(doc, i) or {}).get("kind") for i in inputs}
            mech = "registry check" if blks - {None, "none"} else ("baseline quantities" if None in blks else "none")
        rec["mechanism"] = mech
        rec["baseline_method"] = (doc.get("baseline") or {}).get("method")
        rec["golden_inputs"] = sorted(golden_ids)
        rec["golden_references"] = n_ref
        if doc.get("level") == 1:
            self.audit_ctest(doc, bench_dir, golden_ids, rec)
            self.audit_timing(doc, golden_ids, rec)
        else:
            self.scan_validate(doc, bench_dir, rec)
        if golden_ids:
            self.add("INFO", "MECHANISM", b, f"golden: {len(golden_ids)} of {len(inputs)} inputs, {n_ref} frozen references")
        else:
            self.add("INFO", "MECHANISM", b, f"{mech}: {(rec['baseline_method'] or '')[:200]}")

    def run(self):
        for d in self.bench_dirs():
            self.audit_bench(d)
        if not self.opts.no_pairwise:
            self.pairwise()
        for f, v in sorted(self.shared_changes.items()):
            self.add("REVIEW", "SHARED_HEADER_CHANGED", None, f"{f} changed after {len(v['refs'])} reference(s) were frozen "
                     f"(commits: {'; '.join(sorted(v['commits'])[:6])}): it is compiled into every benchmark that captures, so "
                     f"confirm the change cannot alter a captured output (e.g. one golden run per benchmark)", path=f,
                     data={"references": v["refs"], "commits": sorted(v["commits"])})
        # one tool hash per reference set: references made with an older hpcperf_golden.py
        cur = hashlib.sha256((HERE / "hpcperf_golden.py").read_bytes()).hexdigest()
        older = sorted({r["path"] for r in self.references if r.get("tool_sha256") and r["tool_sha256"] != cur})
        if older:
            self.add("INFO", "REFERENCE_TOOL_OLDER", None, f"{len(older)} reference(s) were made by an earlier hpcperf_golden.py "
                     f"(provenance.tool_sha256 != the current tool); their frozen fields are re-checked above, nothing else follows")
        edited = sorted(r["path"] for r in self.references if r.get("inputs_yaml_changed"))
        if edited:
            self.add("INFO", "INPUTS_YAML_EDITED", None, f"{len(edited)} reference(s) were made from an earlier inputs.yaml (sha256 "
                     f"differs; spec and workload re-checked above)")
        return self.report()

    def pairwise(self):
        """Inputs of one benchmark whose goldens have the same output shape (inputs that differ in an iteration
        count, a time step, a parameter): compare the frozen golden of input B with input A's rule and limit. When
        B's golden passes A's check, that check cannot tell A's computation from B's -- e.g. 2 time steps from 20.
        Reads golden blobs (CPU only, streaming, via hpcperf_golden.metric_value); outputs larger than
        --pairwise-max-bytes are skipped."""
        by_bench = {}
        for r in self.references:
            if r.get("schema") == hg.REF_SCHEMA and not r.get("problems"):
                by_bench.setdefault((r["level"], r["benchmark"]), []).append(r)
        variants = {(x["level"], x["benchmark"]): {i["id"]: i.get("variant") for i in x["inputs"]} for x in self.benchmarks}
        for (lv, b), refs in sorted(by_bench.items()):
            var = variants.get((lv, b), {})
            docs = {}
            for r in refs:
                try:
                    docs[r["input_id"]] = json.loads((self.root / r["path"]).read_text())
                except (OSError, json.JSONDecodeError):
                    pass
            ids = sorted(docs)
            for ia in ids:
                for ib in ids:
                    if ia == ib:
                        continue
                    # an implementation-path variant computes the same result by another code path: equal outputs are expected
                    sev = "INFO" if "implementation-path" in (var.get(ia), var.get(ib)) else "WARN"
                    oa = {o["name"]: o for o in docs[ia]["outputs"]}
                    ob = {o["name"]: o for o in docs[ib]["outputs"]}
                    for name, ea in sorted(oa.items()):
                        eb = ob.get(name)
                        if eb is None or eb["dtype"] != ea["dtype"] or eb["count"] != ea["count"]:
                            continue
                        m = ea["rule"]["metric"]
                        if m == "exact":
                            if eb["sha256"] == ea["sha256"] and ia < ib and ea["count"] > 1:
                                self.add(sev, "PAIRWISE_IDENTICAL", b, f"inputs {ia} and {ib} produce the byte-identical output "
                                         f"{name} ({ea['dtype']}[{ea['count']}]): the check cannot tell the two workloads apart",
                                         input_id=ia, data={"other": ib, "output": name})
                            continue
                        if ea["bytes"] > self.opts.pairwise_max_bytes or m == "trunc_int":
                            continue
                        pa, pb = self.store / ea.get("blob", ""), self.store / eb.get("blob", "")
                        if not (pa.is_file() and pb.is_file()):
                            continue
                        try:
                            v = hg.metric_value(m, pb, pa, ea["dtype"], ea["count"], rule=ea["rule"])
                        except Exception as ex:  # noqa: BLE001
                            self.add("INFO", "PAIRWISE_ERROR", b, f"{ia}/{ib} {name}: {ex}")
                            continue
                        te = ea["rule"].get("tol_effective")
                        rec = {"input": ia, "other": ib, "output": name, "metric": m, "value": v, "limit": te}
                        self.pairwise_rows.append(dict(rec, benchmark=b))
                        if te is not None and v <= te:
                            self.add(sev, "PAIRWISE_INDISTINGUISHABLE", b, f"the frozen golden of input {ib} passes input {ia}'s "
                                     f"check of {name} ({m} {v:.3e} <= {te:.3e}): {ia}'s check cannot tell its workload from "
                                     f"{ib}'s ({json.dumps(docs[ia]['workload'].get('params'))} vs "
                                     f"{json.dumps(docs[ib]['workload'].get('params'))})", input_id=ia, data=rec)

    def report(self):
        counts = {s: sum(1 for f in self.findings if f["severity"] == s) for s in SEVERITIES}
        codes = {}
        for f in self.findings:
            codes.setdefault(f["severity"], {}).setdefault(f["code"], 0)
            codes[f["severity"]][f["code"]] += 1
        return {"schema": SCHEMA, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "repo": str(self.root),
                "git_head": self.head, "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "options": {"levels": self.opts.levels, "bench": self.opts.bench, "no_ctest": self.opts.no_ctest,
                            "build_root": self.opts.build_root, "aggregate_build": self.opts.aggregate_build,
                            "store": str(self.store)},
                "summary": {"benchmarks": len(self.benchmarks),
                            "golden_benchmarks": sum(1 for b in self.benchmarks if b.get("golden_inputs")),
                            "golden_references": sum(1 for r in self.references if r.get("schema") == hg.REF_SCHEMA),
                            "app_references": sum(1 for r in self.references if r.get("app_specific")),
                            "findings": counts, "codes": codes},
                "benchmarks": self.benchmarks, "references": self.references, "pairwise": self.pairwise_rows,
                "findings": self.findings}


# ----------------------------------------------------------------------------- the readable table
def render_table(rep, show_info=False):
    out = []
    s = rep["summary"]
    out.append(f"hpcperf_golden_audit  {rep['repo']}  HEAD {str(rep['git_head'])[:10]}  {rep['generated_utc']}")
    out.append(f"benchmarks {s['benchmarks']} (golden {s['golden_benchmarks']}), golden references {s['golden_references']}, "
               f"app-specific references {s['app_references']}")
    out.append("findings: " + ", ".join(f"{k} {v}" for k, v in s["findings"].items()))
    for sev in SEVERITIES:
        if s["codes"].get(sev):
            out.append(f"  {sev:6s} " + ", ".join(f"{c} {n}" for c, n in sorted(s["codes"][sev].items())))
    out.append("")
    out.append("== mechanisms")
    out.append(f"{'lvl':3s} {'benchmark':27s} {'mechanism':30s} {'in':>3s} {'ref':>3s} {'ctest (per-benchmark build)':44s} registry check")
    for b in rep["benchmarks"]:
        ct = "-"
        if isinstance(b.get("ctest"), list) and b["ctest"]:
            e = b["ctest"][0]
            if e.get("tests"):
                t = e["tests"][0]
                ct = f"{t['wrapper']}" + (f" in={t.get('input_id')}" if t.get("input_id") else "") + \
                     f" pass={t['pass_regex'][0] if t['pass_regex'] else 'NONE'}"
                if t.get("verify_flags"):
                    ct += f" {' '.join(t['verify_flags'])}"
            elif e.get("missing"):
                ct = "no build"
        kinds = sorted({str((i.get('check') or {}).get('kind')) for i in b["inputs"]})
        chk = "/".join(kinds)
        if all((i.get("check") or {}).get("golden") for i in b["inputs"] if i["golden"]) and b.get("golden_inputs"):
            chk += " golden"
        out.append(f"{b['level']!s:3s} {b['benchmark']:27s} {b['mechanism'][:30]:30s} {len(b['inputs']):3d} {b.get('golden_references', 0):3d} "
                   f"{ct[:44]:44s} {chk}")
    out.append("")
    out.append("== golden references (tolerance outputs: worst budget d0/tol, smallest tol_eff/|golden|max, flags)")
    out.append(f"{'reference':52s} {'outs':>4s} {'metrics':18s} {'d0/tol':>7s} {'te/sig':>8s} {'ulps':>8s} {'src':4s} flags")
    for r in rep["references"]:
        if r.get("schema") != hg.REF_SCHEMA:
            continue
        outs = r.get("outputs") or []
        metrics = ",".join(sorted({o["metric"] or "?" for o in outs}))
        bud = max([o.get("budget_used") or 0 for o in outs] or [0])
        rs = [o.get("rel_tol_vs_signal") for o in outs if o.get("rel_tol_vs_signal") is not None]
        ul = [o.get("tol_in_ulps") for o in outs if o.get("tol_in_ulps") is not None]
        flags = sorted({f for o in outs for f in o.get("flags", []) if f != "D0_ZERO"} | set(r.get("problems", [])))
        if r.get("scalar_only"):
            flags.append("SCALAR_ONLY")
        src = "chg" if r.get("sources_changed") else ("wt" if r.get("sources_uncommitted") else "-")
        tolmetric = [o for o in outs if o["metric"] != "exact"]
        out.append(f"{r['benchmark'] + '/' + r['input_id']:52.52s} {len(outs):4d} {metrics[:18]:18s} "
                   f"{(format(bud, '.2f') if tolmetric else '-'):>7s} {(fnum(min(rs)) if rs else '-'):>8s} "
                   f"{(fnum(min(ul)) if ul else '-'):>8s} {src:4s} {','.join(flags)}")
    out.append("")
    for sev in SEVERITIES:
        fs = [f for f in rep["findings"] if f["severity"] == sev]
        if not fs or (sev == "INFO" and not show_info):
            continue
        out.append(f"== {sev} ({len(fs)})")
        for f in fs:
            tag = (f["benchmark"] or "*") + (f"/{f['input_id']}" if f.get("input_id") else "")
            out.append(f"  [{f['code']}] {tag}: {f['message']}")
        out.append("")
    if not show_info:
        out.append(f"(INFO findings: {rep['summary']['findings']['INFO']}; --info to list them)")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="repository root (default: the repository of the current directory)")
    ap.add_argument("--level", default="1,2,3", help="levels to audit (default 1,2,3)")
    ap.add_argument("--bench", nargs="*", help="only these benchmarks (names or level<N>/<name>)")
    ap.add_argument("--json", help="write the machine-readable report (schema hpcperf-golden-audit-1) here")
    ap.add_argument("--table", default="-", help="write the readable table here ('-' = stdout, '' = none)")
    ap.add_argument("--info", action="store_true", help="list INFO findings in the table")
    ap.add_argument("--no-ctest", action="store_true", help="do not ask ctest for the Level 1 test lists")
    ap.add_argument("--ctest", help="the ctest executable (default: the build tree's CMAKE_CTEST_COMMAND, $HPCPERF_CTEST, PATH)")
    ap.add_argument("--build-root", default="build", help="per-benchmark build trees <build-root>/<bench>/cuda (default build)")
    ap.add_argument("--aggregate-build", default="build/all", help="aggregate build tree (<dir>/level1/<bench>; '' = skip)")
    ap.add_argument("--hash-binaries", action="store_true", help="also hash each reference's binary (reports rebuilt binaries)")
    ap.add_argument("--no-pairwise", action="store_true", help="skip the golden-vs-golden comparison of same-shape inputs")
    ap.add_argument("--pairwise-max-bytes", type=int, default=64 << 20, help="largest output compared pairwise (default 64 MiB)")
    ap.add_argument("--strict", action="store_true", help="exit 1 on WARN findings as well")
    a = ap.parse_args(argv)
    try:
        root = Path(a.root).resolve() if a.root else hi.repo_root(Path.cwd())
        levels = [int(x) for x in a.level.split(",") if x.strip()]
    except (hi.InputError, ValueError) as ex:
        print(f"ERROR: {ex}", file=sys.stderr); return 2
    a.levels = levels
    if a.aggregate_build == "":
        a.aggregate_build = None
    aud = Audit(root, a)
    rep = aud.run()
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1, default=str) + "\n")
    if a.table:
        txt = render_table(rep, show_info=a.info)
        if a.table == "-":
            sys.stdout.write(txt)
        else:
            Path(a.table).write_text(txt)
    c = rep["summary"]["findings"]
    if c["HARD"] or (a.strict and c["WARN"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
