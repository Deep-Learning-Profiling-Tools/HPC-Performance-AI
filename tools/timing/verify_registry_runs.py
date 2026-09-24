#!/usr/bin/env python3
"""Verify that timing records of registered inputs actually ran the registered workload.

    verify_registry_runs.py [--json OUT] [--repo DIR] [--rules FILE] PATH...

PATH is a timing results directory (every `hpcperf-timing-*` record below it is checked) or one record.
Read-only: nothing is executed, only records, their raw ROI logs and program output, and repository
files are read.

A record is linked to its input by the workload identity the engine stored (`hpcperf_inputs.py
identity`); this tool checks that the processes of EVERY clean run -- as the ROI log recorded them:
executable, working directory and argv -- ran that input:

  Level 1   the registry's own executable, the registry's arguments element by element (a path
            argument compared by real path, relative ones resolved against the process's working
            directory) in the executable's directory, the input's env, and every argument / build file
            still hashing to the identity's sha256.
  Level 2   the application's own binary (under build/level2/<app>/), the registry selector set to the
            input id, the registry arguments as one contiguous run of the argv (path arguments resolved
            the way run.sh resolves them -- the benchmark directory, then the `search_dirs` the evidence
            rules declare -- and compared by real path AND content hash; a same-named file elsewhere
            never matches), every registered file reached by the program (as an argv path or a
            directory argument holding it, by the working directory, or through a declared copy /
            generated file), no registered option given more than once unless the program's parser is
            declared last-wins and the last occurrence carries the registered value, and every env knob
            the input sets evidenced by the per-application rules (argv / exe / program output / build
            configuration templates) in cases/registry_evidence.yaml.

Verdicts: PASS; FAIL (a contradiction: wrong file, wrong argument, duplicate option, changed file, ...);
INSUFFICIENT (no contradiction, but some part of the input is not evidenced -- listed as gaps); NOT_RUN
(the record has no measured run, e.g. build_not_materialized); SUPERSEDED (the run did get the workload
its identity records, but the registry has since changed the input's definition -- a valid historical
result of the OLD workload, never a result of the current input; unlike an invalidated record it did
not run the wrong workload). Exit 0 only when every record is PASS, NOT_RUN or SUPERSEDED.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_RULES = os.path.join(HERE, "cases", "registry_evidence.yaml")
REPO_PATH_PREFIXES = ("level1/", "level2/", "level3/", "build/")
PLACEHOLDER = re.compile(r"\{(\w+)\}")
NUMBER = re.compile(r"^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")

_sha_cache = {}


def sha(path):
    path = os.path.realpath(path)
    if path not in _sha_cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        _sha_cache[path] = h.hexdigest()
    return _sha_cache[path]


def resolve(p, cwd):
    return os.path.realpath(p if os.path.isabs(p) else os.path.join(cwd, p))


def is_option(tok):
    return tok.startswith("-") and len(tok) > 1 and not NUMBER.match(tok)


def load_rules(path):
    import yaml
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    return {1: doc.get("level1") or {}, 2: doc.get("level2") or {}}


def parse_roi_log(path):
    """exe / cwd / argv of the process that wrote one ROI log."""
    p = {"log": path}
    with open(path, errors="replace") as f:
        for line in f:
            k, _, v = line.rstrip("\n").partition(" ")
            if k in ("exe", "cwd", "rank", "pid"):
                p[k] = v
            elif k == "argv":
                p["argv"] = json.loads(v)
    return p


def clean_runs(raw):
    runs = []
    for d in sorted(glob.glob(os.path.join(raw, "clean.*")), key=lambda x: int(x.rsplit(".", 1)[1])):
        procs = [parse_roi_log(r) for r in sorted(glob.glob(os.path.join(d, "roi.*")))]
        log = os.path.join(d, "run.log")
        out = open(log, errors="replace").read() if os.path.isfile(log) else None
        runs.append({"dir": os.path.basename(d), "procs": procs, "output": out})
    return runs


def fill(template, values):
    """Template with each {name} replaced by the escaped value; None when a name has no value."""
    missing = [n for n in PLACEHOLDER.findall(template) if values.get(n) is None]
    if missing:
        return None
    return PLACEHOLDER.sub(lambda m: re.escape(str(values[m.group(1)])), template)


def template_values(params, rule):
    vals = {k: v for k, v in (params or {}).items() if not isinstance(v, (dict, list))}
    for name, expr in (rule.get("derive") or {}).items():
        try:
            vals[name] = eval(expr, {"__builtins__": {}, "round": round, "int": int, "str": str}, dict(vals))
        except Exception:
            vals[name] = None
    return vals


class Check:
    def __init__(self):
        self.problems, self.gaps, self.evidence = [], [], []

    def fail(self, m):
        if m not in self.problems:
            self.problems.append(m)

    def gap(self, m):
        if m not in self.gaps:
            self.gaps.append(m)

    def ok(self, m):
        if m not in self.evidence:
            self.evidence.append(m)


def apply_templates(c, rule, params, runs, level):
    """Match the rule's argv/exe/output/cwd_file templates; returns the number of templates matched."""
    vals = template_values(params, rule)
    matched = 0
    for kind in ("argv", "exe", "output"):
        for t in rule.get(kind) or []:
            rx = fill(t, vals)
            if rx is None:
                c.ok(f"{kind} template /{t}/ not applicable (param missing)")
                continue
            bad = []
            for r in runs:
                if kind == "output":
                    hay = [r["output"]] if r["output"] is not None else []
                    if not hay:
                        c.gap(f"{r['dir']}: no run.log to match output evidence")
                        continue
                elif kind == "argv":
                    hay = [" ".join(p["argv"][1:]) + " " for p in r["procs"]]
                else:
                    hay = [p["exe"] for p in r["procs"]]
                if not all(re.search(rx, h, re.M) for h in hay):
                    bad.append(r["dir"])
            if bad:
                c.fail(f"{kind} evidence /{rx}/ missing in {','.join(bad)}")
            else:
                matched += 1
                c.ok(f"{kind} /{rx}/")
    for fname, ts in (rule.get("cwd_file") or {}).items():
        for t in ts:
            rx = fill(t, vals)
            if rx is None:
                c.ok(f"cwd_file template /{t}/ not applicable (param missing)")
                continue
            bad = []
            for r in runs:
                for p in r["procs"]:
                    fp = os.path.join(p["cwd"], fname)
                    if not (os.path.isfile(fp) and re.search(rx, open(fp, errors="replace").read(), re.M)):
                        bad.append(r["dir"])
            if bad:
                c.fail(f"{fname} in the process cwd lacks /{rx}/ ({','.join(sorted(set(bad)))})")
            else:
                matched += 1
                c.ok(f"cwd {fname} /{rx}/")
    return matched


def verify_level1(c, repo, ident, rec, runs, rule):
    exe = os.path.realpath(os.path.join(repo, ident.get("binary") or ident["entry"]["path"]))
    want_cwd = os.path.dirname(exe)
    wl = ident["workload"]
    want = [os.path.join(repo, a) if str(a).startswith(REPO_PATH_PREFIXES) else str(a) for a in wl.get("args") or []]
    arg_hash = {os.path.realpath(os.path.join(repo, p)): h for p, h in (ident.get("arg_files_sha256") or {}).items()}
    for r in runs:
        for p in r["procs"]:
            if os.path.realpath(p["exe"]) != exe:
                c.fail(f"{r['dir']}: exe {p['exe']} is not the registry binary {exe}")
            if os.path.realpath(p["cwd"]) != want_cwd:
                c.fail(f"{r['dir']}: cwd {p['cwd']} != {want_cwd}")
            got = p["argv"][1:]
            if len(got) != len(want):
                c.fail(f"{r['dir']}: argv {got} != registry {want}")
                continue
            for g, w in zip(got, want):
                if g == w and not os.path.exists(resolve(w, want_cwd)):
                    continue
                rg, rw = resolve(g, p["cwd"]), resolve(w, want_cwd)
                if os.path.exists(rw) and rg == rw:
                    if rw in arg_hash and sha(rg) != arg_hash[rw]:
                        c.fail(f"argument file {g} does not hash to the identity's value")
                    continue
                c.fail(f"{r['dir']}: argument {g!r} != registry {w!r}")
    denv = (rec.get("inputs") or {}).get("declared_env") or {}
    for k, v in (wl.get("env") or {}).items():
        if denv.get(k) != str(v):
            c.fail(f"env {k}={v} not set for the run (declared {denv.get(k)!r})")
    for p, h in (ident.get("build_files_sha256") or {}).items():
        fp = os.path.join(repo, p)
        if not os.path.isfile(fp) or sha(fp) != h:
            c.fail(f"build file {p} no longer hashes to the identity's value")
    if not c.problems:
        c.ok("exe, cwd and argv are the registry's (element-wise, paths by real path)")
    apply_templates(c, rule, wl.get("params"), runs, 1)


def tok_paths(tok):
    """The path candidates of one argv token: the token, and the value of an --option=value token."""
    out = [tok]
    if tok.startswith("-") and "=" in tok:
        out.append(tok.split("=", 1)[1])
    return out


def dir_copies(rel, repo, rule):
    """Declared copies of a registered file (benchmark-relative `rel`) under `copy_dirs`
    ({source dir: copy dir}, benchmark-relative -> repository-relative): the same relative name."""
    out = []
    for src, dst in (rule.get("copy_dirs") or {}).items():
        src = src.rstrip("/") + "/"
        if rel.startswith(src):
            out.append(os.path.realpath(os.path.join(repo, dst, rel[len(src):])))
    return out


def match_arg(tok, reg, cands, cwd, repo, rule, files_hash):
    """Does argv token `tok` (of a process in `cwd`) stand for registry argument `reg`?
    Returns (matched, note)."""
    target = next((x for x in cands if os.path.exists(x)), None)
    if target is None:                           # a plain value: must be literally the same
        return tok == reg, None
    real = resolve(tok, cwd)
    want_hash = files_hash.get(os.path.realpath(target))
    if real == os.path.realpath(target):
        if want_hash and os.path.isfile(real) and sha(real) != want_hash:
            return False, f"{tok} is the registered file but no longer hashes to the identity's value"
        return True, None
    for src, prefix in (rule.get("copies") or {}).items():
        if real.startswith(os.path.join(repo, prefix)) and os.path.isfile(real) and os.path.isfile(target) \
                and sha(real) == sha(target) and (want_hash is None or sha(real) == want_hash):
            return True, f"{tok}: declared copy of {src}, same sha256"
    return False, None


def verify_level2(c, repo, ident, rec, runs, rule):
    app = ident["benchmark"]
    bdir = os.path.join(repo, "level2", app)
    wl = ident["workload"]
    sel = ident.get("selector")
    denv = (rec.get("inputs") or {}).get("declared_env") or {}
    if not sel or denv.get(sel) != rec["case"]:
        c.fail(f"selector {sel}={denv.get(sel)!r}, not the input id {rec['case']!r}")
    own = os.path.realpath(os.path.join(repo, "build", "level2", app)) + os.sep
    files_hash = {os.path.realpath(os.path.join(bdir, f)): h for f, h in (wl.get("files_sha256") or {}).items()}
    files_hash.update({os.path.realpath(os.path.join(repo, f)): h for f, h in (ident.get("arg_files_sha256") or {}).items()})
    for f, h in files_hash.items():
        if h is None or not os.path.isfile(f) or sha(f) != h:
            c.fail(f"registered file {os.path.relpath(f, repo)} is missing or no longer hashes to the identity's value")
    args = [str(a) for a in wl.get("args") or []]
    search = [""] + list(rule.get("search_dirs") or [])
    cands = [[os.path.join(bdir, sd, a) for sd in search] + ([os.path.join(repo, a)] if a.startswith(REPO_PATH_PREFIXES) else [])
             if not a.startswith("-") else [] for a in args]
    last_wins = rule.get("last_wins")
    reached = set()
    for r in runs:
        for p in r["procs"]:
            argv, cwd = p["argv"], p["cwd"]
            if not os.path.realpath(p["exe"]).startswith(own):
                c.fail(f"{r['dir']}: exe {p['exe']} is not {app}'s own binary")
            # 1. the registry arguments, as one contiguous run of the argv
            if args:
                hit = None
                for i in range(1, len(argv) - len(args) + 1):
                    res = [match_arg(argv[i + j], a, cands[j], cwd, repo, rule, files_hash) for j, a in enumerate(args)]
                    if all(m for m, _ in res):
                        hit = i
                        for _, note in res:
                            if note:
                                c.ok(note)
                        break
                    for m, note in res:
                        if note and not m:
                            c.fail(note)
                if hit is None:
                    c.fail(f"{r['dir']}: registry args {args} are not passed to the program (argv {argv[1:]})")
                else:
                    c.ok(f"registry args {args} passed contiguously (argv[{hit}:{hit + len(args)}])")
                    # 2. registered options given more than once
                    for j, a in enumerate(args):
                        if not is_option(a):
                            continue
                        pos = [k for k, t in enumerate(argv) if t == a]
                        if len(pos) > 1:
                            if not last_wins:
                                c.fail(f"{r['dir']}: option {a} given {len(pos)} times and {app}'s parser is not declared last-wins")
                            elif pos[-1] != hit + j:
                                c.fail(f"{r['dir']}: option {a} given {len(pos)} times; the last one is not the registered value")
                            else:
                                c.ok(f"option {a} repeated; last occurrence is the registry's (last-wins: {last_wins})")
            # 3. registered files reached by the program
            logged = set()
            if rule.get("logged_reads") and r["output"] is not None:
                logged = set(re.findall(rule["logged_reads"], r["output"], re.M))
            for tok in argv[1:]:
                for t in tok_paths(tok):
                    real = resolve(t, cwd)
                    for f in files_hash:
                        if real == f or (os.path.isdir(real) and f.startswith(real + os.sep)):
                            reached.add((r["dir"], f, "argv"))
            for f in files_hash:
                rel = os.path.relpath(f, bdir)
                # a declared copy (copy_dirs) reached by the argv or named by the program's own log as read
                # from its working directory: accepted only with the registered sha256
                for cp in dir_copies(rel, repo, rule):
                    via = []
                    if any(resolve(t, cwd) == cp for tok in argv[1:] for t in tok_paths(tok)):
                        via.append("argv (declared copy)")
                    if os.path.basename(f) in logged and resolve(os.path.basename(f), cwd) == cp:
                        via.append("logged read (declared copy)")
                    if not via:
                        continue
                    if not os.path.isfile(cp) or sha(cp) != files_hash[f]:
                        c.fail(f"{r['dir']}: {os.path.relpath(cp, repo)} is the declared copy of {rel} read by the run, "
                               f"but its content does not hash to the identity's value")
                    else:
                        for v in via:
                            reached.add((r["dir"], f, v))
                if os.path.basename(f) in logged and resolve(os.path.basename(f), cwd) == f:
                    reached.add((r["dir"], f, "logged read"))
                if rel in (rule.get("cwd_files") or []) and os.path.realpath(cwd) == os.path.dirname(f):
                    reached.add((r["dir"], f, "cwd"))
                for kind in ("copies", "generated"):
                    prefix = (rule.get(kind) or {}).get(rel)
                    if not prefix:
                        continue
                    for tok in argv[1:]:
                        real = resolve(tok, cwd)
                        if real.startswith(os.path.join(repo, prefix)) and os.path.isfile(real):
                            if kind == "copies" and (sha(real) != sha(f) or sha(real) != files_hash[f]):
                                c.fail(f"{tok} is declared a copy of {rel} but its sha256 differs")
                            else:
                                reached.add((r["dir"], f, kind))
    generated_used = False
    for f in files_hash:
        how = {h for (d, ff, h) in reached if ff == f}
        dirs = {d for (d, ff, h) in reached if ff == f}
        rel = os.path.relpath(f, bdir)
        if len(dirs) < len(runs):
            c.gap(f"registered file {rel} is not reached by the program in every clean run")
        else:
            c.ok(f"file {rel} reached via {'/'.join(sorted(how))}, sha256 {files_hash[f][:12]}")
            generated_used |= "generated" in how
    matched = apply_templates(c, rule, wl.get("params"), runs, 2)
    if generated_used and not matched:
        c.gap("a registered file reaches the program only through a generated file, and no output/argv template evidences it")
    # 4. every knob the input sets must be evidenced
    covers = set(rule.get("covers") or [])
    for k, v in (wl.get("env") or {}).items():
        if denv.get(k) not in (None, str(v)):
            c.fail(f"env {k} declared as {denv.get(k)!r}, registry value {v!r}")
        vfile = os.path.realpath(os.path.join(bdir, str(v)))
        if vfile in files_hash and any(ff == vfile for (_, ff, _) in reached):
            c.ok(f"knob {k} evidenced as the registered file it names")
        elif k in covers and matched:
            c.ok(f"knob {k} covered by the {app} evidence rules")
        else:
            c.gap(f"knob {k}={v} is not evidenced by argv, files or output")


_current = {}
SUPPLEMENT = "file_identity_supplement.json"


def files_added(recorded, cur):
    """{file: sha256} when the current definition differs from the recorded workload ONLY by input files
    the record's identity did not name (every other field, and every recorded file hash, unchanged)."""
    if not isinstance(cur, dict) or not isinstance(recorded, dict) or cur == recorded:
        return None
    if {k: v for k, v in cur.items() if k != "files_sha256"} != {k: v for k, v in recorded.items() if k != "files_sha256"}:
        return None
    old, new = recorded.get("files_sha256") or {}, cur.get("files_sha256") or {}
    if any(new.get(k) != v for k, v in old.items()):
        return None
    return {k: v for k, v in new.items() if k not in old} or None


def supplement_for(record_path, rec):
    """The supplementary file-identity entry of this record: <results root>/file_identity_supplement.json
    (schema hpcperf-file-identity-supplement-1), written next to the records by whoever established it."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(record_path)))))
    p = os.path.join(root, SUPPLEMENT)
    if not os.path.isfile(p):
        return None
    for e in json.load(open(p)).get("records") or []:
        if (e.get("level"), e.get("app"), e.get("case"), e.get("run_id")) == (rec.get("level"), rec.get("app"), rec.get("case"), rec.get("run_id")):
            return e
    return None


def current_workload(repo, level, app, iid):
    """The input's workload as the registry defines it NOW (None: no registry here, or no such input)."""
    key = (repo, level, app, iid)
    if key not in _current:
        _current[key] = None
        bdir = os.path.join(repo, f"level{level}", app)
        if os.path.isfile(os.path.join(bdir, "inputs.yaml")):
            sys.path.insert(0, os.path.join(repo, "tools", "inputs"))
            import hpcperf_inputs as hi
            doc = hi.load(bdir)
            inp = next((i for i in doc["inputs"] if i["id"] == iid), None)
            _current[key] = hi.registry_identity(doc, inp)["workload"] if inp else "unregistered"
    return _current[key]


def verify_record(path, repo, rules):
    rec = json.load(open(path))
    out = {"record": path, "app": rec.get("app"), "case": rec.get("case"), "level": rec.get("level"),
           "status": rec.get("status"), "git_commit": (rec.get("provenance") or {}).get("git_commit"),
           "exe_sha256": (rec.get("inputs") or {}).get("exe_sha256")}
    c = Check()
    reg = rec.get("registry") or {}
    ident = reg.get("identity") or {}
    raw = os.path.join(repo, (rec.get("provenance") or {}).get("raw_dir") or "")
    runs = clean_runs(raw) if os.path.isdir(raw) else []
    if rec.get("status") != "ok" and not any(r["procs"] for r in runs):
        # never reached the ROI (e.g. build_not_materialized, or run.sh stopped before the program ran)
        out.update(verdict="NOT_RUN", problems=[f"status {rec.get('status')}, no ROI reached in any clean run"],
                   gaps=[], evidence=[], clean_runs=len(runs))
        return out
    if rec.get("status") != "ok":
        c.fail(f"status {rec.get('status')}")
    if not reg.get("identity_complete") or not ident.get("complete"):
        c.fail("workload identity not complete")
    if ident.get("input_id") != rec.get("case") or ident.get("benchmark") != rec.get("app"):
        c.fail(f"identity is for {ident.get('benchmark')}/{ident.get('input_id')}")
    wi = os.path.join(raw, "workload_identity.json")
    if not os.path.isfile(wi) or json.load(open(wi)).get("workload") != ident.get("workload"):
        c.fail("raw workload_identity.json missing or differs from the record's")
    if len(runs) != len((rec.get("roi") or {}).get("runs_s") or []):
        c.fail(f"{len(runs)} clean run dirs but {len((rec.get('roi') or {}).get('runs_s') or [])} ROI samples")
    for r in runs:
        if not r["procs"]:
            c.fail(f"{r['dir']}: no ROI log")
        for p in r["procs"]:
            if "argv" not in p or "cwd" not in p or "exe" not in p:
                c.fail(f"{r['dir']}: ROI log {os.path.basename(p['log'])} lacks exe/cwd/argv")
    cur = current_workload(repo, rec["level"], rec["app"], rec["case"])
    used, file_identity = ident, "recorded"
    added = files_added(ident.get("workload"), cur)
    if added:
        # The registry now names input files this record's identity did not capture. Those files are
        # verified against the CURRENT registry hashes, but that is today's content: it counts for the
        # measurement only through a separate supplement that establishes, with its basis, which content
        # the run read (never by rewriting the stored identity).
        used = json.loads(json.dumps(ident))
        used["workload"].setdefault("files_sha256", {}).update(added)
        sup = supplement_for(path, rec)
        if sup and all((sup.get("files_sha256") or {}).get(k) == v for k, v in added.items()):
            file_identity = "supplement"
            c.ok(f"file identity of {', '.join(sorted(added))} from a supplementary verification: {sup.get('basis', '')}")
        else:
            file_identity = "insufficient"
            c.gap(f"file identity of {', '.join(sorted(added))} was not captured when the run was measured and no "
                  f"supplementary verification establishes it -- file identity INSUFFICIENT")
    if not c.problems:
        rule = rules.get(rec["level"], {}).get(rec["app"]) or {}
        try:
            (verify_level1 if rec["level"] == 1 else verify_level2)(c, repo, used, rec, runs, rule)
        except (KeyError, OSError, re.error) as exc:
            c.fail(f"verifier error: {exc!r}")
    verdict = "FAIL" if c.problems else "INSUFFICIENT" if c.gaps else "PASS"
    out["file_identity"] = file_identity
    if verdict == "PASS" and cur is not None and cur != used.get("workload"):
        verdict = "SUPERSEDED"
        c.gaps.append("the registry's current definition of this input differs from the recorded workload "
                      + ("(input no longer registered)" if cur == "unregistered" else
                         "(" + ", ".join(k for k in sorted(set(cur) | set(ident["workload"]))
                                         if cur.get(k) != used["workload"].get(k)) + " changed)")
                      + " -- a result of the old workload only")
    out.update(verdict=verdict, problems=c.problems, gaps=c.gaps, evidence=c.evidence, clean_runs=len(runs))
    return out


def record_files(paths):
    for p in paths:
        if os.path.isfile(p):
            yield p
            continue
        for f in sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True)):
            try:
                d = json.load(open(f))
            except (ValueError, OSError):
                continue
            if isinstance(d, dict) and str(d.get("schema", "")).startswith("hpcperf-timing-") and "registry" in d:
                yield f


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--rules", default=DEFAULT_RULES)
    ap.add_argument("--json", help="write the per-record verdicts here")
    ap.add_argument("-q", "--quiet", action="store_true", help="print only non-PASS records and the totals")
    a = ap.parse_args(argv)
    repo = os.path.realpath(a.repo)
    rules = load_rules(a.rules)
    res = [verify_record(f, repo, rules) for f in record_files(a.paths)]
    counts = {}
    for r in res:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        if a.quiet and r["verdict"] == "PASS":
            continue
        print(f"{r['verdict']:<12} L{r['level']} {r['app']}/{r['case']}  {os.path.basename(r['record'])}")
        for m in r["problems"]:
            print(f"    problem: {m}")
        for m in r["gaps"]:
            print(f"    gap: {m}")
    print("verify_registry_runs:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "no records",
          f"({len(res)} records)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"schema": "hpcperf-registry-run-verification-1", "counts": counts, "records": res}, f, indent=1)
    return 0 if res and all(r["verdict"] in ("PASS", "NOT_RUN", "SUPERSEDED") for r in res) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
