#!/usr/bin/env python3
"""check_workspace -- static contract checks of a Level 3 benchmark directory (canonical level3/<app> or an
agent workspace copy). No build is run.

    check_workspace.py <benchmark-dir> [--variant NAME] [--json OUT] [--quick]
                       [--agent-mode [--baseline FILE] [--iteration N] [--report OUT.json] [--diff OUT.diff]]

Baseline mode (default; canonical directory and iteration 0 of every run): the source tree must equal the
frozen baseline.

Agent mode (iteration > 0) -- benchmark integrity, protected by default, source tree explicitly mutable:
  SOURCE     src/** and deps/**: may be modified, added or deleted relative to the baseline recorded at
             workspace creation; every change is listed (modified/added/deleted, diff) and the source tree is
             re-hashed (initial_source_hash / current_source_hash). Which part of the source an optimization
             agent SHOULD touch is not decided here: that is the downstream evaluation protocol's business.
  PROTECTED  everything else in the benchmark directory (build.sh, run.sh, validate.sh, benchmark.yaml,
             inputs/**, references/**, provenance/**, workspace metadata, the materialization marker), the
             files benchmark.yaml declares as `inputs`/`references` even when they live under src/, and every
             harness file of the workspace root (hpcperf_env.sh, level2/tools/**, level3/tools/**): must be
             identical to the trusted baseline, otherwise check 4 FAILS ("PROTECTED FILE TAMPERING" /
             "HARNESS TAMPERING") and tools/validate_workspace.sh REFUSES the iteration.

Trusted baseline (agent mode): --baseline FILE (must lie outside the workspace root) or the repository copy
<repo>/.hpcperf/workspace_baselines/<run-id>.json written at workspace creation. The copy inside the workspace root
(workspace_baseline.json) is NOT trusted by default -- a workspace cannot re-declare its own baseline -- and is
accepted only with --allow-workspace-baseline (development use). These checks are file-hash and permission
checks, not an operating-system sandbox: the agent process is not confined by them.

Checks (each PASS/FAIL, all must pass):
  1  benchmark.yaml exists with the required keys (and hip is not claimed validated)
  2  src/ materialized (a real directory)
  3  deps/ present when the lock's layout requires it
  4  source identity: == canonical source_tree_sha256 (baseline mode) / protected files unchanged (agent mode)
  5  build.sh, run.sh, validate.sh exist and are executable
  6  the scripts reference no application source outside the benchmark directory
  7  no symlink under src/ or deps/ escapes the benchmark directory
  8  provenance/source.lock[.variant].yaml: schema hpcperf-source-lock-2, complete, no node-private location
  9  benchmark.yaml identity == lock (artifact filename, archive sha256, tree sha256, size); no scheme-2 fields
 10  required inputs/references listed in benchmark.yaml exist
 11  no credential-looking file / env dump / session record in src+deps (name rules; content rules unless --quick)
 12  no build output / binary artifacts inside src+deps
 13  the scripts do not depend on _upstream paths
 14  artifact publish status consistent (unpublished => no URL; published => immutable https) and the
     materialization marker agrees with the lock (variant, tree hash); no Git LFS metadata
 15  redistribution_status and suite_status declared
"""
import argparse
import json
import os
import re
import stat
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hpcperf_source as hs  # noqa: E402
import hpcperf_lock as hl  # noqa: E402

REQUIRED_BY = ("name", "level", "application", "supported_backends", "validated_backends", "build_entry", "run_entry", "validate_entry")
FORBIDDEN_SRC_REFS = [
    ("upstream-checkout", re.compile(r"_upstream/")),
    ("other-repository-path", re.compile(r"/projects/|/home/[A-Za-z0-9]|\$HOME/|~/")),
    ("parent-directory-source", re.compile(r"\.\./\.\./(level3|_upstream|\.deps)/[^ ]*/src")),
]
SOURCE_TOPS = ("src/", "deps/")


def declared_assets(by):
    """inputs/references of benchmark.yaml, normalised (these stay protected even when they live under src/)."""
    out = []
    for p in (by.get("inputs") or []) + (by.get("references") or []):
        p = str(p).strip().lstrip("./").rstrip("/")
        if p:
            out.append(p)
    return out


def is_under(path, prefix):
    return path == prefix or path.startswith(prefix.rstrip("/") + "/")


def classify(path, assets):
    """source (src/**, deps/**, except declared inputs/references) or protected (everything else)"""
    if path.startswith(SOURCE_TOPS) and not any(is_under(path, a) for a in assets):
        return "source"
    return "protected"


def workspace_root(D):
    return os.path.abspath(os.path.join(D, "..", ".."))


def find_baseline(D, explicit, allow_workspace=False):
    """(path, origin, problem): trusted baseline = explicit file outside the workspace root > the repository copy
    <repo>/.hpcperf/workspace_baselines/<run-id>.json > (only with allow_workspace) <workspace-root>/workspace_baseline.json"""
    wsr = workspace_root(D)
    def inside(p):
        p = os.path.abspath(p)
        return p == wsr or p.startswith(wsr + os.sep)
    if explicit:
        if inside(explicit) and not allow_workspace:
            return None, "explicit", f"--baseline {explicit} lies inside the workspace root (not trusted; pass --allow-workspace-baseline for development)"
        return (explicit, "explicit", None) if os.path.isfile(explicit) else (None, "explicit", f"--baseline {explicit} missing")
    ws = hs.load_yaml(os.path.join(D, "workspace.yaml")) if os.path.isfile(os.path.join(D, "workspace.yaml")) else {}
    run_id = ws.get("run_id")
    root = os.path.abspath(os.path.join(HERE, ".."))
    if run_id:
        p = os.path.join(root, ".hpcperf", "workspace_baselines", f"{run_id}.json")
        if os.path.isfile(p):
            return p, "repository", None
    p = os.path.join(wsr, "workspace_baseline.json")
    if os.path.isfile(p) and allow_workspace:
        return p, "workspace (development, NOT trusted)", None
    return None, None, ("no trusted baseline: the repository copy .hpcperf/workspace_baselines/<run-id>.json is absent and the copy "
                        "inside the workspace is not trusted (--allow-workspace-baseline for development only)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench_dir"); ap.add_argument("--variant"); ap.add_argument("--json"); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--agent-mode", action="store_true"); ap.add_argument("--baseline"); ap.add_argument("--iteration", type=int)
    ap.add_argument("--report"); ap.add_argument("--diff"); ap.add_argument("--allow-workspace-baseline", action="store_true")
    a = ap.parse_args()
    D = os.path.abspath(a.bench_dir)
    results = []

    def rec(no, name, ok, detail=""):
        results.append({"check": no, "name": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        print(f"{'ok  ' if ok else 'FAIL'} {no:>2}: {name}{' -- ' + detail if detail else ''}")

    by = {}
    by_path = os.path.join(D, "benchmark.yaml")
    if os.path.isfile(by_path):
        try:
            by = hs.load_yaml(by_path) or {}
            missing = [k for k in REQUIRED_BY if k not in by]
            rec(1, "benchmark.yaml schema", not missing, f"missing keys: {missing}" if missing else f"{by.get('name')} level {by.get('level')}")
        except Exception as ex:  # noqa: BLE001
            rec(1, "benchmark.yaml schema", False, f"unreadable: {ex}")
    else:
        rec(1, "benchmark.yaml schema", False, "file missing")
    if by.get("validated_backends") and "hip" in [str(b).lower() for b in by.get("validated_backends", [])]:
        rec(1, "benchmark.yaml: hip must not be listed as validated on this node", False, "hip in validated_backends")

    marker = hl.read_marker(D)
    try:
        variant = hl.select_variant(by, a.variant, marker)
    except hs.SourceError as ex:
        variant = a.variant or marker.get("variant"); rec(0, "variant selection", False, str(ex))
    if marker.get("variant") and variant != marker.get("variant"):
        rec(0, "requested variant matches the materialized one", False, f"requested {variant}, materialized {marker.get('variant')}")
    ident = hl.identity_from_benchmark(by, variant)
    expected = ident.get("source_tree_sha256") or (by.get("source_tree_sha256") if not variant else None)
    lock_p = hl.lock_path(D, variant)
    lock = (hs.load_yaml(lock_p) or {}) if os.path.isfile(lock_p) else {}
    lock_v2 = lock.get("schema") == hl.LOCK_SCHEMA

    src_ok = os.path.isdir(os.path.join(D, "src")) and not os.path.islink(os.path.join(D, "src"))
    rec(2, "src/ materialized", src_ok, "" if src_ok else "run tools/prepare_benchmark.sh level3 <app>")
    layout = (lock.get("materialized_tree") or {}).get("layout", ["src/"])
    need_deps = "deps/" in layout
    deps_ok = (not need_deps) or (os.path.isdir(os.path.join(D, "deps")) and not os.path.islink(os.path.join(D, "deps")))
    rec(3, "deps/ present as required by the lock", deps_ok, "required by the lock layout" if need_deps else "not required for this benchmark")

    entries = []
    present = [d for d in ("src", "deps") if os.path.isdir(os.path.join(D, d))]
    tree = None
    if present:
        try:
            entries = hs.manifest(D, present); tree = hs.tree_hash_from_manifest(entries)
        except hs.SourceError as ex:
            rec(4, "source identity", False, str(ex))
    report = None
    if tree is None and not a.agent_mode:
        if not results or results[-1]["check"] != 4:
            rec(4, "source tree hash == canonical baseline", False, "no src/deps to hash")
    elif not a.agent_mode:
        rec(4, "source tree hash == canonical baseline", bool(expected) and tree == expected, f"{tree[:16]}... vs {str(expected)[:16]}...")
    else:
        bp, borigin, bproblem = find_baseline(D, a.baseline, a.allow_workspace_baseline)
        if not bp:
            rec(4, "agent mode: protected files unchanged vs the trusted baseline", False, bproblem)
        else:
            base = json.load(open(bp))
            bfiles = base.get("files", {})
            # harness files of the workspace root (env, launcher, level3 helpers): the scripts inside the benchmark
            # directory source them, so they are part of the trusted surface even though they are outside the agent cwd
            wsr = workspace_root(D)
            hviol = []
            for rel, b in (base.get("harness_files") or {}).items():
                pth = os.path.join(wsr, rel)
                if not os.path.isfile(pth):
                    hviol.append(f"{rel} (deleted)")
                elif hs.sha256_file(pth) != b["sha256"]:
                    hviol.append(f"{rel} (modified)")
            try:
                cur = {e["path"]: e for e in hs.manifest(D)}
            except hs.SourceError as ex:
                cur = {}; rec(4, "agent mode: protected files unchanged vs the trusted baseline", False, str(ex))
            if cur:
                assets = declared_assets(by)
                classes = {"modified": [], "added": [], "deleted": [], "violations": []}
                for pth, b in bfiles.items():
                    c = cur.get(pth)
                    kind = classify(pth, assets)
                    if c is None:
                        (classes["deleted"] if kind == "source" else classes["violations"]).append({"path": pth, "change": "deleted", "class": kind})
                    elif c["sha256"] != b["sha256"] or c["type"] != b["type"]:
                        (classes["modified"] if kind == "source" else classes["violations"]).append({"path": pth, "change": "modified", "class": kind})
                for pth in cur:
                    if pth not in bfiles:
                        kind = classify(pth, assets)
                        (classes["added"] if kind == "source" else classes["violations"]).append({"path": pth, "change": "added", "class": kind})
                v = classes["violations"]
                detail = ("PROTECTED FILE TAMPERING: " + "; ".join(f"{x['path']} ({x['change']})" for x in v[:5])) if v else ""
                if hviol:
                    detail = (detail + "; " if detail else "") + "HARNESS TAMPERING: " + "; ".join(hviol[:5])
                rec(4, "agent mode: protected files (harness, inputs, references, provenance, metadata) unchanged vs the trusted baseline",
                    not v and not hviol, detail or
                    f"{len(classes['modified'])} modified, {len(classes['added'])} added, {len(classes['deleted'])} deleted file(s) inside the source tree (src/, deps/); baseline {os.path.relpath(bp)} ({borigin})")
                src_paths = [p for p in cur if p.startswith(SOURCE_TOPS)]
                cur_tree = hs.tree_hash_from_manifest([cur[p] for p in sorted(src_paths, key=lambda s: s.encode())])
                report = {"schema": "hpcperf-workspace-check-2", "mode": "agent", "iteration": a.iteration, "baseline": os.path.relpath(bp), "baseline_origin": borigin,
                          "source_surface": ["src/**", "deps/**"], "protected_inside_source": [p for p in assets if p.startswith(SOURCE_TOPS)],
                          "harness_violations": hviol,
                          "run_id": base.get("run_id"), "benchmark": base.get("benchmark"), "variant": variant,
                          "initial_source_hash": base.get("canonical_source_tree_sha256"), "current_source_hash": cur_tree,
                          "modified_files": [x["path"] for x in classes["modified"]], "added_files": [x["path"] for x in classes["added"]],
                          "deleted_files": [x["path"] for x in classes["deleted"]], "protected_violations": v}
                if a.diff:
                    ws = hs.load_yaml(os.path.join(D, "workspace.yaml")) if os.path.isfile(os.path.join(D, "workspace.yaml")) else {}
                    can = ws.get("canonical_dir")
                    with open(a.diff, "w") as f:
                        if can and os.path.isdir(can):
                            for x in classes["modified"] + classes["added"] + classes["deleted"]:
                                old = os.path.join(can, x["path"]); new = os.path.join(D, x["path"])
                                r = subprocess.run(["diff", "-u", "--label", f"baseline/{x['path']}", "--label", f"workspace/{x['path']}",
                                                    old if os.path.isfile(old) else "/dev/null", new if os.path.isfile(new) else "/dev/null"], capture_output=True, text=True)
                                f.write(r.stdout)
                            report["diff"] = a.diff
                        else:
                            f.write(f"# diff unavailable: canonical directory {can!r} not reachable from this workspace (hashes recorded in the report)\n")
                            report["diff"] = "unavailable (canonical directory not reachable)"

    scripts = [by.get("build_entry", "./build.sh"), by.get("run_entry", "./run.sh"), by.get("validate_entry", "./validate.sh")]
    exist = [s for s in scripts if os.path.isfile(os.path.join(D, s)) and os.stat(os.path.join(D, s)).st_mode & stat.S_IXUSR]
    rec(5, "build/run/validate entries exist and are executable", len(exist) == 3, ", ".join(scripts))

    bad_refs = []
    for s in scripts:
        p = os.path.join(D, s)
        if not os.path.isfile(p):
            continue
        for i, line in enumerate(open(p, errors="replace"), 1):
            if line.lstrip().startswith("#"):
                continue
            for rule, rx in FORBIDDEN_SRC_REFS:
                if rx.search(line):
                    bad_refs.append(f"{s}:{i} [{rule}]")
    rec(6, "scripts read no application source outside the benchmark directory", not bad_refs, "; ".join(bad_refs[:6]))

    esc = hs.escaping_symlinks(D, entries) if entries else []
    rec(7, "no symlink escaping the benchmark directory", not esc, "; ".join(f"{e['path']} -> {e['target']}" for e in esc[:4]))

    if not lock:
        rec(8, f"provenance/source.lock{hl.suffix(variant)}.yaml valid", False, "file missing")
    elif not lock_v2:
        rec(8, f"provenance/source.lock{hl.suffix(variant)}.yaml valid", False, f"schema {lock.get('schema')!r} (scheme-2 lock; migrate it)")
    else:
        problems = hl.validate_lock(lock, open(lock_p).read())
        rec(8, f"provenance/source.lock{hl.suffix(variant)}.yaml valid (schema 2, complete, no node-private location)", not problems, "; ".join(problems[:4]))

    ok10, detail = False, []
    if lock_v2:
        art = lock["artifact"]
        ok10 = (ident.get("filename") == art["filename"] and ident.get("archive_sha256") == art["sha256"] and ident.get("source_tree_sha256") == art["source_tree_sha256"]
                and int(ident.get("size") or -1) == art["size"] and ident.get("source_version") == lock["benchmark"]["source_version"])
        detail.append("identity fields agree" if ok10 else "benchmark.yaml identity differs from the lock")
        for k in ("source_bundle",):
            if k in by:
                ok10 = False; detail.append(f"scheme-2 key {k} present")
        if variant and any(k in ident for k in ("archive", "compressed_size")):
            ok10 = False; detail.append("scheme-2 variant keys present")
        if os.path.isdir(os.path.join(D, "archives")):
            ok10 = False; detail.append("archives/ directory present (scheme 2)")
    rec(9, "benchmark.yaml identity == source lock; no scheme-2 fields", ok10, "; ".join(detail))

    req = [p for p in (by.get("inputs", []) + by.get("references", []))]
    missing = [p for p in req if not os.path.exists(os.path.join(D, p))]
    rec(10, "required inputs/references exist", not missing, f"{len(req)} listed" + (f"; missing: {missing[:4]}" if missing else ""))

    hits = []
    if entries:
        if a.quick:
            for e in entries:
                for rule, rx in hs.NAME_RULES:
                    if rx.search(e["path"]):
                        hits.append({"path": e["path"], "rule": rule})
        else:
            hits = hs.scan_tree(D, entries, allow=lock.get("scan_allow", []))
    secret_hits = [h for h in hits if h["rule"] != "build-output"]
    build_hits = [h for h in hits if h["rule"] == "build-output"]
    rec(11, "no credential/env-dump/session/profiler files in src+deps", not secret_hits, "; ".join(f"{h['rule']}:{h['path']}" for h in secret_hits[:4]))
    rec(12, "no build output / binary artifacts in src+deps", not build_hits, "; ".join(h["path"] for h in build_hits[:4]))

    ups = []
    for root_dir, dirs, files in os.walk(D):
        dirs[:] = [d for d in dirs if d not in ("src", "deps", "provenance", "__pycache__")]
        for fn in files:
            if fn.endswith((".sh", ".py")):
                p = os.path.join(root_dir, fn)
                for i, line in enumerate(open(p, errors="replace"), 1):
                    if "_upstream/" in line and not line.lstrip().startswith("#") and "fetch.sh" not in fn:
                        ups.append(f"{os.path.relpath(p, D)}:{i}")
    rec(13, "no _upstream dependency in the benchmark scripts (fetch.sh = freeze-time only)", not ups, "; ".join(ups[:5]))

    ok16, d16 = False, []
    if lock_v2:
        prim = lock["artifact"].get("primary") or {}
        ok16 = prim.get("status") in hl.PUBLISH_STATUSES and ((prim.get("status") == "unpublished") == (not prim.get("url")))
        d16.append(f"publish status {prim.get('status')}{' (' + prim['url'] + ')' if prim.get('url') else ''}")
        if marker:
            m_ok = marker.get("source_tree_sha256") == lock["artifact"]["source_tree_sha256"] and (marker.get("variant") or None) == (variant or None)
            ok16 = ok16 and m_ok; d16.append("marker agrees with the lock" if m_ok else "materialization marker disagrees with the lock")
        if any(k in lock for k in ("lfs", "git_lfs")) or any(k in lock["artifact"] for k in ("lfs", "lfs_pointer")):
            ok16 = False; d16.append("Git LFS metadata present")
    rec(14, "artifact publish status consistent, marker agrees with the lock, no LFS metadata", ok16, "; ".join(d16))

    ok17 = by.get("redistribution_status") in hl.REDISTRIBUTION_STATUSES and by.get("suite_status") in hl.SUITE_STATUSES and (not lock_v2 or lock.get("redistribution_status") == by.get("redistribution_status"))
    rec(15, "redistribution_status and suite_status declared (benchmark.yaml == lock)", ok17, f"redistribution {by.get('redistribution_status')}, suite {by.get('suite_status')}")

    fails = [r for r in results if r["status"] == "FAIL"]
    verdict = "PASS" if not fails else "FAIL"
    mode = "agent" if a.agent_mode else "baseline"
    print(f"check_workspace: {verdict} ({len(results) - len(fails)}/{len(results)} checks, {mode} mode) {os.path.relpath(D)}{' variant=' + variant if variant else ''}")
    root = os.path.abspath(os.path.join(HERE, ".."))
    rel = os.path.relpath(D, root) if D.startswith(root + os.sep) else os.path.basename(D)
    if report is not None:
        report["verdict"] = verdict
        if a.report:
            with open(a.report, "w") as f:
                json.dump(report, f, indent=1)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"benchmark_dir": rel, "variant": variant, "mode": mode, "verdict": verdict, "results": results, **({"agent_report": report} if report else {})}, f, indent=1)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
