#!/usr/bin/env python3
"""check_workspace -- static contract checks of a Level 3 benchmark directory (canonical level3/<app> or an
agent workspace copy). Iteration 0 of any optimization run must PASS this check. No build is run.

    check_workspace.py <benchmark-dir> [--variant NAME] [--json OUT] [--quick]

Checks (each reported PASS/FAIL, all must pass):
  1  benchmark.yaml exists and carries the required keys
  2  optimization_scope.yaml exists with modifiable/readonly lists
  3  src/ exists (materialized)
  4  deps/ present when the lock file's layout requires it
  5  source tree hash of src/(+deps/) equals the canonical baseline (benchmark.yaml / source lock)
  6  build.sh, run.sh, validate.sh exist and are executable
  7  build.sh/run.sh/validate.sh reference no application source outside the benchmark directory
  8  no symlink under src/ or deps/ escapes the benchmark directory
  9  provenance/source.lock[.variant].yaml complete
 10  archive sha256 (when the archive is present and not an LFS pointer) and tree sha256 match the lock
 11  required inputs/references listed in benchmark.yaml exist
 12  every optimization-scope pattern points inside the benchmark directory
 13  no credential-looking file / env dump / session record in src+deps (name rules; content rules unless --quick)
 14  no build output / binary artifacts inside src+deps
 15  the scripts do not depend on _upstream paths
"""
import argparse
import json
import os
import re
import stat
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hpcperf_source as hs  # noqa: E402

REQUIRED_BY = ("name", "level", "application", "supported_backends", "validated_backends", "build_entry", "run_entry", "validate_entry", "optimization_scope")
FORBIDDEN_SRC_REFS = [
    ("upstream-checkout", re.compile(r"_upstream/")),
    ("other-repository-path", re.compile(r"/projects/|/home/[A-Za-z0-9]|\$HOME/|~/")),
    ("parent-directory-source", re.compile(r"\.\./\.\./(level3|_upstream|\.deps)/[^ ]*/src")),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench_dir"); ap.add_argument("--variant"); ap.add_argument("--json"); ap.add_argument("--quick", action="store_true")
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
    if by.get("validated_backends") and "hip" in [b.lower() for b in by.get("validated_backends", [])]:
        rec(1, "benchmark.yaml: hip must not be listed as validated on this node", False, "hip in validated_backends")

    scope = {}
    sp = os.path.join(D, by.get("optimization_scope", "./optimization_scope.yaml"))
    if os.path.isfile(sp):
        scope = hs.load_yaml(sp) or {}
        ok = isinstance(scope.get("modifiable"), list) and isinstance(scope.get("readonly"), list)
        rec(2, "optimization_scope.yaml", ok, f"{len(scope.get('modifiable', []))} modifiable / {len(scope.get('readonly', []))} readonly patterns" if ok else "modifiable/readonly lists missing")
    else:
        rec(2, "optimization_scope.yaml", False, "file missing")

    variant = a.variant
    bundle, expected = None, None
    marker_path = os.path.join(D, ".hpcperf-materialized.yaml")
    marker = hs.load_yaml(marker_path) if os.path.isfile(marker_path) else {}
    if by.get("variants"):
        env = by.get("variant_env")
        # the materialization marker says which variant IS in src/; an explicit --variant must agree with it
        variant = variant or marker.get("variant") or (os.environ.get(env) if env else None) or by.get("default_variant")
        if marker.get("variant") and variant != marker.get("variant"):
            rec(0, "requested variant matches the materialized one", False, f"requested {variant}, materialized {marker.get('variant')}")
        v = by["variants"].get(variant) if variant else None
        if v:
            bundle, expected = v, v.get("source_tree_sha256")
    else:
        bundle, expected = by.get("source_bundle"), by.get("source_tree_sha256")
    suffix = f".{variant}" if variant else ""
    lock_path = os.path.join(D, "provenance", f"source.lock{suffix}.yaml")
    lock = hs.load_yaml(lock_path) if os.path.isfile(lock_path) else {}

    src_ok = os.path.isdir(os.path.join(D, "src")) and not os.path.islink(os.path.join(D, "src"))
    rec(3, "src/ materialized", src_ok, "" if src_ok else "run tools/prepare_benchmark.sh level3 <app>")
    layout = lock.get("materialized_tree", {}).get("layout", ["src/"])
    need_deps = "deps/" in layout
    deps_ok = (not need_deps) or (os.path.isdir(os.path.join(D, "deps")) and not os.path.islink(os.path.join(D, "deps")))
    rec(4, "deps/ present as required by the lock", deps_ok, "required by the lock layout" if need_deps else "not required for this benchmark")

    entries = []
    present = [d for d in ("src", "deps") if os.path.isdir(os.path.join(D, d))]
    if present:
        entries = hs.manifest(D, present)
        tree = hs.tree_hash_from_manifest(entries)
        rec(5, "source tree hash == canonical baseline", bool(expected) and tree == expected, f"{tree[:16]}... vs {str(expected)[:16]}...")
    else:
        rec(5, "source tree hash == canonical baseline", False, "no src/deps to hash")

    scripts = [by.get("build_entry", "./build.sh"), by.get("run_entry", "./run.sh"), by.get("validate_entry", "./validate.sh")]
    exist = [s for s in scripts if os.path.isfile(os.path.join(D, s)) and os.stat(os.path.join(D, s)).st_mode & stat.S_IXUSR]
    rec(6, "build/run/validate entries exist and are executable", len(exist) == 3, ", ".join(scripts))

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
    rec(7, "scripts read no application source outside the benchmark directory", not bad_refs, "; ".join(bad_refs[:6]))

    esc = hs.escaping_symlinks(D, entries) if entries else []
    rec(8, "no symlink escaping the benchmark directory", not esc, "; ".join(f"{e['path']} -> {e['target']}" for e in esc[:4]))

    need = ("upstream", "archive", "materialized_tree", "patches", "dependencies", "freeze_timestamp")
    lock_ok = bool(lock) and all(k in lock for k in need)
    rec(9, f"provenance/source.lock{suffix}.yaml complete", lock_ok, "" if lock_ok else f"missing: {[k for k in need if k not in lock]}" if lock else "file missing")

    detail = []
    ok10 = bool(lock) and bool(expected) and lock.get("materialized_tree", {}).get("sha256") == expected
    if bundle and bundle.get("archive"):
        arch = os.path.join(D, bundle["archive"])
        if os.path.isfile(arch) and not hs.is_lfs_pointer(arch):
            got = hs.sha256_file(arch)
            ok10 = ok10 and got == bundle.get("archive_sha256") == lock.get("archive", {}).get("sha256")
            detail.append(f"archive sha256 {'ok' if got == bundle.get('archive_sha256') else 'MISMATCH'}")
        elif os.path.isfile(arch):
            detail.append("archive is an LFS pointer (not verified here)")
        else:
            detail.append("archive absent in this directory (workspace copies omit it)")
    rec(10, "archive/tree hashes consistent between benchmark.yaml and the lock", ok10, "; ".join(detail))

    req = [p for p in (by.get("inputs", []) + by.get("references", []))]
    missing = [p for p in req if not os.path.exists(os.path.join(D, p))]
    rec(11, "required inputs/references exist", not missing, f"{len(req)} listed" + (f"; missing: {missing[:4]}" if missing else ""))

    bad_scope = []
    for key in ("modifiable", "readonly", "excluded"):
        for pat in scope.get(key, []) or []:
            base = re.split(r"[*?\[]", pat, 1)[0]
            if os.path.isabs(pat) or ".." in pat.split("/"):
                bad_scope.append(pat); continue
            base_dir = os.path.join(D, base.rstrip("/") if base else ".")
            if base and not os.path.exists(base_dir) and not os.path.exists(os.path.dirname(base_dir)):
                bad_scope.append(pat)
    rec(12, "optimization-scope patterns lie inside the benchmark directory", not bad_scope, "; ".join(bad_scope[:5]))

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
    rec(13, "no credential/env-dump/session/profiler files in src+deps", not secret_hits, "; ".join(f"{h['rule']}:{h['path']}" for h in secret_hits[:4]))
    rec(14, "no build output / binary artifacts in src+deps", not build_hits, "; ".join(h["path"] for h in build_hits[:4]))

    ups = []
    for root_dir, dirs, files in os.walk(D):
        dirs[:] = [d for d in dirs if d not in ("src", "deps", "archives", "provenance", "__pycache__")]
        for fn in files:
            if fn.endswith((".sh", ".py")):
                p = os.path.join(root_dir, fn)
                for i, line in enumerate(open(p, errors="replace"), 1):
                    if "_upstream/" in line and not line.lstrip().startswith("#") and "fetch.sh" not in fn:
                        ups.append(f"{os.path.relpath(p, D)}:{i}")
    rec(15, "no _upstream dependency in the benchmark scripts (fetch.sh = freeze-time only)", not ups, "; ".join(ups[:5]))

    fails = [r for r in results if r["status"] == "FAIL"]
    verdict = "PASS" if not fails else "FAIL"
    print(f"check_workspace: {verdict} ({len(results) - len(fails)}/{len(results)} checks) {os.path.relpath(D)}{' variant=' + variant if variant else ''}")
    if a.json:
        root = os.path.abspath(os.path.join(HERE, ".."))
        rel = os.path.relpath(D, root) if D.startswith(root + os.sep) else D
        with open(a.json, "w") as f:
            json.dump({"benchmark_dir": rel, "variant": variant, "verdict": verdict, "results": results}, f, indent=1)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
