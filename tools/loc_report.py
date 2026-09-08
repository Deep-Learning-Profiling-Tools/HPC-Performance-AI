#!/usr/bin/env python3
"""loc_report -- code LOC of a materialized Level 3 benchmark by source ownership (cloc code lines: no blank,
no comment lines; documentation, build output, binaries, datasets never counted).

    loc_report.py level3/<app> [--cloc PATH] [--json OUT] [--md OUT]

Categories come from optimization_scope.yaml (`loc_categories`):
    application_owned      globs of the application's own source (src/...)
    bundled_dependency     globs of third-party source shipped inside the upstream tree (src/lib/kokkos, ...)
    benchmark_specific_dependency  globs under deps/ (git trees) -- tarballs under deps/ are extracted to a
                                   temporary directory and counted there
    test                   globs of test code (reported separately, never inside application_owned)
    exclude                globs never counted (docs, examples/data, ...)
plus the agent-modifiable set = files matching `modifiable` minus `readonly`/`excluded` of the scope.
total_materialized_code_loc = application_owned + bundled_dependency + benchmark_specific_dependency (+ test).
"""
import argparse
import fnmatch
import json
import os
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hpcperf_source as hs  # noqa: E402


def match_any(path, globs):
    return any(fnmatch.fnmatchcase(path, g) or fnmatch.fnmatchcase(path, g.rstrip("/") + "/*") for g in globs)


def run_cloc(cloc, args):
    """cloc with --json written to a file (cloc prints warnings to stdout, which would corrupt inline JSON)."""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as jf:
        out_path = jf.name
    try:
        subprocess.run([cloc, "--json", "--quiet", "--skip-uniqueness", f"--out={out_path}"] + args, capture_output=True, text=True, check=False)
        text = open(out_path).read() if os.path.exists(out_path) else ""
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)
    if not text.strip():
        return {"code": 0, "files": 0, "languages": {}}
    data = json.loads(text)
    langs = {k: v["code"] for k, v in data.items() if k not in ("header", "SUM")}
    s = data.get("SUM", {})
    return {"code": int(s.get("code", 0)), "files": int(s.get("nFiles", 0)), "languages": dict(sorted(langs.items(), key=lambda kv: -kv[1]))}


def cloc_files(cloc, root, rel_paths):
    if not rel_paths:
        return {"code": 0, "files": 0, "languages": {}}
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".lst") as lf:
        for p in rel_paths:
            lf.write(os.path.join(root, p) + "\n")
        lst = lf.name
    try:
        return run_cloc(cloc, [f"--list-file={lst}"])
    finally:
        os.remove(lst)


def cloc_dir(cloc, d):
    return run_cloc(cloc, [d])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench_dir"); ap.add_argument("--cloc", default=os.environ.get("CLOC", "cloc")); ap.add_argument("--json"); ap.add_argument("--md")
    a = ap.parse_args()
    D = os.path.abspath(a.bench_dir)
    scope = hs.load_yaml(os.path.join(D, "optimization_scope.yaml"))
    cats = scope.get("loc_categories", {})
    excl = cats.get("exclude", []) or []
    present = [d for d in ("src", "deps") if os.path.isdir(os.path.join(D, d))]
    files = [e["path"] for e in hs.manifest(D, present) if e["type"] == "F" and not match_any(e["path"], excl)]
    res = {"benchmark": os.path.basename(D), "cloc": subprocess.run([a.cloc, "--version"], capture_output=True, text=True).stdout.strip()}
    used = set()
    for key in ("test", "bundled_dependency", "application_owned"):
        globs = cats.get(key, []) or []
        sel = [p for p in files if p not in used and match_any(p, globs)]
        used.update(sel)
        res[f"{key}_code_loc"] = cloc_files(a.cloc, D, sel)
    # benchmark-specific dependencies: git trees under deps/ counted in place, tarballs extracted temporarily
    dep_globs = cats.get("benchmark_specific_dependency", ["deps/*"]) or []
    dep_files = [p for p in files if p not in used and match_any(p, dep_globs)]
    used.update(dep_files)
    tarballs = [p for p in dep_files if p.endswith((".tar.gz", ".tgz", ".tar.xz", ".tar.bz2", ".zip"))]
    plain = [p for p in dep_files if p not in tarballs]
    dep = cloc_files(a.cloc, D, plain)
    per_tarball = {}
    for tb in tarballs:
        with tempfile.TemporaryDirectory(prefix="loc-", dir=os.environ.get("HPCPERF_FREEZE_SCRATCH", "/tmp")) as td:
            p = os.path.join(D, tb)
            if tb.endswith(".zip"):
                import zipfile
                with zipfile.ZipFile(p) as z:
                    z.extractall(td)
            else:
                with tarfile.open(p) as t:
                    t.extractall(td, filter="data")
            r = cloc_dir(a.cloc, td)
        per_tarball[tb] = r["code"]
        dep["code"] += r["code"]; dep["files"] += r["files"]
        for k, v in r["languages"].items():
            dep["languages"][k] = dep["languages"].get(k, 0) + v
    dep["tarballs"] = per_tarball
    res["benchmark_specific_dependency_code_loc"] = dep
    res["unclassified"] = cloc_files(a.cloc, D, [p for p in files if p not in used])
    mod = [p for p in files if match_any(p, scope.get("modifiable", []) or []) and not match_any(p, (scope.get("readonly", []) or []) + (scope.get("excluded", []) or []))]
    res["agent_modifiable_code_loc"] = cloc_files(a.cloc, D, mod)
    res["total_materialized_code_loc"] = sum(res[k]["code"] for k in ("application_owned_code_loc", "bundled_dependency_code_loc", "benchmark_specific_dependency_code_loc", "test_code_loc", "unclassified"))
    summary = {k: res[k]["code"] for k in ("application_owned_code_loc", "bundled_dependency_code_loc", "benchmark_specific_dependency_code_loc", "test_code_loc", "agent_modifiable_code_loc")}
    summary["total_materialized_code_loc"] = res["total_materialized_code_loc"]; summary["unclassified_code_loc"] = res["unclassified"]["code"]
    res["summary"] = summary
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    md = [f"# LOC ({res['benchmark']}) -- cloc code lines", "", "| category | code LOC | files |", "|---|---|---|"]
    for k in ("application_owned_code_loc", "bundled_dependency_code_loc", "benchmark_specific_dependency_code_loc", "test_code_loc", "unclassified", "agent_modifiable_code_loc"):
        md.append(f"| {k} | {res[k]['code']} | {res[k]['files']} |")
    md.append(f"| **total_materialized_code_loc** | **{res['total_materialized_code_loc']}** | |")
    md.append(""); md.append("Top languages (application_owned): " + ", ".join(f"{k} {v}" for k, v in list(res["application_owned_code_loc"]["languages"].items())[:6]))
    if a.md:
        with open(a.md, "w") as f:
            f.write("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
