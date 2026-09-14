#!/usr/bin/env python3
"""loc_report -- code LOC of a materialized Level 3 benchmark by source ownership (cloc code lines: no blank,
no comment lines; documentation, build output, binaries, datasets never counted).

    loc_report.py level3/<app> [--variant NAME] [--cloc PATH] [--json OUT] [--md OUT]

The classification is descriptive metadata about the frozen source, read from the source lock
(`provenance/source.lock[.variant].yaml`, key `source_scope`, written at freeze time from the freeze spec):
    application_owned      globs of the application's own source (src/...)
    bundled                globs of third-party source shipped inside the upstream tree (src/lib/kokkos, ...)
    benchmark_specific     globs under deps/ (git trees) -- tarballs under deps/ are extracted to a temporary
                           directory and counted there
    test                   globs of test code (reported separately, never inside application_owned)
    exclude                globs never counted (docs, examples/data, ...)
total_materialized_code_loc = application_owned + bundled + benchmark_specific + test + unclassified: everything
the artifact unpacks that is code. Dependencies are counted per benchmark, so totals overlap across benchmarks
that ship the same dependency (AMReX in WarpX and Nyx). These figures describe benchmark size and source
ownership; they do not describe what an optimization agent may modify -- the benchmark does not define that.
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
import hpcperf_lock as hl  # noqa: E402

CATEGORY_KEYS = (("test", "test_code_loc"), ("bundled", "bundled_dependency_code_loc"), ("application_owned", "application_owned_code_loc"))


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
    ap.add_argument("bench_dir"); ap.add_argument("--variant"); ap.add_argument("--cloc", default=os.environ.get("CLOC", "cloc"))
    ap.add_argument("--json"); ap.add_argument("--md")
    a = ap.parse_args()
    D = os.path.abspath(a.bench_dir)
    by_p = os.path.join(D, "benchmark.yaml")
    by = hs.load_yaml(by_p) if os.path.isfile(by_p) else {}
    marker = hl.read_marker(D)
    variant = hl.select_variant(by, a.variant, marker)
    if marker.get("variant") and variant != marker.get("variant"):
        sys.exit(f"loc_report: variant {variant} requested but {marker.get('variant')} is materialized")
    lock_p = hl.lock_path(D, variant)
    if not os.path.isfile(lock_p):
        sys.exit(f"loc_report: {lock_p} missing (the source lock carries the source_scope classification)")
    lock = hs.load_yaml(lock_p) or {}
    cats = lock.get("source_scope") or {}
    if not cats:
        sys.exit(f"loc_report: {lock_p} has no source_scope block")
    excl = cats.get("exclude", []) or []
    present = [d for d in ("src", "deps") if os.path.isdir(os.path.join(D, d))]
    files = [e["path"] for e in hs.manifest(D, present) if e["type"] == "F" and not match_any(e["path"], excl)]
    res = {"benchmark": os.path.basename(D), "variant": variant,
           "cloc": subprocess.run([a.cloc, "--version"], capture_output=True, text=True).stdout.strip(),
           "categories_source": os.path.relpath(lock_p, D) + " (source_scope; descriptive source ownership, not an optimization policy)",
           "source_tree_sha256": (lock.get("artifact") or {}).get("source_tree_sha256")}
    used = set()
    for key, out_key in CATEGORY_KEYS:
        globs = cats.get(key, []) or []
        sel = [p for p in files if p not in used and match_any(p, globs)]
        used.update(sel)
        res[out_key] = cloc_files(a.cloc, D, sel)
    # benchmark-specific dependencies: git trees under deps/ counted in place, tarballs extracted temporarily
    dep_globs = cats.get("benchmark_specific", ["deps/*"]) or []
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
    parts = ("application_owned_code_loc", "bundled_dependency_code_loc", "benchmark_specific_dependency_code_loc", "test_code_loc", "unclassified")
    res["total_materialized_code_loc"] = sum(res[k]["code"] for k in parts)
    summary = {k: res[k]["code"] for k in parts if k != "unclassified"}
    summary["total_materialized_code_loc"] = res["total_materialized_code_loc"]; summary["unclassified_code_loc"] = res["unclassified"]["code"]
    res["summary"] = summary
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    md = [f"# LOC ({res['benchmark']}{' / ' + variant if variant else ''}) -- cloc code lines", "",
          f"Classification: `{res['categories_source']}`. cloc code lines exclude blank and comment lines; documentation, examples/data and build output are not counted. "
          "Dependencies are counted per benchmark (totals overlap across benchmarks sharing a dependency). Size and ownership only: no line of this table says what an optimization agent may modify.", "",
          "| category | code LOC | files |", "|---|---|---|"]
    for k in parts:
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
