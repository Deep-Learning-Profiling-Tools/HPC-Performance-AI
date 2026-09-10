#!/usr/bin/env python3
"""freeze_benchmark_source -- turn "exact upstream source + approved patch series (+ pinned dependency
sources)" into a Level 3 source bundle with a recorded identity.

    freeze_benchmark_source.py level3/<app> [--spec provenance/freeze_spec[.variant].yaml] [--variant NAME]
                               [--stage-root DIR] [--keep-stage] [--skip-equivalence] [--no-archive]
                               [--verify-determinism] [--allow-unexpected]

Inputs are ONLY what the spec declares: committed HEAD blobs of pinned git checkouts (never the working
tree, so in-place edits cannot leak), declared submodule checkouts, and pinned dependency files verified
by SHA-256 (copied from the recorded download, or downloaded from the recorded URL when the local copy is
gone). Nothing under .deps/ or the worktree is tarred wholesale. Patches of the approved series are
applied in the staging tree in order; the bundle therefore IS the patched baseline (build.sh applies no
patch). The staging tree is scanned for credential-looking files/contents and build artifacts (any hit
fails the freeze; only paths and rule names are printed), checked for symlinks escaping the tree, hashed
(source_tree_sha256, algorithm hpcperf-tree-1), compared with the tree the recorded results were validated
from (compare_source_trees; an UNEXPECTED difference stops the freeze), and only then archived
deterministically (hpcperf-tar-1 + zstd -19 single-thread) into the maintainer's LOCAL ARTIFACT STAGING
(--staging DIR or $HPCPERF_ARTIFACT_STAGING, outside the git worktree):
    <staging>/level3/<app>/<source_version>/<app>[-<variant>]-<source_version>.tar.zst  (+ artifact.json, SHA256SUMS)
The artifact never enters git; it is published later to project-controlled external storage
(tools/artifacts/publish_artifacts.sh) and materialized by users with tools/prepare_benchmark.sh.

Outputs (under level3/<app>/): provenance/source.lock[.variant].yaml (schema hpcperf-source-lock-2: identity,
verification data, publish status `unpublished`), upstream.lock, patch_series.txt, original_vs_baseline.diff,
SOURCE_MANIFEST[.variant].json, LICENSES.md, equivalence[.variant].{json,md}; benchmark.yaml gets its
source identity fields (source_version, source_tree_sha256, source_artifact / variants.<v>).
"""
import argparse
import datetime
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hpcperf_source as hs  # noqa: E402
import hpcperf_lock as hl  # noqa: E402
import compare_source_trees as cst  # noqa: E402

TOOL_VERSION = "freeze-2.0"


def die(msg):
    print(f"freeze: FAIL -- {msg}", file=sys.stderr)
    sys.exit(1)


def log(msg):
    print(f"freeze: {msg}", flush=True)


def patch_touched_paths(patch_file):
    paths = set()
    with open(patch_file, errors="replace") as f:
        for line in f:
            m = re.match(r"^\+\+\+ (?:b/)?(\S+)", line) or re.match(r"^--- (?:a/)?(\S+)", line)
            if m and m.group(1) != "/dev/null":
                paths.add(m.group(1))
    return sorted(paths)


def apply_patch(tree, patch_file):
    r = subprocess.run(["patch", "-p1", "--forward", "--no-backup-if-mismatch", "-i", os.path.abspath(patch_file)],
                       cwd=tree, capture_output=True, text=True)
    if r.returncode != 0:
        die(f"patch {os.path.basename(patch_file)} did not apply cleanly in {tree}:\n{r.stdout[-1500:]}{r.stderr[-500:]}")
    if "fuzz" in r.stdout or "offset" in r.stdout:
        log(f"  note: {os.path.basename(patch_file)} applied with offset/fuzz: {r.stdout.strip().splitlines()[-1]}")


def download(url, dest):
    log(f"  downloading {url}")
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, dest)


def export_component(R, comp, stage):
    dest = os.path.join(stage, comp["dest"])
    kind = comp["kind"]
    if kind == "git":
        co = os.path.join(R, comp["checkout"])
        if not os.path.isdir(os.path.join(co, ".git")) and not os.path.isfile(os.path.join(co, ".git")):
            die(f"{comp['dest']}: checkout {comp['checkout']} missing (run the application's fetch.sh)")
        head = hs.git_head(co)
        if head != comp["commit"]:
            die(f"{comp['dest']}: {comp['checkout']} is at {head}, spec requires {comp['commit']}")
        excl = [x["path"] for x in comp.get("exclude", [])]
        nf, nl, gitlinks = hs.git_export_tree(co, dest, exclude=excl)
        log(f"  {comp['dest']}: {nf} files, {nl} symlinks from {comp['checkout']} @ {head[:12]} (excluded {len(excl)} paths; {len(gitlinks)} gitlinks)")
        # declared submodules (any nesting depth): exported from their own checkouts at the pinned commits;
        # every other gitlink (undeclared or excluded) stays out of the bundle and is reported
        declared = {s["path"]: s for s in comp.get("submodules", [])}
        for spath, sub_spec in declared.items():
            sub = os.path.join(co, spath)
            if not os.path.exists(os.path.join(sub, ".git")):
                die(f"{comp['dest']}/{spath}: declared submodule is not checked out under {comp['checkout']}")
            sh = hs.git_head(sub)
            if sh != sub_spec["commit"]:
                die(f"{comp['dest']}/{spath}: submodule at {sh}, spec requires {sub_spec['commit']}")
            sf, sl, nested = hs.git_export_tree(sub, os.path.join(dest, spath), exclude=[x["path"] for x in sub_spec.get("exclude", [])])
            undeclared = [f"{spath}/{n}" for n in nested if f"{spath}/{n}" not in declared]
            log(f"  {comp['dest']}/{spath}: submodule {sf} files, {sl} symlinks @ {sh[:12]}" + (f" (nested gitlinks not included: {', '.join(undeclared)})" if undeclared else ""))
        for g in gitlinks:
            if g not in declared and g not in excl:
                log(f"  {comp['dest']}: gitlink {g} not included (not declared in submodules)")
        return {"commit": head}
    if kind == "file":
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        src = os.path.join(R, comp["source"]) if comp.get("source") else None
        if src and os.path.isfile(src):
            shutil.copyfile(src, dest)
            origin = comp["source"]
        elif comp.get("url"):
            cache = os.path.join(R, ".deps", "level3", "_freeze_downloads")
            os.makedirs(cache, exist_ok=True)
            cached = os.path.join(cache, os.path.basename(comp["dest"]))
            if not (os.path.isfile(cached) and hs.sha256_file(cached) == comp["sha256"]):
                download(comp["url"], cached)
            shutil.copyfile(cached, dest)
            origin = comp["url"]
        else:
            die(f"{comp['dest']}: neither a local source nor a url")
        got = hs.sha256_file(dest)
        if got != comp["sha256"]:
            die(f"{comp['dest']}: sha256 {got} != declared {comp['sha256']} (from {origin})")
        os.chmod(dest, 0o644)
        log(f"  {comp['dest']}: {hs.human(os.path.getsize(dest))} sha256 ok (from {origin})")
        return {"sha256": got, "origin": origin}
    die(f"{comp['dest']}: unknown kind {kind}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("app_dir")
    ap.add_argument("--spec"); ap.add_argument("--variant")
    ap.add_argument("--stage-root", default=os.environ.get("HPCPERF_FREEZE_SCRATCH", f"/tmp/hpcperf-freeze-{os.environ.get('USER', 'user')}"))
    ap.add_argument("--keep-stage", action="store_true"); ap.add_argument("--skip-equivalence", action="store_true")
    ap.add_argument("--no-archive", action="store_true"); ap.add_argument("--verify-determinism", action="store_true")
    ap.add_argument("--allow-unexpected", action="store_true", help="record UNEXPECTED equivalence differences but continue (never silent: they stay in the report)")
    ap.add_argument("--staging", default=os.environ.get(hs.STAGING_ENV), help="local artifact staging root (outside the worktree); default $HPCPERF_ARTIFACT_STAGING")
    ap.add_argument("--redistribution-status", choices=hl.REDISTRIBUTION_STATUSES, help="default: the spec's redistribution_status, else 'review'")
    ap.add_argument("--suite-status", choices=hl.SUITE_STATUSES, help="default: the benchmark.yaml value, else 'candidate'")
    a = ap.parse_args()

    app_dir = os.path.abspath(a.app_dir)
    R = os.path.abspath(os.path.join(app_dir, "..", ".."))
    app = os.path.basename(app_dir)
    variant = a.variant
    spec_path = a.spec or os.path.join(app_dir, "provenance", f"freeze_spec{'.' + variant if variant else ''}.yaml")
    spec = hs.load_yaml(spec_path)
    if spec.get("schema") != "hpcperf-freeze-spec-1":
        die(f"{spec_path}: unsupported schema {spec.get('schema')!r}")
    variant = variant or spec.get("variant")
    suffix = f".{variant}" if variant else ""
    prov = os.path.join(app_dir, "provenance"); os.makedirs(prov, exist_ok=True)
    source_version = spec["benchmark_source_version"]
    arch_name = hs.artifact_filename(app, variant, source_version)
    if not a.no_archive and not a.staging:
        die("no artifact staging: pass --staging DIR or set HPCPERF_ARTIFACT_STAGING (never inside the worktree)")
    if a.staging and (os.path.abspath(a.staging) == R or os.path.abspath(a.staging).startswith(R + os.sep)):
        die("the artifact staging must be outside the git worktree")
    staging_dir = hl.staging_entry(a.staging, app, source_version) if a.staging else None
    redistribution = a.redistribution_status or spec.get("redistribution_status") or "review"
    stage_dir = os.path.join(a.stage_root, f"{app}{suffix}")
    if os.path.isdir(stage_dir):
        shutil.rmtree(stage_dir)
    stage = os.path.join(stage_dir, "stage"); os.makedirs(stage)
    log(f"{app}{suffix}: spec {os.path.relpath(spec_path, R)}, staging in {stage}")

    # 1. export / copy every component
    comp_info = {}
    for comp in spec["components"]:
        comp_info[comp["dest"]] = export_component(R, comp, stage)
    for top in ("src", "deps"):
        if os.path.exists(os.path.join(stage, top)) and not os.path.isdir(os.path.join(stage, top)):
            die(f"{top} is not a directory in the staging tree")
    if not os.path.isdir(os.path.join(stage, "src")):
        die("the bundle has no src/ (every component must land under src/ or deps/)")
    for entry in os.listdir(stage):
        if entry not in ("src", "deps"):
            die(f"unexpected top-level entry in the bundle: {entry}")

    # 2. patches (ordered per component), with a pristine copy of the touched files for the diff
    series = []
    diff_chunks = []
    for comp in spec["components"]:
        for i, p in enumerate(comp.get("patches", []), 1):
            pf = os.path.join(app_dir, p["path"])
            if not os.path.isfile(pf):
                die(f"patch {p['path']} missing")
            sub = p.get("apply_in", "")           # patches written relative to a subtree (e.g. tools/toolchain, src/cmake/blt)
            tree = os.path.join(stage, comp["dest"], sub) if sub else os.path.join(stage, comp["dest"])
            touched = [os.path.join(sub, t) if sub else t for t in patch_touched_paths(pf)]
            tree = os.path.join(stage, comp["dest"])
            pristine = tempfile.mkdtemp(prefix="pristine-", dir=stage_dir)
            for t in touched:
                src = os.path.join(tree, t)
                if os.path.isfile(src):
                    os.makedirs(os.path.dirname(os.path.join(pristine, t)), exist_ok=True)
                    shutil.copyfile(src, os.path.join(pristine, t))
            apply_patch(os.path.join(tree, sub) if sub else tree, pf)
            for t in touched:
                old = os.path.join(pristine, t); new = os.path.join(tree, t)
                r = subprocess.run(["diff", "-u", "--label", f"original/{comp['dest']}/{t}", "--label", f"baseline/{comp['dest']}/{t}",
                                    old if os.path.isfile(old) else "/dev/null", new if os.path.isfile(new) else "/dev/null"],
                                   capture_output=True, text=True)
                diff_chunks.append(r.stdout)
            shutil.rmtree(pristine)
            series.append({"order": len(series) + 1, "component": comp["dest"], "path": p["path"], "sha256": hs.sha256_file(pf),
                           "upstream_source": p.get("upstream_source", ""), "category": p.get("category", ""),
                           "touched": [f"{comp['dest']}/{t}" for t in touched]})
            log(f"  applied {p['path']} to {comp['dest']} ({len(touched)} files)")

    # 3. symlink policy + scan + manifest
    entries = hs.manifest(stage)
    esc = hs.escaping_symlinks(stage, entries)
    if esc:
        die("symlinks escaping the bundle: " + ", ".join(f"{e['path']} -> {e['target']}" for e in esc[:10]))
    hits = hs.scan_tree(stage, entries, allow=spec.get("scan_allow", []))
    if hits:
        for h in hits[:50]:
            print(f"freeze: SCAN HIT rule={h['rule']} path={h['path']}", file=sys.stderr)
        die(f"{len(hits)} credential/artifact scan hit(s) in the staging tree -- nothing archived")
    tree_sha = hs.tree_hash_from_manifest(entries)
    total = sum(e["size"] for e in entries if e["type"] == "F")
    log(f"staging tree: {len(entries)} entries, {hs.human(total)} uncompressed, source_tree_sha256={tree_sha}")

    # 4. equivalence with the validated trees
    eq_results = []
    unexpected = 0
    if not a.skip_equivalence:
        touched_all = [t for s in series for t in s["touched"]]
        for eq in spec.get("equivalence", []):
            frozen = os.path.join(stage, eq["archive_path"])
            validated = os.path.join(R, eq["validated_tree"])
            if not os.path.isdir(validated):
                eq_results.append({"archive_path": eq["archive_path"], "validated_tree": eq["validated_tree"], "status": "VALIDATED_TREE_MISSING"})
                log(f"  equivalence {eq['archive_path']}: validated tree {eq['validated_tree']} MISSING (recorded)")
                continue
            prefix = eq["archive_path"].rstrip("/") + "/"
            patched_rel = [t[len(prefix):] for t in touched_all if t.startswith(prefix)]
            res = cst.compare(frozen, validated, ignore=eq.get("ignore", []), generated=eq.get("generated", []),
                              excluded=eq.get("excluded", []), patched=patched_rel, added=eq.get("added", []))
            res["archive_path"] = eq["archive_path"]; res["validated_tree"] = eq["validated_tree"]
            res["status"] = "EQUIVALENT" if res["summary"]["unexpected_total"] == 0 else "UNEXPECTED_DIFFERENCES"
            unexpected += res["summary"]["unexpected_total"]
            eq_results.append(res)
            s = res["summary"]
            log(f"  equivalence {eq['archive_path']} vs {eq['validated_tree']}: {res['status']} (identical {s['identical']}, patch {s['expected_patch_difference']}, generated {s['expected_generated_difference']}, excluded {s['expected_normalization_excluded']}, artifacts {s['expected_build_artifact']}, UNEXPECTED {s['unexpected_total']})")
        hs.write_json(eq_results, os.path.join(prov, f"equivalence{suffix}.json"))
        with open(os.path.join(prov, f"equivalence{suffix}.md"), "w") as f:
            f.write(f"# Source equivalence: frozen bundle vs validated trees ({app}{suffix})\n\n")
            for res in eq_results:
                f.write(f"## {res['archive_path']} vs `{res['validated_tree']}`: **{res['status']}**\n\n")
                if "summary" in res:
                    f.write(cst.to_markdown(res) + "\n")
        if unexpected and not a.allow_unexpected:
            die(f"{unexpected} UNEXPECTED difference(s) between the frozen tree and the validated tree -- migration of {app}{suffix} stopped (see provenance/equivalence{suffix}.md)")

    # 5. deterministic artifact -> local staging (never into the worktree)
    archive_info = {}
    arch = None
    if not a.no_archive:
        os.makedirs(staging_dir, exist_ok=True)
        arch = os.path.join(staging_dir, arch_name)
        tar_tmp = os.path.join(stage_dir, "artifact.tar")
        n = hs.write_deterministic_tar(stage, tar_tmp)
        arch_tmp = arch + f".tmp.{os.getpid()}"
        hs.zstd_compress(tar_tmp, arch_tmp)
        arch_sha = hs.sha256_file(arch_tmp)
        if a.verify_determinism:
            tar2 = os.path.join(stage_dir, "artifact2.tar"); arch2 = os.path.join(stage_dir, "artifact2.tar.zst")
            hs.write_deterministic_tar(stage, tar2); hs.zstd_compress(tar2, arch2)
            same = hs.sha256_file(arch2) == arch_sha
            log(f"determinism check: second archive {'IDENTICAL' if same else 'DIFFERS'} (sha256 {hs.sha256_file(arch2)[:16]} vs {arch_sha[:16]})")
            if not same:
                os.remove(arch_tmp); die("archive is not reproducible")
            os.remove(tar2); os.remove(arch2)
        if os.path.isfile(arch) and hs.sha256_file(arch) != arch_sha:
            os.remove(arch_tmp)
            die(f"{os.path.relpath(arch, a.staging)} already exists in the staging with a DIFFERENT sha256 -- artifacts are immutable; bump benchmark_source_version")
        os.chmod(arch_tmp, 0o640); os.replace(arch_tmp, arch)
        archive_info = {"sha256": arch_sha, "compression": "zstd", "zstd_level": 19, "zstd_version": hs.zstd_version(),
                        "tar_format": hs.TAR_ALGO, "compressed_size": os.path.getsize(arch), "uncompressed_size": total,
                        "tar_entries": n, "file_count": sum(1 for e in entries if e["type"] == "F"), "symlink_count": sum(1 for e in entries if e["type"] == "L")}
        os.remove(tar_tmp)
        log(f"artifact {os.path.relpath(arch, a.staging)}: {hs.human(archive_info['compressed_size'])} compressed / {hs.human(total)} uncompressed, sha256={arch_sha}")

    # 6. provenance files
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    comps = spec["components"]
    app_comp = next(c for c in comps if c["dest"] == "src")
    layout = ["src/", "deps/"] if os.path.isdir(os.path.join(stage, "deps")) else ["src/"]
    eq_records = [{"archive_path": r["archive_path"], "validated_tree": hl.portable_location(r["validated_tree"]), "status": r["status"], **({"summary": r["summary"]} if "summary" in r else {})} for r in eq_results] if not a.skip_equivalence else "SKIPPED"
    lock = None
    if archive_info:
        lock = hl.make_lock(
            name=spec["name"], application=spec["application"], variant=variant, source_version=source_version,
            upstream={"url": app_comp.get("url"), "tag": app_comp.get("ref"), "commit": app_comp["commit"]},
            archive_info=archive_info, tree_sha=tree_sha, entries=len(entries), layout=layout,
            patches=[{"path": s_["path"], "sha256": s_["sha256"], "category": s_["category"], "upstream_reference": s_["upstream_source"], "component": s_["component"], "files": s_["touched"]} for s_ in series],
            dependencies={
                "bundled": [{**b, "component": c["dest"]} for c in comps for b in c.get("bundled", [])],
                "benchmark_specific": [{"path": c["dest"], "project": c.get("project"), "version": c.get("version"), "url": c.get("url"), "commit": c.get("commit"),
                                        "sha256": comp_info[c["dest"]].get("sha256"), "license": c.get("license")} for c in comps if c.get("category") == "benchmark_specific"],
                "environment_provided": spec.get("environment_provided", []),
            },
            components=[{"dest": c["dest"], "kind": c["kind"], "category": c["category"], "url": c.get("url"), "ref": c.get("ref"), "commit": c.get("commit"),
                         "sha256": c.get("sha256"), "checkout": c.get("checkout"), "submodules": c.get("submodules", []),
                         "excluded": c.get("exclude", []), "license": c.get("license")} for c in comps],
            equivalence=eq_records, licenses=spec.get("licenses", []), license_notes=spec.get("license_notes", []),
            redistribution_status=redistribution, source_scope=hl.source_scope_from_optimization_scope(app_dir),
            scan_allow=spec.get("scan_allow", []), freeze_tool_version=TOOL_VERSION, freeze_timestamp=now,
            primary=spec.get("primary"), mirrors=spec.get("mirrors"))
        tmp_lock = os.path.join(prov, f"source.lock{suffix}.yaml.freezing")
        hs.dump_yaml(lock, tmp_lock)
        problems = hl.validate_lock(lock, open(tmp_lock).read())
        if problems:
            os.remove(tmp_lock); die("lock invalid: " + "; ".join(problems))
        os.replace(tmp_lock, os.path.join(prov, f"source.lock{suffix}.yaml"))
        hl.update_staging_metadata(staging_dir, {"filename": arch_name, "benchmark": app, "variant": variant, "source_version": source_version,
                                           "size": archive_info["compressed_size"], "sha256": archive_info["sha256"], "source_tree_sha256": tree_sha,
                                           "format": hs.ARTIFACT_FORMAT, "status": "LOCAL_ARTIFACT_VERIFIED", "remote_status": "REMOTE_ARTIFACT_UNPUBLISHED",
                                           "verified": now, "origin": {"kind": "freeze", "tool": TOOL_VERSION}, "redistribution_status": redistribution})
    with open(os.path.join(prov, f"upstream{suffix}.lock"), "w") as f:
        for c in comps:
            if c["kind"] == "git":
                f.write(f"{c['dest']}\t{c.get('url')}\t{c.get('ref')}\t{c['commit']}\n")
                for s in c.get("submodules", []):
                    f.write(f"{c['dest']}/{s['path']}\t{s.get('url', '')}\t{s.get('ref', '')}\t{s['commit']}\n")
            else:
                f.write(f"{c['dest']}\t{c.get('url')}\t{c.get('version', '')}\tsha256:{c['sha256']}\n")
    with open(os.path.join(prov, f"patch_series{suffix}.txt"), "w") as f:
        f.write("# order\tcomponent\tpatch\tsha256\tcategory\tupstream_source\tfiles\n")
        for s in series:
            f.write(f"{s['order']}\t{s['component']}\t{s['path']}\t{s['sha256']}\t{s['category']}\t{s['upstream_source']}\t{','.join(s['touched'])}\n")
        if not series:
            f.write("# (no patches: the bundle is the exact upstream tree)\n")
    with open(os.path.join(prov, f"original_vs_baseline{suffix}.diff"), "w") as f:
        f.write(f"# {app}{suffix}: exact upstream ({app_comp['commit']}) -> frozen baseline; generated by {TOOL_VERSION} from the applied patch series\n")
        f.write("".join(diff_chunks) if diff_chunks else "# (empty: no patches)\n")
    man = {"schema": "hpcperf-source-manifest-1", "name": spec["name"], "variant": variant, "algorithm": hs.TREE_ALGO,
           "algorithm_description": hs.__doc__.split("Deterministic archive")[0].strip(),
           "source_tree_sha256": tree_sha, "entries": len(entries), "files": sum(1 for e in entries if e["type"] == "F"),
           "symlinks": sum(1 for e in entries if e["type"] == "L"), "uncompressed_size": total,
           "archive": archive_info, "manifest": [{k: e[k] for k in ("path", "type", "sha256", "size", "exec") if k in e} | ({"target": e["target"]} if e["type"] == "L" else {}) for e in entries]}
    hs.write_json(man, os.path.join(prov, f"SOURCE_MANIFEST{suffix}.json"))
    # LICENSES.md: declared licenses + every LICENSE/COPYING/NOTICE file found in the bundle
    lic_files = [e["path"] for e in entries if re.search(r"(^|/)(LICENSE|LICENCE|COPYING|COPYRIGHT|NOTICE)[^/]*$", e["path"], re.I)]
    with open(os.path.join(prov, "LICENSES.md" if not variant else f"LICENSES{suffix}.md"), "w") as f:
        f.write(f"# Licenses of the {spec['application']} source bundle{' (' + variant + ')' if variant else ''}\n\n")
        f.write("Declared components (path in the bundle, project, license, upstream, commit/version):\n\n| path | project | license | upstream | commit / version |\n|---|---|---|---|---|\n")
        for l in spec.get("licenses", []):
            f.write(f"| `{l['path']}` | {l['project']} | {l['license']} | {l.get('url', '')} | {l.get('commit', l.get('version', ''))} |\n")
        f.write(f"\nRedistribution notes:\n\n")
        for n in spec.get("license_notes", []):
            f.write(f"- {n}\n")
        f.write(f"\nLicense/notice files present in the bundle ({len(lic_files)}):\n\n")
        for p in lic_files:
            f.write(f"- `{p}`\n")
    # benchmark.yaml identity fields (from the lock; nothing archive-path related is recorded in git)
    by_path = os.path.join(app_dir, "benchmark.yaml")
    by = hs.load_yaml(by_path) if os.path.isfile(by_path) else {"name": spec["name"], "level": 3, "application": spec["application"]}
    if lock:
        hl.apply_identity(by, lock)
        if a.suite_status:
            by["suite_status"] = a.suite_status
        elif "suite_status" not in by:
            by["suite_status"] = "candidate"
        hs.dump_yaml(by, by_path)
    else:
        log("--no-archive: tree hashed and reported only; lock / benchmark.yaml identity NOT written (an artifact is the identity carrier)")
    log(f"provenance written under {os.path.relpath(prov, R)}; benchmark.yaml identity updated")
    if not a.keep_stage:
        shutil.rmtree(stage_dir)
    else:
        log(f"staging tree kept: {stage}")
    print(f"FREEZE OK {app}{suffix} source_tree_sha256={tree_sha}" + (f" archive_sha256={archive_info['sha256']} compressed={archive_info['compressed_size']} artifact={arch_name} status=LOCAL_ARTIFACT_VERIFIED,REMOTE_ARTIFACT_UNPUBLISHED" if archive_info else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
