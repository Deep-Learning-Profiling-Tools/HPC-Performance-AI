#!/usr/bin/env python3
"""hpcperf_materialize -- materialize a Level 3 benchmark's frozen source bundle into level3/<app>/{src,deps}.

    hpcperf_materialize.py level3/<app> [--variant NAME] [--force-rematerialize]

1. locate archives/<archive> from benchmark.yaml (variant-aware);
2. refuse an un-downloaded Git LFS pointer with an explicit message;
3. verify the archive SHA-256;
4. extract into a temporary staging directory next to the target (same filesystem);
5. verify source_tree_sha256 (hpcperf-tree-1) of the extracted tree;
6. check the src/ (+ deps/) layout against the lock file;
7. static safety: no symlink escaping the tree, no credential-looking file, no build/install/log output,
   complete provenance;
8. atomic rename into place;
9. if src/ or deps/ already exist: recompute their tree hash -- identical: success (idempotent);
   different: FAIL; nothing is ever overwritten unless --force-rematerialize is given explicitly.
"""
import argparse
import datetime
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hpcperf_source as hs  # noqa: E402

MARKER = ".hpcperf-materialized.yaml"


def die(msg, code=1):
    print(f"prepare_benchmark: FAIL -- {msg}", file=sys.stderr)
    sys.exit(code)


def log(msg):
    print(f"prepare_benchmark: {msg}", flush=True)


def select_variant(by, requested):
    variants = by.get("variants")
    if not variants:
        if requested:
            die(f"benchmark has no variants but --variant {requested} was given")
        return None, by.get("source_bundle"), by.get("source_tree_sha256")
    env = by.get("variant_env")
    name = requested or (os.environ.get(env) if env else None) or by.get("default_variant")
    if name not in variants:
        die(f"variant {name!r} unknown; available: {', '.join(variants)}")
    v = variants[name]
    return name, v, v.get("source_tree_sha256")


def existing_hash(app_dir):
    present = [d for d in ("src", "deps") if os.path.lexists(os.path.join(app_dir, d))]
    if not present:
        return None, []
    for d in present:
        if os.path.islink(os.path.join(app_dir, d)):
            die(f"{d} is a symlink -- a materialized source tree must be a real directory")
    return hs.tree_hash(app_dir, present), present


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("app_dir"); ap.add_argument("--variant"); ap.add_argument("--force-rematerialize", action="store_true")
    a = ap.parse_args()
    app_dir = os.path.abspath(a.app_dir)
    by_path = os.path.join(app_dir, "benchmark.yaml")
    if not os.path.isfile(by_path):
        die(f"{by_path} missing")
    by = hs.load_yaml(by_path)
    variant, bundle, expected_tree = select_variant(by, a.variant)
    if not bundle or not bundle.get("archive") or not expected_tree:
        die("benchmark.yaml carries no source_bundle identity (archive / source_tree_sha256) -- the benchmark was not frozen")
    suffix = f".{variant}" if variant else ""
    lock_path = os.path.join(app_dir, "provenance", f"source.lock{suffix}.yaml")
    if not os.path.isfile(lock_path):
        die(f"provenance incomplete: {os.path.relpath(lock_path, app_dir)} missing")
    lock = hs.load_yaml(lock_path)
    if lock.get("materialized_tree", {}).get("sha256") != expected_tree:
        die(f"benchmark.yaml and {os.path.basename(lock_path)} disagree on source_tree_sha256")
    archive = os.path.join(app_dir, bundle["archive"])
    marker = os.path.join(app_dir, MARKER)

    # existing materialization?
    have, present = existing_hash(app_dir)
    if have is not None:
        if have == expected_tree:
            log(f"{'/'.join(present)} already materialized and identical to the frozen baseline ({expected_tree[:16]}...) -- nothing to do")
            hs.dump_yaml({"variant": variant, "source_tree_sha256": expected_tree, "archive_sha256": bundle.get("archive_sha256"),
                          "verified": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "tool": "hpcperf_materialize-1.0"}, marker)
            return 0
        if not a.force_rematerialize:
            die(f"{'/'.join(present)} exist but differ from the frozen baseline (tree {have[:16]}... vs expected {expected_tree[:16]}...). "
                f"Local modifications are never overwritten automatically; pass --force-rematerialize to DISCARD them and re-materialize.", 3)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for d in present:
            old = os.path.join(app_dir, d); trash = os.path.join(app_dir, f".discarded-{d}-{stamp}")
            os.rename(old, trash)
            shutil.rmtree(trash, ignore_errors=True)
            log(f"--force-rematerialize: discarded existing {d} (was {have[:16]}...)")
        if os.path.exists(marker):
            os.remove(marker)

    # archive checks
    if not os.path.isfile(archive):
        die(f"source archive {bundle['archive']} is missing")
    if hs.is_lfs_pointer(archive):
        die(f"source archive is not materialized: {bundle['archive']} is a Git LFS pointer; run `git lfs pull --include \"{os.path.relpath(archive, os.path.dirname(os.path.dirname(app_dir)))}\"` and retry", 4)
    got = hs.sha256_file(archive)
    if got != bundle.get("archive_sha256"):
        die(f"archive sha256 {got} != recorded {bundle.get('archive_sha256')}")
    log(f"archive {bundle['archive']} sha256 ok ({hs.human(os.path.getsize(archive))})")

    # extraction into a sibling temporary directory (same filesystem -> atomic rename)
    tmp = os.path.join(app_dir, f".materialize.tmp.{os.getpid()}")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    try:
        tar_tmp = os.path.join(tmp, "bundle.tar")
        hs.zstd_decompress_to_tar(archive, tar_tmp)
        hs.safe_extract_tar(tar_tmp, tmp)
        os.remove(tar_tmp)
        entries = hs.manifest(tmp)
        tree = hs.tree_hash_from_manifest(entries)
        if tree != expected_tree:
            die(f"extracted tree sha256 {tree} != expected {expected_tree} -- archive content does not match its recorded identity")
        log(f"source_tree_sha256 verified ({len(entries)} entries)")
        layout = lock.get("materialized_tree", {}).get("layout", ["src/"])
        for top in layout:
            if not os.path.isdir(os.path.join(tmp, top.rstrip("/"))):
                die(f"bundle lacks {top} required by the lock file")
        for entry in os.listdir(tmp):
            if entry not in ("src", "deps"):
                die(f"unexpected top-level entry in the bundle: {entry}")
        esc = hs.escaping_symlinks(tmp, entries)
        if esc:
            die("symlinks escaping the source tree: " + ", ".join(f"{e['path']} -> {e['target']}" for e in esc[:10]))
        hits = hs.scan_tree(tmp, entries, allow=lock.get("scan_allow", []))
        if hits:
            for h in hits[:30]:
                print(f"prepare_benchmark: SCAN HIT rule={h['rule']} path={h['path']}", file=sys.stderr)
            die(f"{len(hits)} credential/artifact hit(s) in the extracted tree -- not materialized")
        for f in ("upstream.lock", "patch_series.txt", "SOURCE_MANIFEST.json", "LICENSES.md"):
            name = f.replace(".", f"{suffix}.", 1) if suffix and f != "LICENSES.md" else (f"LICENSES{suffix}.md" if suffix else f)
            if not os.path.isfile(os.path.join(app_dir, "provenance", name)):
                die(f"provenance incomplete: provenance/{name} missing")
        # atomic rename
        for top in ("src", "deps"):
            s = os.path.join(tmp, top)
            if os.path.isdir(s):
                os.rename(s, os.path.join(app_dir, top))
        hs.dump_yaml({"variant": variant, "source_tree_sha256": expected_tree, "archive": bundle["archive"], "archive_sha256": got,
                      "materialized": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "tool": "hpcperf_materialize-1.0",
                      "scan_allow": lock.get("scan_allow", [])}, marker)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"materialized {os.path.relpath(app_dir)}/{{src{',deps' if os.path.isdir(os.path.join(app_dir, 'deps')) else ''}}} (variant {variant or '-'}, tree {expected_tree[:16]}...)")
    print(f"PREPARE OK {os.path.basename(app_dir)}{suffix} source_tree_sha256={expected_tree}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
