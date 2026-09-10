#!/usr/bin/env python3
"""verify_artifact -- verify a Level 3 source artifact against its source lock.

    verify_artifact.py --lock level3/<app>/provenance/source.lock[.variant].yaml
                       (<artifact-file> | --cache-dir DIR | --staging-entry DIR) [--full] [--scratch DIR] [--json OUT]

Checks (each printed ok/FAIL; the verdict is PASS only when all pass):
  1 lock: schema hpcperf-source-lock-2, complete, no node-private location, valid publish/redistribution status
  2 file: exists, size == lock, zstd magic, sha256 == lock
  3 staging metadata (with --staging-entry): artifact.json entry and SHA256SUMS line agree with the lock
  4 (--full) extraction in a scratch directory with the restricted extractor: top-level layout == lock,
    source_tree_sha256 == lock, no symlink escaping the tree, no credential/env-dump/session/profiler/build-output
    hit (paths + rule names only), entry count == lock
Never modifies the artifact, the lock or any benchmark directory. Exit 0 PASS, 1 FAIL, 2 usage.
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import hpcperf_source as hs  # noqa: E402
import hpcperf_lock as hl  # noqa: E402


def verify(lock_path, artifact=None, cache=None, staging_entry=None, full=False, scratch=None, log=print):
    res = {"lock": os.path.relpath(lock_path), "checks": []}

    def rec(no, name, ok, detail=""):
        res["checks"].append({"check": no, "name": name, "status": "PASS" if ok else "FAIL", "detail": detail})
        if log:
            log(f"{'ok  ' if ok else 'FAIL'} {no}: {name}{' -- ' + detail if detail else ''}")
        return ok

    lock = hs.load_yaml(lock_path) or {}
    problems = hl.validate_lock(lock, open(lock_path).read())
    rec(1, "source lock valid", not problems, "; ".join(problems[:4]))
    if problems:
        res["verdict"] = "FAIL"; return res
    a = lock["artifact"]
    res.update({"filename": a["filename"], "sha256": a["sha256"], "size": a["size"], "source_tree_sha256": a["source_tree_sha256"]})
    if staging_entry:
        artifact = os.path.join(staging_entry, a["filename"])
    elif cache:
        artifact = hs.cache_path(cache, a["sha256"])
    res["artifact"] = artifact
    try:
        hs.verify_archive_file(artifact, a["sha256"], a["size"])
        rec(2, "artifact file: size, zstd magic, sha256", True, f"{os.path.basename(artifact)} {a['size']} B sha256 {a['sha256'][:16]}...")
    except hs.SourceError as ex:
        rec(2, "artifact file: size, zstd magic, sha256", False, str(ex))
        res["verdict"] = "FAIL"; return res
    if staging_entry:
        meta_p = os.path.join(staging_entry, "artifact.json")
        sums_p = os.path.join(staging_entry, "SHA256SUMS")
        ok = os.path.isfile(meta_p) and os.path.isfile(sums_p)
        detail = ""
        if ok:
            meta = json.load(open(meta_p))
            ent = next((e for e in meta.get("artifacts", []) if e.get("filename") == a["filename"]), None)
            ok = bool(ent) and ent.get("sha256") == a["sha256"] and int(ent.get("size", -1)) == a["size"] and ent.get("source_tree_sha256") == a["source_tree_sha256"]
            line = hs.sha256sums_line(a["sha256"], a["filename"]).strip()
            ok = ok and line in [l.strip() for l in open(sums_p)]
            detail = "artifact.json entry + SHA256SUMS line agree" if ok else "artifact.json / SHA256SUMS disagree with the lock"
        else:
            detail = "artifact.json or SHA256SUMS missing"
        rec(3, "staging metadata consistent", ok, detail)
    if full:
        scratch = scratch or os.environ.get("HPCPERF_FREEZE_SCRATCH") or f"/tmp/hpcperf-verify-{os.environ.get('USER', 'user')}"
        os.makedirs(scratch, exist_ok=True)
        tmp = os.path.join(scratch, f"verify-{a['sha256'][:12]}-{os.getpid()}")
        try:
            os.makedirs(tmp)
            tar_tmp = os.path.join(tmp, "artifact.tar")
            hs.zstd_decompress_to_tar(artifact, tar_tmp)
            try:
                hs.safe_extract_tar(tar_tmp, tmp)
                rec(4, "restricted extraction (no absolute path, '..', hard link, device)", True)
            except hs.SourceError as ex:
                rec(4, "restricted extraction (no absolute path, '..', hard link, device)", False, str(ex))
                res["verdict"] = "FAIL"; return res
            os.remove(tar_tmp)
            if os.path.isfile(os.path.join(tmp, "ARTIFACT_MANIFEST.json")):
                os.remove(os.path.join(tmp, "ARTIFACT_MANIFEST.json"))
            tops = sorted(os.listdir(tmp))
            rec(5, "top-level layout == lock", tops == sorted(t.rstrip("/") for t in lock["materialized_tree"]["layout"]), f"{tops}")
            entries = hs.manifest(tmp)
            tree = hs.tree_hash_from_manifest(entries)
            rec(6, "source_tree_sha256 == lock", tree == a["source_tree_sha256"], f"{tree[:16]}... vs {a['source_tree_sha256'][:16]}...")
            rec(7, "entry count == lock", len(entries) == lock["materialized_tree"]["entries"], f"{len(entries)} vs {lock['materialized_tree']['entries']}")
            esc = hs.escaping_symlinks(tmp, entries)
            rec(8, "no symlink escaping the tree", not esc, "; ".join(f"{e['path']} -> {e['target']}" for e in esc[:4]))
            hits = hs.scan_tree(tmp, entries, allow=lock.get("scan_allow", []))
            rec(9, "secret / build-output scan (paths + rules only)", not hits, "; ".join(f"{h['rule']}:{h['path']}" for h in hits[:4]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    res["verdict"] = "PASS" if all(c["status"] == "PASS" for c in res["checks"]) else "FAIL"
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("artifact", nargs="?"); ap.add_argument("--lock", required=True)
    ap.add_argument("--cache-dir"); ap.add_argument("--staging-entry"); ap.add_argument("--full", action="store_true")
    ap.add_argument("--scratch"); ap.add_argument("--json")
    a = ap.parse_args()
    if not (a.artifact or a.cache_dir or a.staging_entry):
        ap.error("give an artifact file, --cache-dir or --staging-entry")
    res = verify(a.lock, a.artifact, a.cache_dir, a.staging_entry, a.full, a.scratch)
    print(f"verify_artifact: {res['verdict']} {res.get('filename', '')}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
