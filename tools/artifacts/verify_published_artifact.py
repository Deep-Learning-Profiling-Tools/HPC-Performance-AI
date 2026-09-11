#!/usr/bin/env python3
"""verify_published_artifact -- anonymous download check of ONE published source artifact against its lock.

    verify_published_artifact.py --lock level3/<app>/provenance/source.lock[.variant].yaml --url URL
                                 [--record OUT.yaml] [--cache DIR] [--scratch DIR]

This is the step that breaks the circular dependency between "publish" and "lock update": the URL comes from the
reviewed release plan (the asset is already public), while the expected artifact identity -- size, archive sha256
and source_tree_sha256 -- comes ONLY from the trusted lock metadata in the reviewed checkout. No URL can change
what the artifact is expected to contain. Afterwards the lock may be updated with the real URL (commit M) and the
ordinary user entry (`git clone` of M + `tools/prepare_benchmark.sh`) is verified separately.

Refuses to run when GITHUB_TOKEN or GH_TOKEN is set (an authenticated draft download is not an anonymous fetch),
downloads into a fresh empty cache, verifies size + zstd magic + sha256, extracts with the restricted extractor,
verifies source_tree_sha256 and the layout, and runs the credential/build-output scan. Exit 0 only if all pass.
"""
import argparse
import datetime
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import hpcperf_source as hs  # noqa: E402
import hpcperf_lock as hl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lock", required=True); ap.add_argument("--url", required=True)
    ap.add_argument("--record"); ap.add_argument("--cache"); ap.add_argument("--scratch")
    a = ap.parse_args()
    for v in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(v):
            print(f"verify_published_artifact: FAIL -- {v} is set; an authenticated download is not an anonymous fetch", file=sys.stderr)
            return 2
    lock = hs.load_yaml(a.lock) or {}
    problems = hl.validate_lock(lock, open(a.lock).read())
    if problems:
        print(f"verify_published_artifact: FAIL -- lock invalid: {'; '.join(problems[:3])}", file=sys.stderr)
        return 1
    art = lock["artifact"]
    if not a.url.startswith("https://") and os.environ.get("HPCPERF_ALLOW_INSECURE_FETCH") != "1":
        print("verify_published_artifact: FAIL -- the artifact URL must be https (set HPCPERF_ALLOW_INSECURE_FETCH=1 only in tests)", file=sys.stderr)
        return 1
    if not a.url.endswith(art["filename"]):
        print(f"verify_published_artifact: FAIL -- URL does not end with the artifact filename {art['filename']}", file=sys.stderr)
        return 1
    cache = a.cache or tempfile.mkdtemp(prefix="hpcperf-anon-cache-")
    scratch = a.scratch or tempfile.mkdtemp(prefix="hpcperf-anon-extract-")
    owned = (a.cache is None, a.scratch is None)
    log = lambda m: print(f"verify_published_artifact: {m}", flush=True)  # noqa: E731
    try:
        log(f"anonymous download of {art['filename']} ({art['size']} B) from {a.url}")
        path = hs.download_to_cache(a.url, cache, art["sha256"], art["size"], log=log)
        log(f"size + zstd magic + sha256 verified against the lock ({art['sha256'][:16]}...)")
        tar = os.path.join(scratch, "artifact.tar")
        hs.zstd_decompress_to_tar(path, tar)
        hs.safe_extract_tar(tar, scratch)
        os.remove(tar)
        man = os.path.join(scratch, "ARTIFACT_MANIFEST.json")
        if os.path.isfile(man):
            os.remove(man)
        entries = hs.manifest(scratch)
        tree = hs.tree_hash_from_manifest(entries)
        if tree != art["source_tree_sha256"]:
            raise hs.SourceError(f"source_tree_sha256 {tree} != lock {art['source_tree_sha256']}")
        log(f"source_tree_sha256 verified against the lock ({tree[:16]}..., {len(entries)} entries)")
        tops = sorted(os.listdir(scratch))
        want = sorted(t.rstrip("/") for t in lock["materialized_tree"]["layout"])
        if tops != want:
            raise hs.SourceError(f"top-level layout {tops} != lock {want}")
        esc = hs.escaping_symlinks(scratch, entries)
        if esc:
            raise hs.SourceError(f"escaping symlinks: {[e['path'] for e in esc[:3]]}")
        hits = hs.scan_tree(scratch, entries, allow=lock.get("scan_allow", []))
        if hits:
            raise hs.SourceError(f"{len(hits)} scan hit(s), first rule {hits[0]['rule']} path {hits[0]['path']}")
        log("layout, symlink and secret/build-output scans clean")
    except (hs.SourceError, OSError) as e:
        print(f"verify_published_artifact: FAIL -- {e}", file=sys.stderr)
        return 1
    finally:
        if owned[1]:
            shutil.rmtree(scratch, ignore_errors=True)
        if owned[0]:
            shutil.rmtree(cache, ignore_errors=True)
    if a.record:
        hs.dump_yaml({"schema": "hpcperf-remote-artifact-verification-1", "benchmark": lock["benchmark"]["name"],
                      "variant": lock["benchmark"].get("variant"), "source_version": lock["benchmark"]["source_version"],
                      "artifact": art["filename"], "url": a.url, "url_source": "reviewed release plan",
                      "expected_identity_source": f"trusted lock {os.path.basename(a.lock)} (size, archive sha256, source_tree_sha256)",
                      "size": art["size"], "archive_sha256": art["sha256"], "source_tree_sha256": art["source_tree_sha256"],
                      "anonymous": True, "cache": "fresh empty cache", "verdict": "PASS",
                      "note": "plan-URL anonymous verification; the ordinary user entry (clone + prepare_benchmark.sh with the lock's own URL) is recorded separately",
                      "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}, a.record)
        log(f"record written: {a.record}")
    print(f"REMOTE ARTIFACT VERIFIED {art['filename']} sha256={art['sha256']} tree={art['source_tree_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
