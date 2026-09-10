#!/usr/bin/env python3
"""hpcperf_materialize -- materialize a Level 3 benchmark's frozen source artifact into level3/<app>/{src,deps}
(scheme 3: project-controlled external source artifacts + content-addressed local cache).

    hpcperf_materialize.py level3/<app> [--variant NAME] [--artifact FILE] [--cache-dir DIR] [--offline]
                           [--force-rematerialize] [--status]

Flow (every step prints its status; nothing is written into the benchmark directory before every check passed):
  A  read provenance/source.lock[.variant].yaml: artifact filename, size, sha256, source_tree_sha256, primary url,
     mirrors, layout, dependency requirements;
  B  existing src/ (and deps/): recompute the tree hash -- identical to the lock: READY, nothing to do; different:
     DIRTY -> exit 3 (a user/agent/local modification is never overwritten; --force-rematerialize discards it);
  C  resolve the artifact in this order: --artifact FILE (verified, then copied into the cache) > local cache
     <cache>/sha256/<sha256>.tar.zst > primary https url > mirrors; --offline forbids any remote fetch
     (cache miss -> exit 4); an unpublished primary is reported as such (exit 4), never fetched from a made-up URL;
  D  a download streams into <cache>/.partial/, is size/magic/sha256-checked and only then renamed into the cache;
  E  the archive is extracted with a restricted extractor into a staging directory OUTSIDE the benchmark directory
     (level3/.materialize-staging/, same filesystem): only src/ deps/ (+ ARTIFACT_MANIFEST.json), no absolute
     paths, no '..', no hard links, no devices; symlinks escaping the tree, credential-looking files/contents,
     build output and session/profiler records fail the materialization;
  F  source_tree_sha256 (hpcperf-tree-1) of the extracted tree must equal the lock;
  G  atomic rename of src/ and deps/ into level3/<app>/, then the marker .hpcperf-materialized.yaml.
Status vocabulary: NOT_PREPARED, LOCAL_ARTIFACT_VERIFIED, REMOTE_FETCH_VERIFIED, MATERIALIZED, READY, DIRTY, INVALID.
Exit codes: 0 ok; 1 failure/INVALID; 2 usage; 3 DIRTY (refused); 4 artifact unavailable (offline cache miss,
unpublished, missing); 5 hash/size mismatch of an artifact or of the extracted tree.
"""
import argparse
import datetime
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hpcperf_source as hs  # noqa: E402
import hpcperf_lock as hl  # noqa: E402

TOOL = "hpcperf_materialize-2.0"


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def die(msg, code=1):
    print(f"prepare_benchmark: FAIL -- {msg}", file=sys.stderr)
    sys.exit(code)


def log(msg):
    print(f"prepare_benchmark: {msg}", flush=True)


def status(word, detail=""):
    print(f"prepare_benchmark: STATUS {word}{' -- ' + detail if detail else ''}", flush=True)


def existing_state(app_dir, expected_tree):
    """(state, tree, present): state in NOT_PREPARED | READY | DIRTY | INVALID"""
    present = [d for d in ("src", "deps") if os.path.lexists(os.path.join(app_dir, d))]
    if not present:
        return "NOT_PREPARED", None, []
    for d in present:
        if os.path.islink(os.path.join(app_dir, d)):
            return "INVALID", None, present
    try:
        tree = hs.tree_hash(app_dir, present)
    except hs.SourceError as ex:
        log(f"cannot hash {'/'.join(present)}: {ex}")
        return "INVALID", None, present
    return ("READY" if tree == expected_tree else "DIRTY"), tree, present


def resolve_artifact(lock, explicit, cdir, offline):
    """Returns (path, origin, status_word). origin: explicit|cache|primary|mirror."""
    a = lock["artifact"]
    sha, size, fn = a["sha256"], a["size"], a["filename"]
    if explicit:
        explicit = os.path.abspath(explicit)
        try:
            hs.verify_archive_file(explicit, sha, size)
        except hs.SourceError as ex:
            die(f"--artifact rejected: {ex}", 5)
        log(f"explicit artifact {explicit}: size {size} and sha256 ok")
        p = hs.place_in_cache(explicit, cdir, sha, size)
        log(f"cached as {os.path.relpath(p, cdir)}")
        return p, "explicit", "LOCAL_ARTIFACT_VERIFIED"
    cached = hs.cache_path(cdir, sha)
    if os.path.isfile(cached):
        try:
            hs.verify_archive_file(cached, sha, size)
            log(f"cache hit {os.path.relpath(cached, cdir)}: size {size} and sha256 ok")
            return cached, "cache", "LOCAL_ARTIFACT_VERIFIED"
        except hs.SourceError as ex:
            die(f"cache entry corrupt ({ex}); remove {cached} and retry", 5)
    if offline:
        die(f"--offline: {fn} (sha256 {sha[:16]}...) is not in the cache {cdir}; pass --artifact <file> or fetch it on a connected host", 4)
    prim = a.get("primary") or {}
    urls = []
    if prim.get("status") == "published" and prim.get("url"):
        urls.append((prim["url"], "primary"))
    urls += [(m, "mirror") for m in (a.get("mirrors") or [])]
    if not urls:
        die(f"{fn} is not in the local cache ({cdir}) and the artifact is REMOTE_ARTIFACT_UNPUBLISHED "
            f"(source.lock artifact.primary.status={prim.get('status')}); pass --artifact <file> (local staging copy) "
            f"or publish the artifact first", 4)
    last = None
    for url, origin in urls:
        try:
            p = hs.download_to_cache(url, cdir, sha, size, log=log)
            log(f"{origin} download verified (size {size}, sha256 {sha[:16]}...) -> {os.path.relpath(p, cdir)}")
            return p, origin, "REMOTE_FETCH_VERIFIED"
        except Exception as ex:  # noqa: BLE001
            last = ex
            log(f"{origin} {url}: {ex}")
    die(f"no artifact source succeeded (last error: {last})", 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("app_dir"); ap.add_argument("--variant"); ap.add_argument("--artifact")
    ap.add_argument("--cache-dir"); ap.add_argument("--offline", action="store_true")
    ap.add_argument("--force-rematerialize", action="store_true"); ap.add_argument("--status", action="store_true", help="report the state, change nothing")
    a = ap.parse_args()
    app_dir = os.path.abspath(a.app_dir)
    app = os.path.basename(app_dir)
    by_path = os.path.join(app_dir, "benchmark.yaml")
    if not os.path.isfile(by_path):
        die(f"{by_path} missing", 2)
    by = hs.load_yaml(by_path) or {}
    marker = hl.read_marker(app_dir)
    try:
        variant = hl.select_variant(by, a.variant, marker)
        lock = hl.load_lock(app_dir, variant)
    except hs.SourceError as ex:
        die(str(ex), 2)
    problems = hl.validate_lock(lock, open(hl.lock_path(app_dir, variant)).read())
    if problems:
        die("source.lock invalid: " + "; ".join(problems[:5]), 1)
    ident = hl.identity_from_benchmark(by, variant)
    art = lock["artifact"]
    expected_tree = art["source_tree_sha256"]
    if ident.get("source_tree_sha256") != expected_tree or ident.get("archive_sha256") != art["sha256"]:
        die("benchmark.yaml and the source lock disagree on the source identity", 1)
    if marker.get("variant") and variant != marker.get("variant") and not a.force_rematerialize:
        die(f"variant {marker['variant']} is materialized; requested {variant} -- pass --force-rematerialize to replace it", 3)
    cdir = hs.cache_dir(a.cache_dir)
    label = f"{app}{hl.suffix(variant)}"

    # B. existing tree
    state, have, present = existing_state(app_dir, expected_tree)
    if a.status:
        status(state, f"{label} tree {have[:16] + '...' if have else '-'} expected {expected_tree[:16]}...; cache {'hit' if os.path.isfile(hs.cache_path(cdir, art['sha256'])) else 'miss'} ({cdir}); remote {art['primary'].get('status')}")
        print(f"PREPARE STATUS {label} {state} source_tree_sha256={expected_tree}")
        return 0 if state in ("READY", "NOT_PREPARED") else (3 if state == "DIRTY" else 1)
    if state == "READY":
        status("READY", f"{'/'.join(present)} identical to the frozen baseline {expected_tree[:16]}... -- nothing to do")
        marker.update({"schema": "hpcperf-materialized-1", "benchmark": app, "variant": variant, "source_version": lock["benchmark"]["source_version"],
                       "source_tree_sha256": expected_tree, "artifact_filename": art["filename"], "archive_sha256": art["sha256"],
                       "verified": now(), "tool": TOOL, "scan_allow": lock.get("scan_allow", [])})
        marker.setdefault("materialized", marker.get("verified"))
        hs.dump_yaml(marker, os.path.join(app_dir, hl.MARKER))
        print(f"PREPARE OK {label} READY source_tree_sha256={expected_tree}")
        return 0
    if state in ("DIRTY", "INVALID"):
        if not a.force_rematerialize:
            status(state, f"{'/'.join(present)} exist but {'are not a real tree' if state == 'INVALID' else 'differ from the frozen baseline'} "
                          f"(tree {have[:16] + '...' if have else '?'} vs expected {expected_tree[:16]}...)")
            die("local modifications are never overwritten automatically; pass --force-rematerialize to DISCARD them and re-materialize", 3 if state == "DIRTY" else 1)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for d in present:
            old = os.path.join(app_dir, d); trash = os.path.join(app_dir, f".discarded-{d}-{stamp}")
            if os.path.islink(old):
                os.remove(old)
            else:
                os.rename(old, trash); shutil.rmtree(trash, ignore_errors=True)
            log(f"--force-rematerialize: discarded existing {d} ({state}, was {have[:16] + '...' if have else 'unhashable'})")
        if os.path.exists(os.path.join(app_dir, hl.MARKER)):
            os.remove(os.path.join(app_dir, hl.MARKER))
    else:
        status("NOT_PREPARED", f"{label}: no src/ yet")

    # C/D. artifact
    path, origin, art_status = resolve_artifact(lock, a.artifact, cdir, a.offline)
    status(art_status, f"{art['filename']} via {origin}, size {art['size']}, sha256 {art['sha256'][:16]}...")

    # E/F. extraction into a staging directory outside the benchmark directory (same filesystem)
    stage_root = os.path.join(os.path.dirname(app_dir), ".materialize-staging")
    os.makedirs(stage_root, exist_ok=True)
    tmp = os.path.join(stage_root, f"{label}.{os.getpid()}")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    try:
        tar_tmp = os.path.join(tmp, "artifact.tar")
        hs.zstd_decompress_to_tar(path, tar_tmp)
        try:
            hs.safe_extract_tar(tar_tmp, tmp)
        except hs.SourceError as ex:
            die(f"unsafe archive content: {ex}", 5)
        os.remove(tar_tmp)
        manifest_file = os.path.join(tmp, "ARTIFACT_MANIFEST.json")
        if os.path.isfile(manifest_file):
            os.remove(manifest_file)   # optional descriptive copy; never part of the tree identity
        entries = hs.manifest(tmp)
        tree = hs.tree_hash_from_manifest(entries)
        if tree != expected_tree:
            die(f"extracted tree sha256 {tree} != recorded {expected_tree} -- the artifact does not carry the frozen source", 5)
        status("TREE_VERIFIED", f"source_tree_sha256 {tree[:16]}... ({len(entries)} entries)")
        layout = lock["materialized_tree"]["layout"]
        for top in layout:
            if not os.path.isdir(os.path.join(tmp, top.rstrip("/"))):
                die(f"artifact lacks {top} required by the lock", 5)
        for entry in os.listdir(tmp):
            if entry not in ("src", "deps"):
                die(f"unexpected top-level entry in the artifact: {entry}", 5)
        esc = hs.escaping_symlinks(tmp, entries)
        if esc:
            die("symlinks escaping the source tree: " + ", ".join(f"{e['path']} -> {e['target']}" for e in esc[:10]), 5)
        hits = hs.scan_tree(tmp, entries, allow=lock.get("scan_allow", []))
        if hits:
            for h in hits[:30]:
                print(f"prepare_benchmark: SCAN HIT rule={h['rule']} path={h['path']}", file=sys.stderr)
            die(f"{len(hits)} credential/artifact hit(s) in the extracted tree -- not materialized", 5)
        sfx = hl.suffix(variant)
        for f in (f"upstream{sfx}.lock", f"patch_series{sfx}.txt", f"SOURCE_MANIFEST{sfx}.json", f"LICENSES{sfx}.md"):
            if not os.path.isfile(os.path.join(app_dir, "provenance", f)):
                die(f"provenance incomplete: provenance/{f} missing", 1)
        # G. atomic rename
        for top in ("src", "deps"):
            s = os.path.join(tmp, top)
            if os.path.isdir(s):
                os.rename(s, os.path.join(app_dir, top))
        hs.dump_yaml({"schema": "hpcperf-materialized-1", "benchmark": app, "variant": variant, "source_version": lock["benchmark"]["source_version"],
                      "source_tree_sha256": expected_tree, "artifact_filename": art["filename"], "archive_sha256": art["sha256"],
                      "artifact_origin": origin, "artifact_status": art_status, "materialized": now(), "verified": now(),
                      "tool": TOOL, "scan_allow": lock.get("scan_allow", [])}, os.path.join(app_dir, hl.MARKER))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            os.rmdir(stage_root)
        except OSError:
            pass
    status("MATERIALIZED", f"{os.path.relpath(app_dir)}/{{src{',deps' if os.path.isdir(os.path.join(app_dir, 'deps')) else ''}}} (variant {variant or '-'}, tree {expected_tree[:16]}...)")
    print(f"PREPARE OK {label} MATERIALIZED source_tree_sha256={expected_tree} archive_sha256={art['sha256']} origin={origin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
