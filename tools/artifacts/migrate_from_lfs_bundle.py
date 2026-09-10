#!/usr/bin/env python3
"""migrate_from_lfs_bundle -- one-time migration of a scheme-2 archive (level3/<app>/archives/*.tar.zst, Git LFS
design, abandoned before release) into scheme-3 local artifact staging, without re-freezing.

    migrate_from_lfs_bundle.py level3/<app> [--variant NAME] --staging DIR --source-version hpcperf-l3-v1
                               --redistribution-status cleared|blocked|review [--delete-old] [--scratch DIR]

Steps (copy first, verify, then delete -- never move-then-check):
  1 read the scheme-2 lock (schema hpcperf-source-lock-1) + benchmark.yaml + freeze spec;
  2 copy archives/<old> -> <staging>/level3/<app>/<source_version>/<app>[-<variant>]-<source_version>.tar.zst
    (byte-identical: the archive content is unchanged, only its name and location change);
  3 verify the copy: size, zstd magic, sha256 == old lock; full extraction: layout, source_tree_sha256, scan;
  4 write <staging>/.../artifact.json (+ SHA256SUMS) with status LOCAL_ARTIFACT_VERIFIED / REMOTE_ARTIFACT_UNPUBLISHED;
  5 rewrite provenance/source.lock[.variant].yaml as schema hpcperf-source-lock-2 (all provenance kept: upstream,
    patches, dependencies, components, equivalence, licenses; `migration` block records the old archive path and
    that the sha256 is identical); benchmark.yaml identity -> source_artifact / variants.<v>; the freeze spec's
    benchmark_source_version -> <source_version> (its `archive:` key is dropped);
  6 --delete-old: remove archives/<old> (and the empty archives/ directory) only after steps 3-5 succeeded.
"""
import argparse
import datetime
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import hpcperf_source as hs  # noqa: E402
import hpcperf_lock as hl  # noqa: E402
import verify_artifact as va  # noqa: E402


def die(msg):
    print(f"migrate: FAIL -- {msg}", file=sys.stderr); sys.exit(1)


def log(msg):
    print(f"migrate: {msg}", flush=True)


def portable(s):
    """equivalence records: replace a node-local scratch path with a descriptive placeholder"""
    if isinstance(s, str) and s.startswith("/tmp/"):
        return "<node-local-scratch>/" + "/".join(s.split("/")[3:])
    return s


def source_scope(app_dir):
    sp = os.path.join(app_dir, "optimization_scope.yaml")
    sc = (hs.load_yaml(sp) or {}).get("loc_categories", {}) if os.path.isfile(sp) else {}
    return {"application_owned": sc.get("application_owned", []), "bundled": sc.get("bundled_dependency", []),
            "benchmark_specific": sc.get("benchmark_specific_dependency", []), "test": sc.get("test", []), "exclude": sc.get("exclude", [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("app_dir"); ap.add_argument("--variant"); ap.add_argument("--staging", required=True)
    ap.add_argument("--source-version", required=True); ap.add_argument("--redistribution-status", required=True, choices=hl.REDISTRIBUTION_STATUSES)
    ap.add_argument("--delete-old", action="store_true"); ap.add_argument("--scratch"); ap.add_argument("--suite-status", default="retained", choices=hl.SUITE_STATUSES)
    a = ap.parse_args()
    app_dir = os.path.abspath(a.app_dir); app = os.path.basename(app_dir); variant = a.variant; sfx = hl.suffix(variant)
    lock_p = hl.lock_path(app_dir, variant)
    old = hs.load_yaml(lock_p) or {}
    if old.get("schema") == hl.LOCK_SCHEMA:
        # already migrated: only the deferred deletion of the old archive remains (after the prepare test)
        mig = old.get("migration") or {}
        old_arch = os.path.join(app_dir, mig.get("old_archive_path", "archives/none"))
        if a.delete_old and os.path.isfile(old_arch):
            if hs.sha256_file(old_arch) != old["artifact"]["sha256"]:
                die("old archive sha256 differs from the migrated lock -- not deleting")
            staged = os.path.join(hl.staging_entry(a.staging, app, old["benchmark"]["source_version"]), old["artifact"]["filename"])
            hs.verify_archive_file(staged, old["artifact"]["sha256"], old["artifact"]["size"])
            os.remove(old_arch)
            d = os.path.dirname(old_arch)
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
            log(f"old archive {mig.get('old_archive_path')} deleted (staging copy re-verified: size + sha256)")
        else:
            log(f"{os.path.relpath(lock_p)} is already schema 2 -- nothing to migrate")
        return 0
    if old.get("schema") != "hpcperf-source-lock-1":
        die(f"{lock_p}: unexpected schema {old.get('schema')!r}")
    by_p = os.path.join(app_dir, "benchmark.yaml"); by = hs.load_yaml(by_p)
    spec_p = os.path.join(app_dir, "provenance", f"freeze_spec{sfx}.yaml"); spec = hs.load_yaml(spec_p)
    old_arch_rel = old["archive"]["path"]; old_arch = os.path.join(app_dir, old_arch_rel)
    if not os.path.isfile(old_arch):
        die(f"old archive {old_arch_rel} missing")
    tree = old["materialized_tree"]["sha256"]
    ident = (by.get("variants") or {}).get(variant, {}) if variant else (by.get("source_bundle") or {})
    if ident.get("archive_sha256") != old["archive"]["sha256"] or ident.get("source_tree_sha256") != tree:
        die("benchmark.yaml identity disagrees with the scheme-2 lock")

    # 2. copy into staging (byte-identical, new name)
    fn = hs.artifact_filename(app, variant, a.source_version)
    entry = os.path.join(os.path.abspath(a.staging), "level3", app, a.source_version)
    os.makedirs(entry, exist_ok=True)
    dst = os.path.join(entry, fn)
    if os.path.isfile(dst) and hs.sha256_file(dst) == old["archive"]["sha256"]:
        log(f"staging copy already present: {os.path.relpath(dst, a.staging)}")
    else:
        tmp = dst + f".tmp.{os.getpid()}"
        shutil.copyfile(old_arch, tmp); os.chmod(tmp, 0o640); os.replace(tmp, dst)
        log(f"copied {old_arch_rel} -> {os.path.relpath(dst, a.staging)} ({hs.human(os.path.getsize(dst))})")

    # 5a. build the schema-2 lock in memory first (verification below uses it)
    comps = old.get("components", [])
    app_comp = next(c for c in comps if c["dest"] == "src")
    patches = [{"path": p["path"], "sha256": p["sha256"], "category": p.get("category", ""), "upstream_reference": p.get("upstream_source", ""),
                "component": p.get("component", "src"), "files": p.get("files", [])} for p in old.get("patches", [])]
    eq = old.get("equivalence", [])
    if isinstance(eq, list):
        eq = [{**e, "validated_tree": portable(e.get("validated_tree"))} for e in eq]
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lock = hl.make_lock(name=app, application=old["application"], variant=variant, source_version=a.source_version,
                        upstream={"url": old["upstream"]["url"], "tag": old["upstream"]["tag"], "commit": old["upstream"]["commit"]},
                        archive_info={"sha256": old["archive"]["sha256"], "compressed_size": old["archive"]["compressed_size"], "uncompressed_size": old["archive"].get("uncompressed_size"),
                                      "tar_entries": old["archive"].get("tar_entries"), "file_count": old["archive"].get("file_count"), "symlink_count": old["archive"].get("symlink_count"),
                                      "zstd_version": old["archive"].get("zstd_version")},
                        tree_sha=tree, entries=old["materialized_tree"]["entries"], layout=old["materialized_tree"]["layout"],
                        patches=patches, dependencies=old.get("dependencies", {}), components=comps, equivalence=eq,
                        licenses=old.get("licenses", []), license_notes=spec.get("license_notes", []), redistribution_status=a.redistribution_status,
                        source_scope=source_scope(app_dir), scan_allow=spec.get("scan_allow", []),
                        freeze_tool_version=old.get("freeze_tool_version", "freeze-1.0"), freeze_timestamp=old.get("freeze_timestamp"),
                        migration={"from_schema": "hpcperf-source-lock-1", "from_scheme": "git-lfs source bundle (design abandoned 2026-09-10, never committed, never pushed)",
                                   "old_archive_path": old_arch_rel, "old_upstream_version_label": old.get("benchmark_source_version"),
                                   "archive_sha256_identical": True, "migrated": now, "tool": "migrate_from_lfs_bundle-1.0"})
    tmp_lock = lock_p + ".migrating"
    hs.dump_yaml(lock, tmp_lock)
    problems = hl.validate_lock(lock, open(tmp_lock).read())
    if problems:
        os.remove(tmp_lock); die("new lock invalid: " + "; ".join(problems))

    # 3. verify the staging copy against the new lock (size, sha256, full extraction, tree hash, scan)
    res = va.verify(tmp_lock, artifact=dst, full=True, scratch=a.scratch, log=lambda m: print("  " + m))
    if res["verdict"] != "PASS":
        os.remove(tmp_lock); die("staging copy failed verification -- old archive NOT deleted")

    # 4. staging metadata
    meta_p = os.path.join(entry, "artifact.json")
    meta = json.load(open(meta_p)) if os.path.isfile(meta_p) else {"schema": "hpcperf-artifact-staging-1", "benchmark": app, "source_version": a.source_version, "artifacts": []}
    meta["artifacts"] = [e for e in meta["artifacts"] if e.get("filename") != fn] + [{
        "filename": fn, "variant": variant, "size": lock["artifact"]["size"], "sha256": lock["artifact"]["sha256"],
        "source_tree_sha256": tree, "format": hs.ARTIFACT_FORMAT, "status": "LOCAL_ARTIFACT_VERIFIED", "remote_status": "REMOTE_ARTIFACT_UNPUBLISHED",
        "verified": now, "origin": {"kind": "migrated-from-lfs-bundle", "old_archive_path": old_arch_rel, "sha256_identical": True},
        "redistribution_status": a.redistribution_status, "suite_status": a.suite_status}]
    meta["artifacts"].sort(key=lambda e: e["filename"])
    with open(meta_p, "w") as f:
        json.dump(meta, f, indent=1); f.write("\n")
    with open(os.path.join(entry, "SHA256SUMS"), "w") as f:
        for e in meta["artifacts"]:
            f.write(hs.sha256sums_line(e["sha256"], e["filename"]))
    log(f"staging metadata written: {os.path.relpath(meta_p, a.staging)}, SHA256SUMS")

    # 5b. commit the lock, benchmark.yaml, freeze spec
    os.replace(tmp_lock, lock_p)
    hl.apply_identity(by, lock); by["suite_status"] = a.suite_status
    hs.dump_yaml(by, by_p)
    spec["benchmark_source_version"] = a.source_version; spec.pop("archive", None)
    spec["redistribution_status"] = a.redistribution_status
    hs.dump_yaml(spec, spec_p)
    log(f"lock -> schema 2, benchmark.yaml identity -> source_artifact, freeze spec version -> {a.source_version}")

    # 6. delete the old archive
    if a.delete_old:
        os.remove(old_arch)
        d = os.path.dirname(old_arch)
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
        log(f"old archive {old_arch_rel} deleted (staging copy verified byte-identical)")
    print(f"MIGRATE OK {app}{sfx} {fn} sha256={lock['artifact']['sha256']} tree={tree}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
