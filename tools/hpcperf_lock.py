#!/usr/bin/env python3
"""hpcperf_lock -- the Level 3 source lock (provenance/source.lock[.variant].yaml, schema hpcperf-source-lock-2)
and the benchmark.yaml identity fields derived from it. Shared by freeze, migrate, prepare, verify, publish
and check_workspace so that the schema is defined in exactly one place.

A lock records WHAT the frozen source is (upstream commit, patch series, dependencies, source_tree_sha256),
WHERE the artifact that carries it can be found (immutable https URL(s) once published; `unpublished`
otherwise -- never a local staging path) and HOW to verify it (archive size + sha256, tree sha256).
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hpcperf_source as hs  # noqa: E402

LOCK_SCHEMA = "hpcperf-source-lock-2"
LOCK_SCHEMA_VERSION = 2
MARKER = ".hpcperf-materialized.yaml"
PUBLISH_STATUSES = ("unpublished", "published")
REDISTRIBUTION_STATUSES = ("cleared", "blocked", "review")
SUITE_STATUSES = ("retained", "candidate", "retired")
# strings that must never appear in a committed lock (node-private / floating locations)
FORBIDDEN_LOCATION_RX = re.compile(r"(^|[\s\"'=:(])(/tmp/|/home/|/projects/|/scratch/|\$HOME|~/|file://|/lustre/|/gpfs/)")
FLOATING_URL_RX = re.compile(r"/(latest|main|master|develop|HEAD)(/|$|\.)")


def suffix(variant):
    return f".{variant}" if variant else ""


def lock_path(app_dir, variant=None):
    return os.path.join(app_dir, "provenance", f"source.lock{suffix(variant)}.yaml")


def load_lock(app_dir, variant=None):
    p = lock_path(app_dir, variant)
    if not os.path.isfile(p):
        raise hs.SourceError(f"{os.path.relpath(p)} missing -- the benchmark was not frozen")
    lock = hs.load_yaml(p) or {}
    if lock.get("schema") != LOCK_SCHEMA:
        raise hs.SourceError(f"{os.path.relpath(p)}: schema {lock.get('schema')!r} is not {LOCK_SCHEMA} "
                             f"(a scheme-2 LFS-era lock must be migrated with tools/artifacts/migrate_from_lfs_bundle.py)")
    return lock


def select_variant(by, requested=None, marker=None):
    """(variant name or None) for a benchmark.yaml: explicit request > materialization marker > env > default."""
    variants = by.get("variants")
    if not variants:
        if requested:
            raise hs.SourceError(f"benchmark has no variants but --variant {requested} was given")
        return None
    env = by.get("variant_env")
    name = requested or (marker or {}).get("variant") or (os.environ.get(env) if env else None) or by.get("default_variant")
    if name not in variants:
        raise hs.SourceError(f"variant {name!r} unknown; available: {', '.join(variants)}")
    return name


def identity_from_benchmark(by, variant):
    """The identity block benchmark.yaml carries for a variant (source_artifact or variants.<v>)."""
    if variant:
        return (by.get("variants") or {}).get(variant) or {}
    return by.get("source_artifact") or {}


def identity_fields(lock):
    a = lock["artifact"]
    return {"filename": a["filename"], "archive_sha256": a["sha256"], "source_tree_sha256": a["source_tree_sha256"],
            "size": a["size"], "uncompressed_size": a.get("uncompressed_size"), "file_count": lock["materialized_tree"]["entries"],
            "source_version": lock["benchmark"]["source_version"], "primary_url": a["primary"].get("url"),
            "publish_status": a["primary"].get("status", "unpublished")}


def apply_identity(by, lock):
    """Write the lock's identity into a benchmark.yaml dict (in place) and drop scheme-2 fields."""
    variant = lock["benchmark"].get("variant")
    ident = identity_fields(lock)
    by.pop("source_bundle", None)
    if variant:
        v = by.setdefault("variants", {}).setdefault(variant, {})
        for k in ("archive", "archive_sha256", "compressed_size"):
            v.pop(k, None)
        v.update(ident)
        by["source_version"] = ident["source_version"]
    else:
        by["source_version"] = ident["source_version"]
        by["source_tree_sha256"] = ident["source_tree_sha256"]
        by["source_artifact"] = ident
    by["upstream_version"] = lock["upstream"].get("tag") or lock["upstream"].get("commit")
    by["redistribution_status"] = lock.get("redistribution_status", "review")
    by.setdefault("suite_status", "retained")
    return by


def make_lock(*, name, application, variant, source_version, upstream, archive_info, tree_sha, entries, layout,
              patches, dependencies, components, equivalence, licenses, license_notes, redistribution_status,
              source_scope, scan_allow, freeze_tool_version, freeze_timestamp, primary=None, mirrors=None,
              migration=None, redistribution_notes=None):
    """Assemble a schema-2 lock dict. archive_info: sha256/compressed_size/uncompressed_size/tar_entries/
    file_count/symlink_count/zstd_version. `primary`: {url, status}; a None url is recorded as unpublished."""
    primary = dict(primary or {})
    if not primary.get("url"):
        primary = {"url": None, "status": "unpublished"}
    else:
        primary.setdefault("status", "published")
    return {
        "schema": LOCK_SCHEMA, "schema_version": LOCK_SCHEMA_VERSION,
        "benchmark": {"name": name, "level": 3, "application": application, "variant": variant, "source_version": source_version},
        "upstream": {"repository": upstream.get("url") or upstream.get("repository"), "tag": upstream.get("tag") or upstream.get("ref"), "commit": upstream["commit"]},
        "artifact": {
            "filename": hs.artifact_filename(name, variant, source_version), "format": hs.ARTIFACT_FORMAT, "layout": list(layout),
            "size": int(archive_info["compressed_size"]), "sha256": archive_info["sha256"], "source_tree_sha256": tree_sha,
            "tree_algorithm": hs.TREE_ALGO, "tar_format": hs.TAR_ALGO,
            "compression": {"tool": "zstd", "level": 19, "single_thread": True, "version": archive_info.get("zstd_version")},
            "uncompressed_size": archive_info.get("uncompressed_size"), "tar_entries": archive_info.get("tar_entries"),
            "file_count": archive_info.get("file_count"), "symlink_count": archive_info.get("symlink_count"),
            "immutable": True, "primary": primary, "mirrors": list(mirrors or []),
        },
        "cache": {"content_addressed": True, "layout": "sha256/<archive_sha256>.tar.zst", "env": hs.CACHE_ENV, "default": "$HPC_PERFORMANCE_AI_ROOT/.artifacts"},
        "materialized_tree": {"sha256": tree_sha, "algorithm": hs.TREE_ALGO, "entries": int(entries), "layout": list(layout)},
        "patches": patches, "dependencies": dependencies, "components": components, "equivalence": equivalence,
        "licenses": licenses, "license_notes": license_notes or [],
        "redistribution_status": redistribution_status, "redistribution_notes": redistribution_notes or [],
        "source_scope": source_scope, "scan_allow": scan_allow or [],
        "freeze": {"tool_version": freeze_tool_version, "timestamp": freeze_timestamp,
                   "note": "freeze timestamp and archive metadata are not part of source_tree_sha256; a re-freeze of identical content reproduces the tree hash (and, with the same zstd, the archive sha256)"},
        **({"migration": migration} if migration else {}),
    }


def validate_lock(lock, text=None):
    """Return a list of problems (empty = valid). `text`: the raw file content, for the location scan."""
    p = []
    if lock.get("schema") != LOCK_SCHEMA or lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        p.append(f"schema must be {LOCK_SCHEMA} / {LOCK_SCHEMA_VERSION}")
        return p
    for k in ("benchmark", "upstream", "artifact", "cache", "materialized_tree", "patches", "dependencies", "licenses", "redistribution_status", "source_scope", "freeze"):
        if k not in lock:
            p.append(f"missing key {k}")
    if p:
        return p
    b, u, a, t = lock["benchmark"], lock["upstream"], lock["artifact"], lock["materialized_tree"]
    for k in ("name", "level", "application", "source_version"):
        if not b.get(k) and b.get(k) != 0:
            p.append(f"benchmark.{k} missing")
    if not u.get("commit") or not re.fullmatch(r"[0-9a-f]{40}", str(u.get("commit"))):
        p.append("upstream.commit must be a full 40-hex commit (branch names are not an identity)")
    if not u.get("repository"):
        p.append("upstream.repository missing")
    try:
        want = hs.artifact_filename(b["name"], b.get("variant"), b["source_version"])
        if a.get("filename") != want:
            p.append(f"artifact.filename {a.get('filename')!r} != convention {want!r}")
    except Exception as ex:  # noqa: BLE001
        p.append(str(ex))
    if a.get("format") != hs.ARTIFACT_FORMAT:
        p.append("artifact.format must be tar.zst")
    if not re.fullmatch(r"[0-9a-f]{64}", str(a.get("sha256"))):
        p.append("artifact.sha256 invalid")
    if not isinstance(a.get("size"), int) or a["size"] <= 0:
        p.append("artifact.size must be a positive integer")
    if not re.fullmatch(r"[0-9a-f]{64}", str(a.get("source_tree_sha256"))):
        p.append("artifact.source_tree_sha256 invalid")
    if a.get("source_tree_sha256") != t.get("sha256"):
        p.append("artifact.source_tree_sha256 != materialized_tree.sha256")
    if t.get("algorithm") != hs.TREE_ALGO:
        p.append(f"materialized_tree.algorithm must be {hs.TREE_ALGO}")
    if not t.get("layout") or "src/" not in t["layout"]:
        p.append("materialized_tree.layout must contain src/")
    if a.get("layout") != t.get("layout"):
        p.append("artifact.layout != materialized_tree.layout")
    prim = a.get("primary") or {}
    st = prim.get("status")
    if st not in PUBLISH_STATUSES:
        p.append(f"artifact.primary.status must be one of {PUBLISH_STATUSES}")
    url = prim.get("url")
    if st == "published":
        if not url:
            p.append("artifact.primary.url required when status is published")
    elif url:
        p.append("artifact.primary.url must be null while unpublished (no fabricated future URL)")
    for m in [url] + list(a.get("mirrors") or []):
        if not m:
            continue
        if not re.match(r"^https://", m):
            p.append(f"artifact url must be https: {m}")
        if FLOATING_URL_RX.search(m):
            p.append(f"artifact url looks floating (latest/branch), not immutable: {m}")
        if not m.endswith(a.get("filename", "")):
            p.append(f"artifact url must end with the artifact filename: {m}")
    if lock.get("redistribution_status") not in REDISTRIBUTION_STATUSES:
        p.append(f"redistribution_status must be one of {REDISTRIBUTION_STATUSES}")
    if not lock["cache"].get("content_addressed"):
        p.append("cache.content_addressed must be true")
    for k in ("lfs", "git_lfs", "lfs_pointer"):
        if k in lock or k in a:
            p.append(f"obsolete Git LFS metadata key {k}")
    if text is not None:
        for i, line in enumerate(text.splitlines(), 1):
            if FORBIDDEN_LOCATION_RX.search(line):
                p.append(f"line {i}: node-private or non-immutable location recorded in the lock")
    return p


def lock_text_problems(path):
    lock = hs.load_yaml(path) or {}
    return validate_lock(lock, open(path).read())


def portable_location(s):
    """Provenance records must not carry node-private paths: a /tmp scratch path becomes a descriptive placeholder."""
    if isinstance(s, str) and s.startswith("/tmp/"):
        return "<node-local-scratch>/" + "/".join(s.split("/")[3:])
    return s


def source_scope_from_optimization_scope(app_dir):
    sp = os.path.join(app_dir, "optimization_scope.yaml")
    sc = (hs.load_yaml(sp) or {}).get("loc_categories", {}) if os.path.isfile(sp) else {}
    return {"application_owned": sc.get("application_owned", []), "bundled": sc.get("bundled_dependency", []),
            "benchmark_specific": sc.get("benchmark_specific_dependency", []), "test": sc.get("test", []), "exclude": sc.get("exclude", [])}


def staging_entry(staging, app, source_version):
    return os.path.join(os.path.abspath(staging), "level3", app, source_version)


def update_staging_metadata(entry, record):
    """Merge one artifact record into <entry>/artifact.json and rewrite SHA256SUMS."""
    import json
    meta_p = os.path.join(entry, "artifact.json")
    meta = json.load(open(meta_p)) if os.path.isfile(meta_p) else {"schema": "hpcperf-artifact-staging-1", "benchmark": record.get("benchmark"), "source_version": record.get("source_version"), "artifacts": []}
    meta["artifacts"] = sorted([e for e in meta["artifacts"] if e.get("filename") != record["filename"]] + [record], key=lambda e: e["filename"])
    with open(meta_p, "w") as f:
        json.dump(meta, f, indent=1); f.write("\n")
    with open(os.path.join(entry, "SHA256SUMS"), "w") as f:
        for e in meta["artifacts"]:
            f.write(hs.sha256sums_line(e["sha256"], e["filename"]))
    return meta_p


def read_marker(app_dir):
    p = os.path.join(app_dir, MARKER)
    return (hs.load_yaml(p) or {}) if os.path.isfile(p) else {}
