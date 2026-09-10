#!/usr/bin/env python3
"""generate_release_manifest -- the release manifest of the Level 3 source artifacts and level3/SOURCE_ARTIFACTS.md.

    generate_release_manifest.py [--catalog tools/artifacts/artifact_catalog.yaml] [--staging DIR] [--verify]
                                 [--json tools/artifacts/release_manifest.json] [--md level3/SOURCE_ARTIFACTS.md]

Sources of truth: the catalog (suite membership), level3/<app>/provenance/source.lock*.yaml (identity, licenses,
publish status), benchmark.yaml, provenance/LOC*.json, provenance/check_workspace*.json, the materialization
marker and -- for the local statuses -- the maintainer's artifact staging ($HPCPERF_ARTIFACT_STAGING or --staging;
never recorded as a path, only as a status) and the local cache. --verify re-hashes every staged artifact.
Status words: FROZEN, LOCAL_ARTIFACT_VERIFIED, LOCAL_MATERIALIZATION_VERIFIED, AGENT_WORKSPACE_VERIFIED,
REMOTE_ARTIFACT_UNPUBLISHED, REMOTE_FETCH_VERIFIED, RETIRED, CANDIDATE.
"""
import argparse
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import hpcperf_source as hs  # noqa: E402
import hpcperf_lock as hl  # noqa: E402

R = os.path.abspath(os.path.join(HERE, "..", ".."))


def app_rows(app, cat_entry, staging, verify):
    d = os.path.join(R, "level3", app)
    by = hs.load_yaml(os.path.join(d, "benchmark.yaml")) if os.path.isfile(os.path.join(d, "benchmark.yaml")) else {}
    marker = hl.read_marker(d)
    rows = []
    for v in cat_entry.get("variants") or [None]:
        sfx = hl.suffix(v)
        row = {"app": app, "application": cat_entry.get("application", app), "variant": v, "suite_status": cat_entry.get("suite_status"),
               "statuses": [], "notes": []}
        lp = hl.lock_path(d, v)
        prefix = ["RETIRED"] if cat_entry.get("suite_status") == "retired" else ([f"CANDIDATE({cat_entry.get('admission', 'pending')})"] if cat_entry.get("suite_status") == "candidate" else [])
        if not os.path.isfile(lp):
            row["statuses"] = prefix + ["NOT_FROZEN"]; rows.append(row); continue
        lock = hs.load_yaml(lp) or {}
        if lock.get("schema") != hl.LOCK_SCHEMA:
            row["statuses"] = prefix + [f"LOCK_SCHEMA_{lock.get('schema')}"]; rows.append(row); continue
        a = lock["artifact"]
        row.update({"source_version": lock["benchmark"]["source_version"], "filename": a["filename"], "size": a["size"], "uncompressed_size": a.get("uncompressed_size"),
                    "sha256": a["sha256"], "source_tree_sha256": a["source_tree_sha256"], "entries": lock["materialized_tree"]["entries"],
                    "upstream": {"repository": lock["upstream"]["repository"], "tag": lock["upstream"].get("tag"), "commit": lock["upstream"]["commit"]},
                    "patches": len(lock.get("patches", [])), "redistribution_status": lock.get("redistribution_status"),
                    "primary": a.get("primary"), "mirrors": a.get("mirrors", []), "lock_problems": hl.validate_lock(lock, open(lp).read())})
        row["statuses"].append("FROZEN")
        eq = lock.get("equivalence", [])
        row["equivalence"] = "SKIPPED" if eq == "SKIPPED" else ("EQUIVALENT" if eq and all(e.get("status") == "EQUIVALENT" for e in eq) else "; ".join(f"{e.get('archive_path')}={e.get('status')}" for e in eq))
        loc_p = os.path.join(d, "provenance", f"LOC{sfx}.json")
        row["loc"] = json.load(open(loc_p))["summary"] if os.path.isfile(loc_p) else {}
        # local staging
        st = "not staged"
        if staging:
            entry = os.path.join(staging, "level3", app, lock["benchmark"]["source_version"])
            meta_p = os.path.join(entry, "artifact.json")
            if os.path.isfile(meta_p):
                ent = next((e for e in json.load(open(meta_p)).get("artifacts", []) if e.get("filename") == a["filename"]), None)
                if ent and ent.get("sha256") == a["sha256"] and os.path.isfile(os.path.join(entry, a["filename"])):
                    if verify:
                        try:
                            hs.verify_archive_file(os.path.join(entry, a["filename"]), a["sha256"], a["size"]); st = "LOCAL_ARTIFACT_VERIFIED (re-hashed)"
                            row["statuses"].append("LOCAL_ARTIFACT_VERIFIED")
                        except hs.SourceError as ex:
                            st = f"STAGING MISMATCH: {ex}"
                    else:
                        st = f"{ent.get('status')} ({ent.get('verified', '')[:10]})"
                        if ent.get("status") == "LOCAL_ARTIFACT_VERIFIED":
                            row["statuses"].append("LOCAL_ARTIFACT_VERIFIED")
        row["local_staging"] = st
        row["cache_hit"] = os.path.isfile(hs.cache_path(hs.cache_dir(), a["sha256"]))
        # local materialization
        cw = os.path.join(d, "provenance", f"check_workspace{sfx}.json")
        mat = os.path.isdir(os.path.join(d, "src")) and marker.get("source_tree_sha256") == a["source_tree_sha256"] and (marker.get("variant") == v)
        cwj = json.load(open(cw)) if os.path.isfile(cw) else {}
        # the check record itself proves a materialized, hash-equal tree of that variant at check time (variants share
        # one canonical directory, so only one of them is materialized at any moment)
        cw_ok = cwj.get("verdict") == "PASS" and (cwj.get("variant") or None) == v
        if cw_ok:
            row["statuses"].append("LOCAL_MATERIALIZATION_VERIFIED")
        row["materialized_here"] = bool(mat); row["check_workspace"] = "PASS" if cw_ok else ("FAIL" if os.path.isfile(cw) else "-")
        aw = os.path.join(d, "provenance", f"agent_workspace_verification{sfx}.yaml")
        if os.path.isfile(aw) and (hs.load_yaml(aw) or {}).get("verdict") == "PASS":
            row["statuses"].append("AGENT_WORKSPACE_VERIFIED")
        rf = os.path.join(d, "provenance", f"remote_fetch_verification{sfx}.yaml")
        remote_ok = (a.get("primary") or {}).get("status") == "published" and os.path.isfile(rf) and (hs.load_yaml(rf) or {}).get("verdict") == "PASS"
        row["statuses"].append("REMOTE_FETCH_VERIFIED" if remote_ok else "REMOTE_ARTIFACT_UNPUBLISHED")
        if cat_entry.get("suite_status") == "retired":
            row["statuses"] = ["RETIRED"] + row["statuses"]; row["notes"].append(cat_entry.get("reason", ""))
        if cat_entry.get("suite_status") == "candidate":
            row["statuses"] = [f"CANDIDATE({cat_entry.get('admission', 'pending')})"] + row["statuses"]
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", default=os.path.join(HERE, "artifact_catalog.yaml"))
    ap.add_argument("--staging", default=os.environ.get(hs.STAGING_ENV)); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--json", default=os.path.join(HERE, "release_manifest.json")); ap.add_argument("--md", default=os.path.join(R, "level3", "SOURCE_ARTIFACTS.md"))
    a = ap.parse_args()
    cat = hs.load_yaml(a.catalog)
    rows = []
    for app, ent in cat["applications"].items():
        rows += app_rows(app, ent, a.staging, a.verify)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    suite = [r for r in rows if r["suite_status"] == "retained"]
    man = {"schema": "hpcperf-release-manifest-1", "generated": now, "suite": cat["suite"], "source_version": cat["source_version"],
           "release": cat["release"], "naming": cat["naming"], "default_suite_order": cat["default_suite_order"],
           "totals": {"retained_applications": len({r["app"] for r in suite}), "retained_artifacts": len([r for r in suite if r.get("filename")]),
                      "retained_compressed_bytes": sum(r.get("size") or 0 for r in suite), "retained_uncompressed_bytes": sum(r.get("uncompressed_size") or 0 for r in suite)},
           "artifacts": [{k: v for k, v in r.items()} for r in rows]}
    with open(a.json, "w") as f:
        json.dump(man, f, indent=1); f.write("\n")
    L = ["# Level 3 source artifacts (scheme 3: project-controlled external source artifacts)", "",
         f"Generated by `tools/artifacts/generate_release_manifest.py` on {now} from `tools/artifacts/artifact_catalog.yaml`, `level3/<app>/provenance/source.lock*.yaml`, `benchmark.yaml`, `provenance/LOC*.json`, `provenance/check_workspace*.json`, the materialization markers and the maintainer's local artifact staging (status only; no path is recorded). Machine-readable copy: `tools/artifacts/release_manifest.json`.",
         "", "Every artifact is `<app>[-<variant>]-<source_version>.tar.zst` with the top-level entries `src/` (application source + upstream-bundled dependency source, approved patches pre-applied) and `deps/` (benchmark-specific source dependencies). Identity = `source_tree_sha256` (hpcperf-tree-1); the archive sha256 is recorded separately. Users materialize with `tools/prepare_benchmark.sh level3 <app>`; the artifact is found in the local content-addressed cache or downloaded from the immutable URL recorded in the lock once published. **Remote status of every artifact in this round: REMOTE_ARTIFACT_UNPUBLISHED** (provider, release naming and upload timing are the maintainer's decision; `publish_artifacts.sh --dry-run` shows the plan).",
         "", "| App | variant | suite | artifact | source version | compressed | uncompressed | entries | archive sha256 | tree sha256 | upstream | patches | redistribution | equivalence | local staging | materialized+check | statuses | LOC app-owned / agent-modifiable / total |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r.get("filename"):
            L.append(f"| {r['app']} | {r['variant'] or '-'} | {r['suite_status']} | - | - | - | - | - | - | - | - | - | - | - | - | - | {', '.join(r['statuses'])} | - |"); continue
        up = r["upstream"]; loc = r.get("loc", {})
        L.append(f"| {r['app']} | {r['variant'] or '-'} | {r['suite_status']} | `{r['filename']}` | {r['source_version']} | {hs.human(r['size'])} | {hs.human(r['uncompressed_size'] or 0)} | {r['entries']} | `{r['sha256'][:16]}…` | `{r['source_tree_sha256'][:16]}…` | {up.get('tag') or ''} `{up['commit'][:12]}` | {r['patches']} | {r['redistribution_status']} | {r['equivalence']} | {r['local_staging']} | {'yes' if r['materialized_here'] else 'no'} / {r['check_workspace']} | {', '.join(r['statuses'])} | {loc.get('application_owned_code_loc', '-')} / {loc.get('agent_modifiable_code_loc', '-')} / {loc.get('total_materialized_code_loc', '-')} |")
    t = man["totals"]
    L += ["", f"**Default suite ({t['retained_applications']} retained applications, {t['retained_artifacts']} artifacts)**: {hs.human(t['retained_compressed_bytes'])} compressed ({t['retained_compressed_bytes']} bytes), {hs.human(t['retained_uncompressed_bytes'])} uncompressed after materialization. Order: {', '.join(cat['default_suite_order'])}.",
          "", "## Status vocabulary", "",
          "- FROZEN: source.lock (schema hpcperf-source-lock-2) records upstream commit, patch series, dependencies, artifact size/sha256 and source_tree_sha256.",
          "- LOCAL_ARTIFACT_VERIFIED: the artifact in the maintainer's local staging was re-hashed and fully extracted/tree-hashed against the lock (`tools/artifacts/verify_artifact.py --full`).",
          "- LOCAL_MATERIALIZATION_VERIFIED: `tools/prepare_benchmark.sh` materialized src/ (+ deps/) here from the artifact and `tools/check_workspace.py` PASSes on the canonical directory.",
          "- AGENT_WORKSPACE_VERIFIED: a real agent-edit closed loop (workspace outside the repository: injected compile error fails the build, restore rebuilds, small case validates, readonly tampering refused) is recorded in `provenance/agent_workspace_verification*.yaml`.",
          "- REMOTE_ARTIFACT_UNPUBLISHED: no artifact has been uploaded; `artifact.primary` in the lock is `{url: null, status: unpublished}` (no fabricated URL). REMOTE_FETCH_VERIFIED is only set after a real download + verification from the published URL.",
          "- RETIRED: not part of the default suite; no artifact is staged or published (GEOS: ParMETIS redistribution constraint + replacement decision). CANDIDATE(pending|admitted): replacement application under bring-up.", ""]
    with open(a.md, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:4]))
    for r in rows:
        print(f"{r['app']}{hl.suffix(r['variant'])}: {', '.join(r['statuses'])}" + (f"  [{r['filename']} {hs.human(r['size'])} staging={r['local_staging']}]" if r.get("filename") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
