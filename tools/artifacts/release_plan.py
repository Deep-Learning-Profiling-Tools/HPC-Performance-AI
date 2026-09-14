#!/usr/bin/env python3
"""release_plan -- the publication plan of the Level 3 source artifacts for one release (nothing is uploaded).

    release_plan.py --tag level3-source-hpcperf-l3-v1-rc1 --commit <sha> [--staging DIR] [--catalog FILE]
                    [--json level3/RELEASE_PLAN.json] [--md level3/RELEASE_PLAN.md] [--full]

Provider: GitHub Release assets of this repository (decision 2026-09-11). Planned asset URLs follow
https://github.com/<owner>/<repo>/releases/download/<tag>/<filename>; they are PLANNED, not published: the locks
keep `primary: {url: null, status: unpublished}` until the asset exists and was re-downloaded and verified.
Per artifact: filename, byte size, archive sha256, source_tree_sha256, application license, bundle dependency/data
license review (separately), secret/build-output scan status (re-verified with --full), SOURCE_MANIFEST and
SHA256SUMS, known scientific status, build prerequisites. Excluded: retired applications (GEOS/ParMETIS: never),
candidates not admitted (listed as CANDIDATE_EXCLUDED), anything failing verify_artifact.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import hpcperf_source as hs  # noqa: E402
import hpcperf_lock as hl  # noqa: E402
import verify_artifact as va  # noqa: E402

R = os.path.abspath(os.path.join(HERE, "..", ".."))
OWNER_REPO = "Deep-Learning-Profiling-Tools/HPC-Performance-AI"


def sci_status(by):
    sm = by.get("scale_modes") or {}
    return {"validated_backends": by.get("validated_backends"), "smoke": sm.get("smoke"), "strong": sm.get("strong"), "weak": sm.get("weak"),
            "multi_node": by.get("multi_node"), "gpu_40_80": by.get("gpu_40_80"), "known_limits": by.get("known_limits"), "admission": by.get("admission")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True); ap.add_argument("--commit", required=True)
    ap.add_argument("--staging", default=os.environ.get(hs.STAGING_ENV)); ap.add_argument("--catalog", default=os.path.join(HERE, "artifact_catalog.yaml"))
    ap.add_argument("--json", default=os.path.join(R, "level3", "RELEASE_PLAN.json")); ap.add_argument("--md", default=os.path.join(R, "level3", "RELEASE_PLAN.md"))
    ap.add_argument("--full", action="store_true", help="re-extract every artifact (tree hash + scan)")
    a = ap.parse_args()
    if not a.staging:
        sys.exit("release_plan: --staging or HPCPERF_ARTIFACT_STAGING required (maintainer-side)")
    cat = hs.load_yaml(a.catalog)
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        imm = json.load(open(os.path.join(HERE, "immutable_releases_probe.json")))
    except OSError:
        imm = {"enabled": None, "note": "not probed"}
    assets, excluded = [], []
    for app, ent in cat["applications"].items():
        d = os.path.join(R, "level3", app)
        by = hs.load_yaml(os.path.join(d, "benchmark.yaml")) if os.path.isfile(os.path.join(d, "benchmark.yaml")) else {}
        for v in ent.get("variants") or [None]:
            lp = hl.lock_path(d, v)
            lock = hs.load_yaml(lp) if os.path.isfile(lp) else None
            label = f"{app}{hl.suffix(v)}"
            if ent.get("suite_status") == "retired":
                excluded.append({"artifact": label, "reason": "RETIRED (never published): " + ent.get("reason", "")}); continue
            if ent.get("suite_status") == "candidate" and ent.get("admission") != "admitted":
                excluded.append({"artifact": label, "reason": f"CANDIDATE_EXCLUDED (admission {ent.get('admission')}): " + str(by.get("admission", ""))}); continue
            acc = ent.get("release_acceptance") or by.get("release_acceptance") or "accepted"
            if str(acc).startswith(("on_hold", "under_review")):
                excluded.append({"artifact": label, "reason": f"RELEASE_ACCEPTANCE_{str(acc).upper()}: {by.get('release_acceptance_note', 'acceptance is on hold; not part of an unconditional publish-all')}"}); continue
            if not lock or lock.get("schema") != hl.LOCK_SCHEMA:
                excluded.append({"artifact": label, "reason": "no schema-2 lock"}); continue
            art = lock["artifact"]; entry = hl.staging_entry(a.staging, app, lock["benchmark"]["source_version"]); local = os.path.join(entry, art["filename"])
            res = va.verify(lp, artifact=local, staging_entry=entry, full=a.full, log=None) if os.path.isfile(local) else {"verdict": "FAIL", "checks": [{"check": 2, "name": "artifact file", "status": "FAIL", "detail": "missing from staging"}]}
            lic = lock.get("licenses", []); app_lic = next((l for l in lic if l.get("path") == "src"), {})
            man_p = os.path.join(d, "provenance", f"SOURCE_MANIFEST{hl.suffix(v)}.json")
            rec = {"artifact": label, "application": ent.get("application"), "variant": v, "filename": art["filename"], "size_bytes": art["size"], "sha256": art["sha256"],
                   "source_tree_sha256": art["source_tree_sha256"], "source_version": lock["benchmark"]["source_version"], "upstream": lock["upstream"],
                   "planned_url": f"https://github.com/{OWNER_REPO}/releases/download/{a.tag}/{art['filename']}", "planned_url_status": "PLANNED (not published; lock primary stays url null / unpublished)",
                   "application_license": {"spdx": (app_lic.get("license") or (lock.get("components", [{}])[0].get("license") or {}).get("spdx")), "project": app_lic.get("project")},
                   "bundle_dependency_and_data_license_review": {"status": lock.get("redistribution_status"), "components_reviewed": len(lic), "record": f"level3/{app}/provenance/LICENSES{hl.suffix(v)}.md", "notes": lock.get("license_notes", [])},
                   "verification": {"verdict": res["verdict"], "checks": [c for c in res["checks"]], "mode": "full re-extraction" if a.full else "size/sha256/staging metadata"},
                   "sha256sums_line": hs.sha256sums_line(art["sha256"], art["filename"]).strip(),
                   "source_manifest": {"path": os.path.relpath(man_p, R), "sha256": hs.sha256_file(man_p) if os.path.isfile(man_p) else None, "planned_asset": f"SOURCE_MANIFEST.{label}.json"},
                   "patches": [p["path"] for p in lock.get("patches", [])], "scientific_status": sci_status(by), "build_prerequisites": lock.get("dependencies", {}).get("environment_provided", []),
                   "decision": "PLAN" if res["verdict"] == "PASS" and lock.get("redistribution_status") == "cleared" else "REFUSE"}
            (assets if rec["decision"] == "PLAN" else excluded).append(rec if rec["decision"] == "PLAN" else {"artifact": label, "reason": f"verification {res['verdict']} / redistribution {lock.get('redistribution_status')}"})
    total = sum(x["size_bytes"] for x in assets)
    plan = {"schema": "hpcperf-release-plan-1", "generated": now, "provider": "GitHub Release assets of this repository (no mirror, bucket or paid service)",
            "repository": OWNER_REPO, "tag": a.tag, "target_commit": a.commit, "release_kind": "prerelease (source-artifact release candidate): does not assert that HIP, multi-node or every scientific configuration is validated",
            "immutable_releases": imm, "asset_policy": "assets are never overwritten or deleted after publication; a source change is a new source_version + new tag; every consumer verifies size + sha256 + source_tree_sha256 (prepare_benchmark.sh) -- platform immutability is NOT relied upon",
            "publication_procedure": ["draft release on the target commit (maintainer authorization required)", "upload assets + SHA256SUMS + SOURCE_MANIFEST.<artifact>.json, re-download each asset and compare sha256/size",
                                      "publish as prerelease", "clean clone + empty cache + anonymous (no token) download through prepare_benchmark.sh, tree hash verified (tools/artifacts/remote_fetch_check.sh)",
                                      "only then: locks primary.url = asset URL, status published; provenance/remote_fetch_verification*.yaml written; REMOTE_FETCH_VERIFIED"],
            "totals": {"assets": len(assets), "bytes": total, "human": hs.human(total)}, "assets": assets, "excluded": excluded,
            "extra_assets": ["SHA256SUMS (all planned artifacts)", "SOURCE_MANIFEST.<artifact>.json per artifact", "RELEASE_PLAN.json (this file, at the target commit)"]}
    with open(a.json, "w") as f:
        json.dump(plan, f, indent=1); f.write("\n")
    L = [f"# Release plan: `{a.tag}` (source-artifact prerelease, NOT published)", "",
         f"Generated {now} at target commit `{a.commit}`. Provider: GitHub Release assets of `{OWNER_REPO}`. Immutable releases on the repository: `{imm}` -- the project rule (never overwrite or delete an asset; consumers verify size, sha256 and source_tree_sha256) holds regardless. Nothing has been uploaded; every URL below is PLANNED and the locks stay `unpublished` until the asset exists and was re-downloaded and verified.",
         "", f"**{len(assets)} assets, {hs.human(total)} ({total} bytes)** + SHA256SUMS + one SOURCE_MANIFEST per artifact.", "",
         "| artifact | file | bytes | archive sha256 | source_tree_sha256 | app license | bundle/data review | verify | planned URL |", "|---|---|---|---|---|---|---|---|---|"]
    for x in assets:
        L.append(f"| {x['artifact']} | `{x['filename']}` | {x['size_bytes']} | `{x['sha256']}` | `{x['source_tree_sha256']}` | {x['application_license']['spdx']} | {x['bundle_dependency_and_data_license_review']['status']} ({x['bundle_dependency_and_data_license_review']['components_reviewed']} components, `{x['bundle_dependency_and_data_license_review']['record']}`) | {x['verification']['verdict']} ({x['verification']['mode']}) | `{x['planned_url']}` |")
    L += ["", "## Excluded", ""] + [f"- {e['artifact']}: {e['reason']}" for e in excluded]
    L += ["", "## SHA256SUMS (planned asset)", "", "```"] + [x["sha256sums_line"] for x in assets] + ["```", "", "## Scientific status and build prerequisites (per artifact)", ""]
    for x in assets:
        s = x["scientific_status"]
        L.append(f"- **{x['artifact']}** (upstream {x['upstream'].get('tag')} `{x['upstream']['commit'][:12]}`, patches {len(x['patches'])}): validated backends {s['validated_backends']}; smoke: {s['smoke']}; multi-node: {s['multi_node']}; 40/80: {s['gpu_40_80']}" + (f"; limits: {s['known_limits']}" if s.get("known_limits") else "") + f". Prerequisites: " + "; ".join(f"{p.get('name')} {p.get('version', '')}".strip() for p in x["build_prerequisites"]) + ".")
    L += ["", "## Procedure after the maintainer's upload authorization", ""] + [f"{i + 1}. {step}" for i, step in enumerate(plan["publication_procedure"])] + [""]
    with open(a.md, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"release plan: {len(assets)} assets, {hs.human(total)}; excluded {len(excluded)}; -> {os.path.relpath(a.md, R)}")
    for e in excluded:
        print(f"  excluded {e['artifact']}: {e['reason'][:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
