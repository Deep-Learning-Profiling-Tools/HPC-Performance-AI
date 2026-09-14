#!/usr/bin/env python3
"""publish_plan -- the preflight behind publish_artifacts.sh (see there). Prints the plan table and the
PLAN/REFUSE decision per artifact; exit 0 when every planned artifact passed, 1 when any was refused."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import hpcperf_source as hs  # noqa: E402
import hpcperf_lock as hl  # noqa: E402
import verify_artifact as va  # noqa: E402

R = os.path.abspath(os.path.join(HERE, "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", required=True); ap.add_argument("--app"); ap.add_argument("--release"); ap.add_argument("--full", action="store_true")
    ap.add_argument("--catalog", default=os.path.join(HERE, "artifact_catalog.yaml")); ap.add_argument("--json")
    a = ap.parse_args()
    cat = hs.load_yaml(a.catalog)
    release = a.release or cat["release"]["proposed_release_name"]
    apps = a.app.split(",") if a.app else list(cat["applications"])
    plan, refused = [], 0
    print(f"publish plan: suite {cat['suite']}, source version {cat['source_version']}, intended release/version: {release} (proposed; provider {cat['release'].get('provider') or 'undecided'})")
    print(f"{'application':<10} {'variant':<10} {'decision':<7} {'source ver':<14} {'size':>12} {'archive sha256':<18} {'tree sha256':<18} {'license':<8} {'publish':<12} artifact / reasons")
    for app in apps:
        ent = cat["applications"].get(app)
        if not ent:
            print(f"{app:<10} {'-':<10} REFUSE  not in the catalog"); refused += 1; continue
        d = os.path.join(R, "level3", app)
        by = hs.load_yaml(os.path.join(d, "benchmark.yaml")) if os.path.isfile(os.path.join(d, "benchmark.yaml")) else {}
        for v in ent.get("variants") or [None]:
            reasons = []
            row = {"application": app, "variant": v, "suite_status": ent.get("suite_status"), "intended_release": release}
            if ent.get("suite_status") == "retired":
                reasons.append(f"retired from the default suite ({ent.get('reason', '')[:60]}...)")
            if ent.get("suite_status") == "candidate" and ent.get("admission") != "admitted":
                reasons.append("candidate not admitted")
            acc = ent.get("release_acceptance") or "accepted"
            if str(acc).startswith(("on_hold", "under_review")):
                reasons.append(f"release_acceptance={acc}")
            lp = hl.lock_path(d, v)
            lock = hs.load_yaml(lp) if os.path.isfile(lp) else None
            if not lock or lock.get("schema") != hl.LOCK_SCHEMA:
                reasons.append("no schema-2 source lock")
                row["decision"] = "REFUSE"; row["reasons"] = reasons; plan.append(row); refused += 1
                print(f"{app:<10} {str(v or '-'):<10} REFUSE  {'; '.join(reasons)}"); continue
            art = lock["artifact"]
            row.update({"source_version": lock["benchmark"]["source_version"], "size": art["size"], "sha256": art["sha256"], "source_tree_sha256": art["source_tree_sha256"],
                        "intended_remote_filename": art["filename"], "license_status": lock.get("redistribution_status"), "publish_status": art["primary"].get("status")})
            if lock.get("redistribution_status") != "cleared":
                reasons.append(f"redistribution_status={lock.get('redistribution_status')}")
            ident = hl.identity_from_benchmark(by, v)
            if ident.get("archive_sha256") != art["sha256"] or ident.get("source_tree_sha256") != art["source_tree_sha256"] or ident.get("filename") != art["filename"]:
                reasons.append("benchmark.yaml identity != source.lock")
            entry = os.path.join(a.staging, "level3", app, lock["benchmark"]["source_version"])
            local = os.path.join(entry, art["filename"])
            row["artifact_local_path"] = local
            if not os.path.isfile(local):
                reasons.append("artifact missing from staging")
            else:
                res = va.verify(lp, artifact=local, staging_entry=entry, full=a.full, log=None)
                bad = [c for c in res["checks"] if c["status"] != "PASS"]
                for c in bad:
                    reasons.append(f"check {c['check']} ({c['name']}): {c['detail'][:80]}")
            if art["primary"].get("status") == "published":
                reasons.append("already published (immutable; a new source version needs a new artifact)")
            row["decision"] = "REFUSE" if reasons else "PLAN"; row["reasons"] = reasons
            if reasons:
                refused += 1
            plan.append(row)
            print(f"{app:<10} {str(v or '-'):<10} {row['decision']:<7} {row['source_version']:<14} {art['size']:>12} {art['sha256'][:16] + '…':<18} {art['source_tree_sha256'][:16] + '…':<18} {str(lock.get('redistribution_status')):<8} {str(art['primary'].get('status')):<12} {local if not reasons else '; '.join(reasons)}")
    n_plan = len([p for p in plan if p.get("decision") == "PLAN"])
    print(f"\n{n_plan} artifact(s) PLAN, {refused} REFUSE. Nothing is uploaded by this tool round (REMOTE_ARTIFACT_UNPUBLISHED).")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"release": release, "plan": plan}, f, indent=1)
    return 0 if refused == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
