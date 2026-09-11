#!/usr/bin/env python3
"""github_release_publish -- publish the Level 3 source artifacts as GitHub Release assets, with strict checks.

    github_release_publish.py --plan level3/RELEASE_PLAN.json --mode preflight [--expect-plan-sha256 SHA]
    github_release_publish.py --plan ... --mode draft    [--staging DIR]      (needs HPCPERF_CONFIRM_UPLOAD=yes)
    github_release_publish.py --plan ... --mode publish  --release-id N       (needs HPCPERF_CONFIRM_UPLOAD=yes)
    github_release_publish.py --plan ... --mode verify   --release-id N [--redownload]

Every HTTP call is checked: the transport succeeding is not success. Each endpoint has an expected status code
(201 create release, 201 upload asset, 200 get/patch, 200 asset download); the body must parse as JSON of the
expected shape; and the identity of the returned object is verified (release id, tag, target_commitish, draft and
prerelease flags, asset name/size/state). 401, 403, 404, 422, any 5xx, a timeout, a non-JSON body, an identity
mismatch or an incomplete asset set fail the run with a non-zero exit, and the word "published" is never printed
for such a run. Assets are never overwritten or deleted: an existing asset of the same name aborts the run
(a corrected upload needs a new source version and tag). The working tree's SOURCE_MANIFEST files and the staged
archives must hash-match the reviewed plan before anything is uploaded.

Modes
  preflight  local only, no network: plan schema, per-asset staging file (size + sha256 == plan), working-tree
             SOURCE_MANIFEST hashes == plan, generated SHA256SUMS content, the plan's own sha256.
  draft      create a DRAFT release on the plan's target commit (refusing if the tag already exists), upload the
             archives + SHA256SUMS + SOURCE_MANIFEST.<artifact>.json + RELEASE_PLAN.json, re-download every asset
             and compare its sha256 (an authenticated upload check -- NOT the anonymous REMOTE_FETCH_VERIFIED),
             then GET the release and require the complete asset set.
  publish    flip an existing draft to a published prerelease, then GET again and re-verify identity and assets.
  verify     GET the release and compare names/sizes (with --redownload also sha256) against the plan.
The exit status is 0 only when every check of the mode passed. --api-base exists for the CPU-only mock tests.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

CONFIRM = "HPCPERF_CONFIRM_UPLOAD"
MUTATING = ("draft", "publish")


class Fail(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Api:
    def __init__(self, base, token, timeout, log):
        self.base, self.token, self.timeout, self.log = base.rstrip("/"), token, timeout, log
        self.calls = 0

    def _request(self, method, url, *, data=None, headers=None, expect, accept="application/vnd.github+json", parse_json=True):
        if url.startswith("/"):
            url = self.base + url
        h = {"Authorization": f"Bearer {self.token}", "Accept": accept, "X-GitHub-Api-Version": "2022-11-28",
             "User-Agent": "hpcperf-release/1.0"}
        h.update(headers or {})
        req = urllib.request.Request(url, data=data, headers=h, method=method)
        self.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                status, body = r.status, r.read()
        except urllib.error.HTTPError as e:
            body = e.read()[:400]
            raise Fail(f"{method} {url}: HTTP {e.code} (expected {expect}); body starts: {body[:200]!r}")
        except urllib.error.URLError as e:
            raise Fail(f"{method} {url}: transport/timeout error: {e.reason}")
        except TimeoutError:
            raise Fail(f"{method} {url}: timed out after {self.timeout}s")
        if status != expect:
            raise Fail(f"{method} {url}: HTTP {status}, expected {expect}")
        if not parse_json:
            return body
        try:
            obj = json.loads(body.decode())
        except (ValueError, UnicodeDecodeError) as e:
            raise Fail(f"{method} {url}: HTTP {status} but the body is not valid JSON ({e}); starts: {body[:120]!r}")
        if not isinstance(obj, (dict, list)):
            raise Fail(f"{method} {url}: JSON body is {type(obj).__name__}, expected object/array")
        return obj

    def get(self, url, expect=200, **kw):
        return self._request("GET", url, expect=expect, **kw)

    def post(self, url, obj=None, *, data=None, headers=None, expect=201, parse_json=True):
        body = json.dumps(obj).encode() if obj is not None else data
        hdr = {"Content-Type": "application/json"} if obj is not None else (headers or {})
        return self._request("POST", url, data=body, headers=hdr, expect=expect, parse_json=parse_json)

    def patch(self, url, obj, expect=200):
        return self._request("PATCH", url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"}, expect=expect)


def need(obj, key, where):
    if not isinstance(obj, dict) or key not in obj:
        raise Fail(f"{where}: response JSON lacks the key {key!r}")
    return obj[key]


def asset_paths(plan, staging, repo_root):
    """[(name, local path, size, sha256)] for archives + SHA256SUMS + SOURCE_MANIFESTs + the plan itself."""
    out = []
    for a in plan["assets"]:
        app = a["artifact"].split(".")[0]
        p = os.path.join(staging, "level3", app, a["source_version"], a["filename"])
        out.append((a["filename"], p, a["size_bytes"], a["sha256"]))
    return out


def build_sha256sums(plan):
    return "".join(f"{a['sha256']}  {a['filename']}\n" for a in plan["assets"])


def preflight(plan, plan_path, staging, repo_root, expect_plan_sha, log):
    problems = []
    if plan.get("schema") != "hpcperf-release-plan-1":
        problems.append(f"plan schema {plan.get('schema')!r}")
    for k in ("tag", "target_commit", "assets", "repository", "provider"):
        if not plan.get(k):
            problems.append(f"plan lacks {k}")
    if problems:
        raise Fail("; ".join(problems))
    plan_sha = sha256_file(plan_path)
    log(f"plan {os.path.basename(plan_path)}: sha256 {plan_sha}, tag {plan['tag']}, target commit {plan['target_commit'][:12]}, {len(plan['assets'])} assets")
    if expect_plan_sha and plan_sha != expect_plan_sha:
        raise Fail(f"plan sha256 {plan_sha} != expected {expect_plan_sha} (the reviewed plan is not the one on disk)")
    total = 0
    for name, path, size, sha in asset_paths(plan, staging, repo_root):
        if not os.path.isfile(path):
            raise Fail(f"asset {name}: not in the staging ({path})")
        got_size = os.path.getsize(path)
        if got_size != size:
            raise Fail(f"asset {name}: size {got_size} != plan {size}")
        got = sha256_file(path)
        if got != sha:
            raise Fail(f"asset {name}: sha256 {got} != plan {sha}")
        total += size
        log(f"  {name}: {size} B, sha256 ok")
    # working-tree SOURCE_MANIFESTs must match the plan (never upload an unreviewed manifest)
    for a in plan["assets"]:
        m = a["source_manifest"]
        p = os.path.join(repo_root, m["path"])
        if not os.path.isfile(p):
            raise Fail(f"{m['path']} missing from the working tree")
        got = sha256_file(p)
        if got != m["sha256"]:
            raise Fail(f"{m['path']}: working-tree sha256 {got} != plan {m['sha256']} -- the plan was generated from a different tree; regenerate and re-review the plan")
        log(f"  {m['planned_asset']} <- {m['path']}: sha256 ok")
    sums = build_sha256sums(plan)
    for a in plan["assets"]:
        if a["sha256sums_line"] + "\n" not in sums:
            raise Fail(f"SHA256SUMS line of {a['filename']} disagrees with the plan")
    log(f"SHA256SUMS: {len(plan['assets'])} lines derived from the plan")
    log(f"preflight OK: {len(plan['assets'])} archives + SHA256SUMS + {len(plan['assets'])} manifests + the plan, {total} bytes of archives")
    return {"plan_sha256": plan_sha, "archive_bytes": total, "asset_count": len(plan["assets"]) * 2 + 2}


def upload_all(api, plan, plan_path, staging, repo_root, release, tmpdir, log):
    upload_url = need(release, "upload_url", "release").split("{")[0]
    rid = need(release, "id", "release")
    # the object the API reports must be the object the upload URL addresses, and it must be retrievable
    if f"/releases/{rid}/assets" not in upload_url:
        raise Fail(f"create release: upload_url {upload_url!r} does not address release id {rid} (identity mismatch)")
    back = api.get(f"/repos/{plan['repository']}/releases/{rid}")
    if int(need(back, "id", "release read-back")) != int(rid):
        raise Fail(f"create release: read-back returned id {back.get('id')} for {rid}")
    existing = {a.get("name") for a in api.get(f"/repos/{plan['repository']}/releases/{rid}/assets") or []}
    if existing:
        raise Fail(f"release {rid} already carries assets ({sorted(existing)[:4]}...): assets are never overwritten or deleted; use a new tag/source version")
    os.makedirs(tmpdir, exist_ok=True)
    items = list(asset_paths(plan, staging, repo_root))
    sums_path = os.path.join(tmpdir, "SHA256SUMS")
    with open(sums_path, "w") as f:
        f.write(build_sha256sums(plan))
    items.append(("SHA256SUMS", sums_path, os.path.getsize(sums_path), sha256_file(sums_path)))
    for a in plan["assets"]:
        src = os.path.join(repo_root, a["source_manifest"]["path"])
        dst = os.path.join(tmpdir, a["source_manifest"]["planned_asset"])
        with open(src, "rb") as i, open(dst, "wb") as o:
            o.write(i.read())
        items.append((os.path.basename(dst), dst, os.path.getsize(dst), sha256_file(dst)))
    items.append(("RELEASE_PLAN.json", plan_path, os.path.getsize(plan_path), sha256_file(plan_path)))
    for name, path, size, sha in items:
        with open(path, "rb") as f:
            data = f.read()
        if len(data) != size:
            raise Fail(f"{name}: local file changed while uploading")
        resp = api.post(f"{upload_url}?name={name}", data=data, headers={"Content-Type": "application/octet-stream"}, expect=201)
        if need(resp, "name", f"upload {name}") != name:
            raise Fail(f"upload {name}: response names the asset {resp.get('name')!r}")
        if int(need(resp, "size", f"upload {name}")) != size:
            raise Fail(f"upload {name}: response size {resp.get('size')} != {size}")
        if resp.get("state") not in (None, "uploaded"):
            raise Fail(f"upload {name}: asset state {resp.get('state')!r}")
        aid = need(resp, "id", f"upload {name}")
        blob = api.get(f"/repos/{plan['repository']}/releases/assets/{aid}", accept="application/octet-stream", parse_json=False)
        got = hashlib.sha256(blob).hexdigest()
        if got != sha:
            raise Fail(f"{name}: re-downloaded sha256 {got} != {sha}")
        log(f"  uploaded {name} ({size} B) and re-downloaded identical (authenticated check, not an anonymous fetch)")
    return {n for n, _, _, _ in items}


def check_release_identity(api, plan, rid, want_draft, want_prerelease, expect_assets, log, where):
    rel = api.get(f"/repos/{plan['repository']}/releases/{rid}")
    if int(need(rel, "id", where)) != int(rid):
        raise Fail(f"{where}: release id {rel.get('id')} != {rid}")
    if need(rel, "tag_name", where) != plan["tag"]:
        raise Fail(f"{where}: tag {rel.get('tag_name')!r} != plan tag {plan['tag']!r}")
    tc = rel.get("target_commitish")
    if tc and tc not in (plan["target_commit"], plan["target_commit"][:7], plan["target_commit"][:12]):
        raise Fail(f"{where}: target_commitish {tc!r} != plan target commit {plan['target_commit']}")
    if bool(rel.get("draft")) != want_draft:
        raise Fail(f"{where}: draft={rel.get('draft')}, expected {want_draft}")
    if want_prerelease is not None and bool(rel.get("prerelease")) != want_prerelease:
        raise Fail(f"{where}: prerelease={rel.get('prerelease')}, expected {want_prerelease}")
    assets = api.get(f"/repos/{plan['repository']}/releases/{rid}/assets")
    names = {a.get("name") for a in assets}
    sizes = {a.get("name"): int(a.get("size", -1)) for a in assets}
    if expect_assets is not None:
        missing = expect_assets - names
        extra = names - expect_assets
        if missing or extra:
            raise Fail(f"{where}: asset set differs (missing {sorted(missing)[:4]}, unexpected {sorted(extra)[:4]})")
    for a in plan["assets"]:
        if a["filename"] in sizes and sizes[a["filename"]] != a["size_bytes"]:
            raise Fail(f"{where}: {a['filename']} size {sizes[a['filename']]} != plan {a['size_bytes']}")
    log(f"{where}: release {rid} tag {rel.get('tag_name')} draft={rel.get('draft')} prerelease={rel.get('prerelease')}, {len(names)} assets verified")
    return rel, assets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True); ap.add_argument("--mode", required=True, choices=("preflight", "draft", "publish", "verify"))
    ap.add_argument("--staging", default=os.environ.get("HPCPERF_ARTIFACT_STAGING")); ap.add_argument("--release-id")
    ap.add_argument("--api-base", default=os.environ.get("HPCPERF_GH_API", "https://api.github.com"))
    ap.add_argument("--expect-plan-sha256"); ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--redownload", action="store_true"); ap.add_argument("--json-out"); ap.add_argument("--tmpdir")
    a = ap.parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    log = lambda m: print(f"release: {m}", flush=True)  # noqa: E731
    result = {"mode": a.mode, "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    try:
        plan = json.load(open(a.plan))
        if a.mode in MUTATING and os.environ.get(CONFIRM) != "yes":
            raise Fail(f"{a.mode} refused: set {CONFIRM}=yes only with the maintainer's explicit authorization for this run")
        if not a.staging:
            raise Fail("--staging or HPCPERF_ARTIFACT_STAGING is required")
        pre = preflight(plan, a.plan, a.staging, repo_root, a.expect_plan_sha256, log)
        result.update(pre); result["tag"] = plan["tag"]; result["target_commit"] = plan["target_commit"]
        if a.mode == "preflight":
            result["verdict"] = "PASS"
            log("preflight complete: nothing was uploaded")
        else:
            token = os.environ.get("GITHUB_TOKEN")
            if not token:
                raise Fail("GITHUB_TOKEN is required for API modes")
            api = Api(a.api_base, token, a.timeout, log)
            repo = plan["repository"]
            if a.mode == "draft":
                # the target commit must exist, and the tag must not
                c = api.get(f"/repos/{repo}/commits/{plan['target_commit']}")
                if need(c, "sha", "target commit") != plan["target_commit"]:
                    raise Fail(f"target commit: API returned {c.get('sha')}")
                try:
                    api.get(f"/repos/{repo}/releases/tags/{plan['tag']}")
                    raise Fail(f"a release with tag {plan['tag']} already exists: tags and assets are never reused")
                except Fail as e:
                    if "HTTP 404" not in str(e):
                        raise
                rel = api.post(f"/repos/{repo}/releases", {
                    "tag_name": plan["tag"], "target_commitish": plan["target_commit"], "name": plan["tag"],
                    "draft": True, "prerelease": True,
                    "body": "Level 3 source artifacts (scheme 3), release candidate. See level3/RELEASE_PLAN.md at the target commit."})
                rid = need(rel, "id", "create release")
                if need(rel, "tag_name", "create release") != plan["tag"] or not rel.get("draft"):
                    raise Fail(f"create release: identity mismatch (tag {rel.get('tag_name')!r}, draft {rel.get('draft')})")
                log(f"draft release {rid} created on {plan['target_commit'][:12]}")
                names = upload_all(api, plan, a.plan, a.staging, repo_root, rel, a.tmpdir or os.path.join("/tmp", f"hpcperf-release-{os.getpid()}"), log)
                check_release_identity(api, plan, rid, True, True, names, log, "draft verification")
                result.update({"release_id": rid, "assets": sorted(names), "verdict": "DRAFT_UPLOADED_AND_VERIFIED"})
                log(f"draft release {rid} complete (still a DRAFT, not public). Next: --mode publish --release-id {rid} after authorization.")
            elif a.mode == "publish":
                if not a.release_id:
                    raise Fail("--release-id is required")
                rel, assets = check_release_identity(api, plan, a.release_id, True, True, None, log, "pre-publish check")
                names = {x.get("name") for x in assets}
                expect = {x["filename"] for x in plan["assets"]} | {"SHA256SUMS", "RELEASE_PLAN.json"} | {x["source_manifest"]["planned_asset"] for x in plan["assets"]}
                if names != expect:
                    raise Fail(f"pre-publish: asset set incomplete (missing {sorted(expect - names)[:4]}, unexpected {sorted(names - expect)[:4]})")
                out = api.patch(f"/repos/{repo}/releases/{a.release_id}", {"draft": False, "prerelease": True})
                if bool(out.get("draft")) is not False or bool(out.get("prerelease")) is not True:
                    raise Fail(f"publish: response says draft={out.get('draft')} prerelease={out.get('prerelease')}")
                check_release_identity(api, plan, a.release_id, False, True, expect, log, "post-publish verification")
                result.update({"release_id": a.release_id, "assets": sorted(expect), "verdict": "PUBLISHED_PRERELEASE"})
                log(f"release {a.release_id} published as a prerelease and re-verified. REMOTE_FETCH_VERIFIED still requires the anonymous check (tools/artifacts/remote_fetch_check.sh).")
            else:
                if not a.release_id:
                    raise Fail("--release-id is required")
                rel, assets = check_release_identity(api, plan, a.release_id, False, True, None, log, "verify")
                if a.redownload:
                    by_name = {x.get("name"): x for x in assets}
                    for nm, _p, size, sha in asset_paths(plan, a.staging, repo_root):
                        aid = need(by_name.get(nm, {}), "id", f"asset {nm}")
                        blob = api.get(f"/repos/{repo}/releases/assets/{aid}", accept="application/octet-stream", parse_json=False)
                        got = hashlib.sha256(blob).hexdigest()
                        if got != sha or len(blob) != size:
                            raise Fail(f"{nm}: re-download sha256/size mismatch")
                        log(f"  {nm}: re-downloaded sha256 ok")
                result.update({"release_id": a.release_id, "verdict": "VERIFIED"})
    except Fail as e:
        result.update({"verdict": "FAIL", "error": str(e)})
        print(f"release: FAIL -- {e}", file=sys.stderr)
        if a.json_out:
            json.dump(result, open(a.json_out, "w"), indent=1)
        return 1
    except (OSError, ValueError, KeyError) as e:
        result.update({"verdict": "FAIL", "error": f"{type(e).__name__}: {e}"})
        print(f"release: FAIL -- {type(e).__name__}: {e}", file=sys.stderr)
        if a.json_out:
            json.dump(result, open(a.json_out, "w"), indent=1)
        return 1
    if a.json_out:
        json.dump(result, open(a.json_out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
