#!/bin/bash
# CPU-only mock tests of the release publication path (no real API, no GPU, no network beyond localhost):
#   tools/artifacts/github_release_publish.py   preflight / draft / publish / verify against tools/artifacts/tests/mock_github.py
#   tools/artifacts/verify_published_artifact.py  anonymous plan-URL verification against a local file server
# Every HTTP fault mode must fail with a non-zero exit and must never print a published/verified verdict.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../../.." && pwd)"
ENGINE="$R/tools/artifacts/github_release_publish.py"; ANON="$R/tools/artifacts/verify_published_artifact.py"
pass=0; failn=0
ok()  { echo "ok   $*"; pass=$((pass+1)); }
bad() { echo "FAIL $*"; failn=$((failn+1)); }
T="$(mktemp -d "${TMPDIR:-/tmp}/relmock-XXXXXX")"; trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$T"' EXIT
export PYTHONDONTWRITEBYTECODE=1
filt() { grep -v 'lua:\|posix\|no file\|no field\|\[C\]:\|stack traceback\|in main chunk\|addto' || true; }
cap() { "$@" 2>&1 | filt; return "${PIPESTATUS[0]}"; }

# --- synthetic repo root, staging, artifact, lock and plan -----------------------------------------------
RT="$T/repo"; mkdir -p "$RT/tools/artifacts" "$RT/level3/mini/provenance"
cp -a "$R/tools"/*.py "$RT/tools/"; cp -a "$R/tools/artifacts"/*.py "$RT/tools/artifacts/"
STG="$T/staging/level3/mini/hpcperf-l3-v1"; mkdir -p "$STG"
python3 - "$RT" "$STG" <<'PY'
import json, os, sys, hashlib, tarfile, io, subprocess
RT, STG = sys.argv[1:3]
sys.path.insert(0, os.path.join(RT, "tools")); import hpcperf_source as hs, hpcperf_lock as hl
src = os.path.join(RT, ".stage", "src"); os.makedirs(src, exist_ok=True)
open(os.path.join(src, "main.c"), "w").write("int main(){return 0;}\n")
tar = os.path.join(RT, "a.tar"); hs.write_deterministic_tar(os.path.join(RT, ".stage"), tar)   # the tar must not live inside the staged tree
art = os.path.join(STG, "mini-hpcperf-l3-v1.tar.zst"); hs.zstd_compress(tar, art)
tree = hs.tree_hash(os.path.join(RT, ".stage")); sha = hs.sha256_file(art); size = os.path.getsize(art)
lock = hl.make_lock(name="mini", application="Mini", variant=None, source_version="hpcperf-l3-v1",
    upstream={"url": "https://example.invalid/m.git", "tag": "v1", "commit": "a" * 40},
    archive_info={"sha256": sha, "compressed_size": size, "uncompressed_size": 40}, tree_sha=tree, entries=1,
    layout=["src/"], patches=[], dependencies={}, components=[], equivalence=[], licenses=[{"path": "src", "project": "Mini", "license": "MIT"}],
    license_notes=[], redistribution_status="cleared", source_scope={}, scan_allow=[], freeze_tool_version="t", freeze_timestamp="now")
hs.dump_yaml(lock, os.path.join(RT, "level3/mini/provenance/source.lock.yaml"))
man = {"schema": "hpcperf-source-manifest-1", "name": "mini", "source_tree_sha256": tree, "manifest": []}
hs.write_json(man, os.path.join(RT, "level3/mini/provenance/SOURCE_MANIFEST.json"))
man_sha = hs.sha256_file(os.path.join(RT, "level3/mini/provenance/SOURCE_MANIFEST.json"))
plan = {"schema": "hpcperf-release-plan-1", "provider": "GitHub Release assets (mock)", "repository": "acme/mini",
        "tag": "level3-source-hpcperf-l3-v1-rc1", "target_commit": "b" * 40,
        "assets": [{"artifact": "mini", "application": "Mini", "variant": None, "filename": "mini-hpcperf-l3-v1.tar.zst",
                    "size_bytes": size, "sha256": sha, "source_tree_sha256": tree, "source_version": "hpcperf-l3-v1",
                    "upstream": {"tag": "v1", "commit": "a" * 40}, "planned_url": "https://example.invalid/mini-hpcperf-l3-v1.tar.zst",
                    "application_license": {"spdx": "MIT"}, "bundle_dependency_and_data_license_review": {"status": "cleared", "components_reviewed": 1, "record": "x", "notes": []},
                    "verification": {"verdict": "PASS", "checks": []}, "sha256sums_line": f"{sha}  mini-hpcperf-l3-v1.tar.zst",
                    "source_manifest": {"path": "level3/mini/provenance/SOURCE_MANIFEST.json", "sha256": man_sha, "planned_asset": "SOURCE_MANIFEST.mini.json"},
                    "patches": [], "scientific_status": {}, "build_prerequisites": [], "decision": "PLAN"}],
        "excluded": [], "totals": {"assets": 1, "bytes": size}}
json.dump(plan, open(os.path.join(RT, "level3/RELEASE_PLAN.json"), "w"), indent=1)
print("SIZE", size, "SHA", sha, "TREE", tree)
PY
PLAN="$RT/level3/RELEASE_PLAN.json"; export HPCPERF_ARTIFACT_STAGING="$T/staging"
ENG="$RT/tools/artifacts/github_release_publish.py"

# --- 1. preflight ---------------------------------------------------------------------------------------
out="$(cap python3 "$ENG" --plan "$PLAN" --mode preflight)"; rc=$?
[ $rc -eq 0 ] && echo "$out" | grep -q 'preflight OK' && ok "1a: preflight accepts a consistent plan/staging/working tree" || bad "1a: rc=$rc $out"
PSHA="$(sha256sum "$PLAN" | cut -d' ' -f1)"
out="$(cap python3 "$ENG" --plan "$PLAN" --mode preflight --expect-plan-sha256 "$PSHA")"; rc=$?
[ $rc -eq 0 ] && ok "1b: preflight accepts the reviewed plan sha256" || bad "1b: rc=$rc"
out="$(cap python3 "$ENG" --plan "$PLAN" --mode preflight --expect-plan-sha256 "$(printf 0%.0s $(seq 64))")"; rc=$?
[ $rc -ne 0 ] && echo "$out" | grep -q 'plan sha256' && ok "1c: a plan whose sha256 differs from the reviewed one is refused" || bad "1c: rc=$rc"
python3 - "$RT" <<'PY'
import json, sys, os
p = os.path.join(sys.argv[1], "level3/RELEASE_PLAN.json"); d = json.load(open(p)); d["assets"][0]["size_bytes"] += 1
json.dump(d, open(os.path.join(sys.argv[1], "plan_badsize.json"), "w"), indent=1)
d = json.load(open(p)); d["assets"][0]["sha256"] = "c" * 64; json.dump(d, open(os.path.join(sys.argv[1], "plan_badsha.json"), "w"), indent=1)
d = json.load(open(p)); d["assets"][0]["source_manifest"]["sha256"] = "d" * 64; json.dump(d, open(os.path.join(sys.argv[1], "plan_badman.json"), "w"), indent=1)
PY
for c in "plan_badsize.json:size" "plan_badsha.json:sha256" "plan_badman.json:working-tree sha256"; do IFS=: read -r f needle <<< "$c"
  out="$(cap python3 "$ENG" --plan "$RT/$f" --mode preflight)"; rc=$?
  [ $rc -ne 0 ] && echo "$out" | grep -q "$needle" && ok "1d: preflight refuses $f ($needle)" || bad "1d: $f rc=$rc"; done
out="$(cap env HPCPERF_CONFIRM_UPLOAD= python3 "$ENG" --plan "$PLAN" --mode draft)"; rc=$?
[ $rc -ne 0 ] && echo "$out" | grep -q 'HPCPERF_CONFIRM_UPLOAD' && ok "1e: draft refuses without explicit authorization" || bad "1e: rc=$rc"
out="$(cap env HPCPERF_CONFIRM_UPLOAD=yes GITHUB_TOKEN= python3 "$ENG" --plan "$PLAN" --mode draft)"; rc=$?
[ $rc -ne 0 ] && echo "$out" | grep -q 'GITHUB_TOKEN' && ok "1f: draft refuses without a token" || bad "1f: rc=$rc"

# --- 2. API fault modes: each must fail, none may claim a published/verified result ----------------------
start_mock() { python3 "$HERE/mock_github.py" --fault "$1" > "$T/mock.$1.log" 2>&1 &
    MPID=$!; for i in $(seq 1 50); do P="$(sed -n 's/^MOCK_READY //p' "$T/mock.$1.log" | head -1)"; [ -n "$P" ] && break; sleep 0.1; done; MOCK_PORT="$P"; }
stop_mock() { kill "$MPID" 2>/dev/null; wait "$MPID" 2>/dev/null; }
for f in unauthorized forbidden notfound-commit tag-exists unprocessable server-error-upload invalid-json wrong-id truncated-asset asset-exists incomplete-set; do
    start_mock "$f"
    out="$(cap env HPCPERF_CONFIRM_UPLOAD=yes GITHUB_TOKEN=mock python3 "$ENG" --plan "$PLAN" --mode draft --api-base "http://127.0.0.1:$MOCK_PORT" --timeout 5 --tmpdir "$T/up.$f")"; rc=$?
    if [ $rc -ne 0 ] && ! echo "$out" | grep -qiE 'published|PUBLISHED_PRERELEASE'; then ok "2: fault $f -> non-zero exit, no published claim"; else bad "2: fault $f rc=$rc $(echo "$out" | tail -2)"; fi
    stop_mock
done
start_mock timeout
out="$(cap env HPCPERF_CONFIRM_UPLOAD=yes GITHUB_TOKEN=mock python3 "$ENG" --plan "$PLAN" --mode draft --api-base "http://127.0.0.1:$MOCK_PORT" --timeout 1 --tmpdir "$T/up.timeout")"; rc=$?
[ $rc -ne 0 ] && echo "$out" | grep -qiE 'timed out|timeout|transport' && ok "2: a timeout fails the run" || bad "2: timeout rc=$rc"
stop_mock
# publish-not-applied: draft succeeds, publish must detect that the flags did not change
start_mock publish-not-applied
out="$(cap env HPCPERF_CONFIRM_UPLOAD=yes GITHUB_TOKEN=mock python3 "$ENG" --plan "$PLAN" --mode draft --api-base "http://127.0.0.1:$MOCK_PORT" --timeout 5 --tmpdir "$T/up.pna" --json-out "$T/draft.pna.json")"; rc=$?
RID="$(python3 -c 'import json;print(json.load(open("'"$T"'/draft.pna.json")).get("release_id",""))' 2>/dev/null)"
out="$(cap env HPCPERF_CONFIRM_UPLOAD=yes GITHUB_TOKEN=mock python3 "$ENG" --plan "$PLAN" --mode publish --release-id "$RID" --api-base "http://127.0.0.1:$MOCK_PORT" --timeout 5)"; rc2=$?
[ $rc -eq 0 ] && [ $rc2 -ne 0 ] && echo "$out" | grep -q 'draft=' && ok "2: a PATCH that does not actually publish is detected" || bad "2: publish-not-applied rc=$rc/$rc2"
stop_mock

# --- 3. full success path: draft -> publish -> verify ---------------------------------------------------
start_mock none
out="$(cap env HPCPERF_CONFIRM_UPLOAD=yes GITHUB_TOKEN=mock python3 "$ENG" --plan "$PLAN" --mode draft --api-base "http://127.0.0.1:$MOCK_PORT" --timeout 5 --tmpdir "$T/up.ok" --json-out "$T/draft.json")"; rc=$?
RID="$(python3 -c 'import json;print(json.load(open("'"$T"'/draft.json"))["release_id"])' 2>/dev/null)"
[ $rc -eq 0 ] && [ -n "$RID" ] && echo "$out" | grep -q 'uploaded .* and re-downloaded identical' && echo "$out" | grep -q 'draft verification' && ok "3a: draft uploads every asset, re-downloads each and verifies the complete set" || bad "3a: rc=$rc $(echo "$out" | tail -3)"
echo "$out" | grep -q 'still a DRAFT' && ! echo "$out" | grep -qi 'published as a prerelease' && ok "3b: the draft run never claims publication" || bad "3b"
out="$(cap env HPCPERF_CONFIRM_UPLOAD=yes GITHUB_TOKEN=mock python3 "$ENG" --plan "$PLAN" --mode publish --release-id "$RID" --api-base "http://127.0.0.1:$MOCK_PORT" --timeout 5)"; rc=$?
[ $rc -eq 0 ] && echo "$out" | grep -q 'published as a prerelease and re-verified' && echo "$out" | grep -q 'post-publish verification' && ok "3c: publish flips the draft and re-verifies identity and assets" || bad "3c: rc=$rc"
echo "$out" | grep -q 'REMOTE_FETCH_VERIFIED still requires the anonymous check' && ok "3d: publication does not claim REMOTE_FETCH_VERIFIED" || bad "3d"
out="$(cap env GITHUB_TOKEN=mock python3 "$ENG" --plan "$PLAN" --mode verify --release-id "$RID" --api-base "http://127.0.0.1:$MOCK_PORT" --timeout 5 --redownload)"; rc=$?
[ $rc -eq 0 ] && echo "$out" | grep -q 're-downloaded sha256 ok' && ok "3e: verify re-downloads and compares every archive" || bad "3e: rc=$rc"
out="$(cap env HPCPERF_CONFIRM_UPLOAD=yes GITHUB_TOKEN=mock python3 "$ENG" --plan "$PLAN" --mode draft --api-base "http://127.0.0.1:$MOCK_PORT" --timeout 5 --tmpdir "$T/up.again")"; rc=$?
[ $rc -ne 0 ] && echo "$out" | grep -q 'already exists' && ok "3f: a second draft for the same tag is refused (no asset reuse/overwrite)" || bad "3f: rc=$rc"
stop_mock

# --- 3g-3n. real Git tag checking and remote content verification before publication ---------------------
# a Git tag that exists (without a release) and points elsewhere must block the draft
start_mock git-tag-exists-elsewhere
out="$(cap env HPCPERF_CONFIRM_UPLOAD=yes GITHUB_TOKEN=mock python3 "$ENG" --plan "$PLAN" --mode draft --api-base "http://127.0.0.1:$MOCK_PORT" --timeout 5 --tmpdir "$T/up.gt1")"; rc=$?
[ $rc -ne 0 ] && echo "$out" | grep -q 'Git tag .* already exists' && echo "$out" | grep -q 'DIFFERENT commit' && ok "3g: a Git tag that exists without a release and points elsewhere blocks the draft" || bad "3g: rc=$rc $(echo "$out" | tail -1)"
stop_mock
start_mock annotated-tag-wrong-commit
out="$(cap env HPCPERF_CONFIRM_UPLOAD=yes GITHUB_TOKEN=mock python3 "$ENG" --plan "$PLAN" --mode draft --api-base "http://127.0.0.1:$MOCK_PORT" --timeout 5 --tmpdir "$T/up.gt2")"; rc=$?
[ $rc -ne 0 ] && echo "$out" | grep -q 'annotated' && ok "3h: an annotated tag is resolved through the tag object and its wrong commit blocks the draft" || bad "3h: rc=$rc $(echo "$out" | tail -1)"
stop_mock
# remote content faults: the draft uploads fine, but publication must be refused and no PATCH may reach the API
publish_guard() { # <fault> <needle> <label>
    start_mock "$1"
    local o1 r1 o2 r2 rid
    o1="$(cap env HPCPERF_CONFIRM_UPLOAD=yes GITHUB_TOKEN=mock python3 "$ENG" --plan "$PLAN" --mode draft --api-base "http://127.0.0.1:$MOCK_PORT" --timeout 5 --tmpdir "$T/up.$1" --json-out "$T/draft.$1.json")"; r1=$?
    rid="$(python3 -c "import json;print(json.load(open('$T/draft.$1.json')).get('release_id',''))" 2>/dev/null)"
    if [ -z "$rid" ]; then
        # the upload check already caught it (also acceptable): no publication attempt was possible
        [ $r1 -ne 0 ] && ok "$3 (caught at upload time, no draft id, no publish attempt)" || bad "$3: draft rc=$r1 but no id"
        stop_mock; return
    fi
    o2="$(cap env HPCPERF_CONFIRM_UPLOAD=yes GITHUB_TOKEN=mock python3 "$ENG" --plan "$PLAN" --mode publish --release-id "$rid" --api-base "http://127.0.0.1:$MOCK_PORT" --timeout 5)"; r2=$?
    local still_draft; still_draft="$(cap env GITHUB_TOKEN=mock python3 - "$MOCK_PORT" "$rid" <<'PY'
import json, sys, urllib.request
u=f"http://127.0.0.1:{sys.argv[1]}/repos/acme/mini/releases/{sys.argv[2]}"
r=urllib.request.Request(u, headers={"Authorization":"Bearer mock","Accept":"application/vnd.github+json"})
print(json.load(urllib.request.urlopen(r)).get("draft"))
PY
)"
    if [ $r2 -ne 0 ] && echo "$o2" | grep -q "$2" && [ "$still_draft" = "True" ]; then ok "$3"; else bad "$3: publish rc=$r2 still_draft=$still_draft $(echo "$o2" | tail -1)"; fi
    stop_mock; }
publish_guard asset-content-mismatch 'digest' "3i: an archive with the same size but different content blocks publication (release stays draft)"
publish_guard manifest-replaced 'digest' "3j: a replaced SOURCE_MANIFEST asset (same name) blocks publication (release stays draft)"
publish_guard plan-replaced 'digest' "3k: a replaced RELEASE_PLAN.json asset blocks publication (release stays draft)"
publish_guard asset-not-uploaded 'state' "3l: an asset that is not in state uploaded blocks publication (release stays draft)"
publish_guard asset-state-missing 'state' "3l2: an asset response WITHOUT a state field is rejected (never assumed uploaded)"
publish_guard asset-state-null 'state' "3l3: an asset with state=null is rejected"
# digest-less API: re-download is used, and --no-redownload must then refuse rather than trust the size
start_mock no-digest
out="$(cap env HPCPERF_CONFIRM_UPLOAD=yes GITHUB_TOKEN=mock python3 "$ENG" --plan "$PLAN" --mode draft --api-base "http://127.0.0.1:$MOCK_PORT" --timeout 5 --tmpdir "$T/up.nd" --json-out "$T/draft.nd.json")"; rc=$?
RID="$(python3 -c "import json;print(json.load(open('$T/draft.nd.json'))['release_id'])" 2>/dev/null)"
out="$(cap env HPCPERF_CONFIRM_UPLOAD=yes GITHUB_TOKEN=mock python3 "$ENG" --plan "$PLAN" --mode publish --release-id "$RID" --api-base "http://127.0.0.1:$MOCK_PORT" --timeout 5)"; rc2=$?
[ $rc -eq 0 ] && [ $rc2 -eq 0 ] && echo "$out" | grep -q 'by re-download' && ok "3m: without an API digest the content is verified by re-downloading every asset" || bad "3m: rc=$rc/$rc2 $(echo "$out" | tail -1)"
stop_mock
start_mock no-digest    # fresh state: the previous release was published, so a new draft is needed
out="$(cap env HPCPERF_CONFIRM_UPLOAD=yes GITHUB_TOKEN=mock python3 "$ENG" --plan "$PLAN" --mode draft --api-base "http://127.0.0.1:$MOCK_PORT" --timeout 5 --tmpdir "$T/up.nd2" --json-out "$T/draft.nd2.json")"; rc=$?
RID="$(python3 -c "import json;print(json.load(open('$T/draft.nd2.json'))['release_id'])" 2>/dev/null)"
out="$(cap env HPCPERF_CONFIRM_UPLOAD=yes GITHUB_TOKEN=mock python3 "$ENG" --plan "$PLAN" --mode publish --release-id "$RID" --api-base "http://127.0.0.1:$MOCK_PORT" --timeout 5 --no-redownload)"; rc=$?
[ $rc -ne 0 ] && echo "$out" | grep -q 'no API digest' && ok "3n: --no-redownload refuses when the API offers no digest (size alone is never accepted)" || bad "3n: rc=$rc $(echo "$out" | tail -1)"
stop_mock
# the good path must verify the real tag after publication
start_mock none
out="$(cap env HPCPERF_CONFIRM_UPLOAD=yes GITHUB_TOKEN=mock python3 "$ENG" --plan "$PLAN" --mode draft --api-base "http://127.0.0.1:$MOCK_PORT" --timeout 5 --tmpdir "$T/up.tag" --json-out "$T/draft.tag.json")"; rc=$?
RID="$(python3 -c "import json;print(json.load(open('$T/draft.tag.json'))['release_id'])" 2>/dev/null)"
out="$(cap env HPCPERF_CONFIRM_UPLOAD=yes GITHUB_TOKEN=mock python3 "$ENG" --plan "$PLAN" --mode publish --release-id "$RID" --api-base "http://127.0.0.1:$MOCK_PORT" --timeout 5)"; rc2=$?
[ $rc2 -eq 0 ] && echo "$out" | grep -q 'remote content verified' && echo "$out" | grep -q 'resolves to the plan target' && ok "3o: the good path verifies remote content before publishing and the created Git tag afterwards" || bad "3o: rc=$rc2 $(echo "$out" | tail -2)"
stop_mock

# --- 4. anonymous plan-URL verification (fix for the publish/lock circular dependency) ------------------
SERVE="$T/www"; mkdir -p "$SERVE"; cp "$STG/mini-hpcperf-l3-v1.tar.zst" "$SERVE/"
cat > "$T/fileserver.py" <<'PY'
import functools, http.server, socketserver, sys
class Q(socketserver.TCPServer): allow_reuse_address = True
h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=sys.argv[1])
srv = Q(("127.0.0.1", 0), h); print(f"HTTP_READY {srv.server_address[1]}", flush=True); srv.serve_forever()
PY
python3 "$T/fileserver.py" "$SERVE" > "$T/http.log" 2>&1 & echo $! > "$T/http.pid"
for i in $(seq 1 50); do HPORT="$(sed -n 's/^HTTP_READY //p' "$T/http.log" | head -1)"; [ -n "$HPORT" ] && break; sleep 0.1; done
[ -n "$HPORT" ] && ok "4-setup: local file server on port $HPORT" || bad "4-setup: file server did not start"
LOCK="$RT/level3/mini/provenance/source.lock.yaml"
out="$(cap env -u GITHUB_TOKEN -u GH_TOKEN HPCPERF_ALLOW_INSECURE_FETCH=1 python3 "$RT/tools/artifacts/verify_published_artifact.py" --lock "$LOCK" --url "http://127.0.0.1:$HPORT/mini-hpcperf-l3-v1.tar.zst" --record "$T/rec.yaml")"; rc=$?
[ $rc -eq 0 ] && echo "$out" | grep -q 'REMOTE ARTIFACT VERIFIED' && grep -q 'expected_identity_source' "$T/rec.yaml" && ok "4a: anonymous download verified against the lock's identity (URL from the plan, hashes from the lock)" || bad "4a: rc=$rc $(echo "$out" | tail -2)"

# 4a2: remote_fetch_check.sh hands out --cache/--scratch paths inside a fresh mktemp -d, i.e. directories that do
# not exist yet. They must be created, not assumed (this failed against the live release after a good download).
out="$(cap env -u GITHUB_TOKEN -u GH_TOKEN HPCPERF_ALLOW_INSECURE_FETCH=1 python3 "$RT/tools/artifacts/verify_published_artifact.py" --lock "$LOCK" --url "http://127.0.0.1:$HPORT/mini-hpcperf-l3-v1.tar.zst" --cache "$T/absent-cache/sub" --scratch "$T/absent-scratch/sub")"; rc=$?
[ $rc -eq 0 ] && echo "$out" | grep -q 'REMOTE ARTIFACT VERIFIED' && ok "4a2: a --cache/--scratch directory that does not exist yet is created, not assumed" || bad "4a2: rc=$rc $(echo "$out" | tail -2)"
out="$(cap env GITHUB_TOKEN=x HPCPERF_ALLOW_INSECURE_FETCH=1 python3 "$RT/tools/artifacts/verify_published_artifact.py" --lock "$LOCK" --url "http://127.0.0.1:$HPORT/mini-hpcperf-l3-v1.tar.zst")"; rc=$?
[ $rc -ne 0 ] && echo "$out" | grep -q 'not an anonymous fetch' && ok "4b: a token in the environment makes the anonymous check refuse" || bad "4b: rc=$rc"
out="$(cap env -u GITHUB_TOKEN -u GH_TOKEN python3 "$RT/tools/artifacts/verify_published_artifact.py" --lock "$LOCK" --url "http://127.0.0.1:$HPORT/mini-hpcperf-l3-v1.tar.zst")"; rc=$?
[ $rc -ne 0 ] && echo "$out" | grep -q 'must be https' && ok "4c: a non-https URL is refused outside tests" || bad "4c: rc=$rc"
printf 'x' >> "$SERVE/mini-hpcperf-l3-v1.tar.zst"
out="$(cap env -u GITHUB_TOKEN -u GH_TOKEN HPCPERF_ALLOW_INSECURE_FETCH=1 python3 "$RT/tools/artifacts/verify_published_artifact.py" --lock "$LOCK" --url "http://127.0.0.1:$HPORT/mini-hpcperf-l3-v1.tar.zst")"; rc=$?
[ $rc -ne 0 ] && echo "$out" | grep -qE 'size|sha256' && ! echo "$out" | grep -q 'REMOTE ARTIFACT VERIFIED' && ok "4d: a served file that differs from the lock's size/sha256 fails (a URL cannot redefine the artifact)" || bad "4d: rc=$rc $(echo "$out" | tail -1)"
cp "$STG/mini-hpcperf-l3-v1.tar.zst" "$SERVE/other-name.tar.zst"
out="$(cap env -u GITHUB_TOKEN -u GH_TOKEN HPCPERF_ALLOW_INSECURE_FETCH=1 python3 "$RT/tools/artifacts/verify_published_artifact.py" --lock "$LOCK" --url "http://127.0.0.1:$HPORT/other-name.tar.zst")"; rc=$?
[ $rc -ne 0 ] && echo "$out" | grep -q 'filename' && ok "4e: a URL that does not end with the artifact filename is refused" || bad "4e: rc=$rc"
kill "$(cat "$T/http.pid")" 2>/dev/null
echo
echo "test_release_publish_mock: $pass passed, $failn failed"
[ "$failn" -eq 0 ]
