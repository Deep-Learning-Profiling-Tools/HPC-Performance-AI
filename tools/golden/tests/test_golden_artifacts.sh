#!/usr/bin/env bash
# tools/golden/tests/test_golden_artifacts.sh -- CPU-only tests of tools/golden/golden_artifacts.py (the golden-store
# bundles: pack / fetch / verify / lock-check / plan) on small fixture repositories and stores. No GPU; the only
# network use is a loopback http server (download path). Exit 0 when every test passes.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$(cd "$HERE/.." && pwd)"
GA="$TOOLS/golden_artifacts.py"
FX="$HERE/golden_artifacts_fixture.py"
PY="${PYTHON:-python3}"
command -v zstd >/dev/null || { echo "zstd not on PATH (source hpcperf_env.sh)"; exit 2; }
TMP="$(mktemp -d "${TMPDIR:-/tmp}/golden-artifacts-tests.XXXXXX")"; export TMP
trap 'chmod -R u+w "$TMP" 2>/dev/null; rm -rf "$TMP"' EXIT
unset HPCPERF_GOLDEN_STORE HPCPERF_ARTIFACT_STAGING HPCPERF_GOLDEN_ALLOW_UNCOMMITTED_REFERENCE
export HPCPERF_ARTIFACT_CACHE="$TMP/cache"
pass=0; fail=0
ok()  { echo "ok   $*"; pass=$((pass+1)); }
bad() { echo "FAIL $*"; fail=$((fail+1)); }
eval "$("$PY" "$FX" mkfix "$TMP/fix")"
export SRC
G()  { "$PY" "$GA" --repo "$FR" "$@"; }
GS() { local s="$1"; shift; HPCPERF_GOLDEN_STORE="$s" "$PY" "$GA" --repo "$FR" "$@"; }
has() { [ -f "$1/sha256/${2:0:2}/$2.bin" ]; }
L="$TMP/lockA.yaml"; ST="$TMP/stA"

echo "=== 1: pack (committed references only, exactly their blobs)"
out="$(G pack --staging "$ST" --lock-out "$L" --store "$SRC" 2>&1)"; rc=$?
[ "$rc" = 0 ] && ok "1a: pack exit 0" || bad "1a: rc=$rc $out"
echo "$out" | /usr/bin/grep -q 'u.json is untracked: IGNORED' && echo "$out" | /usr/bin/grep -q 'in2.json differs from the committed tree' \
    && ok "1b: the untracked and the modified reference are reported as ignored" || bad "1b: $out"
"$PY" - "$L" "$ST" <<PY && ok "1c: 4 bundles = 4 groups, members exactly the committed blobs (no orphan/untracked/worktree blob), lock valid, SHA256SUMS agree" || bad "1c: lock content"
import sys, yaml; sys.path.insert(0, "$TOOLS"); import golden_artifacts as ga
text = open(sys.argv[1]).read(); lock = yaml.safe_load(text)
assert ga.validate_lock(lock, text) == [], ga.validate_lock(lock, text)
g = {b["group"]: sorted(m["sha256"] for m in b["members"]) for b in lock["bundles"]}
assert g == {"level1/alpha": sorted(["$A", "$B", "$C"]), "level1/beta": sorted(["$A", "$D"]),
             "level2/gamma": ["$E"], "level3/nyxlike": ["$F"]}, g
allm = {m["sha256"] for b in lock["bundles"] for m in b["members"]}
assert not allm & {"$O", "$U", "$M"}
assert sorted(lock["references"]) == ["level1/alpha/reference/in1.json", "level1/alpha/reference/in2.json",
    "level1/beta/reference/b1.json", "level2/gamma/reference/sub/g.json", "level3/nyxlike/reference/c.smoke.json"]
assert lock["references"]["level1/alpha/reference/in2.json"]["blobs"] == sorted(["$C", "$A"])
nyx = next(b for b in lock["bundles"] if b["group"] == "level3/nyxlike")
assert nyx["content_scan"]["node_private_members"] and nyx["redistribution"]["status"] == "review"
assert all(b["primary"] == {"url": None, "status": "unpublished"} for b in lock["bundles"])
sums = open(sys.argv[2] + "/SHA256SUMS").read().split("\n")
assert all(f"{b['sha256']}  {b['name']}" in sums for b in lock["bundles"]) and "/projects/" not in text
PY

echo
echo "=== 2: determinism and idempotency of pack"
out="$(G pack --staging "$TMP/stB" --lock-out "$TMP/lockB.yaml" --store "$SRC" --verify-determinism 2>&1)"; rc=$?
same=1; for f in "$ST"/*.tar.zst; do cmp -s "$f" "$TMP/stB/$(basename "$f")" || same=0; done
[ "$rc" = 0 ] && cmp -s "$L" "$TMP/lockB.yaml" && cmp -s "$ST/SHA256SUMS" "$TMP/stB/SHA256SUMS" && [ "$same" = 1 ] \
    && ok "2a: a second pack (with --verify-determinism) gives byte-identical bundles, SHA256SUMS and lock" || bad "2a: rc=$rc $out"
out="$(G pack --staging "$ST" --lock-out "$L" --store "$SRC" 2>&1)"; rc=$?
[ "$rc" = 0 ] && [ "$(echo "$out" | /usr/bin/grep -c '^  UNCHANGED')" = 4 ] && ok "2b: re-pack into the same staging: 4 UNCHANGED" || bad "2b: rc=$rc $out"
cp "$ST/$(cd "$ST" && ls golden-level1-alpha-*.tar.zst)" "$TMP/alpha.bak"; chmod u+w "$ST"/golden-level1-alpha-*.tar.zst
"$PY" "$FX" flip "$ST"/golden-level1-alpha-*.tar.zst
out="$(G pack --staging "$ST" --lock-out "$TMP/lockC.yaml" --store "$SRC" 2>&1)"; rc=$?
[ "$rc" = 5 ] && echo "$out" | /usr/bin/grep -q 'never overwritten' && [ ! -e "$TMP/lockC.yaml" ] \
    && ok "2c: a staged bundle with other bytes under the same name is never overwritten (exit 5, no lock)" || bad "2c: rc=$rc $out"
cp "$TMP/alpha.bak" "$ST"/golden-level1-alpha-*.tar.zst
b="$(cd "$ST" && ls golden-level1-alpha-*.tar.zst)"
listing="$(zstd -dc "$ST/$b" | TZ=UTC tar -tvf - 2>&1)"
echo "$listing" | /usr/bin/grep -q '^-rw-r--r-- 0/0 .* 2024-01-01 00:00 sha256/' && [ "$(echo "$listing" | wc -l)" = 3 ] \
    && [ "$(echo "$listing" | awk '{print $NF}')" = "$(echo "$listing" | awk '{print $NF}' | LC_ALL=C sort)" ] \
    && ok "2d: GNU tar reads the bundle: 3 regular members, 0/0, 0644, mtime 2024-01-01, sorted" || bad "2d: $listing"

echo
echo "=== 3: partition below the asset limit"
out="$(G pack --staging "$TMP/stP" --lock-out "$TMP/lockP.yaml" --store "$SRC" --asset-limit 105000 2>&1)"; rc=$?
"$PY" - "$TMP/lockP.yaml" <<PY && [ "$rc" = 0 ] && ok "3a: limit 105000: beta splits into p1of2/p2of2, every bundle below the limit" || bad "3a: rc=$rc $out"
import sys, yaml; sys.path.insert(0, "$TOOLS"); import golden_artifacts as ga
t = open(sys.argv[1]).read(); lock = yaml.safe_load(t); assert ga.validate_lock(lock, t) == []
beta = sorted(b["name"] for b in lock["bundles"] if b["group"] == "level1/beta")
assert len(beta) == 2 and "-p1of2-" in beta[0] and "-p2of2-" in beta[1], beta
assert all(b["size"] < 105000 and ga.compress_bound(b["tar_size"]) < 105000 for b in lock["bundles"])
PY
out="$(GS "$TMP/sP" fetch --from "$TMP/stP" --lock "$TMP/lockP.yaml" level1/beta 2>&1)"; rc=$?
[ "$rc" = 0 ] && has "$TMP/sP" "$A" && has "$TMP/sP" "$D" && ok "3b: fetch from a partitioned staging" || bad "3b: rc=$rc $out"
out="$(G pack --staging "$TMP/stQ" --lock-out "$TMP/lockQ.yaml" --store "$SRC" --asset-limit 50000 2>&1)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q 'cannot fit one release asset' && ok "3c: a blob larger than one asset is refused" || bad "3c: rc=$rc $out"

echo
echo "=== 4: fetch into an empty store, verify, idempotency"
S1="$TMP/s1"
out="$(GS "$S1" fetch --from "$ST" --lock "$L" 2>&1)"; rc=$?
allsame=1; for s in "$A" "$B" "$C" "$D" "$E" "$F"; do cmp -s "$S1/sha256/${s:0:2}/$s.bin" "$SRC/sha256/${s:0:2}/$s.bin" || allsame=0; done
[ "$rc" = 0 ] && [ "$allsame" = 1 ] && ! has "$S1" "$O" && ! has "$S1" "$U" && ! has "$S1" "$M" && [ ! -e "$S1/.incoming" ] \
    && ok "4a: all 6 committed blobs installed byte-identical; orphan/untracked/worktree blobs not; no .incoming left" || bad "4a: rc=$rc $out"
[ "$(stat -c %a "$S1/sha256/${A:0:2}/$A.bin")" = 444 ] && ok "4b: installed blobs are read-only (0444)" || bad "4b: mode"
out="$(GS "$S1" verify --lock "$L" 2>&1)"; rc=$?
[ "$rc" = 0 ] && echo "$out" | /usr/bin/grep -q 'verify: PASS -- ok 6, missing 0, corrupted 0, orphan 0, stray 0' && echo "$out" | /usr/bin/grep -q 'covers 6 of 6' \
    && ok "4c: verify PASS (6 ok, lock covers 6/6)" || bad "4c: rc=$rc $out"
out="$(GS "$S1" fetch --offline --lock "$L" 2>&1)"; rc=$?
[ "$rc" = 0 ] && echo "$out" | /usr/bin/grep -q 'already present 6, missing 0, bundles read 0' && ok "4d: second fetch (offline, no --from) is a no-op" || bad "4d: rc=$rc $out"

echo
echo "=== 5: partial selection"
S2="$TMP/s2"
out="$(GS "$S2" fetch --from "$ST" --lock "$L" level1/beta 2>&1)"; rc=$?
[ "$rc" = 0 ] && has "$S2" "$A" && has "$S2" "$D" && ! has "$S2" "$B" && ! has "$S2" "$C" && ! has "$S2" "$F" \
    && ok "5a: fetch level1/beta installs only beta's blobs (A, D)" || bad "5a: rc=$rc $out"
GS "$S2" verify --lock "$L" level1/beta >/dev/null 2>&1; r1=$?; out="$(GS "$S2" verify --lock "$L" 2>&1)"; r2=$?
[ "$r1" = 0 ] && [ "$r2" = 1 ] && [ "$(echo "$out" | /usr/bin/grep -c '^MISSING')" = 4 ] && ok "5b: verify level1/beta PASS, verify all FAIL with 4 MISSING" || bad "5b: $r1 $r2 $out"
out="$(GS "$S2" fetch --from "$ST" --lock "$L" level3/nyxlike/reference/c.smoke.json 2>&1)"; rc=$?
[ "$rc" = 0 ] && has "$S2" "$F" && ! has "$S2" "$B" && ok "5c: selection by one reference file" || bad "5c: rc=$rc $out"
out="$(GS "$S2" fetch --from "$ST" --lock "$L" level1/nosuch 2>&1)"; rc=$?
[ "$rc" = 2 ] && ok "5d: a selection that matches no committed reference is a usage error" || bad "5d: rc=$rc $out"

echo
echo "=== 6: tampered bundle / lock"
cp -r "$ST" "$TMP/stT"; chmod -R u+w "$TMP/stT"; "$PY" "$FX" flip "$TMP/stT"/golden-level1-alpha-*.tar.zst
out="$(GS "$TMP/s3" fetch --from "$TMP/stT" --lock "$L" level1/alpha 2>&1)"; rc=$?
[ "$rc" = 5 ] && ! has "$TMP/s3" "$A" && ! has "$TMP/s3" "$B" && echo "$out" | /usr/bin/grep -q 'sha256' \
    && ok "6a: one flipped byte: refused before extraction (exit 5), nothing installed" || bad "6a: rc=$rc $out"
cp -r "$ST" "$TMP/stT2"; chmod -R u+w "$TMP/stT2"; f="$(ls "$TMP/stT2"/golden-level1-alpha-*.tar.zst)"; head -c 100 "$f" > "$f.x"; mv "$f.x" "$f"
out="$(GS "$TMP/s3" fetch --from "$TMP/stT2" --lock "$L" level1/alpha 2>&1)"; rc=$?
[ "$rc" = 5 ] && echo "$out" | /usr/bin/grep -q 'size' && ok "6b: truncated bundle refused (size)" || bad "6b: rc=$rc $out"
"$PY" "$FX" lockedit "$L" "$TMP/l1.yaml" bundle-sha level1/alpha
out="$(GS "$TMP/s3" fetch --from "$ST" --lock "$TMP/l1.yaml" level1/alpha 2>&1)"; rc=$?
[ "$rc" = 5 ] && ! has "$TMP/s3" "$B" && ok "6c: lock with another bundle sha256: refused" || bad "6c: rc=$rc $out"
"$PY" "$FX" lockedit "$L" "$TMP/l2.yaml" member-sha level1/alpha
out="$(GS "$TMP/s3" fetch --from "$ST" --lock "$TMP/l2.yaml" 2>&1)"; rc=$?
[ "$rc" = 5 ] && echo "$out" | /usr/bin/grep -q 'members_digest' && ok "6d: lock member list edited: members_digest mismatch, refused" || bad "6d: rc=$rc $out"
"$PY" "$FX" lockedit "$L" "$TMP/l3.yaml" url-unpublished level1/alpha
out="$(GS "$TMP/s3" fetch --from "$ST" --lock "$TMP/l3.yaml" 2>&1)"; rc=$?
[ "$rc" = 5 ] && echo "$out" | /usr/bin/grep -q 'no fabricated future URL' && ok "6e: a URL while unpublished: lock invalid" || bad "6e: rc=$rc $out"
"$PY" "$FX" lockedit "$L" "$TMP/l4.yaml" private-path
out="$(G lock-check --lock "$TMP/l4.yaml" 2>&1)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q 'node-private location' && ok "6f: a node-private path in the lock: invalid" || bad "6f: rc=$rc $out"
for kind in wrong-content extra-member missing-member duplicate size-mismatch pax; do
    rm -rf "$TMP/stK" "$TMP/sK"
    "$PY" "$FX" craft "$kind" "$L" level1/alpha "$TMP/stK" "$TMP/lK.yaml"
    out="$(GS "$TMP/sK" fetch --from "$TMP/stK" --lock "$TMP/lK.yaml" level1/alpha 2>&1)"; rc=$?
    n="$(find "$TMP/sK" -type f 2>/dev/null | wc -l)"
    [ "$rc" = 5 ] && [ "$n" = 0 ] && ok "6g: forged bundle + matching bundle sha in the lock ($kind): refused, nothing installed" || bad "6g $kind: rc=$rc n=$n $out"
done

echo
echo "=== 7: unsafe archive members (path traversal, links, devices)"
for kind in traversal absolute dotdot-inside symlink hardlink directory device; do
    rm -rf "$TMP/stK" "$TMP/sK"
    "$PY" "$FX" craft "$kind" "$L" level1/alpha "$TMP/stK" "$TMP/lK.yaml"
    out="$(GS "$TMP/sK" fetch --from "$TMP/stK" --lock "$TMP/lK.yaml" level1/alpha 2>&1)"; rc=$?
    evil="$(find "$TMP" -name '*evil*' 2>/dev/null | wc -l)"
    n="$(find "$TMP/sK" \( -type f -o -type l \) 2>/dev/null | wc -l)"
    [ "$rc" = 5 ] && [ "$evil" = 0 ] && [ "$n" = 0 ] && echo "$out" | /usr/bin/grep -q 'not a store path\|not a regular file' \
        && ok "7: $kind member refused before anything is written" || bad "7 $kind: rc=$rc evil=$evil n=$n $out"
done

echo
echo "=== 8: missing bundle, unpublished, repair, orphans"
cp -r "$ST" "$TMP/stM"; chmod -R u+w "$TMP/stM"; rm "$TMP/stM"/golden-level1-beta-*.tar.zst
out="$(GS "$TMP/s4" fetch --from "$TMP/stM" --offline --lock "$L" 2>&1)"; rc=$?
[ "$rc" = 4 ] && has "$TMP/s4" "$A" && has "$TMP/s4" "$B" && has "$TMP/s4" "$F" && ! has "$TMP/s4" "$D" \
    && echo "$out" | /usr/bin/grep -q "MISSING $D" && ok "8a: a missing bundle: exit 4, the other bundles still installed, D reported" || bad "8a: rc=$rc $out"
out="$(GS "$TMP/s5" fetch --lock "$L" level1/beta 2>&1)"; rc=$?
[ "$rc" = 4 ] && echo "$out" | /usr/bin/grep -q 'unpublished (no URL in the lock)' && ok "8b: nothing local and unpublished: exit 4" || bad "8b: rc=$rc $out"
chmod u+w "$S1/sha256/${B:0:2}/$B.bin"; "$PY" "$FX" flip "$S1/sha256/${B:0:2}/$B.bin"
out="$(GS "$S1" verify --lock "$L" 2>&1)"; r1=$?
out2="$(GS "$S1" fetch --from "$ST" --lock "$L" 2>&1)"; r2=$?
GS "$S1" verify --lock "$L" >/dev/null 2>&1; r3=$?
[ "$r1" = 1 ] && echo "$out" | /usr/bin/grep -q "^CORRUPTED $B" && [ "$r2" = 0 ] && echo "$out2" | /usr/bin/grep -q 'repaired 1' && [ "$r3" = 0 ] \
    && ok "8c: a corrupted store blob: verify FAIL (CORRUPTED), fetch repairs it, verify PASS" || bad "8c: $r1 $r2 $r3 $out $out2"
mkdir -p "$S1/sha256/${O:0:2}" "$S1/sha256/ab"; cp "$SRC/sha256/${O:0:2}/$O.bin" "$S1/sha256/${O:0:2}/"; echo x > "$S1/sha256/ab/x.tmp"
out="$(GS "$S1" verify --lock "$L" 2>&1)"; rc=$?
[ "$rc" = 0 ] && echo "$out" | /usr/bin/grep -q "^ORPHAN sha256/${O:0:2}/$O.bin" && echo "$out" | /usr/bin/grep -q '^STRAY sha256/ab/x.tmp' \
    && ok "8d: orphan and stray files are listed (not failing, not deleted)" || bad "8d: rc=$rc $out"
chmod u+w "$S1/sha256/${C:0:2}" ; rm -f "$S1/sha256/${C:0:2}/$C.bin"
out="$(GS "$S1" verify --quick --lock "$L" 2>&1)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q "^MISSING $C" && ok "8e: a deleted blob: verify --quick FAIL (MISSING)" || bad "8e: rc=$rc $out"

echo
echo "=== 9: download path (loopback http, hpcperf_source.download_to_cache)"
out="$("$PY" "$FX" serve "$L" "$ST" "$TMP/s6" "$TMP/dl/cache" "$FR" 2>&1)"; rc=$?
nc="$(find "$TMP/dl/cache/sha256" -name '*.tar.zst' 2>/dev/null | wc -l)"
GS "$TMP/s6" verify --lock "$L" >/dev/null 2>&1; rv=$?
[ "$rc" = 0 ] && [ "$nc" = 4 ] && [ "$rv" = 0 ] && ok "9a: fetch through published (loopback) URLs: 4 bundles downloaded into the cache, verify PASS" || bad "9a: rc=$rc nc=$nc rv=$rv $out"
lp="$(echo "$out" | sed -n 's/^LOOPBACK_LOCK=//p')"
out="$(GS "$TMP/s7" fetch --lock "$lp" --cache-dir "$TMP/dl2" 2>&1)"; rc=$?
[ "$rc" = 5 ] && echo "$out" | /usr/bin/grep -q 'url must be https' && ok "9b: the CLI refuses a non-https URL in the lock" || bad "9b: rc=$rc $out"
rm -rf "$TMP/dl3"; out="$("$PY" "$FX" serve "$L" "$ST" "$TMP/s8" "$TMP/dl3/cache" "$FR" tamper 2>&1)"; rc=$?
n="$(find "$TMP/s8" -type f 2>/dev/null | wc -l)"; nc="$(find "$TMP/dl3/cache/sha256" -name '*.tar.zst' 2>/dev/null | wc -l)"
[ "$rc" = 5 ] && [ "$n" = 0 ] && [ "$nc" = 0 ] && ok "9c: a server delivering other bytes: refused, nothing cached or installed" || bad "9c: rc=$rc n=$n nc=$nc $out"

echo
echo "=== 10: stale lock, metadata drift"
R2="$TMP/repo2"; cp -a "$FR" "$R2"
( cd "$R2" && git checkout -q -- . && rm -rf level1/delta && "$PY" - <<PY
import json; p = "level1/alpha/reference/in1.json"; d = json.load(open(p)); d["outputs"][0]["basis"] = "edited"; open(p, "w").write(json.dumps(d, indent=2) + "\n")
PY
  git -c user.email=t@t -c user.name=t commit -qam drift )
out="$("$PY" "$GA" --repo "$R2" lock-check --lock "$L" 2>&1)"; rc=$?
[ "$rc" = 0 ] && echo "$out" | /usr/bin/grep -q 'metadata drift: level1/alpha/reference/in1.json' && ok "10a: a reference edited without a blob change: CURRENT with a metadata-drift note" || bad "10a: rc=$rc $out"
mkdir -p "$R2/level1/delta/reference"; cp "$FR/level1/delta/reference/u.json" "$R2/level1/delta/reference/"
( cd "$R2" && git add level1/delta && git -c user.email=t@t -c user.name=t commit -qm newref )
out="$("$PY" "$GA" --repo "$R2" lock-check --lock "$L" 2>&1)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q 'STALE: bundle golden-level1-delta' && ok "10b: a newly committed reference: lock-check STALE" || bad "10b: rc=$rc $out"
out="$(HPCPERF_GOLDEN_STORE="$TMP/s9" "$PY" "$GA" --repo "$R2" fetch --from "$ST" --lock "$L" level1/delta 2>&1)"; rc=$?
[ "$rc" = 4 ] && echo "$out" | /usr/bin/grep -q "NOT_IN_LOCK $U" && ok "10c: fetch of a blob the lock does not know: exit 4 NOT_IN_LOCK" || bad "10c: rc=$rc $out"
out="$(G lock-check --lock "$L" 2>&1)"; rc=$?
[ "$rc" = 0 ] && echo "$out" | /usr/bin/grep -q 'PASS (CURRENT)' && ok "10d: lock-check of the packed lock against its own commit: PASS" || bad "10d: rc=$rc $out"

echo
echo "=== 11: pack refusals"
R3="$TMP/repo3"; cp -a "$FR" "$R3"; ( cd "$R3" && git checkout -q -- . && rm -rf level1/delta )
cp -r "$SRC" "$TMP/src3"; chmod -R u+w "$TMP/src3"; rm "$TMP/src3/sha256/${D:0:2}/$D.bin"
out="$("$PY" "$GA" --repo "$R3" pack --staging "$TMP/stR1" --lock-out "$TMP/lR1.yaml" --store "$TMP/src3" 2>&1)"; rc=$?
[ "$rc" = 1 ] && [ ! -e "$TMP/lR1.yaml" ] && echo "$out" | /usr/bin/grep -q "blob $D missing" && ok "11a: a referenced blob missing from the store: no lock" || bad "11a: rc=$rc $out"
cp "$SRC/sha256/${D:0:2}/$D.bin" "$TMP/src3/sha256/${D:0:2}/"; chmod u+w "$TMP/src3/sha256/${D:0:2}/$D.bin"; "$PY" "$FX" flip "$TMP/src3/sha256/${D:0:2}/$D.bin"
out="$("$PY" "$GA" --repo "$R3" pack --staging "$TMP/stR2" --lock-out "$TMP/lR2.yaml" --store "$TMP/src3" 2>&1)"; rc=$?
[ "$rc" = 5 ] && [ ! -e "$TMP/lR2.yaml" ] && echo "$out" | /usr/bin/grep -q 'corrupted store' && ok "11b: a corrupted store blob: refused (exit 5)" || bad "11b: rc=$rc $out"
out="$("$PY" "$GA" --repo "$R3" pack --staging "$R3/staging" --lock-out "$TMP/lR3.yaml" --store "$SRC" 2>&1)"; rc=$?
[ "$rc" = 2 ] && ok "11c: a staging inside the repository is refused" || bad "11c: rc=$rc $out"
mkdir -p "$R3/level2/odd/reference"; echo '{"schema": "hpcperf-odd-1", "file": "$HPCPERF_GOLDEN_STORE/sha256/aa/x.bin"}' > "$R3/level2/odd/reference/o.json"
( cd "$R3" && git add level2/odd && git -c user.email=t@t -c user.name=t commit -qm odd )
out="$("$PY" "$GA" --repo "$R3" pack --staging "$TMP/stR4" --lock-out "$TMP/lR4.yaml" --store "$SRC" 2>&1)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q 'add an extractor' && ok "11d: an unknown schema that refers to the store is an error" || bad "11d: rc=$rc $out"
( cd "$R3" && git rm -q -r level2/odd && git -c user.email=t@t -c user.name=t commit -qm unodd )
"$PY" - "$R3" "$TMP/src3" <<'PY'
import sys, json, io, tarfile, hashlib, os
repo, store = sys.argv[1], sys.argv[2]
tok = "gh" + "p_" + "A1b2C3d4" * 4 + "Zz9Y"
bio = io.BytesIO()
with tarfile.open(fileobj=bio, mode="w", format=tarfile.GNU_FORMAT) as tf:
    d = f"export SOMETHING=1\n{tok}\n".encode(); ti = tarfile.TarInfo("plt/job_info"); ti.size = len(d); tf.addfile(ti, io.BytesIO(d))
data = bio.getvalue(); s = hashlib.sha256(data).hexdigest()
os.makedirs(f"{store}/sha256/{s[:2]}", exist_ok=True); open(f"{store}/sha256/{s[:2]}/{s}.bin", "wb").write(data)
p = f"{repo}/level3/nyxlike/reference/c.smoke.json"; doc = json.load(open(p)); doc["archive"] = {"sha256": s, "size": len(data)}
open(p, "w").write(json.dumps(doc) + "\n")
PY
cp "$SRC/sha256/${D:0:2}/$D.bin" "$TMP/src3/sha256/${D:0:2}/$D.bin"
( cd "$R3" && git -c user.email=t@t -c user.name=t commit -qam token )
out="$("$PY" "$GA" --repo "$R3" pack --staging "$TMP/stR5" --lock-out "$TMP/lR5.yaml" --store "$TMP/src3" 2>&1)"; rc=$?
[ "$rc" = 5 ] && [ ! -e "$TMP/lR5.yaml" ] && echo "$out" | /usr/bin/grep -q 'credential-looking content' && ! echo "$out" | /usr/bin/grep -q 'A1b2C3d4' \
    && ok "11e: credential-looking content in an archive blob: refused, rule names only (no content printed)" || bad "11e: rc=$rc $out"

echo
echo "=== 12: publication carry-over and plan (nothing uploaded)"
"$PY" "$FX" lockedit "$L" "$TMP/lpub.yaml" publish level2/gamma same
out="$(G pack --staging "$ST" --lock-out "$TMP/lpub.yaml" --store "$SRC" 2>&1)"; rc=$?
"$PY" - "$TMP/lpub.yaml" <<'PY' && [ "$rc" = 0 ] && ok "12a: re-pack keeps the URL/status of a published bundle whose bytes are unchanged" || bad "12a: rc=$rc $out"
import sys, yaml; lock = yaml.safe_load(open(sys.argv[1]))
b = next(b for b in lock["bundles"] if b["group"] == "level2/gamma"); assert b["primary"]["status"] == "published" and b["primary"]["url"].endswith(b["name"])
PY
"$PY" "$FX" lockedit "$L" "$TMP/lpub2.yaml" publish level2/gamma "$(printf '1%.0s' $(seq 64))"
out="$(G pack --staging "$ST" --lock-out "$TMP/lpub2.yaml" --store "$SRC" 2>&1)"; rc=$?
[ "$rc" = 5 ] && echo "$out" | /usr/bin/grep -q 'PUBLISHED' && ok "12b: a published name that would get other bytes: refused" || bad "12b: rc=$rc $out"
head="$(git -C "$FR" rev-parse HEAD)"
out="$(G plan --tag golden-test-rc1 --commit "$head" --staging "$ST" --lock "$L" --json "$TMP/plan.json" 2>&1)"; rc=$?
[ "$rc" = 1 ] && [ "$(echo "$out" | /usr/bin/grep -c '^REFUSE .*redistribution review')" = 4 ] && ok "12c: plan: every bundle REFUSE while its redistribution is under review" || bad "12c: rc=$rc $out"
"$PY" "$FX" lockedit "$L" "$TMP/lclr.yaml" clear-all
out="$(G plan --tag golden-test-rc1 --commit "$head" --staging "$ST" --lock "$TMP/lclr.yaml" --json "$TMP/plan.json" 2>&1)"; rc=$?
"$PY" - "$TMP/plan.json" <<'PY' && [ "$rc" = 0 ] && ok "12d: cleared: 4 PLAN rows, planned URLs = .../releases/download/<tag>/<bundle>, status PLANNED" || bad "12d: rc=$rc $out"
import sys, json; p = json.load(open(sys.argv[1]))
assert p["schema"] == "hpcperf-golden-release-plan-1" and len(p["assets"]) == 4 and not p["refused"]
assert all(a["planned_url"].endswith("/releases/download/golden-test-rc1/" + a["filename"]) and a["planned_url_status"].startswith("PLANNED") for a in p["assets"])
PY
chmod u+w "$ST"; mv "$ST"/golden-level2-gamma-*.tar.zst "$TMP/"
out="$(G plan --tag golden-test-rc1 --commit "$head" --staging "$ST" --lock "$TMP/lclr.yaml" 2>&1)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q '^REFUSE  golden-level2-gamma.*missing' && ok "12e: plan: a bundle missing from the staging is refused" || bad "12e: rc=$rc $out"

echo
echo "golden_artifacts tests: $pass passed, $fail failed"
[ "$fail" = 0 ]
