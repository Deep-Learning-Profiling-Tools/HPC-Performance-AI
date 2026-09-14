#!/bin/bash
# Static tests of the Level 3 source-distribution tools, scheme 3 (no application, no GPU, no network beyond
# localhost): hpcperf_source.py (tree hash, scan, deterministic archive, restricted extraction, cache/download),
# hpcperf_lock.py, freeze_benchmark_source.py, compare_source_trees.py, prepare_benchmark.sh /
# hpcperf_materialize.py, tools/artifacts/{verify_artifact,publish_plan,generate_release_manifest}.py,
# check_workspace.py, create_agent_workspace.sh, validate_workspace.sh -- on a synthetic mini benchmark built
# from a throwaway git repo, with a throwaway artifact staging and cache.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$(cd "$HERE/.." && pwd)"
pass=0; failn=0
ok()  { echo "ok   $*"; pass=$((pass+1)); }
bad() { echo "FAIL $*"; failn=$((failn+1)); }
TMP="$(mktemp -d "${HPCPERF_FREEZE_SCRATCH:-/tmp}/srctools-XXXXXX")"; [ -n "${KEEP_TMP:-}" ] && echo "TMP=$TMP" || trap 'chmod -R u+w "$TMP" 2>/dev/null; rm -rf "$TMP"' EXIT
export HPCPERF_FREEZE_SCRATCH="$TMP/scratch"; mkdir -p "$HPCPERF_FREEZE_SCRATCH"
export HPCPERF_ARTIFACT_STAGING="$TMP/staging"; mkdir -p "$HPCPERF_ARTIFACT_STAGING"
unset HPCPERF_ARTIFACT_CACHE
export PYTHONDONTWRITEBYTECODE=1
filt() { /usr/bin/grep -v 'lua\|posix\|traceback\|no file\|no field\|\[C\]\|stack\|in main chunk\|addto' || true; }
# cap <cmd...>: combined output through the lmod-noise filter, exit status of the COMMAND (not of the filter)
cap() { "$@" 2>&1 | filt; return "${PIPESTATUS[0]}"; }

# --- a fake repository root with the harness pieces the tools expect ------------------------------------
RT="$TMP/repo"; mkdir -p "$RT/level3/tools" "$RT/level2/tools" "$RT/tools/artifacts" "$RT/_upstream/level3"
cp -a "$TOOLS"/*.py "$TOOLS"/*.sh "$RT/tools/"; cp -a "$TOOLS"/artifacts/*.py "$TOOLS"/artifacts/*.sh "$TOOLS"/artifacts/*.yaml "$RT/tools/artifacts/"
cp -a "$HERE/../../level3/tools/." "$RT/level3/tools/" 2>/dev/null || true
printf '#!/bin/bash\nexport HPC_PERFORMANCE_AI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n' > "$RT/hpcperf_env.sh"
UPREAL="$TMP/checkouts/miniapp"; mkdir -p "$UPREAL/src" "$UPREAL/doc" "$UPREAL/bench" "$UPREAL/lib/thirdparty"
UP="$RT/_upstream/level3/miniapp"; ln -s "$UPREAL" "$UP"
( cd "$UP" && git init -q && git config user.email t@t && git config user.name t
  echo 'int main(){return 0;}' > src/main.cpp; printf '#include <x>\nint k(){return 1;}\n' > src/kernel.cu
  echo 'manual' > doc/manual.txt; echo 'run 100' > bench/in.lj; echo 'ref' > bench/log.ref; echo 'tp' > lib/thirdparty/tp.c
  printf '#!/bin/sh\necho build\n' > src/gen.sh; chmod +x src/gen.sh; ln -s ../bench/in.lj src/in.link; echo 'GPL' > LICENSE
  git add -A && git commit -qm init )
SHA="$(git -C "$UP" rev-parse HEAD)"
B="$RT/level3/miniapp"; mkdir -p "$B/provenance" "$B/patches"
cat > "$B/patches/0001-fix.patch" <<'EOF'
--- a/src/main.cpp
+++ b/src/main.cpp
@@ -1 +1 @@
-int main(){return 0;}
+int main(){return 7;}
EOF
for s in build.sh run.sh validate.sh; do printf '#!/bin/bash\nHERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\nSRC="$HERE/src"\n[ -d "$SRC" ] || { echo "Benchmark source is not prepared. Run: tools/prepare_benchmark.sh level3 miniapp"; exit 3; }\necho %s from "$SRC"\n' "$s" > "$B/$s"; chmod +x "$B/$s"; done
# build.sh: emit a >1 MB "binary" whose content depends on the current source (so a source edit changes its hash)
cat > "$B/build.sh" <<'EOF'
#!/bin/bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; R="$(cd "$HERE/../.." && pwd)"; BK="$(echo "${1:-CUDA}" | tr '[:upper:]' '[:lower:]')"
SRC="$HERE/src"; [ -d "$SRC" ] || { echo "Benchmark source is not prepared. Run: tools/prepare_benchmark.sh level3 miniapp"; exit 3; }
D="$R/build/level3/miniapp/$BK"; mkdir -p "$D"
{ cat "$SRC/src/main.cpp"; head -c 1200000 /dev/zero | tr '\0' 'x'; } > "$D/mini.bin"; chmod +x "$D/mini.bin"
echo "built $D/mini.bin from $SRC"
EOF
chmod +x "$B/build.sh"
# validate.sh: run the binary and write this run's manifest (as the real run.sh scripts do)
cat > "$B/validate.sh" <<'EOF'
#!/bin/bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; R="$(cd "$HERE/../.." && pwd)"; BK="$(echo "${1:-CUDA}" | tr '[:upper:]' '[:lower:]')"
SRC="$HERE/src"; [ -d "$SRC" ] || { echo "Benchmark source is not prepared"; exit 3; }
BIN="$R/build/level3/miniapp/$BK/mini.bin"; [ -x "$BIN" ] || { echo "validate.sh: no binary for backend $BK"; exit 1; }
RD="$R/build/level3/miniapp/$BK/run/smoke.np1"; mkdir -p "$RD"
{ echo "run_id=$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"; echo "app=miniapp"; echo "backend=${1:-CUDA}"; echo "ranks=1"; echo "exit_code=0";
  echo "binary=$BIN"; echo "binary_sha256=$(sha256sum "$BIN" | cut -d' ' -f1)"; echo "utc=$(date -u +%FT%TZ)"; } >> "$RD/run_manifest.txt"
echo "hpcperf-launch: audit summary: 1 verified, 0 mismatch, 0 unverified (of 1 ranks)" > "$RD/stdout.log"
echo "miniapp validation (1 GPU, synthetic): PASS"
EOF
chmod +x "$B/validate.sh"
cat > "$B/provenance/freeze_spec.yaml" <<EOF
schema: hpcperf-freeze-spec-1
name: miniapp
level: 3
application: MiniApp
benchmark_source_version: hpcperf-l3-v1
redistribution_status: cleared
components:
  - dest: src
    category: application
    kind: git
    checkout: _upstream/level3/miniapp
    url: https://example.invalid/miniapp.git
    ref: v1
    commit: $SHA
    exclude: [{path: doc, reason: documentation}]
    patches: [{path: patches/0001-fix.patch, upstream_source: local, category: D}]
    license: {spdx: GPL-2.0-only, files: [LICENSE]}
  - dest: deps/tp/tp.tar.gz
    category: benchmark_specific
    kind: file
    source: .deps/level3/miniapp/downloads/tp.tar.gz
    sha256: __TPSHA__
    project: tp
    version: 1.0
    license: MIT
equivalence:
  - archive_path: src
    validated_tree: _upstream/level3/miniapp
    ignore: ['.git/*']
    excluded: ['doc/*']
licenses:
  - {path: src, project: MiniApp, license: GPL-2.0-only, url: https://example.invalid/miniapp.git, commit: $SHA}
source_scope: {application_owned: ['src/src/*'], bundled: ['src/lib/*'], benchmark_specific: ['deps/*'], test: [], exclude: []}
EOF
mkdir -p "$RT/.deps/level3/miniapp/downloads"; ( cd "$TMP" && mkdir -p tp && echo 'int t;' > tp/t.c && tar -czf "$RT/.deps/level3/miniapp/downloads/tp.tar.gz" tp )
TPSHA="$(sha256sum "$RT/.deps/level3/miniapp/downloads/tp.tar.gz" | cut -d' ' -f1)"; sed -i "s/__TPSHA__/$TPSHA/" "$B/provenance/freeze_spec.yaml"
cat > "$B/benchmark.yaml" <<'EOF'
name: miniapp
level: 3
application: MiniApp
supported_backends: [cuda]
validated_backends: [cuda]
distributed_model: none
default_scale_mode: smoke
build_entry: ./build.sh
run_entry: ./run.sh
validate_entry: ./validate.sh
environment_profile: none
install_root: .deps/level3/miniapp
dependency_installs: []
inputs: [src/bench/in.lj]
references: [src/bench/log.ref]
suite_status: retained
EOF
cd "$RT"
STG="$HPCPERF_ARTIFACT_STAGING/level3/miniapp/hpcperf-l3-v1"; ART="$STG/miniapp-hpcperf-l3-v1.tar.zst"; CACHE="$RT/.artifacts"

# --- 1. freeze: patched baseline, equivalence, deterministic artifact into staging, schema-2 lock ------------
out="$(cap python3 tools/freeze_benchmark_source.py level3/miniapp --verify-determinism)"; rc=$?
echo "$out" | /usr/bin/grep -q 'FREEZE OK' && [ -f "$ART" ] && ok "1a: freeze writes the artifact into the staging (patched, equivalent, deterministic)" || bad "1a: $(echo "$out" | tail -3)"
echo "$out" | /usr/bin/grep -q 'second archive IDENTICAL' && ok "1b: archive reproducible (two builds, same sha256)" || bad "1b: determinism line missing"
echo "$out" | /usr/bin/grep -q 'EQUIVALENT (identical 7, patch 1, generated 0, excluded 1' && ok "1c: equivalence: patched file and excluded doc classified, nothing unexpected" || bad "1c: $(echo "$out" | /usr/bin/grep equivalence)"
for f in source.lock.yaml upstream.lock patch_series.txt original_vs_baseline.diff SOURCE_MANIFEST.json LICENSES.md equivalence.json equivalence.md; do [ -f "$B/provenance/$f" ] || bad "1d: provenance/$f missing"; done; ok "1d: provenance files written"
[ -f "$STG/artifact.json" ] && [ -f "$STG/SHA256SUMS" ] && /usr/bin/grep -q 'miniapp-hpcperf-l3-v1.tar.zst' "$STG/SHA256SUMS" && ok "1e: staging carries artifact.json + SHA256SUMS" || bad "1e"
[ ! -e "$B/archives" ] && ! /usr/bin/grep -q 'source_bundle\|archives/' "$B/benchmark.yaml" && ok "1f: nothing archive-like inside the benchmark directory (no archives/, no source_bundle)" || bad "1f"
python3 - "$B/provenance/source.lock.yaml" <<'PY' && ok "1g: lock is schema hpcperf-source-lock-2, valid, primary unpublished with null url" || bad "1g: lock invalid"
import sys, os; sys.path.insert(0, "tools"); import hpcperf_lock as hl, hpcperf_source as hs
l = hs.load_yaml(sys.argv[1]); p = hl.validate_lock(l, open(sys.argv[1]).read())
assert not p, p; assert l["artifact"]["primary"] == {"url": None, "status": "unpublished"}; assert l["artifact"]["filename"] == "miniapp-hpcperf-l3-v1.tar.zst"
PY
TREE="$(sed -n 's/^source_tree_sha256: //p' "$B/benchmark.yaml")"; ASHA="$(sed -n 's/^  archive_sha256: //p' "$B/benchmark.yaml")"
[ -n "$TREE" ] && [ -n "$ASHA" ] && /usr/bin/grep -q '^source_artifact:' "$B/benchmark.yaml" && ok "1h: benchmark.yaml carries source_version/source_tree_sha256/source_artifact" || bad "1h"
out2="$(cap python3 tools/freeze_benchmark_source.py level3/miniapp)"; TREE2="$(sed -n 's/^source_tree_sha256: //p' "$B/benchmark.yaml")"; ASHA2="$(sed -n 's/^  archive_sha256: //p' "$B/benchmark.yaml")"
[ "$TREE" = "$TREE2" ] && [ "$ASHA" = "$ASHA2" ] && ok "1i: re-freeze reproduces source_tree_sha256 and archive sha256 (immutable staging entry accepted)" || bad "1i: $TREE/$ASHA vs $TREE2/$ASHA2"
out="$(cap python3 tools/freeze_benchmark_source.py level3/miniapp --staging "$RT/inside")"; rc=$?
echo "$out" | /usr/bin/grep -q 'outside the git worktree' && [ ! -d "$RT/inside" ] && ok "1j: a staging inside the worktree is refused" || bad "1j: rc=$rc"

# --- 2. scan / symlink / UNEXPECTED negatives -------------------------------------------------------------
( cd "$UP" && printf 'export GITHUB_TOKEN=ghp_%s\n' "$(printf 'A%.0s' $(seq 36))" > src/toolchain.env && git add -A && git commit -qm leak )
SHA2="$(git -C "$UP" rev-parse HEAD)"; sed -i "s/commit: $SHA\$/commit: $SHA2/; s/commit: $SHA}/commit: $SHA2}/" "$B/provenance/freeze_spec.yaml"
out="$(cap python3 tools/freeze_benchmark_source.py level3/miniapp)"; rc=$?
echo "$out" | /usr/bin/grep -q 'SCAN HIT' && ! echo "$out" | /usr/bin/grep -q 'ghp_A' && ok "2a: credential-looking file/content fails the freeze; no value printed" || bad "2a"
( cd "$UP" && git rm -q src/toolchain.env && ln -s /etc/passwd src/escape && git add -A && git commit -qm esc )
SHA3="$(git -C "$UP" rev-parse HEAD)"; sed -i "s/commit: $SHA2/commit: $SHA3/g" "$B/provenance/freeze_spec.yaml"
out="$(cap python3 tools/freeze_benchmark_source.py level3/miniapp)"
echo "$out" | /usr/bin/grep -q 'escaping' && ok "2b: symlink escaping the bundle fails the freeze" || bad "2b: $(echo "$out" | tail -2)"
( cd "$UP" && git rm -q src/escape && echo 'int main(){return 0;}' > src/main.cpp && git add -A && git commit -qm restore )
SHA4="$(git -C "$UP" rev-parse HEAD)"; sed -i "s/commit: $SHA3/commit: $SHA4/g" "$B/provenance/freeze_spec.yaml"
echo 'stray' > "$UP/src/stray.c"
out="$(cap python3 tools/freeze_benchmark_source.py level3/miniapp)"
echo "$out" | /usr/bin/grep -q 'UNEXPECTED' && ok "2c: unexplained difference vs the validated tree stops the migration" || bad "2c"
rm -f "$UP/src/stray.c"; python3 tools/freeze_benchmark_source.py level3/miniapp > /dev/null 2>&1 || bad "2d: clean re-freeze failed"
TREE="$(sed -n 's/^source_tree_sha256: //p' "$B/benchmark.yaml")"; ASHA="$(sed -n 's/^  archive_sha256: //p' "$B/benchmark.yaml")"
LOCK="$B/provenance/source.lock.yaml"

# --- 3. verify_artifact --------------------------------------------------------------------------------------
out="$(cap python3 tools/artifacts/verify_artifact.py --lock "$LOCK" --staging-entry "$STG" --full)"; rc=$?
[ $rc -eq 0 ] && echo "$out" | /usr/bin/grep -q 'verify_artifact: PASS' && ok "3a: staged artifact verifies fully (size, sha256, extraction, layout, tree hash, scan)" || bad "3a: $(echo "$out" | /usr/bin/grep FAIL)"
cp "$ART" "$TMP/corrupt.tar.zst"; printf 'x' >> "$TMP/corrupt.tar.zst"
out="$(cap python3 tools/artifacts/verify_artifact.py --lock "$LOCK" "$TMP/corrupt.tar.zst")"; rc=$?
[ $rc -ne 0 ] && echo "$out" | /usr/bin/grep -q 'FAIL 2:' && ok "3b: corrupted artifact fails the size/sha256 check" || bad "3b"
python3 - "$LOCK" "$TMP/badtree.lock.yaml" <<'PY'
import sys, yaml; l = yaml.safe_load(open(sys.argv[1])); l["artifact"]["source_tree_sha256"] = "0"*64; l["materialized_tree"]["sha256"] = "0"*64; yaml.safe_dump(l, open(sys.argv[2], "w"), sort_keys=False)
PY
out="$(cap python3 tools/artifacts/verify_artifact.py --lock "$TMP/badtree.lock.yaml" "$ART" --full)"; rc=$?
[ $rc -ne 0 ] && echo "$out" | /usr/bin/grep -q 'FAIL 6:' && ok "3c: tree-hash mismatch between artifact and lock detected" || bad "3c: $(echo "$out" | /usr/bin/grep FAIL)"

# --- 4. prepare: cache / offline / explicit artifact / dirty / force / bad hash / unavailable ------------------
out="$(cap bash tools/prepare_benchmark.sh level3 miniapp --offline)"; rc=$?
[ $rc -eq 4 ] && echo "$out" | /usr/bin/grep -q 'not in the cache' && [ ! -d "$B/src" ] && ok "4a: offline cache miss fails (exit 4), nothing materialized" || bad "4a: rc=$rc $(echo "$out" | tail -1)"
out="$(cap bash tools/prepare_benchmark.sh level3 miniapp)"; rc=$?
[ $rc -eq 4 ] && echo "$out" | /usr/bin/grep -q 'REMOTE_ARTIFACT_UNPUBLISHED' && ok "4b: unpublished artifact + empty cache: refused with the unpublished status (no made-up URL fetched)" || bad "4b: rc=$rc"
out="$(cap bash tools/prepare_benchmark.sh level3 miniapp --artifact /nonexistent.tar.zst)"; rc=$?
[ $rc -eq 5 ] && echo "$out" | /usr/bin/grep -q 'artifact missing' && ok "4c: missing --artifact file refused" || bad "4c: rc=$rc"
out="$(cap bash tools/prepare_benchmark.sh level3 miniapp --artifact "$ART")"; rc=$?
[ $rc -eq 0 ] && [ -f "$B/src/src/main.cpp" ] && [ -f "$B/deps/tp/tp.tar.gz" ] && /usr/bin/grep -q 'return 7' "$B/src/src/main.cpp" && ok "4d: --artifact materializes src/ and deps/ (patched baseline)" || bad "4d: rc=$rc: $(echo "$out" | tail -3)"
[ -f "$CACHE/sha256/$ASHA.tar.zst" ] && echo "$out" | /usr/bin/grep -q 'STATUS LOCAL_ARTIFACT_VERIFIED' && echo "$out" | /usr/bin/grep -q 'STATUS MATERIALIZED' && ok "4e: verified artifact placed in the content-addressed cache; statuses LOCAL_ARTIFACT_VERIFIED + MATERIALIZED printed" || bad "4e"
/usr/bin/grep -q 'artifact_origin: explicit' "$B/.hpcperf-materialized.yaml" && [ ! -d "$B/src/doc" ] && ok "4f: marker records the origin; excluded doc/ absent" || bad "4f"
out="$(cap bash tools/prepare_benchmark.sh level3 miniapp)"; rc=$?
[ $rc -eq 0 ] && echo "$out" | /usr/bin/grep -q 'STATUS READY' && ok "4g: prepare is idempotent on an identical tree (READY)" || bad "4g: rc=$rc"
out="$(cap bash tools/prepare_benchmark.sh level3 miniapp --status)"; rc=$?
[ $rc -eq 0 ] && echo "$out" | /usr/bin/grep -q 'PREPARE STATUS miniapp READY' && ok "4h: --status reports READY" || bad "4h"
echo '// agent change' >> "$B/src/src/main.cpp"
out="$(cap bash tools/prepare_benchmark.sh level3 miniapp)"; rc=$?
[ $rc -eq 3 ] && echo "$out" | /usr/bin/grep -q 'STATUS DIRTY' && /usr/bin/grep -q 'agent change' "$B/src/src/main.cpp" && ok "4i: modified src refused (DIRTY, exit 3), content untouched" || bad "4i: rc=$rc"
out="$(cap bash tools/prepare_benchmark.sh level3 miniapp --status)"; rc=$?
[ $rc -eq 3 ] && echo "$out" | /usr/bin/grep -q 'DIRTY' && ok "4j: --status reports DIRTY (exit 3)" || bad "4j"
out="$(cap bash tools/prepare_benchmark.sh level3 miniapp --force-rematerialize --offline)"; rc=$?
[ $rc -eq 0 ] && ! /usr/bin/grep -q 'agent change' "$B/src/src/main.cpp" && echo "$out" | /usr/bin/grep -q 'cache hit' && ok "4k: --force-rematerialize restores the baseline from the cache, offline" || bad "4k: rc=$rc"
chmod u+w "$CACHE/sha256/$ASHA.tar.zst"; printf 'x' >> "$CACHE/sha256/$ASHA.tar.zst"; rm -rf "$B/src" "$B/deps" "$B/.hpcperf-materialized.yaml"
cmp -s "$ART" "$CACHE/sha256/$ASHA.tar.zst" && bad "4l0: cache entry shares content/inode with the staging artifact" || ok "4l0: corrupting the cache entry leaves the staging artifact intact (real copy, no hard link)"
out="$(cap bash tools/prepare_benchmark.sh level3 miniapp --offline)"; rc=$?
[ $rc -eq 5 ] && echo "$out" | /usr/bin/grep -q 'cache entry corrupt' && [ ! -d "$B/src" ] && ok "4l: corrupted cache entry refused (exit 5), nothing materialized" || bad "4l: rc=$rc"
rm -f "$CACHE/sha256/$ASHA.tar.zst"; cp "$ART" "$CACHE/sha256/$ASHA.tar.zst"
python3 - "$LOCK" "$B/benchmark.yaml" <<'PY'
import sys, yaml
for p in sys.argv[1:]:
    d = yaml.safe_load(open(p)); 
    if "artifact" in d: d["artifact"]["source_tree_sha256"] = "1"*64; d["materialized_tree"]["sha256"] = "1"*64
    else: d["source_tree_sha256"] = "1"*64; d["source_artifact"]["source_tree_sha256"] = "1"*64
    yaml.safe_dump(d, open(p, "w"), sort_keys=False)
PY
out="$(cap bash tools/prepare_benchmark.sh level3 miniapp --offline)"; rc=$?
[ $rc -eq 5 ] && echo "$out" | /usr/bin/grep -q 'extracted tree sha256' && [ ! -d "$B/src" ] && [ ! -d "$RT/level3/.materialize-staging" ] && ok "4m: tree-hash mismatch after extraction refused (exit 5), staging cleaned, nothing placed" || bad "4m: rc=$rc"
python3 - "$LOCK" "$B/benchmark.yaml" "$TREE" <<'PY'
import sys, yaml
t = sys.argv[3]
for p in sys.argv[1:3]:
    d = yaml.safe_load(open(p))
    if "artifact" in d: d["artifact"]["source_tree_sha256"] = t; d["materialized_tree"]["sha256"] = t
    else: d["source_tree_sha256"] = t; d["source_artifact"]["source_tree_sha256"] = t
    yaml.safe_dump(d, open(p, "w"), sort_keys=False)
PY
bash tools/prepare_benchmark.sh level3 miniapp --offline > /dev/null 2>&1 && ok "4n: re-prepare from the cache after restoring the lock" || bad "4n"

# --- 5. download / partial / extractor unit tests (localhost http server) ------------------------------------
python3 - "$ART" "$ASHA" "$TMP" <<'PY' && ok "5a: download: wrong sha -> no cache entry, partial removed; truncated -> size mismatch; correct -> cached" || bad "5a: download unit test"
import sys, os, threading, http.server, socketserver, functools, shutil
sys.path.insert(0, "tools"); import hpcperf_source as hs
art, sha, tmp = sys.argv[1:4]
serve = os.path.join(tmp, "www"); os.makedirs(serve, exist_ok=True); shutil.copy(art, os.path.join(serve, "a.tar.zst"))
open(os.path.join(serve, "short.tar.zst"), "wb").write(open(art, "rb").read()[:os.path.getsize(art) // 2])
H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=serve)
class Q(socketserver.TCPServer): allow_reuse_address = True
srv = Q(("127.0.0.1", 0), H); port = srv.server_address[1]; threading.Thread(target=srv.serve_forever, daemon=True).start()
cache = os.path.join(tmp, "cache2"); size = os.path.getsize(art)
try:
    hs.download_to_cache(f"http://127.0.0.1:{port}/a.tar.zst", cache, "0"*64, size); raise SystemExit("wrong sha accepted")
except hs.SourceError: pass
assert not os.path.exists(hs.cache_path(cache, "0"*64)) and not os.listdir(os.path.join(cache, ".partial"))
try:
    hs.download_to_cache(f"http://127.0.0.1:{port}/short.tar.zst", cache, sha, size); raise SystemExit("short file accepted")
except hs.SourceError: pass
assert not os.path.exists(hs.cache_path(cache, sha))
p = hs.download_to_cache(f"http://127.0.0.1:{port}/a.tar.zst", cache, sha, size)
assert os.path.isfile(p) and hs.sha256_file(p) == sha and not os.listdir(os.path.join(cache, ".partial"))
try:
    hs.download_to_cache(f"http://127.0.0.1:{port}/a.tar.zst", cache, sha, 500); raise SystemExit("oversize accepted")
except hs.SourceError: pass
srv.shutdown()
PY
python3 - "$TMP" <<'PY' && ok "5b: restricted extractor rejects '..' / absolute / hard-link / device entries and unknown top-levels; escaping symlink detected after extraction" || bad "5b: extractor unit test"
import sys, os, tarfile, io
sys.path.insert(0, "tools"); import hpcperf_source as hs
tmp = sys.argv[1]
def mk(name, entries):
    p = os.path.join(tmp, name)
    with tarfile.open(p, "w") as tf:
        for n, kind, extra in entries:
            ti = tarfile.TarInfo(n)
            if kind == "dir": ti.type = tarfile.DIRTYPE
            elif kind == "sym": ti.type = tarfile.SYMTYPE; ti.linkname = extra
            elif kind == "lnk": ti.type = tarfile.LNKTYPE; ti.linkname = extra
            elif kind == "chr": ti.type = tarfile.CHRTYPE
            else: ti.size = len(extra)
            tf.addfile(ti, io.BytesIO(extra) if kind == "file" else None)
    return p
bad = [mk("t1.tar", [("src/", "dir", None), ("src/../evil", "file", b"x")]),
       mk("t2.tar", [("/etc/x", "file", b"x")]),
       mk("t3.tar", [("src/", "dir", None), ("src/a", "file", b"x"), ("src/b", "lnk", "src/a")]),
       mk("t4.tar", [("src/", "dir", None), ("src/dev", "chr", None)]),
       mk("t5.tar", [("other/", "dir", None), ("other/a", "file", b"x")])]
for t in bad:
    d = os.path.join(tmp, "x" + os.path.basename(t)); os.makedirs(d, exist_ok=True)
    try:
        hs.safe_extract_tar(t, d); raise SystemExit(f"{t} accepted")
    except hs.SourceError: pass
t6 = mk("t6.tar", [("src/", "dir", None), ("src/esc", "sym", "/etc/passwd"), ("src/up", "sym", "../../x")])
d = os.path.join(tmp, "x6"); os.makedirs(d, exist_ok=True); hs.safe_extract_tar(t6, d)
esc = hs.escaping_symlinks(d, hs.manifest(d)); assert {e["path"] for e in esc} == {"src/esc", "src/up"}, esc
PY
python3 - <<'PY' && ok "5c: lock validation: unpublished+url, http url, floating url, /tmp path, missing commit, wrong filename all rejected" || bad "5c: lock validation unit test"
import sys, copy; sys.path.insert(0, "tools"); import hpcperf_lock as hl, hpcperf_source as hs
base = hl.make_lock(name="x", application="X", variant=None, source_version="hpcperf-l3-v1", upstream={"url": "https://e/x.git", "tag": "v1", "commit": "a"*40},
    archive_info={"sha256": "b"*64, "compressed_size": 10, "uncompressed_size": 20}, tree_sha="c"*64, entries=1, layout=["src/"], patches=[], dependencies={}, components=[],
    equivalence=[], licenses=[], license_notes=[], redistribution_status="cleared", source_scope={}, scan_allow=[], freeze_tool_version="t", freeze_timestamp="now")
assert not hl.validate_lock(base, "")
def bad(mut, needle):
    l = copy.deepcopy(base); mut(l); p = hl.validate_lock(l, ""); assert p and any(needle in x for x in p), (needle, p)
bad(lambda l: l["artifact"]["primary"].update({"url": "https://h/x-hpcperf-l3-v1.tar.zst"}), "must be null while unpublished")
bad(lambda l: l["artifact"]["primary"].update({"url": "http://h/x-hpcperf-l3-v1.tar.zst", "status": "published"}), "must be https")
bad(lambda l: l["artifact"]["primary"].update({"url": "https://h/latest/x-hpcperf-l3-v1.tar.zst", "status": "published"}), "floating")
bad(lambda l: l["artifact"]["primary"].update({"url": "https://h/v1/other.tar.zst", "status": "published"}), "end with the artifact filename")
bad(lambda l: l["upstream"].update({"commit": "main"}), "40-hex")
bad(lambda l: l["artifact"].update({"filename": "x.tar.zst"}), "convention")
bad(lambda l: l.update({"redistribution_status": "maybe"}), "redistribution_status")
assert any("node-private" in x for x in hl.validate_lock(base, "validated_tree: /tmp/scratch/x\n"))
assert any("node-private" in x for x in hl.validate_lock(base, "path: /home/user/x\n"))
ok_url = copy.deepcopy(base); ok_url["artifact"]["primary"] = {"url": "https://h/rel/v1/x-hpcperf-l3-v1.tar.zst", "status": "published"}; assert not hl.validate_lock(ok_url, "")
PY

# --- 6. secret scan at prepare time (an artifact that carries a token) ----------------------------------------
python3 - "$RT" "$TMP" <<'PY' && out="$(cap bash tools/prepare_benchmark.sh level3 leaky --artifact "$TMP/leaky.tar.zst")"; rc=$?
import sys, os, shutil, yaml; sys.path.insert(0, "tools"); import hpcperf_source as hs, hpcperf_lock as hl
RT, tmp = sys.argv[1:3]; st = os.path.join(tmp, "leaky-stage"); os.makedirs(os.path.join(st, "src"), exist_ok=True)
open(os.path.join(st, "src", "main.c"), "w").write("int main(){}\n"); open(os.path.join(st, "src", "toolchain.env"), "w").write("export GITHUB_TOKEN=ghp_" + "B"*36 + "\n")
tar = os.path.join(tmp, "leaky.tar"); hs.write_deterministic_tar(st, tar); art = os.path.join(tmp, "leaky.tar.zst"); hs.zstd_compress(tar, art)
tree = hs.tree_hash(st); sha = hs.sha256_file(art)
lock = hl.make_lock(name="leaky", application="Leaky", variant=None, source_version="hpcperf-l3-v1", upstream={"url": "https://e/l.git", "tag": "v1", "commit": "d"*40},
    archive_info={"sha256": sha, "compressed_size": os.path.getsize(art), "uncompressed_size": 10}, tree_sha=tree, entries=2, layout=["src/"], patches=[], dependencies={}, components=[],
    equivalence=[], licenses=[], license_notes=[], redistribution_status="cleared", source_scope={}, scan_allow=[], freeze_tool_version="t", freeze_timestamp="now")
b = os.path.join(RT, "level3", "leaky"); os.makedirs(os.path.join(b, "provenance"), exist_ok=True); hs.dump_yaml(lock, hl.lock_path(b))
for f in ("upstream.lock", "patch_series.txt", "SOURCE_MANIFEST.json", "LICENSES.md"): open(os.path.join(b, "provenance", f), "w").write("x\n")
by = {"name": "leaky", "level": 3, "application": "Leaky"}; hl.apply_identity(by, lock); hs.dump_yaml(by, os.path.join(b, "benchmark.yaml"))
PY
[ $rc -eq 5 ] && echo "$out" | /usr/bin/grep -q 'SCAN HIT rule=env-dump' && ! echo "$out" | /usr/bin/grep -q 'ghp_B' && [ ! -d "$RT/level3/leaky/src" ] && ok "6a: artifact carrying a credential-looking file is not materialized (exit 5); paths/rules only" || bad "6a: rc=$rc $(echo "$out" | tail -2)"

# --- 7. check_workspace (baseline mode) ----------------------------------------------------------------------
out="$(cap python3 tools/check_workspace.py level3/miniapp --json "$B/provenance/check_workspace.json")"; rc=$?
[ $rc -eq 0 ] && echo "$out" | /usr/bin/grep -q 'check_workspace: PASS (15/15' && ok "7a: canonical benchmark passes all 15 checks" || bad "7a: rc=$rc: $(echo "$out" | /usr/bin/grep FAIL)"
echo 'SRC2="$R/_upstream/level3/miniapp"' >> "$B/build.sh"; out="$(cap python3 tools/check_workspace.py level3/miniapp)"
echo "$out" | /usr/bin/grep -qE 'FAIL +6:' && echo "$out" | /usr/bin/grep -qE 'FAIL +13:' && ok "7b: _upstream reference in build.sh fails checks 6 and 13" || bad "7b: $(echo "$out" | /usr/bin/grep FAIL)"
sed -i '$d' "$B/build.sh"
touch "$B/src/src/main.o"; out="$(cap python3 tools/check_workspace.py level3/miniapp)"
echo "$out" | /usr/bin/grep -qE 'FAIL +4:' && echo "$out" | /usr/bin/grep -qE 'FAIL +12:' && ok "7c: build output inside src fails the identity (4) and artifact (12) checks" || bad "7c"
rm -f "$B/src/src/main.o"
ln -s /etc/hosts "$B/src/src/esc"; out="$(cap python3 tools/check_workspace.py level3/miniapp)"
echo "$out" | /usr/bin/grep -qE 'FAIL +7:' && ok "7d: escaping symlink fails check 7" || bad "7d"
rm -f "$B/src/src/esc"
mkdir -p "$B/archives"; out="$(cap python3 tools/check_workspace.py level3/miniapp)"; echo "$out" | /usr/bin/grep -qE 'FAIL +9:' && ok "7e: a leftover archives/ directory (scheme 2) fails check 9" || bad "7e"; rmdir "$B/archives"
python3 tools/check_workspace.py level3/miniapp > /dev/null 2>&1 && ok "7f: clean again -> PASS" || bad "7f"

# --- 8. agent workspace: isolation, baseline, agent-mode checks, trusted validation ---------------------------
out="$(cap bash tools/create_agent_workspace.sh level3 miniapp run-001)"; rc=$?
W="$RT/workspaces/run-001/level3/miniapp"
[ $rc -eq 0 ] && [ -f "$W/src/src/main.cpp" ] && [ ! -L "$W/src" ] && [ ! -L "$W/deps" ] && ok "8a: workspace created with real src/deps copies (no symlink back to the canonical tree)" || bad "8a: rc=$rc: $(echo "$out" | tail -3)"
/usr/bin/grep -q "canonical_source_tree_sha256: $TREE" "$W/workspace.yaml" && /usr/bin/grep -q "workspace_initial_tree_sha256: $TREE" "$W/workspace.yaml" && /usr/bin/grep -q 'source_version: hpcperf-l3-v1' "$W/workspace.yaml" && /usr/bin/grep -q 'creation_timestamp:' "$W/workspace.yaml" && ok "8b: workspace.yaml records run id, source version, canonical + initial tree hashes, timestamp" || bad "8b"
[ -f "$RT/workspaces/run-001/workspace_baseline.json" ] && [ ! -w "$RT/workspaces/run-001/workspace_baseline.json" ] && [ -f "$RT/.hpcperf/workspace_baselines/run-001.json" ] && ok "8c: trusted baseline written read-only outside the agent cwd and mirrored under the repository" || bad "8c"
[ -w "$W/src/src/main.cpp" ] && [ ! -w "$W/validate.sh" ] && [ ! -w "$W/src/bench/in.lj" ] && ok "8d: source tree writable; validator and the declared input (under src/) non-writable" || bad "8d"
out="$(cap bash tools/create_agent_workspace.sh level3 miniapp run-001)"; rc=$?; [ $rc -ne 0 ] && ok "8e: an existing run id is refused" || bad "8e"
out="$(cap bash tools/create_agent_workspace.sh level3 miniapp run-002 --dest "$TMP/outside/run-002")"; rc=$?
W2="$TMP/outside/run-002/level3/miniapp"
[ $rc -eq 0 ] && [ -f "$W2/src/src/main.cpp" ] && [ -f "$TMP/outside/run-002/hpcperf_env.sh" ] && ok "8f: --dest creates the workspace outside the repository (self-contained harness copy)" || bad "8f: rc=$rc"
echo '// agent edit' >> "$W/src/src/main.cpp"
python3 tools/check_workspace.py "$W" > /dev/null 2>&1; rc=$?; [ $rc -ne 0 ] && ok "8g: baseline mode: a modified workspace no longer equals the canonical baseline (iteration 0 contract)" || bad "8g"
out="$(cap python3 tools/check_workspace.py "$W" --agent-mode --iteration 1 --report "$TMP/r1.json" --diff "$TMP/r1.diff")"; rc=$?
[ $rc -eq 0 ] && /usr/bin/grep -q '"modified_files": \[' "$TMP/r1.json" && /usr/bin/grep -q 'src/src/main.cpp' "$TMP/r1.json" && /usr/bin/grep -q 'agent edit' "$TMP/r1.diff" && ok "8h: agent mode: a source-tree change passes and is recorded (file list + diff + hashes)" || bad "8h: rc=$rc $(echo "$out" | /usr/bin/grep FAIL)"
python3 - "$TMP/r1.json" "$TREE" <<'PY' && ok "8i: report carries initial_source_hash == canonical and a different current_source_hash" || bad "8i"
import json, sys; r = json.load(open(sys.argv[1])); assert r["initial_source_hash"] == sys.argv[2] and r["current_source_hash"] != sys.argv[2] and r["iteration"] == 1
PY
! /usr/bin/grep -q 'agent edit' "$B/src/src/main.cpp" && ! /usr/bin/grep -q 'agent edit' "$W2/src/src/main.cpp" && python3 tools/check_workspace.py level3/miniapp > /dev/null 2>&1 && ok "8j: the canonical tree and the other workspace are untouched (isolation)" || bad "8j"
chmod u+w "$W/validate.sh"; echo '# tampered' >> "$W/validate.sh"
out="$(cap python3 tools/check_workspace.py "$W" --agent-mode --iteration 2)"; rc=$?
[ $rc -ne 0 ] && echo "$out" | /usr/bin/grep -q 'PROTECTED FILE TAMPERING' && echo "$out" | /usr/bin/grep -q 'validate.sh' && ok "8k: agent mode: a modified validator is protected-file tampering (check 4 FAIL)" || bad "8k: rc=$rc"
out="$(cap bash tools/validate_workspace.sh level3 miniapp "$W" --iteration 2)"; rc=$?
[ $rc -eq 6 ] && echo "$out" | /usr/bin/grep -q 'REFUSED' && /usr/bin/grep -q 'layer: integrity' "$RT/workspaces/run-001/reports/iter-2.verdict.yaml" && /usr/bin/grep -q 'verdict: REFUSED' "$RT/workspaces/run-001/reports/iter-2.verdict.yaml" && ok "8l: trusted validation REFUSES a workspace with a tampered validator (integrity layer, exit 6, no run)" || bad "8l: rc=$rc"
sed -i '$d' "$W/validate.sh"; chmod a-w "$W/validate.sh"
chmod u+w "$W/src/bench/log.ref"; echo 'x' >> "$W/src/bench/log.ref"
out="$(cap bash tools/validate_workspace.sh level3 miniapp "$W" --iteration 3)"; rc=$?
[ $rc -eq 6 ] && echo "$out" | /usr/bin/grep -q 'log.ref' && ok "8m: a modified reference file declared in benchmark.yaml is refused even though it lives under src/ (exit 6)" || bad "8m: rc=$rc"
printf 'ref\n' > "$W/src/bench/log.ref"; chmod a-w "$W/src/bench/log.ref"
# source tree explicitly mutable: bundled dependency source inside src/, a source build script, deps/, new and deleted source files
echo '// dep edit' >> "$W/src/lib/thirdparty/tp.c"; echo '# build config edit' >> "$W/src/src/gen.sh"
printf 'x' >> "$W/deps/tp/tp.tar.gz"; echo 'int n;' > "$W/src/src/new_kernel.cu"; rm -f "$W/src/src/kernel.cu"
out="$(cap python3 tools/check_workspace.py "$W" --agent-mode --iteration 4 --report "$TMP/r4.json")"; rc=$?
python3 - "$TMP/r4.json" <<'PY' && [ $rc -eq 0 ] && ok "8n: src/** (incl. bundled dependency source and build scripts) and deps/** may be modified, added and deleted; every change is recorded" || bad "8n: rc=$rc $(echo "$out" | /usr/bin/grep FAIL)"
import json, sys; r = json.load(open(sys.argv[1]))
assert set(r["modified_files"]) >= {"src/lib/thirdparty/tp.c", "src/src/gen.sh", "deps/tp/tp.tar.gz"}, r["modified_files"]
assert r["added_files"] == ["src/src/new_kernel.cu"] and r["deleted_files"] == ["src/src/kernel.cu"], (r["added_files"], r["deleted_files"])
assert r["protected_violations"] == [] and r["source_surface"] == ["src/**", "deps/**"] and "src/bench/in.lj" in r["protected_inside_source"]
PY
rm -f "$W/src/src/new_kernel.cu"; cp -p "$B/src/src/kernel.cu" "$W/src/src/kernel.cu"; cp -p "$B/src/lib/thirdparty/tp.c" "$W/src/lib/thirdparty/tp.c"; cp -p "$B/src/src/gen.sh" "$W/src/src/gen.sh"; cp -p "$B/deps/tp/tp.tar.gz" "$W/deps/tp/tp.tar.gz"
# protected by default: declared input under src/, benchmark.yaml, a new file outside src/deps, a scope-like file dropped into the workspace
chmod u+w "$W/src/bench/in.lj"; echo 'run 1' >> "$W/src/bench/in.lj"
out="$(cap python3 tools/check_workspace.py "$W" --agent-mode --iteration 4)"; rc=$?
[ $rc -ne 0 ] && echo "$out" | /usr/bin/grep -q 'in.lj' && ok "8n2: a modified input declared in benchmark.yaml is refused although it lives under src/" || bad "8n2: rc=$rc"
printf 'run 100\n' > "$W/src/bench/in.lj"; chmod a-w "$W/src/bench/in.lj"
chmod u+w "$W/benchmark.yaml"; echo 'references: []' >> "$W/benchmark.yaml"
out="$(cap python3 tools/check_workspace.py "$W" --agent-mode --iteration 4)"; rc=$?
[ $rc -ne 0 ] && echo "$out" | /usr/bin/grep -q 'benchmark.yaml' && ok "8n3: a modified benchmark.yaml (the contract that names inputs/references) is refused" || bad "8n3: rc=$rc"
sed -i '$d' "$W/benchmark.yaml"; chmod a-w "$W/benchmark.yaml"
echo 'echo hi' > "$W/helper.sh"
out="$(cap python3 tools/check_workspace.py "$W" --agent-mode --iteration 4)"; rc=$?
[ $rc -ne 0 ] && echo "$out" | /usr/bin/grep -q 'helper.sh (added)' && ok "8n4: a new file outside src/ and deps/ is a protected-surface violation (protected by default)" || bad "8n4: rc=$rc"
rm -f "$W/helper.sh"
printf "modifiable: ['*']\n" > "$W/optimization_scope.yaml"
out="$(cap python3 tools/check_workspace.py "$W" --agent-mode --iteration 4)"; rc=$?
[ $rc -ne 0 ] && echo "$out" | /usr/bin/grep -q 'optimization_scope.yaml (added)' && ok "8n5: an optimization_scope.yaml dropped into the workspace is neither read nor tolerated (no active dependency on the removed file)" || bad "8n5: rc=$rc"
rm -f "$W/optimization_scope.yaml"
python3 tools/check_workspace.py "$W" --agent-mode --iteration 4 > /dev/null 2>&1 && /usr/bin/grep -q 'agent edit' "$W/src/src/main.cpp" && ok "8n6: after the protected-file probes the workspace passes again and the agent's own source edit is still in place (nothing re-materialized it)" || bad "8n6"
out="$(cap bash tools/validate_workspace.sh level3 miniapp "$W" --iteration 5)"; rc=$?
[ $rc -eq 0 ] && /usr/bin/grep -q 'layer: numerical' "$RT/workspaces/run-001/reports/iter-5.verdict.yaml" && /usr/bin/grep -q 'verdict: PASS' "$RT/workspaces/run-001/reports/iter-5.verdict.yaml" && /usr/bin/grep -q 'src/src/main.cpp' "$RT/workspaces/run-001/reports/iter-5.verdict.yaml" && /usr/bin/grep -q 'baseline_origin: repository' "$RT/workspaces/run-001/reports/iter-5.verdict.yaml" && ok "8o: trusted validation runs build + validate on a workspace with only source-tree changes; numerical-layer verdict, modified files and repository baseline recorded" || bad "8o: rc=$rc $(echo "$out" | tail -3)"
# harness tampering (outside the agent cwd, inside the workspace root)
chmod u+w "$RT/workspaces/run-001/level3/tools/l3_common.sh"; echo '# tampered harness' >> "$RT/workspaces/run-001/level3/tools/l3_common.sh"
out="$(cap bash tools/validate_workspace.sh level3 miniapp "$W" --iteration 51)"; rc=$?
[ $rc -eq 6 ] && echo "$out" | /usr/bin/grep -q 'HARNESS TAMPERING' && ok "8o2: a modified harness copy (level3/tools) in the workspace root is refused (exit 6)" || bad "8o2: rc=$rc"
sed -i '$d' "$RT/workspaces/run-001/level3/tools/l3_common.sh"; chmod a-w "$RT/workspaces/run-001/level3/tools/l3_common.sh"
printf '#!/bin/bash\nexit 1\n' > "$TMP/badbuild.sh"; chmod +x "$TMP/badbuild.sh"; cp -p "$W/build.sh" "$TMP/build.sh.keep"; chmod u+w "$W/build.sh"; cp "$TMP/badbuild.sh" "$W/build.sh"
out="$(cap bash tools/validate_workspace.sh level3 miniapp "$W" --iteration 52)"; rc=$?
[ $rc -eq 6 ] && ok "8o3: a replaced build.sh is integrity tampering (refused before any build), not BUILD_FAIL" || bad "8o3: rc=$rc"
cp -p "$TMP/build.sh.keep" "$W/build.sh"; chmod a-w "$W/build.sh"
# an in-workspace baseline alone is not trusted; explicit --baseline inside the workspace is refused
mv "$RT/.hpcperf/workspace_baselines/run-001.json" "$TMP/run-001.baseline.keep"
out="$(cap bash tools/validate_workspace.sh level3 miniapp "$W" --iteration 53)"; rc=$?
[ $rc -eq 6 ] && echo "$out" | /usr/bin/grep -q 'not trusted' && ok "8o4: without the repository baseline copy the workspace's own baseline is NOT trusted (refused)" || bad "8o4: rc=$rc $(echo "$out" | /usr/bin/grep -E 'FAIL +5' | head -1)"
out="$(cap bash tools/validate_workspace.sh level3 miniapp "$W" --iteration 54 --baseline "$RT/workspaces/run-001/workspace_baseline.json")"; rc=$?
[ $rc -eq 6 ] && ok "8o5: --baseline pointing inside the workspace root is refused" || bad "8o5: rc=$rc"
out="$(cap bash tools/validate_workspace.sh level3 miniapp "$W" --iteration 55 --baseline "$TMP/run-001.baseline.keep")"; rc=$?
[ $rc -eq 0 ] && ok "8o6: an explicit baseline outside the workspace is accepted" || bad "8o6: rc=$rc"
mv "$TMP/run-001.baseline.keep" "$RT/.hpcperf/workspace_baselines/run-001.json"
rm -f "$RT/.hpcperf/workspace_baselines/run-001.json"; chmod u+w "$RT/workspaces/run-001/workspace_baseline.json"; python3 - "$RT/workspaces/run-001/workspace_baseline.json" <<'PY'
import json, sys; b = json.load(open(sys.argv[1])); b["files"]["validate.sh"]["sha256"] = "0"*64; json.dump(b, open(sys.argv[1], "w"))
PY
out="$(cap python3 tools/check_workspace.py "$W" --agent-mode --iteration 6)"; rc=$?
[ $rc -ne 0 ] && echo "$out" | /usr/bin/grep -q 'no trusted baseline' && ok "8p: a forged workspace-local baseline is ignored (no trusted baseline -> check 5 FAIL), never consulted" || bad "8p"
out="$(cap python3 tools/check_workspace.py "$W" --agent-mode --iteration 6 --allow-workspace-baseline)"; rc=$?
[ $rc -ne 0 ] && ok "8p2: even with --allow-workspace-baseline (development) the forged baseline only produces a mismatch, never a PASS" || bad "8p2"

# --- 8q-8v. fix C: iteration/run association, build provenance, backend & variant consistency -------------
bash tools/create_agent_workspace.sh level3 miniapp run-003 > /dev/null 2>&1
W3="$RT/workspaces/run-003/level3/miniapp"; R3="$RT/workspaces/run-003/reports"
out="$(cap bash tools/validate_workspace.sh level3 miniapp "$W3" --iteration 1 -- CUDA)"; rc=$?
[ $rc -eq 0 ] && /usr/bin/grep -q 'build_provenance: built_this_iteration' "$R3/iter-1.verdict.yaml" && ok "8q: a built iteration records build_provenance built_this_iteration" || bad "8q: rc=$rc $(/usr/bin/grep build_provenance "$R3/iter-1.verdict.yaml" | head -1)"
python3 - "$R3/iter-1.verdict.yaml" <<'PY' && ok "8r: the verdict lists exactly this iteration's runs (run id, binary, sha256, audit) -- no historical manifest" || bad "8r"
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])); r = d["runs_this_iteration"]
assert d["runs_this_iteration_count"] == 1 and len(r) == 1, d["runs_this_iteration_count"]
assert r[0]["run_id"] and r[0]["binary_sha256"] and r[0]["binary"].endswith("mini.bin"), r[0]
assert r[0]["gpu_binding_audits"] == [{"verified": 1, "mismatch": 0, "unverified": 0}], r[0].get("gpu_binding_audits")
assert d["validated_binaries"] == [r[0]["binary_sha256"]]
PY
RUN1="$(python3 -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["runs_this_iteration"][0]["run_id"])' "$R3/iter-1.verdict.yaml")"
out="$(cap bash tools/validate_workspace.sh level3 miniapp "$W3" --iteration 2 --skip-build -- CUDA)"; rc=$?
RUN2="$(python3 -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["runs_this_iteration"][0]["run_id"])' "$R3/iter-2.verdict.yaml" 2>/dev/null)"
[ -n "$RUN2" ] && [ "$RUN1" != "$RUN2" ] && ok "8r2: the second iteration reports ITS OWN appended run record, not the first iteration's (manifests are appended to)" || bad "8r2: iter1 $RUN1 iter2 $RUN2"
[ $rc -eq 0 ] && /usr/bin/grep -q 'build_provenance: verified_from_build_record' "$R3/iter-2.verdict.yaml" && ok "8s: --skip-build with an unchanged source is covered by the trusted build record" || bad "8s: rc=$rc $(/usr/bin/grep build_provenance "$R3/iter-2.verdict.yaml" | head -1)"
echo '// another agent edit' >> "$W3/src/src/main.cpp"
out="$(cap bash tools/validate_workspace.sh level3 miniapp "$W3" --iteration 3 --skip-build -- CUDA)"; rc=$?
[ $rc -eq 0 ] && /usr/bin/grep -q 'build_provenance: UNVERIFIED' "$R3/iter-3.verdict.yaml" && /usr/bin/grep -q 'NOT proven to be compiled' "$R3/iter-3.verdict.yaml" && ok "8t: --skip-build after a source edit reports UNVERIFIED build provenance (never claims the edit was compiled)" || bad "8t: rc=$rc $(/usr/bin/grep build_provenance "$R3/iter-3.verdict.yaml" | head -1)"
out="$(cap bash tools/validate_workspace.sh level3 miniapp "$W3" --iteration 4 -- CUDA)"; rc=$?
[ $rc -eq 0 ] && /usr/bin/grep -q 'build_provenance: built_this_iteration' "$R3/iter-4.verdict.yaml" && ok "8u: rebuilding the edited source restores a proven build provenance" || bad "8u: rc=$rc"
out="$(cap bash tools/validate_workspace.sh level3 miniapp "$W3" --iteration 5 --backend HIP -- CUDA)"; rc=$?
[ $rc -eq 2 ] && echo "$out" | /usr/bin/grep -q 'disagree' && ok "8v: a backend given twice and inconsistently is refused (build and validation cannot diverge)" || bad "8v: rc=$rc"
# variant consistency: a workspace whose marker names a variant refuses a contradicting environment
FK="$TMP/fakevar/level3/miniapp"; mkdir -p "$FK"
printf 'run_id: fake-001\niteration: 0\n' > "$FK/workspace.yaml"
printf 'variant_env: HPCPERF_MINI_VARIANT\nvariants: {a: {}, b: {}}\n' > "$FK/benchmark.yaml"
printf 'variant: a\n' > "$FK/.hpcperf-materialized.yaml"
out="$(cap env HPCPERF_MINI_VARIANT=b bash tools/validate_workspace.sh level3 miniapp "$FK" --iteration 1 -- CUDA)"; rc=$?
[ $rc -eq 6 ] && echo "$out" | /usr/bin/grep -q 'does not match the materialized variant' && ok "8w: an environment variant that contradicts the materialized one is REFUSED (exit 6) before any build" || bad "8w: rc=$rc"

# --- 9. release manifest / publish plan ----------------------------------------------------------------------
cat > "$TMP/catalog.yaml" <<'EOF'
schema: hpcperf-artifact-catalog-1
suite: test-suite
source_version: hpcperf-l3-v1
release: {proposed_release_name: test-release, provider: null, status: REMOTE_ARTIFACT_UNPUBLISHED}
naming: "<app>-<source_version>.tar.zst"
applications:
  miniapp: {application: MiniApp, suite_status: retained, variants: [null]}
  leaky:   {application: Leaky, suite_status: retained, variants: [null]}
  gone:    {application: Gone, suite_status: retired, reason: test, publish: never, variants: [null]}
default_suite_order: [miniapp]
EOF
out="$(cap python3 tools/artifacts/generate_release_manifest.py --catalog "$TMP/catalog.yaml" --json "$TMP/rel.json" --md "$TMP/rel.md" --verify)"; rc=$?
[ $rc -eq 0 ] && echo "$out" | /usr/bin/grep -q 'miniapp: FROZEN, LOCAL_ARTIFACT_VERIFIED, LOCAL_MATERIALIZATION_VERIFIED, REMOTE_ARTIFACT_UNPUBLISHED' && echo "$out" | /usr/bin/grep -q 'gone: RETIRED' && ok "9a: release manifest reports FROZEN / LOCAL_ARTIFACT_VERIFIED / LOCAL_MATERIALIZATION_VERIFIED / REMOTE_ARTIFACT_UNPUBLISHED and RETIRED" || bad "9a: rc=$rc $out"
! /usr/bin/grep -q "$TMP" "$TMP/rel.md" && ! /usr/bin/grep -q "$TMP" "$TMP/rel.json" && ok "9b: no staging path recorded in the manifest (status only)" || bad "9b"
out="$(cap bash tools/artifacts/publish_artifacts.sh --dry-run --app miniapp,leaky,gone --staging "$HPCPERF_ARTIFACT_STAGING" --catalog "$TMP/catalog.yaml")"; rc=$?
echo "$out" | /usr/bin/grep -qE '^miniapp +- +PLAN' && echo "$out" | /usr/bin/grep -qE '^leaky +- +REFUSE.*missing from staging' && echo "$out" | /usr/bin/grep -qE '^gone +- +REFUSE' && ok "9c: publish plan: PLAN for the verified artifact, REFUSE for a missing artifact and a retired application" || bad "9c: $out"
python3 - "$LOCK" <<'PY'
import sys, yaml; l = yaml.safe_load(open(sys.argv[1])); l["redistribution_status"] = "review"; yaml.safe_dump(l, open(sys.argv[1], "w"), sort_keys=False)
PY
out="$(cap bash tools/artifacts/publish_artifacts.sh --dry-run --app miniapp --staging "$HPCPERF_ARTIFACT_STAGING" --catalog "$TMP/catalog.yaml")"
echo "$out" | /usr/bin/grep -qE '^miniapp +- +REFUSE.*redistribution_status=review' && ok "9d: redistribution_status != cleared -> REFUSE" || bad "9d: $out"
python3 - "$LOCK" <<'PY'
import sys, yaml; l = yaml.safe_load(open(sys.argv[1])); l["redistribution_status"] = "cleared"; yaml.safe_dump(l, open(sys.argv[1], "w"), sort_keys=False)
PY
out="$(cap bash tools/artifacts/publish_artifacts.sh --provider github-release --app miniapp --staging "$HPCPERF_ARTIFACT_STAGING" --catalog "$TMP/catalog.yaml")"; rc=$?
[ $rc -eq 3 ] && echo "$out" | /usr/bin/grep -q 'REFUSED -- no provider adapter' && ok "9e: a real publish is refused (no provider adapter; nothing uploaded)" || bad "9e: rc=$rc"

echo
echo "test_source_tools: $pass passed, $failn failed"
[ "$failn" -eq 0 ]
