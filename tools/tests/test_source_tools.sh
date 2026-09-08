#!/bin/bash
# Static tests of the Level 3 source-distribution tools (no application, no GPU, no network):
#   hpcperf_source.py (tree hash, scan, deterministic archive), freeze_benchmark_source.py,
#   compare_source_trees.py, prepare_benchmark.sh / hpcperf_materialize.py, check_workspace.py,
#   create_agent_workspace.sh -- on a synthetic mini benchmark built from a throwaway git repo.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$(cd "$HERE/.." && pwd)"
pass=0; failn=0
ok()  { echo "ok   $*"; pass=$((pass+1)); }
bad() { echo "FAIL $*"; failn=$((failn+1)); }
TMP="$(mktemp -d "${HPCPERF_FREEZE_SCRATCH:-/tmp}/srctools-XXXXXX")"; trap 'chmod -R u+w "$TMP" 2>/dev/null; rm -rf "$TMP"' EXIT
export HPCPERF_FREEZE_SCRATCH="$TMP/scratch"; mkdir -p "$HPCPERF_FREEZE_SCRATCH"
export PYTHONDONTWRITEBYTECODE=1

# --- a fake repository root with the harness pieces the tools expect ------------------------------------
RT="$TMP/repo"; mkdir -p "$RT/level3/tools" "$RT/level2/tools" "$RT/tools" "$RT/_upstream/level3"
cp -a "$TOOLS"/*.py "$TOOLS"/*.sh "$RT/tools/"; cp -a "$HERE/../../level3/tools/." "$RT/level3/tools/" 2>/dev/null || true
printf '#!/bin/bash\nexport HPC_PERFORMANCE_AI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n' > "$RT/hpcperf_env.sh"
# upstream "application": a git repo with a symlink, an executable, a nested dir and a doc dir; the checkout is
# reached through a directory symlink (as the worktrees share checkouts) -- the tools must resolve that
UPREAL="$TMP/checkouts/miniapp"; mkdir -p "$UPREAL/src" "$UPREAL/doc" "$UPREAL/bench" "$UPREAL/lib/thirdparty"
UP="$RT/_upstream/level3/miniapp"; ln -s "$UPREAL" "$UP"
( cd "$UP" && git init -q && git config user.email t@t && git config user.name t
  echo 'int main(){return 0;}' > src/main.cpp; printf '#include <x>\nint k(){return 1;}\n' > src/kernel.cu
  echo 'manual' > doc/manual.txt; echo 'run 100' > bench/in.lj; echo 'ref' > bench/log.ref; echo 'tp' > lib/thirdparty/tp.c
  printf '#!/bin/sh\necho build\n' > src/gen.sh; chmod +x src/gen.sh; ln -s ../bench/in.lj src/in.link; echo 'GPL' > LICENSE
  git add -A && git commit -qm init )
SHA="$(git -C "$UP" rev-parse HEAD)"
# harness of the mini benchmark
B="$RT/level3/miniapp"; mkdir -p "$B/provenance" "$B/patches"
cat > "$B/patches/0001-fix.patch" <<'EOF'
--- a/src/main.cpp
+++ b/src/main.cpp
@@ -1 +1 @@
-int main(){return 0;}
+int main(){return 7;}
EOF
for s in build.sh run.sh validate.sh; do printf '#!/bin/bash\nHERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\nSRC="$HERE/src"\necho %s from "$SRC"\n' "$s" > "$B/$s"; chmod +x "$B/$s"; done
cat > "$B/provenance/freeze_spec.yaml" <<EOF
schema: hpcperf-freeze-spec-1
name: miniapp
level: 3
application: MiniApp
benchmark_source_version: v1
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
EOF
mkdir -p "$RT/.deps/level3/miniapp/downloads"; ( cd "$TMP" && mkdir -p tp && echo 'int t;' > tp/t.c && tar -czf "$RT/.deps/level3/miniapp/downloads/tp.tar.gz" tp )
TPSHA="$(sha256sum "$RT/.deps/level3/miniapp/downloads/tp.tar.gz" | cut -d' ' -f1)"; sed -i "s/__TPSHA__/$TPSHA/" "$B/provenance/freeze_spec.yaml"
cat > "$B/optimization_scope.yaml" <<'EOF'
schema: hpcperf-optimization-scope-1
benchmark: miniapp
modifiable: ['src/src/*']
readonly: [build.sh, run.sh, validate.sh, benchmark.yaml, optimization_scope.yaml, 'provenance/*', 'src/bench/*', 'src/lib/*', 'deps/*']
excluded: []
loc_categories: {application_owned: ['src/src/*'], bundled_dependency: ['src/lib/*'], benchmark_specific_dependency: ['deps/*'], test: [], exclude: []}
EOF
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
optimization_scope: ./optimization_scope.yaml
environment_profile: none
install_root: .deps/level3/miniapp
dependency_installs: []
inputs: [src/bench/in.lj]
references: [src/bench/log.ref]
EOF
cd "$RT"

# --- 1. freeze: patched baseline, equivalence, deterministic archive, provenance ------------------------
out="$(python3 tools/freeze_benchmark_source.py level3/miniapp --verify-determinism 2>&1)"; rc=$?
[ $rc -eq 0 ] && echo "$out" | /usr/bin/grep -q 'FREEZE OK' && ok "1a: freeze of the synthetic benchmark succeeds (patched, equivalent, deterministic archive)" || bad "1a: rc=$rc: $(echo "$out" | tail -5)"
echo "$out" | /usr/bin/grep -q 'second archive IDENTICAL' && ok "1b: archive reproducible (two builds, same sha256)" || bad "1b: determinism line missing"
echo "$out" | /usr/bin/grep -q 'EQUIVALENT (identical 7, patch 1, generated 0, excluded 1' && ok "1c: equivalence: patched file and excluded doc classified, nothing unexpected" || bad "1c: $(echo "$out" | /usr/bin/grep equivalence)"
for f in source.lock.yaml upstream.lock patch_series.txt original_vs_baseline.diff SOURCE_MANIFEST.json LICENSES.md equivalence.json equivalence.md; do [ -f "$B/provenance/$f" ] || bad "1d: provenance/$f missing"; done; ok "1d: provenance files written"
/usr/bin/grep -q 'return 7' "$B/provenance/original_vs_baseline.diff" && ok "1e: original_vs_baseline.diff shows the patch" || bad "1e: diff content"
TREE="$(sed -n 's/^source_tree_sha256: //p' "$B/benchmark.yaml")"; ARCH="$(sed -n 's/^  archive_sha256: //p' "$B/benchmark.yaml")"
[ -n "$TREE" ] && [ -n "$ARCH" ] && ok "1f: benchmark.yaml carries source_tree_sha256 + archive_sha256" || bad "1f: identity missing"
# re-freeze -> same tree hash (timestamp differs, hash must not)
out2="$(python3 tools/freeze_benchmark_source.py level3/miniapp 2>&1)"; TREE2="$(sed -n 's/^source_tree_sha256: //p' "$B/benchmark.yaml")"
[ "$TREE" = "$TREE2" ] && ok "1g: re-freeze reproduces source_tree_sha256" || bad "1g: $TREE vs $TREE2"

# --- 2. scan / symlink negatives ---------------------------------------------------------------------
( cd "$UP" && printf 'export GITHUB_TOKEN=ghp_%s\n' "$(printf 'A%.0s' $(seq 36))" > src/toolchain.env && git add -A && git commit -qm leak )
SHA2="$(git -C "$UP" rev-parse HEAD)"; sed -i "s/commit: $SHA\$/commit: $SHA2/; s/commit: $SHA}/commit: $SHA2}/" "$B/provenance/freeze_spec.yaml"
out="$(python3 tools/freeze_benchmark_source.py level3/miniapp 2>&1)"; rc=$?
[ $rc -ne 0 ] && echo "$out" | /usr/bin/grep -q 'SCAN HIT' && ! echo "$out" | /usr/bin/grep -q 'ghp_A' && ok "2a: credential-looking file/content fails the freeze; no value printed" || bad "2a: rc=$rc"
( cd "$UP" && git rm -q src/toolchain.env && ln -s /etc/passwd src/escape && git add -A && git commit -qm esc )
SHA3="$(git -C "$UP" rev-parse HEAD)"; sed -i "s/commit: $SHA2/commit: $SHA3/g" "$B/provenance/freeze_spec.yaml"
out="$(python3 tools/freeze_benchmark_source.py level3/miniapp 2>&1)"; rc=$?
[ $rc -ne 0 ] && echo "$out" | /usr/bin/grep -q 'escaping' && ok "2b: symlink escaping the bundle fails the freeze" || bad "2b: rc=$rc: $(echo "$out" | tail -2)"
( cd "$UP" && git rm -q src/escape && echo 'int main(){return 0;}' > src/main.cpp && git add -A && git commit -qm restore )
SHA4="$(git -C "$UP" rev-parse HEAD)"; sed -i "s/commit: $SHA3/commit: $SHA4/g" "$B/provenance/freeze_spec.yaml"
# validated tree with an unexplained extra file -> UNEXPECTED stops the freeze
echo 'stray' > "$UP/src/stray.c"
out="$(python3 tools/freeze_benchmark_source.py level3/miniapp 2>&1)"; rc=$?
[ $rc -ne 0 ] && echo "$out" | /usr/bin/grep -q 'UNEXPECTED' && ok "2c: unexplained difference vs the validated tree stops the migration" || bad "2c: rc=$rc"
rm -f "$UP/src/stray.c"; python3 tools/freeze_benchmark_source.py level3/miniapp > /dev/null 2>&1 || bad "2d: clean re-freeze failed"
TREE="$(sed -n 's/^source_tree_sha256: //p' "$B/benchmark.yaml")"

# --- 3. prepare (materialize) -----------------------------------------------------------------------
out="$(bash tools/prepare_benchmark.sh level3 miniapp 2>&1)"; rc=$?
[ $rc -eq 0 ] && [ -f "$B/src/src/main.cpp" ] && [ -f "$B/deps/tp/tp.tar.gz" ] && /usr/bin/grep -q 'return 7' "$B/src/src/main.cpp" && ok "3a: prepare materializes src/ and deps/ (patched baseline)" || bad "3a: rc=$rc: $(echo "$out" | tail -3)"
[ ! -d "$B/src/doc" ] && ok "3b: excluded doc/ absent from the materialized tree" || bad "3b: doc present"
out="$(bash tools/prepare_benchmark.sh level3 miniapp 2>&1)"; rc=$?
[ $rc -eq 0 ] && echo "$out" | /usr/bin/grep -q 'nothing to do' && ok "3c: prepare is idempotent on an identical tree" || bad "3c: rc=$rc"
echo '// agent change' >> "$B/src/src/main.cpp"
out="$(bash tools/prepare_benchmark.sh level3 miniapp 2>&1)"; rc=$?
[ $rc -eq 3 ] && echo "$out" | /usr/bin/grep -q 'never overwritten' && /usr/bin/grep -q 'agent change' "$B/src/src/main.cpp" && ok "3d: modified src refused (exit 3), content untouched" || bad "3d: rc=$rc"
out="$(bash tools/prepare_benchmark.sh level3 miniapp --force-rematerialize 2>&1)"; rc=$?
[ $rc -eq 0 ] && ! /usr/bin/grep -q 'agent change' "$B/src/src/main.cpp" && ok "3e: --force-rematerialize restores the frozen baseline" || bad "3e: rc=$rc"
# LFS pointer instead of the archive
cp "$B/archives/source_bundle.tar.zst" "$TMP/arch.bak"; python3 - "$B/archives/source_bundle.tar.zst" "$ARCH" <<'PY'
import sys; open(sys.argv[1],'w').write(f"version https://git-lfs.github.com/spec/v1\noid sha256:{sys.argv[2]}\nsize 1\n")
PY
rm -rf "$B/src" "$B/deps"; out="$(bash tools/prepare_benchmark.sh level3 miniapp 2>&1)"; rc=$?
[ $rc -eq 4 ] && echo "$out" | /usr/bin/grep -q 'git lfs pull' && ok "3f: LFS pointer detected with an explicit message (exit 4)" || bad "3f: rc=$rc: $(echo "$out" | tail -1)"
cp "$TMP/arch.bak" "$B/archives/source_bundle.tar.zst"
# corrupted archive
printf 'x' >> "$B/archives/source_bundle.tar.zst"; out="$(bash tools/prepare_benchmark.sh level3 miniapp 2>&1)"; rc=$?
[ $rc -ne 0 ] && echo "$out" | /usr/bin/grep -q 'sha256' && ok "3g: archive sha256 mismatch refused" || bad "3g: rc=$rc"
cp "$TMP/arch.bak" "$B/archives/source_bundle.tar.zst"; bash tools/prepare_benchmark.sh level3 miniapp > /dev/null 2>&1 || bad "3h: re-prepare"

# --- 4. check_workspace ------------------------------------------------------------------------------
out="$(python3 tools/check_workspace.py level3/miniapp 2>&1)"; rc=$?
[ $rc -eq 0 ] && echo "$out" | /usr/bin/grep -q 'check_workspace: PASS (15/15' && ok "4a: canonical benchmark passes all 15 checks" || bad "4a: rc=$rc: $(echo "$out" | /usr/bin/grep FAIL)"
echo 'SRC2="$R/_upstream/level3/miniapp"' >> "$B/build.sh"; out="$(python3 tools/check_workspace.py level3/miniapp 2>&1)"; rc=$?
[ $rc -ne 0 ] && echo "$out" | /usr/bin/grep -qE 'FAIL +7:' && echo "$out" | /usr/bin/grep -qE 'FAIL +15:' && ok "4b: _upstream reference in build.sh fails checks 7 and 15" || bad "4b: $(echo "$out" | /usr/bin/grep FAIL)"
sed -i '$d' "$B/build.sh"
touch "$B/src/src/main.o"; out="$(python3 tools/check_workspace.py level3/miniapp 2>&1)"; rc=$?
[ $rc -ne 0 ] && echo "$out" | /usr/bin/grep -qE 'FAIL +5:' && echo "$out" | /usr/bin/grep -qE 'FAIL +14:' && ok "4c: build output inside src fails the hash (5) and artifact (14) checks" || bad "4c: $(echo "$out" | /usr/bin/grep FAIL)"
rm -f "$B/src/src/main.o"
ln -s /etc/hosts "$B/src/src/esc"; out="$(python3 tools/check_workspace.py level3/miniapp 2>&1)"; rc=$?
[ $rc -ne 0 ] && echo "$out" | /usr/bin/grep -qE 'FAIL +8:' && ok "4d: escaping symlink fails check 8" || bad "4d"
rm -f "$B/src/src/esc"
python3 tools/check_workspace.py level3/miniapp > /dev/null 2>&1 && ok "4e: clean again -> PASS" || bad "4e"

# --- 5. agent workspace ------------------------------------------------------------------------------
out="$(bash tools/create_agent_workspace.sh level3 miniapp run-001 2>&1)"; rc=$?
W="$RT/workspaces/run-001/level3/miniapp"
[ $rc -eq 0 ] && [ -f "$W/src/src/main.cpp" ] && [ ! -L "$W/src" ] && [ ! -e "$W/archives" ] && ok "5a: workspace created with a real src copy, no archives" || bad "5a: rc=$rc: $(echo "$out" | tail -3)"
/usr/bin/grep -q "canonical_source_tree_sha256: $TREE" "$W/workspace.yaml" && /usr/bin/grep -q "workspace_initial_tree_sha256: $TREE" "$W/workspace.yaml" && ok "5b: workspace.yaml records canonical and initial tree hashes" || bad "5b"
[ -w "$W/src/src/main.cpp" ] && [ ! -w "$W/build.sh" ] && [ ! -w "$W/src/bench/in.lj" ] && ok "5c: modifiable src writable, readonly ranges protected" || bad "5c"
[ -f "$RT/workspaces/run-001/hpcperf_env.sh" ] && [ -d "$RT/workspaces/run-001/level3/tools" ] && ok "5d: workspace root carries the harness (env + level3/tools)" || bad "5d"
out="$(bash tools/create_agent_workspace.sh level3 miniapp run-001 2>&1)"; rc=$?
[ $rc -ne 0 ] && ok "5e: an existing run id is refused" || bad "5e"
echo 'x' >> "$W/src/src/main.cpp"; python3 tools/check_workspace.py "$W" > /dev/null 2>&1; rc=$?
[ $rc -ne 0 ] && ok "5f: a modified workspace no longer matches the canonical baseline (check 5 FAIL, as intended for iteration>0)" || bad "5f"
python3 tools/check_workspace.py level3/miniapp > /dev/null 2>&1 && ok "5g: the canonical tree is untouched by the workspace modification" || bad "5g"

echo
echo "test_source_tools: $pass passed, $failn failed"
[ "$failn" -eq 0 ]
