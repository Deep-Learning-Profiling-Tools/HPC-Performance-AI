#!/bin/bash
# CPU-only negative/positive tests for the Level 3 correctness & reproducibility
# helpers in level3/tools/l3_common.sh and l3_check.py. No GPU, no application,
# no build -- these check that the mechanisms which decide PASS/FAIL behave.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../../.." && pwd)"
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
TOOLS="$R/level3/tools"
pass=0; failn=0
ok()  { echo "ok   $*"; pass=$((pass+1)); }
bad() { echo "FAIL $*"; failn=$((failn+1)); }
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# 1. l3_check.require_finite rejects NaN/Inf/non-numbers, accepts finite
py_check() { python3 - "$1" <<'PY'
import os, sys
sys.path.insert(0, os.environ["L3_TOOLS"])
from l3_check import require_finite, ValidationError
try:
    require_finite("x", sys.argv[1]); print("ACCEPT")
except ValidationError:
    print("REJECT")
PY
}
[ "$(L3_TOOLS=$TOOLS py_check nan)"  = REJECT ] && ok "1a: NaN rejected"            || bad "1a: NaN not rejected"
[ "$(L3_TOOLS=$TOOLS py_check inf)"  = REJECT ] && ok "1b: Inf rejected"            || bad "1b: Inf not rejected"
[ "$(L3_TOOLS=$TOOLS py_check abc)"  = REJECT ] && ok "1c: non-number rejected"     || bad "1c: non-number not rejected"
[ "$(L3_TOOLS=$TOOLS py_check 1.5)"  = ACCEPT ] && ok "1d: finite accepted"         || bad "1d: finite rejected"

# 2. l3_capture returns the COMMAND's exit code (not tee's), and saves output
rc=0; l3_capture "$TMP/cap.log" -- bash -c 'echo hello; exit 7' >/dev/null 2>&1 || rc=$?
[ "$rc" -eq 7 ] && ok "2a: l3_capture propagates real exit code (7)" || bad "2a: got rc=$rc, expected 7"
grep -q hello "$TMP/cap.log" && ok "2b: l3_capture saved stdout" || bad "2b: output not saved"

# 3. rc-gate pattern: a failed run must FAIL even if a stale log is present
#    (this is the logic every validator relies on).
echo "OLD PASS-looking log" > "$TMP/stale.log"
fake_validate() { # simulates: run fails (rc=1) but an old log exists
    local rc=0
    bash -c 'exit 1' || rc=$?
    [ "$rc" -eq 0 ] || return 1        # gate: nonzero run -> FAIL, regardless of stale log
    return 0
}
if fake_validate; then bad "3: stale-log gate let a failed run pass"; else ok "3: failed run FAILs even with a stale log present"; fi

# 4. l3_rundir: dry-run must not touch a real result dir (sentinel test)
real="$R/build/level3/__selftest__/run/case.np1"
mkdir -p "$real"; echo SENTINEL > "$real/keep.txt"
d_real="$(l3_rundir "$real")"
[ "$d_real" = "$real" ] && [ ! -e "$real/keep.txt" ] && ok "4a: real run recreates the dir fresh" || bad "4a: real run did not refresh ($d_real)"
echo SENTINEL > "$real/keep.txt"
d_dry="$(HPCPERF_DRY_RUN=1 l3_rundir "$real")"
if [ "$d_dry" != "$real" ] && [ -f "$real/keep.txt" ] && [ "$(cat "$real/keep.txt")" = SENTINEL ]; then
    ok "4b: dry-run used a scratch dir ($(basename "$(dirname "$d_dry")")/$(basename "$d_dry")) and left the real result untouched"
else bad "4b: dry-run touched the real result dir (d_dry=$d_dry)"; fi
d_bad=0; l3_rundir "/tmp/not-under-build" >/dev/null 2>&1 || d_bad=$?
[ "$d_bad" -ne 0 ] && ok "4c: l3_rundir refuses a path outside build/level3/" || bad "4c: accepted an out-of-tree path"
rm -rf "$R/build/level3/__selftest__"

# 5. fingerprint patch handling: missing patch is an error; changed content
#    changes the ordered series hash (cache-invalidation), same content stable.
export CXX=/bin/true FC=/bin/true
fp_missing=0; l3_fingerprint_text app sha cuda deps opts gam "$TMP/nope.patch" >/dev/null 2>&1 || fp_missing=$?
[ "$fp_missing" -ne 0 ] && ok "5a: missing patch is a hard error" || bad "5a: missing patch accepted"
printf 'A\n' > "$TMP/p.patch"
h1="$(l3_fingerprint_text app sha cuda deps opts gam "$TMP/p.patch" 2>/dev/null | sed -n 's/^patch_series_sha256=//p')"
printf 'B\n' > "$TMP/p.patch"   # same name, different content
h2="$(l3_fingerprint_text app sha cuda deps opts gam "$TMP/p.patch" 2>/dev/null | sed -n 's/^patch_series_sha256=//p')"
[ -n "$h1" ] && [ "$h1" != "$h2" ] && ok "5b: same-named patch with changed content changes the series hash (cache invalidated)" || bad "5b: series hash did not change ($h1 vs $h2)"
hn="$(l3_fingerprint_text app sha cuda deps opts gam 2>/dev/null | sed -n 's/^patch_series_sha256=//p')"
[ "$hn" = none ] && ok "5c: empty patch series -> 'none'" || bad "5c: empty series hash '$hn'"

# 6. l3_clean_env.sh: allow-listed names pass, credential-looking names are dropped even under an
#    allow-listed prefix, session/agent names are dropped, --show prints names only (never a value).
CE="$TOOLS/l3_clean_env.sh"
names="$(HPCPERF_SELFTEST_OK=keepme HPCPERF_SELFTEST_TOKEN=secretvalue CLAUDE_SELFTEST=secretvalue SELFTEST_API_KEY=secretvalue \
         "$CE" -- env 2>/dev/null | sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p')"
echo "$names" | /usr/bin/grep -qx HPCPERF_SELFTEST_OK && ok "6a: clean env keeps an allow-listed project variable" || bad "6a: HPCPERF_SELFTEST_OK dropped"
echo "$names" | /usr/bin/grep -qx HPCPERF_SELFTEST_TOKEN && bad "6b: credential-looking name survived through the HPCPERF_ prefix" || ok "6b: HPCPERF_*_TOKEN denied although its prefix is allow-listed"
echo "$names" | /usr/bin/grep -qE '^(CLAUDE_SELFTEST|SELFTEST_API_KEY)$' && bad "6c: agent/credential names survived" || ok "6c: CLAUDE_* and *_API_KEY dropped"
echo "$names" | /usr/bin/grep -qx PATH && ok "6d: PATH survives (the command can run)" || bad "6d: PATH dropped"
show="$(HPCPERF_SELFTEST_TOKEN=secretvalue "$CE" --show -- true 2>&1)"
echo "$show" | /usr/bin/grep -q secretvalue && bad "6e: --show printed a value" || ok "6e: --show prints names only"
echo "$show" | /usr/bin/grep -q 'denied by the credential rule: .*HPCPERF_SELFTEST_TOKEN' && ok "6f: --show names the denied variable" || bad "6f: denied variable not reported"

# 8. lock-file queries used by build.sh (source identity comes from provenance/source.lock*.yaml, never from git)
mkdir -p "$TMP/bench/provenance"
cat > "$TMP/bench/provenance/source.lock.yaml" <<'EOF'
schema: hpcperf-source-lock-2
schema_version: 2
benchmark: {name: bench, level: 3, application: Bench, variant: null, source_version: hpcperf-l3-v1}
upstream: {repository: u, tag: t, commit: aaaa1111}
artifact: {filename: bench-hpcperf-l3-v1.tar.zst, format: tar.zst, layout: [src/, deps/], size: 1, sha256: s, source_tree_sha256: tree9999, primary: {url: null, status: unpublished}, mirrors: []}
materialized_tree: {sha256: tree9999, layout: [src/, deps/]}
patches:
  - {path: patches/0001-a.patch, sha256: x}
  - {path: patches/0002-b.patch, sha256: y}
components:
  - {dest: src, kind: git, commit: aaaa1111, submodules: [{path: sub/one, commit: cccc3333}]}
  - {dest: deps/dep, kind: git, commit: bbbb2222}
EOF
[ "$(l3_source_commit "$TMP/bench")" = aaaa1111 ] && ok "8a: l3_source_commit reads upstream.commit" || bad "8a: $(l3_source_commit "$TMP/bench")"
[ "$(l3_source_tree_sha "$TMP/bench")" = tree9999 ] && ok "8b: l3_source_tree_sha reads artifact.source_tree_sha256" || bad "8b"
[ "$(l3_source_version "$TMP/bench")" = hpcperf-l3-v1 ] && ok "8b2: l3_source_version reads benchmark.source_version" || bad "8b2"
printf 'schema: hpcperf-source-lock-1\nupstream: {commit: zzzz}\n' > "$TMP/bench/provenance/source.lock.old.yaml"
rc=0; l3_source_commit "$TMP/bench" old >/dev/null 2>&1 || rc=$?
[ "$rc" -ne 0 ] && ok "8b3: a scheme-2 (schema 1) lock is refused by the lock queries" || bad "8b3"
[ "$(l3_component_commit "$TMP/bench" deps/dep)" = bbbb2222 ] && ok "8c: l3_component_commit finds a deps component" || bad "8c"
[ "$(l3_submodule_commit "$TMP/bench" src sub/one)" = cccc3333 ] && ok "8d: l3_submodule_commit finds a bundled submodule" || bad "8d"
[ "$(l3_lock_patches "$TMP/bench")" = "0001-a.patch 0002-b.patch" ] && ok "8e: l3_lock_patches lists the series in order" || bad "8e: $(l3_lock_patches "$TMP/bench")"
rc=0; l3_require_materialized "$TMP/bench" >/dev/null 2>&1 || rc=$?
[ "$rc" -eq 3 ] && ok "8f: l3_require_materialized fails (exit 3) without src/" || bad "8f: rc=$rc"
printf 'variant: hypregpu\n' > "$TMP/bench/.hpcperf-materialized.yaml"
[ "$(l3_materialized_variant "$TMP/bench")" = hypregpu ] && ok "8g: l3_materialized_variant reads the marker" || bad "8g"

# 7. l3_run_recorded: records the real exit code of a queue step and never aborts the caller
rcf="$TMP/rc.txt"; steps=0
( set -e; for r in 0 3 0; do l3_run_recorded "$rcf" "step$r" -- bash -c "exit $r"; echo step >> "$TMP/steps"; done )
[ "$(wc -l < "$TMP/steps")" -eq 3 ] && ok "7a: a queue under set -e continues past a step that exits 3" || bad "7a: queue stopped after $(wc -l < "$TMP/steps") step(s)"
[ "$(tr '\n' ' ' < "$rcf")" = "step0 0 step3 3 step0 0 " ] && ok "7b: exit codes recorded verbatim (0 3 0)" || bad "7b: recorded '$(tr '\n' ' ' < "$rcf")'"

# 9. backend/profile isolation of GENERATED state (one frozen source tree per benchmark; .deps/build/install/
#    logs/cache per profile; a profile names its backend; no legacy shared install is ever read).
ISO="$TMP/isoroot"; mkdir -p "$ISO"
iso() { ( L3_R="$ISO"; "$@" ); }                                        # a helper against a throwaway repo root
paths_of() { ( L3_R="$ISO"; l3_paths_profile "$1" "$2" "$3" >/dev/null 2>&1 || exit 9; echo "$L3_SRC $L3_BUILD_DEPS $L3_INSTALL $L3_LOGS $L3_CACHE $L3_BUILD" ); }
pc="$(paths_of app cuda cuda)"; ph="$(paths_of app hip hip)"
if [ -n "$pc" ] && [ -n "$ph" ]; then
    shared=0; for x in $pc; do case " $ph " in *" $x "*) shared=1;; esac; done
    [ "$shared" -eq 0 ] && ok "9a: cuda and hip profiles share no src-copy/dep-build/install/logs/cache/build-tree path" || bad "9a: a generated path is shared between the cuda and hip profiles"
    case "$pc" in *"$ISO/.deps/level3/app/cuda/install "*"$ISO/build/level3/app/cuda") ok "9b: cuda profile = .deps/level3/app/cuda/{src,build,install,logs,cache} + build/level3/app/cuda";; *) bad "9b: unexpected cuda paths: $pc";; esac
    case "$pc" in *"$ISO/.deps/level3/app/cuda/src "*) ok "9c: the profile's src is a build-side copy location under .deps/, never level3/<app>/src";; *) bad "9c: L3_SRC not under the profile root: $pc";; esac
else bad "9a/9b/9c: l3_paths_profile failed for a plain backend profile"; fi
# profile identity must name the backend; conflicts are refused before any directory is created
rc=0; iso l3_paths_profile app2 cuda hip >/dev/null 2>&1 || rc=$?
[ "$rc" -ne 0 ] && [ ! -e "$ISO/.deps/level3/app2" ] && [ ! -e "$ISO/build/level3/app2" ] && ok "9d: profile 'cuda' with BACKEND hip is refused and creates nothing" || bad "9d: conflict accepted (rc=$rc) or directories created"
rc=0; iso l3_paths_profile app2 foo cuda >/dev/null 2>&1 || rc=$?
[ "$rc" -ne 0 ] && [ ! -e "$ISO/.deps/level3/app2" ] && ok "9e: a profile that names no backend ('foo') is refused" || bad "9e: backend-less profile accepted"
acc=0; for c in "hypregpu.cuda cuda" "cpucoarse.cuda cuda" "cpucoarse.hip hip" "cuda132-gcc142-ompi5010 cuda" "clang231-cuda132-offload cuda" "hip-gfx950-adiabatic hip" "cpu-gcc133-adiabatic cpu" "cuda-mytest cuda"; do
    set -- $c; l3_profile_backend_check "$1" "$2" >/dev/null 2>&1 || { acc=1; echo "     unexpectedly refused: $1 for $2"; }; done
[ "$acc" -eq 0 ] && ok "9f: nekRS <variant>.<backend>, toolchain-style and override profiles naming their backend are accepted" || bad "9f: a valid profile was refused"
rej=0; for c in "cpucoarse.cuda hip" "hypregpu.cuda hip" "cuda132-gcc142-ompi5010 hip" "hip-gfx950-adiabatic cuda" "cuda-mytest hip" "cuda bogus"; do
    set -- $c; l3_profile_backend_check "$1" "$2" >/dev/null 2>&1 && { rej=1; echo "     unexpectedly accepted: $1 for $2"; }; done
[ "$rej" -eq 0 ] && ok "9g: profiles naming another backend (and an unknown backend) are refused" || bad "9g: a conflicting profile was accepted"
# build.sh / run.sh / validate.sh derive the profile through ONE helper (deterministic; override honoured)
[ "$(l3_backend_profile LAMMPS cuda)" = cuda ] && [ "$(l3_backend_profile NEKRS cuda hypregpu)" = hypregpu.cuda ] && [ "$(l3_backend_profile NEKRS hip cpucoarse)" = cpucoarse.hip ] \
    && ok "9h: l3_backend_profile derives <backend> / <variant>.<backend>" || bad "9h: $(l3_backend_profile LAMMPS cuda) $(l3_backend_profile NEKRS cuda hypregpu)"
[ "$(HPCPERF_LAMMPS_PROFILE=cuda-mytest l3_backend_profile LAMMPS cuda)" = cuda-mytest ] && ok "9i: HPCPERF_<APP>_PROFILE override is honoured (and 9f/9g check it still names the backend)" || bad "9i: override ignored"
# no silent fallback to a legacy shared install
mkdir -p "$ISO/.deps/level3/app/install"; printf 'schema=l3-2\napplication=app\nbackend=cuda arch=sm_100\n' > "$ISO/.deps/level3/app/install/.hpcperf-l3-fingerprint"
inst="$(paths_of app cuda cuda | awk '{print $3}')"; note="$( ( L3_R="$ISO"; l3_paths_profile app cuda cuda 2>&1 >/dev/null ) )"
[ "$inst" = "$ISO/.deps/level3/app/cuda/install" ] && echo "$note" | /usr/bin/grep -q 'legacy shared install .* not used' && ok "9j: a legacy .deps/level3/app/install is reported and NOT used (profile install stays .deps/level3/app/cuda/install)" || bad "9j: legacy install handling ($inst; $note)"
rc=0; l3_fingerprint_expect_backend "$ISO/.deps/level3/app/cuda/install" cuda >/dev/null 2>&1 || rc=$?
[ "$rc" -ne 0 ] && ok "9k: run-side gate refuses an unbuilt profile even though a fingerprinted legacy install exists (no fallback)" || bad "9k: unbuilt profile accepted"
# fingerprints of two profiles never overwrite each other; backend recorded == profile backend
fpc="$(l3_fingerprint_text app sha cuda deps opts gam 2>/dev/null)"; fph="$(l3_fingerprint_text app sha hip deps opts gam 2>/dev/null)"
l3_fingerprint_write "$ISO/.deps/level3/app/cuda/install" "$fpc"; l3_fingerprint_write "$ISO/.deps/level3/app/hip/install" "$fph"
[ -f "$ISO/.deps/level3/app/cuda/install/.hpcperf-l3-fingerprint" ] && [ -f "$ISO/.deps/level3/app/hip/install/.hpcperf-l3-fingerprint" ] \
    && ! cmp -s "$ISO/.deps/level3/app/cuda/install/.hpcperf-l3-fingerprint" "$ISO/.deps/level3/app/hip/install/.hpcperf-l3-fingerprint" \
    && ok "9l: cuda and hip fingerprints live in their own profile installs and differ" || bad "9l: fingerprints missing or identical"
rc=0; l3_fingerprint_expect_backend "$ISO/.deps/level3/app/cuda/install" cuda >/dev/null 2>&1 || rc=$?; rc2=0; l3_fingerprint_expect_backend "$ISO/.deps/level3/app/cuda/install" hip >/dev/null 2>&1 || rc2=$?
[ "$rc" -eq 0 ] && [ "$rc2" -ne 0 ] && ok "9m: a CUDA fingerprint serves a CUDA request and is refused for HIP" || bad "9m: rc=$rc rc2=$rc2"
rc=0; l3_fingerprint_check "$ISO/.deps/level3/app/hip/install" "$fpc" >/dev/null 2>&1 || rc=$?
[ "$rc" -ne 0 ] && ok "9n: build-side fingerprint check refuses a CUDA configuration against the hip profile's install" || bad "9n: mismatch accepted"
# static: every active Level 3 wrapper goes through l3_paths_profile; no legacy helper; no unprofiled generated path
L3D="$R/level3"; miss=""; leg=""; hard=""
for a in lammps sparta warpx specfem3d nekrs exaca; do for s in build run validate; do
    /usr/bin/grep -q "l3_paths_profile $a \"\$PROFILE\" \"\$MODEL\"" "$L3D/$a/$s.sh" || miss="$miss $a/$s.sh"; done; done
for a in nyx cp2k qmcpack dftfe; do for s in build run; do /usr/bin/grep -q 'l3_paths_profile '"$a"' "$PROFILE" ' "$L3D/$a/$s.sh" || miss="$miss $a/$s.sh"; done; done
[ -z "$miss" ] && ok "9o: build/run/validate of the six migrated apps (and build/run of nyx/cp2k/qmcpack/dftfe) call l3_paths_profile with the backend" || bad "9o: missing profile call in:$miss"
leg="$(/usr/bin/grep -l -w 'l3_paths' "$L3D"/*/build.sh "$L3D"/*/run.sh "$L3D"/*/validate.sh 2>/dev/null || true)"
[ -z "$leg" ] && ! /usr/bin/grep -q '^l3_paths()' "$L3D/tools/l3_common.sh" && ok "9p: the legacy shared-path helper l3_paths has no definition and no caller" || bad "9p: legacy l3_paths still present: $leg"
hard="$(/usr/bin/grep -n -E '\.deps/level3/[a-z0-9]+/(install|logs|src|build)\b' "$L3D"/*/build.sh "$L3D"/*/run.sh "$L3D"/*/validate.sh 2>/dev/null || true)"
[ -z "$hard" ] && ok "9q: no active wrapper hardcodes an unprofiled .deps/level3/<app>/{install,logs,src,build}" || bad "9q: unprofiled generated-state path: $hard"
nk=0; for s in build run validate; do /usr/bin/grep -q 'hypregpu is CUDA-only' "$L3D/nekrs/$s.sh" || nk=1; done
[ "$nk" -eq 0 ] && ok "9r: nekRS build/run/validate refuse the undefined hypregpu x HIP combination explicitly" || bad "9r: nekRS hypregpu x HIP refusal missing"
src=0; for a in lammps sparta warpx specfem3d nekrs exaca; do /usr/bin/grep -q 'l3_require_materialized "\$HERE"' "$L3D/$a/build.sh" || src=1; done
[ "$src" -eq 0 ] && ok "9s: the migrated build scripts still read application source from the ONE materialized level3/<app>/src (not a per-backend copy)" || bad "9s: a build.sh no longer requires the materialized tree"

# 10. local scratch outside the worktree (CP2K toolchain, QMCPACK LLVM) is still workspace x source x profile specific
W1="$TMP/ws-one"; W2="$TMP/ws-two"; mkdir -p "$W1" "$W2"; TREE_A="d877d2d4de4643ce90c1ced68b0cc5d18e389f5055efbf3309cbeba059ef6fa2"; TREE_B="6bde03184c09b236179b2ff4c8200e8489666ed68aebbbdc11462bfb050b1c0e"
sd() { ( L3_R="$1"; l3_local_scratch_dir "$2" "$3" "$4" ); }
s1="$(sd "$W1" cp2k-toolchain "$TREE_A" cuda132-gcc142-ompi5010)"; s2="$(sd "$W2" cp2k-toolchain "$TREE_A" cuda132-gcc142-ompi5010)"
[ -n "$s1" ] && [ "$s1" != "$s2" ] && ok "10a: two worktrees, same source and profile -> different scratch directories" || bad "10a: $s1 vs $s2"
[ "$(sd "$W1" cp2k-toolchain "$TREE_A" cuda132-gcc142-ompi5010)" = "$s1" ] && ok "10b: same worktree/source/profile -> the same directory every time" || bad "10b: unstable"
[ "$(sd "$W1" cp2k-toolchain "$TREE_B" cuda132-gcc142-ompi5010)" != "$s1" ] && ok "10c: a different frozen source tree -> a different directory" || bad "10c"
[ "$(sd "$W1" cp2k-toolchain "$TREE_A" cuda132-gcc133-ompi5010)" != "$s1" ] && ok "10d: a different profile -> a different directory" || bad "10d"
case "$s1" in "${TMPDIR:-/tmp}/hpcperf-l3-scratch/cp2k-toolchain/"*"/${TREE_A:0:12}/cuda132-gcc142-ompi5010") ok "10e: layout <base>/hpcperf-l3-scratch/<component>/<root12>/<source12>/<profile>";; *) bad "10e: $s1";; esac
case "$s1" in *"$W1"*|*"ws-one"*) bad "10f: the workspace path leaks into the scratch path";; *) ok "10f: only a hash of the workspace root appears in the path";; esac
[ "$(HPCPERF_L3_SCRATCH_BASE="$TMP/base" sd "$W1" cp2k-toolchain "$TREE_A" p-cuda)" = "$TMP/base/hpcperf-l3-scratch/cp2k-toolchain/$(printf '%s' "$(realpath "$W1")" | sha256sum | cut -c1-12)/${TREE_A:0:12}/p-cuda" ] && ok "10g: HPCPERF_L3_SCRATCH_BASE relocates the base; root hash = sha256(realpath root)[:12]" || bad "10g"
rc=0; sd "$W1" cp2k-toolchain "" cuda >/dev/null 2>&1 || rc=$?; [ "$rc" -ne 0 ] && ok "10h: a missing source identity is refused" || bad "10h"
/usr/bin/grep -q 'l3_local_scratch_dir cp2k-toolchain "\$TREE_SHA" "\$PROFILE"' "$R/level3/cp2k/build.sh" && /usr/bin/grep -q 'HPCPERF_CP2K_TOOLCHAIN_SCRATCH:-' "$R/level3/cp2k/build.sh" \
    && ok "10i: cp2k/build.sh derives its toolchain scratch from the helper and keeps the explicit override" || bad "10i"
/usr/bin/grep -q 'l3_local_scratch_dir qmcpack-llvm "\$EXPECT_SHA" "\$PROFILE"' "$R/level3/qmcpack/toolchain/build_llvm.sh" && /usr/bin/grep -q 'HPCPERF_LLVM_SCRATCH:-' "$R/level3/qmcpack/toolchain/build_llvm.sh" \
    && ok "10j: qmcpack/toolchain/build_llvm.sh derives its scratch from the helper and keeps the explicit override" || bad "10j"
leg="$(/usr/bin/grep -l 'hpcperf-l3-b2-scratch' "$R"/level3/*/build.sh "$R"/level3/*/run.sh "$R"/level3/*/validate.sh "$R"/level3/*/toolchain/*.sh "$R"/level3/tools/*.sh 2>/dev/null | /usr/bin/grep -v -E ':[0-9]+:#' || true)"
if [ -z "$leg" ]; then ok "10k: no active script still uses the shared /tmp/hpcperf-l3-b2-scratch location as a default"; else
    # comments describing the legacy location are fine; a code line is not
    code="$(/usr/bin/grep -n 'hpcperf-l3-b2-scratch' $leg | /usr/bin/grep -v -E ':[0-9]+:\s*#' || true)"
    [ -z "$code" ] && ok "10k: the shared /tmp/hpcperf-l3-b2-scratch location survives only in comments (legacy note)" || bad "10k: code still uses the shared scratch: $code"; fi

echo
echo "test_l3_infra: $pass passed, $failn failed"
[ "$failn" -eq 0 ]
