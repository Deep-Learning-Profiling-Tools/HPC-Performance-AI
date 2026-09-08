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

# 7. l3_run_recorded: records the real exit code of a queue step and never aborts the caller
rcf="$TMP/rc.txt"; steps=0
( set -e; for r in 0 3 0; do l3_run_recorded "$rcf" "step$r" -- bash -c "exit $r"; echo step >> "$TMP/steps"; done )
[ "$(wc -l < "$TMP/steps")" -eq 3 ] && ok "7a: a queue under set -e continues past a step that exits 3" || bad "7a: queue stopped after $(wc -l < "$TMP/steps") step(s)"
[ "$(tr '\n' ' ' < "$rcf")" = "step0 0 step3 3 step0 0 " ] && ok "7b: exit codes recorded verbatim (0 3 0)" || bad "7b: recorded '$(tr '\n' ' ' < "$rcf")'"

echo
echo "test_l3_infra: $pass passed, $failn failed"
[ "$failn" -eq 0 ]
