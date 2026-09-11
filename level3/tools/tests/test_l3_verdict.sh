#!/bin/bash
# Verdict classes of level3/tools/l3_verdict.py (CPU only): exit code 3 (Nyx heat/cool I_R_CHECK_PENDING)
# and 4 (UNSUPPORTED_LAYOUT) must stay their own classes in every summary -- never PASS -- and an exit
# code that contradicts the log is FAIL. The queue-continuation half (a step exiting 3 does not abort
# a queue) is test 7 of test_l3_infra.sh (l3_run_recorded).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../../.." && pwd)"
V="$R/level3/tools/l3_verdict.py"
pass=0; failn=0
ok()  { echo "ok   $*"; pass=$((pass+1)); }
bad() { echo "FAIL $*"; failn=$((failn+1)); }
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mk() { printf '%s\n' "$2" > "$TMP/$1"; }
mk pass.log "hpcperf-launch: audit summary: 2 verified, 0 mismatch, 0 unverified (of 2 ranks)
Nyx CUDA validation (2 GPU, cases: minisb lya_adiabatic; crit): PASS"
mk pend.log "    lya_heatcool.np1.vs-CPU: diagnostic field(s) I_R reported, NOT accepted -> case verdict I_R_CHECK_PENDING
Nyx CUDA validation (1 GPU, cases: lya_heatcool; crit): STATE_AND_PARTICLES_PASS; I_R_CHECK_PENDING [x] -- heat/cool acceptance incomplete, not a full regression PASS"
mk unsup.log "    minisb.np2.vs-1GPU: UNSUPPORTED_LAYOUT -- a different but legal box layout is outside this validator's scope
Nyx CUDA validation (2 GPU, cases: minisb; crit): UNSUPPORTED_LAYOUT [minisb.np2.vs-1GPU] -- plotfile comparison not performed for these pairs; not a PASS"
mk fail.log "validate.sh: FAIL -- minisb.np2.vs-1GPU: plotfile comparison (rel_tol 2e-10): DISAGREE
Nyx CUDA validation (2 GPU, cases: minisb): FAIL"
mk noverdict.log "hpcperf-launch: audit summary: 0 verified, 0 mismatch, 1 unverified (of 1 ranks)
some output that never reaches a verdict line"
mk firstbatch.log "LAMMPS CUDA validation (4 GPU, in.lj thermo vs reference log): PASS"
c() { python3 "$V" classify --rc "$1" --log "$TMP/$2"; }
[ "$(c 0 pass.log)" = PASS ] && ok "1: rc 0 + ': PASS' line -> PASS" || bad "1: $(c 0 pass.log)"
[ "$(c 0 firstbatch.log)" = PASS ] && ok "2: first-batch style PASS line -> PASS" || bad "2: $(c 0 firstbatch.log)"
[ "$(c 3 pend.log)" = PENDING ] && ok "3: rc 3 + I_R_CHECK_PENDING -> PENDING (its own class)" || bad "3: $(c 3 pend.log)"
[ "$(c 0 pend.log)" = FAIL ] && ok "4: rc 0 with a PENDING line is inconsistent -> FAIL, never PASS" || bad "4: $(c 0 pend.log)"
[ "$(c 3 pass.log)" = FAIL ] && ok "5: rc 3 with a PASS line is inconsistent -> FAIL" || bad "5: $(c 3 pass.log)"
[ "$(c 4 unsup.log)" = UNSUPPORTED_LAYOUT ] && ok "6: rc 4 + UNSUPPORTED_LAYOUT -> UNSUPPORTED_LAYOUT" || bad "6: $(c 4 unsup.log)"
[ "$(c 1 fail.log)" = FAIL ] && ok "7: rc 1 -> FAIL" || bad "7: $(c 1 fail.log)"
[ "$(c 0 noverdict.log)" = FAIL ] && ok "8: rc 0 without a verdict line -> FAIL" || bad "8: $(c 0 noverdict.log)"
[ "$(c 124 pass.log)" = FAIL ] && ok "9: timeout (124) -> FAIL even with a stale PASS line" || bad "9: $(c 124 pass.log)"
[ "$(c 0 absent.log)" = MISSING ] && ok "10: missing log -> MISSING" || bad "10: $(c 0 absent.log)"
# summary: every class counted separately; the PENDING row is never marked PASS; unverified audits listed
printf 'pass.log 0\npend.log 3\nunsup.log 4\nfail.log 1\nfirstbatch.log 0\n' > "$TMP/rc.txt"
python3 "$V" summary --rc-file "$TMP/rc.txt" --log "$TMP"/{pass,pend,unsup,fail,noverdict,firstbatch}.log > "$TMP/sum.md" 2>&1; rc=$?
[ "$rc" -eq 0 ] && /usr/bin/grep -q 'PASS=2, PENDING=1, UNSUPPORTED_LAYOUT=1, FAIL=1, REFUSED=0, BUILD_FAIL=0, MISSING=1' "$TMP/sum.md" && ok "11: summary counts PASS/PENDING/UNSUPPORTED_LAYOUT/FAIL/REFUSED/BUILD_FAIL/MISSING separately" || bad "11: rc=$rc: $(tail -3 "$TMP/sum.md")"
/usr/bin/grep -E '^\| pend\.log ' "$TMP/sum.md" | /usr/bin/grep -q '\*\*PENDING\*\*' && ! /usr/bin/grep -E '^\| pend\.log ' "$TMP/sum.md" | /usr/bin/grep -q '\*\*PASS\*\*' && ok "12: the rc-3 row is labelled PENDING, not PASS" || bad "12: $(/usr/bin/grep -E '^\| pend\.log ' "$TMP/sum.md")"
/usr/bin/grep -q 'unverified ranks.*noverdict.log \[0/0/1\]' "$TMP/sum.md" && ok "13: unverified launcher audit listed as a binding-evidence gap" || bad "13: $(/usr/bin/grep 'unverified ranks' "$TMP/sum.md")"
echo
# workspace-integrity / build layers (tools/validate_workspace.sh): never PASS, never PENDING
printf 'validate_workspace: REFUSED -- the workspace violates the contract\n' > "$TMP/ref.log"
[ "$(python3 "$V" classify --rc 6 --log "$TMP/ref.log")" = REFUSED ] && ok "rc 6 + REFUSED line -> REFUSED" || bad "rc 6 REFUSED"
[ "$(python3 "$V" classify --rc 3 --log "$TMP/ref.log")" = FAIL ] && ok "rc 3 with a REFUSED line is FAIL, never PENDING" || bad "rc 3 refused"
[ "$(python3 "$V" classify --rc 6 --log "$TMP/pass.log")" = FAIL ] && ok "rc 6 with a PASS line is FAIL (contradiction)" || bad "rc 6 pass line"
printf 'validate_workspace: BUILD_FAIL (build layer, exit 7)\n' > "$TMP/bf.log"
[ "$(python3 "$V" classify --rc 7 --log "$TMP/bf.log")" = BUILD_FAIL ] && ok "rc 7 + BUILD_FAIL line -> BUILD_FAIL" || bad "rc 7"
[ "$(python3 "$V" classify --rc 0 --log "$TMP/ref.log")" = FAIL ] && ok "rc 0 with a REFUSED line is FAIL" || bad "rc 0 refused"
echo "test_l3_verdict: $pass passed, $failn failed"
[ "$failn" -eq 0 ]
