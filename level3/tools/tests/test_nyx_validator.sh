#!/bin/bash
# Negative/positive tests for the Nyx validator (CPU only, no GPU, no application, no build).
#
# Part A: nyx_fcompare_check.py against STUB fcompare/fextrema tools whose output is injected
#         (the reviewer's cases: rc=1 with Ne abs=1 rel=inf; rc=1 with a missing variable and the
#         "not all variables present" warning; rc=1 with a NaN row -- all of which the previous awk
#         parser accepted) plus normal control, legal zero/zero reference, zero reference vs nonzero
#         test, duplicate row, truncated table, header variable-set mismatch, diagnostic field.
# Part B: the REAL level3/nyx/validate.sh call chain in OFFLINE mode (HPCPERF_NYX_OFFLINE=1) on
#         synthetic run directories, with the same stub tools: the outer verdict/exit code must
#         follow the comparator (FAIL on every injected defect, PASS on the control, exit 3 for a
#         diagnostic field).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../../.." && pwd)"
CHK="$R/level3/nyx/nyx_fcompare_check.py"
pass=0; failn=0
ok()  { echo "ok   $*"; pass=$((pass+1)); }
bad() { echo "FAIL $*"; failn=$((failn+1)); }
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP" "$R/build/level3/nyx/__selftest__"*' EXIT
VARS="density xmom ymom zmom rho_E rho_e Temp Ne phi_grav grav_x grav_y grav_z I_R pressure particle_count particle_mass_density"

# --- synthetic plotfile: an AMReX Header (single level, 8 grids) --------------------------------
mk_header() { # mk_header <plotdir> [time] [vars...]; MK_BOXES="xlo xhi ylo yhi zlo zhi" lines (physical) override the default 2x2x2 layout
    local d=$1 t=${2:-2.86455272510386e-06}; shift; [ $# -gt 0 ] && shift; local vars=("$@"); [ ${#vars[@]} -gt 0 ] || vars=($VARS)
    local boxes=() l x0 x1 y0 y1 z0 z1
    if [ -n "${MK_BOXES:-}" ]; then while IFS= read -r l; do [ -n "$l" ] && boxes+=("$l"); done <<< "$MK_BOXES"
    else for b in "0 4" "4 8"; do for c in "0 4" "4 8"; do for e in "0 4" "4 8"; do boxes+=("$b $c $e"); done; done; done; fi
    mkdir -p "$d/Level_0" "$d/DM"
    { echo "HyperCLaw-V1.1"; echo "${#vars[@]}"; printf '%s\n' "${vars[@]}"; echo 3; echo "$t"; echo 0; echo "0 0 0 "; echo "8 8 8 "; echo ""
      echo "((0,0,0) (31,31,31) (0,0,0)) "; echo "10 "; echo "0.25 0.25 0.25 "; echo 0; echo 0
      echo "0 ${#boxes[@]} $t"; echo 10
      for l in "${boxes[@]}"; do read -r x0 x1 y0 y1 z0 z1 <<< "$l"; echo "$x0 $x1"; echo "$y0 $y1"; echo "$z0 $z1"; done
      echo "Level_0/Cell"; } > "$d/Header"
    # DM particle header: version, dim, nreal, names, nint, names, is_checkpoint, nparticles
    { echo "Version_Two_Dot_Zero_double"; echo 3; echo 4; printf 'mass\nxvel\nyvel\nzvel\n'; echo 0; echo 0; echo 32768; } > "$d/DM/Header"
}
# stub fextrema: prints "name min max" for every variable of the plotfile Header; VALUE can be overridden per var
mk_tools() { # mk_tools <dir> ; env FEXTREMA_OVERRIDE="Ne:0:0,rho_e:nan:1"
    local t=$1; mkdir -p "$t"
    cat > "$t/amrex_fextrema" <<'EOF'
#!/bin/bash
plt=$1; n=$(sed -n 2p "$plt/Header"); i=0
sed -n "3,$((2+n))p" "$plt/Header" | while read -r v; do lo=1.0; hi=2.0
  IFS=, read -ra ov <<< "${FEXTREMA_OVERRIDE:-}"; for o in "${ov[@]}"; do IFS=: read -r name mn mx <<< "$o"; [ "$name" = "$v" ] && { lo=$mn; hi=$mx; }; done
  printf ' %-24s %24s %24s\n' "$v" "$lo" "$hi"; done
EOF
    # stub fcompare: prints the canned table from $FCOMPARE_TABLE (file) and exits $FCOMPARE_RC
    cat > "$t/amrex_fcompare" <<'EOF'
#!/bin/bash
cat "$FCOMPARE_TABLE"; exit "${FCOMPARE_RC:-0}"
EOF
    printf '#!/bin/bash\necho stub\n' > "$t/amrex_fvolumesum"; printf '#!/bin/bash\necho stub\n' > "$t/particle_compare"
    chmod +x "$t"/*
}
table() { # table <rc-independent canned fcompare output>: one row per var with given abs/rel; overrides "var=abs:rel"
    local extra="$1"; shift
    { echo ""; echo "            variable name             absolute error            relative error"; echo "                                         (||A - B||)         (||A - B||/||A||)"
      echo " ----------------------------------------------------------------------------"; echo " level = 0"
      for v in $VARS; do a=1e-14; r=1e-14
          for o in "$@"; do IFS='=' read -r name vals <<< "$o"; [ "$name" = "$v" ] && { IFS=: read -r a r <<< "$vals"; }; done
          case "$a" in SKIP) continue;; MSG) printf ' %-24s  %-50s\n' "$v" "$r"; continue;; esac
          printf ' %-24s  %24s  %24s\n' "$v" "$a" "$r"; done
      [ -n "$extra" ] && echo "$extra"; true; }
}
T="$TMP/tools"; mk_tools "$T"; export FEXTREMA_OVERRIDE="Ne:0:0"   # Ne identically zero in the reference (as in the real deck)
REF="$TMP/ref/plt00010"; TST="$TMP/tst/plt00010"; mk_header "$REF"; mk_header "$TST"
run_chk() { # run_chk <table-file> <rc> [extra args]
    local tbl=$1 rc=$2; shift 2
    FCOMPARE_TABLE="$tbl" FCOMPARE_RC="$rc" python3 "$CHK" "$REF" "$TST" --rel_tol 2e-10 --fcompare "$T/amrex_fcompare" --fextrema "$T/amrex_fextrema" --out "$TMP/out.txt" "$@" >"$TMP/chk.out" 2>&1
}
# A1 control: everything tiny, Ne zero/zero (abs 0 rel 0 as fcompare prints), rc 0 -> AGREE
table "" "Ne=0:0" > "$TMP/t1"; run_chk "$TMP/t1" 0; [ $? -eq 0 ] && ok "A1: control (all within 2e-10, Ne zero/zero) AGREE" || bad "A1: control rejected: $(cat "$TMP/chk.out")"
# A2 reviewer case: rc=1, Ne abs=1 rel=inf (reference norm zero -> inf) -> must FAIL (zero-reference abs rule)
table "" "Ne=1:inf" > "$TMP/t2"; run_chk "$TMP/t2" 1; rc=$?; [ $rc -eq 1 ] && ok "A2: Ne abs=1 rel=inf on a zero reference -> DISAGREE (rc 1)" || bad "A2: rc=$rc: $(cat "$TMP/chk.out")"
# A3 reviewer case: rc=1, a variable missing + fcompare's warning line -> STRUCTURAL
table " WARNING: not all variables present in both files" "Temp=SKIP:x" > "$TMP/t3"; run_chk "$TMP/t3" 1; rc=$?; [ $rc -eq 2 ] && ok "A3: missing variable + WARNING -> STRUCTURAL (rc 2)" || bad "A3: rc=$rc: $(cat "$TMP/chk.out")"
# A4 reviewer case: rc=1, NaN message row -> STRUCTURAL
table "" "rho_e=MSG:< NaN present in B > " > "$TMP/t4"; run_chk "$TMP/t4" 1; rc=$?; [ $rc -eq 2 ] && ok "A4: '< NaN present in B >' row -> STRUCTURAL" || bad "A4: rc=$rc: $(cat "$TMP/chk.out")"
# A5 numeric nan in the rel column on a non-zero reference -> not finite -> DISAGREE or STRUCTURAL, never AGREE
table "" "Temp=1e-3:nan" > "$TMP/t5"; run_chk "$TMP/t5" 1; rc=$?; [ $rc -ne 0 ] && ok "A5: rel=nan on a non-zero reference rejected (rc $rc)" || bad "A5: rel=nan accepted"
# A6 inf rel on a NON-zero reference (fextrema says Temp is non-zero) -> DISAGREE
table "" "Temp=1e-3:inf" > "$TMP/t6"; run_chk "$TMP/t6" 1; rc=$?; [ $rc -eq 1 ] && ok "A6: rel=inf on a non-zero reference -> DISAGREE" || bad "A6: rc=$rc: $(cat "$TMP/chk.out")"
# A7 zero reference, nonzero test, fcompare says agree (rc 0, abs tiny) -> abs rule 0 -> DISAGREE (and parser/tool inconsistency is STRUCTURAL) => non-zero
table "" "Ne=1e-20:0" > "$TMP/t7"; run_chk "$TMP/t7" 0; rc=$?; [ $rc -ne 0 ] && ok "A7: zero reference vs 1e-20 test -> rejected (rc $rc, exact rule)" || bad "A7: accepted"
# A8 duplicate row -> STRUCTURAL
{ table "" ; printf ' %-24s  %24s  %24s\n' Temp 1e-14 1e-14; } > "$TMP/t8"; run_chk "$TMP/t8" 0; rc=$?; [ $rc -eq 2 ] && ok "A8: duplicate row -> STRUCTURAL" || bad "A8: rc=$rc"
# A9 truncated table (last 3 rows missing) -> STRUCTURAL
table "" | head -n -3 > "$TMP/t9"; run_chk "$TMP/t9" 0; rc=$?; [ $rc -eq 2 ] && ok "A9: truncated table -> STRUCTURAL" || bad "A9: rc=$rc"
# A10 header variable-set mismatch (test plotfile lacks pressure) -> STRUCTURAL before fcompare
mk_header "$TMP/tst2/plt00010" 2.86455272510386e-06 $(echo $VARS | sed 's/ pressure//'); table "" > "$TMP/t10"
FCOMPARE_TABLE="$TMP/t10" FCOMPARE_RC=0 python3 "$CHK" "$REF" "$TMP/tst2/plt00010" --rel_tol 2e-10 --fcompare "$T/amrex_fcompare" --fextrema "$T/amrex_fextrema" >"$TMP/chk.out" 2>&1; rc=$?
[ $rc -eq 2 ] && ok "A10: header variable set differs -> STRUCTURAL" || bad "A10: rc=$rc"
# A11 header time mismatch -> STRUCTURAL
mk_header "$TMP/tst3/plt00010" 3.0e-06; FCOMPARE_TABLE="$TMP/t1" FCOMPARE_RC=0 python3 "$CHK" "$REF" "$TMP/tst3/plt00010" --rel_tol 2e-10 --fcompare "$T/amrex_fcompare" --fextrema "$T/amrex_fextrema" >"$TMP/chk.out" 2>&1; rc=$?
[ $rc -eq 2 ] && ok "A11: simulation time differs -> STRUCTURAL" || bad "A11: rc=$rc"
# A12 raw non-finite value in the reference (fextrema max=nan) -> STRUCTURAL even if fcompare is happy
FEXTREMA_OVERRIDE="Ne:0:0,rho_e:1:nan" run_chk "$TMP/t1" 0; rc=$?; [ $rc -eq 2 ] && ok "A12: non-finite raw value (fextrema) -> STRUCTURAL" || bad "A12: rc=$rc"
# A13 diagnostic field over tolerance is reported, others gate: rc from fcompare 1 (it counts I_R) -> AGREE
table "" "Ne=0:0" "I_R=0.29:1.3" > "$TMP/t13"; run_chk "$TMP/t13" 1 --diagnostic I_R; rc=$?; [ $rc -eq 0 ] && /usr/bin/grep -q 'diagnostic=\[.I_R@L0: abs=2.9' "$TMP/chk.out" && ok "A13: diagnostic I_R (rel 1.3) reported, state gated -> AGREE" || bad "A13: rc=$rc: $(cat "$TMP/chk.out")"
# A14 the same I_R row WITHOUT the diagnostic flag -> DISAGREE
run_chk "$TMP/t13" 1; rc=$?; [ $rc -eq 1 ] && ok "A14: I_R gated when not declared diagnostic -> DISAGREE" || bad "A14: rc=$rc"
# A15 a genuine over-tolerance state variable -> DISAGREE
table "" "Ne=0:0" "Temp=1e-3:5e-9" > "$TMP/t15"; run_chk "$TMP/t15" 1; rc=$?; [ $rc -eq 1 ] && ok "A15: Temp rel 5e-9 > 2e-10 -> DISAGREE" || bad "A15: rc=$rc"
# A16 parser/tool inconsistency: table says agree but fcompare rc=1 and no diagnostic -> STRUCTURAL
run_chk "$TMP/t1" 1; rc=$?; [ $rc -eq 2 ] && ok "A16: AGREE table but fcompare rc=1 -> STRUCTURAL (inconsistency)" || bad "A16: rc=$rc"
# --- box layouts: legal re-blocking is UNSUPPORTED_LAYOUT (exit 4, explicit), real defects are STRUCTURAL ----
chk_layout() { # chk_layout <plotdir with another layout> -> rc, output in chk.out (control table, rc 0)
    FCOMPARE_TABLE="$TMP/t1" FCOMPARE_RC=0 python3 "$CHK" "$REF" "$1" --rel_tol 2e-10 --fcompare "$T/amrex_fcompare" --fextrema "$T/amrex_fextrema" >"$TMP/chk.out" 2>&1
}
REBLOCK=$'0 8 0 4 0 4\n0 8 0 4 4 8\n0 8 4 8 0 4\n0 8 4 8 4 8'                       # 4 boxes, same 32^3 cells as the 8-box layout
DEFAULT8="$(for b in "0 4" "4 8"; do for c in "0 4" "4 8"; do for e in "0 4" "4 8"; do echo "$b $c $e"; done; done; done)"
MK_BOXES="$REBLOCK" mk_header "$TMP/lay1/plt00010"; chk_layout "$TMP/lay1/plt00010"; rc=$?
[ $rc -eq 4 ] && /usr/bin/grep -q 'UNSUPPORTED_LAYOUT.*legal re-blocking' "$TMP/chk.out" && ok "A17: same cells, different partition (4 vs 8 boxes) -> UNSUPPORTED_LAYOUT (exit 4), reported explicitly" || bad "A17: rc=$rc: $(cat "$TMP/chk.out")"
MK_BOXES="$(echo "$DEFAULT8" | head -n 7)" mk_header "$TMP/lay2/plt00010"; chk_layout "$TMP/lay2/plt00010"; rc=$?
[ $rc -eq 2 ] && /usr/bin/grep -q 'coverage deficit' "$TMP/chk.out" && ok "A18: a missing box (7 of 8, level 0 not covered) -> STRUCTURAL" || bad "A18: rc=$rc: $(cat "$TMP/chk.out")"
MK_BOXES="$(echo "$DEFAULT8" | head -n 7; echo "$DEFAULT8" | head -n 1)" mk_header "$TMP/lay3/plt00010"; chk_layout "$TMP/lay3/plt00010"; rc=$?
[ $rc -eq 2 ] && /usr/bin/grep -q 'overlap' "$TMP/chk.out" && ok "A19: overlapping boxes -> STRUCTURAL" || bad "A19: rc=$rc: $(cat "$TMP/chk.out")"
MK_BOXES="$(echo "$DEFAULT8" | head -n 7; echo "4 12 4 8 4 8")" mk_header "$TMP/lay4/plt00010"; chk_layout "$TMP/lay4/plt00010"; rc=$?
[ $rc -eq 2 ] && /usr/bin/grep -q 'outside the level domain' "$TMP/chk.out" && ok "A20: a box outside the domain -> STRUCTURAL" || bad "A20: rc=$rc: $(cat "$TMP/chk.out")"
MK_BOXES="$(echo "$DEFAULT8" | tac)" mk_header "$TMP/lay5/plt00010"; chk_layout "$TMP/lay5/plt00010"; rc=$?
[ $rc -eq 0 ] && /usr/bin/grep -q 'RESULT: AGREE' "$TMP/chk.out" && ok "A21: the same boxes listed in another order -> identical layout, compared normally" || bad "A21: rc=$rc: $(cat "$TMP/chk.out")"

# --- Part B: real validate.sh chain, offline, synthetic run dirs, stub tools -----------------------
VAL="$R/level3/nyx/validate.sh"
if [ -f "$VAL" ]; then
    SUB="run.selftest-$$"; PROF="__selftest__"; export HPCPERF_L3_RUN_SUBDIR="$SUB" HPCPERF_NYX_PROFILE="${PROF}-adiabatic" HPCPERF_NYX_CPU_PROFILE="${PROF}-cpu"
    GR="$R/build/level3/nyx/${PROF}-adiabatic/$SUB"; CR="$R/build/level3/nyx/${PROF}-cpu/$SUB"
    mk_run() { # mk_run <dir> [dm particle count]
        local d=$1 np=${2:-32768}; mkdir -p "$d"; mk_header "$d/plt00000" 0.0; mk_header "$d/plt00010"
        sed -i "s/^32768$/$np/" "$d/plt00000/DM/Header" "$d/plt00010/DM/Header"
        printf '#  nstep time dt z a\n%s\n' "$(for s in $(seq 0 10); do echo "       $s 1.7e7 2.6e5 100 0.0099"; done)" > "$d/runlog"
        printf 'run_id=selftest\nexit_code=0\nbinary_sha256=deadbeef\ndeck_sha256=cafe\nsteps=10\n' > "$d/run_manifest.txt"
    }
    # MiniSB's IC count (Exec/MiniSB/ic_sb_32.ascii) is 32686 when the upstream checkout is present; the count check is skipped otherwise
    NP_SB="$(head -1 "$R/_upstream/level3/Nyx/Exec/MiniSB/ic_sb_32.ascii" 2>/dev/null | tr -d ' ')"; NP_SB="${NP_SB:-32768}"
    for d in "$GR/minisb.smoke.np1" "$GR/minisb.smoke.np1.rerun" "$GR/minisb.smoke.np2" "$CR/minisb.smoke.np1"; do mk_run "$d" "$NP_SB"; done
    printf '#!/bin/bash\necho " density  1.0e+10"\n' > "$T/amrex_fvolumesum"; chmod +x "$T/amrex_fvolumesum"
    export HPCPERF_NYX_OFFLINE=1 HPCPERF_NYX_TOOLS_DIR="$T" HPCPERF_NYX_SKIP_PARTICLES=1 HPCPERF_NYX_CASES=minisb HPCPERF_NYX_REPORT_DIR="$TMP/reports"
    chain() { FCOMPARE_TABLE="$1" FCOMPARE_RC="$2" HPCPERF_GPUS="$3" bash "$VAL" CUDA > "$TMP/val.out" 2>&1; echo $?; }
    # ic_count for minisb reads the upstream IC file; if the checkout is absent the count check is skipped by the script (empty want)
    rc=$(chain "$TMP/t1" 0 2); /usr/bin/grep -q ': PASS$' "$TMP/val.out" && [ "$rc" -eq 0 ] && ok "B1: validate.sh offline chain, control -> PASS (exit 0)" || bad "B1: rc=$rc: $(tail -3 "$TMP/val.out")"
    rc=$(chain "$TMP/t2" 1 2); [ "$rc" -eq 1 ] && /usr/bin/grep -q 'FAIL' "$TMP/val.out" && ok "B2: chain, Ne abs=1 rel=inf -> FAIL (exit 1)" || bad "B2: rc=$rc: $(tail -3 "$TMP/val.out")"
    rc=$(chain "$TMP/t3" 1 2); [ "$rc" -eq 1 ] && /usr/bin/grep -q 'STRUCTURAL' "$TMP/val.out" && ok "B3: chain, missing variable warning -> FAIL/STRUCTURAL" || bad "B3: rc=$rc: $(tail -3 "$TMP/val.out")"
    rc=$(chain "$TMP/t4" 1 2); [ "$rc" -eq 1 ] && /usr/bin/grep -q 'STRUCTURAL' "$TMP/val.out" && ok "B4: chain, NaN row -> FAIL/STRUCTURAL" || bad "B4: rc=$rc: $(tail -3 "$TMP/val.out")"
    rc=$(chain "$TMP/t9" 0 2); [ "$rc" -eq 1 ] && ok "B5: chain, truncated table -> FAIL" || bad "B5: rc=$rc"
    rc=$(chain "$TMP/t8" 0 2); [ "$rc" -eq 1 ] && ok "B6: chain, duplicate row -> FAIL" || bad "B6: rc=$rc"
    rc=$(chain "$TMP/t15" 1 2); [ "$rc" -eq 1 ] && ok "B7: chain, Temp over tolerance -> FAIL" || bad "B7: rc=$rc"
    # heat/cool case: diagnostic I_R -> exit 3 (STATE_AND_PARTICLES_PASS; I_R_CHECK_PENDING), never PASS
    export HPCPERF_NYX_HEATCOOL=YES HPCPERF_NYX_PROFILE="${PROF}-heatcool" HPCPERF_NYX_CPU_PROFILE="${PROF}-cpuhc" HPCPERF_NYX_CASES=lya_heatcool
    GH="$R/build/level3/nyx/${PROF}-heatcool/$SUB"; CH="$R/build/level3/nyx/${PROF}-cpuhc/$SUB"
    for d in "$GH/lya_heatcool.smoke.np1" "$GH/lya_heatcool.smoke.np1.rerun" "$GH/lya_heatcool.smoke.np2" "$CH/lya_heatcool.smoke.np1"; do mk_run "$d"; done
    rc=$(chain "$TMP/t13" 1 2); [ "$rc" -eq 3 ] && /usr/bin/grep -q 'I_R_CHECK_PENDING' "$TMP/val.out" && ! /usr/bin/grep -q '): PASS$' "$TMP/val.out" && ok "B8: chain, heat/cool with diagnostic I_R -> exit 3, I_R_CHECK_PENDING, no PASS" || bad "B8: rc=$rc: $(tail -2 "$TMP/val.out")"
    table "" "Ne=0:0" "I_R=0.29:1.3" "Temp=1e-3:1e-3" > "$TMP/t15hc"   # Temp rel 1e-3 > the heat/cool tolerance 5e-5
    rc=$(chain "$TMP/t15hc" 1 2); [ "$rc" -eq 1 ] && ok "B9: chain, heat/cool with a state variable over tolerance -> FAIL (not PENDING)" || bad "B9: rc=$rc"
    # reports never land in the run directories in offline mode
    [ -z "$(ls "$GH/lya_heatcool.smoke.np2" | /usr/bin/grep -E 'fcompare|particle_compare')" ] && [ -n "$(ls "$TMP/reports" 2>/dev/null)" ] && ok "B10: offline reports written under HPCPERF_NYX_REPORT_DIR, run dirs untouched" || bad "B10: report placement"
    # a legally re-blocked run directory: validate.sh exits 4 with UNSUPPORTED_LAYOUT -- not PASS, not FAIL, not PENDING-as-PASS
    MK_BOXES="$REBLOCK" mk_run "$GH/lya_heatcool.smoke.np2"
    rc=$(chain "$TMP/t13" 0 2); [ "$rc" -eq 4 ] && /usr/bin/grep -q '): UNSUPPORTED_LAYOUT \[' "$TMP/val.out" && ! /usr/bin/grep -q '): PASS$' "$TMP/val.out" && ok "B11: chain, re-blocked layout -> exit 4, UNSUPPORTED_LAYOUT verdict, no PASS" || bad "B11: rc=$rc: $(tail -2 "$TMP/val.out")"
    # a defective layout (missing box) is a FAIL of the case, not UNSUPPORTED
    MK_BOXES="$(echo "$DEFAULT8" | head -n 7)" mk_run "$GH/lya_heatcool.smoke.np2"
    rc=$(chain "$TMP/t13" 0 2); [ "$rc" -eq 1 ] && /usr/bin/grep -q 'STRUCTURAL.*coverage deficit' "$TMP/val.out" && ok "B12: chain, missing box -> FAIL (STRUCTURAL: coverage deficit)" || bad "B12: rc=$rc: $(tail -2 "$TMP/val.out")"
    mk_run "$GH/lya_heatcool.smoke.np2"
    unset HPCPERF_NYX_OFFLINE HPCPERF_NYX_TOOLS_DIR HPCPERF_NYX_SKIP_PARTICLES HPCPERF_NYX_CASES HPCPERF_NYX_REPORT_DIR HPCPERF_NYX_HEATCOOL HPCPERF_NYX_PROFILE HPCPERF_NYX_CPU_PROFILE HPCPERF_L3_RUN_SUBDIR
else
    echo "# validate.sh not found -- Part B skipped"
fi
echo "test_nyx_validator: $pass passed, $failn failed"
[ "$failn" -eq 0 ]
