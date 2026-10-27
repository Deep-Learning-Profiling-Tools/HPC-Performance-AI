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
    # stub fcompare: prints the canned table from $FCOMPARE_TABLE (file) and exits $FCOMPARE_RC; records its arguments
    cat > "$t/amrex_fcompare" <<'EOF'
#!/bin/bash
[ -n "${FCOMPARE_ARGS_LOG:-}" ] && echo "$*" >> "$FCOMPARE_ARGS_LOG"
ref="${@: -2:1}"   # a separate canned table for comparisons against an extracted frozen reference (validate.sh [3] default)
if [ -n "${FCOMPARE_TABLE_FROZEN:-}" ] && [[ "$ref" == *frozen-reference* ]]; then cat "$FCOMPARE_TABLE_FROZEN"; exit "${FCOMPARE_RC_FROZEN:-0}"; fi
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
    # MiniSB's IC count (Exec/MiniSB/ic_sb_32.ascii) is 32686 when the frozen bundle is materialized (level3/nyx/src)
    # or, before that, the freeze-time checkout is present; the count check uses 32768 otherwise
    NP_SB="$( { head -1 "$R/level3/nyx/src/Exec/MiniSB/ic_sb_32.ascii" 2>/dev/null || head -1 "$R/_upstream/level3/Nyx/Exec/MiniSB/ic_sb_32.ascii" 2>/dev/null; } | tr -d ' ')"; NP_SB="${NP_SB:-32768}"
    for d in "$GR/minisb.smoke.np1" "$GR/minisb.smoke.np1.rerun" "$GR/minisb.smoke.np2" "$CR/minisb.smoke.np1"; do mk_run "$d" "$NP_SB"; done
    printf '#!/bin/bash\necho " density  1.0e+10"\n' > "$T/amrex_fvolumesum"; chmod +x "$T/amrex_fvolumesum"
    # Part B exercises check [3] against the CPU reference run directory: the explicit offline mode (unchanged code path)
    export HPCPERF_CPU_REFERENCE=1
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

    # --- Part C: the DEFAULT check [3] -- frozen B200 reference (nyx_reference.py freeze/fetch), no CPU run ------------
    unset HPCPERF_CPU_REFERENCE
    REFT="$R/level3/nyx/nyx_reference.py"
    export HPCPERF_GOLDEN_STORE="$TMP/store" HPCPERF_NYX_REFERENCE_DIR="$TMP/refs"
    # the synthetic references below are not committed files: allowed for the harness only (C20 checks the refusal)
    export HPCPERF_GOLDEN_ALLOW_UNCOMMITTED_REFERENCE=1
    cat > "$TMP/mkpart.py" <<'PY'
import os, sys, numpy as np
# mkpart.py <run_dir> <n> <final scale> : chk00000/DM and chk00010/DM in AMReX checkpoint particle format (ids, double)
run, n, scale = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
rng = np.random.default_rng(7)
pos0 = rng.random((n, 3)) * 8.0
fin = np.column_stack([pos0 + 0.01, np.full(n, 1.0e9), rng.standard_normal((n, 3)) * 100.0])
fin[:, 4:] *= scale                                   # perturb the velocities by the given relative factor
for tag, reals in (("chk00000", np.column_stack([pos0, np.full(n, 1.0e9), np.zeros((n, 3))])), ("chk00010", fin)):
    d = os.path.join(run, tag, "DM"); os.makedirs(os.path.join(d, "Level_0"), exist_ok=True)
    ints = np.column_stack([np.arange(1, n + 1), np.zeros(n)]).astype("<i4")
    with open(os.path.join(d, "Level_0", "DATA_00000"), "wb") as f:
        f.write(ints.tobytes()); f.write(reals.astype("<f8").tobytes())
    with open(os.path.join(d, "Header"), "w") as f:
        f.write("Version_Two_Dot_Zero_double\n3\n4\nmass\nxvel\nyvel\nzvel\n0\n1\n%d\n%d\n0\n1\n0 %d 0\n" % (n, n + 1, n))
PY
    FR="$R/build/level3/nyx/${PROF}-frozen/$SUB"; FC="$R/build/level3/nyx/${PROF}-frozencpu/$SUB"
    mk_frun() { # mk_frun <dir> <backend> <velocity scale> [case]: synthetic run with plotfiles, checkpoint particles, full manifest
        local d=$1 be=$2 sc=$3 cs=${4:-minisb}; mk_run "$d"; python3 "$TMP/mkpart.py" "$d" 64 "$sc"   # (mk_header's loop variables are global)
        printf 'run_id=selftest-%s\napp=nyx\nbackend=%s\ncase=%s\nmode=smoke\nranks=1\ngrid=32x32x32\nmax_grid_size=16\nboxes=8\nsteps=10\nexit_code=0\nbinary_sha256=deadbeef\ndeck_sha256=cafe\nic_sha256=ic=beef\n' \
            "$be" "$be" "$cs" > "$d/run_manifest.txt"
    }
    # the frozen GPU run and its CPU validation: d0 = 1e-9 in Temp (fcompare table), 1e-9 in the particle velocities
    mk_frun "$FR/golden/minisb.smoke.np1" CUDA 1.0; mk_frun "$FC/minisb.smoke.np1" CPU 1.000000001
    table "" "Ne=0:0" "Temp=1e-3:1e-9" > "$TMP/tfreeze"
    freeze_report() { # freeze_report <gpu run> <cpu run> <case> <tol> [diag...]: the [3] reports validate.sh writes
        local g=$1 c=$2 cs=$3 tl=$4; shift 4; local dg=(); [ $# -gt 0 ] && dg=(--diagnostic "$@")
        FCOMPARE_TABLE="$TMP/tfreeze" FCOMPARE_RC="${FREEZE_RC:-0}" python3 "$CHK" "$c/plt00010" "$g/plt00010" --rel_tol "$tl" "${dg[@]}" --fcompare "$T/amrex_fcompare" \
            --fextrema "$T/amrex_fextrema" --out "$g/fcompare.$cs.np1.vs-CPU.txt" --label "$cs.np1.vs-CPU" > /dev/null 2>&1
        python3 "$R/level3/nyx/nyx_particle_compare.py" "$c" "$g" 10 --rel_tol "$tl" > "$g/particle_compare.$cs.np1.vs-CPU.txt" 2>&1
    }
    freeze_report "$FR/golden/minisb.smoke.np1" "$FC/minisb.smoke.np1" minisb 1e-8
    frz() { python3 "$REFT" freeze --steps 10 --mode smoke --allow-dirty "$@" > "$TMP/frz.out" 2>&1; }
    frz --case minisb --gpu-run "$FR/golden/minisb.smoke.np1" --cpu-run "$FC/minisb.smoke.np1" --rel-tol 1e-8 --out "$TMP/refs/minisb.smoke.json"; rc=$?
    TE="$(python3 "$REFT" tol --rel-tol 1e-8 --d0 1e-9)"
    [ $rc -eq 0 ] && python3 - "$TMP/refs/minisb.smoke.json" "$TE" <<'PY' && ok "C1: freeze from an agreeing [3] pair: archive in the store, d0 Temp 1e-9 -> rel_tol (1e-8 - d0)/(1 + d0) = $TE, particle d0 measured exactly" || bad "C1: freeze rc=$rc: $(cat "$TMP/frz.out")"
import json, sys
d = json.load(open(sys.argv[1])); m, p = d["comparison"]["mesh"], d["comparison"]["particles"]
assert m["d0_max"] == 1e-9 and m["d0_max_variable"] == "Temp" and m["zero_reference_fields"] == ["Ne"], m
assert repr(m["rel_tol_effective"]) == sys.argv[2] and m["rel_tol_effective"] < 1e-8 - 1e-9, m
assert abs(p["d0_max"] - 1e-9) < 1e-12 and p["rel_tol_effective"] < 1e-8 - p["d0_max"], p
assert len(d["archive"]["sha256"]) == 64 and d["archive"]["dm_particles"] == 32768 and len(d["archive"]["members"]) == 6
PY
    frz --case minisb --gpu-run "$FR/golden/minisb.smoke.np1" --cpu-run "$FC/minisb.smoke.np1" --rel-tol 1e-8 --out "$TMP/refs/minisb.smoke.json"; rc=$?
    [ $rc -ne 0 ] && /usr/bin/grep -q 'exists' "$TMP/frz.out" && ok "C2: freeze refuses to overwrite an existing reference (needs --replace --reason)" || bad "C2: rc=$rc $(cat "$TMP/frz.out")"
    table "" "Ne=0:0" "Temp=1e-3:2e-8" > "$TMP/tfreeze"; FREEZE_RC=1 freeze_report "$FR/golden/minisb.smoke.np1" "$FC/minisb.smoke.np1" minisb 1e-8
    frz --case minisb --gpu-run "$FR/golden/minisb.smoke.np1" --cpu-run "$FC/minisb.smoke.np1" --rel-tol 1e-8 --out "$TMP/refs/x.json"; rc=$?
    [ $rc -ne 0 ] && [ ! -f "$TMP/refs/x.json" ] && /usr/bin/grep -q 'did not AGREE' "$TMP/frz.out" && ok "C3: freeze refuses a run whose [3] CPU comparison did not agree" || bad "C3: rc=$rc $(cat "$TMP/frz.out")"
    table "" "Ne=0:0" "Temp=1e-3:1e-9" > "$TMP/tfreeze"; freeze_report "$FR/golden/minisb.smoke.np1" "$FC/minisb.smoke.np1" minisb 1e-8
    sed -i 's/^deck_sha256=.*/deck_sha256=other/' "$FC/minisb.smoke.np1/run_manifest.txt"
    frz --case minisb --gpu-run "$FR/golden/minisb.smoke.np1" --cpu-run "$FC/minisb.smoke.np1" --rel-tol 1e-8 --out "$TMP/refs/x.json"; rc=$?
    [ $rc -ne 0 ] && /usr/bin/grep -q 'differ in deck_sha256' "$TMP/frz.out" && ok "C4: freeze refuses a CPU run of another deck" || bad "C4: rc=$rc $(cat "$TMP/frz.out")"
    # heat/cool reference (I_R diagnostic)
    mk_frun "$FR/golden/lya_heatcool.smoke.np1" CUDA 1.0 lya_heatcool; mk_frun "$FC/lya_heatcool.smoke.np1" CPU 1.000000001 lya_heatcool
    table "" "Ne=0:0" "Temp=1e-3:1e-13" "I_R=0.29:1.3" > "$TMP/tfreeze"; FREEZE_RC=1 freeze_report "$FR/golden/lya_heatcool.smoke.np1" "$FC/lya_heatcool.smoke.np1" lya_heatcool 5e-5 I_R
    frz --case lya_heatcool --gpu-run "$FR/golden/lya_heatcool.smoke.np1" --cpu-run "$FC/lya_heatcool.smoke.np1" --rel-tol 5e-5 --diagnostic I_R --out "$TMP/refs/lya_heatcool.smoke.json"; rc=$?
    [ $rc -eq 0 ] && python3 -c "import json,sys; m=json.load(open(sys.argv[1]))['comparison']['mesh']; assert m['diagnostic_fields']==['I_R'] and m['d0_max_variable']=='Temp' and m['d0_max']==1e-13" "$TMP/refs/lya_heatcool.smoke.json" \
        && ok "C5: heat/cool freeze: I_R recorded as diagnostic, not part of the d0 budget" || bad "C5: rc=$rc $(cat "$TMP/frz.out")"
    # validate.sh chain, default mode, synthetic candidate runs (particles compared for real; stub fcompare)
    unset HPCPERF_NYX_SKIP_PARTICLES HPCPERF_NYX_HEATCOOL
    export HPCPERF_NYX_PROFILE="${PROF}-frozen" HPCPERF_NYX_CPU_PROFILE="${PROF}-nocpu" HPCPERF_NYX_CASES=minisb HPCPERF_L3_RUN_SUBDIR="$SUB.cand"
    CD="$R/build/level3/nyx/${PROF}-frozen/$SUB.cand"; NOCPU="$R/build/level3/nyx/${PROF}-nocpu"
    for d in "$CD/minisb.smoke.np1" "$CD/minisb.smoke.np2"; do mk_frun "$d" CUDA 1.0; done
    export FCOMPARE_ARGS_LOG="$TMP/fcargs"
    fchain() { rm -f "$FCOMPARE_ARGS_LOG"; FCOMPARE_TABLE="$1" FCOMPARE_RC="$2" HPCPERF_GPUS="$3" bash "$VAL" CUDA > "$TMP/val.out" 2>&1; echo $?; }
    rc=$(fchain "$TMP/t1" 0 2)
    [ "$rc" -eq 0 ] && /usr/bin/grep -q ': PASS$' "$TMP/val.out" && /usr/bin/grep -q 'minisb.np2.vs-frozen-B200: plotfile comparison' "$TMP/val.out" \
        && /usr/bin/grep -q -- "--rel_tol $TE " "$TMP/fcargs" && ! /usr/bin/grep -q 'CPU profile now' "$TMP/val.out" && [ ! -e "$NOCPU" ] \
        && ok "C6: default [3] = frozen B200 reference: PASS, fcompare called with the reduced rel_tol $TE, no CPU run, no CPU run tree" || bad "C6: rc=$rc: $(tail -4 "$TMP/val.out"); args: $(cat "$TMP/fcargs" 2>/dev/null)"
    table "" "Ne=0:0" "Temp=1e-3:9.5e-9" > "$TMP/tC7"     # within the original 1e-8, outside the budget (1e-8 - 1e-9)/(1 + 1e-9)
    rc=$(FCOMPARE_TABLE_FROZEN="$TMP/tC7" FCOMPARE_RC_FROZEN=1 fchain "$TMP/t1" 0 2)
    [ "$rc" -eq 1 ] && /usr/bin/grep -q 'np2.vs-1GPU: plotfile comparison rel_tol 2e-10: AGREE' "$TMP/val.out" && /usr/bin/grep -q 'FAIL -- minisb.np2.vs-frozen-B200: plotfile comparison.*Temp@L0' "$TMP/val.out" \
        && ok "C7: perturbed field vs the frozen reference (Temp rel 9.5e-9: inside the original 1e-8, outside the d0-reduced budget) -> FAIL ([2] still AGREE)" || bad "C7: rc=$rc: $(tail -3 "$TMP/val.out")"
    table "" "Ne=0:0" "Temp=1e-3:8.5e-9" > "$TMP/tC8"
    rc=$(FCOMPARE_TABLE_FROZEN="$TMP/tC8" FCOMPARE_RC_FROZEN=0 fchain "$TMP/t1" 0 2); [ "$rc" -eq 0 ] && ok "C8: Temp rel 8.5e-9 vs the frozen reference (inside the budget) -> PASS" || bad "C8: rc=$rc: $(tail -3 "$TMP/val.out")"
    python3 "$TMP/mkpart.py" "$CD/minisb.smoke.np2" 64 1.0000000095    # particle velocities: 9.5e-9 relative (budget 1e-8 - 1e-9)
    python3 "$TMP/mkpart.py" "$CD/minisb.smoke.np1" 64 1.0000000095
    rc=$(fchain "$TMP/t1" 0 2)
    [ "$rc" -eq 1 ] && /usr/bin/grep -q 'vs-frozen-B200: DM particle comparison' "$TMP/val.out" && ok "C9: perturbed DM velocities (9.5e-9 relative) -> FAIL against the frozen particles" || bad "C9: rc=$rc: $(tail -3 "$TMP/val.out")"
    for d in "$CD/minisb.smoke.np1" "$CD/minisb.smoke.np2"; do python3 "$TMP/mkpart.py" "$d" 64 1.0; done
    mv "$TMP/refs/minisb.smoke.json" "$TMP/minisb.smoke.json.keep"
    rc=$(fchain "$TMP/t1" 0 2)
    [ "$rc" -eq 1 ] && /usr/bin/grep -q 'frozen B200 reference .* missing' "$TMP/val.out" && ! /usr/bin/grep -q 'CPU profile now' "$TMP/val.out" && [ ! -e "$NOCPU" ] \
        && ok "C10: missing frozen reference -> FAIL, no CPU run (no fallback)" || bad "C10: rc=$rc: $(tail -3 "$TMP/val.out")"
    mv "$TMP/minisb.smoke.json.keep" "$TMP/refs/minisb.smoke.json"
    BLOB="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['archive']['sha256'])" "$TMP/refs/minisb.smoke.json")"; BP="$TMP/store/sha256/${BLOB:0:2}/$BLOB.bin"
    mv "$BP" "$BP.keep"; rc=$(fchain "$TMP/t1" 0 2)
    [ "$rc" -eq 1 ] && /usr/bin/grep -q 'unusable.*not available' "$TMP/val.out" && ok "C11: frozen archive missing from the store -> FAIL" || bad "C11: rc=$rc: $(tail -3 "$TMP/val.out")"
    cp "$BP.keep" "$BP"; printf 'X' | dd of="$BP" bs=1 seek=2000 conv=notrunc 2>/dev/null; rc=$(fchain "$TMP/t1" 0 2)
    [ "$rc" -eq 1 ] && /usr/bin/grep -q 'unusable.*corrupted' "$TMP/val.out" && ok "C12: corrupted frozen archive (sha256 mismatch) -> FAIL" || bad "C12: rc=$rc: $(tail -3 "$TMP/val.out")"
    mv -f "$BP.keep" "$BP"
    cp "$TMP/refs/minisb.smoke.json" "$TMP/minisb.smoke.json.keep"
    python3 -c "import json,sys; p=sys.argv[1]; d=json.load(open(p)); d['comparison']['mesh']['rel_tol_effective']=1e-8; open(p,'w').write(json.dumps(d))" "$TMP/refs/minisb.smoke.json"
    rc=$(fchain "$TMP/t1" 0 2)
    [ "$rc" -eq 1 ] && /usr/bin/grep -q 'is not (tol - d0)/(1 + d0)' "$TMP/val.out" && ok "C13: a reference JSON whose budget was edited is refused -> FAIL" || bad "C13: rc=$rc: $(tail -3 "$TMP/val.out")"
    cp "$TMP/minisb.smoke.json.keep" "$TMP/refs/minisb.smoke.json"
    sed -i 's/^deck_sha256=.*/deck_sha256=other/' "$CD/minisb.smoke.np2/run_manifest.txt"; rc=$(fchain "$TMP/t1" 0 2)
    [ "$rc" -eq 1 ] && /usr/bin/grep -q 'workload differs from the frozen reference: deck_sha256' "$TMP/val.out" && ok "C14: a run of another deck than the frozen reference -> FAIL" || bad "C14: rc=$rc: $(tail -3 "$TMP/val.out")"
    sed -i 's/^deck_sha256=.*/deck_sha256=cafe/' "$CD/minisb.smoke.np2/run_manifest.txt"
    rm -rf "$TMP/fd15"; python3 "$REFT" fetch --ref "$TMP/refs/minisb.smoke.json" --case minisb --mode smoke --steps 20 --run "$CD/minisb.smoke.np2" --dest "$TMP/fd15" > "$TMP/f15.out" 2>&1; rc=$?
    [ "$rc" -eq 1 ] && /usr/bin/grep -q 'no frozen reference for this configuration' "$TMP/f15.out" && ok "C15: another step count than the frozen one -> refused (validate.sh FAIL), never a CPU run" || bad "C15: rc=$rc: $(cat "$TMP/f15.out")"
    # heat/cool against its frozen reference: I_R stays diagnostic -> exit 3, never PASS
    export HPCPERF_NYX_HEATCOOL=YES HPCPERF_NYX_CASES=lya_heatcool
    for d in "$CD/lya_heatcool.smoke.np1" "$CD/lya_heatcool.smoke.np2"; do mk_frun "$d" CUDA 1.0 lya_heatcool; done
    rc=$(fchain "$TMP/t13" 1 2)
    [ "$rc" -eq 3 ] && /usr/bin/grep -q 'vs-frozen-B200:I_R' "$TMP/val.out" && ! /usr/bin/grep -q '): PASS$' "$TMP/val.out" \
        && ok "C16: heat/cool vs the frozen reference: I_R reported, exit 3 (I_R_CHECK_PENDING), never PASS" || bad "C16: rc=$rc: $(tail -2 "$TMP/val.out")"
    # explicit offline mode still runs today's [3]: the CPU run directory is required (and used)
    rc=$(HPCPERF_CPU_REFERENCE=1 fchain "$TMP/t13" 1 2)
    [ "$rc" -eq 1 ] && /usr/bin/grep -q 'OFFLINE -- CPU reference .* missing\|CPU reference run exited' "$TMP/val.out" && ok "C17: HPCPERF_CPU_REFERENCE=1 selects the CPU-reference [3] (here: no CPU run dir -> FAIL)" || bad "C17: rc=$rc: $(tail -2 "$TMP/val.out")"
    rc=0; HPCPERF_NYX_FREEZE_REFERENCE=1 bash "$VAL" CUDA > "$TMP/val.out" 2>&1 || rc=$?
    [ "$rc" -eq 2 ] && /usr/bin/grep -q 'set HPCPERF_CPU_REFERENCE=1' "$TMP/val.out" && ok "C18: freezing without the CPU-reference check is refused (exit 2)" || bad "C18: rc=$rc: $(tail -2 "$TMP/val.out")"
    # agent workspace layout (tools/golden is not copied into a workspace): fetch resolves the canonical repository's
    # store tool through workspace.yaml; the archive is still verified against the (protected) reference JSON
    WSN="$TMP/ws/level3/nyx"; mkdir -p "$WSN"; cp "$R/level3/nyx/"nyx_*.py "$WSN/"; printf 'schema: hpcperf-workspace-1\ncanonical_dir: %s\n' "$R/level3/nyx" > "$WSN/workspace.yaml"
    rm -rf "$TMP/fd19"; python3 "$WSN/nyx_reference.py" fetch --ref "$TMP/refs/minisb.smoke.json" --case minisb --mode smoke --steps 10 --run "$CD/minisb.smoke.np2" --dest "$TMP/fd19" > "$TMP/f19.out" 2>&1; rc=$?
    rm -f "$WSN/workspace.yaml"; rm -rf "$TMP/fd19b"; python3 "$WSN/nyx_reference.py" fetch --ref "$TMP/refs/minisb.smoke.json" --case minisb --mode smoke --steps 10 --run "$CD/minisb.smoke.np2" --dest "$TMP/fd19b" > "$TMP/f19b.out" 2>&1; rc2=$?
    [ "$rc" -eq 0 ] && [ -f "$TMP/fd19/plt00010/Header" ] && [ "$rc2" -eq 1 ] && /usr/bin/grep -q 'hpcperf_golden.py not found' "$TMP/f19b.out" \
        && ok "C19: agent-workspace layout: the frozen archive comes from the canonical repository's store (verified); without workspace.yaml -> refused" || bad "C19: rc=$rc rc2=$rc2: $(cat "$TMP/f19.out" "$TMP/f19b.out")"
    # a frozen reference counts only when committed and unmodified (adiabatic MiniSB again)
    unset HPCPERF_NYX_HEATCOOL; export HPCPERF_NYX_CASES=minisb
    rm -rf "$TMP/fd20"; env -u HPCPERF_GOLDEN_ALLOW_UNCOMMITTED_REFERENCE python3 "$REFT" fetch --ref "$TMP/refs/minisb.smoke.json" --case minisb --mode smoke --steps 10 --run "$CD/minisb.smoke.np2" --dest "$TMP/fd20" > "$TMP/f20.out" 2>&1; rc=$?
    unset HPCPERF_GOLDEN_ALLOW_UNCOMMITTED_REFERENCE; rc2=$(fchain "$TMP/t1" 0 2); export HPCPERF_GOLDEN_ALLOW_UNCOMMITTED_REFERENCE=1
    [ "$rc" -eq 1 ] && /usr/bin/grep -q 'not usable.*committed and unmodified' "$TMP/f20.out" && [ "$rc2" -eq 1 ] && /usr/bin/grep -q 'frozen B200 reference unusable.*committed' "$TMP/val.out" \
        && ok "C20: an uncommitted reference JSON is refused (fetch exit 1, validate.sh FAIL) without the harness override" || bad "C20: rc=$rc rc2=$rc2: $(cat "$TMP/f20.out"; tail -2 "$TMP/val.out")"
    rc=$(fchain "$TMP/t1" 0 2); [ "$rc" -eq 0 ] && /usr/bin/grep -q 'WARNING -- the frozen reference is NOT the committed file' "$TMP/val.out" \
        && ok "C20b: with the harness override the uncommitted reference is used and validate.sh says so loudly" || bad "C20b: rc=$rc: $(tail -3 "$TMP/val.out")"
    cp "$TMP/refs/minisb.smoke.json" "$TMP/minisb.smoke.json.keep"
    python3 -c "import json,sys; p=sys.argv[1]; d=json.load(open(p)); d['comparison']['mesh']['d0_max']=-0.5; open(p,'w').write(json.dumps(d))" "$TMP/refs/minisb.smoke.json"
    rm -rf "$TMP/fd21"; python3 "$REFT" fetch --ref "$TMP/refs/minisb.smoke.json" --case minisb --mode smoke --steps 10 --run "$CD/minisb.smoke.np2" --dest "$TMP/fd21" > "$TMP/f21.out" 2>&1; rc=$?
    [ "$rc" -eq 1 ] && /usr/bin/grep -q 'is not a measured deviation' "$TMP/f21.out" && ok "C21: a negative d0 (which would widen the budget) is refused" || bad "C21: rc=$rc: $(cat "$TMP/f21.out")"
    cp "$TMP/minisb.smoke.json.keep" "$TMP/refs/minisb.smoke.json"
    # freeze mode freezes only a case whose every earlier check passed: [4] (baryon mass) failing -> NOT frozen
    cat > "$T/amrex_fvolumesum" <<'EOF'
#!/bin/bash
case "$*" in *plt00010*) [ -n "${FVS_DRIFT:-}" ] && { echo " density  1.1e+10"; exit 0; } ;; esac
echo " density  1.0e+10"
EOF
    chmod +x "$T/amrex_fvolumesum"
    G22="$R/build/level3/nyx/${PROF}-frozen/$SUB.c22"; C22="$R/build/level3/nyx/${PROF}-frozencpu/$SUB.c22"
    mk_frun "$G22/minisb.smoke.np1" CUDA 1.0; mk_frun "$G22/minisb.smoke.np1.rerun" CUDA 1.0; mk_frun "$C22/minisb.smoke.np1" CPU 1.000000001
    frz22() { rm -rf "$TMP/refs22"; HPCPERF_L3_RUN_SUBDIR="$SUB.c22" HPCPERF_NYX_CPU_PROFILE="${PROF}-frozencpu" HPCPERF_NYX_REFERENCE_DIR="$TMP/refs22" \
              HPCPERF_CPU_REFERENCE=1 HPCPERF_NYX_FREEZE_REFERENCE=1 FCOMPARE_TABLE="$TMP/t1" FCOMPARE_RC=0 HPCPERF_GPUS=1 bash "$VAL" CUDA > "$TMP/val.out" 2>&1; echo $?; }
    rc=$(FVS_DRIFT=1 frz22)
    [ "$rc" -eq 1 ] && /usr/bin/grep -q 'baryon mass not conserved' "$TMP/val.out" && /usr/bin/grep -q 'minisb: NOT frozen' "$TMP/val.out" && [ ! -e "$TMP/refs22/minisb.smoke.json" ] \
        && ok "C22: freeze mode, [3] vs CPU agrees but [4] failed -> NOT frozen" || bad "C22: rc=$rc: $(tail -4 "$TMP/val.out")"
    rc=$(frz22)   # control: every check passes -> the freeze is attempted (it succeeds from a committed harness; refused from a dirty one)
    /usr/bin/grep -q 'FROZEN minisb.smoke\|has uncommitted changes' "$TMP/val.out" && ! /usr/bin/grep -q 'NOT frozen' "$TMP/val.out" \
        && ok "C22b: control -- all checks pass -> the freeze is attempted (rc=$rc)" || bad "C22b: rc=$rc: $(tail -4 "$TMP/val.out")"
    rm -rf "$G22" "$C22"
    unset FCOMPARE_ARGS_LOG HPCPERF_GOLDEN_STORE HPCPERF_NYX_REFERENCE_DIR HPCPERF_GOLDEN_ALLOW_UNCOMMITTED_REFERENCE
    rm -rf "$R/build/level3/nyx/${PROF}-frozen" "$R/build/level3/nyx/${PROF}-frozencpu" "$NOCPU"
    unset HPCPERF_NYX_OFFLINE HPCPERF_NYX_TOOLS_DIR HPCPERF_NYX_SKIP_PARTICLES HPCPERF_NYX_CASES HPCPERF_NYX_REPORT_DIR HPCPERF_NYX_HEATCOOL HPCPERF_NYX_PROFILE HPCPERF_NYX_CPU_PROFILE HPCPERF_L3_RUN_SUBDIR
else
    echo "# validate.sh not found -- Part B skipped"
fi

# --- Part D (optional, no GPU): the REAL AMReX comparison tools and the REAL frozen reference minisb.smoke -------------
# The candidate is the frozen reference's own data (extracted from the golden store) placed in synthetic run directories:
# unperturbed -> PASS (r = 0); one Temp cell raised to a relative 9.98e-9 of max|Temp| (inside the original 1e-8, outside
# the d0-reduced budget 9.955e-9) -> FAIL; 9.9e-9 -> PASS. Runs only where the CPU profile's tools, the reference JSON
# and its archive exist (a materialized + built worktree); otherwise skipped and said so.
RTOOLS="$R/.deps/level3/nyx/cpu-gcc133-adiabatic/install/bin"; RREF="$R/level3/nyx/reference/minisb.smoke.json"
if [ -f "$VAL" ] && [ -x "$RTOOLS/amrex_fcompare" ] && [ -f "$RREF" ] && git -C "$R" ls-files --error-unmatch "$RREF" > /dev/null 2>&1 && \
   [ -z "$(git -C "$R" status --porcelain -- "$RREF")" ] && \
   python3 "$R/tools/golden/hpcperf_golden.py" store-get "$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['archive']['sha256'])" "$RREF")" > "$TMP/blob.path" 2>/dev/null; then
    SUBD="run.selftest-real-$$"; DR="$R/build/level3/nyx/__selftest__-real/$SUBD"
    mk_real() { # mk_real <run_dir> <relative Temp perturbation of max|Temp| at its maximum cell, 0 = none>
        python3 - "$RREF" "$(tail -1 "$TMP/blob.path")" "$1" "$2" <<'PY'
import json, os, re, shutil, sys, tarfile
import numpy as np
ref, blob, run, eps = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], float(sys.argv[4])
os.makedirs(run, exist_ok=True)
with tarfile.open(blob) as tf:
    for m in tf.getmembers():
        p = os.path.join(run, m.name); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").write(tf.extractfile(m).read())
shutil.copytree(os.path.join(run, "plt00010"), os.path.join(run, "plt00000"))      # mass check: identical masses
with open(os.path.join(run, "runlog"), "w") as f:
    f.write("#  nstep time dt z a\n" + "".join(f"       {s} 1.7e7 2.6e5 100 0.0099\n" for s in range(11)))
w = ref["workload"]
with open(os.path.join(run, "run_manifest.txt"), "w") as f:
    f.write("".join(f"{k}={w[k]}\n" for k in ("case", "mode", "steps", "grid", "max_grid_size", "boxes", "deck_sha256", "ic_sha256"))
            + "run_id=selftest-real\nexit_code=0\nranks=1\nbackend=CUDA\nbinary_sha256=selftest\n")
if eps:
    plt = os.path.join(run, "plt00010"); H = open(os.path.join(plt, "Header")).read().splitlines()
    names = [H[2 + k].strip() for k in range(int(H[1]))]; ic = names.index("Temp")
    CH = open(os.path.join(plt, "Level_0", "Cell_H")).read().splitlines()
    fods = [l.split()[1:] for l in CH if l.startswith("FabOnDisk:")]
    best = None
    for fn, off in fods:                                    # find the global max of Temp
        with open(os.path.join(plt, "Level_0", fn), "rb") as fh:
            fh.seek(int(off)); hdr = fh.readline().decode()
            lo, hi = [list(map(int, g.split(","))) for g in re.findall(r"\(\(([-\d,]+)\) \(([-\d,]+)\)", hdr)[0]]
            n = np.prod([h - l + 1 for l, h in zip(lo, hi)]); start = fh.tell()
            fh.seek(start + 8 * n * ic); a = np.frombuffer(fh.read(8 * n), "<f8")
            k = int(np.argmax(np.abs(a)))
            if best is None or abs(a[k]) > abs(best[2]): best = (fn, start + 8 * (n * ic + k), a[k])
    fn, pos, v = best
    with open(os.path.join(plt, "Level_0", fn), "r+b") as fh:
        fh.seek(pos); fh.write(np.array([v + eps * abs(v)], "<f8").tobytes())
PY
    }
    export HPCPERF_L3_RUN_SUBDIR="$SUBD" HPCPERF_NYX_PROFILE="__selftest__-real" HPCPERF_NYX_CASES=minisb HPCPERF_NYX_OFFLINE=1 HPCPERF_NYX_REPORT_DIR="$TMP/realreports"
    real() { rm -rf "$DR"; mk_real "$DR/minisb.smoke.np1" "$1"; mk_real "$DR/minisb.smoke.np1.rerun" "$1"
             HPCPERF_GPUS=1 bash "$VAL" CUDA > "$TMP/real.out" 2>&1; echo $?; }
    rc=$(real 0)
    [ "$rc" -eq 0 ] && /usr/bin/grep -q 'minisb.np1.vs-frozen-B200: plotfile comparison rel_tol 9.955.*AGREE' "$TMP/real.out" && ok "D1: real fcompare/particles vs the real frozen minisb reference (its own data) -> PASS" || bad "D1: rc=$rc: $(tail -3 "$TMP/real.out")"
    # D2/D3 call the comparator directly (each AMReX tool start-up costs ~7 s of MPI initialisation here) on the extracted
    # frozen reference, at the rel_tol validate.sh derived for it
    FDR="$TMP/realreports/frozen-reference.minisb.smoke"; TM="$(sed -n 's/.*plotfile comparison rel_tol \([0-9.e+-]*\): AGREE.*/\1/p' "$TMP/real.out" | tail -1)"
    dchk() { rm -rf "$DR"; mk_real "$DR/minisb.smoke.np1" "$1"
             python3 "$CHK" "$FDR/plt00010" "$DR/minisb.smoke.np1/plt00010" --rel_tol "$TM" --fcompare "$RTOOLS/amrex_fcompare" --fextrema "$RTOOLS/amrex_fextrema" > "$TMP/real2.out" 2>&1; echo $?; }
    # perturbations placed from the reference's own budget TM: halfway between TM and the original 1e-8, and 0.99 TM
    EOUT="$(python3 -c "import sys; t=float(sys.argv[1]); print(repr((t + 1e-8) / 2))" "${TM:-0}")"; EIN="$(python3 -c "import sys; print(repr(0.99 * float(sys.argv[1])))" "${TM:-0}")"
    rc=$(dchk "$EOUT")
    [ -n "$TM" ] && [ "$rc" -eq 1 ] && /usr/bin/grep -q 'DISAGREE.*Temp@L0' "$TMP/real2.out" \
        && ok "D2: one Temp cell +$EOUT of max|Temp| (inside 1e-8, outside the d0-reduced $TM) -> DISAGREE, real fcompare" || bad "D2: rc=$rc tol=$TM: $(cat "$TMP/real2.out")"
    rc=$(dchk "$EIN")
    [ "$rc" -eq 0 ] && ok "D3: the same cell +$EIN (inside the budget) -> AGREE, real fcompare" || bad "D3: rc=$rc: $(cat "$TMP/real2.out")"
    # agent workspace: the reference copy must be byte-identical to the canonical committed one
    WSR="$TMP/wsr/level3/nyx"; mkdir -p "$WSR/reference"; cp "$R/level3/nyx/"nyx_*.py "$WSR/"; printf 'schema: hpcperf-workspace-1\ncanonical_dir: %s\n' "$R/level3/nyx" > "$WSR/workspace.yaml"
    cp "$RREF" "$WSR/reference/"; rm -rf "$DR"; mk_real "$DR/minisb.smoke.np1" 0
    rm -rf "$TMP/fd4"; python3 "$WSR/nyx_reference.py" fetch --ref "$WSR/reference/minisb.smoke.json" --case minisb --mode smoke --steps 10 --run "$DR/minisb.smoke.np1" --dest "$TMP/fd4" > "$TMP/d4.out" 2>&1; rc=$?
    printf ' ' >> "$WSR/reference/minisb.smoke.json"
    rm -rf "$TMP/fd4b"; python3 "$WSR/nyx_reference.py" fetch --ref "$WSR/reference/minisb.smoke.json" --case minisb --mode smoke --steps 10 --run "$DR/minisb.smoke.np1" --dest "$TMP/fd4b" > "$TMP/d4b.out" 2>&1; rc2=$?
    [ "$rc" -eq 0 ] && /usr/bin/grep -q '^REF_COMMITTED=canonical$' "$TMP/d4.out" && [ "$rc2" -eq 1 ] && /usr/bin/grep -q 'differs from the canonical committed' "$TMP/d4b.out" \
        && ok "D4: agent workspace: a byte-identical copy of the committed reference is accepted; a modified copy is refused" || bad "D4: rc=$rc rc2=$rc2: $(cat "$TMP/d4.out" "$TMP/d4b.out")"
    rm -rf "$R/build/level3/nyx/__selftest__-real"
    unset HPCPERF_L3_RUN_SUBDIR HPCPERF_NYX_PROFILE HPCPERF_NYX_CASES HPCPERF_NYX_OFFLINE HPCPERF_NYX_REPORT_DIR
else
    echo "# Part D skipped: needs the CPU profile's AMReX tools, the COMMITTED level3/nyx/reference/minisb.smoke.json and its archive in the golden store"
fi
echo "test_nyx_validator: $pass passed, $failn failed"
[ "$failn" -eq 0 ]
