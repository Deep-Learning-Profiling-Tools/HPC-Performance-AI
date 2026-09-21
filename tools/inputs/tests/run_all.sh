#!/bin/bash
# Regression tests of the input registry (tools/inputs/hpcperf_inputs.py) and of the
# run.sh input selectors. No GPU execution: the tool is exercised on the committed
# inputs.yaml files and on log fixtures, the run.sh guards through HPCPERF_DRY_RUN=1
# (those cases are skipped when the app is not built in this worktree).
#   usage: tools/inputs/tests/run_all.sh          (source hpcperf_env.sh first: needs PyYAML)
set -u -o pipefail   # pipefail: a failing tool exit must not be masked by the noise filter
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../../.." && pwd)"
TOOL="$R/tools/inputs/hpcperf_inputs.py"
FX="$HERE/fixtures"
pass=0; failn=0; skip=0
ok()  { echo "ok   $*"; pass=$((pass+1)); }
bad() { echo "FAIL $*"; failn=$((failn+1)); }
noise() { grep -viE 'lua|posix|no file|stack traceback|\[C\]|addto|no field' | grep -vE '^\s*$'; }
unset HPCPERF_GPUS HPCPERF_NP HPCPERF_SCALE_MODE HPCPERF_DRY_RUN HPCPERF_HIPBONE_INPUT HPCPERF_QUICKSILVER_INPUT_ID HPCPERF_LAMMPS_INPUT \
      HPCPERF_LAMMPS_STEPS HPCPERF_LAMMPS_STRONG HPCPERF_LAMMPS_LOCAL HPCPERF_QUICKSILVER_STEPS HPCPERF_QUICKSILVER_INPUT
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# ---- 1. every committed inputs.yaml validates; default_input is registered ----------------
for d in level1/background_subtraction level2/hipbone level2/quicksilver level3/lammps; do
    out="$(python3 "$TOOL" validate "$R/$d" 2>&1 | noise)"; rc=$?
    [ $rc -eq 0 ] && grep -q ': OK' <<<"$out" && ok "validate $d" || bad "validate $d: $out"
    n="$(python3 "$TOOL" list "$R/$d" 2>/dev/null | wc -l)"
    [ "$n" -ge 2 ] && ok "$d registers $n inputs" || bad "$d: expected >= 2 inputs, got $n"
    python3 "$TOOL" list "$R/$d" 2>/dev/null | grep -q '(default)' && ok "$d: default input marked" || bad "$d: no default input"
done

# ---- 2. unknown ids and parameters are refused (exit 2), never guessed ---------------------
python3 "$TOOL" args "$R/level2/hipbone" no-such-input >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -eq 2 ] && grep -q "unknown input id" "$TMP/e" && ok "unknown input id -> exit 2" || bad "unknown id: rc=$rc $(cat "$TMP/e")"
python3 "$TOOL" param "$R/level3/lammps" lj-32k no_such_key >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -eq 2 ] && ok "unknown parameter -> exit 2" || bad "unknown parameter: rc=$rc"
python3 "$TOOL" args "$R/level1/hotspot" x >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -ne 0 ] && grep -q "no inputs.yaml" "$TMP/e" && ok "benchmark without inputs.yaml -> error" || bad "missing inputs.yaml not reported (rc=$rc)"

# ---- 3. schema errors are caught ----------------------------------------------------------
mkdir -p "$TMP/bad"
cat > "$TMP/bad/inputs.yaml" <<'EOF'
schema: hpcperf-inputs-1
benchmark: bad
level: 2
selector: HPCPERF_BAD_INPUT
default_input: nothere
entry: {kind: run.sh, path: level2/bad/run.sh}
timing: {scope: x, kind: total, unit: parsec, regex: 'time (\d+)', select: only}
baseline: {quantities: [{name: q, regex: 'q=(\d+)', compare: {rule: bogus}}]}
inputs:
  - {id: A_Bad, case: c, variant: nonsense, source: {kind: custom}, params: {}, backends_validated: []}
  - {id: dup, case: c, variant: size, source: {kind: upstream-file}, params: {}, backends_validated: []}
  - {id: dup, case: c, variant: size, source: {kind: derived}, params: {}, backends_validated: []}
EOF
out="$(python3 "$TOOL" validate "$TMP/bad" 2>&1 | noise)"; rc=$?
[ $rc -eq 1 ] && ok "invalid inputs.yaml -> exit 1" || bad "invalid inputs.yaml accepted (rc=$rc)"
for msg in "unit must be" "compare.rule must be" "id must match" "duplicate id" "variant must be" "must state 'derivation'" "must state 'upstream'" "default_input 'nothere'" "regex needs a named group"; do
    grep -q -- "$msg" <<<"$out" && ok "schema check: $msg" || bad "schema check missing: $msg"
done

# ---- 4. timing parsing: correct line, unit conversion, per-iteration work, failure modes ---
t="$(python3 "$TOOL" parse-timing "$R/level2/hipbone" "$FX/hipbone_nx24_p14.log" 2>&1 | noise)"
python3 - "$t" <<'PY' && ok "hipbone: elapsed field of the hipBone: line, seconds" || bad "hipbone parse: $t"
import json,sys; d=json.loads(sys.argv[1]); assert abs(d["main_compute_s"]-0.2876)<1e-9 and d["unit"]=="s" and d["kind"]=="total"
PY
t="$(python3 "$TOOL" parse-timing "$R/level2/quicksilver" "$FX/quicksilver_two_tables.log" 2>&1 | noise)"
python3 - "$t" <<'PY' && ok "quicksilver: 'main' from the Cumulative table (not Last Cycle), us -> s" || bad "quicksilver parse: $t"
import json,sys; d=json.loads(sys.argv[1]); assert abs(d["main_compute_s"]-7.057)<1e-6 and d["raw_value"]==7.057e6 and d["unit"]=="us"
PY
t="$(python3 "$TOOL" parse-timing "$R/level1/background_subtraction" "$FX/bgsub_r102.log" --input w4096-h2048-merged0-r102 2>&1 | noise)"
python3 - "$t" <<'PY' && ok "background_subtraction: per-frame us x (repeat-2) frames" || bad "bgsub parse: $t"
import json,sys; d=json.loads(sys.argv[1]); assert d["work_count"]==100 and abs(d["main_compute_s"]-100*63.5e-6)<1e-9
PY
t="$(python3 "$TOOL" parse-timing "$R/level3/lammps" "$FX/lammps_lj.log" 2>&1 | noise)"
python3 - "$t" <<'PY' && ok "lammps: Loop time line" || bad "lammps parse: $t"
import json,sys; d=json.loads(sys.argv[1]); assert abs(d["main_compute_s"]-0.852877)<1e-9
PY
python3 "$TOOL" parse-timing "$R/level2/hipbone" "$FX/hipbone_nx24_p14.log" --rc 134 >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -eq 1 ] && grep -q "failed run" "$TMP/e" && ok "nonzero exit code: timer not used" || bad "failed run parsed (rc=$rc)"
python3 "$TOOL" parse-timing "$R/level2/hipbone" "$FX/no_timer.log" >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -eq 1 ] && grep -q "no matching line" "$TMP/e" && ok "missing timer line -> error, no E2E substitute" || bad "missing timer accepted (rc=$rc)"
python3 "$TOOL" parse-timing "$R/level2/hipbone" "$FX/hipbone_two_lines.log" >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -eq 1 ] && grep -q "exactly one" "$TMP/e" && ok "ambiguous (two timer lines) -> error" || bad "ambiguous timer accepted (rc=$rc)"
python3 "$TOOL" parse-timing "$R/level2/hipbone" "$FX/hipbone_nan.log" >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -eq 1 ] && ok "non-finite timer value -> error" || bad "nan timer accepted (rc=$rc)"
python3 "$TOOL" parse-timing "$R/level2/hipbone" "$TMP/does-not-exist.log" >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -eq 1 ] && grep -q "missing" "$TMP/e" && ok "missing log -> error" || bad "missing log accepted (rc=$rc)"

# ---- 5. baseline extraction / comparison ---------------------------------------------------
python3 "$TOOL" extract "$R/level2/hipbone" "$FX/hipbone_nx24_p14.log" > "$TMP/hb.json" 2>/dev/null
python3 - "$TMP/hb.json" <<'PY' && ok "hipbone: residual norms and iteration count extracted" || bad "hipbone extract"
import json,sys; d=json.load(open(sys.argv[1])); assert d["cg_iterations"]["value"]==100 and d["r_norm_final"]["value"]<1e-8 and d["dofs"]["value"]==37595375
PY
python3 - "$TMP/hb.json" > "$TMP/hb_baseline.json" <<'PY'
import json,sys; print(json.dumps({"input_id":"coral2-nx24-p14","quantities":json.load(open(sys.argv[1]))}))
PY
python3 "$TOOL" compare "$R/level2/hipbone" "$TMP/hb_baseline.json" "$FX/hipbone_nx24_p14.log" >/dev/null 2>&1 && ok "hipbone: log compares equal to its own baseline" || bad "hipbone self-compare failed"
python3 "$TOOL" compare "$R/level2/hipbone" "$TMP/hb_baseline.json" "$FX/hipbone_bad_residual.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 1 ] && grep -q '"r_norm_initial"' "$TMP/c.json" && ok "hipbone: initial residual off by 5% -> compare fails (1% rule)" || bad "bad initial residual accepted (rc=$rc)"
python3 "$TOOL" compare "$R/level2/hipbone" "$TMP/hb_baseline.json" "$FX/hipbone_99_iterations.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 1 ] && grep -q '"cg_iterations"' "$TMP/c.json" && ok "hipbone: 99 instead of 100 CG iterations -> compare fails (exact)" || bad "iteration count mismatch accepted (rc=$rc)"
python3 - "$TMP/hb.json" <<'PY2' && ok "hipbone: final residual is recorded (rule record, tolerance TBD)" || bad "hipbone final residual not recorded"
import json,sys; d=json.load(open(sys.argv[1])); assert d["r_norm_final"]["value"] > 0
PY2
python3 "$TOOL" extract "$R/level3/lammps" "$FX/lammps_rhodo.log" --input rhodo-32k > "$TMP/rh.json" 2>/dev/null
python3 - "$TMP/rh.json" <<'PY' && ok "lammps rhodo: per-input override reads thermo_style multi blocks (first=step 0, last=step 100)" || bad "rhodo extract: $(cat "$TMP/rh.json")"
import json,sys; d=json.load(open(sys.argv[1]))
assert abs(d["toteng_step0"]["value"]-(-25356.2057))<1e-6 and abs(d["toteng_step100"]["value"]-(-25290.7364))<1e-6 and abs(d["press_step100"]["value"]-6.7352)<1e-6
PY
python3 "$TOOL" extract "$R/level2/quicksilver" "$FX/quicksilver_two_tables.log" > "$TMP/qs.json" 2>/dev/null
python3 - "$TMP/qs.json" <<'PY' && ok "quicksilver: four PASS lines present, no FAIL, last-cycle tallies recorded" || bad "quicksilver extract: $(cat "$TMP/qs.json")"
import json,sys; d=json.load(open(sys.argv[1]))
assert all(d[k]["present"] for k in ("pass_ratios","pass_facet","pass_no_loss","pass_fluence")) and not d["fail_marker"]["present"]
assert d["final_cycle_census"]["value"]==91801 and d["final_cycle_num_seg"]["value"]==1868057 and abs(d["final_cycle_scalar_flux"]["value"]-5.918521e5)<1
PY
python3 "$TOOL" extract "$R/level1/background_subtraction" "$FX/bgsub_r102.log" > "$TMP/bg.json" 2>/dev/null
python3 - "$TMP/bg.json" > "$TMP/bg_base.json" <<'PY'
import json,sys; d=json.load(open(sys.argv[1])); assert d["max_error"]["value"]==0 and d["pass_marker"]["present"]; print(json.dumps({"quantities":d}))
PY
[ $? -eq 0 ] && ok "background_subtraction: Max error 0 + PASS extracted" || bad "bgsub extract"
python3 "$TOOL" compare "$R/level1/background_subtraction" "$TMP/bg_base.json" "$FX/bgsub_fail.log" >/dev/null 2>&1; rc=$?
[ $rc -eq 1 ] && ok "background_subtraction: 'Max error is 3' + FAIL -> compare fails" || bad "bgsub FAIL log accepted (rc=$rc)"

# ---- 6. run.sh selectors (dry-run; needs the built app) -------------------------------------
export HPCPERF_DRY_RUN=1
if [ -x "$R/build/level2/hipbone/cuda/hipBone" ]; then
    out="$(HPCPERF_HIPBONE_INPUT=bogus "$R/level2/hipbone/run.sh" CUDA 2>&1 | noise)"; rc=${PIPESTATUS[0]}
    grep -q "unknown input id" <<<"$out" && ok "hipbone run.sh: unknown id refused" || bad "hipbone unknown id: $out"
    out="$(HPCPERF_HIPBONE_INPUT=sweep-nx16-p8 "$R/level2/hipbone/run.sh" CUDA -nx 8 2>&1 | noise)"
    grep -q "mutually exclusive" <<<"$out" && ok "hipbone run.sh: id + extra args refused" || bad "hipbone id+args: $out"
    out="$(HPCPERF_HIPBONE_INPUT=sweep-nx16-p8 "$R/level2/hipbone/run.sh" CUDA 2>&1 | noise)"
    grep -q -- '-nx 16 -ny 16 -nz 16 -p 8 -v' <<<"$out" && ok "hipbone run.sh: id resolves to its registered args" || bad "hipbone id args: $out"
    out="$("$R/level2/hipbone/run.sh" CUDA 2>&1 | noise)"
    grep -q -- '-nx 24 -ny 24 -nz 24 -p 14' <<<"$out" && ! grep -q 'input=' <<<"$out" && ok "hipbone run.sh: default command unchanged" || bad "hipbone default changed: $out"
else skip=$((skip+1)); echo "skip hipbone run.sh (not built)"; fi
if [ -x "$R/build/level2/quicksilver/cuda/qs" ]; then
    out="$(HPCPERF_GPUS=2 HPCPERF_QUICKSILVER_INPUT_ID=coral2-p1-1rank "$R/level2/quicksilver/run.sh" CUDA 2>&1 | noise)"
    grep -q "single-rank deck" <<<"$out" && ok "quicksilver run.sh: verbatim deck at 2 ranks refused" || bad "quicksilver 2-rank verbatim: $out"
    out="$(HPCPERF_QUICKSILVER_STEPS=5 HPCPERF_QUICKSILVER_INPUT_ID=coral2-p1-1rank "$R/level2/quicksilver/run.sh" CUDA 2>&1 | noise)"
    grep -q "mutually exclusive" <<<"$out" && ok "quicksilver run.sh: id + size knob refused" || bad "quicksilver id+knob: $out"
    out="$(HPCPERF_QUICKSILVER_INPUT_ID=coral2-p2-1rank "$R/level2/quicksilver/run.sh" CUDA 2>&1 | noise)"
    grep -q 'Coral2_P2_1.inp$' <<<"$(grep 'command:' <<<"$out")" && ok "quicksilver run.sh: verbatim deck passed with -i only" || bad "quicksilver verbatim cmd: $out"
    out="$("$R/level2/quicksilver/run.sh" CUDA 2>&1 | noise)"
    grep -q 'mesh=8x8x8 particles=100000 steps=20$' <<<"$out" && ok "quicksilver run.sh: default command unchanged" || bad "quicksilver default changed: $out"
else skip=$((skip+1)); echo "skip quicksilver run.sh (not built)"; fi
if [ -x "$R/build/level3/lammps/cuda/lmp_kokkos_cuda" ] && [ -f "$R/level3/lammps/src/bench/in.lj" ]; then
    out="$(HPCPERF_SCALE_MODE=strong HPCPERF_LAMMPS_INPUT=lj-2m "$R/level3/lammps/run.sh" CUDA 2>&1 | noise)"
    grep -q "mutually exclusive" <<<"$out" && ok "lammps run.sh: id + strong mode refused" || bad "lammps id+strong: $out"
    out="$(HPCPERF_LAMMPS_STEPS=10 HPCPERF_LAMMPS_INPUT=lj-2m "$R/level3/lammps/run.sh" CUDA 2>&1 | noise)"
    grep -q "mutually exclusive" <<<"$out" && ok "lammps run.sh: id + steps knob refused" || bad "lammps id+steps: $out"
    out="$(HPCPERF_LAMMPS_INPUT=eam-32k "$R/level3/lammps/run.sh" CUDA 2>&1 | noise)"
    deck="$R/build/level3/lammps/cuda/run/.dryrun/in.eam.input.eam-32k"
    grep -q "in.eam" <<<"$out" && grep -q "^pair_coeff .* $R/level3/lammps/src/bench/Cu_u3.eam" "$deck" && ok "lammps run.sh: eam deck derived with absolute potential path (frozen src untouched)" || bad "lammps eam deck: $(grep pair_coeff "$deck" 2>&1)"
    cmp -s "$R/level3/lammps/src/bench/in.eam" <(sed -e 's/^run             ${steps}/run             100/' -e "s#$R/level3/lammps/src/bench/##" "$deck") && ok "lammps: derived eam deck differs from upstream only in run/path substitution" || bad "lammps: derived eam deck drifted from upstream"
    out="$("$R/level3/lammps/run.sh" CUDA 2>&1 | noise)"
    grep -q 'mode=smoke ranks=1 box=20x20x20' <<<"$out" && ! grep -q 'input=' <<<"$out" && ok "lammps run.sh: default smoke command unchanged" || bad "lammps default changed: $out"
else skip=$((skip+1)); echo "skip lammps run.sh (not built/materialized)"; fi
unset HPCPERF_DRY_RUN

# ---- 7. frozen Level 3 source tree untouched by the input machinery -----------------------
if [ -f "$R/level3/lammps/provenance/source.lock.yaml" ] && [ -d "$R/level3/lammps/src" ]; then
    st="$("$R/tools/prepare_benchmark.sh" level3 lammps --status 2>&1 | noise | grep -o 'PREPARE STATUS lammps [A-Z_]*')"
    [ "$st" = "PREPARE STATUS lammps READY" ] && ok "lammps frozen tree still READY (source_tree_sha256 unchanged)" || bad "lammps frozen tree: $st"
else skip=$((skip+1)); echo "skip frozen-tree check (lammps not materialized)"; fi

echo; echo "inputs tests: $pass passed, $failn failed, $skip skipped"
[ $failn -eq 0 ]
