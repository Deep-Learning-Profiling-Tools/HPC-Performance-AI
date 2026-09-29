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
      HPCPERF_LAMMPS_STEPS HPCPERF_LAMMPS_STRONG HPCPERF_LAMMPS_LOCAL HPCPERF_QUICKSILVER_STEPS HPCPERF_QUICKSILVER_INPUT \
      HPCPERF_TEALEAF_INPUT HPCPERF_TEALEAF_DECK HPCPERF_SPARTA_INPUT HPCPERF_SPARTA_STRONG HPCPERF_SPARTA_LOCAL FAKE_MODE \
      HPCPERF_CLOVERLEAF_INPUT HPCPERF_CLOVERLEAF_DECK HPCPERF_LAGHOS_INPUT HPCPERF_LAGHOS_ARGS HPCPERF_LAGHOS_RS HPCPERF_LAGHOS_EPM HPCPERF_LAMMPS_VARIANT
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# ---- 1. every committed inputs.yaml validates; default_input is registered ----------------
for d in level1/background_subtraction level2/hipbone level2/quicksilver level3/lammps level2/tealeaf level3/sparta level2/cloverleaf level2/laghos; do
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
mkdir -p "$TMP/noregistry"; python3 "$TOOL" args "$TMP/noregistry" x >/dev/null 2>"$TMP/e"; rc=$?
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
mkbase() { # $1 bench_dir  $2 input_id  $3 quantities.json  -> stdout baseline json with workload
python3 - "$R" "$1" "$2" "$3" <<'PY'
import json, sys; sys.path.insert(0, sys.argv[1] + "/tools/inputs"); import hpcperf_inputs as hi
doc = hi.load(sys.argv[2]); inp = hi.get_input(doc, sys.argv[3])
print(json.dumps({"benchmark": doc["benchmark"], "input_id": inp["id"], "workload": hi.workload_identity(doc, inp), "quantities": json.load(open(sys.argv[4]))}))
PY
}
python3 "$TOOL" extract "$R/level2/hipbone" "$FX/hipbone_nx24_p14.log" > "$TMP/hb.json" 2>/dev/null
python3 - "$TMP/hb.json" <<'PY' && ok "hipbone: residual norms and iteration count extracted" || bad "hipbone extract"
import json,sys; d=json.load(open(sys.argv[1])); assert d["cg_iterations"]["value"]==100 and d["r_norm_final"]["value"]<1e-8 and d["dofs"]["value"]==37595375
PY
mkbase "$R/level2/hipbone" coral2-nx24-p14 "$TMP/hb.json" > "$TMP/hb_baseline.json"
python3 "$TOOL" compare "$R/level2/hipbone" "$TMP/hb_baseline.json" "$FX/hipbone_nx24_p14.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 3 ] && grep -q '"ok": true' "$TMP/c.json" && grep -q '"verdict": "INCOMPLETE"' "$TMP/c.json" && ok "hipbone: log vs its own baseline -> nothing fails, but exit 3 INCOMPLETE (required r_norm_final still record)" || bad "hipbone self-compare rc=$rc"
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
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['max_error']['value']==0 and d['pass_marker']['present']" "$TMP/bg.json" && mkbase "$R/level1/background_subtraction" w4096-h2048-merged0-r102 "$TMP/bg.json" > "$TMP/bg_base.json"
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
if [ -x "$R/build/level2/tealeaf/cuda/cuda-tealeaf" ]; then
    out="$(HPCPERF_TEALEAF_DECK=tea.in HPCPERF_TEALEAF_INPUT=bm4-1000sq-10steps "$R/level2/tealeaf/run.sh" CUDA 2>&1 | noise)"
    grep -q "mutually exclusive" <<<"$out" && ok "tealeaf run.sh: id + HPCPERF_TEALEAF_DECK refused" || bad "tealeaf id+deck: $out"
    out="$(HPCPERF_TEALEAF_INPUT=bm4-1000sq-10steps "$R/level2/tealeaf/run.sh" CUDA --solver cg 2>&1 | noise)"
    grep -q "mutually exclusive" <<<"$out" && ok "tealeaf run.sh: id + extra args refused" || bad "tealeaf id+args: $out"
    out="$(HPCPERF_TEALEAF_INPUT=bm6-8000sq-10steps "$R/level2/tealeaf/run.sh" CUDA 2>&1 | noise)"
    grep -q -- '--file .*/Benchmarks/tea_bm_6.in' <<<"$out" && grep -q -- '--out .*/tea.input.bm6-8000sq-10steps.out' <<<"$out" && ok "tealeaf run.sh: id resolves to its deck and a per-input log" || bad "tealeaf id deck: $out"
    out="$("$R/level2/tealeaf/run.sh" CUDA 2>&1 | noise)"
    grep -q -- '--file .*/Benchmarks/tea_bm_5.in' <<<"$out" && grep -q -- '--out .*/run/tea.out' <<<"$out" && ! grep -q 'input=' <<<"$out" && ok "tealeaf run.sh: default command unchanged" || bad "tealeaf default changed: $out"
else skip=$((skip+1)); echo "skip tealeaf run.sh (not built)"; fi
if [ -n "$(find "$R/build/level3/sparta/cuda" -maxdepth 2 -name spa_kokkos_cuda -type f 2>/dev/null)" ] && [ -f "$R/level3/sparta/src/bench/in.sphere" ]; then
    out="$(HPCPERF_SCALE_MODE=strong HPCPERF_SPARTA_INPUT=collide-1m "$R/level3/sparta/run.sh" CUDA 2>&1 | noise)"
    grep -q "mutually exclusive" <<<"$out" && ok "sparta run.sh: id + strong mode refused" || bad "sparta id+strong: $out"
    out="$(HPCPERF_SPARTA_LOCAL=20 HPCPERF_SPARTA_INPUT=collide-1m "$R/level3/sparta/run.sh" CUDA 2>&1 | noise)"
    grep -q "mutually exclusive" <<<"$out" && ok "sparta run.sh: id + size knob refused" || bad "sparta id+knob: $out"
    out="$(HPCPERF_SPARTA_INPUT=sphere-1m "$R/level3/sparta/run.sh" CUDA 2>&1 | noise)"
    grep -q -- '-in in.sphere -var x 40 -var y 50 -var z 50' <<<"$out" && grep -q 'log.input.sphere-1m.np1.sparta' <<<"$out" && ok "sparta run.sh: id resolves to deck + size + per-input log" || bad "sparta sphere: $out"
    out="$("$R/level3/sparta/run.sh" CUDA 2>&1 | noise)"
    grep -q -- '-in in.collide -var x 10 -var y 10 -var z 10' <<<"$out" && grep -q 'log.smoke.np1.sparta' <<<"$out" && ! grep -q 'input=' <<<"$out" && ok "sparta run.sh: default smoke command unchanged" || bad "sparta default changed: $out"
else skip=$((skip+1)); echo "skip sparta run.sh (not built/materialized)"; fi
unset HPCPERF_DRY_RUN

# ---- 7. frozen Level 3 source tree untouched by the input machinery -----------------------
if [ -f "$R/level3/lammps/provenance/source.lock.yaml" ] && [ -d "$R/level3/lammps/src" ]; then
    st="$("$R/tools/prepare_benchmark.sh" level3 lammps --status 2>&1 | noise | grep -o 'PREPARE STATUS lammps [A-Z_]*')"
    [ "$st" = "PREPARE STATUS lammps READY" ] && ok "lammps frozen tree still READY (source_tree_sha256 unchanged)" || bad "lammps frozen tree: $st"
else skip=$((skip+1)); echo "skip frozen-tree check (lammps not materialized)"; fi
if [ -f "$R/level3/sparta/provenance/source.lock.yaml" ] && [ -d "$R/level3/sparta/src" ]; then
    st="$("$R/tools/prepare_benchmark.sh" level3 sparta --status 2>&1 | noise | grep -o 'PREPARE STATUS sparta [A-Z_]*')"
    [ "$st" = "PREPARE STATUS sparta READY" ] && ok "sparta frozen tree still READY (source_tree_sha256 unchanged)" || bad "sparta frozen tree: $st"
else skip=$((skip+1)); echo "skip frozen-tree check (sparta not materialized)"; fi

# ---- 8. negative cases: the tool must never report an unverified result as verified ---------
# 8a. a science field missing from the log -> the comparison fails on that quantity
grep -v '^CG: initial res norm' "$FX/hipbone_nx24_p14.log" > "$TMP/hb_nofield.log"
python3 "$TOOL" compare "$R/level2/hipbone" "$TMP/hb_baseline.json" "$TMP/hb_nofield.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 1 ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); c={x['name']:x for x in d['checks']}; assert not d['ok'] and not d['verified'] and not c['r_norm_initial']['ok'] and 'no matching line' in c['r_norm_initial']['error']" "$TMP/c.json" 2>/dev/null \
    && ok "neg: missing science field (r_norm_initial) -> compare fails, verified=false" || bad "neg: missing field accepted (rc=$rc) $(cat "$TMP/c.json")"
# 8b. NaN / Inf in science fields -> not finite -> fails (also for a record-only quantity)
sed -e 's/^CG: initial res norm .*/CG: initial res norm inf /' -e 's/^CG: it 100, r norm [^,]*,/CG: it 100, r norm nan,/' "$FX/hipbone_nx24_p14.log" > "$TMP/hb_naninf.log"
python3 "$TOOL" compare "$R/level2/hipbone" "$TMP/hb_baseline.json" "$TMP/hb_naninf.log" >"$TMP/c.json" 2>/dev/null; rc=$?
# hipBone's value regexes use [0-9.eE+-]+, so a printed nan/inf does not match at all and surfaces as
# "no matching line"; a regex that does capture the token (\S+, fake benchmark below) yields "not finite".
# Both are failures of that quantity; neither ever becomes a float nan that compares equal to itself.
[ $rc -eq 1 ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); c={x['name']:x for x in d['checks']}; assert not d['ok'] and not c['r_norm_initial']['ok'] and ('not finite' in c['r_norm_initial']['error'] or 'no matching line' in c['r_norm_initial']['error']) and not c['r_norm_final']['ok'] and ('not finite' in c['r_norm_final']['error'] or 'no matching line' in c['r_norm_final']['error'])" "$TMP/c.json" 2>/dev/null \
    && ok "neg: inf initial residual + nan final residual -> both quantities fail (the record rule too)" || bad "neg: nan/inf accepted (rc=$rc) $(cat "$TMP/c.json")"
python3 "$TOOL" extract "$R/level2/hipbone" "$TMP/hb_naninf.log" 2>/dev/null | grep -q '"value": null' && ok "neg: extract reports nan/inf as no value (never a float nan)" || bad "neg: extract emitted a non-finite value"
# 8c. a modified final result (LAMMPS TotEng at step 100 shifted by 1e-3) -> the 1e-5 rule fails
python3 "$TOOL" extract "$R/level3/lammps" "$FX/lammps_lj.log" --input lj-32k > "$TMP/lj.json" 2>/dev/null
mkbase "$R/level3/lammps" lj-32k "$TMP/lj.json" > "$TMP/lj_baseline.json"
python3 - "$FX/lammps_lj.log" "$TMP/lj_mod.log" <<'PY'
import re,sys
out=[]
for ln in open(sys.argv[1]):
    if re.match(r'^\s+100\s+', ln):
        f=ln.split(); f[4]="%.7f" % (float(f[4])+1e-3); ln="  ".join(f)+"\n"    # TotEng column of thermo_style one
    out.append(ln)
open(sys.argv[2],"w").write("".join(out))
PY
python3 "$TOOL" compare "$R/level3/lammps" "$TMP/lj_baseline.json" "$FX/lammps_lj.log" --input lj-32k >/dev/null 2>&1 && ok "neg-control: unmodified lj log compares equal (rc 0, verified)" || bad "neg-control: unmodified lj log failed"
python3 "$TOOL" compare "$R/level3/lammps" "$TMP/lj_baseline.json" "$TMP/lj_mod.log" --input lj-32k >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 1 ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); c={x['name']:x for x in d['checks']}; assert not c['toteng_step100']['ok'] and c['toteng_step100']['rel_err']>1e-5 and c['temp_step100']['ok']" "$TMP/c.json" 2>/dev/null \
    && ok "neg: TotEng@100 shifted by 1e-3 -> toteng_step100 fails the 1e-5 rule, other fields still pass" || bad "neg: modified result accepted (rc=$rc) $(cat "$TMP/c.json")"
# 8d. a baseline that belongs to another input / another benchmark is refused (exit 2)
python3 "$TOOL" compare "$R/level2/hipbone" "$TMP/hb_baseline.json" "$FX/hipbone_nx24_p14.log" --input sweep-nx16-p8 >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -eq 2 ] && grep -q "belongs to input 'coral2-nx24-p14'" "$TMP/e" && ok "neg: baseline of another input -> exit 2, refused" || bad "neg: cross-input baseline accepted (rc=$rc) $(cat "$TMP/e")"
python3 "$TOOL" compare "$R/level2/hipbone" "$TMP/lj_baseline.json" "$FX/hipbone_nx24_p14.log" >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -eq 2 ] && grep -q "belongs to benchmark 'lammps'" "$TMP/e" && ok "neg: baseline of another benchmark -> exit 2, refused" || bad "neg: cross-benchmark baseline accepted (rc=$rc)"
# 8e. nonzero exit code: neither the timer nor the comparison uses the output
python3 "$TOOL" compare "$R/level2/hipbone" "$TMP/hb_baseline.json" "$FX/hipbone_nx24_p14.log" --rc 134 >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 1 ] && grep -q '"verified": false' "$TMP/c.json" && grep -q 'exited 134' "$TMP/c.json" && ok "neg: candidate exited 134 -> compare fails without reading the science fields" || bad "neg: failed candidate compared (rc=$rc)"
# 8f. exit 0 but the log says FAIL (background_subtraction really does this: printf FAIL; return 0)
python3 "$TOOL" compare "$R/level1/background_subtraction" "$TMP/bg_base.json" "$FX/bgsub_fail.log" --rc 0 >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 1 ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); c={x['name']:x for x in d['checks']}; assert not c['fail_marker']['ok'] and not c['pass_marker']['ok'] and not c['max_error']['ok']" "$TMP/c.json" 2>/dev/null \
    && ok "neg: exit 0 + 'FAIL' + 'Max error is 3' -> all three markers fail" || bad "neg: FAIL log with exit 0 accepted (rc=$rc)"
# 8h. record-only rules: nothing fails, but nothing is verified either -> exit 3, never 0
mkdir -p "$TMP/reconly"
cat > "$TMP/reconly/inputs.yaml" <<'EOF'
schema: hpcperf-inputs-1
benchmark: hipbone
level: 2
selector: HPCPERF_HIPBONE_INPUT
default_input: coral2-nx24-p14
entry: {kind: run.sh, path: level2/hipbone/run.sh, backend_arg: CUDA}
timing: {scope: x, kind: total, unit: s, regex: '^hipBone: \d+, \d+, (?P<value>[0-9.eE+-]+), \d+,', select: only}
baseline:
  quantities:
    - {name: r_norm_final, regex: '^CG: it 100, r norm (?P<value>[0-9.eE+-]+)', select: only, compare: {rule: record}}
    - {name: r_norm_initial, regex: '^CG: initial res norm (?P<value>[0-9.eE+-]+)', select: only, compare: {rule: record}}
inputs:
  - {id: coral2-nx24-p14, case: c, variant: default, source: {kind: upstream-parameterized, upstream: x}, params: {}, args: [], backends_validated: [cuda]}
EOF
mkbase "$TMP/reconly" coral2-nx24-p14 "$TMP/hb.json" > "$TMP/reconly_baseline.json"     # identity of the record-only registry's own input
python3 "$TOOL" compare "$TMP/reconly" "$TMP/reconly_baseline.json" "$FX/hipbone_99_iterations.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 3 ] && grep -q '"record_only": true' "$TMP/c.json" && grep -q '"verified": false' "$TMP/c.json" && ok "neg: all rules record-only -> exit 3 (inconclusive), verified=false, ok=true" || bad "neg: record-only compare returned rc=$rc $(cat "$TMP/c.json")"
# 8i. baseline and candidate are the same output file -> refused (exit 2)
mkbase "$R/level2/hipbone" coral2-nx24-p14 "$TMP/hb.json" | python3 -c "import json,sys; b=json.load(sys.stdin); b['log']=sys.argv[1]; print(json.dumps(b))" "$FX/hipbone_nx24_p14.log" > "$TMP/hb_same.json"
python3 "$TOOL" compare "$R/level2/hipbone" "$TMP/hb_same.json" "$FX/hipbone_nx24_p14.log" >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -eq 2 ] && grep -q "same output file" "$TMP/e" && ok "neg: baseline and candidate read the same file -> exit 2, refused" || bad "neg: same-file compare accepted (rc=$rc)"
cp "$FX/hipbone_nx24_p14.log" "$TMP/hb_copy.log"
python3 "$TOOL" compare "$R/level2/hipbone" "$TMP/hb_same.json" "$TMP/hb_copy.log" >/dev/null 2>&1; rc=$?
[ $rc -eq 3 ] && ok "neg-control: a different file with the same content is compared normally (exit 3 = hipBone's INCOMPLETE, not a refusal)" || bad "neg-control: copy refused (rc=$rc)"

# ---- 8j. compile-time inputs: an unmaterialized configuration is registered but never run (nothing substituted)
mkdir -p "$TMP/ct"
cat > "$TMP/ct/inputs.yaml" <<'EOF'
schema: hpcperf-inputs-1
benchmark: ct
level: 1
selector: null
default_input: class-b
entry: {kind: binary, path: build/fake/fake_bin}
timing: {scope: x, kind: total, unit: s, regex: '^time (?P<value>[0-9.]+) s$', select: only}
baseline: {quantities: [{name: pass_marker, regex: '^PASS$', compare: {rule: present}}]}
coverage: {status: BLOCKED, reason: other classes need regenerated build parameters, blocker: class headers not generated}
inputs:
  - {id: class-b, case: c, variant: build-config, input_form: compile-time, build_config: {CLASS: B}, source: {kind: upstream-parameterized, upstream: x}, params: {}, args: [], backends_validated: [cuda]}
  - {id: class-c, case: c, variant: build-config, input_form: compile-time, materialized: false, build_config: {CLASS: C}, source: {kind: upstream-parameterized, upstream: x}, params: {}, args: [], backends_validated: []}
EOF
python3 "$TOOL" validate "$TMP/ct" >/dev/null 2>&1 && ok "compile-time: registry with coverage BLOCKED and an unmaterialized class validates" || bad "compile-time registry invalid: $(python3 "$TOOL" validate "$TMP/ct" 2>&1 | noise)"
mkdir -p "$TMP/cov1" "$TMP/cov2" "$TMP/cov3"
sed -e 's/status: BLOCKED, reason: other classes need regenerated build parameters, blocker: class headers not generated/status: MULTI_INPUT/' "$TMP/ct/inputs.yaml" > "$TMP/cov1/inputs.yaml"
out="$(python3 "$TOOL" validate "$TMP/cov1" 2>&1 | noise || true)"; echo "$out" | grep -q 'MULTI_INPUT needs at least two runnable' && ok "coverage: MULTI_INPUT with one runnable input (the other unmaterialized) is refused" || bad "coverage MULTI_INPUT with one runnable input accepted"
sed -e 's/status: BLOCKED, reason: other classes need regenerated build parameters, blocker: class headers not generated/status: SINGLE_INPUT, reason: only class B is materialized/' "$TMP/ct/inputs.yaml" > "$TMP/cov2/inputs.yaml"
python3 "$TOOL" validate "$TMP/cov2" >/dev/null 2>&1 && ok "coverage: SINGLE_INPUT = exactly one runnable input validates" || bad "coverage SINGLE_INPUT with one runnable input refused: $(python3 "$TOOL" validate "$TMP/cov2" 2>&1 | noise)"
sed -e 's/materialized: false, //' "$TMP/ct/inputs.yaml" > "$TMP/cov3/inputs.yaml"
out="$(python3 "$TOOL" validate "$TMP/cov3" 2>&1 | noise || true)"; echo "$out" | grep -q 'BLOCKED means at most one runnable' && ok "coverage: BLOCKED with two runnable inputs is refused (it is MULTI_INPUT)" || bad "coverage BLOCKED with two runnable inputs accepted"
sed -e 's/materialized: false, //' -e 's/input_form: compile-time, build_config: {CLASS: C}/build_config: {CLASS: C}/' "$TMP/ct/inputs.yaml" > "$TMP/ct/bad.yaml"; mkdir -p "$TMP/ct2"; cp "$TMP/ct/bad.yaml" "$TMP/ct2/inputs.yaml"
python3 "$TOOL" validate "$TMP/ct2" 2>&1 | noise | grep -q 'compile-time input needs build_config\|materialized' ; [ $? -eq 0 ] || true
python3 "$TOOL" args "$TMP/ct" class-c >/dev/null 2>&1; rc=$?
[ $rc -eq 0 ] && ok "compile-time: args/show of an unmaterialized input still work (read/display)" || bad "compile-time args rc=$rc"
python3 - "$R" "$TMP/ct" <<'PY' && ok "compile-time: build_command refuses an unmaterialized input (never runs the class-B binary for it)" || bad "compile-time build_command did not refuse"
import sys; sys.path.insert(0, sys.argv[1] + "/tools/inputs"); import hpcperf_inputs as hi
from pathlib import Path
doc = hi.load(sys.argv[2]); inp = hi.get_input(doc, "class-c")
try:
    hi.build_command(doc, inp, Path(sys.argv[1]), Path(sys.argv[2]), 1); sys.exit(1)
except hi.InputError as ex:
    assert "not materialized" in str(ex); sys.exit(0)
PY
python3 - "$R" "$TMP/ct" <<'PY' && ok "compile-time: build_config is part of the workload identity (class-b != class-c)" || bad "compile-time workload identity"
import sys; sys.path.insert(0, sys.argv[1] + "/tools/inputs"); import hpcperf_inputs as hi
doc = hi.load(sys.argv[2]); a = hi.workload_identity(doc, hi.get_input(doc, "class-b")); b = hi.workload_identity(doc, hi.get_input(doc, "class-c"))
assert a["params"]["_build_config"] == {"CLASS": "B"} and hi.workload_mismatch(a, b)
PY
# ---- 8n. per-input timing override (a sweep-like input times its own summary line; another input has no timer)
mkdir -p "$TMP/tov/build/fake" "$TMP/tov/level1/tov"; touch "$TMP/tov/hpcperf_env.sh"
printf '#!/bin/sh\ncase "$1" in sweep) echo "block time 1.0 s"; echo "block time 2.0 s"; echo "best: sum 3.0 s";; short) echo "no timer here";; *) echo "time 4.0 s";; esac\necho PASS\n' > "$TMP/tov/build/fake/tov_bin"; chmod +x "$TMP/tov/build/fake/tov_bin"
cat > "$TMP/tov/level1/tov/inputs.yaml" <<'EOF'
schema: hpcperf-inputs-1
benchmark: tov
level: 1
selector: null
default_input: plain
entry: {kind: binary, path: build/fake/tov_bin}
timing: {scope: the benchmark's time line, kind: total, unit: s, regex: '^time (?P<value>[0-9.]+) s$', select: only}
baseline: {quantities: [{name: pass_marker, regex: '^PASS$', compare: {rule: present}}]}
coverage: {status: MULTI_INPUT}
inputs:
  - {id: plain, case: c, variant: default, source: {kind: upstream-file, upstream: x}, params: {}, args: [plain], backends_validated: [cuda]}
  - {id: sweep, case: c, variant: parameter, source: {kind: upstream-file, upstream: x}, params: {}, args: [sweep], backends_validated: [cuda],
     timing: {scope: the best block of the sweep, kind: total, unit: s, regex: '^best: sum (?P<value>[0-9.]+) s$', select: only}}
  - {id: short, case: c, variant: case, source: {kind: upstream-file, upstream: x}, params: {}, args: [short], backends_validated: [cuda],
     timing: {kind: none, status: NEEDS_TIMING_SUPPORT, reason: this deck is shorter than the timed window}}
EOF
python3 "$TOOL" validate "$TMP/tov/level1/tov" >/dev/null 2>&1 && ok "per-input timing: registry with two overrides validates" || bad "per-input timing registry invalid: $(python3 "$TOOL" validate "$TMP/tov/level1/tov" 2>&1 | noise)"
sed -e "s/regex: '^best: sum (?P<value>\[0-9.\]+) s\$'/regex: '^best: sum [0-9.]+ s$'/" "$TMP/tov/level1/tov/inputs.yaml" > "$TMP/tov/bad.yaml"; mkdir -p "$TMP/tov2/level1/tov"; cp "$TMP/tov/bad.yaml" "$TMP/tov2/level1/tov/inputs.yaml"
out="$(python3 "$TOOL" validate "$TMP/tov2/level1/tov" 2>&1 | noise || true)"; echo "$out" | grep -q "input 'sweep'.timing.regex needs a named group" && ok "per-input timing: the override is validated like the benchmark timer (regex without value group refused)" || bad "per-input timing override not validated: $out"
python3 "$TOOL" measure "$TMP/tov/level1/tov" sweep --out "$TMP/tov_sweep" --warmup 1 --reps 2 --timeout 30 >/dev/null 2>&1; rc=$?
python3 - "$TMP/tov_sweep/measurement.json" <<'PY' && ok "per-input timing: the sweep input is timed from its own summary line (3.0 s, timing_override recorded), exit 0" || bad "per-input timing sweep measurement (rc=$rc)"
import json, sys; d = json.load(open(sys.argv[1]))
assert d["timing_override"] is True and abs(d["summary"]["main_compute_s"]["median"] - 3.0) < 1e-9 and d["summary"]["timing_status"] == "NATIVE", d["summary"]
PY
python3 "$TOOL" measure "$TMP/tov/level1/tov" short --out "$TMP/tov_short" --warmup 1 --reps 2 --timeout 30 >/dev/null 2>&1; rc=$?
python3 - "$TMP/tov_short/measurement.json" <<'PY' && [ $rc -eq 0 ] && ok "per-input timing: kind none on one input -> NEEDS_TIMING_SUPPORT for that input only, runs recorded, exit 0" || bad "per-input timing none (rc=$rc)"
import json, sys; d = json.load(open(sys.argv[1]))
assert d["summary"]["timing_status"] == "NEEDS_TIMING_SUPPORT" and d["summary"]["run_completed"] is True and d["summary"]["main_compute_s"] is None, d["summary"]
PY
python3 "$TOOL" measure "$TMP/tov/level1/tov" plain --out "$TMP/tov_plain" --warmup 1 --reps 2 --timeout 30 >/dev/null 2>&1
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['timing_override'] is False and abs(d['summary']['main_compute_s']['median']-4.0)<1e-9" "$TMP/tov_plain/measurement.json" && ok "per-input timing: inputs without an override keep the benchmark timer" || bad "benchmark timer changed by the override feature"
# ---- 8p. the selector helper sourced a second time (run.sh -> hpcperf_launch_common.sh) keeps the input's arguments
out="$(bash -c 'source "$1/tools/inputs/hpcperf_input_selector.sh"; HPCPERF_REMHOS_INPUT=cube-remap-rs1; hpcperf_apply_input "$1/level2/remhos" HPCPERF_REMHOS_INPUT || exit 9; source "$1/tools/inputs/hpcperf_input_selector.sh"; printf "%s " "${HPCPERF_INPUT_ARGS[@]}"' _ "$R" 2>/dev/null)"
[ "$out" = "-rs 1 -dt 0.02 " ] && ok "selector helper: re-sourcing after hpcperf_apply_input keeps HPCPERF_INPUT_ARGS (remhos cube-remap-rs1: '$out')" || bad "selector helper re-source dropped the arguments: '$out'"
# ---- 8o. identity: the registry-side identity tools/timing stores with every ROI measurement of an input
python3 "$TOOL" identity "$R/level1/cg" class-c > "$TMP/id_c.json" 2>/dev/null; rc=$?
python3 - "$TMP/id_c.json" <<'PY' && [ $rc -eq 0 ] && ok "identity: NPB class-c -> its own binary, build_config, class header sha256, registry sha/git blob, complete" || bad "identity class-c (rc=$rc)"
import json, sys; d = json.load(open(sys.argv[1]))
assert d["schema"] == "hpcperf-workload-identity-1" and d["input_id"] == "class-c" and d["benchmark"] == "cg"
assert d["binary"] == "build/cg/cuda-classC/cg_cuda" and d["build_config"]["CLASS"] == "C"
assert d["build_files_sha256"]["level1/cg/inputs/npbparams.C.hpp"] and d["registry"]["sha256"] and d["registry"]["git_blob"]
assert d["workload"]["params"]["_build_config"]["CLASS"] == "C" and d["complete"] is True
PY
python3 "$TOOL" identity "$R/level2/sw4lite" pointsource-h0.02 > "$TMP/id_s.json" 2>/dev/null; rc=$?
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['selector']=='HPCPERF_SW4LITE_INPUT_ID' and d['workload']['files_sha256']['inputs/pointsource-h0p02.in'] and d['complete']" "$TMP/id_s.json" && [ $rc -eq 0 ] \
    && ok "identity: a deck input records its selector and the deck's sha256" || bad "identity sw4lite deck (rc=$rc)"
python3 "$TOOL" identity "$R/level2/sw4lite" no-such-input >/dev/null 2>&1; rc=$?
[ $rc -eq 2 ] && ok "identity: an unknown input id is refused (exit 2)" || bad "identity unknown id rc=$rc"
mkdir -p "$TMP/idmiss"; touch "$TMP/idmiss/hpcperf_env.sh"; mkdir -p "$TMP/idmiss/level1/m/data"
cat > "$TMP/idmiss/level1/m/inputs.yaml" <<'EOF'
schema: hpcperf-inputs-1
benchmark: m
level: 1
selector: null
default_input: a
entry: {kind: binary, path: build/m/m_bin}
timing: {scope: x, kind: total, unit: s, regex: '^time (?P<value>[0-9.]+) s$', select: only}
baseline: {quantities: [{name: pass_marker, regex: '^PASS$', compare: {rule: present}}]}
coverage: {status: SINGLE_INPUT, reason: test registry}
inputs:
  - {id: a, case: c, variant: default, source: {kind: upstream-file, upstream: x}, params: {}, args: ["build/m/data/gone.bin"], backends_validated: [cuda]}
  - {id: b, case: c, variant: build-config, input_form: compile-time, materialized: false, build_config: {CLASS: C}, source: {kind: upstream-parameterized, upstream: x}, params: {}, args: [], backends_validated: []}
EOF
python3 "$TOOL" identity "$TMP/idmiss/level1/m" a > "$TMP/id_m.json" 2>/dev/null; rc=$?
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['complete'] is False and d['arg_files_sha256']['build/m/data/gone.bin'] is None" "$TMP/id_m.json" && [ $rc -eq 3 ] \
    && ok "identity: a missing argument file -> complete false, exit 3 (no timing result for that input)" || bad "identity missing file rc=$rc"
python3 "$TOOL" identity "$TMP/idmiss/level1/m" b >/dev/null 2>&1; rc=$?
[ $rc -eq 3 ] && ok "identity: an unmaterialized compile-time input -> exit 3" || bad "identity unmaterialized rc=$rc"
# ---- 8l. repository-relative file arguments of a binary entry are passed as absolute paths (measure runs in the run dir)
mkdir -p "$TMP/relarg/build/fake"; printf '#!/bin/sh\nexit 0\n' > "$TMP/relarg/build/fake/fake_bin"; chmod +x "$TMP/relarg/build/fake/fake_bin"
mkdir -p "$TMP/relarg/level1/relarg/data"; echo x > "$TMP/relarg/level1/relarg/data/in.txt"
cat > "$TMP/relarg/level1/relarg/inputs.yaml" <<'EOF'
schema: hpcperf-inputs-1
benchmark: relarg
level: 1
selector: null
default_input: a
entry: {kind: binary, path: build/fake/fake_bin}
timing: {scope: x, kind: total, unit: s, regex: '^time (?P<value>[0-9.]+) s$', select: only}
baseline: {quantities: [{name: pass_marker, regex: '^PASS$', compare: {rule: present}}]}
inputs:
  - {id: a, case: c, variant: default, source: {kind: upstream-file, upstream: x}, params: {}, args: ["-f", "level1/relarg/data/in.txt", "-n", "8192", "level1/relarg/data/missing.txt"], files: [data/in.txt], backends_validated: [cuda]}
EOF
python3 - "$R" "$TMP/relarg" <<'PY' && ok "binary entry: an existing repo-relative path argument becomes absolute; other arguments (numbers, missing paths) are passed verbatim" || bad "repo-relative argument resolution"
import sys; sys.path.insert(0, sys.argv[1] + "/tools/inputs"); import hpcperf_inputs as hi
from pathlib import Path
root = Path(sys.argv[2]); doc = hi.load(str(root / "level1/relarg")); inp = hi.get_input(doc, "a")
cmd, env, exe = hi.build_command(doc, inp, root, root / "level1/relarg", 1)
assert cmd[2] == str(root / "level1/relarg/data/in.txt"), cmd
assert cmd[4] == "8192" and cmd[5] == "level1/relarg/data/missing.txt", cmd
assert hi.workload_identity(doc, inp)["args"][1] == "level1/relarg/data/in.txt"
PY
# ---- 8m. measure stops after a timed-out run (the remaining repetitions are not started; the record says so)
mkdir -p "$TMP/slow/build/fake" "$TMP/slow/level1/slow"; touch "$TMP/slow/hpcperf_env.sh"; printf '#!/bin/sh\nsleep 3\necho "time 1.0 s"\necho PASS\n' > "$TMP/slow/build/fake/slow_bin"; chmod +x "$TMP/slow/build/fake/slow_bin"
sed -e 's/benchmark: relarg/benchmark: slow/' -e 's#build/fake/fake_bin#build/fake/slow_bin#' -e 's/args: \[.*\], files: \[data\/in.txt\], /args: [], /' "$TMP/relarg/level1/relarg/inputs.yaml" > "$TMP/slow/level1/slow/inputs.yaml"
python3 "$TOOL" measure "$TMP/slow/level1/slow" a --out "$TMP/slow_out" --warmup 1 --reps 3 --timeout 1 >/dev/null 2>&1; rc=$?
python3 - "$TMP/slow_out/measurement.json" <<'PY' && ok "measure: a timed-out warm-up (rc 124) stops the measurement -- 1 run recorded, 'aborted' note, exit 1" || bad "measure timeout abort (rc=$rc): $(python3 -c "import json,sys; d=json.load(open('$TMP/slow_out/measurement.json')); print(len(d['runs']), d.get('aborted'))" 2>&1)"
import json, sys; d = json.load(open(sys.argv[1]))
assert len(d["runs"]) == 1 and d["runs"][0]["exit_code"] == 124 and "timeout" in d.get("aborted", ""), (len(d["runs"]), d.get("aborted"))
assert d["summary"]["run_completed"] is False
PY
[ $rc -eq 1 ] || bad "measure timeout abort exit code $rc (expected 1)"

# ---- 8k. the shared run.sh selector helper (hpcperf_apply_input): knobs exported, args collected, conflicts refused
mkdir -p "$TMP/sel/level2/selapp"
cat > "$TMP/sel/level2/selapp/inputs.yaml" <<'EOF'
schema: hpcperf-inputs-1
benchmark: selapp
level: 2
selector: HPCPERF_SELAPP_INPUT
default_input: big
entry: {kind: run.sh, path: level2/selapp/run.sh}
timing: {scope: x, kind: total, unit: s, regex: '^time (?P<value>[0-9.]+) s$', select: only}
baseline: {quantities: [{name: pass_marker, regex: '^PASS$', compare: {rule: present}}]}
inputs:
  - {id: big, case: c, variant: default, source: {kind: upstream-parameterized, upstream: x}, params: {n: 256}, env: {SELAPP_N: "256", HPCPERF_SCALE_MODE: weak}, args: ["--extra", "1"], backends_validated: [cuda]}
EOF
: > "$TMP/sel/hpcperf_env.sh"; mkdir -p "$TMP/sel/level1" "$TMP/sel/tools/inputs"; cp "$TOOL" "$TMP/sel/tools/inputs/hpcperf_inputs.py"
sel_out="$(cd "$TMP/sel" && bash -c 'source "'"$R"'/tools/inputs/hpcperf_input_selector.sh"; unset SELAPP_N HPCPERF_SCALE_MODE; export HPCPERF_SELAPP_INPUT=big; hpcperf_apply_input level2/selapp HPCPERF_SELAPP_INPUT && echo "N=$SELAPP_N MODE=$HPCPERF_SCALE_MODE ID=$HPCPERF_INPUT_ID ARGS=${HPCPERF_INPUT_ARGS[*]}"' 2>&1 | noise)"
grep -q 'N=256 MODE=weak ID=big ARGS=--extra 1' <<<"$sel_out" && ok "selector helper: knobs exported, args collected, input id recorded" || bad "selector helper: $sel_out"
sel_out="$(cd "$TMP/sel" && bash -c 'source "'"$R"'/tools/inputs/hpcperf_input_selector.sh"; export SELAPP_N=128 HPCPERF_SELAPP_INPUT=big; hpcperf_apply_input level2/selapp HPCPERF_SELAPP_INPUT; echo "rc=$?"' 2>&1 | noise)"
grep -q 'mutually exclusive' <<<"$sel_out" && grep -q 'rc=2' <<<"$sel_out" && ok "selector helper: a conflicting pre-set knob is refused (exit 2)" || bad "selector helper conflict: $sel_out"
sel_out="$(cd "$TMP/sel" && bash -c 'source "'"$R"'/tools/inputs/hpcperf_input_selector.sh"; export HPCPERF_SELAPP_INPUT=nope; hpcperf_apply_input level2/selapp HPCPERF_SELAPP_INPUT; echo "rc=$?"' 2>&1 | noise)"
grep -q 'rc=2' <<<"$sel_out" && ok "selector helper: unknown id -> exit 2" || bad "selector helper unknown: $sel_out"
sel_out="$(cd "$TMP/sel" && bash -c 'source "'"$R"'/tools/inputs/hpcperf_input_selector.sh"; unset HPCPERF_SELAPP_INPUT; hpcperf_apply_input level2/selapp HPCPERF_SELAPP_INPUT; echo "rc=$? ID=${HPCPERF_INPUT_ID:-none}"' 2>&1 | noise)"
grep -q 'rc=0 ID=none' <<<"$sel_out" && ok "selector helper: without the selector nothing changes" || bad "selector helper noop: $sel_out"

# ---- 9. measure end-to-end on a fake benchmark (no GPU): exit codes, FAIL-with-exit-0, stale logs, record-only ----
FR="$TMP/fakerepo"; mkdir -p "$FR/level1/fake" "$FR/level1/fakerec" "$FR/build/fake"; : > "$FR/hpcperf_env.sh"
cat > "$FR/build/fake/fake_bin" <<'EOF'
#!/bin/bash
case "${FAKE_MODE:-ok}" in
  ok)      echo "time 1.500 s"; echo "energy 42.000000"; echo "PASS" ;;
  exit1)   echo "starting"; exit 1 ;;
  failmark) echo "time 1.500 s"; echo "energy 42.000000"; echo "FAIL" ;;
  nan)     echo "time 1.500 s"; echo "energy nan"; echo "PASS" ;;
  notimer) echo "energy 42.000000"; echo "PASS" ;;
esac
EOF
chmod +x "$FR/build/fake/fake_bin"
cat > "$FR/level1/fake/inputs.yaml" <<'EOF'
schema: hpcperf-inputs-1
benchmark: fake
level: 1
selector: null
default_input: a
entry: {kind: binary, path: build/fake/fake_bin}
timing: {scope: fake timer, kind: total, unit: s, regex: '^time (?P<value>[0-9.]+) s$', select: only}
baseline:
  quantities:
    - {name: energy, regex: '^energy (?P<value>\S+)$', select: only, compare: {rule: rel, tol: 1.0e-6}}
    - {name: pass_marker, regex: '^PASS$', compare: {rule: present}}
    - {name: fail_marker, regex: '^FAIL$', compare: {rule: absent}}
inputs:
  - {id: a, case: c, variant: default, source: {kind: custom, derivation: test}, params: {}, args: [], backends_validated: []}
EOF
sed -e 's/^benchmark: fake$/benchmark: fakerec/' -e "s/compare: {rule: rel, tol: 1.0e-6}/compare: {rule: record}/" -e '/pass_marker/d' -e '/fail_marker/d' "$FR/level1/fake/inputs.yaml" > "$FR/level1/fakerec/inputs.yaml"
mrun() { python3 "$TOOL" measure "$FR/level1/$1" a --out "$TMP/m_$2" --warmup 1 --reps 2 --timeout 30 >"$TMP/m_$2.out" 2>&1; echo $?; }
rc=$(FAKE_MODE=ok mrun fake ok)
[ "$rc" -eq 0 ] && grep -q 'run_completed=True timing_ok=True timing_status=NATIVE native_check=PASS baseline_saved=True comparison_rules=READY' "$TMP/m_ok.out" && [ -f "$TMP/m_ok/baseline.json" ] && ok "measure: healthy fake run -> completed, native PASS, baseline saved, rules READY" || bad "measure ok: rc=$rc $(cat "$TMP/m_ok.out")"
rc=$(FAKE_MODE=exit1 mrun fake exit1)
[ "$rc" -eq 1 ] && grep -q 'run_completed=False timing_ok=False' "$TMP/m_exit1.out" && [ ! -f "$TMP/m_exit1/baseline.json" ] && grep -q '"main_compute_s": null' "$TMP/m_exit1/rep1/result.json" && ok "measure: binary exits 1 -> run_completed=False, no timer value, no baseline, exit 1" || bad "measure exit1: rc=$rc $(cat "$TMP/m_exit1.out")"
rc=$(FAKE_MODE=failmark mrun fake failmark)
grep -q 'native_check=FAIL' "$TMP/m_failmark.out" && grep -q 'baseline_saved=False' "$TMP/m_failmark.out" && grep -q 'run_completed=True' "$TMP/m_failmark.out" && ok "measure: exit 0 but 'FAIL' printed -> run_completed=True, native_check=FAIL, baseline NOT saved" || bad "measure failmark: rc=$rc $(cat "$TMP/m_failmark.out")"
rc=$(FAKE_MODE=nan mrun fake nan)
grep -q 'baseline_self_consistent=False' "$TMP/m_nan.out" && grep -q '"value": null' "$TMP/m_nan/rep1/result.json" && ok "measure: 'energy nan' -> quantity has no value, baseline not self-consistent" || bad "measure nan: rc=$rc $(cat "$TMP/m_nan.out")"
rc=$(FAKE_MODE=notimer mrun fake notimer)
[ "$rc" -eq 1 ] && grep -q 'timing_ok=False' "$TMP/m_notimer.out" && grep -q 'no matching line' "$TMP/m_notimer/rep1/result.json" && ! grep -q '"main_compute_s": [0-9]' "$TMP/m_notimer/rep1/result.json" && ok "measure: no timer line -> timing_ok=False, wall time never substituted, exit 1" || bad "measure notimer: rc=$rc"
# stale log: a previous run's stdout.log in the run directory must never be read
mkdir -p "$TMP/m_stale/rep1"; printf 'time 9.999 s\nenergy 42.000000\nPASS\n' > "$TMP/m_stale/rep1/stdout.log"
rc=$(FAKE_MODE=exit1 mrun fake stale)
! grep -q '9.999' "$TMP/m_stale/rep1/stdout.log" && grep -q '"main_compute_s": null' "$TMP/m_stale/rep1/result.json" && grep -q '"exit_code": 1' "$TMP/m_stale/rep1/result.json" && ok "measure: stale stdout.log from an earlier run is removed, not reused" || bad "measure stale: rc=$rc $(head -5 "$TMP/m_stale/rep1/result.json" 2>/dev/null)"
rc=$(FAKE_MODE=ok mrun fakerec rec)
grep -q 'comparison_rules=NONE' "$TMP/m_rec.out" && grep -q 'NEEDS_VALIDATION=energy' "$TMP/m_rec.out" && grep -q 'native_check=NONE' "$TMP/m_rec.out" && ok "measure: record-only rules -> comparison_rules=NONE, NEEDS_VALIDATION listed, native_check=NONE (never PASS)" || bad "measure record-only: $(cat "$TMP/m_rec.out")"
python3 "$TOOL" compare "$FR/level1/fakerec" "$TMP/m_rec/baseline.json" "$TMP/m_rec/rep2/stdout.log" >/dev/null 2>&1; rc=$?
[ $rc -eq 3 ] && ok "compare: record-only baseline vs a later run -> exit 3, not 0" || bad "compare record-only rc=$rc"
python3 "$TOOL" compare "$FR/level1/fakerec" "$TMP/m_rec/baseline.json" "$TMP/m_rec/rep1/stdout.log" >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -eq 2 ] && grep -q "same output file" "$TMP/e" && ok "compare: baseline.json's own source log as candidate -> refused" || bad "compare same log rc=$rc"
python3 "$TOOL" status "$FR/level1/fakerec" "$TMP/m_rec/measurement.json" 2>/dev/null | grep -q 'NEEDS_VALIDATION=energy' && ok "status: re-derives the vocabulary from measurement.json" || bad "status subcommand"

# ---- 10. acceptance semantics of compare: exit 0 only for a COMPLETE, passing science comparison ----
# 10.1 one rule passes (dofs) and one fails (cg_iterations 99 != 100): the whole comparison fails (exit 1, FAIL)
python3 "$TOOL" compare "$R/level2/hipbone" "$TMP/hb_baseline.json" "$FX/hipbone_99_iterations.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 1 ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); c={x['name']:x for x in d['checks']}; assert d['verdict']=='FAIL' and not d['ok'] and not d['verified'] and c['dofs']['ok'] and not c['cg_iterations']['ok'] and d['failed']==['cg_iterations']" "$TMP/c.json" 2>/dev/null \
    && ok "acceptance: dofs passes + cg_iterations fails -> exit 1, verdict FAIL (a pass never outweighs a failure)" || bad "acceptance 10.1: rc=$rc $(cat "$TMP/c.json")"
# 10.2 configuration checks pass (iterations, DOFs, initial residual) but the REQUIRED final result is only recorded:
#      exit 3 / INCOMPLETE -- never 0, even though nothing failed (the real hipBone registry)
python3 "$TOOL" compare "$R/level2/hipbone" "$TMP/hb_baseline.json" "$FX/hipbone_nx24_p14.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 3 ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); c={x['name']:x for x in d['checks']}; assert d['ok'] and not d['complete'] and not d['verified'] and d['verdict']=='INCOMPLETE' and d['required_pending']==['r_norm_final'] and c['cg_iterations']['ok'] and c['dofs']['ok'] and c['r_norm_initial']['ok'] and c['r_norm_initial']['role']=='diagnostic'" "$TMP/c.json" 2>/dev/null \
    && ok "acceptance: iterations/DOFs/initial residual pass, final residual only recorded -> exit 3, verdict INCOMPLETE, required_pending=[r_norm_final]" || bad "acceptance 10.2: rc=$rc $(cat "$TMP/c.json")"
# 10.3 every REQUIRED result verified, one DIAGNOSTIC quantity only recorded -> exit 0 / PASS; a missing diagnostic is noted, not a failure
mkdir -p "$FR/level1/fakediag"
cat > "$FR/level1/fakediag/inputs.yaml" <<'EOF'
schema: hpcperf-inputs-1
benchmark: fakediag
level: 1
selector: null
default_input: a
entry: {kind: binary, path: build/fake/fake_bin}
timing: {scope: fake timer, kind: total, unit: s, regex: '^time (?P<value>[0-9.]+) s$', select: only}
baseline:
  quantities:
    - {name: energy, role: required, regex: '^energy (?P<value>\S+)$', select: only, compare: {rule: rel, tol: 1.0e-6}}
    - {name: pass_marker, role: required, regex: '^PASS$', compare: {rule: present}}
    - {name: iterations, role: diagnostic, regex: '^iterations (?P<value>\d+)$', select: only, compare: {rule: record}}
inputs:
  - {id: a, case: c, variant: default, source: {kind: custom, derivation: test}, params: {}, args: [], backends_validated: []}
EOF
printf 'time 1.500 s\nenergy 42.000000\niterations 17\nPASS\n' > "$TMP/diag_base.log"; printf 'time 1.400 s\nenergy 42.000010\niterations 19\nPASS\n' > "$TMP/diag_cand.log"; printf 'time 1.400 s\nenergy 42.000010\nPASS\n' > "$TMP/diag_nodiag.log"
python3 "$TOOL" extract "$FR/level1/fakediag" "$TMP/diag_base.log" > "$TMP/dq.json" 2>/dev/null
mkbase "$FR/level1/fakediag" a "$TMP/dq.json" > "$TMP/diag_baseline.json"
python3 "$TOOL" compare "$FR/level1/fakediag" "$TMP/diag_baseline.json" "$TMP/diag_cand.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 0 ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['verdict']=='PASS' and d['verified'] and d['complete'] and d['diagnostic_recorded']==['iterations'] and d['required_pending']==[]" "$TMP/c.json" 2>/dev/null \
    && ok "acceptance: required energy (rel 1e-6) + marker pass, diagnostic iterations recorded (17 -> 19) -> exit 0, verdict PASS" || bad "acceptance 10.3: rc=$rc $(cat "$TMP/c.json")"
python3 "$TOOL" compare "$FR/level1/fakediag" "$TMP/diag_baseline.json" "$TMP/diag_nodiag.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 0 ] && grep -q '"diagnostic_missing": \[' "$TMP/c.json" && grep -q '"iterations"' "$TMP/c.json" && ok "acceptance: a missing DIAGNOSTIC field is reported (diagnostic_missing) but does not fail the comparison" || bad "acceptance 10.3b: rc=$rc"
printf 'time 1.400 s\nenergy 42.001000\niterations 19\nPASS\n' > "$TMP/diag_bad.log"
python3 "$TOOL" compare "$FR/level1/fakediag" "$TMP/diag_baseline.json" "$TMP/diag_bad.log" >/dev/null 2>&1; rc=$?
[ $rc -eq 1 ] && ok "acceptance: the same registry with the required energy off by 2.4e-5 -> exit 1 (a failure is never downgraded to incomplete)" || bad "acceptance 10.3c: rc=$rc"
# 10.4 PARTIAL propagates through CLI exit code, JSON, measure summary, status line and a shell caller -- never PASS
mkdir -p "$FR/level1/fakepartial"
cat > "$FR/level1/fakepartial/inputs.yaml" <<'EOF'
schema: hpcperf-inputs-1
benchmark: fakepartial
level: 1
selector: null
default_input: a
entry: {kind: binary, path: build/fake/fake_bin}
timing: {scope: fake timer, kind: total, unit: s, regex: '^time (?P<value>[0-9.]+) s$', select: only}
baseline:
  quantities:
    - {name: pass_marker, role: required, regex: '^PASS$', compare: {rule: present}}
    - {name: energy, role: required, regex: '^energy (?P<value>\S+)$', select: only, compare: {rule: record}}
inputs:
  - {id: a, case: c, variant: default, source: {kind: custom, derivation: test}, params: {}, args: [], backends_validated: []}
EOF
rc=$(FAKE_MODE=ok mrun fakepartial partial)
grep -q 'comparison_rules=PARTIAL' "$TMP/m_partial.out" && grep -q 'baseline_verdict=INCOMPLETE' "$TMP/m_partial.out" && grep -q 'NEEDS_VALIDATION=energy' "$TMP/m_partial.out" && grep -q 'native_check=PASS' "$TMP/m_partial.out" \
    && ok "propagation: measure -> native_check=PASS (marker) but comparison_rules=PARTIAL, baseline_verdict=INCOMPLETE, NEEDS_VALIDATION=energy" || bad "propagation measure: $(cat "$TMP/m_partial.out")"
python3 -c "import json,sys; s=json.load(open(sys.argv[1]))['summary']; assert s['comparison_rules']=='PARTIAL' and s['baseline_verdict']=='INCOMPLETE' and s['needs_validation']==['energy'] and all(not c['verified'] for c in s['baseline_checks'])" "$TMP/m_partial/measurement.json" 2>/dev/null \
    && ok "propagation: measurement.json carries comparison_rules=PARTIAL, baseline_verdict=INCOMPLETE, every self-check verified=false" || bad "propagation json"
python3 "$TOOL" status "$FR/level1/fakepartial" "$TMP/m_partial/measurement.json" 2>/dev/null | grep -q 'comparison_rules=PARTIAL baseline_verdict=INCOMPLETE' && ok "propagation: status re-derives PARTIAL/INCOMPLETE" || bad "propagation status"
python3 "$TOOL" compare "$FR/level1/fakepartial" "$TMP/m_partial/baseline.json" "$TMP/m_partial/rep2/stdout.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 3 ] && grep -q '"verdict": "INCOMPLETE"' "$TMP/c.json" && grep -q '"verified": false' "$TMP/c.json" && ok "propagation: compare CLI -> exit 3, JSON verdict INCOMPLETE / verified false" || bad "propagation cli rc=$rc"
caller_saw=none
if python3 "$TOOL" compare "$FR/level1/fakepartial" "$TMP/m_partial/baseline.json" "$TMP/m_partial/rep2/stdout.log" >/dev/null 2>&1; then caller_saw=PASS; else case $? in 0) caller_saw=PASS;; 1) caller_saw=FAIL;; 2) caller_saw=REFUSED;; 3) caller_saw=INCOMPLETE;; esac; fi
[ "$caller_saw" = INCOMPLETE ] && ok "propagation: a shell caller using 'if compare' does not take the PASS branch (sees INCOMPLETE)" || bad "propagation caller saw $caller_saw"
# 10.5 same input_id, different WORKLOAD (params) -> refused; same workload, different CODE identity -> compared normally
python3 - "$TMP/m_ok/baseline.json" > "$TMP/wl_diff.json" <<'PY'
import json,sys; b=json.load(open(sys.argv[1])); b["workload"]=dict(b["workload"]); b["workload"]["params"]={"n": 10}; print(json.dumps(b))
PY
python3 "$TOOL" compare "$FR/level1/fake" "$TMP/wl_diff.json" "$TMP/m_ok/rep2/stdout.log" >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -eq 2 ] && grep -q "differs in: params" "$TMP/e" && ok "identity: same input_id, baseline recorded with other params -> exit 2, refused (names the differing key)" || bad "identity workload rc=$rc $(cat "$TMP/e")"
python3 - "$TMP/m_ok/baseline.json" > "$TMP/code_diff.json" <<'PY'
import json,sys; b=json.load(open(sys.argv[1])); b["code_identity"]={"entry_sha256":"0000deadbeef","git":{"head":"optimized-branch","dirty":True}}; print(json.dumps(b))
PY
python3 "$TOOL" compare "$FR/level1/fake" "$TMP/code_diff.json" "$TMP/m_ok/rep2/stdout.log" >/dev/null 2>&1; rc=$?
[ $rc -eq 0 ] && ok "identity: same workload, different binary/git identity (an optimized build) -> compared normally, exit 0" || bad "identity code rc=$rc"
grep -q '"workload"' "$TMP/m_ok/baseline.json" && grep -q '"files_sha256"' "$TMP/m_ok/baseline.json" && grep -q 'informational only' "$TMP/m_ok/baseline.json" && ok "identity: baseline.json records workload (params/args/env/files sha256) and code identity separately" || bad "identity baseline fields"
python3 - "$TMP/hb_baseline.json" > "$TMP/hb_legacy.json" <<'PY'
import json,sys; b=json.load(open(sys.argv[1])); del b["workload"]; print(json.dumps(b))
PY
python3 "$TOOL" compare "$R/level2/hipbone" "$TMP/hb_legacy.json" "$TMP/hb_copy.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 3 ] && grep -q 'no workload identity' "$TMP/c.json" && grep -q '"workload_status": "not-established"' "$TMP/c.json" && ok "identity: a baseline without workload identity (pre round 3) is shown but exit 3 INCOMPLETE, never PASS" || bad "identity legacy rc=$rc"

# ---- 11. workload identity gate: no formal PASS without an ESTABLISHED identity on both sides ----
# helper: a baseline JSON carrying the registry's workload identity (what `measure` writes)
# 11a. baseline lacks workload (a pre-round-3 record): shown, but exit 3 / INCOMPLETE -- never 0
python3 - "$TMP/m_ok/baseline.json" > "$TMP/wl_none.json" <<'PY'
import json,sys; b=json.load(open(sys.argv[1])); b.pop("workload", None); b.pop("workload_migration", None); print(json.dumps(b))
PY
python3 "$TOOL" compare "$FR/level1/fake" "$TMP/wl_none.json" "$TMP/m_ok/rep2/stdout.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 3 ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['ok'] and not d['verified'] and d['verdict']=='INCOMPLETE' and d['workload_status']=='not-established' and '(workload identity not established)' in d['required_pending'] and d['checks']" "$TMP/c.json" 2>/dev/null \
    && ok "identity gate: baseline without workload -> values shown, exit 3 INCOMPLETE (workload_status=not-established), never PASS" || bad "identity gate 11a rc=$rc $(cat "$TMP/c.json")"
# 11b. candidate side unknown (no --input, baseline names no input): exit 3, not 0
python3 - "$TMP/m_ok/baseline.json" > "$TMP/wl_noinput.json" <<'PY'
import json,sys; b=json.load(open(sys.argv[1])); b.pop("input_id", None); print(json.dumps(b))
PY
python3 "$TOOL" compare "$FR/level1/fake" "$TMP/wl_noinput.json" "$TMP/m_ok/rep2/stdout.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 3 ] && grep -q '"workload_status": "not-established"' "$TMP/c.json" && ok "identity gate: candidate workload unknown (no input id on either side) -> exit 3, not PASS" || bad "identity gate 11b rc=$rc"
# 11c. identity present but incomplete / empty
for variant in empty missing_files null_params; do
python3 - "$TMP/m_ok/baseline.json" "$variant" > "$TMP/wl_$variant.json" <<'PY'
import json,sys; b=json.load(open(sys.argv[1])); v=sys.argv[2]
if v=="empty": b["workload"]={}
elif v=="missing_files": b["workload"].pop("files_sha256")
elif v=="null_params": b["workload"]["params"]=None
print(json.dumps(b))
PY
python3 "$TOOL" compare "$FR/level1/fake" "$TMP/wl_$variant.json" "$TMP/m_ok/rep2/stdout.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 3 ] && grep -q '"workload_status": "not-established"' "$TMP/c.json" && ok "identity gate: workload $variant -> exit 3 INCOMPLETE" || bad "identity gate 11c $variant rc=$rc"
done
# 11d. deleting the identity from a MISMATCHING baseline turns a refusal (2) into INCOMPLETE (3), never into 0
python3 "$TOOL" compare "$FR/level1/fake" "$TMP/wl_diff.json" "$TMP/m_ok/rep2/stdout.log" >/dev/null 2>&1; rc1=$?
python3 - "$TMP/wl_diff.json" > "$TMP/wl_diff_stripped.json" <<'PY'
import json,sys; b=json.load(open(sys.argv[1])); del b["workload"]; print(json.dumps(b))
PY
python3 "$TOOL" compare "$FR/level1/fake" "$TMP/wl_diff_stripped.json" "$TMP/m_ok/rep2/stdout.log" >/dev/null 2>&1; rc2=$?
[ $rc1 -eq 2 ] && [ $rc2 -eq 3 ] && ok "identity gate: mismatching baseline refused (2); with its identity deleted -> 3, not 0" || bad "identity gate 11d rc1=$rc1 rc2=$rc2"
# 11e. a failing rule still fails (1) even when the identity is not established
printf 'time 1.500 s\nenergy 43.000000\nPASS\n' > "$TMP/wl_badcand.log"
python3 "$TOOL" compare "$FR/level1/fake" "$TMP/wl_none.json" "$TMP/wl_badcand.log" >/dev/null 2>&1; rc=$?
[ $rc -eq 1 ] && ok "identity gate: identity not established AND energy off -> exit 1 (failure is never masked as incomplete)" || bad "identity gate 11e rc=$rc"
# 11f. migration with trusted evidence (the measurement.json the baseline came from) -> new file, then exit 0
python3 "$TOOL" migrate-baseline "$FR/level1/fake" "$TMP/wl_none.json" --input a --evidence "$TMP/m_ok/measurement.json" --note "test migration" --out "$TMP/wl_migrated.json" >"$TMP/mig.json" 2>"$TMP/e"; rc=$?
[ $rc -eq 0 ] && grep -q '"workload_migration"' "$TMP/wl_migrated.json" && grep -q '"source"' "$TMP/wl_migrated.json" && grep -q 'test migration' "$TMP/wl_migrated.json" && [ -f "$TMP/wl_none.json" ] && ! grep -q '"workload"' "$TMP/wl_none.json" \
    && ok "migration: identity attached from the evidence measurement into a NEW file with source/evidence/basis; the old record is untouched" || bad "migration 11f rc=$rc $(cat "$TMP/e")"
python3 "$TOOL" compare "$FR/level1/fake" "$TMP/wl_migrated.json" "$TMP/m_ok/rep2/stdout.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 0 ] && grep -q '"workload_status": "established"' "$TMP/c.json" && grep -q 'migrated:' "$TMP/c.json" && ok "migration: the migrated record compares formally (exit 0, workload established, migration source shown)" || bad "migration compare rc=$rc"
python3 "$TOOL" migrate-baseline "$FR/level1/fake" "$TMP/wl_none.json" --input a --evidence "$TMP/m_ok/measurement.json" --out "$TMP/wl_migrated.json" >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -ne 0 ] && grep -q 'not overwriting' "$TMP/e" && ok "migration: never overwrites an existing migration record" || bad "migration overwrite rc=$rc"
# 11g. migration is refused when the evidence does not fit (other input id / other registry file / other command)
python3 - "$TMP/m_ok/measurement.json" > "$TMP/ev_bad.json" <<'PY'
import json,sys; m=json.load(open(sys.argv[1])); m["input_id"]="b"; print(json.dumps(m))
PY
python3 "$TOOL" migrate-baseline "$FR/level1/fake" "$TMP/wl_none.json" --input a --evidence "$TMP/ev_bad.json" --out "$TMP/wl_mig_bad.json" >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -ne 0 ] && grep -q "migration refused" "$TMP/e" && [ ! -f "$TMP/wl_mig_bad.json" ] && ok "migration: evidence naming another input id -> refused, nothing written" || bad "migration 11g rc=$rc $(cat "$TMP/e")"
python3 - "$TMP/m_ok/measurement.json" > "$TMP/ev_bad2.json" <<'PY'
import json,sys; m=json.load(open(sys.argv[1])); m["inputs_yaml_sha256"]="0"*64; print(json.dumps(m))
PY
python3 "$TOOL" migrate-baseline "$FR/level1/fake" "$TMP/wl_none.json" --input a --evidence "$TMP/ev_bad2.json" --out "$TMP/wl_mig_bad2.json" >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -ne 0 ] && grep -q "inputs_yaml_sha256" "$TMP/e" && ok "migration: evidence recorded under another inputs.yaml with a dirty/unknown git tree -> refused (entry at run time not recoverable)" || bad "migration 11g2 rc=$rc"
# a clean commit whose registry entry equals the current one is accepted as evidence (real repo history: hipbone coral2-nx24-p14 at c6efde5)
if git -C "$R" cat-file -e c6efde5:level2/hipbone/inputs.yaml 2>/dev/null; then
python3 - "$TMP/hb_legacy.json" "$FX/hipbone_nx24_p14.log" "$R" > "$TMP/hb_ev.json" <<'PY'
import json,sys,hashlib
print(json.dumps({"benchmark":"hipbone","input_id":"coral2-nx24-p14","inputs_yaml_sha256":"not-the-current-sha","selector":{"HPCPERF_HIPBONE_INPUT":"coral2-nx24-p14"},
                  "command":["bash","run.sh","CUDA"],"git":{"head":"c6efde5","dirty":False},"runs":[{"label":"rep1","log":sys.argv[2]}],"started_utc":"2026-09-21T00:00:00Z"}))
PY
python3 - "$TMP/hb_legacy.json" "$FX/hipbone_nx24_p14.log" > "$TMP/hb_legacy2.json" <<'PY'
import json,sys; b=json.load(open(sys.argv[1])); b["log"]=sys.argv[2]; print(json.dumps(b))
PY
python3 "$TOOL" migrate-baseline "$R/level2/hipbone" "$TMP/hb_legacy2.json" --input coral2-nx24-p14 --evidence "$TMP/hb_ev.json" --out "$TMP/hb_migrated.json" >"$TMP/mig.json" 2>"$TMP/e"; rc=$?
[ $rc -eq 0 ] && grep -q 'clean commit c6efde5' "$TMP/hb_migrated.json" && ok "migration: registry file changed since the run, but the entry at the run's clean commit equals the current one (git show) -> accepted, basis recorded" || bad "migration git-history rc=$rc $(cat "$TMP/e")"
else skip=$((skip+1)); echo "skip migration git-history case (commit not in this clone)"; fi
# dirty run + changed registry: refused without --manual-basis; accepted with --registry-commit whose entry equals now AND a stated basis (recorded as kind manual)
if git -C "$R" cat-file -e c6efde5:level2/hipbone/inputs.yaml 2>/dev/null; then
python3 - "$TMP/hb_ev.json" > "$TMP/hb_ev_dirty.json" <<'PY'
import json,sys; e=json.load(open(sys.argv[1])); e["git"]={"head":"c6efde5","dirty":True}; print(json.dumps(e))
PY
python3 "$TOOL" migrate-baseline "$R/level2/hipbone" "$TMP/hb_legacy2.json" --input coral2-nx24-p14 --evidence "$TMP/hb_ev_dirty.json" --out "$TMP/hb_mig_dirty.json" >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -ne 0 ] && grep -q "registry-commit" "$TMP/e" && [ ! -f "$TMP/hb_mig_dirty.json" ] && ok "migration: dirty run + changed registry without a named commit/basis -> refused" || bad "migration dirty rc=$rc $(cat "$TMP/e")"
python3 "$TOOL" migrate-baseline "$R/level2/hipbone" "$TMP/hb_legacy2.json" --input coral2-nx24-p14 --evidence "$TMP/hb_ev_dirty.json" --registry-commit c6efde5 --out "$TMP/hb_mig_dirty.json" >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -ne 0 ] && grep -q "manual-basis" "$TMP/e" && ok "migration: dirty run + named commit but no stated basis -> refused" || bad "migration dirty no-basis rc=$rc $(cat "$TMP/e")"
python3 "$TOOL" migrate-baseline "$R/level2/hipbone" "$TMP/hb_legacy2.json" --input coral2-nx24-p14 --evidence "$TMP/hb_ev_dirty.json" --registry-commit c6efde5 --manual-basis "test: run.sh echoed the args; entry unchanged" --out "$TMP/hb_mig_dirty.json" >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -eq 0 ] && grep -q '"kind": "manual"' "$TMP/hb_mig_dirty.json" && grep -q 'entry unchanged' "$TMP/hb_mig_dirty.json" && grep -q '"registry_commit_checked": "c6efde5"' "$TMP/hb_mig_dirty.json" && grep -q '"run_log_lines"' "$TMP/hb_mig_dirty.json" \
    && ok "migration: dirty run + named commit (entry equal) + stated basis -> accepted as kind=manual, basis/commit/log evidence recorded" || bad "migration dirty manual rc=$rc $(cat "$TMP/e")"
else skip=$((skip+1)); echo "skip manual migration case"; fi
# 11h. upstream reference bound to a known input (controlled adaptation) -> comparable; bound elsewhere -> refused; unbound -> incomplete
python3 - "$TMP/wl_none.json" > "$TMP/ref_bound.json" <<'PY'
import json,sys; b=json.load(open(sys.argv[1])); b["reference_binding"]={"kind":"upstream-reference","bound_input":"a","source":"fake upstream log","evidence":"deck/size/version read from the log header","adapted_by":"tests/run_all.sh"}; print(json.dumps(b))
PY
python3 "$TOOL" compare "$FR/level1/fake" "$TMP/ref_bound.json" "$TMP/m_ok/rep2/stdout.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 0 ] && grep -q '"comparison_kind": "upstream-reference"' "$TMP/c.json" && ok "reference binding: upstream reference bound to this input with evidence -> comparable (exit 0, kind upstream-reference)" || bad "reference binding rc=$rc"
python3 - "$TMP/ref_bound.json" > "$TMP/ref_other.json" <<'PY'
import json,sys; b=json.load(open(sys.argv[1])); b["reference_binding"]["bound_input"]="other"; print(json.dumps(b))
PY
python3 "$TOOL" compare "$FR/level1/fake" "$TMP/ref_other.json" "$TMP/m_ok/rep2/stdout.log" >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -eq 2 ] && grep -q "bound to input 'other'" "$TMP/e" && ok "reference binding: bound to another input -> exit 2, refused" || bad "reference binding other rc=$rc"
python3 - "$TMP/ref_bound.json" > "$TMP/ref_noev.json" <<'PY'
import json,sys; b=json.load(open(sys.argv[1])); b["reference_binding"].pop("evidence"); print(json.dumps(b))
PY
python3 "$TOOL" compare "$FR/level1/fake" "$TMP/ref_noev.json" "$TMP/m_ok/rep2/stdout.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 3 ] && grep -q 'lacks evidence' "$TMP/c.json" && ok "reference binding: binding without evidence -> exit 3, not PASS" || bad "reference binding noev rc=$rc"
# 11i. self-consistency never counts the baseline run against itself
python3 -c "import json,sys; s=json.load(open(sys.argv[1]))['summary']; assert s['baseline_from_run']=='rep1' and s['independent_runs_compared']==1 and all(c['run']!='rep1' for c in s['baseline_checks'])" "$TMP/m_ok/measurement.json" 2>/dev/null \
    && ok "self-consistency: baseline_from_run=rep1, independent_runs_compared=1 (rep2 only) -- the baseline is not compared with itself" || bad "self-consistency fields"

# ---- 12. Quicksilver: what compare actually covers (real registry, real-format fixture) ----
python3 "$TOOL" extract "$R/level2/quicksilver" "$FX/quicksilver_two_tables.log" > "$TMP/qsq.json" 2>/dev/null
mkbase "$R/level2/quicksilver" p1-profile-8c-100k-20s "$TMP/qsq.json" > "$TMP/qs_baseline.json"
python3 "$TOOL" compare "$R/level2/quicksilver" "$TMP/qs_baseline.json" "$FX/quicksilver_two_tables.log" >"$TMP/c.json" 2>/dev/null; rc=$?
# the last-cycle tallies are compared exactly (deterministic build: the deck seed is never used by the
# transport, every run of an input reproduces census / segments / scalar flux bit for bit)
[ $rc -eq 0 ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); c={x['name']:x for x in d['checks']}; assert d['verdict']=='PASS' and d['required_pending']==[] and c['final_cycle_scalar_flux']['rule']=='exact' and c['final_cycle_scalar_flux']['ok'] and all(c[k]['ok'] for k in ('pass_ratios','pass_facet','pass_no_loss','pass_fluence','fail_marker','final_cycle_census','final_cycle_num_seg'))" "$TMP/c.json" 2>/dev/null \
    && ok "quicksilver coverage: identical log -> native checks ok and the exact tally comparison verifies the scalar flux -> exit 0 PASS" || bad "quicksilver coverage control rc=$rc $(cat "$TMP/c.json")"
# change the extracted science result (scalar flux x 1.5) but keep every PASS:: line -> the exact rule catches it
sed -E 's/^(\s+19\s.*\s)5\.918521e\+05(\s)/\18.877782e+05\2/' "$FX/quicksilver_two_tables.log" > "$TMP/qs_flux_changed.log"
grep -q '8.877782e+05' "$TMP/qs_flux_changed.log" || bad "quicksilver fixture edit did not apply"
python3 "$TOOL" compare "$R/level2/quicksilver" "$TMP/qs_baseline.json" "$TMP/qs_flux_changed.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 1 ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); c={x['name']:x for x in d['checks']}; assert (not d['ok']) and d['verdict']=='FAIL' and d['failed']==['final_cycle_scalar_flux'] and abs(c['final_cycle_scalar_flux']['value']-887778.2)<1 and abs(c['final_cycle_scalar_flux']['baseline']-591852.1)<1" "$TMP/c.json" 2>/dev/null \
    && ok "quicksilver coverage: scalar flux changed x1.5 with all PASS:: lines kept -> detected: exit 1 FAIL on final_cycle_scalar_flux (591852.1 -> 887778.2)" || bad "quicksilver coverage flux rc=$rc $(cat "$TMP/c.json")"
sed -e '/^PASS:: Fluence/d' "$FX/quicksilver_two_tables.log" > "$TMP/qs_nofluence.log"
python3 "$TOOL" compare "$R/level2/quicksilver" "$TMP/qs_baseline.json" "$TMP/qs_nofluence.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 1 ] && grep -q '"failed": \[' "$TMP/c.json" && grep -q '"pass_fluence"' "$TMP/c.json" && ok "quicksilver coverage: a missing upstream PASS:: line -> exit 1 FAIL (the native checks are what is verified)" || bad "quicksilver coverage marker rc=$rc"

# ---- 13. invalidated measurements (INVALIDATED.json) are never compared, migrated, counted ----
rm -rf "$TMP/m_inv"; cp -r "$TMP/m_ok" "$TMP/m_inv"
sed -i "s#$TMP/m_ok/#$TMP/m_inv/#g" "$TMP/m_inv/baseline.json" "$TMP/m_inv/measurement.json"
python3 "$TOOL" compare "$FR/level1/fake" "$TMP/m_inv/baseline.json" "$TMP/m_ok/rep2/stdout.log" >/dev/null 2>&1; rc0=$?
echo '{"schema": "hpcperf-invalidation-1", "reason": "test: the registry args never reached the program"}' > "$TMP/m_inv/INVALIDATED.json"
python3 "$TOOL" compare "$FR/level1/fake" "$TMP/m_inv/baseline.json" "$TMP/m_ok/rep2/stdout.log" >/dev/null 2>"$TMP/e"; rc=$?
[ $rc0 -eq 0 ] && [ $rc -eq 2 ] && grep -q "baseline is invalidated" "$TMP/e" \
    && ok "invalidation: a baseline that verified (exit 0) is refused once its measurement is invalidated (exit 2)" || bad "invalidation compare baseline rc0=$rc0 rc=$rc $(cat "$TMP/e")"
python3 "$TOOL" compare "$FR/level1/fake" "$TMP/m_ok/baseline.json" "$TMP/m_inv/rep2/stdout.log" >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -eq 2 ] && grep -q "candidate log is invalidated" "$TMP/e" && ok "invalidation: an invalidated run log is refused as a candidate (exit 2)" || bad "invalidation compare candidate rc=$rc"
python3 "$TOOL" migrate-baseline "$FR/level1/fake" "$TMP/wl_none.json" --input a --evidence "$TMP/m_inv/measurement.json" --out "$TMP/wl_mig_inv.json" >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -ne 0 ] && [ ! -e "$TMP/wl_mig_inv.json" ] && grep -q "invalidated" "$TMP/e" && ok "invalidation: migrate-baseline refuses invalidated evidence, writes nothing" || bad "invalidation migrate rc=$rc"
out="$(python3 "$TOOL" status "$FR/level1/fake" "$TMP/m_inv/measurement.json" 2>&1)"; rc=$?
[ $rc -eq 3 ] && case "$out" in INVALIDATED*) true ;; *) false ;; esac && ok "invalidation: status reports INVALIDATED (exit 3), no verdict" || bad "invalidation status rc=$rc $out"
mkdir -p "$TMP/mdir/level1-fake"; rm -rf "$TMP/mdir/level1-fake/a"; cp -r "$TMP/m_inv" "$TMP/mdir/level1-fake/a"
python3 - "$TMP/mdir" "$FR/level1/fake" <<'PY' && ok "invalidation: the audit loads an invalidated measurement as INVALIDATED, not completed, no verdict" || bad "invalidation audit"
import sys, os
sys.path.insert(0, os.path.join(os.environ.get("R", "."), "tools", "inputs"))
import hpcperf_inputs_audit as au
m = au.load_measurement(sys.argv[1], 1, "fake", "a", sys.argv[2])
assert m["invalidated"] and not m["run_completed"] and m["timing_status"] == "INVALIDATED" and m["baseline_verdict"] is None and m["native_check"] is None, m
PY

# ---- 14. a redefined input (remhos periodic-hexagon-p0: order 3 made explicit) ----
printf 'Final mass u:  0.3884079556\nMax value u:   0.7994561099\nMass loss u:   1.0e-12\n' > "$TMP/rh_o2.log"
python3 "$TOOL" extract "$R/level2/remhos" "$TMP/rh_o2.log" --input periodic-hexagon-p0 > "$TMP/rh_q.json" 2>/dev/null
mkbase "$R/level2/remhos" periodic-hexagon-p0 "$TMP/rh_q.json" > "$TMP/rh_base_cur.json"
python3 - "$TMP/rh_base_cur.json" > "$TMP/rh_base_o2.json" <<'PY'
import json, sys; b = json.load(open(sys.argv[1])); w = b["workload"]
assert w["args"][-2:] == ["-o", "3"], w["args"]
w["args"] = w["args"][:-2]; w["params"].pop("o")          # the definition before order 3 was made explicit
print(json.dumps(b))
PY
python3 "$TOOL" compare "$R/level2/remhos" "$TMP/rh_base_o2.json" "$TMP/rh_o2.log" --input periodic-hexagon-p0 >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -eq 2 ] && grep -q "not the candidate's workload" "$TMP/e" && ok "redefined input: a baseline of the old (order 2) workload is refused for the order-3 input (exit 2)" || bad "redefined input: old baseline rc=$rc $(cat "$TMP/e")"
if [ -x "$R/build/level2/remhos/cuda/remhos" ]; then
    SH_DIR="$TMP/rh_shim"; mkdir -p "$SH_DIR"
    for t in mpirun mpiexec srun; do printf '#!/bin/sh\necho "%s" >> "%s/hit"\nexit 97\n' "$t" "$SH_DIR" > "$SH_DIR/$t"; chmod +x "$SH_DIR/$t"; done
    for id in periodic-hexagon-p0 periodic-square-p5; do
        cmd="$(PATH="$SH_DIR:$PATH" HPCPERF_DRY_RUN=1 HPCPERF_REMHOS_INPUT=$id bash "$R/level2/remhos/run.sh" CUDA 2>&1 | grep 'command:')"
        n_o="$(printf '%s\n' "$cmd" | tr ' ' '\n' | grep -c '^-o$')"
        last="$(printf '%s\n' "$cmd" | tr ' ' '\n' | grep -A1 '^-o$' | tail -1)"
        [ "$n_o" = 1 ] && [ "$last" = 3 ] && [ ! -e "$SH_DIR/hit" ] && ok "remhos run.sh $id: exactly one -o in the command, value 3 (dry run, nothing started)" \
            || bad "remhos run.sh $id: $n_o x -o, value '$last', shims hit: $(cat "$SH_DIR/hit" 2>/dev/null)"
    done
else
    skip=$((skip+1)); echo "skip remhos run.sh -o count (not built)"
fi

# ---- 15. correctness checks (inputs.yaml `check:`, `check`, file-sourced quantities, `near`, verdict) ----
CK="$TMP/ck"; mkdir -p "$CK/build/fake" "$CK/level1/ck" "$CK/shared"; touch "$CK/hpcperf_env.sh"
# the fake benchmark: prints a result line, records whether the timing switch was set, writes result.txt
# into its cwd and one file per run into a shared directory; FAKE_RC forces its exit code
cat > "$CK/build/fake/ck_bin" <<'EOF'
#!/bin/sh
echo "args: $*"
echo "verify=${HPCPERF_SKIP_VERIFY:-unset}"
echo "value 42" > result.txt
echo "shared $$ $*" > "$FAKE_SHARED/out_$$.txt"
[ "${FAKE_NOPASS:-}" = 1 ] || echo "RESULT OK"
exit "${FAKE_RC:-0}"
EOF
chmod +x "$CK/build/fake/ck_bin"
cat > "$CK/level1/ck/verify.py" <<'EOF'
import os, subprocess, sys
r = subprocess.run([sys.argv[1]] + sys.argv[2:], capture_output=True, text=True)
mode = os.environ.get("FAKE_CHECK", "pass")
if mode == "pass": print("PASS: verified " + " ".join(sys.argv[2:]))
elif mode == "silent": pass
elif mode == "fail": print("FAIL: mismatch")
elif mode == "exit1": print("PASS: printed but"); sys.exit(1)
EOF
cat > "$CK/level1/ck/inputs.yaml" <<'EOF'
schema: hpcperf-inputs-1
benchmark: ck
level: 1
selector: null
default_input: a
entry: {kind: binary, path: build/fake/ck_bin}
timing: {kind: none, status: NEEDS_TIMING_SUPPORT, reason: fake}
baseline:
  quantities:
  - {name: ok_line, regex: '^RESULT OK$', compare: {rule: present}}
  - {name: file_value, regex: '^value (?P<value>\d+)$', source: {file: result.txt}, compare: {rule: exact}}
  - {name: shared_marker, regex: '^shared ', source: {file: 'outputs/out_*.txt', select_file: newest}, compare: {rule: present}}
  - {name: bad_marker, regex: '^BAD', source: {file: 'outputs/out_*.txt', select_file: newest}, compare: {rule: absent}}
coverage: {status: MULTI_INPUT}
check:
  kind: standalone
  command: [python3, '{bench_dir}/verify.py', '{exe}', '{args}']
  outputs: [{glob: '{repo}/shared/out_*.txt', new: true}]
  pass_regex: '^PASS: verified'
  fail_regex: '^FAIL'
  basis: fake checker for the tests
  covers: all
inputs:
  - {id: a, case: c, variant: default, source: {kind: upstream-file, upstream: x}, params: {n: 7, tag: seven}, args: [level1/ck/verify.py, '7'], backends_validated: [cuda]}
  - id: b
    case: c
    variant: size
    source: {kind: upstream-file, upstream: x}
    params: {n: 8}
    args: ['8']
    backends_validated: [cuda]
    check:
      kind: post_run
      run_env: {HPCPERF_SKIP_VERIFY: null}
      outputs: [{glob: '{repo}/shared/out_*.txt', new: true}]
      pass_regex: '^RESULT OK$'
      fail_regex: '^RESULT BAD'
      basis: the program's own line
      covers: [file_value]
  - id: c
    case: c
    variant: size
    source: {kind: upstream-file, upstream: x}
    params: {n: 9}
    args: ['9']
    backends_validated: [cuda]
    check: {kind: none, reason: nothing to check yet}
  - id: d
    case: c
    variant: size
    source: {kind: upstream-file, upstream: x}
    params: {n: 10}
    args: ['10']
    backends_validated: [cuda]
    check:
      kind: post_run
      command: [python3, '{bench_dir}/verify.py', /bin/true, '{param:n}', '{arg:0}', '{log}', '{outputs}']
      outputs: [{glob: '{repo}/shared/out_*.txt', new: true}]
      pass_regex: '^PASS: verified'
      fail_regex: '^FAIL'
      basis: checker after the run, reads the run's outputs
      covers: all
EOF
python3 "$TOOL" validate "$CK/level1/ck" >/dev/null 2>&1 && ok "check: a registry with benchmark-level and per-input check blocks (standalone, post_run, none) validates" || bad "check registry invalid: $(python3 "$TOOL" validate "$CK/level1/ck" 2>&1 | noise)"
# schema negatives: each variant must be refused with a specific message
ckneg() {  # ckneg <label> <sed expr> <expected message>
    mkdir -p "$TMP/ckneg/level1/ck"; sed -e "$2" "$CK/level1/ck/inputs.yaml" > "$TMP/ckneg/level1/ck/inputs.yaml"
    out="$(python3 "$TOOL" validate "$TMP/ckneg/level1/ck" 2>&1 | noise || true)"
    grep -q -- "$3" <<<"$out" && ok "check schema: $1" || bad "check schema: $1 not refused: $out"
}
ckneg "unknown placeholder" "s/'{exe}', '{args}'/'{exe}', '{binary}'/" "unknown template placeholder {binary}"
ckneg "unknown parameter" "s/'{param:n}', '{arg:0}'/'{param:nope}', '{arg:0}'/" "no parameter 'nope'"
ckneg "argument index out of range" "s/'{param:n}', '{arg:0}'/'{param:n}', '{arg:5}'/" "only 1 argument"
ckneg "covers names an unknown quantity" "s/covers: \[file_value\]/covers: [no_such_quantity]/" "unknown quantity 'no_such_quantity'"
ckneg "kind none needs a reason" "s/check: {kind: none, reason: nothing to check yet}/check: {kind: none}/" "kind none needs a reason"
ckneg "standalone cannot set run_env" "s/^  covers: all\$/  covers: all\n  run_env: {X: '1'}/" "run_env applies to post_run"
ckneg "basis required" "s/  basis: fake checker for the tests/  basis: ''/" "check.basis missing"
ckneg "{log} only for post_run" "s/'{exe}', '{args}'/'{exe}', '{log}'/" "not available for this check kind"
ckneg "near needs value and tol" "s/compare: {rule: exact}}/compare: {rule: near, tol: 0.1}}/" "rule near needs value"
# dry run renders the templates without running anything
python3 "$TOOL" check "$CK/level1/ck" a --out "$TMP/ck_dry" --dry-run > "$TMP/ck_dry.json" 2>/dev/null; rc=$?
python3 - "$TMP/ck_dry.json" "$CK" <<'PY' && [ ! -e "$TMP/ck_dry" ] && ok "check --dry-run: {exe} {args} rendered (repository path absolute), nothing executed" || bad "check dry-run rc=$rc $(cat "$TMP/ck_dry.json")"
import json, sys; d = json.load(open(sys.argv[1])); ck = sys.argv[2]
assert d["dry_run"] and d["command"] == ["python3", f"{ck}/level1/ck/verify.py", f"{ck}/build/fake/ck_bin", f"{ck}/level1/ck/verify.py", "7"], d["command"]
assert d["outputs"] == [{"glob": f"{ck}/shared/out_*.txt", "new": True}], d["outputs"]
PY
python3 "$TOOL" check "$CK/level1/ck" d --out "$TMP/ck_dry2" --dry-run > "$TMP/ck_dry2.json" 2>/dev/null
python3 - "$TMP/ck_dry2.json" "$TMP/ck_dry2" <<'PY' && ok "check --dry-run: {param:n} {arg:0} {log} {outputs} rendered for a post_run checker" || bad "check dry-run d: $(cat "$TMP/ck_dry2.json")"
import json, sys; d = json.load(open(sys.argv[1])); o = sys.argv[2]
assert d["command"][2:] == ["/bin/true", "10", "10", f"{o}/run/stdout.log", f"{o}/outputs"], d["command"]
PY
# standalone: PASS / silent exit 0 / FAIL line / exit 1 with a PASS line / missing checker
export FAKE_SHARED="$CK/shared"
FAKE_CHECK=pass python3 "$TOOL" check "$CK/level1/ck" a --out "$TMP/ck_a" >/dev/null 2>&1; rc=$?
python3 - "$TMP/ck_a/check.json" <<'PY' && [ $rc -eq 0 ] && ok "check standalone: PASS line + exit 0 -> verdict PASS (exit 0), record carries workload, command, log sha256, outputs" || bad "check standalone PASS rc=$rc $(cat "$TMP/ck_a/check.json" 2>/dev/null | head -c 600)"
import json, sys; d = json.load(open(sys.argv[1]))
assert d["schema"] == "hpcperf-inputs-check-1" and d["verdict"] == "PASS" and d["matched_line"].startswith("PASS: verified") and d["kind"] == "standalone"
assert d["workload"]["params"]["n"] == 7 and d["check"]["exit_code"] == 0 and len(d["check"]["log_sha256"]) == 64 and d["run"] is None
assert d["covers"] == "all" and d["basis"] and d["git"] and d["started_utc"] and d["finished_utc"]
PY
FAKE_CHECK=silent python3 "$TOOL" check "$CK/level1/ck" a --out "$TMP/ck_a2" >/dev/null 2>&1; rc=$?
[ $rc -eq 1 ] && grep -q '"verdict": "FAIL"' "$TMP/ck_a2/check.json" && grep -q "no pass line" "$TMP/ck_a2/check.json" && ok "check standalone: exit 0 without the pass line -> FAIL, not PASS" || bad "check silent rc=$rc"
FAKE_CHECK=fail python3 "$TOOL" check "$CK/level1/ck" a --out "$TMP/ck_a3" >/dev/null 2>&1; rc=$?
[ $rc -eq 1 ] && grep -q '"matched_line": "FAIL: mismatch"' "$TMP/ck_a3/check.json" && ok "check standalone: a FAIL line -> FAIL with the line recorded" || bad "check fail-line rc=$rc"
FAKE_CHECK=exit1 python3 "$TOOL" check "$CK/level1/ck" a --out "$TMP/ck_a4" >/dev/null 2>&1; rc=$?
[ $rc -eq 1 ] && grep -q "checker exited 1" "$TMP/ck_a4/check.json" && ok "check standalone: exit 1 with a PASS line -> FAIL (the exit code wins)" || bad "check exit1 rc=$rc"
mkdir -p "$TMP/ckmiss/level1/ck" "$TMP/ckmiss/build/fake" "$TMP/ckmiss/shared"; cp "$CK/build/fake/ck_bin" "$TMP/ckmiss/build/fake/"; touch "$TMP/ckmiss/hpcperf_env.sh"
sed -e "s#\[python3, '{bench_dir}/verify.py', '{exe}', '{args}'\]#['{bench_dir}/no_such_checker', '{exe}']#" "$CK/level1/ck/inputs.yaml" > "$TMP/ckmiss/level1/ck/inputs.yaml"
FAKE_SHARED="$TMP/ckmiss/shared" python3 "$TOOL" check "$TMP/ckmiss/level1/ck" a --out "$TMP/ck_a5" >/dev/null 2>&1; rc=$?
[ $rc -eq 2 ] && grep -q '"verdict": "ERROR"' "$TMP/ck_a5/check.json" && ok "check standalone: a checker that cannot start -> ERROR (exit 2), never PASS" || bad "check missing checker rc=$rc"
python3 "$TOOL" check "$CK/level1/ck" c --out "$TMP/ck_c" >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -eq 2 ] && grep -q "has no correctness check (nothing to check yet)" "$TMP/e" && ok "check: an input with kind none is refused with its reason (exit 2)" || bad "check none rc=$rc $(cat "$TMP/e")"
# post_run: the program's own line; HPCPERF_SKIP_VERIFY never reaches the run; outputs collected (new files only)
echo "stale" > "$CK/shared/out_stale.txt"; touch -d '2000-01-01' "$CK/shared/out_stale.txt"
HPCPERF_SKIP_VERIFY=1 python3 "$TOOL" check "$CK/level1/ck" b --out "$TMP/ck_b" >/dev/null 2>&1; rc=$?
python3 - "$TMP/ck_b" <<'PY' && [ $rc -eq 0 ] && ok "check post_run: program pass line -> PASS; HPCPERF_SKIP_VERIFY unset for the run; only the run's new shared file collected (sha256 recorded)" || bad "check post_run b rc=$rc $(cat "$TMP/ck_b/check.json" 2>/dev/null | head -c 800)"
import json, os, sys; o = sys.argv[1]; d = json.load(open(os.path.join(o, "check.json")))
assert d["verdict"] == "PASS" and d["run"]["exit_code"] == 0 and d["run"]["skip_verify"] is False and d["check"]["note"]
assert "verify=unset" in open(d["run"]["log"]).read(), open(d["run"]["log"]).read()
outs = [x for x in d["outputs"] if "sha256" in x]
assert len(outs) == 1 and os.path.basename(outs[0]["source"]).startswith("out_") and not outs[0]["source"].endswith("out_stale.txt") and len(outs[0]["sha256"]) == 64, d["outputs"]
assert os.path.isfile(outs[0]["copied_to"])
PY
FAKE_RC=3 python3 "$TOOL" check "$CK/level1/ck" b --out "$TMP/ck_b2" >/dev/null 2>&1; rc=$?
[ $rc -eq 1 ] && grep -q "benchmark run exited 3" "$TMP/ck_b2/check.json" && ok "check post_run: benchmark exit 3 -> FAIL without reading its output" || bad "check post_run rc rc=$rc"
FAKE_NOPASS=1 python3 "$TOOL" check "$CK/level1/ck" b --out "$TMP/ck_b3" >/dev/null 2>&1; rc=$?
[ $rc -eq 1 ] && grep -q "no pass line" "$TMP/ck_b3/check.json" && ok "check post_run: run exit 0 without the pass line -> FAIL" || bad "check post_run nopass rc=$rc"
FAKE_CHECK=pass python3 "$TOOL" check "$CK/level1/ck" d --out "$TMP/ck_d" >/dev/null 2>&1; rc=$?
python3 - "$TMP/ck_d" <<'PY' && [ $rc -eq 0 ] && ok "check post_run with a checker: the run's log and outputs directory are handed to the checker" || bad "check post_run d rc=$rc $(cat "$TMP/ck_d/check.json" 2>/dev/null | head -c 600)"
import json, os, sys; o = sys.argv[1]; d = json.load(open(os.path.join(o, "check.json")))
assert d["verdict"] == "PASS" and d["run"]["exit_code"] == 0 and d["matched_line"] == f"PASS: verified 10 10 {o}/run/stdout.log {o}/outputs"
PY
# file-sourced quantities through measure (result.txt in the run directory, the shared file under outputs/)
python3 "$TOOL" measure "$CK/level1/ck" b --out "$TMP/ck_m" --warmup 0 --reps 2 --timeout 30 >/dev/null 2>&1; rc=$?
python3 - "$TMP/ck_m/measurement.json" <<'PY' && ok "file-sourced quantities: value read from result.txt, markers from the newest outputs/ file; baseline verdict PASS over the independent run" || bad "file-sourced measure rc=$rc $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['runs'][0].get('baseline_quantities'), d['runs'][0].get('outputs'), d['summary']['baseline_verdict'])" "$TMP/ck_m/measurement.json" 2>&1)"
import json, sys; d = json.load(open(sys.argv[1]))
q = d["runs"][0]["baseline_quantities"]
assert q["file_value"]["value"] == 42 and q["shared_marker"]["present"] and not q["bad_marker"]["present"] and "error" not in q["bad_marker"], q
assert d["summary"]["baseline_verdict"] == "PASS", d["summary"]["baseline_verdict"]
PY
python3 - "$R" "$CK/level1/ck" "$TMP/ck_m" <<'PY' && ok "file-sourced quantities: a missing source file is an error -- 'absent' does not pass on it, compare fails" || bad "missing source file accepted"
import json, os, sys; sys.path.insert(0, sys.argv[1] + "/tools/inputs"); import hpcperf_inputs as hi
from pathlib import Path
doc = hi.load(sys.argv[2]); inp = hi.get_input(doc, "b")
log = Path(sys.argv[3]) / "rep1" / "stdout.log"
cur = hi.extract(doc, log, inp, run_dir=Path(sys.argv[3]) / "nowhere")
assert cur["bad_marker"].get("error") and cur["file_value"]["value"] is None, cur
b = json.load(open(Path(sys.argv[3]) / "baseline.json"))
res = hi.compare(doc, b["quantities"], cur, inp)
assert not res["ok"] and "bad_marker" in res["failed"] and "file_value" in res["failed"], res["failed"]
PY
# the near rule: a reference constant with a relative tolerance (no baseline needed)
python3 - "$R" <<'PY' && ok "near rule: |v - value| <= tol |value| passes, beyond it fails, and it needs no baseline value" || bad "near rule"
import sys; sys.path.insert(0, sys.argv[1] + "/tools/inputs"); import hpcperf_inputs as hi
doc = {"baseline": {"quantities": [{"name": "l2", "regex": "x", "compare": {"rule": "near", "value": 0.00355178, "tol": 1e-3}}]}}
ok_ = hi.compare(doc, {}, {"l2": {"value": 0.003554}}, None); bad_ = hi.compare(doc, {}, {"l2": {"value": 0.00360}}, None)
assert ok_["verdict"] == "PASS" and ok_["checks"][0]["reference"] == 0.00355178, ok_
assert bad_["verdict"] == "FAIL" and bad_["checks"][0]["rel_err"] > 1e-3, bad_
assert hi.native_check(doc, {"l2": {"value": 0.003554}})["status"] == "PASS"
PY
# the combined verdict (correctness_verdict / verdict sub-command)
python3 - "$R" "$CK/level1/ck" "$TMP/ck_a/check.json" <<'PY' && ok "correctness_verdict: FAIL wins; compare PASS is PASS; INCOMPLETE + covering check PASS is PASS; non-covering / ERROR / stale checks leave INCOMPLETE" || bad "correctness_verdict"
import copy, json, sys; sys.path.insert(0, sys.argv[1] + "/tools/inputs"); import hpcperf_inputs as hi
doc = hi.load(sys.argv[2]); inp = hi.get_input(doc, "a"); chk = json.load(open(sys.argv[3]))
V = lambda cv, pend, c: hi.correctness_verdict(doc, inp, cv, pend, c)["verdict"]
assert V("FAIL", [], chk) == "FAIL" and V("PASS", [], dict(chk, verdict="FAIL")) == "FAIL"
assert V("PASS", [], None) == "PASS" and V("INCOMPLETE", ["q"], chk) == "PASS"           # covers all
assert V("INCOMPLETE", ["q"], dict(chk, covers=["other"])) == "INCOMPLETE" and V("INCOMPLETE", ["q"], dict(chk, covers=["q"])) == "PASS"
assert V("NONE", None, chk) == "PASS" and V("NONE", None, None) == "INCOMPLETE" and V("INCOMPLETE", ["q"], dict(chk, verdict="ERROR")) == "INCOMPLETE"
stale = copy.deepcopy(chk); stale["workload"]["params"]["n"] = 99
r = hi.correctness_verdict(doc, inp, "INCOMPLETE", ["q"], stale); assert r["verdict"] == "INCOMPLETE" and r["check_stale"], r
PY
python3 "$TOOL" verdict "$CK/level1/ck" b --measurement "$TMP/ck_m/measurement.json" --check "$TMP/ck_b/check.json" > "$TMP/v.json" 2>/dev/null; rc=$?
[ $rc -eq 0 ] && grep -q '"verdict": "PASS"' "$TMP/v.json" && ok "verdict sub-command: measurement PASS + check PASS -> PASS (exit 0)" || bad "verdict rc=$rc $(cat "$TMP/v.json")"
python3 "$TOOL" verdict "$CK/level1/ck" b --check "$TMP/ck_b2/check.json" > "$TMP/v.json" 2>/dev/null; rc=$?
[ $rc -eq 1 ] && grep -q '"verdict": "FAIL"' "$TMP/v.json" && ok "verdict sub-command: a failed check -> FAIL (exit 1)" || bad "verdict fail rc=$rc"
python3 "$TOOL" verdict "$CK/level1/ck" c > "$TMP/v.json" 2>/dev/null; rc=$?
[ $rc -eq 3 ] && grep -q '"verdict": "INCOMPLETE"' "$TMP/v.json" && ok "verdict sub-command: nothing measured, no check -> INCOMPLETE (exit 3)" || bad "verdict incomplete rc=$rc"
python3 "$TOOL" verdict "$CK/level1/ck" a --check "$TMP/ck_b/check.json" >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -eq 2 ] && grep -q "not a check record of ck/a" "$TMP/e" && ok "verdict sub-command: another input's check record is refused" || bad "verdict wrong input rc=$rc"
# the audit reads check records: <checks>/<bench>/<input>/check.json
mkdir -p "$TMP/ckaudit/ck/a" "$TMP/ckaudit/ck/b"; cp "$TMP/ck_a/check.json" "$TMP/ckaudit/ck/a/"; cp "$TMP/ck_b2/check.json" "$TMP/ckaudit/ck/b/"
python3 - "$R" "$CK" "$TMP/ckaudit" <<'PY' && ok "audit --checks: per-input has_check / check verdict / combined verdict (a PASS, b FAIL, c INCOMPLETE without a check)" || bad "audit checks"
import sys, os; sys.path.insert(0, os.path.join(sys.argv[1], "tools", "inputs")); import hpcperf_inputs_audit as au
rows = au.audit(sys.argv[2], None, {}, sys.argv[3]); r = [x for x in rows if x["benchmark"] == "ck"][0]
assert r["has_check"] == {"a": True, "b": True, "c": False, "d": True}, r["has_check"]
assert r["checks"]["a"]["verdict"] == "PASS" and r["checks"]["b"]["verdict"] == "FAIL" and "c" not in r["checks"]
assert r["verdicts"] == {"a": "PASS", "b": "FAIL", "c": "INCOMPLETE", "d": "INCOMPLETE"}, r["verdicts"]
assert r["verdict_counts"] == {"PASS": 1, "INCOMPLETE": 2, "FAIL": 1}
PY
unset FAKE_SHARED
# real registries: the wired checks render, the closed-form and channel_shuffle rules behave
python3 "$TOOL" check "$R/level1/bfs" graph4096 --out "$TMP/ck_bfs" --dry-run > "$TMP/ck_bfs.json" 2>/dev/null; rc=$?
python3 - "$TMP/ck_bfs.json" "$R" <<'PY' && ok "bfs: check renders verify.py <exe> <absolute graph path> (standalone, covers all)" || bad "bfs check render rc=$rc $(cat "$TMP/ck_bfs.json")"
import json, sys; d = json.load(open(sys.argv[1])); R = sys.argv[2]
assert d["kind"] == "standalone" and d["command"][:2] == ["python3", f"{R}/level1/bfs/verify.py"] and d["command"][3] == f"{R}/level1/bfs/data/graph4096.txt", d["command"]
PY
python3 "$TOOL" check "$R/level1/hotspot" g512-p2-t200 --out "$TMP/ck_hs" --dry-run > "$TMP/ck_hs.json" 2>/dev/null
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['command'][3:6]==['512','2','200'] and d['command'][6].endswith('/data/temp_512'), d['command']" "$TMP/ck_hs.json" && ok "hotspot: check renders grid / pyramid height / sim_time from the params and the data files from the args" || bad "hotspot check render: $(cat "$TMP/ck_hs.json")"
python3 "$TOOL" check "$R/level2/examinimd" snap-ta06a --out "$TMP/x" --dry-run >/dev/null 2>"$TMP/e"; rc=$?
[ $rc -eq 2 ] && grep -q "SNAP deck" "$TMP/e" && ok "examinimd: the SNAP input's per-input 'kind: none' overrides the benchmark check with its reason" || bad "examinimd snap rc=$rc"
for d in level1/bfs level1/hotspot level1/srad_v1 level1/gaussian_elimination level1/channel_shuffle level2/haccabanapm level2/examinimd level2/exacmech level2/kripke level2/comb level2/branson level2/exampm level2/p3_vlp4d level2/p3_heat3d level2/miniem level2/cabanapic level2/hipbone level2/miniweather level2/quicksilver level2/shaw level1/ao_bench; do
    python3 "$TOOL" validate "$R/$d" >/dev/null 2>&1 || bad "validate $d after the correctness wiring: $(python3 "$TOOL" validate "$R/$d" 2>&1 | noise | head -3)"
done; ok "every registry touched by the correctness wiring validates"
printf 'Elapsed time: 1.5 [s]\nL2_norm: 0.00355178\n' > "$TMP/h3.log"
python3 "$TOOL" extract "$R/level2/p3_heat3d" "$TMP/h3.log" --input n512-1000 > "$TMP/h3.json" 2>/dev/null
python3 - "$R" "$TMP/h3.json" "$TMP/h3.log" <<'PY' && ok "p3_heat3d: L2_norm is a required 'near' quantity against the closed form (0.00355178 passes, 0.0036 fails, upstream-documented value)" || bad "p3_heat3d near rule"
import json, sys; sys.path.insert(0, sys.argv[1] + "/tools/inputs"); import hpcperf_inputs as hi
from pathlib import Path
doc = hi.load(sys.argv[1] + "/level2/p3_heat3d"); inp = hi.get_input(doc, "n512-1000")
cur = json.load(open(sys.argv[2])); assert cur["l2_norm"]["value"] == 0.00355178
assert hi.compare(doc, {}, cur, inp)["verdict"] == "PASS"
assert hi.compare(doc, {}, {"l2_norm": {"value": 0.0036}, "elapsed_line": {"present": True}}, inp)["verdict"] == "FAIL"
el = {"present": True}
assert hi.native_check(doc, {"l2_norm": {"value": 0.00013108}, "elapsed_line": el}, hi.get_input(doc, "n1024-200"))["status"] == "PASS"
assert hi.native_check(doc, {"l2_norm": {"value": 0.017546}, "elapsed_line": el}, hi.get_input(doc, "n256-1000"))["status"] == "PASS"
assert hi.native_check(doc, {"l2_norm": {"value": 0.0180}, "elapsed_line": el}, hi.get_input(doc, "n256-1000"))["status"] == "FAIL"
PY
printf '(N=1 C=32 W=224 H=224)\nAverage time of channel shuffle (NHWC): 0.1 (ms)\nAverage time of channel shuffle (NCHW): 0.2 (ms)\n' > "$TMP/cs_ok.log"
printf '(N=1 C=32 W=224 H=224)\n' > "$TMP/cs_skip.log"
python3 - "$R" "$TMP/cs_ok.log" "$TMP/cs_skip.log" <<'PY' && ok "channel_shuffle: the 'Average time' lines are required evidence that the built-in memcmp ran -- a skip-verify style log no longer passes on the absent failure marker alone" || bad "channel_shuffle evidence rule"
import sys; sys.path.insert(0, sys.argv[1] + "/tools/inputs"); import hpcperf_inputs as hi
from pathlib import Path
doc = hi.load(sys.argv[1] + "/level1/channel_shuffle"); inp = hi.get_input(doc, "g2-w255-h255")
assert hi.compare(doc, {}, hi.extract(doc, Path(sys.argv[2]), inp), inp)["verdict"] == "PASS"
assert hi.compare(doc, {}, hi.extract(doc, Path(sys.argv[3]), inp), inp)["verdict"] == "FAIL"
PY
# an aborted sweep (upstream's int numel overflows at N=16, C=512, W=H=512; cudaMalloc fails; exit 0) is a
# failure of the check, not a pass on the configurations that ran before it
python3 - "$R" <<'PY' && ok "channel_shuffle: the check's fail line covers the aborted sweep ('Device memory allocation failed')" || bad "channel_shuffle aborted-sweep rule"
import re, sys; sys.path.insert(0, sys.argv[1] + "/tools/inputs"); import hpcperf_inputs as hi
doc = hi.load(sys.argv[1] + "/level1/channel_shuffle"); inp = hi.get_input(doc, "g2-w255-h255")
fail = hi.check_of(doc, inp)["fail_regex"]
assert re.search(fail, "Device memory allocation failed. Exit"), fail
assert re.search(fail, "Failed to pass channel shuffle (NCHW) check"), fail
assert not re.search(fail, "Average time of channel shuffle (NCHW): 0.2 (ms)"), fail
PY
if [ -x "$R/build/gaussian_elimination/cuda/gaussian_elimination_cuda" ]; then :; fi
python3 - "$R" <<'PY' && ok "gaussian_elimination verify.py: the -s branch rebuilds create_matrix's Toeplitz system (a[i][j] = 10 exp(-0.01|i-j|), b = 1) and accepts the exact solution" || bad "gaussian verify -s branch"
import os, subprocess, sys, tempfile, numpy as np
R = sys.argv[1]; n = 16
idx = np.arange(n); a = 10.0 * np.exp(-0.01 * np.abs(idx[:, None] - idx[None, :])); x = np.linalg.solve(a, np.ones(n))
d = tempfile.mkdtemp(); exe = os.path.join(d, "fake_gauss")
open(exe, "w").write("#!/bin/sh\necho 'Create matrix internally in parse, size = 16'\necho 'The final solution is: '\necho '" + " ".join(f"{v:.12g}" for v in x) + "'\n"); os.chmod(exe, 0o755)
r = subprocess.run(["python3", f"{R}/level1/gaussian_elimination/verify.py", exe, "-s", "16"], capture_output=True, text=True)
assert r.returncode == 0 and r.stdout.startswith("PASS: residual"), r.stdout + r.stderr
open(exe, "w").write("#!/bin/sh\necho 'The final solution is: '\necho '" + " ".join("1.0" for _ in x) + "'\n"); os.chmod(exe, 0o755)
r = subprocess.run(["python3", f"{R}/level1/gaussian_elimination/verify.py", exe, "-s", "16"], capture_output=True, text=True)
assert r.returncode == 1 and r.stdout.startswith("FAIL: residual"), r.stdout
PY
python3 - "$R" <<'PY' && ok "comb check_run.py: clean proc + summary files PASS; a mismatch line, a missing summary or no test-comm phase FAIL" || bad "comb check_run.py"
import os, subprocess, sys, tempfile
R = sys.argv[1]
def run(files):
    d = tempfile.mkdtemp()
    for n, c in files.items(): open(os.path.join(d, n), "w").write(c)
    r = subprocess.run(["python3", f"{R}/level2/comb/check_run.py", d], capture_output=True, text=True); return r.returncode, r.stdout
ok = {"Comb_07_proc0000": "Starting test Comm mpi\ntest-comm:  num 1 avg 0.3 s\n", "Comb_07_summary": "Args x\ntest-comm:  num 1 avg 0.37 s min 0.37 s max 0.37 s\n", "Comb_07_summary.csv": "x"}
assert run(ok)[0] == 0 and "PASS: Comb halo check" in run(ok)[1]
bad = dict(ok); bad["Comb_07_proc0000"] = "test pre-comm 0x1 1 zone 5(1 2 3) g5(1 2 3) = 3.000000 expected -1.000000 next 7.000000\n"
assert run(bad)[0] == 1 and "1 halo value mismatch" in run(bad)[1]
assert run({"Comb_07_proc0000": ok["Comb_07_proc0000"]})[0] == 1
nosum = dict(ok); nosum["Comb_07_summary"] = "Args x\nbench-comm: num 1\n"; assert run(nosum)[0] == 1 and "test-comm" in run(nosum)[1]
PY
python3 - "$R" <<'PY' && ok "branson check_log.py: conservation within 1e-9 with GPU transport PASSes; a violated balance, a CPU fallback or a missing FOM line FAILs; --steps N enforced" || bad "branson check_log.py"
import os, subprocess, sys, tempfile
R = sys.argv[1]
def log(rad="1.0e-12", gpu=True, fom=True, steps=2):
    s = ""
    for i in range(steps):
        s += "Step: %d\n" % (i + 1) + ("Transferring 25 cell(s) to the GPU\n" if gpu else "GPU kernel not available\n")
        s += "Emission E: 1.0e3\nSource E: 0.0\nPre census E: 2.0e2\nPre mat E: 5.0e3\nPost mat E: 5.1e3\nAbsorption E: 1.0e2\nExit E: 1.0\n"
        s += "Radiation conservation: %s\nMaterial conservation: 1.0e-12\n" % rad
        s += "   0   1.5000   0.1   0.2\n   1   1.2000   0.1   0.2\n"          # T_e rows (cell, T_e, ...)
    if fom: s += "Total Photons transported: 100\nPhotons Per Second (FOM): 1.0e6\n"
    d = tempfile.mkdtemp(); p = os.path.join(d, "run.log"); open(p, "w").write(s); return p
def run(*args):
    r = subprocess.run(["python3", f"{R}/level2/branson/check_log.py"] + list(args), capture_output=True, text=True); return r.returncode, r.stdout
assert run("gpu", log())[0] == 0 and "PASS: branson log check (gpu)" in run("gpu", log())[1]
assert run("gpu", log(rad="1.0e-4"))[0] == 1 and run("gpu", log(gpu=False))[0] == 1 and run("gpu", log(fom=False))[0] == 1
assert run("gpu", log(steps=2), "--steps", "5")[0] == 1 and run("gpu", log(steps=5), "--steps", "5")[0] == 0
assert run("cmp", log(), log())[0] == 0
PY


# ---- 16. second-pass correctness checks: branson reference, exampm final state / dumps, shaw norms, vlp4d nrj, aobench image, quicksilver seed scatter ----
P2="$TMP/p2"; mkdir -p "$P2"
# branson check_log.py cmp: energies, T_e and the transported photon count of two logs
bransonlog() {  # $1 file  $2 photons  $3 post-mat E
printf 'Step: 1 of 1\nEmission E: 1.0\nSource E: 0.0\nPre census E: 0.5\nPre mat E: 2.0\nAbsorption E: 0.3\nExit E: 0.1\nPost mat E: %s\nRadiation conservation: 1e-13\nMaterial conservation: 1e-13\n 3 cell(s) to the GPU\n  0  1.5000  0.2  0.3\n  1  1.4000  0.2  0.3\nTotal Photons transported: %s\nPhotons Per Second (FOM): 1.0\n' "$3" "$2" > "$1"
}
bransonlog "$P2/b_gpu.log" 100000 2.10; bransonlog "$P2/b_cpu.log" 101000 2.12; bransonlog "$P2/b_cpu_far.log" 120000 2.12
python3 "$R/level2/branson/check_log.py" cmp "$P2/b_gpu.log" "$P2/b_cpu.log" > "$P2/b1.out" 2>&1; rc1=$?
python3 "$R/level2/branson/check_log.py" cmp "$P2/b_gpu.log" "$P2/b_cpu_far.log" > "$P2/b2.out" 2>&1; rc2=$?
[ $rc1 -eq 0 ] && grep -q "^PASS: branson log check (cmp)" "$P2/b1.out" && grep -q "total photons transported: GPU 100000  CPU 101000" "$P2/b1.out" \
    && [ $rc2 -eq 1 ] && grep -q "transported photon count differs" "$P2/b2.out" \
    && ok "branson check_log.py cmp: energies / T_e / transported photons within the margins pass; a 20 % photon-count difference fails" \
    || bad "branson cmp rc1=$rc1 rc2=$rc2 $(tail -2 "$P2/b1.out" "$P2/b2.out" | tr '\n' '|')"
# the hohlraum decks print no per-cell temperature table: the T_e criterion does not apply, energies and
# the photon count are still compared (a table on one side only stays an error)
sed '/^  [0-9]  [0-9.]*  0\.2  0\.3$/d; /cell(s) to the GPU/d' "$P2/b_gpu.log" > "$P2/b_gpu_note.log"
sed '/^  [0-9]  [0-9.]*  0\.2  0\.3$/d; /cell(s) to the GPU/d' "$P2/b_cpu.log" > "$P2/b_cpu_note.log"
python3 "$R/level2/branson/check_log.py" cmp "$P2/b_gpu_note.log" "$P2/b_cpu_note.log" > "$P2/b3.out" 2>&1; rc3=$?
python3 "$R/level2/branson/check_log.py" cmp "$P2/b_gpu.log" "$P2/b_cpu_note.log" > "$P2/b4.out" 2>&1; rc4=$?
[ $rc3 -eq 0 ] && grep -q "no per-cell temperature table" "$P2/b3.out" && [ $rc4 -eq 1 ] && grep -q "T_e cell count differs" "$P2/b4.out" \
    && ok "branson check_log.py cmp: no T_e table in either log -> criterion not applicable, PASS on energies and photons; a table on one side only -> FAIL" \
    || bad "branson cmp without T_e rc3=$rc3 rc4=$rc4 $(tail -2 "$P2/b3.out" "$P2/b4.out" | tr '\n' '|')"
bash -n "$R/level2/branson/check_reference.sh" && ok "branson check_reference.sh: valid shell" || bad "branson check_reference.sh syntax"
python3 "$TOOL" check "$R/level2/branson" hohlraum-multi-node --out "$P2/b_dry" --dry-run 2>/dev/null | python3 -c "
import json, sys; d = json.load(sys.stdin)
assert d['command'][:2] == ['bash', '$R/level2/branson/check_reference.sh'] and d['command'][3].endswith('/level2/branson/inputs/3D_hohlraum_multi_node.xml') and d['command'][2].endswith('/run/stdout.log'), d
" && ok "branson check: the reference checker gets the run's log and the registered deck (dry run)" || bad "branson check dry-run"
# exampm: the final-state line -> rules (count exact, bounds, volume within 1 %); dumps checker refuses an empty directory
printf 'Time 0.000000 / 0.250000\nExaMPM final state: step 250 time 0.25 particles 15400000 initial 15400000 pos_min 0.0012 pos_max 0.9876 volume_ratio 1.0004 v_mean 0.1 -0.2 -0.3 x_mean 0.5 0.5 0.31\n' > "$P2/ex_ok.log"
sed 's/particles 15400000 initial/particles 15399999 initial/' "$P2/ex_ok.log" > "$P2/ex_lost.log"
sed 's/volume_ratio 1.0004/volume_ratio 1.02/' "$P2/ex_ok.log" > "$P2/ex_vol.log"
sed 's/pos_max 0.9876/pos_max 1.5/' "$P2/ex_ok.log" > "$P2/ex_out.log"
python3 "$TOOL" extract "$R/level2/exampm" "$P2/ex_ok.log" --input dambreak-0.005 > "$P2/ex_q.json" 2>/dev/null
mkbase "$R/level2/exampm" dambreak-0.005 "$P2/ex_q.json" > "$P2/ex_base.json"
python3 "$TOOL" compare "$R/level2/exampm" "$P2/ex_base.json" "$P2/ex_ok.log" --input dambreak-0.005 > "$P2/ex_c1.json" 2>/dev/null; rc1=$?
python3 "$TOOL" compare "$R/level2/exampm" "$P2/ex_base.json" "$P2/ex_lost.log" --input dambreak-0.005 > /dev/null 2>&1; rc2=$?
python3 "$TOOL" compare "$R/level2/exampm" "$P2/ex_base.json" "$P2/ex_vol.log" --input dambreak-0.005 > /dev/null 2>&1; rc3=$?
python3 "$TOOL" compare "$R/level2/exampm" "$P2/ex_base.json" "$P2/ex_out.log" --input dambreak-0.005 > /dev/null 2>&1; rc4=$?
[ $rc1 -eq 0 ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['verdict']=='PASS' and not d['required_pending'], d" "$P2/ex_c1.json" \
    && [ $rc2 -eq 1 ] && [ $rc3 -eq 1 ] && [ $rc4 -eq 1 ] \
    && ok "exampm final-state rules: identical state PASS (no pending required quantity); a lost particle, a 2 % volume change and a particle outside the box each FAIL" \
    || bad "exampm final-state rules rc=$rc1/$rc2/$rc3/$rc4"
for id in dambreak-0.01 dambreak-0.05-upstream; do
    python3 "$TOOL" extract "$R/level2/exampm" "$P2/ex_ok.log" --input $id 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); assert d['particles_final']['value']==15400000 and d['volume_ratio']['value']==1.0004, d" \
        && ok "exampm $id: the same final-state quantities apply" || bad "exampm $id quantities"
done
python3 "$TOOL" extract "$R/level2/exampm" "$P2/ex_ok.log" --input freefall-0.01 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); assert set(d)=={'last_time_line'}, d" \
    && ok "exampm freefall-0.01: keeps the benchmark-level quantities (the box of the free fall is not the unit cube)" || bad "exampm freefall quantities"
mkdir -p "$P2/nodumps"; python3 "$R/level2/exampm/check_dambreak.py" "$P2/nodumps" > "$P2/db.out" 2>&1; rc=$?
[ $rc -eq 1 ] && grep -q "^FAIL: exampm dam-break check" "$P2/db.out" && grep -q "at least 2 expected" "$P2/db.out" \
    && ok "exampm check_dambreak.py: no dumps -> FAIL (nothing verified)" || bad "check_dambreak.py empty dir rc=$rc $(cat "$P2/db.out")"
python3 "$TOOL" check "$R/level2/exampm" dambreak-0.01 --out "$P2/ex_dry" --dry-run 2>/dev/null | python3 -c "
import json, sys; d = json.load(sys.stdin)
assert d['outputs'][0]['new'] is True and d['outputs'][0]['glob'].endswith('build/level2/exampm/cuda/run/particles_*.h5') and d['command'][-2:] == ['--min-dumps', '2'], d
" && ok "exampm check: only the dumps written by this run are captured (new: true), checker gets the outputs directory (dry run)" || bad "exampm check dry-run"
# shaw: final-state norms, exact
printf 'hpcperf final state: nrm2(vp) = 1.2345678901234567e-03 nrm2(sp) = 9.8765432109876543e+01\nloopTime = 1.5\n' > "$P2/sh_ok.log"
sed 's/9.8765432109876543e+01/9.8765432109876700e+01/' "$P2/sh_ok.log" > "$P2/sh_diff.log"   # differs in the 15th digit
python3 "$TOOL" extract "$R/level2/shaw" "$P2/sh_ok.log" --input prem-500x2500 > "$P2/sh_q.json" 2>/dev/null
mkbase "$R/level2/shaw" prem-500x2500 "$P2/sh_q.json" > "$P2/sh_base.json"
python3 "$TOOL" compare "$R/level2/shaw" "$P2/sh_base.json" "$P2/sh_ok.log" --input prem-500x2500 > "$P2/sh_c.json" 2>/dev/null; rc1=$?
python3 "$TOOL" compare "$R/level2/shaw" "$P2/sh_base.json" "$P2/sh_diff.log" --input prem-500x2500 > /dev/null 2>&1; rc2=$?
[ $rc1 -eq 0 ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['verdict']=='PASS', d" "$P2/sh_c.json" && [ $rc2 -eq 1 ] \
    && ok "shaw final-state norms: exact rule -- identical PASS, a 15th-digit change FAIL" || bad "shaw norms rc=$rc1/$rc2"
# vlp4d sld10: file-sourced rows from outputs/nrj.out, exact; check_nrj.py structure
mkdir -p "$P2/vl/outputs"; python3 -c "
for i in range(41): print(f'{i*0.01:.13e} {-0.8-0.003*i:.13e} {(1e-15 if i%2 else -2e-16):.13e}')" > "$P2/vl/outputs/nrj.out"
printf 'Number of total iterations : 40\n' > "$P2/vl/stdout.log"
python3 - "$R" "$P2/vl" <<'PY' && ok "vlp4d sld10: e_norm_final / mass_final come from the captured nrj.out (last row), exact rules" || bad "vlp4d file-sourced quantities"
import sys, json; sys.path.insert(0, sys.argv[1] + "/tools/inputs"); import hpcperf_inputs as hi
from pathlib import Path
doc = hi.load(sys.argv[1] + "/level2/p3_vlp4d"); inp = hi.get_input(doc, "sld10")
q = hi.extract(doc, Path(sys.argv[2]) / "stdout.log", inp, run_dir=sys.argv[2])
assert abs(q["e_norm_final"]["value"] - (-0.8 - 0.003 * 40)) < 1e-12 and q["mass_final"]["value"] == -2e-16 and q["iterations"]["value"] == 40, q
r = hi.compare(doc, q, q, inp); assert r["verdict"] == "PASS" and not r["required_pending"], r
q2 = dict(q); q2["e_norm_final"] = dict(q["e_norm_final"], value=q["e_norm_final"]["value"] + 1e-12)
assert hi.compare(doc, q, q2, inp)["verdict"] == "FAIL"
q3 = hi.extract(doc, Path(sys.argv[2]) / "stdout.log", hi.get_input(doc, "sld10-large"), run_dir=sys.argv[2])
assert set(q3) == {"iterations"}, q3          # the large deck keeps its own quantities (the Landau fit is its check)
PY
python3 "$R/level2/p3_vlp4d/check_nrj.py" "$P2/vl/outputs/nrj.out" --lines 41 > "$P2/vl1.out" 2>&1; rc1=$?
head -40 "$P2/vl/outputs/nrj.out" > "$P2/vl_short.out"; python3 "$R/level2/p3_vlp4d/check_nrj.py" "$P2/vl_short.out" --lines 41 > /dev/null 2>&1; rc2=$?
sed '5s/.*/4.0000000000000e-02 -8.1e-01 1.0e-06/' "$P2/vl/outputs/nrj.out" > "$P2/vl_mass.out"; python3 "$R/level2/p3_vlp4d/check_nrj.py" "$P2/vl_mass.out" --lines 41 > /dev/null 2>&1; rc3=$?
[ $rc1 -eq 0 ] && grep -q "^PASS: vlp4d nrj check" "$P2/vl1.out" && [ $rc2 -eq 1 ] && [ $rc3 -eq 1 ] \
    && ok "vlp4d check_nrj.py: 41 finite rows PASS; 40 rows FAIL; a 1e-6 mass residual FAIL" || bad "check_nrj.py rc=$rc1/$rc2/$rc3"
# aobench image check
python3 -c "
import sys; w,h=4,3; sys.stdout.buffer.write(b'P6\n%d %d\n255\n' % (w,h) + bytes(range(w*h*3)))" > "$P2/ao.ppm"
sha="$(sha256sum "$P2/ao.ppm" | cut -d' ' -f1)"; echo "$sha  ao.ppm" > "$P2/ref.sha256"; echo "0000  ao.ppm" > "$P2/ref_wrong.sha256"
python3 "$R/level1/ao_bench/check_ppm.py" "$P2/ao.ppm" --reference "$P2/ref.sha256" > "$P2/ao1.out" 2>&1; rc1=$?
python3 "$R/level1/ao_bench/check_ppm.py" "$P2/ao.ppm" --reference "$P2/ref_wrong.sha256" > "$P2/ao2.out" 2>&1; rc2=$?
python3 "$R/level1/ao_bench/check_ppm.py" "$P2/ao.ppm" --reference "$P2/ref_missing.sha256" > "$P2/ao3.out" 2>&1; rc3=$?
[ $rc1 -eq 0 ] && grep -q "^PASS: aobench image check" "$P2/ao1.out" && [ $rc2 -eq 1 ] && grep -q "sha256 differs" "$P2/ao2.out" \
    && [ $rc3 -eq 1 ] && grep -q "no reference image hash" "$P2/ao3.out" \
    && ok "aobench check_ppm.py: byte-identical PASS; a different image FAIL; no captured reference FAIL (never PASS by default)" \
    || bad "check_ppm.py rc=$rc1/$rc2/$rc3"
python3 "$TOOL" check "$R/level1/ao_bench" iter100 --out "$P2/ao_dry" --dry-run 2>/dev/null | python3 -c "
import json, sys; d = json.load(sys.stdin)
assert d['outputs'][0]['glob'].endswith('/ao_dry/run/ao.ppm') and d['command'][-1].endswith('level1/ao_bench/reference/iter100.sha256'), d
" && ok "aobench check: ao.ppm of the run directory is captured and compared with the repository reference (dry run)" || bad "aobench check dry-run"
# quicksilver seed scatter statistics
qslog() {  # $1 file  $2 flux
printf '      99       150684        16384         3165            0        53241      1334462        66755       106691      1454458            0      2000      3000    %s    1.0    2.0    3.0\nPASS:: Absorption / Fission / Scatter Ratios maintained with 1%% tolerance\nPASS:: Collision to Facet Crossing Ratio maintained even balanced within 1%% tolerance\nPASS:: No Particles Lost During Run\nPASS:: Fluence is homogenous across cells with 6%% tolerance\n' "$2" > "$1"
}
for s in 11 22 33; do mkdir -p "$P2/qs/seed-$s"; echo "rc=0 seed=$s" > "$P2/qs/seed-$s/DONE"; done
qslog "$P2/qs/seed-11/stdout.log" 1.00e+02; qslog "$P2/qs/seed-22/stdout.log" 1.02e+02; qslog "$P2/qs/seed-33/stdout.log" 0.98e+02
python3 "$R/level2/quicksilver/seed_stats.py" "$P2/qs" --k 4 > "$P2/qs.out" 2>&1; rc=$?
[ $rc -eq 0 ] && grep -q "seed-varied runs used: 3" "$P2/qs.out" && grep -qE "scalar_flux +mean 100 +sigma 2 " "$P2/qs.out" && grep -q "tol 8.000e-02" "$P2/qs.out" \
    && ok "quicksilver seed_stats.py: mean / sigma of the final-cycle scalar flux over the seeds and the k-sigma near rule it implies (nothing written)" \
    || bad "seed_stats.py rc=$rc $(cat "$P2/qs.out" | tr '\n' '|')"
bash -n "$R/level2/quicksilver/seed_scatter.sh" && ok "quicksilver seed_scatter.sh: valid shell" || bad "seed_scatter.sh syntax"
for d in level2/branson level2/exampm level2/shaw level2/p3_vlp4d level1/ao_bench level2/quicksilver level2/examinimd level2/hipbone level2/miniweather level2/miniem; do
    python3 "$TOOL" validate "$R/$d" >/dev/null 2>&1 && ok "validate $d (second pass)" || bad "validate $d"
done

echo; echo "inputs tests: $pass passed, $failn failed, $skip skipped"
[ $failn -eq 0 ]
