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
[ $rc -eq 3 ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); c={x['name']:x for x in d['checks']}; assert d['verdict']=='INCOMPLETE' and d['required_pending']==['final_cycle_scalar_flux'] and d['diagnostic_recorded']==['final_cycle_census','final_cycle_num_seg'] and all(c[k]['ok'] for k in ('pass_ratios','pass_facet','pass_no_loss','pass_fluence','fail_marker'))" "$TMP/c.json" 2>/dev/null \
    && ok "quicksilver coverage: identical log -> native checks ok, scalar flux only recorded -> exit 3 INCOMPLETE (no numeric baseline comparison exists)" || bad "quicksilver coverage control rc=$rc $(cat "$TMP/c.json")"
# change the extracted science result (scalar flux x 1.5) but keep every PASS:: line
sed -E 's/^(\s+19\s.*\s)5\.918521e\+05(\s)/\18.877782e+05\2/' "$FX/quicksilver_two_tables.log" > "$TMP/qs_flux_changed.log"
grep -q '8.877782e+05' "$TMP/qs_flux_changed.log" || bad "quicksilver fixture edit did not apply"
python3 "$TOOL" compare "$R/level2/quicksilver" "$TMP/qs_baseline.json" "$TMP/qs_flux_changed.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 3 ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); c={x['name']:x for x in d['checks']}; assert d['ok'] and d['verdict']=='INCOMPLETE' and c['final_cycle_scalar_flux']['ok'] and abs(c['final_cycle_scalar_flux']['value']-887778.2)<1 and abs(c['final_cycle_scalar_flux']['baseline']-591852.1)<1 and c['final_cycle_scalar_flux']['rule']=='record'" "$TMP/c.json" 2>/dev/null \
    && ok "quicksilver coverage: scalar flux changed x1.5 with all PASS:: lines kept -> NOT detected as a failure (record: 591852.1 -> 887778.2 shown), exit 3 INCOMPLETE -- the differential comparison is not ready" || bad "quicksilver coverage flux rc=$rc $(cat "$TMP/c.json")"
sed -e '/^PASS:: Fluence/d' "$FX/quicksilver_two_tables.log" > "$TMP/qs_nofluence.log"
python3 "$TOOL" compare "$R/level2/quicksilver" "$TMP/qs_baseline.json" "$TMP/qs_nofluence.log" >"$TMP/c.json" 2>/dev/null; rc=$?
[ $rc -eq 1 ] && grep -q '"failed": \[' "$TMP/c.json" && grep -q '"pass_fluence"' "$TMP/c.json" && ok "quicksilver coverage: a missing upstream PASS:: line -> exit 1 FAIL (the native checks are what is verified)" || bad "quicksilver coverage marker rc=$rc"

echo; echo "inputs tests: $pass passed, $failn failed, $skip skipped"
[ $failn -eq 0 ]
