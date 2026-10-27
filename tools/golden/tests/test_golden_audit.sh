#!/usr/bin/env bash
# tools/golden/tests/test_golden_audit.sh -- CPU-only tests of tools/golden/hpcperf_golden_audit.py (no GPU, no real
# benchmark, no CMake build).
#
# A fake repository holds one fake Level 1 benchmark (a Python "binary" that captures its outputs through
# hpcperf_golden_capture.py), its two references made by the real `hpcperf_golden.py make`, a fake ctest (prints a
# --show-only=json-v1 payload the test controls), a fake timing table and a fake Level 2 app with a validate.sh. Every
# negative test breaks exactly one link (spec, args, reference, provenance, ctest, registry check, ...) and expects
# the audit's finding code and exit status; the clean state must give exit 0 and change nothing. Exit 0 when every
# test passes.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$(cd "$HERE/.." && pwd)"
G="$TOOLS/hpcperf_golden.py"
A="$TOOLS/hpcperf_golden_audit.py"
PY="${PYTHON:-python3}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/golden-audit-tests.XXXXXX")"
[ -n "${HPCPERF_AUDIT_TEST_KEEP:-}" ] && echo "keeping $TMP" || trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
ok()  { echo "ok   $*"; pass=$((pass+1)); }
bad() { echo "FAIL $*"; fail=$((fail+1)); }
GIT() { git -C "$FR" -c user.email=t@t -c user.name=t "$@"; }
export HPCPERF_GOLDEN_STORE="$TMP/store"
unset HPCPERF_GOLDEN_ALLOW_UNCOMMITTED_REFERENCE HPCPERF_SKIP_VERIFY HPCPERF_CPU_REFERENCE HPCPERF_CTEST

FR="$TMP/repo"; B="$FR/level1/fake"; BD="$FR/build/fake/cuda"
mkdir -p "$B/cuda" "$BD" "$FR/tools/timing/cases" "$FR/level2/app"
: > "$FR/hpcperf_env.sh"
printf 'build/\n' > "$FR/.gitignore"
echo "// source" > "$B/cuda/main.cu"
cat > "$B/CMakeLists.txt" <<'CM'
add_test(NAME fake_run COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/golden/hpcperf_golden.py
         run ${CMAKE_CURRENT_SOURCE_DIR} --exe $<TARGET_FILE:fake> --workdir ${CMAKE_CURRENT_BINARY_DIR}/golden-ctest)
CM
cat > "$BD/fake" <<PYBIN
#!$(command -v "$PY")
# fake benchmark: y[i] = sin(i*k) (f64, max_abs), idx[i] = i*k (i32, exact), small[i] = 1e-12 i (f64, max_abs 1e-8),
# flat = 7 (i32, exact); CPU reference y.cpu = y + FAKE_CPU_DELTA (default 1e-10), small.cpu = small
import os, sys
sys.path.insert(0, "$TOOLS")
import numpy as np, hpcperf_golden_capture as gc
n, k = int(sys.argv[1]), float(sys.argv[2])
y = np.sin(np.arange(n) * k)
small = np.arange(n) * 1e-12
gc.capture("y", y, "f64"); gc.capture("idx", (np.arange(n) * int(k * 10)).astype(np.int32), "i32")
gc.capture("small", small, "f64"); gc.capture("flat", np.full(n, 7, np.int32), "i32")
if os.environ.get("HPCPERF_CPU_REFERENCE") == "1":
    gc.capture("y.cpu", y + float(os.environ.get("FAKE_CPU_DELTA", "1e-10")), "f64")
    gc.capture("small.cpu", small, "f64")
    print("PASS")
print("Average kernel execution time 1.0 (ms)")
PYBIN
chmod +x "$BD/fake"
cat > "$B/inputs.yaml" <<'YAML'
schema: hpcperf-inputs-1
benchmark: fake
level: 1
selector: null
default_input: n1000
entry: {kind: binary, path: build/fake/cuda/fake}
timing: {scope: fake, kind: per_iteration, unit: ms, regex: '^Average kernel execution time (?P<value>[0-9.eE+-]+) \(ms\)', select: only, work: {key: repeat, offset: 0}}
baseline:
  method: fake
  reference: fake
  quantities:
  - {name: fail_marker, role: diagnostic, regex: '^FAIL$', compare: {rule: absent}}
golden:
  mode: data
  cpu_check: {pass_regex: '^PASS$', fail_regex: '^FAIL$'}
  outputs:
  - {name: y, dtype: f64, rule: {metric: max_abs, tol: 1.0e-8}, cpu: y.cpu, basis: fake upstream abs 1e-8}
  - {name: idx, dtype: i32, rule: {metric: exact}, basis: fake exact}
  - {name: small, dtype: f64, rule: {metric: max_abs, tol: 1.0e-8}, cpu: small.cpu, basis: fake abs on a tiny signal}
  - {name: flat, dtype: i32, rule: {metric: exact}, basis: fake constant output}
check:
  kind: post_run
  run_env: {HPCPERF_GOLDEN_DUMP: '{run_dir}/golden-capture'}
  command: [python3, '{repo}/tools/golden/hpcperf_golden.py', check, '{bench_dir}', '{input_id}', --dump, '{run_dir}/golden-capture']
  pass_regex: '^PASS: golden check'
  fail_regex: '^FAIL: golden check'
  basis: fake golden check
  covers: all
coverage: {status: MULTI_INPUT, reason: fake}
inputs:
- id: n1000
  case: fake
  variant: default
  source: {kind: custom, derivation: fake}
  params: {n: 1000, repeat: 1}
  args: ['1000', '0.1']
  runtime_config: {gpus: 1}
  build_config: null
  backends_validated: [cuda]
- id: n2000
  case: fake
  variant: size
  source: {kind: custom, derivation: fake}
  params: {n: 2000, repeat: 1}
  args: ['2000', '0.1']
  runtime_config: {gpus: 1}
  build_config: null
  backends_validated: [cuda]
- id: probe-n300
  purpose: correctness-probe
  probe_basis: fake probe
  case: fake
  variant: parameter
  source: {kind: custom, derivation: fake}
  params: {n: 300, repeat: 1}
  args: ['300', '0.3']
  runtime_config: {gpus: 1}
  build_config: null
  backends_validated: [cuda]
YAML
cat > "$FR/tools/timing/cases/level1_extra.tsv" <<'TSV'
# fake extra timing cases
#app	case	args	env	timeout_s	notes
fake	n2000	2000 0.1	-	-	a registered input
TSV
# a fake Level 2 app: only its validate.sh is scanned (static)
cat > "$FR/level2/app/inputs.yaml" <<'YAML'
schema: hpcperf-inputs-1
benchmark: app
level: 2
selector: HPCPERF_APP_INPUT
default_input: small
entry: {kind: run.sh, path: level2/app/run.sh, backend_arg: CUDA}
timing: {kind: none, status: NEEDS_TIMING_SUPPORT, reason: fake}
baseline:
  method: fake
  reference: fake
  quantities:
  - {name: ok_marker, role: required, regex: '^OK$', compare: {rule: present}}
inputs:
- {id: small, case: fake, variant: default, source: {kind: custom, derivation: fake}, params: {n: 1}, backends_validated: [cuda]}
YAML
echo 'echo run' > "$FR/level2/app/run.sh"
cat > "$FR/level2/app/validate.sh" <<'SH'
#!/usr/bin/env bash
case "${HPCPERF_CPU_REFERENCE:-}" in
  ''|0*) CPU_REF=0 ;;
  *)     CPU_REF=1 ;;
esac
bash "$HERE/run.sh" CUDA
if [ "$CPU_REF" = 1 ]; then
    bash "$HERE/build.sh" Serial
    bash "$HERE/run.sh" Serial
else
    echo "CPU reference: skipped (default; frozen reference instead)"
fi
[ "$CPU_REF" = 1 ] && bash "$HERE/run.sh" OpenMP
bash "$HERE/run.sh" --arch Sequential
SH
# the fake ctest: prints the payload of <test dir>/ctest.json
cat > "$TMP/ctest" <<'SH'
#!/usr/bin/env bash
d=""; while [ $# -gt 0 ]; do [ "$1" = --test-dir ] && { d="$2"; shift; }; shift; done
cat "$d/ctest.json"
SH
chmod +x "$TMP/ctest"
: > "$BD/CTestTestfile.cmake"
ctest_json() {   # ctest_json <command tail after the python interpreter as a JSON list> <pass regex JSON> [extra properties JSON]
    "$PY" - "$BD/ctest.json" "$1" "$2" "${3:-[]}" <<'PY'
import json, sys
out, cmd, pr, extra = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3]), json.loads(sys.argv[4])
props = [{"name": "FAIL_REGULAR_EXPRESSION", "value": ["FAIL: golden check"]}, {"name": "TIMEOUT", "value": 600.0}]
if pr is not None:
    props.append({"name": "PASS_REGULAR_EXPRESSION", "value": pr})
json.dump({"kind": "ctestInfo", "version": {"major": 1, "minor": 0}, "tests": [
    {"name": "fake_run", "command": cmd, "properties": props + extra}]}, open(out, "w"))
PY
}
GOLDEN_CMD="[\"/usr/bin/python3\", \"$FR/level1/fake/../../tools/golden/hpcperf_golden.py\", \"run\", \"$B\", \"--exe\", \"$BD/fake\", \"--workdir\", \"$BD/golden-ctest\"]"
ctest_json "$GOLDEN_CMD" '["PASS: golden check"]'
touch -d '1 hour ago' "$B/CMakeLists.txt"
GIT init -q && GIT add -A && GIT commit -qm fake || { echo "FAIL setup: git"; exit 1; }
touch "$BD/fake"
for i in n1000 n2000 probe-n300; do
    "$PY" "$G" make "$B" "$i" --workdir "$TMP/mk-$i" > "$TMP/make-$i.log" 2>&1 || { echo "FAIL setup: make $i"; cat "$TMP/make-$i.log"; exit 1; }
done
GIT add level1/fake/reference && GIT commit -qm "fake references" || { echo "FAIL setup: commit references"; exit 1; }
BASE="$(GIT rev-parse HEAD)"

audit() {   # audit <tag> [audit args...]  -> $TMP/<tag>.json / .txt, rc in $rc
    local tag="$1"; shift
    "$PY" "$A" --root "$FR" --ctest "$TMP/ctest" --aggregate-build '' --json "$TMP/$tag.json" --table "$TMP/$tag.txt" "$@" > /dev/null 2>"$TMP/$tag.err"
    rc=$?
}
has() {     # has <tag> <severity> <code> [input]: the finding exists
    "$PY" - "$TMP/$1.json" "$2" "$3" "${4:-}" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); sev, code, inp = sys.argv[2], sys.argv[3], sys.argv[4]
sys.exit(0 if any(f["severity"] == sev and f["code"] == code and (not inp or f.get("input_id") == inp) for f in r["findings"]) else 1)
PY
}
nhard() { "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['summary']['findings']['HARD'])" "$TMP/$1.json"; }
fixtimes() { touch -d '1 hour ago' "$B/CMakeLists.txt"; touch "$BD/CTestTestfile.cmake" "$BD/fake"; }
reset() { GIT checkout -q -- . && GIT clean -qfd && ctest_json "$GOLDEN_CMD" '["PASS: golden check"]' && fixtimes; }
hardreset() { GIT reset -q --hard "$BASE" && GIT clean -qfd && ctest_json "$GOLDEN_CMD" '["PASS: golden check"]' && fixtimes; }

echo "=== 1: the clean fake repository"
before="$(cd "$FR" && git status --porcelain; find "$FR" -path "$FR/.git" -prune -o -type f -newer "$TMP/ctest" -print | sort | xargs -r sha256sum)"
audit clean
after="$(cd "$FR" && git status --porcelain; find "$FR" -path "$FR/.git" -prune -o -type f -newer "$TMP/ctest" -print | sort | xargs -r sha256sum)"
[ "$rc" = 0 ] && [ "$(nhard clean)" = 0 ] && ok "1a: exit 0, no HARD finding" || { bad "1a: rc=$rc hard=$(nhard clean)"; cat "$TMP/clean.txt" "$TMP/clean.err"; }
[ "$before" = "$after" ] && ok "1b: the audit changed nothing in the repository" || bad "1b: the tree changed"
"$PY" - "$TMP/clean.json" <<'PY' && ok "1c: report schema, 3 golden references (one a probe), the golden ctest of the default input, the golden registry check" || bad "1c: report content"
import json, sys
r = json.load(open(sys.argv[1]))
assert r["schema"] == "hpcperf-golden-audit-1"
assert r["summary"]["golden_references"] == 3, r["summary"]
b = next(x for x in r["benchmarks"] if x["benchmark"] == "fake")
t = b["ctest"][0]["tests"][0]
assert t["wrapper"] == "golden" and t["input_id"] == "n1000", t
assert all(i["check"].get("golden") for i in b["inputs"]), b["inputs"]
assert b["mechanism"] == "golden"
PY
has clean WARN ZERO_OUTPUT_PASSES && ok "1d: 'small' (|golden| <= 1e-9 against tol 1e-8): an all-zero output passes -> WARN" || bad "1d: no ZERO_OUTPUT_PASSES"
has clean WARN GOLDEN_CONSTANT && ok "1e: 'flat' (every value 7, exact) -> WARN GOLDEN_CONSTANT" || bad "1e: no GOLDEN_CONSTANT"
audit strict --strict
[ "$rc" = 1 ] && ok "1f: --strict turns the WARN findings into exit 1" || bad "1f: --strict rc=$rc"
! has clean WARN BUDGET_CONSUMED && ! has clean REVIEW BUDGET_CONSUMED && ok "1g: d0 1e-10 of tol 1e-8 is no budget finding" || bad "1g: budget"

echo
echo "=== 2: registry vs frozen reference"
sed -i 's/rule: {metric: max_abs, tol: 1.0e-8}, cpu: y.cpu/rule: {metric: max_abs, tol: 2.0e-8}, cpu: y.cpu/' "$B/inputs.yaml"
audit spec
[ "$rc" = 1 ] && has spec HARD REFERENCE_SPEC_MISMATCH n1000 && has spec HARD REFERENCE_RULE_MISMATCH n1000 \
    && ok "2a: a tolerance changed in inputs.yaml after freezing -> HARD REFERENCE_SPEC_MISMATCH / RULE_MISMATCH, exit 1" || bad "2a: rc=$rc"
reset
sed -i "s/args: \['2000', '0.1'\]/args: ['2000', '0.2']/" "$B/inputs.yaml"
audit args
[ "$rc" = 1 ] && has args HARD REFERENCE_WORKLOAD_MISMATCH n2000 && has args HARD REFERENCE_COMMAND_MISMATCH n2000 \
    && ok "2b: the registry's arguments of an input changed -> HARD WORKLOAD_MISMATCH + COMMAND_MISMATCH" || bad "2b: rc=$rc"
has args WARN TIMING_CASE_UNREGISTERED && ok "2c: the timing extra case no longer matches a registered input -> WARN TIMING_CASE_UNREGISTERED" \
    || bad "2c: no TIMING_CASE_UNREGISTERED"
reset
GIT rm -q level1/fake/reference/n2000.json && GIT commit -qm "drop n2000"
audit missing
[ "$rc" = 1 ] && has missing HARD REFERENCE_MISSING n2000 && ok "2d: a golden input without a reference and without a reason -> HARD REFERENCE_MISSING" || bad "2d: rc=$rc"
"$PY" - "$B/inputs.yaml" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
s = s.replace("- id: n2000\n", "- id: n2000\n  # NO GOLDEN YET (decision pending): fake reason\n")
open(p, "w").write(s)
PY
audit recorded
[ "$rc" = 0 ] && has recorded REVIEW REFERENCE_MISSING_RECORDED n2000 && ok "2e: the same with a 'NO GOLDEN' comment in the input's block -> REVIEW, exit 0" \
    || bad "2e: rc=$rc"
hardreset
cp "$B/reference/n1000.json" "$B/reference/n9999.json"
audit orphan
[ "$rc" = 1 ] && has orphan HARD REFERENCE_ORPHAN && has orphan HARD REFERENCE_UNCOMMITTED && ok "2f: a reference of an unregistered input (untracked) -> HARD ORPHAN + UNCOMMITTED" \
    || bad "2f: rc=$rc"
rm -f "$B/reference/n9999.json"

echo
echo "=== 3: tampered or foreign references"
"$PY" - "$B/reference/n1000.json" <<'PY'
import json, sys
p = sys.argv[1]; r = json.load(open(p))
for o in r["outputs"]:
    if o["name"] == "y":
        o["rule"]["tol_effective"] = 1.0e-8          # = tol, as if d0 had been ignored
open(p, "w").write(json.dumps(r, indent=2) + "\n")
PY
audit tamper
[ "$rc" = 1 ] && has tamper HARD REFERENCE_TOL_EFFECTIVE_MISMATCH n1000 && has tamper HARD REFERENCE_UNCOMMITTED n1000 \
    && ok "3a: an edited tol_effective -> HARD TOL_EFFECTIVE_MISMATCH (+ UNCOMMITTED)" || bad "3a: rc=$rc"
GIT checkout -q -- level1/fake/reference
side="$(GIT commit-tree -m side "$(GIT rev-parse HEAD^{tree})")"
"$PY" - "$B/reference/n1000.json" "$side" <<'PY'
import json, sys
p = sys.argv[1]; r = json.load(open(p)); r["provenance"]["git_head"] = sys.argv[2]
open(p, "w").write(json.dumps(r, indent=2) + "\n")
PY
GIT commit -qam "reference from a side commit"
audit side
[ "$rc" = 1 ] && has side HARD REFERENCE_NOT_ANCESTOR n1000 && ok "3b: provenance git_head not an ancestor of HEAD -> HARD REFERENCE_NOT_ANCESTOR" || bad "3b: rc=$rc"
"$PY" - "$B/reference/n1000.json" <<'PY'
import json, sys
p = sys.argv[1]; r = json.load(open(p)); r["provenance"]["git_head"] = "0123456789abcdef0123456789abcdef01234567"; r["provenance"]["source_clean"] = False
open(p, "w").write(json.dumps(r, indent=2) + "\n")
PY
GIT commit -qam "reference from an unknown commit, dirty"
audit unknown
[ "$rc" = 1 ] && has unknown HARD REFERENCE_UNKNOWN_COMMIT n1000 && has unknown HARD REFERENCE_SOURCE_DIRTY n1000 \
    && ok "3c: an unknown provenance commit / a dirty source tree -> HARD" || bad "3c: rc=$rc"
hardreset
echo "// a later change" >> "$B/cuda/main.cu"; GIT commit -qam "fake: change the kernel after freezing"
audit srcchg
[ "$rc" = 0 ] && has srcchg REVIEW REFERENCE_SOURCES_CHANGED n1000 && /usr/bin/grep -q "change the kernel after freezing" "$TMP/srcchg.txt" \
    && ok "3d: a source commit after the reference -> REVIEW REFERENCE_SOURCES_CHANGED naming the commit, exit 0" || bad "3d: rc=$rc"
echo "// uncommitted" >> "$B/cuda/main.cu"
audit srcwt
has srcwt REVIEW REFERENCE_SOURCES_UNCOMMITTED n1000 && ok "3e: an uncommitted source change -> REVIEW REFERENCE_SOURCES_UNCOMMITTED" || bad "3e"
hardreset
rm -rf "$HPCPERF_GOLDEN_STORE.away"; mv "$HPCPERF_GOLDEN_STORE" "$HPCPERF_GOLDEN_STORE.away"
audit noblob
[ "$rc" = 0 ] && has noblob WARN BLOB_MISSING && ok "3f: a golden blob missing from the store -> WARN BLOB_MISSING" || bad "3f: rc=$rc"
mv "$HPCPERF_GOLDEN_STORE.away" "$HPCPERF_GOLDEN_STORE"

echo
echo "=== 4: ctest"
ctest_json "[\"/usr/bin/true\"]" '["PASS"]'
audit ct_direct
[ "$rc" = 1 ] && has ct_direct HARD CTEST_NOT_GOLDEN && has ct_direct WARN CTEST_EXTRA_NON_GOLDEN && ok "4a: a golden benchmark whose ctest runs the binary directly -> HARD CTEST_NOT_GOLDEN" \
    || bad "4a: rc=$rc"
ctest_json "[\"/usr/bin/python3\", \"$FR/tools/golden/hpcperf_golden.py\", \"run\", \"$B\", \"n2000\", \"--exe\", \"$BD/fake\"]" '["PASS: golden check"]'
audit ct_input
[ "$rc" = 1 ] && has ct_input HARD CTEST_DEFAULT_NOT_CHECKED && has ct_input WARN CTEST_EXTRA_INPUT \
    && ok "4b: the only golden ctest checks another performance input than default_input -> HARD DEFAULT_NOT_CHECKED" || bad "4b: rc=$rc"
"$PY" - "$BD/ctest.json" "$B" "$BD/fake" <<'PY'
import json, sys
out, b, exe = sys.argv[1:4]
props = [{"name": "PASS_REGULAR_EXPRESSION", "value": ["PASS: golden check"]}, {"name": "FAIL_REGULAR_EXPRESSION", "value": ["FAIL: golden check"]}]
t = lambda name, extra: {"name": name, "command": ["/usr/bin/python3", "/x/tools/golden/hpcperf_golden.py", "run", b] + extra + ["--exe", exe], "properties": props}
json.dump({"kind": "ctestInfo", "tests": [t("fake_run", []), t("fake_probe", ["probe-n300"])]}, open(out, "w"))
PY
audit ct_probe
[ "$rc" = 0 ] && has ct_probe INFO CTEST_PROBE_TEST && ! has ct_probe WARN CTEST_PROBE_WOULD_BE_TIMED \
    && ok "4k: default + probe golden ctests -> INFO CTEST_PROBE_TEST, exit 0" || bad "4k: rc=$rc"
echo "# makes one timing case per ctest test" > "$FR/tools/timing/gen_cases.py"
audit ct_probe2
has ct_probe2 WARN CTEST_PROBE_WOULD_BE_TIMED && ok "4l: ... and a gen_cases.py without a probe filter -> WARN CTEST_PROBE_WOULD_BE_TIMED" || bad "4l"
rm -f "$FR/tools/timing/gen_cases.py"
ctest_json "[\"/usr/bin/python3\", \"$FR/tools/golden/hpcperf_golden.py\", \"run\", \"$B\", \"--exe\", \"$TMP/other_binary\"]" '["PASS: golden check"]'
audit ct_exe
[ "$rc" = 1 ] && has ct_exe HARD CTEST_EXE_MISMATCH && ok "4c: the golden ctest runs another binary than the registry entry -> HARD" || bad "4c: rc=$rc"
ctest_json "$GOLDEN_CMD" '["PASS|SKIP_VERIFY"]'
audit ct_skip
[ "$rc" = 1 ] && has ct_skip HARD CTEST_GOLDEN_REGEX && ok "4d: a golden ctest pass regex that also matches SKIP_VERIFY -> HARD" || bad "4d: rc=$rc"
ctest_json "$GOLDEN_CMD" '["PASS: golden check"]' '[{"name": "ENVIRONMENT", "value": ["HPCPERF_GOLDEN_ALLOW_UNCOMMITTED_REFERENCE=1"]}]'
audit ct_env
[ "$rc" = 1 ] && has ct_env HARD CTEST_ENVIRONMENT && ok "4e: a ctest ENVIRONMENT that accepts uncommitted references -> HARD" || bad "4e: rc=$rc"
ctest_json "$GOLDEN_CMD" '["PASS: golden check"]' '[{"name": "WILL_FAIL", "value": true}]'
audit ct_willfail
[ "$rc" = 1 ] && has ct_willfail HARD CTEST_WILL_FAIL && ok "4f: WILL_FAIL inverts the verdict -> HARD" || bad "4f: rc=$rc"
reset
touch "$B/CMakeLists.txt"; touch -d '1 hour ago' "$BD/CTestTestfile.cmake"
audit ct_stale
has ct_stale WARN CTEST_BUILD_STALE && ok "4g: CMakeLists.txt newer than the configured CTestTestfile.cmake -> WARN CTEST_BUILD_STALE" || bad "4g"
touch -d '1 hour ago' "$B/CMakeLists.txt"; touch "$BD/CTestTestfile.cmake"
# a non-golden benchmark's ctest: exit-code only / generic word / skip marker
"$PY" - "$B/inputs.yaml" <<'PY'
import sys, yaml
p = sys.argv[1]; d = yaml.safe_load(open(p)); d.pop("golden"); d.pop("check")
yaml.safe_dump(d, open(p, "w"), sort_keys=False)
PY
GIT rm -rq level1/fake/reference && GIT commit -qam "fake: no golden"
sed -i 's/hpcperf_golden.py/verify.py/' "$B/CMakeLists.txt"; touch -d '1 hour ago' "$B/CMakeLists.txt"
ctest_json "[\"$BD/fake\", \"1000\", \"0.1\"]" 'null'
audit ng_exit
[ "$rc" = 0 ] && has ng_exit WARN CTEST_EXIT_CODE_ONLY && ok "4h: a non-golden ctest without a pass regex -> WARN CTEST_EXIT_CODE_ONLY" || bad "4h: rc=$rc"
ctest_json "[\"$BD/fake\", \"1000\", \"0.1\"]" '["Passed"]'
audit ng_generic
has ng_generic WARN CTEST_PASS_REGEX && ok "4i: pass regex 'Passed' -> WARN (a bare success word)" || bad "4i"
ctest_json "[\"$BD/fake\", \"1000\", \"0.1\"]" '["PASS|SKIP"]'
audit ng_skip
[ "$rc" = 1 ] && has ng_skip HARD CTEST_PASS_REGEX && ok "4j: pass regex matching a skip marker -> HARD" || bad "4j: rc=$rc"
"$PY" - "$B/inputs.yaml" <<'PY'
import sys, yaml
p = sys.argv[1]; d = yaml.safe_load(open(p))
d["check"] = {"kind": "post_run", "command": ["python3", "{bench_dir}/check_out.py", "{log}"], "pass_regex": "^PASS: fake reference check",
              "fail_regex": "^FAIL", "basis": "fake", "covers": "all"}
yaml.safe_dump(d, open(p, "w"), sort_keys=False)
PY
ctest_json "[\"/usr/bin/python3\", \"$B/verify.py\", \"$BD/fake\", \"1000\"]" '["PASS: sanity"]'
audit ng_div
has ng_div WARN CTEST_REGISTRY_DIVERGE && ok "4m: ctest runs verify.py, the registry check check_out.py -> WARN CTEST_REGISTRY_DIVERGE" || bad "4m"
hardreset

echo
echo "=== 5: the registry check of a golden input"
sed -i "s/check, '{bench_dir}', '{input_id}', --dump/check, '{bench_dir}', n1000, --dump/" "$B/inputs.yaml"
audit chk_id
[ "$rc" = 1 ] && has chk_id HARD CHECK_NOT_GOLDEN n2000 && ok "5a: a check command with a fixed input id (n2000 checked against n1000's reference) -> HARD" || bad "5a: rc=$rc"
reset
sed -i "s/pass_regex: '^PASS: golden check'/pass_regex: 'PASS|SKIP_VERIFY'/" "$B/inputs.yaml"
audit chk_rx
[ "$rc" = 1 ] && has chk_rx HARD CHECK_NOT_GOLDEN && ok "5b: a registry pass regex that matches SKIP_VERIFY -> HARD" || bad "5b: rc=$rc"
reset
sed -i "s/run_env: {HPCPERF_GOLDEN_DUMP: '{run_dir}\/golden-capture'}/run_env: {HPCPERF_GOLDEN_DUMP: '{run_dir}\/elsewhere'}/" "$B/inputs.yaml"
audit chk_dump
[ "$rc" = 1 ] && has chk_dump HARD CHECK_NOT_GOLDEN && ok "5c: run_env captures into another directory than --dump -> HARD" || bad "5c: rc=$rc"
reset
"$PY" - "$B/inputs.yaml" <<'PY'
import sys, yaml
p = sys.argv[1]; d = yaml.safe_load(open(p)); d["inputs"][1]["check"] = {"kind": "none", "reason": "fake"}
yaml.safe_dump(d, open(p, "w"), sort_keys=False)
PY
audit chk_none
[ "$rc" = 1 ] && has chk_none HARD CHECK_MISSING n2000 && ok "5d: a golden input whose own check is kind none -> HARD CHECK_MISSING" || bad "5d: rc=$rc"
reset

echo
echo "=== 6: tolerance budget, printed resolution, validate.sh (static)"
hardreset
touch "$BD/fake"
FAKE_CPU_DELTA=6e-9 "$PY" "$G" make "$B" n1000 --replace --reason "test: d0 = 0.6 tol" --workdir "$TMP/mk-b" > "$TMP/make-b.log" 2>&1 \
    && GIT commit -qam "fake: d0 0.6 tol" || bad "6-setup: make --replace"
audit budget
has budget REVIEW BUDGET_CONSUMED n1000 && ok "6a: d0 = 0.6 tol -> REVIEW BUDGET_CONSUMED" || bad "6a"
hardreset
"$PY" - "$TOOLS" <<'PY' && ok "6b: printed_quanta: %g (trailing zeros stripped), %e and fixed-point formats" || bad "6b: printed_quanta"
import sys; sys.path.insert(0, sys.argv[1]); import hpcperf_golden_audit as a
q = {s: u for s, _, u in a.printed_quanta(["34.48", "-12.9274", "0.00479705", "-100.021"])}
assert abs(q["34.48"] - 1e-4) < 1e-12 and abs(q["-12.9274"] - 1e-4) < 1e-12 and abs(q["0.00479705"] - 1e-8) < 1e-15, q
assert abs(q["-100.021"] - 1e-3) < 1e-12, q
q = {s: u for s, _, u in a.printed_quanta(["8.581906e+07", "1.278669e+08"])}
assert abs(q["8.581906e+07"] - 10.0) < 1e-9 and abs(q["1.278669e+08"] - 100.0) < 1e-6, q
q = {s: u for s, _, u in a.printed_quanta(["1.50", "20.25", "0.10"])}
assert all(abs(u - 0.01) < 1e-15 for u in q.values()), q
PY
"$PY" - "$TOOLS" <<'PY' && ok "6c: pass_regex_quality: skip markers HARD, bare words WARN, the golden line specific" || bad "6c: pass_regex_quality"
import sys; sys.path.insert(0, sys.argv[1]); import hpcperf_golden_audit as a
assert a.pass_regex_quality("^PASS: golden check", golden=True) == (None, None)
assert a.pass_regex_quality("SKIP_VERIFY|PASS")[0] == "HARD"
assert a.pass_regex_quality("PASS")[0] == "WARN" and a.pass_regex_quality("Passed")[0] == "WARN"
assert a.pass_regex_quality(None)[0] == "WARN"
assert a.pass_regex_quality("^PASS: residual")[0] is None
assert a.pass_regex_quality("golden", golden=True)[0] == "HARD"          # also matches the FAIL line
assert a.pass_regex_quality("^FAIL", golden=True)[0] == "HARD"           # never matches the PASS line
PY
"$PY" - "$B/inputs.yaml" <<'PY'
import sys, yaml
p = sys.argv[1]; d = yaml.safe_load(open(p))
d["inputs"].append({"id": "n1000-k1p", "case": "fake", "variant": "parameter", "source": {"kind": "custom", "derivation": "k + 1e-14"},
                    "params": {"n": 1000, "repeat": 1, "k": 0.10000000000001}, "args": ["1000", "0.10000000000001"],
                    "runtime_config": {"gpus": 1}, "build_config": None, "backends_validated": ["cuda"]})
yaml.safe_dump(d, open(p, "w"), sort_keys=False)
PY
GIT commit -qam "fake: an input that differs by 1e-14 in k" && touch "$BD/fake" && "$PY" "$G" make "$B" n1000-k1p --workdir "$TMP/mk-p" > "$TMP/make-p.log" 2>&1 \
    && GIT add level1/fake/reference && GIT commit -qm "fake: n1000-k1p reference" || bad "6-setup: pairwise input"
audit pair
has pair WARN PAIRWISE_INDISTINGUISHABLE n1000 && has pair WARN PAIRWISE_IDENTICAL && ok "6e: an input whose golden passes another input's check -> WARN PAIRWISE_INDISTINGUISHABLE / IDENTICAL" \
    || bad "6e: pairwise"
audit nopair --no-pairwise
! has nopair WARN PAIRWISE_INDISTINGUISHABLE && ok "6f: --no-pairwise skips it" || bad "6f"
hardreset
audit l2 --level 2
"$PY" - "$TMP/l2.json" <<'PY' && ok "6d: validate.sh: only the ungated CPU run is a default-path candidate; the gated ones are offline" || bad "6d: validate.sh scan"
import json, sys
r = json.load(open(sys.argv[1])); v = next(b for b in r["benchmarks"] if b["benchmark"] == "app")["validate_sh"]
lines = [d["line"] for d in v["default_path_cpu_candidates"]]
assert v["cpu_reference_gate"] and "CPU_REF" in v["gate_vars"], v
assert lines == [14], v                              # bash run.sh --arch Sequential (ungated)
assert v["offline_cpu_lines"] == 3, v                # build.sh Serial, run.sh Serial, [ CPU_REF = 1 ] && run.sh OpenMP
PY

echo
echo "golden audit tests: $pass passed, $fail failed"
[ "$fail" = 0 ]
