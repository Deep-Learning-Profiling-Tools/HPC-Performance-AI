#!/usr/bin/env bash
# tools/golden/tests/run_all.sh -- CPU-only tests of the golden-reference tooling (no GPU, no real benchmark).
#
# A fake repository with one fake Level 1 benchmark (a Python "binary" that captures its outputs through
# hpcperf_golden_capture.py) exercises make / check / run and their refusals, and the C header is compiled and
# checked against hashlib. Exit 0 when every test passes.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$TOOLS/../.." && pwd)"
G="$TOOLS/hpcperf_golden.py"
PY="${PYTHON:-python3}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/golden-tests.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
ok()  { echo "ok   $*"; pass=$((pass+1)); }
bad() { echo "FAIL $*"; fail=$((fail+1)); }
CC="${CC_FOR_TESTS:-/usr/bin/gcc}"; CXX="${CXX_FOR_TESTS:-/usr/bin/g++}"

echo "=== 1: hpcperf_golden.h (C99 and C++), SHA-256 against hashlib, the CPU-reference gate"
if "$CC" -std=c99 -Wall -Wextra -Werror -O2 -I"$TOOLS" "$HERE/golden_capture_test.c" -o "$TMP/gct" 2>"$TMP/cc.log" &&
   "$CXX" -std=c++20 -Wall -Wextra -Werror -x c++ -fsyntax-only -I"$TOOLS" "$HERE/golden_capture_test.c" 2>>"$TMP/cc.log"; then
    ok "1a: the header compiles warning-free as C99 and C++20"
else
    bad "1a: compile: $(head -5 "$TMP/cc.log")"
fi
allok=1
for n in 0 1 55 56 63 64 65 4096 100003; do
    head -c "$n" /dev/urandom > "$TMP/in.bin"; rm -rf "$TMP/d"
    HPCPERF_GOLDEN_DUMP="$TMP/d" "$TMP/gct" 13 < "$TMP/in.bin" > /dev/null
    "$PY" - "$TMP/in.bin" "$TMP/d" <<'PY' || allok=0
import sys, json, hashlib
data = open(sys.argv[1], 'rb').read(); m = [json.loads(l) for l in open(sys.argv[2] + '/manifest.jsonl')]
w = hashlib.sha256(data).hexdigest(); w4 = hashlib.sha256(data[:len(data) // 4 * 4]).hexdigest()
assert [r['name'] for r in m] == ['blob.streamed', 'blob.whole', 'blob.f32']
assert m[0]['sha256'] == w and m[1]['sha256'] == w and m[2]['sha256'] == w4 and m[2]['count'] == len(data) // 4
assert open(sys.argv[2] + '/blob.streamed.bin', 'rb').read() == data
PY
done
[ "$allok" = 1 ] && ok "1b: streamed / whole / f32 captures match hashlib sha256 and the bytes at 9 lengths" || bad "1b: sha256 / bytes mismatch"
rm -rf "$TMP/d"; HPCPERF_GOLDEN_DUMP="$TMP/d" HPCPERF_GOLDEN_MODE=hash "$TMP/gct" < "$TMP/in.bin" > /dev/null
[ -f "$TMP/d/manifest.jsonl" ] && ! ls "$TMP/d"/*.bin >/dev/null 2>&1 && /usr/bin/grep -q '"data": false' "$TMP/d/manifest.jsonl" \
    && ok "1c: HPCPERF_GOLDEN_MODE=hash records the sha256 only" || bad "1c: hash mode wrote data"
rm -rf "$TMP/d"; "$TMP/gct" < "$TMP/in.bin" > "$TMP/o"; [ ! -e "$TMP/d" ] && /usr/bin/grep -q 'cpu_reference_enabled=0' "$TMP/o" \
    && ok "1d: without HPCPERF_GOLDEN_DUMP nothing is captured; the CPU reference is off by default" || bad "1d: default not inert"
g() { env -u HPCPERF_SKIP_VERIFY -u HPCPERF_CPU_REFERENCE "$@" "$TMP/gct" < /dev/null | sed -n 's/cpu_reference_enabled=//p'; }
r="$(g HPCPERF_CPU_REFERENCE=1)$(g HPCPERF_CPU_REFERENCE=0)$(g HPCPERF_CPU_REFERENCE=)$(g HPCPERF_CPU_REFERENCE=1 HPCPERF_SKIP_VERIFY=1)$(g HPCPERF_SKIP_VERIFY=0)"
[ "$r" = "10000" ] && ok "1e: gate: =1 on; =0 / empty / SKIP_VERIFY=1 / unset off" || bad "1e: gate truth table $r (want 10000)"
HPCPERF_GOLDEN_DUMP="$TMP/nonexistent/sub/dir" "$TMP/gct" < "$TMP/in.bin" > /dev/null 2>&1; rc=$?
[ "$rc" = 97 ] && ok "1f: an unwritable capture directory ends the run with exit 97 (never a partial capture)" || bad "1f: rc=$rc"

echo
echo "=== 2: a fake repository and benchmark"
FR="$TMP/repo"; B="$FR/level1/fake"
mkdir -p "$B/cuda" "$FR/build/fake/cuda"
: > "$FR/hpcperf_env.sh"
echo "// source" > "$B/cuda/main.cu"
cat > "$FR/build/fake/cuda/fake" <<PYBIN
#!$(command -v "$PY")
# fake benchmark: y[i] = sin(i*k) + bias (f64), idx[i] = i*k (i32); CPU reference y.cpu = y + 1e-9 (f64)
import os, sys
sys.path.insert(0, "$TOOLS")
import numpy as np, hpcperf_golden_capture as gc
n, k = int(sys.argv[1]), float(sys.argv[2])
y = np.sin(np.arange(n) * k) + float(os.environ.get("FAKE_BIAS", "0"))
if os.environ.get("FAKE_NAN"):
    y[3] = np.nan
idx = (np.arange(n) * int(k * 10)).astype(np.int32)
if os.environ.get("FAKE_IDX"):
    idx[0] += 1
img = np.linspace(0.0, 200.0, n) + float(os.environ.get("FAKE_IMG", "0"))
gc.capture("y", y, "f64"); gc.capture("idx", idx, "i32"); gc.capture("img", img, "f32")
if os.environ.get("HPCPERF_CPU_REFERENCE") == "1":
    print("CPU_REFERENCE_RAN")
    gc.capture("y.cpu", np.sin(np.arange(n) * k) + 1e-9, "f64")
    gc.capture("img.cpu", np.linspace(0.0, 200.0, n), "f32")
    print("FAIL" if os.environ.get("FAKE_CPU_FAIL") else "PASS")
else:
    print("SKIP_VERIFY")
print("Average kernel execution time 1.0 (ms)")
PYBIN
chmod +x "$FR/build/fake/cuda/fake"
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
  - {name: img, dtype: f32, rule: {metric: trunc_int, tol: 3, frac_tol: 0.001}, cpu: img.cpu, basis: fake trunc_int}
check:
  kind: post_run
  run_env: {HPCPERF_GOLDEN_DUMP: '{run_dir}/golden-capture'}
  command: [python3, '{repo}/../golden-tool/hpcperf_golden.py', check, '{bench_dir}', '{input_id}', --dump, '{run_dir}/golden-capture']
  pass_regex: '^PASS: golden check'
  fail_regex: '^FAIL: golden check'
  basis: fake golden check
  covers: all
coverage: {status: SINGLE_INPUT, reason: fake}
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
YAML
# the check block calls the real tool through a stable path next to the fake repository
ln -s "$TOOLS" "$TMP/golden-tool"
( cd "$FR" && git init -q && git -c user.email=t@t -c user.name=t add -A && git -c user.email=t@t -c user.name=t commit -qm fake ) \
    && ok "2a: fake repository committed" || bad "2a: git"
export HPCPERF_GOLDEN_STORE="$TMP/store"
"$PY" "$G" validate "$B" | /usr/bin/grep -q 'fake: OK' && ok "2b: validate accepts the spec" || bad "2b: validate"

echo
echo "=== 3: make"
echo "// dirty" >> "$B/cuda/main.cu"; touch -d '2 hours ago' "$B/cuda/main.cu"
out="$("$PY" "$G" make "$B" n1000 --workdir "$TMP/mk" 2>&1)"; rc=$?
[ "$rc" = 2 ] && echo "$out" | /usr/bin/grep -q 'uncommitted changes' && [ ! -e "$B/reference/n1000.json" ] \
    && ok "3a: make refuses uncommitted sources" || bad "3a: rc=$rc $out"
( cd "$FR" && git checkout -q level1/fake/cuda/main.cu )
touch "$B/cuda/main.cu"; touch -d '1 hour ago' "$FR/build/fake/cuda/fake"
out="$("$PY" "$G" make "$B" n1000 --workdir "$TMP/mk" 2>&1)"; rc=$?
[ "$rc" = 2 ] && echo "$out" | /usr/bin/grep -q 'older than' && ok "3b: make refuses a binary older than its sources" || bad "3b: rc=$rc $out"
touch "$FR/build/fake/cuda/fake"
out="$(FAKE_CPU_FAIL=1 "$PY" "$G" make "$B" n1000 --workdir "$TMP/mk" 2>&1)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q 'did not pass' && [ ! -e "$B/reference/n1000.json" ] \
    && ok "3c: a failing one-time CPU validation freezes nothing" || bad "3c: rc=$rc $out"
out="$("$PY" "$G" make "$B" n1000 --workdir "$TMP/mk" 2>&1)"; rc=$?
[ "$rc" = 0 ] && [ -f "$B/reference/n1000.json" ] && ok "3d: make freezes the reference" || bad "3d: rc=$rc $out"
out="$("$PY" "$G" run "$B" --workdir "$TMP/rn0" 2>&1)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q 'is untracked' && ok "3h: a reference that is not committed is not trusted (FAIL)" || bad "3h: rc=$rc $out"
out="$(HPCPERF_GOLDEN_ALLOW_UNCOMMITTED_REFERENCE=1 "$PY" "$G" run "$B" --workdir "$TMP/rn0" --json "$TMP/rn0.json" 2>&1)"; rc=$?
[ "$rc" = 0 ] && echo "$out" | /usr/bin/grep -q 'WARNING' && "$PY" -c "import json,sys;sys.exit(0 if json.load(open('$TMP/rn0.json'))['reference_committed'] is False else 1)" \
    && ok "3i: the test-only override passes but records reference_committed false" || bad "3i: rc=$rc"
( cd "$FR" && git -c user.email=t@t -c user.name=t add level1/fake/reference && git -c user.email=t@t -c user.name=t commit -qm ref1 )
"$PY" - "$B/reference/n1000.json" "$TMP/store" <<'PY' && ok "3e: d0, tol_effective, exact sha, trunc_int budget and the blob are recorded" || bad "3e: reference content"
import json, sys, os
r = json.load(open(sys.argv[1])); o = {e['name']: e for e in r['outputs']}
y = o['y']['rule']; assert abs(y['d0_vs_cpu_reference'] - 1e-9) < 1e-12 and abs(y['tol_effective'] - (1e-8 - 1e-9)) < 1e-12, y
assert o['idx']['rule']['metric'] == 'exact' and 'blob' not in o['idx']
im = o['img']['rule']; assert im['d0_vs_cpu_reference'] == 0 and im['tol_effective'] == 3 and im['frac_tol_effective'] == 0.001, im
assert os.path.isfile(os.path.join(sys.argv[2], o['y']['blob']))
p = r['provenance']; assert p['cpu_validation']['pass_line'] == 'PASS' and p['source_clean'] is True and p['env']['HPCPERF_CPU_REFERENCE'] == '1'
assert r['workload']['args'] == ['1000', '0.1']
PY
out="$("$PY" "$G" make "$B" n1000 --workdir "$TMP/mk" 2>&1)"; rc=$?
[ "$rc" = 2 ] && echo "$out" | /usr/bin/grep -q 'exists (frozen)' && ok "3f: an existing reference is never replaced silently" || bad "3f: rc=$rc"
out="$("$PY" "$G" make "$B" n1000 --replace --workdir "$TMP/mk" 2>&1)"; rc=$?
[ "$rc" = 2 ] && echo "$out" | /usr/bin/grep -q 'needs --reason' && ok "3g: --replace without --reason is refused" || bad "3g: rc=$rc"

echo
echo "=== 4: check / run"
chk() { "$PY" "$G" run "$B" --workdir "$TMP/rn" 2>&1; }
out="$(chk)"; rc=$?
[ "$rc" = 0 ] && echo "$out" | /usr/bin/grep -q '^PASS: golden check fake/n1000' && ! /usr/bin/grep -q CPU_REFERENCE_RAN "$TMP/rn/run/stdout.log" \
    && ok "4a: run: a plain run (no CPU reference) matches the frozen reference" || bad "4a: rc=$rc $out"
out="$(HPCPERF_CPU_REFERENCE=1 chk)"; rc=$?
[ "$rc" = 0 ] && ! /usr/bin/grep -q CPU_REFERENCE_RAN "$TMP/rn/run/stdout.log" \
    && ok "4b: run never passes the caller's HPCPERF_CPU_REFERENCE to the benchmark" || bad "4b: CPU reference ran under run"
out="$(FAKE_BIAS=5e-9 chk)"; rc=$?
[ "$rc" = 0 ] && ok "4c: a deviation inside tol - d0 passes" || bad "4c: rc=$rc $out"
out="$(FAKE_BIAS=9.5e-9 chk)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q 'FAIL y: max_abs' \
    && ok "4d: a deviation inside the upstream tol but beyond tol - d0 fails (the CPU budget is preserved)" || bad "4d: rc=$rc $out"
out="$(FAKE_NAN=1 chk)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q 'FAIL y' && ok "4e: a NaN where the golden is finite fails" || bad "4e: rc=$rc"
out="$(FAKE_IDX=1 chk)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q 'FAIL idx: exact' && ok "4f: one changed integer fails the exact rule" || bad "4f: rc=$rc"
out="$(FAKE_IMG=0.5 chk)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q 'FAIL img: trunc_int' && ok "4g: trunc_int: >0.1% of pixels one level off fails" || bad "4g: rc=$rc $out"
out="$(HPCPERF_GOLDEN_STORE="$TMP/empty-store" chk)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q 'golden blob .* missing' && ok "4h: a missing golden blob fails (no fallback)" || bad "4h: rc=$rc $out"
bp="$("$PY" -c "import json;print(json.load(open('$B/reference/n1000.json'))['outputs'][0]['blob'])")"
cp "$TMP/store/$bp" "$TMP/blob.keep"; printf 'X' | dd of="$TMP/store/$bp" bs=1 seek=8 conv=notrunc 2>/dev/null
out="$(chk)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q 'does not match its sha256' && ok "4i: a corrupted golden blob fails" || bad "4i: rc=$rc $out"
cp "$TMP/blob.keep" "$TMP/store/$bp"
cp -r "$TMP/rn/capture" "$TMP/cap2"; printf 'X' | dd of="$TMP/cap2/y.bin" bs=1 seek=0 conv=notrunc 2>/dev/null
out="$("$PY" "$G" check "$B" n1000 --dump "$TMP/cap2" 2>&1)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q 'does not match its recorded sha256' && ok "4j: a capture altered after the run fails" || bad "4j: rc=$rc $out"
rm -rf "$TMP/cap3"; mkdir -p "$TMP/cap3"; out="$("$PY" "$G" check "$B" n1000 --dump "$TMP/cap3" 2>&1)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q 'no capture manifest' && ok "4k: a run that captured nothing fails" || bad "4k: rc=$rc"

echo
echo "=== 5: frozen spec and workload"
cp "$B/inputs.yaml" "$TMP/inputs.keep"
sed -i 's/tol: 1.0e-8/tol: 1.0e-6/' "$B/inputs.yaml"
out="$(chk)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q 'differs from the one frozen' && ok "5a: a tolerance edited after freezing is refused until an explicit make --replace" || bad "5a: rc=$rc $out"
cp "$TMP/inputs.keep" "$B/inputs.yaml"
sed -i "s/args: \['1000', '0.1'\]/args: ['1000', '0.2']/" "$B/inputs.yaml"
out="$(chk)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q 'another workload' && ok "5b: a reference of another workload is refused" || bad "5b: rc=$rc $out"
cp "$TMP/inputs.keep" "$B/inputs.yaml"
out="$("$PY" "$G" status "$B" 2>&1)"; rc=$?
[ "$rc" = 0 ] && echo "$out" | /usr/bin/grep -q 'fake/n1000: OK' && ok "5c: status OK" || bad "5c: rc=$rc $out"
( cd "$FR" && git -c user.email=t@t -c user.name=t add -A && git -c user.email=t@t -c user.name=t commit -qm ref )
touch "$FR/build/fake/cuda/fake"
out="$("$PY" "$G" make "$B" n1000 --replace --reason "test" --workdir "$TMP/mk" 2>&1)"; rc=$?
[ "$rc" = 0 ] && "$PY" -c "import json,sys;r=json.load(open('$B/reference/n1000.json'));sys.exit(0 if r['replaces']['reason']=='test' and r['replaces']['previous_sha256'] else 1)" \
    && ok "5d: make --replace --reason records what it replaced" || bad "5d: rc=$rc $out"
out="$("$PY" "$G" run "$B" --workdir "$TMP/rn5" 2>&1)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q 'modified relative to HEAD' && ok "5e: a replaced but uncommitted reference is not trusted until committed" || bad "5e: rc=$rc $out"
touch "$FR/build/fake/cuda/fake"
out="$("$PY" "$G" make "$B" n1000 --replace --reason "test again" --workdir "$TMP/mk2" 2>&1)"; rc=$?
[ "$rc" = 0 ] && ! echo "$out" | /usr/bin/grep -q 'uncommitted changes' \
    && ok "5f: a replaced (modified, uncommitted) reference does not make the next make refuse (reference/ is excluded from the dirty check)" || bad "5f: rc=$rc $out"
( cd "$FR" && git -c user.email=t@t -c user.name=t add level1/fake/reference && git -c user.email=t@t -c user.name=t commit -qm ref2 )

echo
echo "=== 6: spec validation"
"$PY" - "$TOOLS" <<'PY' && ok "6a: invalid specs are rejected (hash + tolerance, integer tolerance, no basis, match without count, no cpu_check)" || bad "6a"
import sys; sys.path.insert(0, sys.argv[1]); import hpcperf_golden as g
base = lambda o, **kw: dict({"mode": "data", "cpu_check": {"pass_regex": "PASS"}, "outputs": [o]}, **kw)
o = {"name": "y", "dtype": "f64", "rule": {"metric": "max_abs", "tol": 1e-8}, "basis": "b"}
assert not g.validate_spec(base(o))
assert g.validate_spec(base(o, mode="hash"))
assert g.validate_spec(base(dict(o, dtype="i32")))
assert g.validate_spec(base(dict(o, basis="")))
assert g.validate_spec(base({"match": "y.*", "dtype": "f64", "rule": {"metric": "exact"}, "basis": "b"}))
assert g.validate_spec({"outputs": [o]})
assert g.validate_spec(base(dict(o, rule={"metric": "trunc_int", "tol": 3})))
assert not g.validate_spec({"unsupported": "multiple valid colorings"})
PY

echo
echo "=== 7: tools/inputs integration (post_run check, CPU-reference switch never inherited)"
out="$(HPCPERF_CPU_REFERENCE=1 HPCPERF_GOLDEN_MODE=hash "$PY" "$REPO/tools/inputs/hpcperf_inputs.py" check "$B" n1000 --out "$TMP/ck" 2>&1)"; rc=$?
v="$("$PY" -c "import json;print(json.load(open('$TMP/ck/check.json'))['verdict'])" 2>/dev/null)"
[ "$rc" = 0 ] && [ "$v" = PASS ] && ! /usr/bin/grep -q CPU_REFERENCE_RAN "$TMP/ck/run/stdout.log" \
    && ok "7a: hpcperf_inputs.py check: PASS through the golden check; the caller's HPCPERF_CPU_REFERENCE / GOLDEN_MODE are dropped" \
    || bad "7a: rc=$rc verdict=$v $out"
out="$(HPCPERF_CPU_REFERENCE=1 "$PY" "$REPO/tools/inputs/hpcperf_inputs.py" measure "$B" n1000 --out "$TMP/ms" --warmup 0 --reps 1 2>&1)"; rc=$?
! /usr/bin/grep -q CPU_REFERENCE_RAN "$TMP/ms/rep1/stdout.log" && /usr/bin/grep -q SKIP_VERIFY "$TMP/ms/rep1/stdout.log" && [ ! -e "$TMP/ms/rep1/golden-capture" ] \
    && ok "7b: hpcperf_inputs.py measure never runs the CPU reference nor captures" || bad "7b: rc=$rc $out"

echo
echo "=== 8: log-sourced outputs, after_run CPU validation, max_relabs (the Level 2 pattern)"
L="$FR/level2/fakelog"; mkdir -p "$L" "$FR/build/fakelog"
cat > "$FR/build/fakelog/app" <<PYBIN
#!$(command -v "$PY")
# prints 'Step# n Stress: a b c' for 5 steps; FAKE_SHIFT perturbs the GPU values
import os
s = float(os.environ.get("FAKE_SHIFT", "0"))
for n in range(1, 6):
    print("Step# %d Stress: %.6e %.6e %.6e" % (n, 100.0 * n + s, -3.0 * n, 0.5))
print("done")
PYBIN
cat > "$FR/build/fakelog/cpu_check.sh" <<'SH'
#!/bin/sh
# the one-time CPU validation: writes the "CPU" values (1e-7 relative off) and compares them with the GPU log
gpu_log="$1"; ref="$2"
[ -n "$HPCPERF_CPU_REFERENCE" ] || { echo "cpu_check: not an explicit CPU-reference run"; echo "CHECK FAIL"; exit 1; }
awk 'BEGIN { for (n = 1; n <= 5; n++) printf("Step# %d Stress: %.6e %.6e %.6e\n", n, 100 * n * (1 + 1e-7), -3 * n, 0.5) }' > "$ref"
grep -c '^Step#' "$gpu_log" | grep -q '^5$' && echo "CHECK PASS" || { echo "CHECK FAIL"; exit 1; }
SH
chmod +x "$FR/build/fakelog/app" "$FR/build/fakelog/cpu_check.sh"
cat > "$L/inputs.yaml" <<'YAML'
schema: hpcperf-inputs-1
benchmark: fakelog
level: 2
selector: null
default_input: d5
entry: {kind: binary, path: build/fakelog/app}
timing: {kind: none, status: NEEDS_TIMING_SUPPORT, reason: fake}
baseline: {method: fake, reference: fake, quantities: [{name: done, role: diagnostic, regex: '^done$', compare: {rule: present}}]}
golden:
  cpu_check:
    after_run: true
    command: ['{repo}/build/fakelog/cpu_check.sh', '{log}', '{run_dir}/ref.out']
    pass_regex: '^CHECK PASS'
    fail_regex: '^CHECK FAIL'
  outputs:
  - name: stress
    dtype: f64
    log: {regex: '^Step# \d+ Stress: (\S+) (\S+) (\S+)'}
    rule: {metric: max_relabs, rtol: 1.0e-5, atol: 1.0e-6, tol: 1}
    cpu: {log: {file: '{run_dir}/ref.out'}}
    basis: fake |gpu - ref| <= 1e-5 |ref| + 1e-6
check:
  kind: post_run
  command: [python3, '{repo}/../golden-tool/hpcperf_golden.py', check, '{bench_dir}', '{input_id}', --dump, '{run_dir}/golden-capture', --log, '{log}']
  pass_regex: '^PASS: golden check'
  fail_regex: '^FAIL: golden check'
  basis: fake
  covers: all
coverage: {status: SINGLE_INPUT, reason: fake}
inputs:
- {id: d5, case: fake, variant: default, source: {kind: custom, derivation: fake}, params: {}, args: [], runtime_config: {gpus: 1}, build_config: null, backends_validated: [cuda]}
YAML
( cd "$FR" && git -c user.email=t@t -c user.name=t add -A && git -c user.email=t@t -c user.name=t commit -qm fakelog )
touch "$FR/build/fakelog/app"
out="$("$PY" "$G" make "$L" d5 --workdir "$TMP/mkl" 2>&1)"; rc=$?
( cd "$FR" && git -c user.email=t@t -c user.name=t add level2/fakelog/reference && git -c user.email=t@t -c user.name=t commit -qm ref3 )
"$PY" - "$L/reference/d5.json" <<'PY' && [ "$rc" = 0 ] && ok "8a: make: GPU run, then the after_run CPU validation; 15 log values frozen, d0 from the CPU log (~0.01 of the normalised tolerance)" || bad "8a: rc=$rc $out"
import json, sys
r = json.load(open(sys.argv[1])); e = r['outputs'][0]
assert e['count'] == 15 and e['rule']['metric'] == 'max_relabs', e
d0 = e['rule']['d0_vs_cpu_reference']; assert 0.005 < d0 < 0.02 and abs(e['rule']['tol_effective'] - (1 - d0) / (1 + 1e-5 * d0)) < 1e-12, d0
assert r['provenance']['gpu_run']['log'].endswith('gpu_run.log')
PY
out="$("$PY" "$REPO/tools/inputs/hpcperf_inputs.py" check "$L" d5 --out "$TMP/ckl" 2>&1)"; rc=$?
[ "$rc" = 0 ] && ok "8b: registry check reads the values from the run's log and passes" || bad "8b: rc=$rc $out"
out="$(FAKE_SHIFT=0.005 "$PY" "$REPO/tools/inputs/hpcperf_inputs.py" check "$L" d5 --out "$TMP/ckl" 2>&1)"; rc=$?
[ "$rc" = 1 ] && /usr/bin/grep -q 'FAIL stress: max_relabs' "$TMP/ckl/check.log" && ok "8c: a shift beyond rtol |g| + atol fails" || bad "8c: rc=$rc"
out="$(FAKE_SHIFT=0.0002 "$PY" "$REPO/tools/inputs/hpcperf_inputs.py" check "$L" d5 --out "$TMP/ckl" 2>&1)"; rc=$?
[ "$rc" = 0 ] && ok "8d: a shift inside the budget passes" || bad "8d: rc=$rc $(tail -3 "$TMP/ckl/check.log")"

echo
echo "=== 9: the correctness gate: canonical directory only, machine-readable verdict, both SHA-256 paths"
cp -r "$B" "$FR/level1/fake_copy"
out="$("$PY" "$G" run "$FR/level1/fake_copy" --workdir "$TMP/rc" 2>&1)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q 'not the canonical benchmark directory' \
    && ok "9a: a copy of the benchmark (e.g. a candidate workspace) never supplies the registry or the reference" || bad "9a: rc=$rc $out"
out="$("$PY" "$G" make "$FR/level1/fake_copy" n1000 --replace --reason x --workdir "$TMP/mkc" 2>&1)"; rc=$?
[ "$rc" = 2 ] && echo "$out" | /usr/bin/grep -q 'not the canonical benchmark directory' \
    && ok "9b: make refuses a copy (a candidate never makes a golden reference)" || bad "9b: rc=$rc $out"
rm -rf "$FR/level1/fake_copy"
"$PY" "$G" run "$B" --workdir "$TMP/rj" --json "$TMP/rj.json" >/dev/null 2>&1
FAKE_BIAS=9.5e-9 "$PY" "$G" run "$B" --workdir "$TMP/rj2" --json "$TMP/rj2.json" >/dev/null 2>&1
"$PY" - "$TMP/rj.json" "$TMP/rj2.json" "$FR/build/fake/cuda/fake" <<'PY' && ok "9c: --json: verdict, per-output value/limit, reference and binary sha256 (PASS and FAIL)" || bad "9c: json"
import json, sys, hashlib
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
sha = hashlib.sha256(open(sys.argv[3], 'rb').read()).hexdigest()
assert a["verdict"] == "PASS" and a["run"]["binary_sha256"] == sha and a["reference_sha256"] and a["run"]["cpu_reference"] is False
assert b["verdict"] == "FAIL" and any(o["name"] == "y" and not o["ok"] and o["value"] > o["limit"] for o in b["outputs"]), b
PY
head -c 5000017 /dev/urandom > "$TMP/r.bin"
rm -rf "$TMP/s1" "$TMP/s2"
HPCPERF_GOLDEN_DUMP="$TMP/s1" "$TMP/gct" 4099 < "$TMP/r.bin" > /dev/null
HPCPERF_GOLDEN_DUMP="$TMP/s2" HPCPERF_GOLDEN_PORTABLE_SHA=1 "$TMP/gct" 4099 < "$TMP/r.bin" > /dev/null
[ "$(cut -c1-400 "$TMP/s1/manifest.jsonl")" = "$(cut -c1-400 "$TMP/s2/manifest.jsonl")" ] && /usr/bin/grep -q "$(sha256sum "$TMP/r.bin" | cut -c1-64)" "$TMP/s1/manifest.jsonl" \
    && ok "9d: the libcrypto and the portable SHA-256 give the same digests (= sha256sum)" || bad "9d: digests differ"

# the distribution tool (golden_artifacts.py) has its own CPU-only suite on fixture repositories
if bash "$(dirname "${BASH_SOURCE[0]}")/test_golden_artifacts.sh" > "$TMP/artifacts.log" 2>&1; then
    ok "10a: golden_artifacts.py suite ($(grep -o '[0-9]* passed' "$TMP/artifacts.log" | tail -1))"
else
    bad "10a: golden_artifacts.py suite: $(grep -E '^FAIL|failed' "$TMP/artifacts.log" | head -3)"
fi
# the read-only consistency audit (hpcperf_golden_audit.py) has its own CPU-only suite on a fake repository
if bash "$(dirname "${BASH_SOURCE[0]}")/test_golden_audit.sh" > "$TMP/audit.log" 2>&1; then
    ok "10b: hpcperf_golden_audit.py suite ($(grep -o '[0-9]* passed' "$TMP/audit.log" | tail -1))"
else
    bad "10b: hpcperf_golden_audit.py suite: $(grep -E '^FAIL|failed' "$TMP/audit.log" | head -3)"
fi

echo
echo "golden tests: $pass passed, $fail failed"
[ "$fail" = 0 ]
