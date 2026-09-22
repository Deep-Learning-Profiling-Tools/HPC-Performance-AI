#!/usr/bin/env bash
# Self-tests for tools/timing. CPU only: no GPU, no build tree, no nsys required.
# Groups that need something absent (a build tree, nsys) SKIP instead of failing.
#
#   bash tools/timing/tests/run_all.sh
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$TOOLS/../.." && pwd)"
# exported because the python heredocs below read them from the environment
export TOOLS REPO

pass=0; failn=0; skipn=0
ok()   { echo "ok   $*"; pass=$((pass+1)); }
bad()  { echo "FAIL $*"; failn=$((failn+1)); }
skip() { echo "skip $*"; skipn=$((skipn+1)); }
# the login shell on this site prints lmod noise on every subshell
noise() { /usr/bin/grep -viE 'lua|posix|traceback|no file|no field|in function|main chunk|in \?'; }

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

echo "=== 1: case table integrity"
CASES="$TOOLS/cases.tsv"
if [ ! -f "$CASES" ]; then
    bad "1a: cases.tsv missing"
else
    ncol_bad=0
    while IFS= read -r line; do
        case "$line" in ''|\#*) continue ;; esac
        n=$(awk -F'\t' '{print NF}' <<< "$line")
        [ "$n" -eq 7 ] || ncol_bad=$((ncol_bad+1))
    done < "$CASES"
    [ "$ncol_bad" -eq 0 ] && ok "1a: every row has 7 tab-separated fields" \
                          || bad "1a: $ncol_bad rows with the wrong field count"

    # empty fields must be "-": bash's `IFS=$'\t' read` collapses runs of tabs
    if /usr/bin/grep -qP '\t\t' "$CASES"; then
        bad "1b: cases.tsv contains an empty field (consecutive tabs) -- columns would shift"
    else
        ok "1b: no consecutive tabs (empty fields use the - placeholder)"
    fi

    nrows=$(/usr/bin/grep -cvE '^#|^$' "$CASES")
    [ "$nrows" -ge 40 ] && ok "1c: $nrows cases listed" || bad "1c: only $nrows cases listed"

    # every benchmark directory in level1/ should have a row
    missing=""
    for d in "$REPO"/level1/*/; do
        b="$(basename "$d")"
        [ -f "$d/CMakeLists.txt" ] || continue
        /usr/bin/grep -q "^$b	" "$CASES" || missing="$missing $b"
    done
    [ -z "$missing" ] && ok "1d: every level1 benchmark has a case row" \
                      || bad "1d: no case row for:$missing"

    # the nine verify.py-wrapped rows must record the inner binary, never python
    pyrows=$(awk -F'\t' '$6=="verify.py"' "$CASES" | wc -l)
    pybad=$(awk -F'\t' '$6=="verify.py" && ($2 ~ /python/ || $3 ~ /verify\.py/)' "$CASES" | wc -l)
    [ "$pyrows" -ge 1 ] && { [ "$pybad" -eq 0 ] \
        && ok "1e: $pyrows verify.py rows record the inner binary, not the wrapper" \
        || bad "1e: $pybad verify.py rows still point at python/verify.py"; } \
        || skip "1e: no verify.py rows in the table"
fi

echo
echo "=== 2: measure_level1.sh argument handling and dry run"
M="$TOOLS/measure_level1.sh"
bash -n "$M" && ok "2a: shell syntax" || bad "2a: shell syntax"

out="$(bash "$M" --build-root "$TMP" 2>&1 | noise)"
case "$out" in *"give a benchmark name"*) ok "2b: refuses to run without a case selector" ;;
                                      *) bad "2b: missing selector not refused: $out" ;; esac

out="$(bash "$M" --build-root "$TMP" --repeats x daxpy 2>&1 | noise)"
case "$out" in *"--repeats must be a number"*) ok "2c: rejects a non-numeric --repeats" ;;
                                            *) bad "2c: bad --repeats accepted" ;; esac

out="$(bash "$M" --build-root "$TMP" --bogus daxpy 2>&1 | noise)"
case "$out" in *"unknown option"*) ok "2d: rejects an unknown option" ;;
                                *) bad "2d: unknown option accepted" ;; esac

if [ -d "$REPO/build/gcc13/level1" ] || [ -d "$REPO/build/all/level1" ]; then
    root="$REPO/build/all"; [ -d "$root/level1" ] || root="$REPO/build/gcc13"
    dry="$(bash "$M" --build-root "$root" --dry-run daxpy 2>&1 | noise)"
    case "$dry" in *HPCPERF_SKIP_VERIFY*) ok "2e: dry run passes HPCPERF_SKIP_VERIFY" ;;
                                       *) bad "2e: dry run does not set HPCPERF_SKIP_VERIFY" ;; esac
    case "$dry" in *"env -i"*) ok "2f: dry run uses env -i (nsys records the environment)" ;;
                            *) bad "2f: dry run does not use env -i" ;; esac
    case "$dry" in *ctest*) bad "2g: constructed command contains ctest" ;;
                         *) ok "2g: constructed command does not go through ctest" ;; esac
    case "$dry" in *"/build/timing/"*) ok "2h: nsys report path lands under build/timing" ;;
                                    *) bad "2h: nsys report path is not under build/timing" ;; esac
    dry2="$(bash "$M" --build-root "$root" --dry-run --keep-verify daxpy 2>&1 | noise)"
    case "$dry2" in *HPCPERF_SKIP_VERIFY*) bad "2i: --keep-verify still sets HPCPERF_SKIP_VERIFY" ;;
                                        *) ok "2i: --keep-verify does not set the skip variable" ;; esac
    # the allow-list must not leak a planted credential into the profiled command
    export HPCPERF_TIMING_TOKEN=supersecretvalue CLAUDE_TIMING=alsosecret TIMING_API_KEY=thirdsecret
    dry3="$(bash "$M" --build-root "$root" --dry-run daxpy 2>&1 | noise)"
    unset HPCPERF_TIMING_TOKEN CLAUDE_TIMING TIMING_API_KEY
    if echo "$dry3" | /usr/bin/grep -qE 'supersecretvalue|alsosecret|thirdsecret|HPCPERF_TIMING_TOKEN|CLAUDE_TIMING|TIMING_API_KEY'; then
        bad "2j: a planted credential reached the constructed command"
    else
        ok "2j: planted credentials are not in the constructed command (allow-list holds)"
    fi
else
    skip "2e-2j: no Level 1 build tree"
fi

echo
echo "=== 3: summarize.py -- interval union, CSV parsing, schema"
python3 - <<'PY' && ok "3a-3f: see lines above" || bad "3a-3f: python assertions failed"
import importlib.util, os, sys, csv, json, tempfile
spec = importlib.util.spec_from_file_location("summarize", os.path.join(os.environ["TOOLS"], "summarize.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
fail = []

# the interval union is the one algorithm that is easy to get wrong
cases = [
    ("disjoint",   [(0, 10), (20, 30)],              20.0, 30.0, 20.0),
    ("overlap",    [(0, 10), (5, 15)],               15.0, 15.0, 20.0),
    ("contained",  [(0, 100), (10, 20)],            100.0, 100.0, 110.0),
    ("adjacent",   [(0, 10), (10, 20)],              20.0, 20.0, 20.0),
    ("unsorted",   [(20, 30), (0, 10)],              20.0, 30.0, 20.0),
    ("identical",  [(5, 15), (5, 15)],               10.0, 10.0, 20.0),
    ("single",     [(7, 9)],                          2.0,  2.0,  2.0),
    ("empty",      [],                                0.0,  0.0,  0.0),
]
for name, iv, exp_u, exp_s, exp_n in cases:
    u, s, n = m.merge_intervals(iv)
    if (u, s, n) != (exp_u, exp_s, exp_n):
        fail.append(f"merge_intervals[{name}] = {(u, s, n)}, expected {(exp_u, exp_s, exp_n)}")
print(f"ok   3a: interval union/span/sum correct on {len(cases)} shapes" if not fail else "")

# require_finite must reject NaN and infinity
for bad_v in (float("nan"), float("inf"), float("-inf")):
    try:
        m.finite("t", bad_v); fail.append(f"finite() accepted {bad_v}")
    except ValueError:
        pass
print("ok   3b: finite() rejects NaN and infinity")

# a missing column must raise, not silently become zero
try:
    m.need({"Other": "1"}, "Total Time (ns)", "x.csv"); fail.append("need() accepted a missing column")
except KeyError:
    pass
print("ok   3c: a missing CSV column raises")

# memcpy direction comes from the Operation text, not a column position
d = tempfile.mkdtemp()
with open(os.path.join(d, "rep_cuda_gpu_mem_time_sum.csv"), "w") as f:
    f.write("Time (%),Total Time (ns),Count,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Operation\n"
            "60.0,3000000,2,1500000.0,1.0,1,2,0.0,[CUDA memcpy Host-to-Device]\n"
            "40.0,2000000,1,2000000.0,1.0,1,2,0.0,[CUDA memcpy Device-to-Host]\n"
            "0.0,1000,1,1000.0,1.0,1,2,0.0,[CUDA memset]\n")
with open(os.path.join(d, "rep_cuda_gpu_mem_size_sum.csv"), "w") as f:
    f.write("Total (MB),Count,Avg (MB),Med (MB),Min (MB),Max (MB),StdDev (MB),Operation\n"
            "16.000,2,8.0,8.0,8.0,8.0,0.0,[CUDA memcpy Host-to-Device]\n")
mo = m.parse_memops(d)
if abs(mo["h2d_s"] - 0.003) > 1e-9 or abs(mo["d2h_s"] - 0.002) > 1e-9 or abs(mo["memset_s"] - 1e-6) > 1e-12:
    fail.append(f"parse_memops times wrong: {mo}")
if abs(mo["h2d_mb"] - 16.0) > 1e-9:
    fail.append(f"parse_memops sizes wrong: {mo}")
print("ok   3d: memcpy/memset time and size keyed off the Operation text")

# kernels are sorted by total time, ns -> s
with open(os.path.join(d, "rep_cuda_gpu_kern_sum.csv"), "w") as f:
    f.write("Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n"
            "30.0,1000000,10,100000.0,1.0,1,2,0.0,small\n"
            "70.0,2000000,20,100000.0,1.0,1,2,0.0,big\n")
ks = m.parse_kernels(d)
if [k["name"] for k in ks] != ["big", "small"] or abs(ks[0]["total_s"] - 0.002) > 1e-12:
    fail.append(f"parse_kernels wrong: {ks}")
print("ok   3e: kernels parsed and sorted by total time")

# the aggregate CSV must refuse a foreign schema
if m.SCHEMA != "hpcperf-timing-1":
    fail.append(f"unexpected schema name {m.SCHEMA}")
print("ok   3f: schema name is hpcperf-timing-1")

if fail:
    print("FAIL " + "; ".join(fail), file=sys.stderr)
    sys.exit(1)
PY

echo
echo "=== 4: skip-verify switch is default-off in every patched benchmark"
patched=0; badlog=""
for f in $(/usr/bin/grep -rl 'hpcperf_skip_verify' "$REPO"/level1/*/cuda/ "$REPO"/level1/*/common/ 2>/dev/null); do
    patched=$((patched+1))
    # the variable must be the only trigger: no unconditional skip, no default-on
    if /usr/bin/grep -qE 'HPCPERF_SKIP_VERIFY"?\s*\)\s*==\s*NULL|skip_verify\s*=\s*true' "$f"; then
        badlog="$badlog $f"
    fi
done
[ "$patched" -ge 20 ] && ok "4a: $patched files carry the switch" || bad "4a: only $patched files carry the switch"
[ -z "$badlog" ] && ok "4b: skipping is never enabled by default" || bad "4b: default-on suspicion:$badlog"

# ctest command lines must never carry the variable
if /usr/bin/grep -rn 'HPCPERF_SKIP_VERIFY' "$REPO"/level1/*/CMakeLists.txt >/dev/null 2>&1; then
    bad "4c: a CMakeLists sets HPCPERF_SKIP_VERIFY -- ctest would stop verifying"
else
    ok "4c: no CMakeLists sets HPCPERF_SKIP_VERIFY (ctest still verifies)"
fi

echo
echo "=== 5: gen_cases.py resolves a working ctest, never a broken PATH one"
G="$TOOLS/gen_cases.py"
# a fixture build tree whose CMakeCache.txt names a stub ctest, plus a PATH whose
# ctest is deliberately broken (this is the situation on the reference node, where
# ~/.local/bin/ctest is a pip shim with no cmake module)
FB="$TMP/fakebuild"; mkdir -p "$FB/level1/demo" "$TMP/badbin"
printf '#!/bin/sh\nexit 1\n' > "$TMP/badbin/ctest"; chmod +x "$TMP/badbin/ctest"
cat > "$TMP/stubctest" <<'STUB'
#!/bin/sh
case "$1" in
  --version) echo "ctest version 9.9.9"; exit 0 ;;
esac
echo '{"kind":"ctestInfo","tests":[{"name":"demo_run","command":["/bin/true","7"],
      "properties":[{"name":"WORKING_DIRECTORY","value":"/tmp"},{"name":"TIMEOUT","value":600}]}]}'
STUB
chmod +x "$TMP/stubctest"
echo "CMAKE_CTEST_COMMAND:INTERNAL=$TMP/stubctest" > "$FB/CMakeCache.txt"
touch "$FB/level1/demo/CTestTestfile.cmake"

out="$(PATH="$TMP/badbin:$PATH" python3 "$G" --build-root "$FB" --out "$TMP/out.tsv" 2>&1 | noise)"
case "$out" in *"ctest 9.9.9 at $TMP/stubctest"*) ok "5a: prefers CMAKE_CTEST_COMMAND over a broken PATH ctest" ;;
                                               *) bad "5a: did not use the build tree's ctest: $out" ;; esac
if /usr/bin/grep -q "^demo	/bin/true	7	" "$TMP/out.tsv" 2>/dev/null; then
    ok "5b: parses the stub ctest's json-v1 into a case row"
else
    bad "5b: no demo row written"
fi

# no cache and no working ctest anywhere -> hard error naming all three attempts
rm "$FB/CMakeCache.txt"
PY3="$(command -v python3)"
out="$(PATH="$TMP/badbin:/usr/bin:/bin" "$PY3" "$G" --build-root "$FB" --out "$TMP/out2.tsv" 2>&1 | noise)"
if echo "$out" | /usr/bin/grep -q 'CMAKE_CTEST_COMMAND' && echo "$out" | /usr/bin/grep -q 'HPCPERF_CTEST'; then
    ok "5c: with no usable ctest it fails naming every resolution attempt"
else
    bad "5c: resolution failure is not reported with its trail: $out"
fi
out="$(PATH="$TMP/badbin:/usr/bin:/bin" HPCPERF_CTEST="$TMP/stubctest" "$PY3" "$G" --build-root "$FB" --out "$TMP/out3.tsv" 2>&1 | noise)"
case "$out" in *"$TMP/stubctest"*) ok "5d: \$HPCPERF_CTEST is honoured when the cache is absent" ;;
                                *) bad "5d: HPCPERF_CTEST ignored: $out" ;; esac

echo
echo "=== 6: Level 2 case table, FOM extraction and measurement command"
CASES2="$TOOLS/cases_l2.tsv"
MEAS2="$TOOLS/measure_level2.sh"
if [ ! -f "$CASES2" ]; then
    bad "6a: cases_l2.tsv missing"
else
    ncol_bad=0
    while IFS= read -r line; do
        case "$line" in ''|\#*) continue ;; esac
        n=$(awk -F'\t' '{print NF}' <<< "$line")
        [ "$n" -eq 10 ] || ncol_bad=$((ncol_bad+1))
    done < "$CASES2"
    [ "$ncol_bad" -eq 0 ] && ok "6a: every Level 2 row has 10 tab-separated fields" \
                          || bad "6a: $ncol_bad Level 2 rows with the wrong field count"

    if /usr/bin/grep -qP '\t\t' "$CASES2"; then
        bad "6b: cases_l2.tsv contains an empty field (consecutive tabs)"
    else
        ok "6b: no consecutive tabs in cases_l2.tsv"
    fi

    # one row per level2 application that has a run.sh, and no row for anything else
    missing=""; extra=""
    for d in "$REPO"/level2/*/; do
        a="$(basename "$d")"
        [ -f "$d/run.sh" ] || continue
        /usr/bin/grep -q "^$a	" "$CASES2" || missing="$missing $a"
    done
    while IFS=$'\t' read -r a _rest; do
        case "$a" in ''|\#*) continue ;; esac
        [ -f "$REPO/level2/$a/run.sh" ] || extra="$extra $a"
    done < "$CASES2"
    [ -z "$missing" ] && ok "6c: every level2 app with a run.sh has a case row" \
                      || bad "6c: no case row for:$missing"
    [ -z "$extra" ] && ok "6d: no case row without a level2 run.sh" \
                    || bad "6d: case rows with no run.sh:$extra"
fi

# every FOM pattern must compile, expose exactly one group, and declare a direction;
# a blank FOM row must be blank in all five FOM columns, so a half-filled row cannot
# silently produce a number with no unit.
if [ -f "$CASES2" ]; then
    out="$(python3 - "$CASES2" <<'PYEOF' 2>&1
import csv, re, sys
bad = []
nfom = 0
for r in csv.reader(open(sys.argv[1]), delimiter="\t"):
    if not r or r[0].startswith("#"):
        continue
    app, be, gpus, tmo, name, unit, better, src, rx, note = r
    if not tmo.isdigit() or int(tmo) <= 0:
        bad.append(f"{app}: timeout_s={tmo!r}")
    if name == "-":
        if [unit, better, src, rx] != ["-", "-", "-", "-"]:
            bad.append(f"{app}: no fom_name but other FOM columns are filled")
        continue
    nfom += 1
    if better not in ("higher", "lower"):
        bad.append(f"{app}: fom_better={better!r}")
    if src != "stdout":
        bad.append(f"{app}: fom_source={src!r} is not implemented")
    if unit == "-" and app != "remhos":
        bad.append(f"{app}: FOM without a unit")
    try:
        pat = re.compile(rx, re.M)
    except re.error as exc:
        bad.append(f"{app}: fom_regex does not compile: {exc}")
        continue
    if pat.groups != 1:
        bad.append(f"{app}: fom_regex has {pat.groups} capture groups, need exactly 1")
print(f"NFOM={nfom}")
for b in bad:
    print("BAD", b)
PYEOF
)"
    nfom="$(echo "$out" | sed -n 's/^NFOM=//p')"
    if echo "$out" | /usr/bin/grep -q '^BAD'; then
        bad "6e: FOM columns are inconsistent: $(echo "$out" | /usr/bin/grep '^BAD' | head -3 | tr '\n' ';')"
    else
        ok "6e: all $nfom FOM patterns compile with one capture group and a direction"
    fi
fi

# FOM extraction against synthetic stdout: last match wins, commas are stripped,
# a case without a FOM stays empty, a pattern that misses reports not_matched.
mk_l2_raw() {   # mk_l2_raw <app> <fom_name> <fom_regex> <stdout text>
    local d="$L2/$1/r-$1"
    mkdir -p "$d"
    { echo "schema=hpcperf-timing-raw-1"; echo "level=2"; echo "run_id=r-$1"
      echo "utc=1970-01-01T00:00:00Z"; echo "benchmark=$1"; echo "backend=CUDA"
      echo "runner=level2/$1/run.sh"; echo "gpus=1"; echo "timeout_s=60"
      echo "repeats=1"; echo "warmup=0"; echo "profiled=1"; echo "profiler_in_wall=1"
      echo "run_status=ok"; echo "fom_name=$2"; echo "fom_unit=u"; echo "fom_better=higher"
      echo "fom_source=$([ "$2" = "-" ] && echo - || echo stdout)"; echo "fom_regex=$3"
      echo "fom_note="; echo "hostname=h"; echo "gpu_csv="
      echo "gpu_audit=audit summary: 1 verified, 0 mismatch, 0 unverified"; } > "$d/run_meta.txt"
    echo 1500000000 > "$d/wall_ns.txt"
    echo 0 > "$d/exit_codes.txt"
    printf '%s\n' "$4" > "$d/run.log"
}
SUM="$TOOLS/summarize.py"
L2="$TMP/l2raw"
mk_l2_raw commas 'Lookups/s' 'Lookups/s:\s*([0-9,]+)' 'Lookups/s:   1,234,567'
mk_l2_raw anchored 'FOM' '^FOM:\s+([0-9.eE+-]+)' 'FOM RHS: 49.4
FOM: 10.5
FOM: 11.5'
mk_l2_raw nofom '-' '-' 'no metric here at all'
mk_l2_raw misses 'Ghost' 'Ghost = ([0-9.]+)' 'the application printed something else'
out="$(python3 "$SUM" --raw-root "$TMP/none" --raw-root-l2 "$L2" --out-root "$TMP/l2out" 2>&1 | noise)"
if echo "$out" | /usr/bin/grep -q 'level2 json_written=4'; then
    ok "6f: four Level 2 records written"
else
    bad "6f: summarize did not write four Level 2 records: $out"
fi
res="$(python3 - "$TMP/l2out/summary_level2.csv" <<'PYEOF' 2>&1
import csv, sys
by = {r["app"]: r for r in csv.DictReader(open(sys.argv[1]))}
checks = [
    ("comma stripped", by["commas"]["fom_value"] == "1234567.0"),
    ("comma status ok", by["commas"]["fom_status"] == "ok"),
    ("last match wins", by["anchored"]["fom_value"] == "11.5"),
    ("anchored ^FOM: only", by["anchored"]["fom_status"] == "ok"),
    ("blank stays blank", by["nofom"]["fom_value"] == "" and by["nofom"]["fom_status"] == "none"),
    ("blank name empty", by["nofom"]["fom_name"] == ""),
    ("miss -> not_matched", by["misses"]["fom_value"] == "" and by["misses"]["fom_status"] == "not_matched"),
    ("wall carries profiler", by["commas"]["wall_includes_profiler"] == "1"),
    ("fom marked profiled", by["commas"]["fom_from_profiled_run"] == "1"),
    ("audit parsed clean", by["commas"]["gpu_audit_ok"] == "1"),
]
print(";".join(n for n, okk in checks if not okk) or "ALLOK")
PYEOF
)"
[ "$res" = "ALLOK" ] && ok "6g: FOM extraction handles commas, last-match, blanks and misses" \
                     || bad "6g: FOM extraction wrong: $res"

# a run whose FOM pattern misses, and a single profiled run, must both be caveated
cav="$(python3 - "$TMP/l2out/level2" <<'PYEOF' 2>&1
import glob, json, sys
txt = " ".join(json.dumps(json.load(open(p))["caveats"]) for p in glob.glob(sys.argv[1] + "/*/*.json"))
print("PROF" if "includes nsys overhead" in txt else "-", "MISS" if "did not match" in txt else "-")
PYEOF
)"
case "$cav" in
    "PROF MISS") ok "6h: single-profiled-run and pattern-miss caveats are emitted" ;;
    *)           bad "6h: expected both caveats, got '$cav'" ;;
esac

# the measurement command: no validate.sh, reports under build/timing-l2, and a
# planted credential must not reach the child environment (deny beats allow-list)
if [ ! -x "$MEAS2" ]; then
    bad "6i: measure_level2.sh is not executable"
else
    out="$(HPCPERF_SESSION_TOKEN=planted-l2-credential \
           "$MEAS2" --dry-run xsbench 2>&1 | noise)"
    fails=""
    echo "$out" | /usr/bin/grep -q 'HPCPERF_GPUS=1'            || fails="$fails no-gpus"
    echo "$out" | /usr/bin/grep -q 'build/timing-l2'           || fails="$fails no-raw-root"
    echo "$out" | /usr/bin/grep -q 'run.sh CUDA'               || fails="$fails no-runner"
    echo "$out" | /usr/bin/grep -q 'validate.sh'               && fails="$fails calls-validate"
    echo "$out" | /usr/bin/grep -q 'planted-l2-credential'     && fails="$fails leaks-value"
    echo "$out" | /usr/bin/grep -q 'HPCPERF_SESSION_TOKEN'     && fails="$fails leaks-name"
    echo "$out" | /usr/bin/grep -q 'repeats=1'                 || fails="$fails no-single-run"
    [ -z "$fails" ] && ok "6i: dry-run command is single-run, profiled, validate-free and credential-free" \
                    || bad "6i: dry-run command wrong:$fails"

    # SESSION is on the deny list, so even if it were allow-listed it must be dropped
    if /usr/bin/grep -q "ENV_DENY=" "$MEAS2" && /usr/bin/grep -q 'SESSION' "$MEAS2"; then
        ok "6j: measure_level2.sh carries a credential deny rule that beats the allow-list"
    else
        bad "6j: no credential deny rule in measure_level2.sh"
    fi
fi

echo
echo "timing tests: $pass passed, $failn failed, $skipn skipped"
[ "$failn" -eq 0 ]
