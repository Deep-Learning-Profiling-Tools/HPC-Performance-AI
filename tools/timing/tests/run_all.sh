#!/usr/bin/env bash
# Self-tests for tools/timing. CPU only: no GPU, no build tree, no nsys required.
# Groups that need something absent (a build tree, nsys) SKIP instead of failing.
#
#   bash tools/timing/tests/run_all.sh
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$TOOLS/../.." && pwd)"

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
echo "timing tests: $pass passed, $failn failed, $skipn skipped"
[ "$failn" -eq 0 ]
