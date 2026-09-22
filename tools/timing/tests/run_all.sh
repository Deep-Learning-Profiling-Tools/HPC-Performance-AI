#!/usr/bin/env bash
# Self-tests for tools/timing. CPU only: no GPU, no profiler, no build tree required.
# Groups that need something absent (a C compiler, gfortran, a build tree) SKIP instead
# of failing.
#
#   bash tools/timing/tests/run_all.sh
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$TOOLS/../.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
# exported because the python heredocs below read them from the environment
export TOOLS REPO TMP
export PYTHONDONTWRITEBYTECODE=1

pass=0; failn=0; skipn=0
ok()   { echo "ok   $*"; pass=$((pass+1)); }
bad()  { echo "FAIL $*"; failn=$((failn+1)); }
skip() { echo "skip $*"; skipn=$((skipn+1)); }
# the login shell on this site prints lmod noise on every subshell
noise() { /usr/bin/grep -viE 'lua|posix|traceback|no file|no field|in function|main chunk|in \?'; }
# a python check prints ALLOK or the list of what failed
pycheck() {   # pycheck <label> ; python source on stdin
    local label="$1" res
    res="$(python3 - 2>&1 | noise | tail -5)"
    if [ "$res" = "ALLOK" ]; then ok "$label"; else bad "$label: $(echo "$res" | tr '\n' ' ')"; fi
}

echo "=== 1: case tables"
out="$(python3 "$TOOLS/cases.py" check 2>&1 | noise)"
if echo "$out" | /usr/bin/grep -q '^cases: level2:'; then
    ok "1a: cases.py check passes ($(echo "$out" | tr '\n' ' ' | sed 's/cases: //g'))"
else
    bad "1a: cases.py check: $out"
fi
if /usr/bin/grep -lP '\t\t|\t$' "$TOOLS"/cases/*.tsv >/dev/null 2>&1; then
    bad "1b: empty field in $(/usr/bin/grep -lP '\t\t|\t$' "$TOOLS"/cases/*.tsv | tr '\n' ' ') (write '-')"
else
    ok "1b: no empty fields in the case tables (bash read would shift columns)"
fi
mkdir -p "$TMP/fb/level1"
rows="$(python3 "$TOOLS/cases.py" resolve --level 1 --build-root "$TMP/fb" --no-env-check all 2>&1 | noise)"
n1="$(printf '%s\n' "$rows" | /usr/bin/grep -c .)"
nbad="$(printf '%s\n' "$rows" | awk -F'\t' 'NF!=17' | wc -l)"
nbm="$(ls -d "$REPO"/level1/*/CMakeLists.txt 2>/dev/null | wc -l)"
if [ "$nbad" -eq 0 ] && [ "$n1" -ge "$nbm" ]; then ok "1c: level 1 resolves to $n1 cases of 17 fields ($nbm benchmarks)"
else bad "1c: level 1 resolve: $n1 rows, $nbad malformed"; fi
if printf '%s\n' "$rows" | awk -F'\t' '$9 ~ /verify\.py|python/' | /usr/bin/grep -q .; then
    bad "1d: a Level 1 case runs a python wrapper instead of the binary"
else
    ok "1d: verify.py-wrapped benchmarks are measured through their inner binary"
fi
rows2="$(python3 "$TOOLS/cases.py" resolve --level 2 --no-env-check amg2023 2>&1 | noise)"
cases2="$(printf '%s\n' "$rows2" | cut -f3 | tr '\n' ' ')"
env128="$(printf '%s\n' "$rows2" | awk -F'\t' '$3=="n128"{print $8}')"
if [ "$cases2" = "default n128 n192 " ] && [ "$env128" = "HPCPERF_AMG_N=128" ]; then
    ok "1e: the sweep n{} expands to one case per value with its own input"
else
    bad "1e: sweep expansion: cases='$cases2' env(n128)='$env128'"
fi

pycheck "1f: refusal rules (reserved, credential-like, bad names, sweeps, stray shell variables)" <<'PY'
import os, sys, shutil
sys.path.insert(0, os.environ["TOOLS"])
import cases
bad = []
def refused(fn, *a):
    try:
        fn(*a)
    except cases.CaseError:
        return True
    return False
for text in ("HPCPERF_ROI_LOG=x", "HPCPERF_GPUS=2", "MY_API_KEY=1", "A_SESSION_ID=1", "lower=1", "NOEQUALS", "A=1;A=2"):
    if not refused(cases.parse_env, text, "t"):
        bad.append(f"parse_env accepted {text!r}")
if not refused(cases.expand_sweep, "n", [("A", "1|2")], "t"):
    bad.append("sweep without {} accepted")
if not refused(cases.expand_sweep, "n{}", [("A", "1|2"), ("B", "3|4")], "t"):
    bad.append("two sweeping variables accepted")
if not refused(cases.expand_sweep, "n{}", [("A", "1")], "t"):
    bad.append("{} without a sweep accepted")
if not refused(cases.expand_sweep, "n{}", [("A", "a b|c")], "t"):
    bad.append("sweep value unusable in a case name accepted")
rows, _ = cases.level2_rows("CUDA")
amg = [r for r in rows if r["app"] == "amg2023" and r["case"] == "default"]
if not refused(cases.refuse_undeclared, amg, {"HPCPERF_AMG_N": "64"}):
    bad.append("a stray HPCPERF_AMG_N in the shell was not refused for amg2023/default")
n128 = [r for r in rows if r["app"] == "amg2023" and r["case"] == "n128"]
try:
    cases.refuse_undeclared(n128, {"HPCPERF_AMG_N": "64"})     # declared by the case: fine
except cases.CaseError as e:
    bad.append(f"declared variable refused: {e}")
# a case that sets a variable its run.sh never reads
tmp = os.path.join(os.environ["TMP"], "casedir")
os.makedirs(tmp, exist_ok=True)
shutil.copy(os.path.join(cases.CASES, "level2_apps.tsv"), tmp)
with open(os.path.join(tmp, "level2_cases.tsv"), "w") as f:
    f.write("amg2023\tbogus\t1\tHPCPERF_NOT_READ_BY_RUNSH=1\t-\t-\t-\t-\n")
cases.CASES = tmp
if not refused(cases.level2_rows, "CUDA"):
    bad.append("a variable run.sh does not read was accepted")
with open(os.path.join(tmp, "level2_cases.tsv"), "w") as f:
    f.write("amg2023\tdefault\t1\t-\t-\t-\t-\t-\namg2023\tdefault\t1\t-\t-\t-\t-\t-\n")
if not refused(cases.level2_rows, "CUDA"):
    bad.append("a duplicate case was accepted")
with open(os.path.join(tmp, "level2_cases.tsv"), "w") as f:
    f.write("amg2023\tdefault\t1\t\t-\t-\t-\t-\n")
if not refused(cases.level2_rows, "CUDA"):
    bad.append("an empty field was accepted")
print("ALLOK" if not bad else "\n".join(bad))
PY

echo
echo "=== 2: ROI log v2 and the vendor-neutral analysis"
pycheck "2a: ROI log parsing: entries, excluded time, unterminated, unmatched, version check" <<'PY'
import os, sys
sys.path.insert(0, os.environ["TOOLS"])
import analysis
tmp = os.environ["TMP"]
p = os.path.join(tmp, "roi.100")
open(p, "w").write("""# hpcperf-roi-log 2
pid 100
rank 0
clock CLOCK_MONOTONIC CLOCK_REALTIME
host h
exe /bin/x
cwd /tmp
argv ["x", "1"]
B 1000 5000
E 3000 7000
x 500 2
B 10000 14000
E 11000 15000
B 20000 24000
U 25000 29000
unmatched_end 1
""")
log = analysis.parse_roi_log(p)
r = analysis.roi_from_log(log)
bad = []
want = {"entries": 3, "gross_ns": 3000, "excluded_ns": 500, "excludes": 2, "wall_ns": 2500,
        "first_begin_real": 5000, "last_end_real": 15000, "unterminated": True, "unmatched_end": 1}
for k, v in want.items():
    if r[k] != v:
        bad.append(f"{k}={r[k]} want {v}")
if log["argv"] != ["x", "1"] or log["rank"] != "0":
    bad.append("header fields")
q = os.path.join(tmp, "roi.101")
open(q, "w").write("# hpcperf-roi-log 2\npid 101\nB 0 0\nE 4000 4000\n")
job = analysis.clean_roi([log, analysis.parse_roi_log(q)])
if job["wall_ns"] != 4000 or job["processes"] != 2 or job["imbalance_ns"] != 1500:
    bad.append(f"multi-process: {job}")
v1 = os.path.join(tmp, "roi.v1")
open(v1, "w").write("# hpcperf-roi-log 1\nB 0 0\n")
try:
    analysis.parse_roi_log(v1)
    bad.append("a version-1 log was accepted")
except ValueError:
    pass
try:
    analysis.finite("x", float("nan"))
    bad.append("NaN accepted")
except ValueError:
    pass
print("ALLOK" if not bad else "\n".join(bad))
PY

pycheck "2b: clipping to ROI minus excludes, union, overlap, null vs 0" <<'PY'
import os, sys
sys.path.insert(0, os.environ["TOOLS"])
import analysis
from collectors import Marker, Interval
import collectors.nvidia_nsys as nsys

class Fake:
    def markers(self):
        return [Marker(1, "roi", 100, 200), Marker(1, "exclude", 140, 160), Marker(1, "roi", 300, 400)]
    def intervals(self):
        return iter(sorted([
            Interval(50, 120, "compute", ("k", 1), 1, 0),      # clipped to 100..120
            Interval(110, 130, "compute", ("k", 2), 1, 0),     # overlaps the first
            Interval(150, 170, "copy_d2d", ("c", 8), 1, 1000), # half inside the exclude
            Interval(190, 310, "fill", ("f", 0), 1, 0),        # spans two ROI entries
            Interval(500, 600, "compute", ("k", 1), 1, 0),     # after the ROI
        ]))
    def op_names(self, keys):
        return {k: str(k) for k in keys}
    def runtime_calls(self, windows=None):
        return None

res = analysis.analyze_trace(Fake(), nsys.CAPABILITIES)
d, w, pr = res["device"], res["whole_process"], res["profiled_roi"]
bad = []
def eq(name, got, want):
    if got is None or abs(got - want) > 1e-15:
        bad.append(f"{name}={got} want {want}")
eq("profiled wall", pr["wall_s"], 180e-9)
eq("excluded", pr["excluded_s"], 20e-9)
eq("compute", d["compute_s"], 40e-9)
eq("d2d", d["copy_d2d_s"], 10e-9)
eq("fill", d["fill_s"], 20e-9)
eq("busy (union)", d["busy_s"], 60e-9)
eq("op time sum", d["op_time_sum_s"], 70e-9)
eq("overlap", d["overlap_s"], 10e-9)
eq("whole busy", w["busy_s"], 320e-9)
if d["compute_ops"] != 2 or d["fill_ops"] != 1 or w["compute_ops"] != 3:
    bad.append(f"op counts roi={d['compute_ops']},{d['fill_ops']} whole={w['compute_ops']}")
if d["copy_d2d_bytes"] != 500:
    bad.append(f"bytes pro rata: {d['copy_d2d_bytes']}")
if d["collective_s"] is not None:
    bad.append("an unobservable category is not null")
if d["copy_h2d_s"] != 0:
    bad.append("an observable empty category is not 0")
if res["ops"][0]["share"] <= 0 or abs(sum(o["share"] for o in res["ops"]) - 1) > 1e-9:
    bad.append("op shares")
none = analysis.analyze_trace(Fake(), frozenset())
if none["device"] is not None or none["whole_process"] is not None:
    bad.append("a collector without capabilities produced device numbers")
print("ALLOK" if not bad else "\n".join(bad))
PY

echo
echo "=== 3: the nvidia_nsys adapter on a synthetic sqlite export, summarize end to end"
pycheck "3a: nsys adapter: markers via text and StringIds, activity, runtime calls, env names only" <<'PY'
import json, os, sqlite3, sys
sys.path.insert(0, os.environ["TOOLS"])
import analysis
import collectors.nvidia_nsys as nsys
tmp = os.environ["TMP"]

def make_trace(path, pid=7):
    P, T = pid << 24, (pid << 24) + 3
    db = sqlite3.connect(path)
    db.executescript("""
      create table StringIds(id integer primary key, value text);
      create table NVTX_EVENTS(start int, end int, eventType int, text text, textId int, globalTid int);
      create table CUPTI_ACTIVITY_KIND_KERNEL(start int, end int, demangledName int, globalPid int);
      create table CUPTI_ACTIVITY_KIND_MEMCPY(start int, end int, copyKind int, bytes int, globalPid int);
      create table CUPTI_ACTIVITY_KIND_MEMSET(start int, end int, bytes int, globalPid int);
      create table CUPTI_ACTIVITY_KIND_RUNTIME(start int, end int, nameId int, globalTid int);
      create table ENUM_CUDA_MEMCPY_OPER(id int, name text, label text);
      create table TARGET_INFO_SYSTEM_ENV(name text, value text);
    """)
    db.executemany("insert into StringIds values(?,?)",
                   [(1, "work_kernel"), (2, "hpcperf:exclude"), (3, "cudaLaunchKernel"),
                    (4, "cudaDeviceSynchronize"), (5, "check_kernel")])
    db.executemany("insert into NVTX_EVENTS values(?,?,?,?,?,?)",
                   [(1000, 2000, 59, "hpcperf:roi", None, T), (1400, 1600, 59, None, 2, T),
                    (1100, 1200, 59, "unrelated", None, T), (1000, 2000, 34, "hpcperf:roi", None, T)])
    db.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?)",
                   [(1100, 1300, 1, P), (1450, 1550, 5, P), (1700, 1800, 1, P), (2500, 2600, 1, P)])
    db.executemany("insert into CUPTI_ACTIVITY_KIND_MEMCPY values(?,?,?,?,?)",
                   [(500, 600, 1, 4096, P), (1800, 1900, 8, 1000, P), (2100, 2200, 2, 4096, P),
                    (2300, 2350, 12, 64, P), (2360, 2370, 10, 64, P)])     # unified DtoH, peer
    db.executemany("insert into CUPTI_ACTIVITY_KIND_MEMSET values(?,?,?,?)", [(400, 450, 64, P)])
    db.executemany("insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,?)",
                   [(1050, 1060, 3, T), (1650, 1660, 3, T), (1900, 1990, 4, T), (2400, 2410, 3, T)])
    db.executemany("insert into ENUM_CUDA_MEMCPY_OPER values(?,?,?)",
                   [(1, "CUDA_MEMCPY_OPER_HTOD", "HtoD"), (2, "CUDA_MEMCPY_OPER_DTOH", "DtoH"),
                    (8, "CUDA_MEMCPY_OPER_DTOD", "DtoD")])
    db.execute("insert into TARGET_INFO_SYSTEM_ENV values('DeviceEnvironment', ?)",
               ("PATH=/usr/bin;HOME=/h;FAKE_API_KEY=planted-value-xyz",))
    db.commit()
    db.close()

os.makedirs(os.path.join(tmp, "nsys"), exist_ok=True)
path = os.path.join(tmp, "nsys", "trace.sqlite")
make_trace(path)
t = nsys.open(os.path.join(tmp, "nsys"))
bad = []
m = t.markers()
if sorted(x.kind for x in m) != ["exclude", "roi"] or len({x.proc for x in m}) != 1:
    bad.append(f"markers {m}")
res = analysis.analyze_trace(t, nsys.CAPABILITIES)
d = res["device"]
# ROI 1000..2000 minus 1400..1600: kernels 1100-1300 (200) + 1700-1800 (100); d2d 1800-1900 (100)
if d["compute_ops"] != 2 or abs(d["compute_s"] - 300e-9) > 1e-15:
    bad.append(f"compute {d['compute_ops']} {d['compute_s']}")
if d["copy_d2d_ops"] != 1 or d["copy_h2d_ops"] != 0 or d["copy_d2h_ops"] != 0 or d["fill_ops"] != 0:
    bad.append("copy/fill counts inside the ROI")
w = res["whole_process"]
if (w["compute_ops"], w["copy_h2d_ops"], w["copy_d2h_ops"], w["copy_other_ops"], w["fill_ops"]) != (4, 1, 2, 1, 1):
    bad.append(f"whole process counts {w}")
rt = res["runtime_api_roi"]
if rt["calls"] != 3 or rt["sync_calls"] != 1:     # 1050, 1650, 1900(sync); 2400 is outside
    bad.append(f"runtime calls inside the ROI windows {rt}")
names = {o["name"] for o in res["ops"]}
if "work_kernel" not in names or "check_kernel" in names:
    bad.append(f"ops table {names}")
info = t.info()
if info.get("recorded_env_names") != ["FAKE_API_KEY", "HOME", "PATH"]:
    bad.append(f"env names {info.get('recorded_env_names')}")
if "planted-value-xyz" in json.dumps(info):
    bad.append("an environment VALUE leaked into info()")
t.close()
print("ALLOK" if not bad else "\n".join(bad))
PY

# a synthetic raw directory in the engine's layout, summarized to JSON + CSV
mkraw() {   # mkraw <dir> <collector> <with_roi 0|1> <fom stdout>
    local d="$1" col="$2" roi="$3"
    mkdir -p "$d/clean.0" "$d/clean.1"
    {
        echo "schema=hpcperf-timing-raw-2"; echo "run_id=$(basename "$d")"; echo "utc=2026-09-22T00:00:00Z"
        echo "level=2"; echo "app=$(basename "$(dirname "$(dirname "$d")")")"; echo "case=$(basename "$(dirname "$d")")"
        echo "backend=CUDA"; echo "gpus=1"; echo "cwd=/tmp"; echo "timeout_s=60"; echo "case_env=HPCPERF_X=1"
        echo "argv=bash level2/x/run.sh CUDA"; echo "fom_name=Rate"; echo "fom_unit=u/s"; echo "fom_better=higher"
        echo "fom_source=stdout"; echo 'fom_regex=^Rate:\s+([0-9.,eE+-]+)'; echo "roi_excludes=-"; echo "verify_vs_roi=outside"
        echo "notes=-"; echo "roi_where=level2/x/main.cpp:10"; echo "warmup_runs=0"; echo "clean_runs=2"
        echo "profiled_runs=1"; echo "skip_verify=0"; echo "collector=$col"; echo "collector_version=-"
        echo "env_script=-"; echo "env_allow=PATH HOME"; echo "env_deny_regex=(TOKEN|SECRET|PASSWD|PASSWORD|CREDENTIAL|PRIVATE_KEY|API_KEY|SESSION)"
        echo "platform_id=test-platform"; echo 'device_json={"schema":"hpcperf-device-1","device":{"vendor":"test","platform_id":"test-platform"},"host":{}}'
        echo "git_commit=abc"; echo "git_dirty=0"; echo "status=ok"
    } > "$d/run_meta.txt"
    local i
    for i in 0 1; do
        echo "start_ns=1000000000 end_ns=1000900000 rc=0" > "$d/clean.$i/run.txt"
        printf '%s\n' "$4" > "$d/clean.$i/run.log"
        if [ "$roi" = 1 ]; then
            printf '# hpcperf-roi-log 2\npid 9\nrank 0\nargv ["x"]\nB 0 1000100000\nE %d 1000%d\n' \
                "$((500000 + i * 100000))" "$((600000 + i * 100000))" > "$d/clean.$i/roi.9"
        fi
    done
}
RAW="$TMP/raw/level2"
mkraw "$RAW/appa/default/r1" none 1 "$(printf 'Rate: 1,234\nRate: 2,500.5')"
mkraw "$RAW/appb/default/r1" none 0 "nothing"
mkraw "$RAW/appc/default/r1" none 1 "no metric line"
out="$(python3 "$TOOLS/summarize.py" --raw-root "$TMP/raw" --out-root "$TMP/res" 2>&1 | noise)"
cp "$TMP/res/summary_level2.csv" "$TMP/first.csv" 2>/dev/null
python3 "$TOOLS/summarize.py" --raw-root "$TMP/raw" --out-root "$TMP/res" >/dev/null 2>&1
if cmp -s "$TMP/first.csv" "$TMP/res/summary_level2.csv"; then
    ok "3b: summarize regenerates byte-identical CSVs from the same raw data"
else
    bad "3b: summarize output is not deterministic ($out)"
fi
pycheck "3c: records: ROI median, null device columns, FOM last match/commas, roi_missing" <<'PY'
import csv, json, os
tmp = os.environ["TMP"]
rows = {r["app"]: r for r in csv.DictReader(open(os.path.join(tmp, "res", "summary_level2.csv")))}
bad = []
a = rows.get("appa", {})
# clean.0 ROI 0.5 ms, clean.1 0.6 ms -> median 0.55 ms
if abs(float(a.get("roi_wall_s") or 0) - 0.00055) > 1e-12:
    bad.append(f"roi median {a.get('roi_wall_s')}")
if a.get("roi_runs") != "2" or a.get("status") != "ok":
    bad.append(f"runs/status {a.get('roi_runs')} {a.get('status')}")
for c in ("device_busy_s", "device_compute_s", "host_gap_s", "device_compute_ops"):
    if a.get(c) != "":
        bad.append(f"{c}={a.get(c)!r} with the none collector (must be empty, not 0)")
if a.get("fom_value") != "2500.5" or a.get("fom_status") != "ok":
    bad.append(f"fom {a.get('fom_value')} {a.get('fom_status')}")
if a.get("inputs_env") != "HPCPERF_X=1":
    bad.append(f"inputs_env {a.get('inputs_env')}")
if abs(float(a.get("pre_roi_s") or 0) - 0.0001) > 1e-9:
    bad.append(f"pre_roi_s {a.get('pre_roi_s')}")
if rows.get("appb", {}).get("status") != "roi_missing":
    bad.append(f"no ROI record -> status {rows.get('appb', {}).get('status')}")
if rows.get("appc", {}).get("fom_status") != "not_matched" or rows.get("appc", {}).get("fom_value") != "":
    bad.append("an unmatched FOM is not left empty")
rec = json.load(open(os.path.join(tmp, "res", "level2", "appa", "default", "r1.json")))
if rec["schema"] != "hpcperf-timing-2" or rec["device"] is not None:
    bad.append("schema / device block")
if not any("No collector" in c for c in rec["caveats"]):
    bad.append("the none collector is not caveated")
print("ALLOK" if not bad else "\n".join(bad))
PY

pycheck "3d: summarize with the nsys adapter: host gap, inflation, credential-name caveat, no values" <<'PY'
import json, os, shutil, sys
sys.path.insert(0, os.environ["TOOLS"])
tmp = os.environ["TMP"]
src = os.path.join(tmp, "raw", "level2", "appa", "default", "r1")
dst = os.path.join(tmp, "raw2", "level2", "appn", "default", "r1")
shutil.copytree(src, dst)
meta = open(os.path.join(dst, "run_meta.txt")).read().replace("collector=none", "collector=nvidia_nsys")
meta = meta.replace("app=appa", "app=appn")
open(os.path.join(dst, "run_meta.txt"), "w").write(meta)
os.makedirs(os.path.join(dst, "prof"))
shutil.copy(os.path.join(tmp, "nsys", "trace.sqlite"), os.path.join(dst, "prof", "trace.sqlite"))
import summarize
rec = summarize.build_record(dst)
bad = []
if rec["status"] != "ok":
    bad.append(f"status {rec['status']}")
# profiled ROI in the synthetic trace: 1000 ns - 200 ns exclude = 800 ns; clean median 0.55 ms
if abs(rec["roi"]["profiled_wall_s"] - 800e-9) > 1e-15:
    bad.append(f"profiled wall {rec['roi']['profiled_wall_s']}")
dev = rec["device"]
if abs(dev["host_gap_s"] - (0.00055 - dev["busy_s"])) > 1e-15:
    bad.append("host gap is not roi_wall - busy")
if not any("credential deny rule" in c for c in rec["caveats"]):
    bad.append("a deny-matching recorded variable name is not caveated")
if "planted-value-xyz" in json.dumps(rec):
    bad.append("an environment value reached the record")
if not any("conformance" in c for c in rec["caveats"]):
    bad.append("a platform without a conformance record is not caveated")
print("ALLOK" if not bad else "\n".join(bad))
PY

echo
echo "=== 4: the C/C++ marker header (compile modes, log format, default off, shared state)"
CC_BIN="$(command -v cc || command -v gcc || true)"
CXX_BIN="$(command -v c++ || command -v g++ || true)"
if [ -z "$CC_BIN" ]; then
    skip "4: no C compiler"
else
    cat > "$TMP/t_roi.c" <<'C'
#include "hpcperf_roi.h"
void tu2_end(void);
int main(void) {
    int i;
    HPCPERF_ROI_END();                        /* unmatched */
    HPCPERF_ROI_EXCLUDE_BEGIN();              /* outside an ROI: ignored */
    HPCPERF_ROI_EXCLUDE_END();
    for (i = 0; i < 3; i++) {
        HPCPERF_ROI_BEGIN();
        HPCPERF_ROI_BEGIN();                  /* nested: only the outermost counts */
        HPCPERF_ROI_EXCLUDE_BEGIN_SYNC();
        HPCPERF_ROI_EXCLUDE_BEGIN();
        HPCPERF_ROI_EXCLUDE_END();
        HPCPERF_ROI_EXCLUDE_END();
        HPCPERF_ROI_END();
        tu2_end();                            /* the outer END lives in another file */
    }
    for (i = 0; i < 20000; i++) {             /* > 8192 events: exercises the flush */
        HPCPERF_ROI_BEGIN_SYNC();
        HPCPERF_ROI_EXCLUDE_BEGIN();
        HPCPERF_ROI_EXCLUDE_END();
        HPCPERF_ROI_END_SYNC();
    }
    HPCPERF_ROI_BEGIN();                      /* never ended -> U at exit */
    return 0;
}
C
    printf '#include "hpcperf_roi.h"\nvoid tu2_end(void) { HPCPERF_ROI_END(); }\n' > "$TMP/t_roi2.c"
    modes_ok=""; modes_bad=""
    for m in "-std=c99" "-std=c11" "-std=gnu11" "-std=c99 -DHPCPERF_ROI_ROCTX" "-std=c99 -DHPCPERF_ROI_NO_ANNOTATION"; do
        if "$CC_BIN" $m -Wall -Wextra -O2 -I"$TOOLS/roi" "$TMP/t_roi.c" "$TMP/t_roi2.c" -o "$TMP/t_roi" 2> "$TMP/cc.log"; then
            modes_ok="$modes_ok [$m]"
        else
            modes_bad="$modes_bad [$m: $(head -3 "$TMP/cc.log" | tr '\n' ' ')]"
        fi
    done
    [ -z "$modes_bad" ] && ok "4a: compiles in$modes_ok" || bad "4a: does not compile:$modes_bad"
    if [ -n "$CXX_BIN" ]; then
        printf '#include "hpcperf_roi.h"\n#include <vector>\nint main() { std::vector<int> v(3); HPCPERF_ROI_BEGIN(); v[0] = 1; HPCPERF_ROI_END(); return 0; }\n' > "$TMP/t_roi.cpp"
        if "$CXX_BIN" -std=c++17 -Wall -Wextra -I"$TOOLS/roi" "$TMP/t_roi.cpp" -o "$TMP/t_roi_cpp" 2> "$TMP/cxx.log"; then
            ok "4b: compiles as C++17"
        else
            bad "4b: C++17: $(head -3 "$TMP/cxx.log" | tr '\n' ' ')"
        fi
    else
        skip "4b: no C++ compiler"
    fi
    "$CC_BIN" -std=c99 -O2 -I"$TOOLS/roi" "$TMP/t_roi.c" "$TMP/t_roi2.c" -o "$TMP/t_roi" 2>/dev/null
    mkdir -p "$TMP/off"
    ( cd "$TMP/off" && env -u HPCPERF_ROI_LOG ../t_roi )
    if [ -n "$(ls -A "$TMP/off")" ]; then
        bad "4c: a file was written although HPCPERF_ROI_LOG is unset: $(ls "$TMP/off")"
    else
        ok "4c: default off: without HPCPERF_ROI_LOG nothing is written"
    fi
    ( cd "$TMP" && HPCPERF_ROI_LOG="$TMP/on" ./t_roi )
    pycheck "4d: log v2 from C: 20004 entries, nesting, cross-file END, excludes summed, U, unmatched, no overflow" <<'PY'
import glob, os, sys
sys.path.insert(0, os.environ["TOOLS"])
import analysis
tmp = os.environ["TMP"]
logs = glob.glob(os.path.join(tmp, "on.*"))
bad = []
if len(logs) != 1:
    print(f"expected one log, got {logs}"); sys.exit()
log = analysis.parse_roi_log(logs[0])
r = analysis.roi_from_log(log)
if r["entries"] != 20004:
    bad.append(f"entries {r['entries']}")
if r["excludes"] != 20003:
    bad.append(f"excludes {r['excludes']}")
if not r["unterminated"] or r["unmatched_end"] != 1 or r["overflow"] != 0:
    bad.append(f"flags {r}")
if not (0 <= r["excluded_ns"] <= r["gross_ns"]) or r["wall_ns"] != r["gross_ns"] - r["excluded_ns"]:
    bad.append("excluded time not inside the gross ROI")
if open(logs[0]).read().count("# hpcperf-roi-log 2") != 1:
    bad.append("the header was written more than once across flushes")
if not log["argv"] or not log["argv"][0].endswith("t_roi"):
    bad.append(f"argv {log['argv']}")
print("ALLOK" if not bad else "\n".join(bad))
PY
    # strict ISO C with a system header first must fail loudly, not silently lose the clock
    printf '#include <stdio.h>\n#include "hpcperf_roi.h"\nint main(void) { return 0; }\n' > "$TMP/t_late.c"
    if "$CC_BIN" -std=c99 -I"$TOOLS/roi" "$TMP/t_late.c" -o "$TMP/t_late" 2> "$TMP/late.log"; then
        ok "4e: strict C99 with a system header first still compiles (glibc exposes clock_gettime)"
    elif /usr/bin/grep -q 'needs POSIX clock_gettime' "$TMP/late.log"; then
        ok "4e: strict C99 with a system header first fails with the header's own message"
    else
        bad "4e: strict C99 late include: $(head -2 "$TMP/late.log" | tr '\n' ' ')"
    fi
fi

FC_BIN="$(command -v gfortran || true)"
if [ -z "$FC_BIN" ] || [ -z "$CC_BIN" ]; then
    skip "4f: Fortran interface (no gfortran)"
else
    cat > "$TMP/t_roi.f90" <<'F'
program t
  use hpcperf_roi
  implicit none
  integer :: i
  do i = 1, 4
    call hpcperf_roi_begin_sync()
    call hpcperf_roi_exclude_begin_sync()
    call hpcperf_roi_exclude_end()
    call hpcperf_roi_end_sync()
  end do
end program t
F
    if ( cd "$TMP" && "$FC_BIN" -c "$TOOLS/roi/hpcperf_roi.f90" -o hpcperf_roi_f.o \
         && "$CC_BIN" -O2 -I"$TOOLS/roi" -c "$TOOLS/roi/hpcperf_roi_fortran.c" -o hpcperf_roi_c.o \
         && "$FC_BIN" t_roi.f90 hpcperf_roi_f.o hpcperf_roi_c.o -o t_roi_f ) > "$TMP/f.log" 2>&1 \
       && ( cd "$TMP" && HPCPERF_ROI_LOG="$TMP/fort" ./t_roi_f ); then
        e="$(cat "$TMP"/fort.* 2>/dev/null | /usr/bin/grep -c '^E ')"
        x="$(cat "$TMP"/fort.* 2>/dev/null | awk '/^x /{n+=$3} END{print n+0}')"
        [ "$e" = 4 ] && [ "$x" = 4 ] && ok "4f: Fortran module + C shim: 4 entries, 4 excludes" \
                                   || bad "4f: Fortran log has $e entries, $x excludes"
    else
        bad "4f: Fortran interface does not build/run: $(head -3 "$TMP/f.log" | tr '\n' ' ')"
    fi
fi

echo
echo "=== 5: the Python marker API (same log format)"
pycheck "5a: hpcperf_roi.py writes a log the analysis reads identically; off without the variable" <<'PY'
import glob, os, subprocess, sys
tmp, tools = os.environ["TMP"], os.environ["TOOLS"]
prog = r"""
import sys; sys.path.insert(0, sys.argv[1])
import hpcperf_roi as roi
calls = []
roi.set_device_sync(lambda: calls.append(1))
roi.end()                                   # unmatched
for i in range(3):
    with roi.region(sync=True):
        with roi.region(sync=False):        # nested
            with roi.excluded(sync=True):
                pass
for i in range(5000):                       # > the flush threshold
    roi.begin(); roi.exclude_begin(); roi.exclude_end(); roi.end()
roi.begin()                                 # unterminated
print("syncs", len(calls))
"""
env = dict(os.environ, HPCPERF_ROI_LOG=os.path.join(tmp, "py"))
out = subprocess.run([sys.executable, "-c", prog, os.path.join(tools, "roi")], env=env,
                     capture_output=True, text=True)
env_off = {k: v for k, v in os.environ.items() if k != "HPCPERF_ROI_LOG"}
off = subprocess.run([sys.executable, "-c", prog, os.path.join(tools, "roi")], env=env_off,
                     capture_output=True, text=True, cwd=tmp)
sys.path.insert(0, tools)
import analysis
bad = []
logs = glob.glob(os.path.join(tmp, "py.*"))
if len(logs) != 1:
    print(f"logs {logs} {out.stderr}"); sys.exit()
r = analysis.roi_from_log(analysis.parse_roi_log(logs[0]))
if (r["entries"], r["excludes"], r["unterminated"], r["unmatched_end"]) != (5004, 5003, True, 1):
    bad.append(f"python log {r}")
if "syncs 9" not in out.stdout:        # 3 begin + 3 end + 3 exclude syncs, only when measuring
    bad.append(f"device sync calls: {out.stdout.strip()}")
if "syncs 0" not in off.stdout:
    bad.append(f"device sync called although not measuring: {off.stdout.strip()}")
print("ALLOK" if not bad else "\n".join(bad))
PY

echo
echo "=== 6: markers are in place (static check of the sources and build scripts)"
pycheck "6a: every Level 1 benchmark: include + BEGIN + END in each backend, CMake include path" <<'PY'
import glob, os, re
repo = os.environ["REPO"]
bad = []
src_ext = (".c", ".cc", ".cpp", ".cu", ".hip", ".h", ".hpp", ".cuh")
for cm in sorted(glob.glob(os.path.join(repo, "level1", "*", "CMakeLists.txt"))):
    bm = os.path.dirname(cm)
    name = os.path.basename(bm)
    if "tools/timing/roi" not in open(cm).read():
        bad.append(f"{name}: CMakeLists has no tools/timing/roi include path")
    for be in ("cuda", "hip"):
        dirs = [os.path.join(bm, be), os.path.join(bm, "common")]
        if not os.path.isdir(dirs[0]):
            continue
        text = ""
        for d in dirs:
            for p in glob.glob(os.path.join(d, "**", "*"), recursive=True):
                if p.endswith(src_ext) and os.path.isfile(p):
                    text += open(p, errors="replace").read()
        if not re.search(r"HPCPERF_ROI_BEGIN(_SYNC)?\(", text) or not re.search(r"HPCPERF_ROI_END(_SYNC)?\(", text):
            bad.append(f"{name}/{be}: no BEGIN/END marker")
        if '#include "hpcperf_roi.h"' not in text:
            bad.append(f"{name}/{be}: header not included")
print("ALLOK" if not bad else "\n".join(bad[:10]))
PY
pycheck "6b: every Level 2 application: BEGIN + END in its sources, CPATH export in build.sh" <<'PY'
import glob, os, re
repo = os.environ["REPO"]
bad = []
rx = re.compile(r"HPCPERF_ROI_(BEGIN|END)(_SYNC)?\(|hpcperf_roi_(begin|end)(_sync)?\(")
for run in sorted(glob.glob(os.path.join(repo, "level2", "*", "run.sh"))):
    app = os.path.dirname(run)
    name = os.path.basename(app)
    kinds = set()
    for p in glob.glob(os.path.join(app, "**", "*"), recursive=True):
        if "/build/" in p or not os.path.isfile(p):
            continue
        if not p.endswith((".c", ".cc", ".cpp", ".cxx", ".C", ".cu", ".h", ".hpp", ".f90", ".F90")):
            continue
        for m in rx.finditer(open(p, errors="replace").read()):
            kinds.add("begin" if "begin" in m.group(0).lower() else "end")
    if kinds != {"begin", "end"}:
        bad.append(f"{name}: markers {sorted(kinds)}")
    b = os.path.join(app, "build.sh")
    if not os.path.isfile(b) or "tools/timing/roi" not in open(b).read():
        bad.append(f"{name}: build.sh does not put tools/timing/roi on the include path")
print("ALLOK" if not bad else "\n".join(bad[:10]))
PY
if /usr/bin/grep -rn 'HPCPERF_ROI_LOG\|HPCPERF_SKIP_VERIFY' "$REPO"/level1/*/CMakeLists.txt "$REPO"/level2/*/validate.sh >/dev/null 2>&1; then
    bad "6c: ctest or validate.sh sets a measurement variable -- validation would change"
else
    ok "6c: no ctest command or validate.sh sets HPCPERF_ROI_LOG / HPCPERF_SKIP_VERIFY"
fi
BR=""
for cand in "$REPO/build/gcc13" "$REPO/build/all"; do [ -d "$cand/level1" ] && { BR="$cand"; break; }; done
if [ -z "$BR" ]; then
    skip "6d: no Level 1 build tree"
else
    nb=0; missing=""
    for exe in "$BR"/level1/*/*_cuda; do
        [ -x "$exe" ] || continue
        nb=$((nb+1))
        /usr/bin/grep -q 'hpcperf:roi' "$exe" || missing="$missing $(basename "$exe")"
    done
    [ "$nb" -gt 0 ] && [ -z "$missing" ] && ok "6d: all $nb built Level 1 binaries carry the markers ($BR)" \
                                         || bad "6d: $nb binaries, without markers:$missing"
fi

echo
echo "=== 7: front-ends, clean environment and the credential deny rule"
for f in measure_level1.sh measure_level2.sh lib/engine.sh lib/collectors.sh probes/conformance/run_conformance.sh; do
    bash -n "$TOOLS/$f" 2>/dev/null || bad "7a: shell syntax of $f"
done
ok "7a: shell syntax of the front-ends, engine, collectors and conformance runner"
for M in measure_level1.sh measure_level2.sh; do
    out="$(bash "$TOOLS/$M" --build-root "$TMP/fb" 2>&1 | noise)"
    [ "$M" = measure_level2.sh ] && out="$(bash "$TOOLS/$M" 2>&1 | noise)"
    case "$out" in *"give a selection"*) ;; *) bad "7b: $M runs without a selection: $out" ;; esac
    out="$(bash "$TOOLS/$M" --build-root "$TMP/fb" --clean-runs 0 x 2>&1 | noise)"
    [ "$M" = measure_level2.sh ] && out="$(bash "$TOOLS/$M" --clean-runs 0 x 2>&1 | noise)"
    case "$out" in *"--clean-runs must be a positive number"*) ;; *) bad "7b: $M accepts --clean-runs 0" ;; esac
    out="$(bash "$TOOLS/$M" --bogus x 2>&1 | noise)"
    case "$out" in *"unknown option"*) ;; *) bad "7b: $M accepts an unknown option" ;; esac
    out="$(bash "$TOOLS/$M" --help 2>&1 | noise)"
    case "$out" in *"--dry-run"*) ;; *) bad "7b: $M --help prints no options" ;; esac
done
ok "7b: argument handling (selection required, --clean-runs, unknown options, --help)"

mkdir -p "$TMP/fb/level1/daxpy"
out="$(env PLANTED_API_KEY=planted-xyz HPCPERF_UNRELATED=1 bash "$TOOLS/measure_level1.sh" \
       --build-root "$TMP/fb" --dry-run --collector none daxpy 2>&1 | noise)"
if echo "$out" | /usr/bin/grep -q 'HPCPERF_ROI_LOG=<run dir>/roi' \
   && echo "$out" | /usr/bin/grep -q 'HPCPERF_SKIP_VERIFY=1' \
   && echo "$out" | /usr/bin/grep -q "$TMP/fb/level1/daxpy" \
   && ! echo "$out" | /usr/bin/grep -q 'PLANTED_API_KEY\|planted-xyz\|HPCPERF_UNRELATED'; then
    ok "7c: level 1 dry run: env -i allow-list, ROI log, skip-verify, binary from the build root"
else
    bad "7c: level 1 dry run: $(echo "$out" | head -8 | tr '\n' ' ')"
fi
out="$(bash "$TOOLS/measure_level2.sh" --dry-run --collector none amg2023/n128 2>&1 | noise)"
if echo "$out" | /usr/bin/grep -q 'HPCPERF_AMG_N=128' && echo "$out" | /usr/bin/grep -q 'HPCPERF_GPUS=1' \
   && echo "$out" | /usr/bin/grep -q 'level2/amg2023/run.sh CUDA'; then
    ok "7d: level 2 dry run carries the case input, the GPU count and run.sh"
else
    bad "7d: level 2 dry run: $(echo "$out" | head -8 | tr '\n' ' ')"
fi
out="$(env HPCPERF_AMG_N=64 bash "$TOOLS/measure_level2.sh" --dry-run --collector none amg2023/default 2>&1 | noise)"
case "$out" in *"not declared by the case"*) ok "7e: a stray input variable in the shell is refused" ;;
             *) bad "7e: stray HPCPERF_AMG_N not refused: $(echo "$out" | head -3 | tr '\n' ' ')" ;; esac

res="$(
    LEVEL=1; ENV_SCRIPT_ABS=""
    . "$TOOLS/lib/engine.sh" 2>/dev/null
    BASE_ALLOW="$BASE_ALLOW MY_SESSION_ID"          # an allow-listed name the deny rule must still beat
    export MY_SESSION_ID=planted-session PLANTED_TOKEN=planted-token OTHER_VAR=1
    run_clean "$TMP/envdump.txt" "$TMP" -- env
    _allowed_pairs >/dev/null
    if /usr/bin/grep -q 'planted-\|OTHER_VAR' "$TMP/envdump.txt"; then echo "LEAK"
    elif ! /usr/bin/grep -q '^PATH=' "$TMP/envdump.txt"; then echo "NOPATH"
    else echo "OK:$DENIED_NAMES"; fi
)"
case "$res" in "OK: MY_SESSION_ID") ok "7f: run_clean passes only allow-listed names; the deny rule beats the allow-list" ;;
               *) bad "7f: clean environment: $res" ;; esac

echo
echo "=== 8: hardware neutrality -- interface-only platforms refuse instead of guessing"
pycheck "8a: AMD / TPU collectors and probes are interface-only and refuse; none works everywhere" <<'PY'
import os, sys
sys.path.insert(0, os.environ["TOOLS"])
sys.path.insert(0, os.path.join(os.environ["TOOLS"], "probes"))
import collectors
bad = []
for n in ("amd_rocprofv3", "tpu_xprof"):
    m = collectors.get(n)
    if m.VERIFIED:
        bad.append(f"{n} claims VERIFIED")
    try:
        m.open("/nonexistent")
        bad.append(f"{n}.open did not refuse")
    except NotImplementedError:
        pass
none = collectors.get("none")
t = none.open("/nonexistent")
if none.CAPABILITIES or t.markers() or list(t.intervals()) or t.runtime_calls() is not None:
    bad.append("none collector is not empty")
try:
    collectors.get("bogus")
    bad.append("unknown collector accepted")
except KeyError:
    pass
import device
os.environ["TPU_NAME"] = "fake"
try:
    device.probe_tpu()
    bad.append("TPU probe guessed instead of refusing")
except NotImplementedError:
    pass
print("ALLOK" if not bad else "\n".join(bad))
PY
res="$(. "$TOOLS/lib/collectors.sh"; collector_wrap amd_rocprofv3 /tmp/x 2>/dev/null; a=$?
       collector_wrap tpu_xprof /tmp/x 2>/dev/null; b=$?; collector_wrap none /tmp/x; c=$?
       echo "$a $b $c ${#COLLECTOR_ARGV[@]} $(backend_env_allow XLA | tr -s ' \n' ' ')")"
case "$res" in "2 2 0 0 TPU_NAME"*) ok "8b: the measurement side refuses interface-only collectors; XLA allow-list defined" ;;
               *) bad "8b: collector_wrap / backend_env_allow: $res" ;; esac

echo
echo "=== 9: Level 1 skip-verify switch is default-off"
patched=0; badlog=""
for f in $(/usr/bin/grep -rl 'hpcperf_skip_verify' "$REPO"/level1/*/cuda/ "$REPO"/level1/*/common/ 2>/dev/null); do
    patched=$((patched+1))
    if /usr/bin/grep -qE 'HPCPERF_SKIP_VERIFY"?\s*\)\s*==\s*NULL|skip_verify\s*=\s*true' "$f"; then
        badlog="$badlog $f"
    fi
done
[ "$patched" -ge 20 ] && [ -z "$badlog" ] && ok "9a: $patched files carry the switch, never enabled by default" \
                                          || bad "9a: $patched files; default-on suspicion:$badlog"

echo
echo "=== 10: gen_cases.py resolves a working ctest and writes one row per ctest test"
G="$TOOLS/gen_cases.py"
FB="$TMP/fakebuild"; mkdir -p "$FB/level1/demo" "$FB/level1/multi" "$TMP/badbin"
printf '#!/bin/sh\nexit 1\n' > "$TMP/badbin/ctest"; chmod +x "$TMP/badbin/ctest"
cat > "$TMP/stubctest" <<'STUB'
#!/bin/sh
case "$1" in --version) echo "ctest version 9.9.9"; exit 0 ;; esac
case "$*" in
  *multi*) echo '{"kind":"ctestInfo","tests":[
      {"name":"multi_small","command":["/bin/true","1"],"properties":[{"name":"WORKING_DIRECTORY","value":"/tmp"}]},
      {"name":"multi_large","command":["/bin/true","2"],"properties":[{"name":"WORKING_DIRECTORY","value":"/tmp"}]}]}' ;;
  *) echo '{"kind":"ctestInfo","tests":[{"name":"demo_run","command":["/bin/true","7"],
      "properties":[{"name":"WORKING_DIRECTORY","value":"/tmp"},{"name":"TIMEOUT","value":600}]}]}' ;;
esac
STUB
chmod +x "$TMP/stubctest"
echo "CMAKE_CTEST_COMMAND:INTERNAL=$TMP/stubctest" > "$FB/CMakeCache.txt"
touch "$FB/level1/demo/CTestTestfile.cmake" "$FB/level1/multi/CTestTestfile.cmake"
out="$(PATH="$TMP/badbin:$PATH" python3 "$G" --build-root "$FB" --out "$TMP/out.tsv" 2>&1 | noise)"
case "$out" in *"ctest 9.9.9 at $TMP/stubctest"*) ok "10a: prefers CMAKE_CTEST_COMMAND over a broken PATH ctest" ;;
                                               *) bad "10a: did not use the build tree's ctest: $out" ;; esac
if /usr/bin/grep -q "^demo	default	/bin/true	7	" "$TMP/out.tsv" 2>/dev/null \
   && [ "$(/usr/bin/grep -c '^multi	' "$TMP/out.tsv" 2>/dev/null)" = 2 ]; then
    ok "10b: one test -> case 'default'; several tests -> one case each"
else
    bad "10b: rows: $(/usr/bin/grep -v '^#' "$TMP/out.tsv" 2>/dev/null | cut -f1-4 | tr '\t\n' ' ;')"
fi
if /usr/bin/grep -q "$TMP" "$TMP/out.tsv" 2>/dev/null; then
    bad "10c: an absolute build path leaked into the generated table"
else
    ok "10c: paths are stored with placeholders, not absolute build paths"
fi
rm "$FB/CMakeCache.txt"
PY3="$(command -v python3)"
out="$(PATH="$TMP/badbin:/usr/bin:/bin" "$PY3" "$G" --build-root "$FB" --out "$TMP/out2.tsv" 2>&1 | noise)"
if echo "$out" | /usr/bin/grep -q 'CMAKE_CTEST_COMMAND' && echo "$out" | /usr/bin/grep -q 'HPCPERF_CTEST'; then
    ok "10d: with no usable ctest it fails naming every resolution attempt"
else
    bad "10d: resolution failure is not reported with its trail: $out"
fi

echo
echo "=== 11: FOM extraction (the application's own metric)"
pycheck "11a: last match wins, commas stripped, blank stays blank, a miss is not_matched" <<'PY'
import os, sys
sys.path.insert(0, os.environ["TOOLS"])
import summarize
tmp = os.environ["TMP"]
p = os.path.join(tmp, "fom.log")
open(p, "w").write("FOM RHS: 49.4\nFOM: 10.5\nLookups/s: 1,234,567\nFOM: 11.5\n")
bad = []
def f(regex, name="F"):
    return summarize.extract_fom(p, {"fom_name": name, "fom_unit": "u", "fom_better": "higher",
                                     "fom_source": "stdout", "fom_regex": regex})
r = f(r"^FOM:\s+([0-9.eE+-]+)")
if (r["value"], r["status"]) != (11.5, "ok"):
    bad.append(f"last match {r}")
r = f(r"Lookups/s:\s+([0-9.,]+)")
if r["value"] != 1234567.0:
    bad.append(f"commas {r}")
r = f(r"^Nope:\s+([0-9.]+)")
if (r["value"], r["status"]) != (None, "not_matched"):
    bad.append(f"miss {r}")
r = summarize.extract_fom(p, {"fom_name": "-"})
if (r["value"], r["status"]) != (None, "none"):
    bad.append(f"blank {r}")
try:
    f(r"(a)(b)")
    bad.append("two capture groups accepted")
except RuntimeError:
    pass
print("ALLOK" if not bad else "\n".join(bad))
PY

echo
echo "tools/timing tests: $pass passed, $failn failed, $skipn skipped"
[ "$failn" -eq 0 ]
