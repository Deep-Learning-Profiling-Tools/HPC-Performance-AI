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
nbad="$(printf '%s\n' "$rows" | awk -F'\t' 'NF!=19' | wc -l)"
nbm="$(ls -d "$REPO"/level1/*/CMakeLists.txt 2>/dev/null | wc -l)"
if [ "$nbad" -eq 0 ] && [ "$n1" -ge "$nbm" ]; then ok "1c: level 1 resolves to $n1 cases of 19 fields ($nbm benchmarks)"
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
res="$(cd "$TMP" && LEVEL=1 RAW_ROOT=rel/raw bash -c '. "$1/lib/engine.sh" 2>/dev/null; echo "$RAW_ROOT"' _ "$TOOLS")"
case "$res" in "$TMP/rel/raw") ok "7g: a relative --raw-root becomes absolute (the ROI log path is read from other cwds)" ;;
               *) bad "7g: relative raw root stays '$res'" ;; esac

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
echo "=== 12: the web page generator (report.py) and the application timer"
python3 - <<'PY'
import json, os, glob
tmp = os.environ["TMP"]
src = glob.glob(os.path.join(tmp, "res", "level2", "appa", "default", "*.json"))[0]
r = json.load(open(src))
r["run_id"] = "r2"
r["utc"] = "2026-09-23T00:00:00Z"
r["roi"]["wall_s"] *= 1.10
r["ops"] = [{"name": "k<int>(float*) </script><script>alert(1)</script>", "category": "compute", "count": 1,
             "total_s": 1e-3, "avg_s": 1e-3, "min_s": 1e-3, "max_s": 1e-3, "share": 1.0}]
json.dump(r, open(os.path.join(os.path.dirname(src), "r2.json"), "w"))
PY
R="$TOOLS/report.py"
python3 "$R" --results-root "$TMP/res" --out "$TMP/rep" >/dev/null 2>&1
python3 "$R" --results-root "$TMP/res" --out "$TMP/rep2" >/dev/null 2>&1
if cmp -s "$TMP/rep/index.html" "$TMP/rep2/index.html" && cmp -s "$TMP/rep/README.md" "$TMP/rep2/README.md"; then
    ok "12a: the same records give byte-identical pages (no wall-clock time in them)"
else
    bad "12a: report output is not deterministic"
fi
pycheck "12b: interactive page: inputs x platforms with null, latest run, history, escaping, no paths, Markdown" <<'PY'
import json, os, re
tmp, repo = os.environ["TMP"], os.environ["REPO"]
h = open(os.path.join(tmp, "rep", "index.html")).read()
md = open(os.path.join(tmp, "rep", "README.md")).read()
bad = []
m = re.search(r'<script type="application/json" id="timing-data">(.*?)</script>', h, re.S)
if not m:
    print("no embedded data"); raise SystemExit
D = json.loads(m.group(1))
plats = [p["id"] for p in D["platforms"]]
if "test-platform" not in plats or "nvidia-b200.cuda13.2" not in plats:
    bad.append(f"platforms {plats} (measured ones and those with a conformance record)")
apps2 = {a["app"]: a for a in D["levels"]["2"]}
a = apps2.get("appa")
cell = a and [c for c in a["cases"] if c["case"] == "default"][0]["cells"]
if not cell or cell.get("nvidia-b200.cuda13.2") is not None:
    bad.append("a platform without a measurement of appa is not null")
c = (cell or {}).get("test-platform") or {}
if [x["run_id"] for x in c.get("history", [])] != ["r1", "r2"] or c.get("run", {}).get("run_id") != "r2":
    bad.append("the latest successful run / its history is wrong")
if not c.get("prev") or abs(c["run"]["roi"]["wall_s"] / c["prev"]["roi_s"] - 1.1) > 1e-9:
    bad.append("the previous successful run is not recorded")
amg = apps2.get("amg2023")
if not amg or sorted(x["case"] for x in amg["cases"]) != ["default", "n128", "n192"] \
        or any(v is not None for x in amg["cases"] for v in x["cells"].values()):
    bad.append("inputs from the case tables without a measurement are not all null")
b = [x for x in apps2.get("appb", {}).get("cases", []) if x["case"] == "default"]
if not b or (b[0]["cells"].get("test-platform") or {}).get("run", {}).get("status") != "roi_missing":
    bad.append("a combination without a successful run does not carry its status")
if len(D["levels"]["1"]) < 50:
    bad.append("Level 1 benchmarks from the case tables are missing")
if "</script><script>alert" in h or "alert(1)" not in h:
    bad.append("an operation name was not kept as inert JSON text")
if repo in h or repo in md:
    bad.append("an absolute path of the checkout reached the page")
for gone in ("Is the ROI the right region", "Platforms and collectors", "How to read and regenerate", "Needs attention"):
    if gone in h or gone in md:
        bad.append(f"section still present: {gone}")
if not re.search(r"^\| appa \| default \| test-platform \| .* \| 2 \| \+10\.0% \|$", md, re.M):
    bad.append("the Markdown row for appa is missing its run count / change")
if "<script>alert" in md:
    bad.append("an operation name was not escaped in the Markdown")
print("ALLOK" if not bad else "\n".join(bad))
PY
pycheck "12c: application timer: unit scaling, last match wins, ROI difference, missing log" <<'PY'
import os, sys
sys.path.insert(0, os.environ["TOOLS"])
import summarize
tmp = os.environ["TMP"]
p = os.path.join(tmp, "timer.log")
open(p, "w").write("main    1   9.000e+06\nmain    1   7.402e+06\n")
bad = []
t = summarize.extract_app_timer(p, (r"^main\s+1\s+([0-9.eE+-]+)", 1e-6))
if t["status"] != "ok" or abs(t["value_s"] - 7.402) > 1e-9:
    bad.append(f"extraction {t}")
if summarize.extract_app_timer(p, None) is not None:
    bad.append("an app without a timer pattern got a timer block")
if summarize.extract_app_timer(os.path.join(tmp, "nope.log"), ("x(1)", 1.0))["status"] != "log_missing":
    bad.append("missing log")
import cases
try:
    cases.check_app_timer({"app_timer_regex": "(a)(b)", "app_timer_unit": "s", "_where": "t"})
    bad.append("two capture groups accepted")
except cases.CaseError:
    pass
try:
    cases.check_app_timer({"app_timer_regex": "(a)", "app_timer_unit": "min", "_where": "t"})
    bad.append("unknown unit accepted")
except cases.CaseError:
    pass
print("ALLOK" if not bad else "\n".join(bad))
PY
out="$(python3 "$TOOLS/summarize.py" --raw-root "$TMP/raw" --out-root "$TMP/res3" --run-id nope --no-report 2>&1 | noise)"
case "$out" in *"json_written=0 "*) ok "12d: summarize --run-id processes only the named run" ;;
               *) bad "12d: --run-id filter: $out" ;; esac
[ ! -e "$TMP/res3/report" ] && ok "12e: --no-report writes no page" || bad "12e: --no-report still wrote report/"
out="$(bash "$TOOLS/measure_level1.sh" --build-root "$TMP/fb" --dry-run --collector none daxpy 2>&1 | noise)"
case "$out" in *summarizing*) bad "12f: a dry run summarized" ;;
               *) ok "12f: a dry run neither measures nor summarizes" ;; esac

echo "=== 13: Level 3 (the applications' own timers)"
rows3="$(python3 "$TOOLS/cases.py" resolve --level 3 --no-env-check all 2>&1 | noise)"
n3="$(printf '%s\n' "$rows3" | /usr/bin/grep -c .)"
nbad3="$(printf '%s\n' "$rows3" | awk -F'\t' 'NF!=19' | wc -l)"
napps3="$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import cases; print(len(cases.level3_apps_in_suite()))' "$TOOLS" 2>&1 | noise)"
if [ "$nbad3" -eq 0 ] && [ "$n3" -ge "$napps3" ] && [ "$napps3" -gt 0 ]; then
    ok "13a: level 3 resolves to $n3 cases of 19 fields ($napps3 applications)"
else bad "13a: level 3 resolve: $n3 rows, $nbad3 malformed, $napps3 apps: $(printf '%s' "$rows3" | head -3)"; fi
# every case is a declared input of a real run.sh; an undeclared variable in the shell is refused
out="$(HPCPERF_LAMMPS_STEPS=7 python3 "$TOOLS/cases.py" resolve --level 3 lammps 2>&1 | noise)"
case "$out" in *"not declared by the case"*) bad "13b: a declared variable was reported as undeclared: $out" ;;
               *) ok "13b: a variable the case declares may also be set in the shell" ;; esac
out="$(HPCPERF_EXACA_SEED=3 python3 "$TOOLS/cases.py" resolve --level 3 exaca 2>&1 | noise)"
case "$out" in *"not declared by the case"*HPCPERF_EXACA_SEED*) ok "13c: an undeclared Level 3 input set in the shell is refused" ;;
               *) bad "13c: undeclared HPCPERF_EXACA_SEED not refused: $out" ;; esac

# synthetic evidence of one clean run per application (the lines each extractor anchors on,
# numbers taken from real 2-GPU runs on dgx003); the expected region time follows the definition
python3 - <<'PY'
import os
T = os.environ["TMP"] + "/l3"
def w(app, rel, text):
    p = os.path.join(T, app, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(text)
w("lammps", "run.log", """Loop time of 0.5 on 2 procs for 10 steps with 32000 atoms

Loop time of 0.0709434 on 2 procs for 100 steps with 32000 atoms

Performance: 608936.013 tau/day, 1409.574 timesteps/s, 45.106 Matom-step/s
99.1% CPU use with 2 MPI tasks x 1 OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0.0011158  | 0.0011724  | 0.0012289  |   0.2 |  1.65
Comm    | 0.065329   | 0.065331   | 0.065334   |   0.0 | 92.09
Other   |            | 0.0007312  |            |       |  1.03

Nlocal:  16000 ave
""")
w("sparta", "run.log", """Loop time of 0.0529465 on 2 procs for 30 steps with 10000 particles

Loop time of 0.0411685 on 2 procs for 100 steps with 10000 particles

Performance: 2429.041 timesteps/s, 24.290 Mparticle-step/s
""")
w("warpx", "run.log", """STEP 1 ends. TIME = 1.7e-15 DT = 1.7e-15
Evolve time = 0.01901006 s; This step = 0.01901006 s; Avg. per step = 0.01901006 s
STEP 2 ends. TIME = 3.4e-15 DT = 1.7e-15
Evolve time = 0.0242073 s; This step = 0.00519724 s; Avg. per step = 0.01210365 s
Total Time                     : 0.211231649
""")
w("nyx", "run.log", """[STEP 1] Coarse TimeStep time: 0.044260893
checkPoint() time = 0.010267096 secs.
Write plotfile time = 0.009074775  seconds
[STEP 2] Coarse TimeStep time: 0.035151909
Run time = 0.467080722
""")
w("nekrs", "run.log", """initialization took 43.4189 s
>>> runtime statistics (step= 30  totalElapsed= 44.2363s):
name                    time          abs%  rel%  calls
  solve                 8.17425e-01s  100.0
    min                 2.55411e-02s
    max                 6.43613e-02s
    flops/rank          2.60077e+10
    checkpointing       8.04749e-03s   1.0        3
    pressureSolve       4.86723e-01s  59.5        30
      preconditioner    4.34207e-01s  53.1  89.2  121

occa max memory usage:     32311330 bytes
""")
w("specfem3d", "run.log", "solver done\n")
w("specfem3d", "app/smoke.np2/OUTPUT_FILES/output_solver.txt", """ Elapsed time in seconds =    7.6978100000002492E-004
 Time steps done =            5  out of         5000
 Elapsed time in seconds =   0.56034900799999998
 Time steps done =         5000  out of         5000
 Writing the seismograms in parallel took    3.92535515E-02  seconds
 Total elapsed time in seconds =   0.59985799399999995
""")
w("exaca", "run.log", """Time spent initializing data = 0.0388126 s
Time spent performing CA calculations = 1.6139 s
Time spent collecting and printing output data = 0.0607356 s
Max/min rank time in CA cell capture = 0.760942 / 0.734621 s
""")
w("qmcpack", "run.log", """Stack timer profile
Timer         Inclusive_time  Exclusive_time  Calls       Time_per_call
Total          289.5648     0.9414              1     289.564786534
  DMCBatched   277.1801     1.1801              1     277.180135472
    DMCBatched::Production   270.0000     2.0000              1     270.000000000
      DMCBatched::RunSteps   260.0000     10.0000           2500     0.104000000
    DMCBatched::Startup   6.0000     6.0000              1     6.000000000
  Startup        0.5827     0.5827              1       0.582700529
  VMCBatched    10.8606    0.8606              1      10.860585595
    VMCBatched::Production   10.0000     10.0000              1     10.000000000

QMCPACK execution completed successfully
""")
w("dftfe", "run.log", """Wall time for the above scf iteration: 8.87e-01 seconds
Wall time for the above scf iteration: 2.51e-01 seconds
---------------MD STEP 0 ------------------
Time taken for updateAtomPositionsAndMoveMesh: 17.94307
Wall time for the above scf iteration: 0.38868 seconds
Wall time for the above scf iteration: 0.25 seconds
---------------MD STEP 1 ------------------
Time taken for updateAtomPositionsAndMoveMesh: 17.75514
Wall time for the above scf iteration: 0.3 seconds
---------------MD STEP 2 ------------------
DFT-FE Program ends. Elapsed wall time since start of the program: 218.90105 seconds.
""")
w("cp2k", "run.log", "cp2k done\n")
w("cp2k", "app/h2o64.smoke.np2.t8/H2O-64-1.ener", """#     Step Nr.          Time[fs]        Kin.[a.u.]          Temp[K]            Pot.[a.u.]        Cons Qty[a.u.]        UsedTime[s]
         0            0.000000         0.272187779       300.000000000     -1101.031005889     -1100.758818110         0.000000000
         1            0.500000         0.273748897       301.720633893     -1101.039289568     -1100.765540671        17.978639553
         2            1.000000         0.273748897       301.720633893     -1101.039289568     -1100.765540671         2.260197000
         3            1.500000         0.273748897       301.720633893     -1101.039289568     -1100.765540671         2.291654000
""")
PY
pycheck "13d: every extractor reads its application's own timer from synthetic evidence" <<'PY'
import os, sys
sys.path.insert(0, os.environ["TOOLS"])
import apptimers
T = os.environ["TMP"] + "/l3"
want = {"lammps": (0.0709434, 100), "sparta": (0.0411685, 100), "warpx": (0.0242073, 2),
        "nyx": (0.044260893 + 0.035151909, 2), "nekrs": (0.817425 - 0.00804749, 30),
        "specfem3d": (0.560349008, 5000), "exaca": (1.6139, None), "cp2k": (2.260197 + 2.291654, 2),
        "qmcpack": (270.0, 2500), "dftfe": (17.94307 + 17.75514 + 0.38868 + 0.25 + 0.3, 2)}
bad = []
for app, (wall, steps) in want.items():
    try:
        r = apptimers.extract(app, os.path.join(T, app))
    except Exception as exc:
        bad.append(f"{app}: {exc}")
        continue
    if abs(r["wall_s"] - wall) > 1e-9 or r["steps"] != steps:
        bad.append(f"{app}: got {r['wall_s']} / {r['steps']}, want {wall} / {steps}")
r = apptimers.extract("nekrs", os.path.join(T, "nekrs"))
if abs(r["excluded_s"] - 0.00804749) > 1e-12 or r["setup_s"] != 43.4189:
    bad.append(f"nekrs excluded/setup {r['excluded_s']} {r['setup_s']}")
r = apptimers.extract("nyx", os.path.join(T, "nyx"))
if abs(r["excluded_s"] - (0.010267096 + 0.009074775)) > 1e-12:
    bad.append(f"nyx excluded {r['excluded_s']}")
missing = sorted(set(apptimers.TIMERS) - set(want))
if missing:
    bad.append(f"no synthetic evidence for {missing}")
print("ALLOK" if not bad else "\n".join(bad))
PY
pycheck "13e: a run without the timer fails loudly (no fallback to the process wall clock)" <<'PY'
import os, sys
sys.path.insert(0, os.environ["TOOLS"])
import apptimers
d = os.environ["TMP"] + "/l3empty"
os.makedirs(d, exist_ok=True)
open(os.path.join(d, "run.log"), "w").write("the application printed nothing useful\nLoop time of garbage\n")
bad = []
for app in sorted(apptimers.TIMERS):
    try:
        r = apptimers.extract(app, d)
        bad.append(f"{app}: returned {r['wall_s']} from a log without its timer")
    except apptimers.TimerMissing:
        pass
    except Exception as exc:
        bad.append(f"{app}: {type(exc).__name__} instead of TimerMissing: {exc}")
# an incomplete SPECFEM run (last report before the final step) is not a result
p = os.path.join(d, "app", "r", "OUTPUT_FILES")
os.makedirs(p, exist_ok=True)
open(os.path.join(p, "output_solver.txt"), "w").write(
    " Elapsed time in seconds =   0.5\n Time steps done =         500  out of         5000\n")
try:
    apptimers.extract("specfem3d", d)
    bad.append("specfem3d: incomplete run accepted")
except apptimers.TimerMissing:
    pass
print("ALLOK" if not bad else "\n".join(bad))
PY
pycheck "13f: every timer definition cites source lines that exist in the frozen tree (when materialized)" <<'PY'
import os, re, sys
sys.path.insert(0, os.environ["TOOLS"])
import apptimers
R = os.environ["REPO"]
bad, checked = [], 0
for app, t in apptimers.TIMERS.items():
    src = os.path.join(R, "level3", app, "src")
    if not os.path.isdir(src):
        continue
    for w in t.where:
        path, _, lines = w.partition(":")
        full = os.path.normpath(os.path.join(src, path))
        if not os.path.isfile(full):
            bad.append(f"{app}: {w}: no such file")
            continue
        n = sum(1 for _ in open(full, errors="replace"))
        last = max(int(x) for x in re.findall(r"\d+", lines)) if lines else 0
        if last > n:
            bad.append(f"{app}: {w}: file has {n} lines")
        checked += 1
    if not (t.definition and t.reduction and t.device_sync):
        bad.append(f"{app}: incomplete definition")
print("ALLOK" if not bad else "\n".join(bad))
PY
out="$(bash "$TOOLS/measure_level3.sh" --dry-run --collector none lammps 2>&1 | noise)"
if echo "$out" | /usr/bin/grep -q 'HPCPERF_L3_RUN_SUBDIR=run.timing-' && ! echo "$out" | /usr/bin/grep -q 'HPCPERF_ROI_LOG' \
   && echo "$out" | /usr/bin/grep -q "application's own timer" && echo "$out" | /usr/bin/grep -q 'HPCPERF_GPUS=2'; then
    ok "13g: a Level 3 dry run uses its own run-directory tree, no ROI log, 2 GPUs"
else bad "13g: level 3 dry run: $out"; fi

# summarize: a synthetic Level 3 raw run -> an app_timer record; a run without its timer -> app_timer_missing
mkraw3() {   # mkraw3 <run_id> <with timer: 1|0>
    local d="$TMP/raw3/level3/lammps/strong.s8/$1"
    mkdir -p "$d/clean.0"
    cp "$TMP/l3/lammps/run.log" "$d/clean.0/run.log"
    [ "$2" = 1 ] || echo "nothing" > "$d/clean.0/run.log"
    echo "hpcperf-launch: audit summary: 2 verified, 0 mismatch, 0 unverified (of 2 ranks)" >> "$d/clean.0/run.log"
    echo "start_ns=1000000000 end_ns=5000000000 rc=0" > "$d/clean.0/run.txt"
    printf '%s\n' "schema=hpcperf-timing-raw-2" "run_id=$1" "utc=2026-09-29T00:00:00Z" "level=3" "app=lammps" \
        "case=strong.s8" "backend=CUDA" "gpus=2" "case_env=HPCPERF_SCALE_MODE=strong" "argv=bash level3/lammps/run.sh CUDA" \
        "fom_name=Performance" "fom_unit=Matom-step/s" "fom_better=higher" "fom_source=stdout" \
        "fom_regex=^Performance: .*?([0-9.eE+-]+) Matom-step/s" "verify_vs_roi=none" "nvtx_roi=-" "region=app_timer" \
        "warmup_runs=0" "clean_runs=1" "profiled_runs=0" "collector=none" "platform_id=test-platform" "status=ok" > "$d/run_meta.txt"
}
mkraw3 20260929T000000Z-1 1; mkraw3 20260929T000001Z-2 0
python3 "$TOOLS/summarize.py" --raw-root "$TMP/raw3" --out-root "$TMP/res4" > "$TMP/sum4.log" 2>&1
pycheck "13h: summarize turns Level 3 evidence into the common record (roi.source app_timer, caveats, CSV, page)" <<'PY'
import csv, glob, json, os
T = os.environ["TMP"]
bad = []
recs = {os.path.basename(p): json.load(open(p)) for p in glob.glob(T + "/res4/level3/lammps/strong.s8/*.json")}
a, b = recs.get("20260929T000000Z-1.json"), recs.get("20260929T000001Z-2.json")
if not a or not b:
    bad.append(f"records: {sorted(recs)} ({open(T + '/sum4.log').read()[-300:]})")
else:
    R = a["roi"]
    if a["status"] != "ok" or R["source"] != "app_timer" or abs(R["wall_s"] - 0.0709434) > 1e-12 or R["steps"] != 100:
        bad.append(f"ok record: {a['status']} {R.get('source')} {R.get('wall_s')} {R.get('steps')}")
    if not R.get("definition") or not R.get("where") or a["app_timer"] is not None:
        bad.append("definition / where missing, or app_timer filled for Level 3")
    if a["fom"]["value"] != 45.106:
        bad.append(f"fom {a['fom']}")
    if not any("No numerical acceptance" in c for c in a["caveats"]):
        bad.append("no caveat for an input without numerical acceptance")
    if not any("not device-synchronized" in c for c in a["caveats"]):
        bad.append("LAMMPS section caveat missing")
    if abs(a["context"]["outside_region_s"] - (4.0 - 0.0709434)) > 1e-9:
        bad.append(f"outside_region_s {a['context'].get('outside_region_s')}")
    if b["status"] != "app_timer_missing":
        bad.append(f"record without timer has status {b['status']}")
rows = list(csv.DictReader(open(T + "/res4/summary_level3.csv")))
if len(rows) != 2 or {r["roi_source"] for r in rows} != {"app_timer"}:
    bad.append(f"summary_level3.csv rows {len(rows)}")
page = open(T + "/res4/report/index.html").read()
md = open(T + "/res4/report/README.md").read()
if 'data-level="3"' not in page or "## Level 3" not in md or "strong.s8" not in md:
    bad.append("page / README has no Level 3")
print("ALLOK" if not bad else "\n".join(bad))
PY

# an application the table does not profile by default: skipped with its reason, --profile-all overrides
out="$(bash "$TOOLS/measure_level3.sh" --dry-run qmcpack 2>&1 | noise)"
out2="$(bash "$TOOLS/measure_level3.sh" --dry-run --profile-all qmcpack 2>&1 | noise)"
if echo "$out" | /usr/bin/grep -q 'profiled=0 collector=none' && echo "$out" | /usr/bin/grep -q 'skipped by default' \
   && echo "$out2" | /usr/bin/grep -q 'profiled=1' && ! echo "$out2" | /usr/bin/grep -q 'skipped by default'; then
    ok "13i: QMCPACK is not profiled by default (reason shown), --profile-all profiles it"
else bad "13i: profile default: $out // $out2"; fi
d="$TMP/raw3/level3/lammps/strong.s8/20260929T000002Z-3"
mkdir -p "$d"; cp -r "$TMP/raw3/level3/lammps/strong.s8/20260929T000000Z-1/clean.0" "$d/"
sed -e 's/20260929T000000Z-1/20260929T000002Z-3/' "$TMP/raw3/level3/lammps/strong.s8/20260929T000000Z-1/run_meta.txt" > "$d/run_meta.txt"
echo "profile_skipped=no (a planted reason for the test)" >> "$d/run_meta.txt"
python3 "$TOOLS/summarize.py" --raw-root "$TMP/raw3" --out-root "$TMP/res5" --run-id 20260929T000002Z-3 --no-report > /dev/null 2>&1
if /usr/bin/grep -q 'Not profiled by default.*a planted reason for the test' "$TMP/res5/level3/lammps/strong.s8/20260929T000002Z-3.json" 2>/dev/null; then
    ok "13j: a record the table kept from the profiler carries the table's reason"
else bad "13j: no 'not profiled by default' caveat with the reason"; fi

echo
echo "tools/timing tests: $pass passed, $failn failed, $skipn skipped"
[ "$failn" -eq 0 ]
