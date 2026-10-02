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
nbad="$(printf '%s\n' "$rows" | awk -F'\t' 'NF!=20' | wc -l)"
nbm="$(ls -d "$REPO"/level1/*/CMakeLists.txt 2>/dev/null | wc -l)"
if [ "$nbad" -eq 0 ] && [ "$n1" -ge "$nbm" ]; then ok "1c: level 1 resolves to $n1 cases of 20 fields ($nbm benchmarks)"
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
echo "=== 1r: registered inputs (inputs.yaml -> generated cases/level<N>_registry.tsv)"
pycheck "1r1: registry tables: one case per registered input (counts from the registry), case == input_id, no drift" <<'PY'
import os, subprocess, sys, glob
sys.path.insert(0, os.environ["TOOLS"])
import cases, gen_registry_cases as g
bad = []
repo = os.environ["REPO"]
for lvl, fn in ((1, cases.level1_registry_rows), (2, cases.level2_registry_rows), (3, cases.level3_registry_rows)):
    rows, _ = fn("CUDA")
    n = 0
    for f in glob.glob(os.path.join(repo, f"level{lvl}", "*", "inputs.yaml")):
        out = subprocess.run([sys.executable, os.path.join(repo, "tools/inputs/hpcperf_inputs.py"), "list", os.path.dirname(f)],
                             capture_output=True, text=True).stdout
        n += len([l for l in out.splitlines() if l.strip()])
    if len(rows) != n:
        bad.append(f"level {lvl}: {len(rows)} registry cases for {n} registered inputs")
    if any(r["case"] != r["input_id"] or not r["input_id"] for r in rows):
        bad.append(f"level {lvl}: a registry case without its input id")
if g.check():
    bad.append("drift: " + "; ".join(g.check()))
print("ALLOK" if not bad else "\n".join(bad))
PY
pycheck "1r2: registry Level 1: a compile-time input runs its OWN binary (NPB class A/C), generated data per input id" <<'PY'
import os, sys, shlex
sys.path.insert(0, os.environ["TOOLS"])
import cases
rows, _ = cases.level1_registry_rows("CUDA")
by = {(r["app"], r["case"]): r for r in rows}
bad = []
for b in ("cg", "ep", "ft", "is", "mg"):
    for cls, d in (("class-a", "cuda-classA"), ("class-c", "cuda-classC"), ("class-b", "cuda")):
        exe = shlex.split(by[(b, cls)]["argv"])[0]
        if not exe.endswith(f"/build/{b}/{d}/{b}_cuda"):
            bad.append(f"{b}/{cls} runs {exe}")
for app, iid, data in (("aes", "plaintext-4mib", "input_4MB.hex"), ("aes", "plaintext-16mib", "input_16MB.hex"),
                       ("pagerank", "nodes4096", "4096.data"), ("pagerank", "nodes1024", "1024.data")):
    argv = shlex.split(by[(app, iid)]["argv"])
    if not any(a.endswith(data) and os.path.isabs(a) for a in argv):
        bad.append(f"{app}/{iid}: {data} not passed as an absolute path: {argv}")
print("ALLOK" if not bad else "\n".join(bad))
PY
pycheck "1r3: registry Level 2: selector allowed only as the registry declares it; stray / conflicting / hand-written selector and knobs refused" <<'PY'
import os, sys, shutil
sys.path.insert(0, os.environ["TOOLS"])
import cases
bad = []
def refused(fn, *a):
    try:
        fn(*a); return False
    except cases.CaseError:
        return True
rows, _ = cases.level2_registry_rows("CUDA")
reg = cases.registry_l2()
for r in rows:
    sel = reg[r["app"]][0]
    if r["env"] != f"{sel}={r['input_id']}":
        bad.append(f"{r['app']}/{r['case']}: env {r['env']!r}")
if "HPCPERF_XSBENCH_INPUT" not in cases.allowed_env("xsbench", ""):
    bad.append("the registry selector of xsbench is not an allowed input variable")
for internal in ("HPCPERF_INPUT_ARGS", "HPCPERF_INPUT_ID"):
    if internal in cases.allowed_env("xsbench", ""):
        bad.append(f"{internal} (set inside run.sh by the selector helper) is settable by a case")
kz = [r for r in rows if r["app"] == "kripke" and r["case"] == "z64-g64-q128"]
kd, _ = cases.level2_rows("CUDA")
kd = [r for r in kd if r["app"] == "kripke"]
if not refused(cases.refuse_undeclared, kd, {"HPCPERF_KRIPKE_INPUT": "z64-g64-q128"}):
    bad.append("a selector in the shell was not refused for the hand-written kripke case")
if not refused(cases.refuse_undeclared, kd, {"KRIPKE_ZONES": "8,8,8"}):
    bad.append("a registry knob in the shell was not refused")
if not refused(cases.refuse_undeclared, kz, {"HPCPERF_KRIPKE_INPUT": "z32-g32-q64"}):
    bad.append("a conflicting selector value in the shell was not refused")
try:
    cases.refuse_undeclared(kz, {"HPCPERF_KRIPKE_INPUT": "z64-g64-q128", "UNRELATED": "1"})
except cases.CaseError as e:
    bad.append(f"an identical declared value was refused: {e}")
if not refused(cases.parse_env, "HPCPERF_GITHUB_TOKEN=x", "t"):
    bad.append("a credential-looking variable was accepted")
tmp = os.path.join(os.environ["TMP"], "casedir_reg")
os.makedirs(tmp, exist_ok=True)
for f in ("level2_apps.tsv", "level2_registry.tsv"):
    shutil.copy(os.path.join(cases.CASES, f), tmp)
with open(os.path.join(tmp, "level2_cases.tsv"), "w") as f:
    f.write("kripke\tsneaky\t1\tHPCPERF_KRIPKE_INPUT=z64-g64-q128\t-\t-\t-\t-\n")
orig = cases.CASES
cases.CASES = tmp
if not refused(cases.level2_rows, "CUDA"):
    bad.append("a hand-written case that sets the registry selector was accepted")
cases.CASES = orig
print("ALLOK" if not bad else "\n".join(bad))
PY
pycheck "1r4: drift: a hand edit of a generated registry table is detected" <<'PY'
import os, sys, shutil
sys.path.insert(0, os.environ["TOOLS"])
import cases, gen_registry_cases as g
tmp = os.path.join(os.environ["TMP"], "casedir_drift")
os.makedirs(tmp, exist_ok=True)
for f in os.listdir(cases.CASES):
    shutil.copy(os.path.join(cases.CASES, f), tmp)
p = os.path.join(tmp, "level1_registry.tsv")
text = open(p).read().replace("build/cg/cuda-classC/cg_cuda", "build/cg/cuda/cg_cuda", 1)
open(p, "w").write(text)
cases.CASES = tmp
problems = g.check()
print("ALLOK" if any("level1_registry.tsv" in m for m in problems) else f"not detected: {problems}")
PY
out="$(bash "$TOOLS/measure_level1.sh" --registry --dry-run --no-profile cg/class-a cg/class-c 2>&1 || true)"
if printf '%s\n' "$out" | /usr/bin/grep -q 'build/cg/cuda-classA/cg_cuda' && printf '%s\n' "$out" | /usr/bin/grep -q 'build/cg/cuda-classC/cg_cuda' \
   && printf '%s\n' "$out" | /usr/bin/grep -q 'input    class-a (registry level1/cg/inputs.yaml'; then
    ok "1r5: registry dry run: class-a / class-c commands name their materialized binaries and the registry input"
else
    bad "1r5: registry dry run: $(printf '%s\n' "$out" | /usr/bin/grep -E 'command|input' | head -4 | tr '\n' ' ')"
fi

echo
pycheck "1r4: registry Level 3: one case per registered input, selector env (+ the build-variant variable of a ReaxFF input), GPUs from runtime_config, NVTX range / profile default from the apps table; stray, conflicting and hand-written selector refused" <<'PY'
import os, sys, shutil
sys.path.insert(0, os.environ["TOOLS"]); sys.path.insert(0, os.path.join(os.environ["REPO"], "tools", "inputs"))
import cases, hpcperf_inputs as hi
R = os.environ["REPO"]; bad = []
def refused(fn, *a):
    try:
        fn(*a); return False
    except cases.CaseError:
        return True
rows, apps = cases.level3_registry_rows("CUDA")
reg = cases.registry_l3()
for r in rows:
    sel = reg[r["app"]][0]
    doc = hi.load(os.path.join(R, "level3", r["app"])); inp = hi.get_input(doc, r["input_id"])
    variant = (inp.get("params") or {}).get("variant")
    want = f"{sel}={r['input_id']}" + (f";HPCPERF_LAMMPS_VARIANT={variant}" if variant and variant != "default" else "")
    if r["env"] != want or r["case"] != r["input_id"] or r["level"] != "3":
        bad.append(f"{r['app']}/{r['case']}: env {r['env']!r}, want {want!r}")
    if r["gpus"] != str((inp.get("runtime_config") or {}).get("gpus", 1)):
        bad.append(f"{r['app']}/{r['case']}: gpus {r['gpus']} != runtime_config")
    if r["nvtx_roi"] != apps[r["app"]]["nvtx_roi"] or r["profile"] != apps[r["app"]]["profile"] or r["verify_vs_roi"] != "outside":
        bad.append(f"{r['app']}/{r['case']}: nvtx/profile/verify not from the apps table")
if not any(r["app"] == "warpx" and r["nvtx_roi"] == "WarpX::Evolve()" for r in rows):
    bad.append("warpx rows lack the NVTX range")
if not any(r["app"] == "qmcpack" and r["profile"].startswith("no (") for r in rows):
    bad.append("qmcpack rows lack the no-profile default")
rx = [r for r in rows if r["app"] == "lammps" and r["case"].startswith("reaxff-")]
if len(rx) != 2 or any("HPCPERF_LAMMPS_VARIANT=reaxff" not in r["env"] for r in rx):
    bad.append(f"ReaxFF cases do not carry the build-variant variable: {[r['env'] for r in rx]}")
if any("HPCPERF_LAMMPS_VARIANT" in r["env"] for r in rows if r["app"] == "lammps" and not r["case"].startswith("reaxff-")):
    bad.append("a non-ReaxFF LAMMPS case carries a build-variant variable")
if "HPCPERF_LAMMPS_INPUT" not in cases.allowed_env("lammps", "", level=3):
    bad.append("the Level 3 registry selector is not an allowed input variable")
lj = [r for r in rows if r["app"] == "lammps" and r["case"] == "lj-32k"]
hand, _ = cases.level3_rows("CUDA"); hand = [r for r in hand if r["app"] == "lammps"]
if not refused(cases.refuse_undeclared, hand, {"HPCPERF_LAMMPS_INPUT": "lj-32k"}):
    bad.append("a Level 3 selector in the shell was not refused for the hand-written lammps case")
if not refused(cases.refuse_undeclared, lj, {"HPCPERF_LAMMPS_INPUT": "lj-2m"}):
    bad.append("a conflicting Level 3 selector value in the shell was not refused")
cp = [r for r in rows if r["app"] == "cp2k" and r["case"] == "h2o-64"]
if not refused(cases.refuse_undeclared, cp, {"HPCPERF_CP2K_SYSTEM": "32"}):
    bad.append("a Level 3 registry knob in the shell was not refused")
try:
    cases.refuse_undeclared(lj, {"HPCPERF_LAMMPS_INPUT": "lj-32k"})
except cases.CaseError as e:
    bad.append(f"an identical declared selector value was refused: {e}")
tmp = os.path.join(os.environ["TMP"], "casedir_reg3"); os.makedirs(tmp, exist_ok=True)
for f in ("level3_apps.tsv", "level3_registry.tsv", "level2_registry.tsv"):
    shutil.copy(os.path.join(cases.CASES, f), tmp)
with open(os.path.join(tmp, "level3_cases.tsv"), "w") as f:
    f.write("lammps\tsneaky\t1\tHPCPERF_LAMMPS_INPUT=lj-32k\t-\t-\t-\t-\t-\n")
orig = cases.CASES; cases.CASES = tmp
if not refused(cases.level3_rows, "CUDA"):
    bad.append("a hand-written Level 3 case that sets the registry selector was accepted")
cases.CASES = orig
print("ALLOK" if not bad else "\n".join(bad))
PY
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
# the executables the registry cases run (build/<benchmark>/cuda*/...), else a legacy build root
exes="$(/usr/bin/awk -F'\t' '!/^#/ {print $5}' "$TOOLS/cases/level1_registry.tsv" | sed "s#{REPO}#$REPO#" | sort -u)"
BR="registry"
if [ -z "$(for e in $exes; do [ -x "$e" ] && echo y && break; done)" ]; then
    BR=""; exes=""
    for cand in "$REPO/build/gcc13" "$REPO/build/all"; do
        [ -d "$cand/level1" ] && { BR="$cand"; exes="$(ls "$cand"/level1/*/*_cuda 2>/dev/null)"; break; }
    done
fi
if [ -z "$BR" ]; then
    skip "6d: no Level 1 build tree"
else
    nb=0; absent=0; missing=""
    for exe in $exes; do
        [ -x "$exe" ] || { absent=$((absent+1)); continue; }
        nb=$((nb+1))
        /usr/bin/grep -q 'hpcperf:roi' "$exe" || missing="$missing ${exe#$REPO/}"
    done
    [ "$nb" -gt 0 ] && [ -z "$missing" ] && ok "6d: all $nb built Level 1 binaries carry the markers ($BR; $absent not built)" \
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
if [c.get("kind") for c in D.get("campaigns", [{}])] != ["cases"]:
    bad.append(f"case-table records should give one campaign of kind 'cases': {[c.get('kind') for c in D.get('campaigns', [])]}")
D = D["campaigns"][0]
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
# an invalidated raw run (INVALIDATED.json) builds no record, and an existing record of it is not loaded
python3 "$TOOLS/summarize.py" --raw-root "$TMP/raw" --out-root "$TMP/res_inv" --no-report >/dev/null 2>&1
nrec="$(find "$TMP/res_inv" -name '*.json' -path '*/level*' | wc -l)"
for r in $(find "$TMP/raw" -name run_meta.txt -exec dirname {} \;); do echo '{"reason": "test"}' > "$r/INVALIDATED.json"; done
o1="$(python3 "$TOOLS/summarize.py" --raw-root "$TMP/raw" --out-root "$TMP/res_inv2" --no-report 2>&1 | noise | /usr/bin/grep '^summarize: json')"
o2="$(python3 "$TOOLS/summarize.py" --raw-root "$TMP/raw" --out-root "$TMP/res_inv" --csv-only --no-report 2>&1 | noise | /usr/bin/grep '^summarize: json')"
find "$TMP/raw" -name INVALIDATED.json -delete
case "$o1|$o2" in
    *"json_written=0 "*"invalidated_raw=$nrec "*"|"*" records=0 "*"invalidated_records=$nrec"*)
        ok "12g: $nrec invalidated raw runs build no record and their existing records are not loaded" ;;
    *) bad "12g: invalidated runs still summarized ($nrec runs): $o1 / $o2" ;;
esac

echo "=== 13: registry run verifier (verify_registry_runs.py) -- negative cases"
# A fake repository and records in the engine's layout: a Level 2 app whose registry names
# wk/SLD10.dat, and an MFEM-like app with option arguments. Each record is verified read-only.
pycheck "13a-13o: same-name file, relative/absolute path, controlled copy, dropped args, duplicate options" <<'PY'
import json, os, sys
sys.path.insert(0, os.environ["TOOLS"])
import verify_registry_runs as V
T = os.path.join(os.environ["TMP"], "vr"); R = os.path.join(T, "repo")
def w(p, s):
    os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").write(s)
w(f"{R}/level2/vl/wk/SLD10.dat", "registered deck\n")
w(f"{R}/level2/vl/other/SLD10.dat", "a different deck with the same name\n")
w(f"{R}/build/level2/vl/run/SLD10.dat", "registered deck\n")          # byte-identical copy
w(f"{R}/build/level2/vl/bad/SLD10.dat", "tampered copy\n")
w(f"{R}/build/level2/vl/vlp4d", ""); w(f"{R}/build/level2/rh/remhos", "")
os.makedirs(f"{R}/build/level2/vl/cwd", exist_ok=True)
deck_sha = V.sha(f"{R}/level2/vl/wk/SLD10.dat")
rules_plain = {1: {}, 2: {"vl": {"search_dirs": ["wk"]}, "rh": {}, "lw": {"last_wins": "test parser"}}}
rules_copy = {1: {}, 2: {"vl": {"search_dirs": ["wk"], "copies": {"wk/SLD10.dat": "build/level2/vl/"}}}}
n = [0]
def record(app, args, argv, cwd, files=None, sel="SEL", selval=None):
    n[0] += 1
    raw = f"{T}/raw/{n[0]}"; wl = {"input_id": "x", "args": args, "env": {}, "files_sha256": files or {}, "params": {}}
    ident = {"schema": "hpcperf-workload-identity-1", "benchmark": app, "input_id": "x", "level": 2, "complete": True,
             "selector": sel, "arg_files_sha256": {}, "workload": wl}
    for i in range(2):
        w(f"{raw}/clean.{i}/roi.{100 + i}", f"# hpcperf-roi-log 2\npid {100 + i}\nrank 0\nexe {argv[0]}\ncwd {cwd}\nargv {json.dumps(argv)}\nB 1 1\nE 2 2\n")
        w(f"{raw}/clean.{i}/run.log", "done\n")
    w(f"{raw}/workload_identity.json", json.dumps(ident))
    rec = {"schema": "hpcperf-timing-2", "level": 2, "app": app, "case": "x", "status": "ok",
           "registry": {"input_id": "x", "identity": ident, "identity_complete": True},
           "inputs": {"declared_env": {sel: selval or "x"}, "processes": []},
           "roi": {"runs_s": [1.0, 1.0]}, "provenance": {"raw_dir": os.path.relpath(raw, R), "git_commit": "t"}}
    p = f"{T}/rec/{n[0]}.json"; w(p, json.dumps(rec)); return p
bad = []
def expect(label, path, rules, verdict, needle=""):
    r = V.verify_record(path, os.path.realpath(R), rules)
    txt = " | ".join(r["problems"] + r["gaps"])
    if r["verdict"] != verdict or needle not in txt:
        bad.append(f"{label}: got {r['verdict']} ({txt}), want {verdict} /{needle}/")
exe, cwd = f"{R}/build/level2/vl/vlp4d", f"{R}/build/level2/vl/cwd"
files = {"wk/SLD10.dat": deck_sha}
# (1) same name, different content / location
expect("13a same-name other deck", record("vl", ["SLD10.dat"], [exe, f"{R}/level2/vl/other/SLD10.dat"], cwd, files),
       rules_plain, "FAIL", "not passed")
expect("13b bare same name in the process cwd (a different file)", record("vl", ["SLD10.dat"], [exe, "SLD10.dat"], f"{R}/level2/vl/other", files),
       rules_plain, "FAIL", "not passed")
# (2) the registered file by absolute and by cwd-relative path
expect("13c absolute path", record("vl", ["SLD10.dat"], [exe, f"{R}/level2/vl/wk/SLD10.dat"], cwd, files), rules_plain, "PASS")
expect("13d relative path", record("vl", ["SLD10.dat"], [exe, "../../../../level2/vl/wk/SLD10.dat"], cwd, files), rules_plain, "PASS")
# (3) a byte-identical copy counts only under a declared copy rule; a differing copy never
cp = record("vl", ["SLD10.dat"], [exe, f"{R}/build/level2/vl/run/SLD10.dat"], cwd, files)
expect("13e undeclared copy", cp, rules_plain, "FAIL", "not passed")
expect("13f declared identical copy", cp, rules_copy, "PASS")
expect("13g declared copy with other content", record("vl", ["SLD10.dat"], [exe, f"{R}/build/level2/vl/bad/SLD10.dat"], cwd, files),
       rules_copy, "FAIL", "")
# (4) registry arguments that never reach the program
rx = f"{R}/build/level2/rh/remhos"
expect("13h registry args dropped", record("rh", ["-rs", "1", "-dt", "0.02"], [rx, "-m", "cube.mesh", "-rs", "4", "-dt", "0.0025", "-pa"], cwd),
       rules_plain, "FAIL", "not passed")
# (5) run.sh defaults plus registered overrides: the duplicated option is refused ...
expect("13i duplicate option, parser not last-wins", record("rh", ["-rs", "1", "-dt", "0.02"], [rx, "-rs", "4", "-dt", "0.0025", "-rs", "1", "-dt", "0.02", "-pa"], cwd),
       rules_plain, "FAIL", "given 2 times")
# ... accepted after the fix that drops the overridden defaults ...
expect("13j overrides replace the defaults", record("rh", ["-rs", "1", "-dt", "0.02"], [rx, "-m", "cube.mesh", "-pa", "-rs", "1", "-dt", "0.02"], cwd),
       rules_plain, "PASS")
# ... and for a declared last-wins parser only when the LAST occurrence is the registry's
w(f"{R}/build/level2/lw/app", "")
lw = f"{R}/build/level2/lw/app"
expect("13k last-wins, registry last", record("lw", ["-s", "small"], [lw, "-s", "large", "-s", "small"], cwd), rules_plain, "PASS")
expect("13l last-wins, registry not last", record("lw", ["-s", "small"], [lw, "-s", "small", "-s", "large"], cwd), rules_plain, "FAIL", "last one")
# the selector must name the input; the binary must be the app's own
expect("13m wrong selector", record("rh", [], [rx], cwd, selval="y"), rules_plain, "FAIL", "selector")
expect("13n foreign binary", record("rh", [], [exe], cwd), rules_plain, "FAIL", "own binary")
# an env knob with no evidence rule is a gap, not a pass
p = record("rh", [], [rx], cwd)
d = json.load(open(p)); d["registry"]["identity"]["workload"]["env"] = {"HPCPERF_RH_KNOB": "3"}
json.dump(d, open(p, "w")); ri = json.load(open(f"{R}/{d['provenance']['raw_dir']}/workload_identity.json"))
ri["workload"]["env"] = {"HPCPERF_RH_KNOB": "3"}; json.dump(ri, open(f"{R}/{d['provenance']['raw_dir']}/workload_identity.json", "w"))
expect("13o unevidenced knob", p, rules_plain, "INSUFFICIENT", "HPCPERF_RH_KNOB")
print("ALLOK" if not bad else "\n".join(bad))
PY
# a changed input definition (remhos periodic-hexagon-p0: order 3 made explicit): a record of the old
# workload is SUPERSEDED (kept, never a result of the current input); the current workload with ONE -o
# passes; a second -o (run.sh default + registry) is refused -- MFEM is not last-wins
pycheck "13q-13s: changed definition -> SUPERSEDED; single effective -o; duplicated -o refused (real remhos registry)" <<'PY'
import json, os, sys
sys.path.insert(0, os.environ["TOOLS"]); sys.path.insert(0, os.path.join(os.environ["REPO"], "tools", "inputs"))
import verify_registry_runs as V, hpcperf_inputs as hi
R = os.environ["REPO"]; T = os.path.join(os.environ["TMP"], "vq")
doc = hi.load(os.path.join(R, "level2", "remhos")); inp = hi.get_input(doc, "periodic-hexagon-p0")
cur = hi.registry_identity(doc, inp)
bad = []
if cur["workload"]["args"][-2:] != ["-o", "3"] or cur["workload"]["params"].get("o") != 3:
    bad.append(f"registry does not state order 3: {cur['workload']['args']}")
old_wl = json.loads(json.dumps(cur["workload"])); old_wl["args"] = old_wl["args"][:-2]; old_wl["params"].pop("o", None)
exe = f"{R}/build/level2/remhos/cuda/remhos"
echo = "   --mesh /x/data/periodic-hexagon.mesh\n   --problem 0\n   --refine-serial 2\n   --order {o}\n   --time-step 0.005\n"
n = [0]
def rec(wl, argv, o):
    n[0] += 1; raw = f"{T}/raw/{n[0]}"; os.makedirs(raw, exist_ok=True)
    ident = dict(cur, workload=wl)
    for i in range(3):
        d = f"{raw}/clean.{i}"; os.makedirs(d, exist_ok=True)
        open(f"{d}/roi.{i}", "w").write(f"# hpcperf-roi-log 2\npid {i}\nrank 0\nexe {exe}\ncwd {R}/build/level2/remhos/cuda/run\nargv {json.dumps([exe] + argv)}\nB 1 1\nE 2 2\n")
        open(f"{d}/run.log", "w").write(echo.format(o=o))
    json.dump(ident, open(f"{raw}/workload_identity.json", "w"))
    r = {"schema": "hpcperf-timing-2", "level": 2, "app": "remhos", "case": "periodic-hexagon-p0", "status": "ok",
         "registry": {"identity": ident, "identity_complete": True}, "inputs": {"declared_env": {cur["selector"]: "periodic-hexagon-p0"}},
         "roi": {"runs_s": [1, 1, 1]}, "provenance": {"raw_dir": os.path.relpath(raw, R)}}
    p = f"{T}/{n[0]}.json"; json.dump(r, open(p, "w")); return p
rules = V.load_rules(os.path.join(os.environ["TOOLS"], "cases", "registry_evidence.yaml"))
base = ["-ho", "3", "-lo", "5", "-fct", "2", "-pa", "-d", "cuda", "-no-vis", "-m", f"{R}/level2/remhos/data/periodic-hexagon.mesh",
        "-p", "0", "-rs", "2", "-dt", "0.005", "-tf", "10"]
for label, wl, argv, o, want in [
        ("13q old order-2 record", old_wl, ["-o", "2"] + base, 2, "SUPERSEDED"),
        ("13r current workload, one -o 3", cur["workload"], base + ["-o", "3"], 3, "PASS"),
        ("13s run.sh -o 2 AND registry -o 3", cur["workload"], ["-o", "2"] + base + ["-o", "3"], 3, "FAIL")]:
    v = V.verify_record(rec(wl, argv, o), R, rules)
    if v["verdict"] != want:
        bad.append(f"{label}: {v['verdict']} ({v['problems'] + v['gaps']}), want {want}")
print("ALLOK" if not bad else "\n".join(bad))
PY
# a registered input whose run fails before the ROI (MiniEM bdot/blob today) stays a failure: NOT_RUN,
# never PASS, and summarize counts it as not ok
pycheck "13t: a run that aborts before the ROI is NOT_RUN, never PASS" <<'PY'
import json, os, sys
sys.path.insert(0, os.environ["TOOLS"])
import verify_registry_runs as V
T = os.path.join(os.environ["TMP"], "vt"); raw = os.path.join(T, "raw"); os.makedirs(os.path.join(raw, "clean.0"), exist_ok=True)
open(os.path.join(raw, "clean.0", "run.log"), "w").write("terminate called after throwing an instance of 'Teuchos::Exceptions::InvalidParameterName'\n")
open(os.path.join(raw, "clean.0", "run.txt"), "w").write("rc=134\n")
ident = {"benchmark": "miniem", "input_id": "maxwell-bdot-small", "complete": True, "selector": "HPCPERF_MINIEM_INPUT",
         "workload": {"args": [], "env": {}, "params": {}, "files_sha256": {}}}
rec = {"schema": "hpcperf-timing-2", "level": 2, "app": "miniem", "case": "maxwell-bdot-small", "status": "clean_failed",
       "registry": {"identity": ident, "identity_complete": True}, "inputs": {"declared_env": {"HPCPERF_MINIEM_INPUT": "maxwell-bdot-small"}},
       "roi": {"runs_s": [], "wall_s": None}, "provenance": {"raw_dir": raw}}
p = os.path.join(T, "r.json"); json.dump(rec, open(p, "w"))
v = V.verify_record(p, os.path.join(T, "norepo"), {1: {}, 2: {}})
print("ALLOK" if v["verdict"] == "NOT_RUN" else f"verdict {v['verdict']}: {v['problems']}")
PY
out="$(python3 "$TOOLS/verify_registry_runs.py" --repo "$TMP/vr/repo" "$TMP/vr/rec" 2>&1 | noise | tail -1)"
case "$out" in *"records)"*) ok "13p: the command line verifies a directory of records ($out)" ;;
               *) bad "13p: verify_registry_runs.py cli: $out" ;; esac

echo "=== 14: a registry dry run executes nothing"
# Not every run.sh honours HPCPERF_DRY_RUN (the direct-launch ones ignore it), so the dry run must
# never start run.sh or a benchmark at all. Shims for every way the engine starts a program
# (env -i, timeout, mpirun/mpiexec/srun) record a sentinel and exit without running anything.
SHIM="$TMP/shim"; SENT="$TMP/executed"; mkdir -p "$SHIM"
for t in env timeout mpirun mpiexec srun; do
    printf '#!/bin/sh\necho "%s $*" >> "%s"\nexit 97\n' "$t" "$SENT" > "$SHIM/$t"; chmod +x "$SHIM/$t"
done
rm -f "$SENT"
o1="$(PATH="$SHIM:$PATH" bash "$TOOLS/measure_level1.sh" --registry --dry-run --collector none \
      --raw-root "$TMP/dr/raw" --results-root "$TMP/dr/res" all 2>&1 | noise)"
o2="$(PATH="$SHIM:$PATH" bash "$TOOLS/measure_level2.sh" --registry --dry-run --collector none \
      --raw-root "$TMP/dr/raw" --results-root "$TMP/dr/res" all 2>&1 | noise)"
o3="$(PATH="$SHIM:$PATH" bash "$TOOLS/measure_level3.sh" --registry --dry-run --collector none \
      --raw-root "$TMP/dr/raw" --results-root "$TMP/dr/res" all 2>&1 | noise)"
n1="$(printf '%s\n' "$o1" | /usr/bin/grep -c '^    command ')"; n2="$(printf '%s\n' "$o2" | /usr/bin/grep -c '^    command ')"
n3="$(printf '%s\n' "$o3" | /usr/bin/grep -c '^    command ')"
r1="$(/usr/bin/grep -vc '^#' "$TOOLS/cases/level1_registry.tsv")"; r2="$(/usr/bin/grep -vc '^#' "$TOOLS/cases/level2_registry.tsv")"
r3="$(/usr/bin/grep -vc '^#' "$TOOLS/cases/level3_registry.tsv")"
if [ -e "$SENT" ]; then
    bad "14a: the registry dry run started a program: $(head -3 "$SENT" | tr '\n' ' ')"
elif [ "$n1" != "$r1" ] || [ "$n2" != "$r2" ] || [ "$n3" != "$r3" ]; then
    bad "14a: dry run planned $n1/$r1 Level 1, $n2/$r2 Level 2 and $n3/$r3 Level 3 registry cases"
elif [ -e "$TMP/dr/raw" ] || [ -e "$TMP/dr/res" ]; then
    bad "14a: the dry run wrote raw or result directories"
else
    ok "14a: registry dry run planned all $n1 Level 1 + $n2 Level 2 + $n3 Level 3 inputs, started nothing, wrote nothing"
fi
# positive control: the same shims DO see a real (non-dry) run -- which they stop before any program runs
rm -f "$SENT"
PATH="$SHIM:$PATH" bash "$TOOLS/measure_level1.sh" --registry --collector none --no-profile --no-summary \
    --clean-runs 1 --raw-root "$TMP/dr2/raw" --results-root "$TMP/dr2/res" daxpy/"$(/usr/bin/awk -F'\t' '!/^#/ && $1=="daxpy" {print $2; exit}' "$TOOLS/cases/level1_registry.tsv")" >/dev/null 2>&1
[ -s "$SENT" ] && ok "14b: positive control -- a non-dry run is caught by the shims ($(head -1 "$SENT" | cut -c1-40)...)" \
               || bad "14b: the shims did not see a non-dry run; 14a proves nothing"

echo "=== 15: registered-input report (registry_view.py + report.py)"
# Synthetic records of the REAL registry's remhos periodic-hexagon-p0 (current definition: -o 3) and
# miniem darcy-hex: an old-definition record (SUPERSEDED), an INVALIDATED one, a current 3-run
# record plus a 2-run adaptive extension of the same configuration, a newer 3-run record built from
# another binary (a separate measurement), and a failed MiniEM attempt.
pycheck "15a-15h, 15k: pooling, INVALIDATED/SUPERSEDED never current, failed input listed, nulls, determinism, vs previous only within one protocol" <<'PY'
import json, os, sys
sys.path.insert(0, os.environ["TOOLS"]); sys.path.insert(0, os.path.join(os.environ["REPO"], "tools", "inputs"))
import hpcperf_inputs as hi, report, registry_view as RV
R = os.environ["REPO"]; T = os.path.join(os.environ["TMP"], "rv"); root = os.path.join(T, "results")
doc = hi.load(os.path.join(R, "level2", "remhos")); cur = hi.registry_identity(doc, hi.get_input(doc, "periodic-hexagon-p0"))
old_wl = json.loads(json.dumps(cur["workload"])); old_wl["args"] = old_wl["args"][:-2]; old_wl["params"].pop("o", None)
exe = f"{R}/build/level2/remhos/cuda/remhos"
base = ["-ho", "3", "-lo", "5", "-fct", "2", "-pa", "-d", "cuda", "-no-vis", "-m", f"{R}/level2/remhos/data/periodic-hexagon.mesh",
        "-p", "0", "-rs", "2", "-dt", "0.005", "-tf", "10"]
PLAT = "test-platform"
def rec(run_id, wl, argv, runs, exe_sha="aa", commit="c0ffee", app="remhos", case="periodic-hexagon-p0", status="ok",
        invalid=False, sel="HPCPERF_REMHOS_INPUT", order=3, warm=0):
    raw = os.path.join(T, "raw", app, case, run_id); os.makedirs(raw, exist_ok=True)
    ident = dict(cur, workload=wl, benchmark=app, input_id=case, selector=sel)
    for i, v in enumerate(runs if status == "ok" else [None]):
        d = f"{raw}/clean.{i}"; os.makedirs(d, exist_ok=True)
        if status == "ok":
            open(f"{d}/roi.{i}", "w").write(f"# hpcperf-roi-log 2\npid {i}\nrank 0\nexe {exe}\ncwd {R}\nargv {json.dumps([exe] + argv)}\nB 1 1\nE 2 2\n")
        open(f"{d}/run.log", "w").write(f"   --mesh /x/data/periodic-hexagon.mesh\n   --problem 0\n   --refine-serial 2\n   --order {order}\n   --time-step 0.005\n")
    json.dump(ident, open(f"{raw}/workload_identity.json", "w"))
    if invalid:
        json.dump({"reason": "test: another workload"}, open(f"{raw}/INVALIDATED.json", "w"))
    import statistics
    r = {"schema": "hpcperf-timing-2", "level": 2, "app": app, "case": case, "status": status, "run_id": run_id,
         "utc": "2026-01-01T00:00:%02dZ" % int(run_id[-2:]), "platform": PLAT,
         "registry": {"input_id": case, "identity": ident, "identity_complete": True, "identity_sha256": "id-" + json.dumps(wl, sort_keys=True)[:40]},
         "inputs": {"declared_env": {sel: case}, "processes": [], "exe_sha256": exe_sha},
         "roi": {"runs_s": runs if status == "ok" else [], "wall_s": statistics.median(runs) if status == "ok" else None},
         "measurement": {"protocol": {"warmup_runs": warm, "clean_runs": len(runs), "profiled_runs": 0}, "collector": {"name": "none"}},
         "device": None, "provenance": {"raw_dir": os.path.relpath(raw, R), "git_commit": commit}, "caveats": []}
    os.makedirs(f"{root}/level2/{app}/{case}", exist_ok=True)
    json.dump(r, open(f"{root}/level2/{app}/{case}/{run_id}.json", "w"))
rec("run01", old_wl, ["-o", "2"] + base, [1.50, 1.52, 1.51], commit="old0001", order=2)          # SUPERSEDED
rec("run02", cur["workload"], ["-o", "2"] + base, [1.40, 1.41, 1.42], invalid=True, commit="bad0002", order=2)  # INVALIDATED
rec("run03", cur["workload"], base + ["-o", "3"], [2.72, 2.40, 2.38])                               # current, 3 runs
rec("run04", cur["workload"], base + ["-o", "3"], [2.38, 2.39])                                     # adaptive +2, same config
rec("run05", cur["workload"], base + ["-o", "3"], [9.0, 9.1, 9.2], exe_sha="bb")                    # other binary: separate
rec("run07", cur["workload"], base + ["-o", "3"], [9.5, 9.6, 9.7], exe_sha="bb", warm=1)            # another protocol: a new chain
rec("run08", cur["workload"], base + ["-o", "3"], [9.6, 9.7, 9.8], exe_sha="bb", warm=1)            # same protocol as run07
json.dump({"schema": "hpcperf-timing-measurement-groups-1", "groups": [{"id": "g1", "level": 2, "app": "remhos",
           "case": "periodic-hexagon-p0", "base_run_id": "run03", "extension_run_ids": ["run04"], "evidence": ["test"]}]},
          open(os.path.join(root, "measurement_groups.json"), "w"))                                  # the explicit link
ms = hi.load(os.path.join(R, "level2", "miniem")); mcur = hi.registry_identity(ms, hi.get_input(ms, "darcy-hex"))
rec("run06", mcur["workload"], [], [], app="miniem", case="darcy-hex", status="clean_failed", sel="HPCPERF_MINIEM_INPUT")
bad = []
b1 = report.build_bundle([root]); b2 = report.build_bundle([root])
if json.dumps(b1, sort_keys=True) != json.dumps(b2, sort_keys=True): bad.append("15a: two builds differ")
c = b1["campaigns"][0]
if c["kind"] != "registry": bad.append("15b: registry records not rendered as the registry view")
rows = {i["input_id"]: i for a in c["levels"]["2"] for i in a["inputs"]}
nreg = sum(1 for x in RV.registered_inputs(R) if x["level"] == 2)
if len(rows) != nreg: bad.append(f"15c: {len(rows)} Level 2 inputs listed, registry has {nreg}")
h = rows["periodic-hexagon-p0"]; m = h["cells"].get(PLAT)
if not m or m["set"]["run_ids"] != ["run08"]:
    bad.append(f"15d: current should be the newest configuration run08 (another binary / protocol is a separate measurement): {m and m['set']['run_ids']}")
sets = {tuple(s["run_ids"]): s for s in h["sets"]}
p = sets.get(("run03", "run04"))
if not p or p["n"] != 5 or abs(p["median"] - 2.39) > 1e-9: bad.append(f"15e: adaptive 3+2 not pooled into 5 samples: {p}")
if sets.get(("run01",), {}).get("verdict") != "SUPERSEDED" or sets[("run01",)]["current_definition"]: bad.append("15f: old definition not SUPERSEDED")
if sets[("run01",)].get("vs_previous") is not None or p.get("vs_previous") is not None:
    bad.append("15f: vs previous computed across workload definitions")
if not sets.get(("run05",)) or sets[("run05",)]["vs_previous"] is None: bad.append("15f: vs previous missing between same-workload measurements")
if sets.get(("run07",), {}).get("vs_previous") is not None: bad.append("15k: vs previous computed across protocols (warm-up 0 -> 1)")
if not sets.get(("run08",)) or sets[("run08",)]["vs_previous"] is None: bad.append("15k: vs previous missing between two measurements of the same protocol")
if "warm-up 1" not in sets.get(("run07",), {}).get("protocol_key", ""): bad.append(f"15k: protocol key not shown: {sets.get(('run07',), {}).get('protocol_key')}")
if m and m["set"]["run_ids"] != ["run08"]: bad.append(f"15d/15k: current should now be run08: {m['set']['run_ids']}")
if any(tuple(s["run_ids"]) == ("run02",) for s in h["sets"]) or not any(a["verdict"] == "INVALIDATED" for a in h["attempts"]):
    bad.append("15g: INVALIDATED record used as a measurement or not shown in the attempts")
f = rows["darcy-hex"]
if f["status"] != "RUN_FAILED" or any(f["cells"].values()): bad.append(f"15h: failed input: status {f['status']}")
if m and (m.get("device") is not None or m["roi"].get("profiler_inflation") is not None): bad.append("15h: no-profile fields not null")
if c["counts"]["roi_success"]["level2"] != 1: bad.append(f"15h: counts {c['counts']['roi_success']}")
report.write([root], os.path.join(T, "page"))
md = open(os.path.join(T, "page", "README.md")).read()
if "darcy-hex**: RUN_FAILED" not in md or "SUPERSEDED remhos / periodic-hexagon-p0" not in md: bad.append("15h: README lacks the failed / superseded entries")
print("ALLOK" if not bad else "\n".join(bad))
PY
if command -v node >/dev/null 2>&1; then
    printf '%s\n' '[{"name":"15i overview","level":"2","expect":["1 ROI timing SUCCESS"]},
 {"name":"15i hexagon","level":"2","app":"remhos","input":"periodic-hexagon-p0","platform":"test-platform","expect":["run05","SUPERSEDED","INVALIDATED","earlier definition","Where the process spends its time","Device activity inside the ROI","no collector observed this run","Runs of this input"]},
 {"name":"15i failed","level":"2","app":"miniem","input":"darcy-hex","platform":"test-platform","expect":["run failed","NOT_RUN"],"absent":["ROI (median of"]}]' > "$TMP/rv/checks.json"
    out="$(node "$HERE/page_smoke.js" "$TMP/rv/page/index.html" "$TMP/rv/checks.json" 2>&1)"
    [ $? -eq 0 ] && ok "15i: the page's own script renders the synthetic campaign (DOM shim, $(echo "$out" | grep -c '^ok') checks)" \
                 || bad "15i: page smoke: $(echo "$out" | grep FAIL | head -3 | tr '\n' ' ')"
else
    skip "15i: node not available for the page smoke test"
fi

echo "=== 16: pooling needs an explicit measurement group"
pycheck "16a-16h: linked 3+2 -> 5; unlinked 3+3 -> 2; other campaign -> 2; inconsistent or malformed group refused; duplicate record -> once" <<'PY'
import json, os, sys
sys.path.insert(0, os.environ["TOOLS"])
import registry_view as RV
T = os.path.join(os.environ["TMP"], "rg")
def rec(root, run_id, runs, exe="aa", ident="id1", warm=0, verdict="PASS"):
    return {"schema": "hpcperf-timing-2", "level": 2, "app": "app", "case": "in", "run_id": run_id, "status": "ok", "utc": run_id,
            "platform": "p", "registry": {"identity_sha256": ident, "identity": {"workload": {"k": ident}}},
            "inputs": {"exe_sha256": exe}, "provenance": {"git_commit": "c"},
            "measurement": {"protocol": {"warmup_runs": warm, "clean_runs": len(runs), "profiled_runs": 0}, "collector": {"name": "none"}},
            "roi": {"runs_s": runs}, "_root": os.path.realpath(root), "_verdict": verdict, "_path": os.path.join(root, run_id)}
def grp(root, base, ext):
    return {"id": f"{base}+{ext}", "level": 2, "app": "app", "case": "in", "base_run_id": base, "extension_run_ids": [ext],
            "_root": os.path.realpath(root)}
A, B = os.path.join(T, "campaignA"), os.path.join(T, "campaignB")
bad = []
sets, pr = RV.measurements([rec(A, "r1", [1.0, 1.2, 1.4]), rec(A, "r2", [1.1, 1.1])], [grp(A, "r1", "r2")])
if [len(m["samples"]) for m in sets] != [5] or not sets[0]["adaptive"] or pr:
    bad.append(f"16a linked 3+2: {[m['run_ids'] for m in sets]} {pr}")
sets, pr = RV.measurements([rec(A, "r1", [1.0, 1.1, 1.2]), rec(A, "r3", [1.0, 1.1, 1.2])], [])
if [len(m["samples"]) for m in sets] != [3, 3] or any(m["adaptive"] for m in sets):
    bad.append(f"16b unlinked, same configuration, same campaign: {[m['run_ids'] for m in sets]}")
sets, pr = RV.measurements([rec(A, "r1", [1.0, 1.1, 1.2]), rec(B, "r9", [1.0, 1.1, 1.2])], [grp(A, "r1", "r9")])
if [len(m["samples"]) for m in sets] != [3, 3] or not pr:
    bad.append(f"16c other campaign, same configuration (even with a cross-directory link): {[m['run_ids'] for m in sets]} {pr}")
for label, other in (("binary", dict(exe="bb")), ("workload", dict(ident="id2")), ("protocol", dict(warm=1))):
    sets, pr = RV.measurements([rec(A, "r1", [1.0, 1.1, 1.2]), rec(A, "r2", [1.1, 1.1], **other)], [grp(A, "r1", "r2")])
    if [len(m["samples"]) for m in sets] != [3, 2] or not pr or "not pooled" not in pr[0]:
        bad.append(f"16d linked but different {label}: {[m['run_ids'] for m in sets]} {pr}")
# a malformed member list is rejected as a whole, never repaired: no sample counted twice
def g2(base, ext):
    return {"id": "bad", "level": 2, "app": "app", "case": "in", "base_run_id": base, "extension_run_ids": ext, "_root": os.path.realpath(A)}
three, two = rec(A, "r1", [1.0, 1.2, 1.4]), rec(A, "r2", [1.1, 1.1])
for label, g, reason in (("16f extension listed twice", g2("r1", ["r2", "r2"]), "more than once"),
                         ("16g base listed as an extension", g2("r1", ["r1", "r2"]), "base run id is also listed"),
                         ("16h missing base", g2("", ["r2"]), "base_run_id"),
                         ("16h empty extension list", g2("r1", []), "extension_run_ids"),
                         ("16h extension list not a list", g2("r1", "r2"), "extension_run_ids")):
    sets, pr = RV.measurements([three, two], [g])
    if sorted(len(m["samples"]) for m in sets) != [2, 3] or not pr or reason not in pr[0] or any(m["adaptive"] for m in sets):
        bad.append(f"{label}: {[len(m['samples']) for m in sets]} {pr}")
# the same record under two roots (a copied results directory): loaded once
for root in (A, B):
    d = os.path.join(root, "level2", "app", "in"); os.makedirs(d, exist_ok=True)
    r = rec(root, "r1", [1.0, 1.1, 1.2])
    for k in ("_root", "_verdict", "_path"):
        r.pop(k)
    json.dump(r, open(os.path.join(d, "r1.json"), "w"))
recs, dups = RV.load_records([A, B])
if len(recs) != 1 or len(dups) != 1:
    bad.append(f"16e duplicate record: {len(recs)} loaded, {len(dups)} duplicates")
print("ALLOK" if not bad else "\n".join(bad))
PY

echo "=== 17: input-file identity through declared copies and logged reads (MiniEM-like)"
pycheck "17a-17e: copies with the registered content pass; changed deck / changed solver config / unlogged config do not" <<'PY'
import json, os, sys
sys.path.insert(0, os.environ["TOOLS"])
import verify_registry_runs as V
T = os.path.join(os.environ["TMP"], "fi"); R = os.path.join(T, "repo")
def w(p, s):
    os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").write(s)
w(f"{R}/level2/me/src/decks/deck.xml", "<deck/>\n"); w(f"{R}/level2/me/src/decks/solver.xml", "<solver/>\n")
w(f"{R}/build/level2/me/exe", "")
rules = {1: {}, 2: {"me": {"copy_dirs": {"src/decks": "build/level2/me/decks"}, "logged_reads": r"^Loading solver config from (\S+)$"}}}
files = {"src/decks/deck.xml": V.sha(f"{R}/level2/me/src/decks/deck.xml"), "src/decks/solver.xml": V.sha(f"{R}/level2/me/src/decks/solver.xml")}
n = [0]
def case(deck, solver, logged=True):
    n[0] += 1
    w(f"{R}/build/level2/me/decks/deck.xml", deck); w(f"{R}/build/level2/me/decks/solver.xml", solver)
    V._sha_cache.clear()
    raw = f"{T}/raw/{n[0]}"; cwd = f"{R}/build/level2/me/decks"
    ident = {"benchmark": "me", "input_id": "x", "complete": True, "selector": "SEL", "arg_files_sha256": {},
             "workload": {"input_id": "x", "args": [], "env": {}, "params": {}, "files_sha256": files}}
    for i in range(2):
        w(f"{raw}/clean.{i}/roi.{i}", f"# hpcperf-roi-log 2\npid {i}\nrank 0\nexe {R}/build/level2/me/exe\ncwd {cwd}\n"
          f"argv {json.dumps([R + '/build/level2/me/exe', '--inputFile=deck.xml'])}\nB 1 1\nE 2 2\n")
        w(f"{raw}/clean.{i}/run.log", "Loading solver config from solver.xml\n" if logged else "\n")
    w(f"{raw}/workload_identity.json", json.dumps(ident))
    r = {"schema": "hpcperf-timing-2", "level": 2, "app": "me", "case": "x", "status": "ok", "run_id": f"r{n[0]}",
         "registry": {"identity": ident, "identity_complete": True}, "inputs": {"declared_env": {"SEL": "x"}},
         "roi": {"runs_s": [1, 1]}, "provenance": {"raw_dir": os.path.relpath(raw, R)}}
    p = f"{T}/rec/{n[0]}.json"; w(p, json.dumps(r))
    return V.verify_record(p, R, rules)
bad = []
v = case("<deck/>\n", "<solver/>\n")
if v["verdict"] != "PASS" or not any("declared copy" in e for e in v["evidence"]):
    bad.append(f"17a identical copies: {v['verdict']} {v['problems'] + v['gaps']}")
v = case("<deck changed='1'/>\n", "<solver/>\n")
if v["verdict"] != "FAIL" or "deck.xml" not in " ".join(v["problems"]):
    bad.append(f"17b changed deck, same name and argv: {v['verdict']}")
v = case("<deck/>\n", "<solver changed='1'/>\n")
if v["verdict"] != "FAIL" or "solver.xml" not in " ".join(v["problems"]):
    bad.append(f"17c changed solver config: {v['verdict']}")
v = case("<deck/>\n", "<solver/>\n", logged=False)
if v["verdict"] != "INSUFFICIENT" or "solver.xml" not in " ".join(v["gaps"]):
    bad.append(f"17d solver config not named by the run: {v['verdict']}")
old = {"input_id": "x", "args": [], "env": {}, "params": {}, "files_sha256": {}}
if V.files_added(old, dict(old, files_sha256=files)) != files or V.files_added(old, dict(old, params={"a": 1})) is not None:
    bad.append("17e files_added does not isolate an added-files-only change")
print("ALLOK" if not bad else "\n".join(bad))
PY
if [ -f "$REPO/build/level2/miniem/cuda/decks/maxwell-large.xml" ]; then
pycheck "17f: MiniEM record measured before its files were registered: INSUFFICIENT without a complete, bound supplement" <<'PY'
import json, os, sys
sys.path.insert(0, os.environ["TOOLS"]); sys.path.insert(0, os.path.join(os.environ["REPO"], "tools", "inputs"))
import verify_registry_runs as V, hpcperf_inputs as hi
R = os.environ["REPO"]; T = os.path.join(os.environ["TMP"], "fm"); root = os.path.join(T, "results")
doc = hi.load(os.path.join(R, "level2", "miniem")); cur = hi.registry_identity(doc, hi.get_input(doc, "maxwell-large-weak48"))
ident = json.loads(json.dumps(cur)); ident["workload"]["files_sha256"] = {}          # as captured before the files were registered
exe = f"{R}/build/level2/miniem/cuda/PanzerMiniEM_BlockPrec"; cwd = f"{R}/build/level2/miniem/cuda/decks"
raw = os.path.join(T, "raw")
for i in range(3):
    d = f"{raw}/clean.{i}"; os.makedirs(d, exist_ok=True)
    open(f"{d}/roi.{i}", "w").write(f"# hpcperf-roi-log 2\npid {i}\nrank 0\nexe {exe}\ncwd {cwd}\nargv " + json.dumps([exe,
        "--inputFile=maxwell-large.xml", "--solver=MueLu", "--linAlgebra=Tpetra", "--numTimeSteps=3", "--x-elements=48",
        "--y-elements=48", "--z-elements=48", "--stacked-timer"]) + "\nB 1 1\nE 2 2\n")
    open(f"{d}/run.log", "w").write("Loading solver config from solverMueLu.xml\nLoading solver config from solverMueLuCuda.xml\n")
json.dump(ident, open(f"{raw}/workload_identity.json", "w"))
os.makedirs(f"{root}/level2/miniem/maxwell-large-weak48", exist_ok=True)
p = f"{root}/level2/miniem/maxwell-large-weak48/rX.json"
json.dump({"schema": "hpcperf-timing-2", "level": 2, "app": "miniem", "case": "maxwell-large-weak48", "status": "ok", "run_id": "rX",
           "registry": {"identity": ident, "identity_complete": True},
           "inputs": {"declared_env": {"HPCPERF_MINIEM_INPUT": "maxwell-large-weak48"}},
           "roi": {"runs_s": [1, 1, 1]}, "provenance": {"raw_dir": os.path.relpath(raw, R)}}, open(p, "w"))
rules = V.load_rules(V.DEFAULT_RULES)
bad = []
v = V.verify_record(p, R, rules)
if v["verdict"] != "INSUFFICIENT" or v["file_identity"] != "insufficient":
    bad.append(f"no supplement: {v['verdict']} {v['file_identity']}")
full = {"level": 2, "app": "miniem", "case": "maxwell-large-weak48", "run_id": "rX", "files_sha256": cur["workload"]["files_sha256"],
        "basis": "test basis", "evidence": ["test source"], "record_sha256": V.sha(p), "raw_dir": os.path.relpath(raw, R)}
def with_sup(entry, schema="hpcperf-file-identity-supplement-1"):
    json.dump({"schema": schema, "records": [entry]}, open(f"{root}/{V.SUPPLEMENT}", "w"))
    return V.verify_record(p, R, rules)
for label, entry, schema in (
        ("hashes differ from the registry", dict(full, files_sha256=dict(full["files_sha256"], **{"src/decks/solverMueLu.xml": "0" * 64})), None),
        ("identity + current hashes only, no basis / evidence", {k: full[k] for k in ("level", "app", "case", "run_id", "files_sha256")}, None),
        ("empty basis", dict(full, basis=" "), None),
        ("no evidence sources", dict(full, evidence=[]), None),
        ("not bound to the record (record_sha256)", dict(full, record_sha256="0" * 64), None),
        ("not bound to the raw runs (raw_dir)", dict(full, raw_dir="elsewhere"), None),
        ("wrong schema", full, "other-schema")):
    v = with_sup(entry, schema or "hpcperf-file-identity-supplement-1")
    if v["verdict"] == "PASS" or v["file_identity"] != "insufficient":
        bad.append(f"a supplement with {label} was accepted: {v['verdict']} {v['file_identity']}")
v = with_sup(full)
if v["verdict"] != "PASS" or v["file_identity"] != "supplement":
    bad.append(f"complete matching supplement: {v['verdict']} {v['problems'] + v['gaps']}")
if json.load(open(f"{raw}/workload_identity.json"))["workload"]["files_sha256"] != {}:
    bad.append("the stored identity was changed")
print("ALLOK" if not bad else "\n".join(bad))
PY
else
    skip "17f: MiniEM build decks not present"
fi

echo "=== 18: stability depends on the sample size"
pycheck "18a-18d: range rule below 10 samples, IQR rule from 10, one outlier in 20, two levels flagged" <<'PY'
import os, sys
sys.path.insert(0, os.environ["TOOLS"])
import registry_view as RV
bad = []
st, rule, iqr, two = RV.stability([1.0, 1.05, 1.12, 1.0, 1.01])            # 5 samples, range 12 %
if st or "max-min" not in rule: bad.append(f"18a 5 samples, range 12 %: {st} {rule}")
st, rule, iqr, two = RV.stability([1.0, 1.01, 1.02, 1.0, 1.01])            # 5 samples, range 2 %
if not st: bad.append("18a 5 samples, range 2 %: unstable")
s = [1.0 + 0.001 * i for i in range(19)] + [1.6]                            # 20 samples, one outlier
st, rule, iqr, two = RV.stability(s)
if not st or "IQR" not in rule or two: bad.append(f"18b 20 samples with one outlier: {st} {rule} two={two}")
s = [1.0, 1.01] * 5 + [1.3, 1.31] * 5                                       # 20 samples, two groups of 10
st, rule, iqr, two = RV.stability(s)
if st or not two: bad.append(f"18c two levels: stable={st} two={two}")
s = [1.0 + 0.002 * i for i in range(20)]                                    # 20 samples, 4 % steady spread
st, rule, iqr, two = RV.stability(s)
if not st or two: bad.append(f"18d steady 20: {st} {two}")
print("ALLOK" if not bad else "\n".join(bad))
PY

echo "=== 19: placement probe (probes/bindprobe.c) and the record's placement block"
BP="$TMP/bp"; mkdir -p "$BP"
if command -v cc >/dev/null 2>&1 && cc -O2 -shared -fPIC -o "$BP/bindprobe.so" "$TOOLS/probes/bindprobe.c" 2> "$BP/cc.log"; then
    # a process that wrote an ROI log records its placement; one that did not records nothing
    env -i PATH="$PATH" LD_PRELOAD="$BP/bindprobe.so" HPCPERF_PLACEMENT_LOG="$BP/bind" HPCPERF_ROI_LOG="$BP/roi" \
        /bin/sh -c 'echo "# fake" > "$HPCPERF_ROI_LOG.$$"; exit 0'
    env -i PATH="$PATH" LD_PRELOAD="$BP/bindprobe.so" HPCPERF_PLACEMENT_LOG="$BP/bind_plain" HPCPERF_ROI_LOG="$BP/roi_plain" \
        /bin/sh -c 'exit 0'
    mine="$(/usr/bin/grep Cpus_allowed_list /proc/self/status | cut -f2)"
    n=$(ls "$BP"/bind.* 2>/dev/null | wc -l); np=$(ls "$BP"/bind_plain.* 2>/dev/null | wc -l)
    if [ "$n" = 1 ] && [ "$np" = 0 ] && /usr/bin/grep -q "^cpus_allowed_end $mine\$" "$BP"/bind.* \
       && /usr/bin/grep -q "^roi_log 1$" "$BP"/bind.* && /usr/bin/grep -q "^task .* last_cpu [0-9]" "$BP"/bind.*; then
        ok "19a: the probe records the ROI process's CPU set ($mine), thread placement; a plain shell writes nothing"
    else
        bad "19a: bind files: roi=$n plain=$np; $(cat "$BP"/bind.* 2>/dev/null | head -20 | tr '\n' '|')"
    fi
else
    skip "19a: no C compiler for the placement probe"
fi
# summarize: bind.<pid> next to the ROI log -> placement block; consistent across clean runs
mkraw "$RAW/appd/default/r1" none 1 "Rate: 1"
echo "bind_probe=bindprobe.c@test" >> "$RAW/appd/default/r1/run_meta.txt"
for i in 0 1; do
    printf '# hpcperf-bind-log 1\npid 9\nexe /x/app\nroi_log 1\ncpu_start 3\ncpu_end 5\ncpus_allowed_start 0-7\nmems_allowed_start 0\ncpus_allowed_end 0-7\nmems_allowed_end 0\nthreads 2\nvoluntary_ctxt_switches 3\nnonvoluntary_ctxt_switches 1\ntask 9 cpus 0-7 last_cpu 5 name app\ntask 10 cpus 0-7 last_cpu 6 name cuda-EvtHandlr\ngpu minor 2 bus 0000:52:00.0\nenv CUDA_VISIBLE_DEVICES=GPU-abc\n' \
        > "$RAW/appd/default/r1/clean.$i/bind.9"
done
mkraw "$RAW/appe/default/r1" none 1 "Rate: 1"
python3 "$TOOLS/summarize.py" --raw-root "$TMP/raw" --out-root "$TMP/out19" --no-report >/dev/null 2>&1
pycheck "19b: placement block from bind logs (consistent), empty without the probe" <<'PY'
import json, os, glob
bad = []
d = json.load(open(glob.glob(os.path.join(os.environ["TMP"], "out19/level2/appd/default/*.json"))[0]))
p = d["placement"]
if p["probe"] != "bindprobe.c@test": bad.append(f"probe id {p['probe']}")
if len(p["clean_runs"]) != 2 or len(p["clean_runs"][0]) != 1: bad.append("one process per clean run expected")
q = p["clean_runs"][0][0]
if q["cpus_allowed"] != "0-7" or q["gpus"] != ["0000:52:00.0"] or q["last_cpus"] != [5, 6] or q["thread_cpusets"] != {"0-7": 2}:
    bad.append(f"process placement {q}")
if q["env"].get("CUDA_VISIBLE_DEVICES") != "GPU-abc": bad.append("env not kept")
s = p["summary"]
if not (s["consistent"] and s["cpus_allowed"] == ["0-7"] and s["gpus"] == ["0000:52:00.0"] and s["processes_recorded"] == 2):
    bad.append(f"summary {s}")
e = json.load(open(glob.glob(os.path.join(os.environ["TMP"], "out19/level2/appe/default/*.json"))[0]))["placement"]
if e["probe"] is not None or e["clean_runs"] != [[], []] or e["summary"]["consistent"]: bad.append(f"no-probe record {e}")
import csv
rows = {r["app"]: r for r in csv.DictReader(open(os.path.join(os.environ["TMP"], "out19/summary_level2.csv")))}
if rows["appd"]["placement_gpus"] != "0000:52:00.0" or rows["appd"]["placement_consistent"] != "1" or rows["appe"]["placement_consistent"] != "":
    bad.append(f"csv {rows['appd']['placement_gpus']} {rows['appd']['placement_consistent']!r} {rows['appe']['placement_consistent']!r}")
print("ALLOK" if not bad else "\n".join(bad))
PY

echo "=== 20: warm-up runs are recorded, never counted"
mkraw "$RAW/appf/default/r1" none 1 "Rate: 1"
mkdir -p "$RAW/appf/default/r1/warmup.0"
echo "start_ns=1000000000 end_ns=1000900000 rc=0" > "$RAW/appf/default/r1/warmup.0/run.txt"
printf '# hpcperf-roi-log 2\npid 9\nrank 0\nargv ["x"]\nB 0 1000100000\nE 2500000 1002600000\n' > "$RAW/appf/default/r1/warmup.0/roi.9"   # 2.5 ms, ~5x the clean runs
python3 "$TOOLS/summarize.py" --raw-root "$TMP/raw" --out-root "$TMP/out20" --no-report >/dev/null 2>&1
pycheck "20a: roi.warmup_runs_s holds the warm-up ROI; wall_s / runs_s unchanged" <<'PY'
import json, os, glob
bad = []
d = json.load(open(glob.glob(os.path.join(os.environ["TMP"], "out20/level2/appf/default/*.json"))[0]))
r = d["roi"]
if r["warmup_runs_s"] != [0.0025]: bad.append(f"warmup_runs_s {r['warmup_runs_s']}")
if abs(r["wall_s"] - 0.00055) > 1e-9 or len(r["runs_s"]) != 2: bad.append(f"clean statistics changed: {r['wall_s']} {r['runs_s']}")
e = json.load(open(glob.glob(os.path.join(os.environ["TMP"], "out20/level2/appe/default/*.json"))[0]))["roi"]
if e["warmup_runs_s"] != []: bad.append(f"no warm-up dir -> {e['warmup_runs_s']}")
print("ALLOK" if not bad else "\n".join(bad))
PY

echo "=== 21: the GPU a profiled run's kernels executed on is joined per process"
pycheck "21a: nsys gpus_used follows (pid, cudaId) -> gpuId, not the ordinal alone" <<'PY'
import os, sqlite3, sys
sys.path.insert(0, os.environ["TOOLS"])
from collectors import nvidia_nsys
d = os.path.join(os.environ["TMP"], "t21", "prof"); os.makedirs(d, exist_ok=True)
db = sqlite3.connect(os.path.join(d, "trace.sqlite"))
db.executescript("""
create table TARGET_INFO_GPU (id int, busLocation text);
insert into TARGET_INFO_GPU values (0, '0000:43:00.0'), (1, '0000:52:00.0');
create table TARGET_INFO_CUDA_DEVICE (gpuId int, cudaId int, pid int);
insert into TARGET_INFO_CUDA_DEVICE values (0, 0, 4000), (1, 1, 4000), (1, 0, 5000);  -- 4000: a helper seeing both GPUs; 5000: the app under CUDA_VISIBLE_DEVICES=<52:00.0>
create table CUPTI_ACTIVITY_KIND_KERNEL (globalPid int, deviceId int);
insert into CUPTI_ACTIVITY_KIND_KERNEL values (((1 << 24) | 5000) << 24, 0), (((1 << 24) | 5000) << 24, 0);
""")
db.commit(); db.close()
t = nvidia_nsys.open(os.path.dirname(d) + "/prof")
info = t.info(); t.close()
bad = []
if info.get("gpus_used") != ["0000:52:00.0"]: bad.append(f"gpus_used {info.get('gpus_used')} (the helper's ordinal 0 must not win)")
if info.get("gpus_visible") != ["0000:43:00.0", "0000:52:00.0"]: bad.append(f"gpus_visible {info.get('gpus_visible')}")
print("ALLOK" if not bad else "\n".join(bad))
PY

echo "=== 13: Level 3 (the applications' own timers)"
rows3="$(python3 "$TOOLS/cases.py" resolve --level 3 --no-env-check all 2>&1 | noise)"
n3="$(printf '%s\n' "$rows3" | /usr/bin/grep -c .)"
nbad3="$(printf '%s\n' "$rows3" | awk -F'\t' 'NF!=20' | wc -l)"
napps3="$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import cases; print(len(cases.level3_apps_in_suite()))' "$TOOLS" 2>&1 | noise)"
if [ "$nbad3" -eq 0 ] && [ "$n3" -ge "$napps3" ] && [ "$napps3" -gt 0 ]; then
    ok "13a: level 3 resolves to $n3 cases of 20 fields ($napps3 applications)"
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
out="$(bash "$TOOLS/measure_level3.sh" --registry --dry-run --collector none lammps/lj-32k 2>&1 | noise)"
if echo "$out" | /usr/bin/grep -q 'HPCPERF_LAMMPS_INPUT=lj-32k' && echo "$out" | /usr/bin/grep -q 'HPCPERF_GPUS=1' \
   && echo "$out" | /usr/bin/grep -q 'registry level3/lammps/inputs.yaml; workload identity: ok' && echo "$out" | /usr/bin/grep -q 'apptimers.py: lammps'; then
    ok "13k: a Level 3 registry dry run sets the selector and the input's GPU count and captures the workload identity"
else bad "13k: level 3 registry dry run: $(echo "$out" | head -8 | tr '\n' ' ' | cut -c1-300)"; fi

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

echo "=== 22: --page-level keeps one level of a published page as a current campaign"
python3 "$TOOLS/report.py" --results-root "$TMP/out19" --out "$TMP/page22a" >/dev/null 2>&1
python3 "$TOOLS/report.py" --results-root "$TMP/out19" --page-level "$TMP/page22a/index.html:2" --out "$TMP/page22b" >/dev/null 2>&1
pycheck "22a: the page's level 2 becomes a second, non-historical campaign; the Markdown names its source page" <<'PY'
import json, os, re
bad = []
t = open(os.path.join(os.environ["TMP"], "page22b", "index.html")).read()
d = json.loads(re.search(r'<script type="application/json" id="timing-data">(.*?)</script>', t, re.S).group(1).replace("<\\/", "</"))
c = d["campaigns"]
if len(c) != 2: bad.append(f"{len(c)} campaigns")
else:
    x = c[1]
    if x.get("historical") or x.get("kind") != "cases" or list(x["levels"].keys()) != ["2"]: bad.append(f"campaign {x.get('historical')} {x.get('kind')} {list(x['levels'].keys())}")
    if x.get("source_page") != "index.html" or x.get("level_only") != "2" or not x.get("records"): bad.append(f"provenance {x.get('source_page')} {x.get('level_only')} {x.get('records')}")
    if not str(x.get("title", "")).startswith("Level 2"): bad.append(f"title {x.get('title')}")
md = open(os.path.join(os.environ["TMP"], "page22b", "README.md")).read()
if "as published in index.html" not in md: bad.append("markdown lacks the source note")
print("ALLOK" if not bad else "\n".join(bad))
PY

echo "=== 23: Level 2 CPU-binding policy (cases/level2_binding.tsv, --bind-policy)"
be_h="$(python3 "$TOOLS/cases.py" binding-env hipbone 2>&1 | noise | tr '\n' ' ')"
be_a="$(python3 "$TOOLS/cases.py" binding-env amg2023 2>&1 | noise | tr '\n' ' ')"
be_x="$(python3 "$TOOLS/cases.py" binding-env xsbench 2>&1 | noise | tr '\n' ' ')"
if [ "$be_h" = "HPCPERF_CPUS_PER_RANK=4 OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=close " ] \
   && [ "$be_a" = "HPCPERF_CPUS_PER_RANK=1 " ] && [ -z "$be_x" ]; then
    ok "23a: hipBone gets PE=4 + pinned OpenMP threads, a single-host-thread MPI app PE=1, a direct-exec app nothing"
else bad "23a: binding env hipbone=[$be_h] amg2023=[$be_a] xsbench=[$be_x]"; fi
o_e="$(bash "$TOOLS/measure_level2.sh" --dry-run --collector none hipbone/default 2>&1 | noise)"
o_r="$(bash "$TOOLS/measure_level2.sh" --dry-run --collector none --bind-policy runtime hipbone/default 2>&1 | noise)"
if echo "$o_e" | /usr/bin/grep -q 'protocol=warmup:1,clean:3,profiled:1' && echo "$o_e" | /usr/bin/grep -q 'bind_policy=explicit' \
   && echo "$o_e" | /usr/bin/grep -q 'HPCPERF_CPUS_PER_RANK=4' && echo "$o_e" | /usr/bin/grep -q 'OMP_PROC_BIND=close' \
   && echo "$o_r" | /usr/bin/grep -q 'bind_policy=runtime' && ! echo "$o_r" | /usr/bin/grep -q 'HPCPERF_CPUS_PER_RANK'; then
    ok "23b: Level 2 defaults are 1 warm-up + 3 clean + 1 profiled with explicit binding; --bind-policy runtime adds no binding variables"
else bad "23b: dry runs: $(echo "$o_e" | /usr/bin/grep -E 'protocol|env ' | head -2 | cut -c1-160 | tr '\n' ' ') / $(echo "$o_r" | /usr/bin/grep -E 'protocol' | cut -c1-160)"; fi
o_b="$(bash "$TOOLS/measure_level2.sh" --dry-run --collector none --bind-policy nowhere hipbone/default 2>&1 | noise)"
case "$o_b" in *"--bind-policy must be explicit or runtime"*) ok "23c: an unknown binding policy is refused" ;;
               *) bad "23c: unknown policy not refused: $(echo "$o_b" | head -2 | tr '\n' ' ')" ;; esac
mkdir -p "$TMP/bind"; cp "$TOOLS/cases/level2_binding.tsv" "$TMP/bind/keep.tsv"
printf 'zzz_not_an_app\tmpirun\t1\t1\tno\tplanted\n' >> "$TOOLS/cases/level2_binding.tsv"
o_c="$(python3 "$TOOLS/cases.py" check 2>&1 | noise)"; cp "$TMP/bind/keep.tsv" "$TOOLS/cases/level2_binding.tsv"
sed -i 's/^hipbone\tmpirun\t4\t4\tyes/hipbone\tmpirun\t4\t2\tyes/' "$TOOLS/cases/level2_binding.tsv"
o_d="$(python3 "$TOOLS/cases.py" check 2>&1 | noise)"; cp "$TMP/bind/keep.tsv" "$TOOLS/cases/level2_binding.tsv"
if echo "$o_c" | /usr/bin/grep -q 'zzz_not_an_app, which is not a level2 application' \
   && echo "$o_d" | /usr/bin/grep -q 'cpus_per_rank must equal host_threads'; then
    ok "23d: cases.py check refuses a binding row without an application and idle bound cores"
else bad "23d: $(echo "$o_c" | tail -1) / $(echo "$o_d" | tail -1)"; fi
d="$TMP/raw23/level2/x/default/20260930T000000Z-9"
mkraw "$d" none 1 "Rate: 5.0"
if [ -d "$d" ]; then
    printf 'bind_policy=explicit\nbind_launcher=mpirun\nbind_host_threads=4\nbind_cpus_per_rank=4\nbind_omp_pin=yes\nbind_env=HPCPERF_CPUS_PER_RANK=4;OMP_NUM_THREADS=4;OMP_PLACES=cores;OMP_PROC_BIND=close\n' >> "$TMP/raw23/level2/x/default/20260930T000000Z-9/run_meta.txt"
    python3 "$TOOLS/summarize.py" --raw-root "$TMP/raw23" --out-root "$TMP/res23" --no-report > /dev/null 2>&1
    pycheck "23e: the record carries placement.policy (kind, cpus_per_rank, env) and the CSV a placement_policy column" <<'PY'
import csv, glob, json, os
bad = []
rs = glob.glob(os.path.join(os.environ["TMP"], "res23", "level2", "x", "default", "*.json"))
if not rs: bad.append("no record")
else:
    p = json.load(open(rs[0]))["placement"]["policy"]
    if p.get("kind") != "explicit" or p.get("cpus_per_rank") != 4 or p.get("omp_pin") != "yes" or "OMP_PLACES=cores" not in p.get("env", []):
        bad.append(f"policy {p}")
    rows = list(csv.DictReader(open(os.path.join(os.environ["TMP"], "res23", "summary_level2.csv"))))
    if not rows or rows[0].get("placement_policy") != "explicit:PE=4:omp_pin": bad.append(f"csv {rows and rows[0].get('placement_policy')}")
print("ALLOK" if not bad else "\n".join(bad))
PY
else
    bad "23e: the synthetic Level 2 raw run was not created"
fi


# ---- 24. campaign notes carried by several results roots are shown once (annotations.json) --------------------
N24="$(mktemp -d)"; mkdir -p "$N24/r1" "$N24/r2"
printf '{"schema":"hpcperf-timing-annotations-1","campaign":{"a":"x"},"notes":["shared note","only in r1"],"inputs":[]}\n' > "$N24/r1/annotations.json"
printf '{"schema":"hpcperf-timing-annotations-1","campaign":{"b":"y"},"notes":["shared note","only in r2"],"inputs":[]}\n' > "$N24/r2/annotations.json"
if python3 - "$HERE/.." "$N24" <<'PY'
import sys; sys.path.insert(0, sys.argv[1]); import registry_view as RV
ann, meta = RV.load_annotations([sys.argv[2] + "/r1", sys.argv[2] + "/r2"])
assert meta["notes"] == ["shared note", "only in r1", "only in r2"], meta["notes"]
assert meta["campaign"] == {"a": "x", "b": "y"}
PY
then ok "24: a note present in two roots' annotations.json is listed once, root-specific notes kept in root order"; else bad "24: duplicate campaign notes"; fi
rm -rf "$N24"


# ---- 25. Level 3 run verification: manifest, launcher and placement evidence (registered lammps/lj-32k, synthetic raw) ----
pycheck "25: Level 3 verification PASSes on consistent evidence and FAILs on a wrong selector / rank count / GPU count / foreign or changed binary / deck / steps / audit; a missing manifest is INSUFFICIENT; app_timer_missing is judged on its evidence" <<'PY'
import json, os, sys
sys.path.insert(0, os.environ["TOOLS"]); sys.path.insert(0, os.path.join(os.environ["REPO"], "tools", "inputs"))
import verify_registry_runs as V, hpcperf_inputs as hi
R = os.environ["REPO"]; T = os.path.join(os.environ["TMP"], "vr3")
doc = hi.load(os.path.join(R, "level3", "lammps")); inp = hi.get_input(doc, "lj-32k"); ident = hi.registry_identity(doc, inp)
exe = os.path.join(R, "build", "level3", "lammps", "cuda", "lmp_kokkos_cuda")
if not os.path.isfile(exe) or not os.path.isfile(os.path.join(R, "level3", "lammps", "src", "bench", "in.lj")):
    print("ALLOK"); sys.exit()                       # LAMMPS not built or its source bundle not materialized here: nothing to hash / verify
exe_sha = V.sha(exe); rules = V.load_rules(V.DEFAULT_RULES)
n = [0]
def w(p, s):
    os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").write(s)
def record(manifest=None, status="ok", sel_val="lj-32k", gpus="1", steps=100, exe_rec=None, audit="1 verified, 0 mismatch, 0 unverified",
           buses=("0000:43:00.0",), with_manifest=True, binary=None):
    n[0] += 1; raw = f"{T}/raw/{n[0]}"; run = f"{raw}/clean.0"; binary = binary or exe
    derived = f"{raw}/in.lj.input.lj-32k"; w(derived, "run ${steps}\n")
    m = {"run_id": "r", "app": "lammps", "backend": "CUDA", "profile": "cuda", "variant": "default", "mode": "smoke", "input_id": "lj-32k",
         "deck": "bench/in.lj", "ranks": "1", "atoms": "32000", "steps": "100", "gpu_aware": "on", "exit_code": "0",
         "binary": binary, "binary_sha256": exe_sha, "input": derived, "input_sha256": V.sha(derived),
         "fingerprint_sha256": "f" * 64, "log": "x", "utc": "t"}
    m.update(manifest or {})
    if with_manifest:
        w(f"{run}/app/run.timing-r-c0/run_manifest.txt", "".join(f"{k}={v}\n" for k, v in m.items()))
    log = ("# LAMMPS CUDA profile=cuda: mode=smoke input=lj-32k deck=bench/in.lj ranks=1 box=20x20x20 fcc cells = 32000 atoms (32000/rank), 100 steps, gpu-aware=on, log=x\n"
           "hpcperf-launch: launch:    backend=CUDA site=x launcher=mpirun ranks=1 (one per GPU) on 1 node(s) [h], ranks/node=1\n"
           f"hpcperf-launch: audit summary: {audit} (of 1 ranks)\n")
    w(f"{run}/run.log", log)
    w(f"{run}/bind.1", "# hpcperf-bind-log 1\npid 1\nexe " + binary + "\nhost h\n" + "".join(f"gpu minor 1 bus {b}\n" for b in buses))
    w(f"{raw}/workload_identity.json", json.dumps(ident))
    rec = {"schema": "hpcperf-timing-2", "level": 3, "app": "lammps", "case": "lj-32k", "status": status, "run_id": f"r{n[0]}",
           "registry": {"input_id": "lj-32k", "identity": ident, "identity_complete": True},
           "inputs": {"declared_env": {"HPCPERF_LAMMPS_INPUT": sel_val}, "processes": [], "exe_sha256": exe_sha if exe_rec is None else exe_rec},
           "measurement": {"gpus": gpus, "protocol": {"warmup_runs": 0, "clean_runs": 1, "profiled_runs": 0}},
           "roi": {"runs_s": [0.02] if status == "ok" else [], "wall_s": 0.02 if status == "ok" else None, "steps": steps, "source": "app_timer"},
           "provenance": {"raw_dir": raw, "git_commit": "t"}}
    p = f"{T}/rec/{n[0]}.json"; w(p, json.dumps(rec)); return p
bad = []
def expect(label, path, verdict, needle=""):
    r = V.verify_record(path, R, rules); txt = " | ".join(r["problems"] + r["gaps"])
    if r["verdict"] != verdict or needle not in txt:
        bad.append(f"{label}: got {r['verdict']} ({txt[:160]}), want {verdict} /{needle}/")
expect("25a consistent evidence", record(), "PASS")
expect("25b wrong selector value", record(sel_val="lj-2m"), "FAIL", "selector")
expect("25c two ranks in the manifest", record({"ranks": "2"}), "FAIL", "ranks=")
expect("25d HPCPERF_GPUS 2", record(gpus="2"), "FAIL", "runtime_config.gpus")
expect("25e binary outside the repository", record({"binary": "/usr/bin/true", "binary_sha256": V.sha("/usr/bin/true")}, exe_rec=V.sha("/usr/bin/true"), binary="/usr/bin/true"), "FAIL", "own binary")
expect("25f manifest sha differs from the file", record({"binary_sha256": "deadbeef" * 8}, exe_rec="deadbeef" * 8), "FAIL", "no longer hashes")
expect("25g wrong deck in the manifest", record({"deck": "bench/in.eam"}), "FAIL", "deck=")
expect("25h wrong step count in the manifest", record({"steps": "50"}), "FAIL", "steps=")
expect("25i timer reports other steps", record(steps=50), "FAIL", "timer reports")
expect("25j audit mismatch", record(audit="0 verified, 1 mismatch, 0 unverified"), "FAIL", "audit")
expect("25k two GPUs held", record(buses=("0000:43:00.0", "0000:52:00.0")), "FAIL", "held 2")
expect("25l no manifest harvested", record(with_manifest=False), "INSUFFICIENT", "no run manifest")
expect("25m app_timer_missing judged on its evidence", record(status="app_timer_missing", steps=None), "PASS")
expect("25n app_timer_missing without a manifest is NOT_RUN", record(status="app_timer_missing", steps=None, with_manifest=False), "NOT_RUN")
print("ALLOK" if not bad else "\n".join(bad))
PY

# ---- 26. the registered Level 3 inputs on the page, in the Markdown twin and in registry_current.csv; an embedded page level as history ----
pycheck "26: a Level 3 registry record is a SUCCESS row with run verification, the page/README show it with the application-timer columns, registry_current.csv lists it, --page-level PATH:3:history is an earlier campaign, a NO_TIMED_REGION input is neither SUCCESS nor run failed" <<'PY'
import csv, json, os, sys
sys.path.insert(0, os.environ["TOOLS"]); sys.path.insert(0, os.path.join(os.environ["REPO"], "tools", "inputs"))
import verify_registry_runs as V, hpcperf_inputs as hi, registry_view as RV, report, summarize
R = os.environ["REPO"]; T = os.path.join(os.environ["TMP"], "rv3"); root = os.path.join(T, "results")
doc = hi.load(os.path.join(R, "level3", "lammps")); inp = hi.get_input(doc, "lj-32k"); ident = hi.registry_identity(doc, inp)
exe = os.path.join(R, "build", "level3", "lammps", "cuda", "lmp_kokkos_cuda")
if not os.path.isfile(exe) or not os.path.isfile(os.path.join(R, "level3", "lammps", "src", "bench", "in.lj")):
    print("ALLOK"); sys.exit()                       # LAMMPS not built or its source bundle not materialized here: nothing to hash / verify
exe_sha = V.sha(exe); PLAT = "test-platform"
def w(p, s):
    os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").write(s)
raw = f"{T}/raw/lammps/lj-32k/run01"; run = f"{raw}/clean.0"; derived = f"{raw}/in.lj.input.lj-32k"; w(derived, "run ${steps}\n")
m = {"run_id": "r", "app": "lammps", "backend": "CUDA", "profile": "cuda", "variant": "default", "mode": "smoke", "input_id": "lj-32k", "deck": "bench/in.lj",
     "ranks": "1", "atoms": "32000", "steps": "100", "gpu_aware": "on", "exit_code": "0", "binary": exe, "binary_sha256": exe_sha,
     "input": derived, "input_sha256": V.sha(derived), "fingerprint_sha256": "f" * 64, "log": "x", "utc": "t"}
w(f"{run}/app/run.timing-r-c0/run_manifest.txt", "".join(f"{k}={v}\n" for k, v in m.items()))
w(f"{run}/run.log", "# LAMMPS CUDA profile=cuda: mode=smoke input=lj-32k deck=bench/in.lj ranks=1 box=20x20x20 fcc cells = 32000 atoms (32000/rank), 100 steps, gpu-aware=on, log=x\n"
  "hpcperf-launch: launch:    backend=CUDA site=x launcher=mpirun ranks=1 (one per GPU) on 1 node(s) [h], ranks/node=1\nhpcperf-launch: audit summary: 1 verified, 0 mismatch, 0 unverified (of 1 ranks)\n")
w(f"{run}/bind.1", "# hpcperf-bind-log 1\npid 1\nexe " + exe + "\nhost h\ngpu minor 1 bus 0000:43:00.0\n")
w(f"{raw}/workload_identity.json", json.dumps(ident))
rec = {"schema": "hpcperf-timing-2", "level": 3, "app": "lammps", "case": "lj-32k", "status": "ok", "run_id": "run01", "utc": "2026-01-01T00:00:01Z",
       "platform": PLAT, "registry": {"input_id": "lj-32k", "identity": ident, "identity_complete": True, "identity_sha256": "id-l3"},
       "inputs": {"declared_env": {"HPCPERF_LAMMPS_INPUT": "lj-32k"}, "processes": [], "exe_sha256": exe_sha},
       "measurement": {"gpus": "1", "protocol": {"warmup_runs": 0, "clean_runs": 1, "profiled_runs": 0}, "collector": {"name": "none"}, "region": "app_timer"},
       "roi": {"runs_s": [0.024], "wall_s": 0.024, "steps": 100, "source": "app_timer", "definition": "d", "where": ["x:1"], "reduction": "r", "device_sync": "s", "parts": {}},
       "context": {"process_wall_s": 4.0}, "device": None, "fom": {"name": "Performance", "value": 45.1, "unit": "Matom-step/s", "status": "ok"},
       "placement": {"policy": {"kind": "application"}, "summary": {"gpus": ["0000:43:00.0"]}},
       "provenance": {"raw_dir": raw, "git_commit": "c0ffee"}, "caveats": []}
w(f"{root}/level3/lammps/lj-32k/run01.json", json.dumps(rec))
# a Level 3 input the registry declares NO_TIMED_REGION (no time-step loop): its record is app_timer_missing by construction
rec2 = dict(rec, app="cp2k", case="regtest-gpw-h2o-geoopt", status="app_timer_missing", run_id="run02", utc="2026-01-01T00:00:02Z",
            registry={"input_id": "regtest-gpw-h2o-geoopt", "identity": hi.registry_identity(hi.load(os.path.join(R, "level3", "cp2k")), hi.get_input(hi.load(os.path.join(R, "level3", "cp2k")), "regtest-gpw-h2o-geoopt")), "identity_complete": True, "identity_sha256": "id-c"},
            inputs={"declared_env": {"HPCPERF_CP2K_INPUT": "regtest-gpw-h2o-geoopt"}, "processes": [], "exe_sha256": None},
            roi={"runs_s": [], "wall_s": None, "steps": None, "source": "app_timer"}, fom=None, provenance={"raw_dir": f"{T}/raw/cp2k/none", "git_commit": "c0ffee"},
            caveats=["The application's timer was not found in the clean run"])
w(f"{root}/level3/cp2k/regtest-gpw-h2o-geoopt/run02.json", json.dumps(rec2))
bad = []
rows, recs, meta, orph = RV.current_view([root], R)
ntr = next((r for r in rows if r["level"] == 3 and r["benchmark"] == "cp2k" and r["input_id"] == "regtest-gpw-h2o-geoopt"), None)
if not ntr or ntr["status"] != "NO_TIMED_REGION" or ntr["current"] is not None:
    bad.append(f"26n: NO_TIMED_REGION input status {ntr and ntr['status']}")
row = next((r for r in rows if r["level"] == 3 and r["benchmark"] == "lammps" and r["input_id"] == "lj-32k"), None)
if not row or row["status"] != "SUCCESS" or row["run_verification"] != "PASS":
    bad.append(f"26a: row {row and (row['status'], row['run_verification'])}")
k = RV.counts(rows, recs, orph)
if k["roi_success"].get("level3") != 1 or k["registered_inputs"]["level3"] != 43 or "roi_not_supported_level3" in k:
    bad.append(f"26b: counts {k['roi_success']} {k['registered_inputs']}")
if sum(1 for r in rows if r["level"] == 3 and r["status"] == "NOT_MEASURED") != 41:
    bad.append("26c: the other 41 Level 3 inputs are not NOT_MEASURED")
if k.get("no_timed_region") != ["cp2k/regtest-gpw-h2o-geoopt"] or k["run_failed"] != []:
    bad.append(f"26o: counts no_timed_region {k.get('no_timed_region')} run_failed {k['run_failed']}")
if "no timed region by construction 1" not in report.md_status_line({"counts": k}):
    bad.append("26p: the Markdown status line does not count the no-timed-region input")
out = os.path.join(T, "page"); report.write([root], out)
b = report.build_bundle([root]); c = b["campaigns"][0]
app = next((a for a in c["levels"]["3"] if a["app"] == "lammps"), None)
cell = next((i for i in app["inputs"] if i["input_id"] == "lj-32k"), {}).get("cells", {}).get(PLAT) if app else None
if not cell or cell["roi"].get("source") != "app_timer" or cell["roi"].get("steps") != 100 or cell["measurement"].get("gpus") != "1":
    bad.append(f"26d: page cell {cell and (cell['roi'].get('source'), cell['roi'].get('steps'))}")
md = open(os.path.join(out, "README.md")).read(); html = open(os.path.join(out, "index.html")).read()
l3 = md[md.index("### Level 3"):]
if "| lammps | lj-32k | SUCCESS |" not in l3 or "timed region" not in l3 or "24 ms" not in l3 or "| PASS |" not in l3:
    bad.append("26e: the Markdown Level 3 table lacks the row / timer columns")
if "registered Level 1 / 2 / 3 inputs" not in md or 'data-level="3"' not in html:
    bad.append("26f: status line / page level 3 tab")
summarize.write_registry_current(root)
rows_csv = [r for r in csv.DictReader(open(os.path.join(root, "registry_current.csv"))) if r["level"] == "3"]
if len(rows_csv) != 43 or not any(r["input_id"] == "lj-32k" and r["status"] == "SUCCESS" and r["run_verification"] == "PASS" for r in rows_csv):
    bad.append(f"26g: registry_current.csv level 3 rows {len(rows_csv)}")
if not any(r["input_id"] == "regtest-gpw-h2o-geoopt" and r["status"] == "NO_TIMED_REGION" for r in rows_csv):
    bad.append("26q: registry_current.csv does not carry NO_TIMED_REGION")
h = report.load_page_level(os.path.join(out, "index.html") + ":3:history")
if not h["historical"] or "earlier campaign" not in h["title"] or h["level_only"] != "3":
    bad.append(f"26h: page-level history {h.get('historical')} {h.get('title')}")
cur = report.load_page_level(os.path.join(out, "index.html") + ":3")
if cur["historical"]:
    bad.append("26i: a page level without :history became historical")
# 26j: campaign metadata of several roots merges one level deep -- a Level 3 root adds protocol.level3 and keeps the Level 1/2 protocol
ra, rb = os.path.join(T, "annA"), os.path.join(T, "annB")
w(os.path.join(ra, RV.ANNOTATIONS), json.dumps({"schema": "hpcperf-timing-annotations-1", "campaign": {"id": "a", "title": "A", "protocol": {"level1": "P1", "level2": "P2"}}, "notes": ["n1"], "inputs": []}))
w(os.path.join(rb, RV.ANNOTATIONS), json.dumps({"schema": "hpcperf-timing-annotations-1", "campaign": {"title": "B", "protocol": {"level3": "P3"}, "level3": "L3 text"}, "notes": ["n1", "n2"], "inputs": []}))
_ann, meta = RV.load_annotations([ra, rb])
camp = meta["campaign"]
if camp.get("protocol") != {"level1": "P1", "level2": "P2", "level3": "P3"} or camp.get("title") != "B" or camp.get("id") != "a" or meta["notes"] != ["n1", "n2"]:
    bad.append(f"26j: campaign merge {camp} {meta['notes']}")
md3 = report.render_md_registry({"campaign": camp, "counts": k, "measured_from": None, "generated_from": None, "records": 1, "notes": [], "levels": {}})
if "- Level 3 protocol: P3" not in md3 or "- Level 1 protocol: P1" not in md3 or "- L3 text" not in md3:
    bad.append("26k: the Markdown twin does not list the Level 3 protocol line")
print("ALLOK" if not bad else "\n".join(bad))
PY

echo
echo "tools/timing tests: $pass passed, $failn failed, $skipn skipped"
[ "$failn" -eq 0 ]
