#!/usr/bin/env bash
# tools/golden/tests/test_scrub_provenance.sh -- CPU-only tests of tools/golden/scrub_provenance.py on fixture files in a
# temporary git repository (and, read-only, on the repository's committed references): only the declared provenance
# fields go, every key outside the provenance is byte-identical, the result is deterministic and idempotent, `scrubbed`
# names what was removed but never a removed value, unknown schemas / reformatted files / malformed blocks are refused
# untouched, and `verify --against` catches any change outside the declared fields. Section 5: the Level 2 CPU reference
# schemas made before the golden work (hipBone / miniWeather without a provenance block, NekBone / PENNANT with in-block
# free text, PENNANT's own writer, Branson's relative vs absolute log_kept_at), each in its real serialisation.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$TOOLS/../.." && pwd)"
S="$TOOLS/scrub_provenance.py"
PY="${PYTHON:-python3}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/scrub-tests.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
pass=0; fail=0
ok()  { echo "ok   $*"; pass=$((pass+1)); }
bad() { echo "FAIL $*"; fail=$((fail+1)); }
GIT() { git -C "$TMP/repo" -c user.email=t@t -c user.name=t "$@"; }
sha() { sha256sum "$1" | cut -c1-64; }

mkdir -p "$TMP/repo/level1/fake/reference" "$TMP/repo/level3/nyx/reference" "$TMP/repo/level2/remhos/reference/validate" \
         "$TMP/repo/level2/branson/reference/validate"
UUID="GPU-0a1b2c3d-1111-2222-3333-444455556666"; HOSTQ="node17.example.org"
"$PY" - "$TMP/repo" "$UUID" "$HOSTQ" <<'PY'
import json, sys
R, uuid, host = sys.argv[1:4]
golden = {"schema": "hpcperf-golden-1", "benchmark": "fake", "level": 1, "input_id": "n1000", "frozen": True,
          "workload": {"args": ["1000", "0.1"], "host": "kept: a workload key named host is not provenance"},
          "inputs_yaml_sha256": "ab" * 32, "spec": {"mode": "data", "outputs": []},
          "outputs": [{"name": "y", "dtype": "f64", "count": 1000, "bytes": 8000, "sha256": "cd" * 32,
                       "rule": {"metric": "max_abs", "tol": 1e-08, "d0_vs_cpu_reference": 1.0000000000000002e-09,
                                "tol_effective": 9.000000000000001e-09}, "blob": "sha256/cd/" + "cd" * 32 + ".bin",
                       "stats": {"min": -0.9999996829318346, "max": 0.1, "sum": 12.34, "nonfinite": 0}}],
          "provenance": {"made_utc": "2026-10-26T06:15:32Z", "host": host,
                         "gpu": {"name": "NVIDIA B200", "uuid": uuid, "driver": "595.58.03"}, "cuda_toolkit": "13.2.1",
                         "git_head": "01" * 20, "source_clean": True, "binary": "build/fake/cuda/fake",
                         "binary_sha256": "ef" * 32, "command": ["build/fake/cuda/fake", "1000", "0.1"], "gpu_run": None,
                         "env": {"HPCPERF_CPU_REFERENCE": "1"}, "store": "x"}}
open(f"{R}/level1/fake/reference/n1000.json", "w").write(json.dumps(golden, indent=2) + "\n")
nyx = {"schema": "hpcperf-nyx-frozen-reference-1", "benchmark": "nyx", "level": 3, "case": "minisb", "mode": "smoke", "steps": 10,
       "workload": {"grid": "32x32x32"}, "archive": {"sha256": "aa" * 32, "size": 10, "members": [{"path": "plt00010/Header", "size": 3, "sha256": "bb" * 32}]},
       "comparison": {"mesh": {"tol": 1e-08, "d0_max": 3.25e-13, "rel_tol_effective": 9.999674999999e-09}},
       "provenance": {"made_utc": "2026-10-27T09:19:00Z", "host": host, "gpu": {"index": "1", "name": "NVIDIA B200", "uuid": uuid, "driver": "595.58.03"},
                      "cuda": "13.2.1", "git_head": "02" * 20, "harness_dirty": [], "gpu_run": {"run_id": "r1", "dir": "build/level3/nyx/p/run.x/minisb.smoke.np1"},
                      "note": None}}
open(f"{R}/level3/nyx/reference/minisb.smoke.json", "w").write(json.dumps(nyx, indent=1) + "\n")
remhos = {"schema": "hpcperf-remhos-cpu-backend-reference-1", "benchmark": "remhos", "cases": {"c1": {"final_mass_u": 0.1191358858, "log_sha256": "11" * 32}},
          "note": f"the CUDA build (committed sources) on {host.split('.')[0]}; every value equals the earlier run (d0 = 0)",
          "reference_run": {"binary_sha256": "22" * 32, "host": host.split(".")[0], "cpu_model": "XEON 8570", "slurm_job": "2956114",
                            "log_kept_at": "/projects/someone/results/x.log (outside the repository)", "finished_utc": "2026-10-27T08:00:34Z"},
          "tolerances": {"mass_rel": 1e-12, "max_rel": 1e-10}}
open(f"{R}/level2/remhos/reference/validate/cpu-backend-pa.json", "w").write(json.dumps(remhos, indent=1, sort_keys=True) + "\n")
branson = {"schema": "hpcperf-branson-reference-1", "benchmark": "branson", "final_step": {"photons_transported": 50000, "energy": 1.5},
           "note": "CPU-only run, d0 = 0", "reference_run": {"allocation": f"{host.split('.')[0]}, Slurm job 2956114", "binary_sha256": "33" * 32,
           "log_kept_at": "/projects/someone/branson.log", "log_sha256": "44" * 32, "mpi_ranks": 1, "finished_utc": "2026-10-27T08:01:47Z"},
           "tolerances": {"energy_rel": 1e-12}}
open(f"{R}/level2/branson/reference/validate/marshak.json", "w").write(json.dumps(branson, indent=1, sort_keys=True) + "\n")
PY
( cd "$TMP/repo" && git init -q && GIT add -A && GIT commit -qm fixtures ) && ok "0: fixture repository (4 schemas) committed" || bad "0: git"
F1="$TMP/repo/level1/fake/reference/n1000.json"; F3="$TMP/repo/level3/nyx/reference/minisb.smoke.json"
F2="$TMP/repo/level2/remhos/reference/validate/cpu-backend-pa.json"; FB="$TMP/repo/level2/branson/reference/validate/marshak.json"
ALL=("$F1" "$F3" "$F2" "$FB")
declare -A H0; for f in "${ALL[@]}"; do H0[$f]="$(sha "$f")"; done

echo "=== 1: plan writes nothing; apply removes exactly the declared fields"
out="$("$PY" "$S" plan "${ALL[@]}" 2>&1)"; rc=$?
same=1; for f in "${ALL[@]}"; do [ "$(sha "$f")" = "${H0[$f]}" ] || same=0; done
[ "$rc" = 0 ] && [ "$same" = 1 ] && echo "$out" | /usr/bin/grep -q 'would change 4' && ok "1a: plan lists 4 files, writes none" || bad "1a: rc=$rc $out"
out="$("$PY" "$S" apply "${ALL[@]}" 2>&1)"; rc=$?
"$PY" - "$TMP/repo" "$UUID" "$HOSTQ" <<'PY' && [ "$rc" = 0 ] && ok "1b: apply: host / gpu.uuid / gpu.index / allocation / slurm_job / log_kept_at gone, platform kept, scrubbed lists the paths, no removed value anywhere in the files" || bad "1b: rc=$rc $out"
import json, sys
R, uuid, host = sys.argv[1:4]
g = json.load(open(f"{R}/level1/fake/reference/n1000.json")); p = g["provenance"]
assert "host" not in p and p["gpu"] == {"name": "NVIDIA B200", "driver": "595.58.03"}, p
assert p["scrubbed"]["removed"] == ["gpu.uuid", "host"] and p["scrubbed"]["rule"] == "hpcperf-provenance-scrub-1", p["scrubbed"]
assert g["workload"]["host"].startswith("kept"), "a key named host outside the provenance must stay"
assert list(p)[-1] == "scrubbed" and p["cuda_toolkit"] == "13.2.1" and p["git_head"] == "01" * 20 and p["binary_sha256"] == "ef" * 32
n = json.load(open(f"{R}/level3/nyx/reference/minisb.smoke.json"))["provenance"]
assert "host" not in n and n["gpu"] == {"name": "NVIDIA B200", "driver": "595.58.03"} and n["scrubbed"]["removed"] == ["gpu.index", "gpu.uuid", "host"], n
r = json.load(open(f"{R}/level2/remhos/reference/validate/cpu-backend-pa.json"))
assert sorted(r["reference_run"]["scrubbed"]["removed"]) == ["host", "log_kept_at", "slurm_job"] and r["reference_run"]["cpu_model"] == "XEON 8570"
assert r["note"] == "the CUDA build (committed sources); every value equals the earlier run (d0 = 0)", r["note"]
assert r["reference_run"]["scrubbed"]["redacted_text"] == ["note: host name"]
b = json.load(open(f"{R}/level2/branson/reference/validate/marshak.json"))["reference_run"]
assert sorted(b["scrubbed"]["removed"]) == ["allocation", "log_kept_at"] and b["mpi_ranks"] == 1 and "redacted_text" not in b["scrubbed"]
for f in ("level1/fake/reference/n1000.json", "level3/nyx/reference/minisb.smoke.json", "level2/remhos/reference/validate/cpu-backend-pa.json",
          "level2/branson/reference/validate/marshak.json"):
    t = open(f"{R}/{f}").read()
    for needle in (uuid, host, host.split(".")[0], "2956114", "/projects/someone"):
        assert needle not in t, (f, needle)
PY
"$PY" - "$TMP/repo" <<'PY' && ok "1c: every key outside the provenance block (and the remhos/branson free-text note) is identical in value, order and serialisation" || bad "1c"
import json, subprocess, sys
R = sys.argv[1]
for f, prov, ind, sk in (("level1/fake/reference/n1000.json", {"provenance"}, 2, False), ("level3/nyx/reference/minisb.smoke.json", {"provenance"}, 1, False),
                         ("level2/remhos/reference/validate/cpu-backend-pa.json", {"reference_run", "note"}, 1, True),
                         ("level2/branson/reference/validate/marshak.json", {"reference_run", "note"}, 1, True)):
    old = json.loads(subprocess.run(["git", "-C", R, "show", f"HEAD:{f}"], capture_output=True, text=True, check=True).stdout)
    new = json.load(open(f"{R}/{f}"))
    assert [k for k in old if k not in prov] == [k for k in new if k not in prov], f
    for k in old:
        if k not in prov:
            assert json.dumps(old[k]) == json.dumps(new[k]), (f, k)
    blank = lambda d: {k: ("<p>" if k in prov else v) for k, v in d.items()}
    assert json.dumps(blank(old), indent=ind, sort_keys=sk) == json.dumps(blank(new), indent=ind, sort_keys=sk), f
PY
declare -A H1; for f in "${ALL[@]}"; do H1[$f]="$(sha "$f")"; done
out="$("$PY" "$S" apply "${ALL[@]}" 2>&1)"; rc=$?
same=1; for f in "${ALL[@]}"; do [ "$(sha "$f")" = "${H1[$f]}" ] || same=0; done
[ "$rc" = 0 ] && [ "$same" = 1 ] && echo "$out" | /usr/bin/grep -q 'changed 0, unchanged 4' && ok "1d: idempotent: a second apply changes no byte" || bad "1d: rc=$rc $out"
mkdir -p "$TMP/det"; for f in "${ALL[@]}"; do GIT show "HEAD:${f#$TMP/repo/}" > "$TMP/det/$(basename "$f")"; done
"$PY" "$S" apply "$TMP/det"/*.json > /dev/null 2>&1
same=1; for f in "${ALL[@]}"; do [ "$(sha "$TMP/det/$(basename "$f")")" = "${H1[$f]}" ] || same=0; done
[ "$same" = 1 ] && ok "1e: deterministic: scrubbing another copy of the originals gives the same bytes" || bad "1e: copies differ"
out="$("$PY" "$S" verify --against HEAD "${ALL[@]}" 2>&1)"; rc=$?
[ "$rc" = 0 ] && [ "$(echo "$out" | /usr/bin/grep -c '^ok ')" = 4 ] && ok "1f: verify --against HEAD: 4 files are the deterministic scrub of HEAD, outside the provenance byte-identical" || bad "1f: rc=$rc $out"

echo
echo "=== 2: verify catches any other change"
cp "$F1" "$TMP/keep1"
"$PY" -c "import json,sys;p=sys.argv[1];d=json.load(open(p));d['outputs'][0]['rule']['tol_effective']=9.1e-09;open(p,'w').write(json.dumps(d,indent=2)+'\n')" "$F1"
out="$("$PY" "$S" verify --against HEAD "$F1" 2>&1)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q "key 'outputs' differs" && ok "2a: a changed tol_effective (outside the provenance) -> verify FAIL naming the key" || bad "2a: rc=$rc $out"
cp "$TMP/keep1" "$F1"
"$PY" -c "import json,sys;p=sys.argv[1];d=json.load(open(p));d['provenance']['git_head']='03'*20;open(p,'w').write(json.dumps(d,indent=2)+'\n')" "$F1"
out="$("$PY" "$S" verify --against HEAD "$F1" 2>&1)"; rc=$?
[ "$rc" = 1 ] && echo "$out" | /usr/bin/grep -q 'not the deterministic scrub' && ok "2b: a provenance change beyond the declared fields (git_head) -> verify FAIL" || bad "2b: rc=$rc $out"
cp "$TMP/keep1" "$F1"
"$PY" -c "import json,sys;p=sys.argv[1];d=json.load(open(p));d['provenance']['gpu']['driver']='1.0';open(p,'w').write(json.dumps(d,indent=2)+'\n')" "$F1"
out="$("$PY" "$S" verify --against HEAD "$F1" 2>&1)"; rc=$?
[ "$rc" = 1 ] && ok "2c: a removed-but-also-edited platform field (driver) -> verify FAIL" || bad "2c: rc=$rc $out"
cp "$TMP/keep1" "$F1"
printf ' ' >> "$F1"; out="$("$PY" "$S" verify --against HEAD "$F1" 2>&1)"; rc=$?
[ "$rc" = 1 ] && ok "2d: a trailing byte -> verify FAIL" || bad "2d: rc=$rc $out"
cp "$TMP/keep1" "$F1"

echo
echo "=== 3: refusals leave the file untouched"
"$PY" -c "import json;open('$TMP/unknown.json','w').write(json.dumps({'schema':'hpcperf-other-1','provenance':{'host':'h1234'}},indent=2)+'\n')"
h="$(sha "$TMP/unknown.json")"; out="$("$PY" "$S" apply "$TMP/unknown.json" 2>&1)"; rc=$?
[ "$rc" = 2 ] && [ "$(sha "$TMP/unknown.json")" = "$h" ] && echo "$out" | /usr/bin/grep -q 'has no scrub rule' && ok "3a: an unknown schema is refused (exit 2), not touched" || bad "3a: rc=$rc $out"
printf '{"schema": "x"}' > "$TMP/noschema.json"; "$PY" -c "import json;open('$TMP/noschema2.json','w').write(json.dumps({'provenance':{'host':'h1234'}})+'\n')"
out="$("$PY" "$S" apply "$TMP/noschema2.json" 2>&1)"; rc=$?
[ "$rc" = 2 ] && /usr/bin/grep -q h1234 "$TMP/noschema2.json" && ok "3b: no schema at all -> refused" || bad "3b: rc=$rc $out"
GIT show HEAD:level1/fake/reference/n1000.json | sed 's/^  "outputs"/   "outputs"/' > "$TMP/reformatted.json"
h="$(sha "$TMP/reformatted.json")"; out="$("$PY" "$S" apply "$TMP/reformatted.json" 2>&1)"; rc=$?
[ "$rc" = 2 ] && [ "$(sha "$TMP/reformatted.json")" = "$h" ] && echo "$out" | /usr/bin/grep -q 'not the exact json.dumps output' \
    && ok "3c: a file that is not the exact json.dumps output of its content (re-serialising would change other bytes) -> refused" || bad "3c: rc=$rc $out"
"$PY" -c "import json;open('$TMP/nogpu.json','w').write(json.dumps({'schema':'hpcperf-golden-1','provenance':{'host':'h1234','gpu':'NVIDIA B200 GPU-x'}},indent=2)+'\n')"
h="$(sha "$TMP/nogpu.json")"; out="$("$PY" "$S" apply "$TMP/nogpu.json" 2>&1)"; rc=$?
[ "$rc" = 2 ] && [ "$(sha "$TMP/nogpu.json")" = "$h" ] && ok "3d: provenance.gpu not a mapping -> refused (never guesses)" || bad "3d: rc=$rc $out"
"$PY" -c "import json;open('$TMP/noprov.json','w').write(json.dumps({'schema':'hpcperf-golden-1','outputs':[]},indent=2)+'\n')"
out="$("$PY" "$S" apply "$TMP/noprov.json" 2>&1)"; rc=$?
[ "$rc" = 2 ] && echo "$out" | /usr/bin/grep -q 'no provenance block' && ok "3e: no provenance block -> refused" || bad "3e: rc=$rc $out"
"$PY" -c "import json;open('$TMP/badrec.json','w').write(json.dumps({'schema':'hpcperf-golden-1','provenance':{'host':'h1234','scrubbed':'by hand'}},indent=2)+'\n')"
out="$("$PY" "$S" apply "$TMP/badrec.json" 2>&1)"; rc=$?
[ "$rc" = 2 ] && /usr/bin/grep -q h1234 "$TMP/badrec.json" && ok "3f: a foreign 'scrubbed' record -> refused" || bad "3f: rc=$rc $out"
out="$("$PY" "$S" verify --against HEAD "$TMP/unknown.json" 2>&1)"; rc=$?
[ "$rc" = 2 ] && ok "3g: verify of a file outside the repository / not in REV -> refused (exit 2)" || bad "3g: rc=$rc $out"

echo
echo "=== 4: the repository's committed references (read-only, in memory)"
"$PY" - "$REPO" "$TOOLS" <<'PY'; rc=$?
import json, subprocess, sys
R, T = sys.argv[1:3]
sys.path.insert(0, T); import scrub_provenance as sp
files = [l for l in subprocess.run(["git", "-C", R, "ls-files", "level*/*/reference/*.json", "level*/*/reference/*/*.json"],
                                   capture_output=True, text=True).stdout.split() if l]
n = {"scrub": 0, "clean": 0, "no rule": 0}
for f in files:
    raw = subprocess.run(["git", "-C", R, "show", f"HEAD:{f}"], capture_output=True, text=True).stdout
    try:
        out, removed, redacted = sp.scrub_text(raw, f, [])
    except sp.Refused as ex:
        if "has no scrub rule" in str(ex):
            n["no rule"] += 1; continue
        print("REFUSED", ex); sys.exit(1)
    doc, rule, fmt = sp.parse(raw, f); new = json.loads(out)
    assert not sp.outside_problems(doc, new, rule, fmt), f
    assert sp.scrub_text(out, f, [])[0] == out, f"{f}: not idempotent"
    n["scrub" if out != raw else "clean"] += 1
print(f"   committed references: {n['scrub']} would change, {n['clean']} already clean, {n['no rule']} of another schema (not touched)")
PY
[ "$rc" = 0 ] && ok "4a: every committed reference of a known schema scrubs without refusal, idempotently, nothing outside the provenance changes" || bad "4a: rc=$rc"
"$PY" - "$REPO" "$TOOLS" <<'PY'; rc=$?
import subprocess, sys
R, T = sys.argv[1:3]
sys.path.insert(0, T); import scrub_provenance as sp
files = [l for l in subprocess.run(["git", "-C", R, "ls-files", "level*/*/reference/*.json", "level*/*/reference/*/*.json"],
                                   capture_output=True, text=True).stdout.split() if l]
n = {"fixed point": 0, "no rule": 0}
for f in files:
    raw = open(f"{R}/{f}").read()
    try:
        out = sp.scrub_text(raw, f, [])[0]
    except sp.Refused as ex:
        if "has no scrub rule" in str(ex):
            n["no rule"] += 1; continue
        print("REFUSED", ex); sys.exit(1)
    if out != raw:
        print(f"   {f}: a scrub would still change it (machine identity left in its provenance)"); sys.exit(1)
    n["fixed point"] += 1
print(f"   working-tree references: {n['fixed point']} of a known schema already scrubbed, {n['no rule']} of another schema")
PY
[ "$rc" = 0 ] && ok "4b: every working-tree reference of a known schema is a fixed point of the scrub (no host / batch job / absolute log location left)" || bad "4b: rc=$rc"

echo
echo "=== 5: the Level 2 CPU references made before the golden work (hipBone, miniWeather, NekBone, PENNANT, Branson cases)"
R2="$TMP/repo2"; mkdir -p "$R2"; GIT2() { git -C "$R2" -c user.email=t@t -c user.name=t "$@"; }
"$PY" - "$R2" <<'PY'
import json, os, re, sys
R = sys.argv[1]
def put(rel, text):
    os.makedirs(os.path.dirname(f"{R}/{rel}"), exist_ok=True); open(f"{R}/{rel}", "w").write(text)
hb = {"schema": "hpcperf-hipbone-reference-1", "input_id": "nx9-p14",
      "note": "CPU reference run 2026-09-30 on node17 (Slurm job 1397595), the same build/x/hipBone binary, OCCA OpenMP backend (8 threads, cores 100-107), exact registered arguments",
      "dofs": 1953125, "cg_iterations": 100, "r_norm_initial": 988.576225155911, "r_norm_final": 7.619353900899e-09,
      "runs": [{"log": "nx9.OpenMP.1.log", "sha256": "9c" * 32, "dofs": 1953125, "r_norm_final": 7.619353900899e-09}],
      "rule": {"r_norm_final": "|log10 ratio| <= 0.1", "dofs": "exact"}}
put("level2/hipbone/reference/nx9-p14.json", json.dumps(hb, indent=1))                       # json.dump: no newline
mw = {"schema": "hpcperf-miniweather-reference-1", "key": "thermal-64x32-10s", "nx": 64, "nz": 32, "sim_time": 10, "data_spec": "DATA_SPEC_THERMAL",
      "note": "CPU reference 2026-09-30 on node17 (Slurm job 1397595): build.sh OPENMP (YAKL OpenMP, -O3, no fast-math), 4 threads on cores 108-111; 4304 s",
      "d_mass": -9.210783e-13, "d_te": -7.842791e-09, "runs": [{"log": "t.openmp.1.log", "sha256": "ab" * 32}], "rule": {"d_te": "abs diff to reference <= 1e-07"}}
put("level2/miniweather/reference/thermal-64x32-10s.json", json.dumps(mw, indent=1))
nk = {"schema": "hpcperf-nekbone-cg-reference-1", "input_id": "n12-e78125",
      "provenance": {"made_utc": "2026-10-21T03:18:21Z", "SIZE_sha256": "60" * 32, "log": "n12.cpu.log", "log_sha256": "35" * 32,
                     "binary": "CPU build of the same source (build.sh CPU, nvfortran 25.7 -O2/-O3, no -acc/-cuda), one rank on node17",
                     "host": "node17.example.org", "note": "Solve Time =   0.3034E+03"},
      "rule": {"r0_tol": 0.0001, "x_tol": 1e-10, "text": "timed solve"},
      "run": {"solves": [{"r0": 5911.8, "iter": 101, "rN": 1.35e-07}], "xnorm": 7329.375063369254, "xf": 28422761.905461155, "nelt": 78125}}
put("level2/nekbone/reference/n12-e78125.json", json.dumps(nk, indent=1))
pn = {"schema": "hpcperf-pennant-reference-1", "input_id": "sedov-64",
      "provenance": {"made_utc": "2026-10-20T23:59:55Z", "deck": "sedov.pnt", "deck_sha256": "27" * 32, "log": "run.log", "log_sha256": "a9" * 32,
                     "binary": "lanl/PENNANT master 617414f CPU build, 1 MPI rank", "omp_threads": 32, "host": "node17.example.org",
                     "note": "upstream deck test/sedov/sedov.pnt run to tstop=1.0 (3882 cycles) on node17, 2026-10-20"},
      "rule": {"rel_tol": 1e-05, "zero_floor": 1e-08},
      "columns": {"zr": {"n": 4096, "scale": 1.0, "sum": 4.5e3, "sample_stride": 113, "samples": [[1, 1.0], [114, 0.5], [227, -2.5e-07]]}}}
put("level2/pennant/reference/sedov-64.json",                                               # check_xy.py's writer
    re.sub(r'\[\s*(\d+),\s*(-?[0-9.eE+-]+)\s*\]', r'[\1, \2]', json.dumps(pn, indent=1)) + "\n")
br = {"schema": "hpcperf-branson-reference-1", "benchmark": "branson", "input_id": "lb", "note": "24-rank run: 2 h 56 min wall",
      "final_step": {"photons_transported": 251242903, "post_mat_e": 0.017737}, "tolerances": {"energy_rel": 0.05},
      "reference_run": {"allocation": "node17, Slurm job 1397595 (24 of its 32 cores)", "binary_sha256": "39" * 32,
                        "launch": "taskset -c 16-31,96-99,108-111 mpirun -np 24", "log_sha256": "95" * 32, "mpi_ranks": 24,
                        "log_kept_at": "results directory r-2026-09-30/branson/lb/cpu.log (outside the repository)"}}
put("level2/branson/reference/lb.json", json.dumps(br, indent=1, sort_keys=True) + "\n")
br2 = json.loads(json.dumps(br)); br2["input_id"] = "lb2"; br2["reference_run"]["log_kept_at"] = "/home/someone/results/cpu.log (outside the repository)"
put("level2/branson/reference/lb2.json", json.dumps(br2, indent=1, sort_keys=True) + "\n")
PY
( cd "$R2" && git init -q && GIT2 add -A && GIT2 commit -qm fixtures ) && ok "5.0: fixture repository (5 Level 2 schemas, 3 serialisations) committed" || bad "5.0: git"
N=("$R2"/level2/*/reference/*.json); declare -A N0; for f in "${N[@]}"; do N0[$f]="$(sha "$f")"; done
out="$("$PY" "$S" apply --host node17 "${N[@]}" 2>&1)"; rc=$?
"$PY" - "$R2" <<'PY' && [ "$rc" = 0 ] && ok "5a: apply --host node17: host gone, ' on node17' / '(Slurm job N)' dropped from note / binary, all else verbatim (core lists, dates, run times); relative log_kept_at kept, absolute removed; record after note (no block) or last in the block" || bad "5a: rc=$rc $out"
import json, sys
R = sys.argv[1]; L = lambda p: json.load(open(f"{R}/level2/{p}"))
hb = L("hipbone/reference/nx9-p14.json")
assert hb["note"] == "CPU reference run 2026-09-30, the same build/x/hipBone binary, OCCA OpenMP backend (8 threads, cores 100-107), exact registered arguments", hb["note"]
assert list(hb)[:4] == ["schema", "input_id", "note", "scrubbed"], list(hb)
assert hb["scrubbed"]["removed"] == [] and hb["scrubbed"]["redacted_text"] == ["note: batch job", "note: host name"], hb["scrubbed"]
mw = L("miniweather/reference/thermal-64x32-10s.json")
assert mw["note"] == "CPU reference 2026-09-30: build.sh OPENMP (YAKL OpenMP, -O3, no fast-math), 4 threads on cores 108-111; 4304 s", mw["note"]
nk = L("nekbone/reference/n12-e78125.json")["provenance"]
assert "host" not in nk and nk["binary"].endswith("no -acc/-cuda), one rank") and nk["note"] == "Solve Time =   0.3034E+03", nk
assert list(nk)[-1] == "scrubbed" and nk["scrubbed"]["removed"] == ["host"] and nk["scrubbed"]["redacted_text"] == ["binary: host name"]
pn = L("pennant/reference/sedov-64.json")["provenance"]
assert "host" not in pn and pn["note"] == "upstream deck test/sedov/sedov.pnt run to tstop=1.0 (3882 cycles), 2026-10-20" and pn["omp_threads"] == 32, pn
br = L("branson/reference/lb.json")["reference_run"]
assert "allocation" not in br and br["log_kept_at"].startswith("results directory") and br["launch"].startswith("taskset -c 16-31,96-99,108-111")
assert br["scrubbed"]["removed"] == ["allocation"] and "redacted_text" not in br["scrubbed"]
br2 = L("branson/reference/lb2.json")["reference_run"]
assert "log_kept_at" not in br2 and br2["scrubbed"]["removed"] == ["allocation", "log_kept_at"], br2
import glob
for f in glob.glob(f"{R}/level2/*/reference/*.json"):
    t = open(f).read()
    for needle in ("node17", "example.org", "1397595", "Slurm", "/home/someone"):
        assert needle not in t, (f, needle)
t = open(f"{R}/level2/pennant/reference/sedov-64.json").read()
assert '    [114, 0.5],\n' in t and t.endswith("}\n"), "pennant: one sample per line and the trailing newline kept"
assert not open(f"{R}/level2/hipbone/reference/nx9-p14.json").read().endswith("\n"), "hipbone: still no trailing newline"
PY
"$PY" - "$R2" <<'PY' && ok "5b: independent raw-byte check: every top-level value outside the provenance set is the identical byte string, same order, same framing" || bad "5b"
import json, re, subprocess, sys
R = sys.argv[1]; dec = json.JSONDecoder(); ws = re.compile(r"[ \t\n\r]*")
def spans(raw):
    i = raw.index("{") + 1; out = []
    while True:
        i = ws.match(raw, i).end()
        if raw[i] == "}":
            return out, raw[i:]
        k, i = dec.raw_decode(raw, i); i = ws.match(raw, i).end() + 1; i = ws.match(raw, i).end()
        _, j = dec.raw_decode(raw, i); out.append((k, raw[i:j])); i = ws.match(raw, j).end(); i += raw[i] == ","
prov = {"hipbone": {"note", "scrubbed"}, "miniweather": {"note", "scrubbed"}, "nekbone": {"provenance"}, "pennant": {"provenance"},
        "branson": {"reference_run", "note"}}
for f in subprocess.run(["git", "-C", R, "ls-files"], capture_output=True, text=True).stdout.split():
    old = subprocess.run(["git", "-C", R, "show", f"HEAD:{f}"], capture_output=True, text=True).stdout; new = open(f"{R}/{f}").read()
    p = prov[f.split("/")[1]]
    (so, to), (sn, tn) = spans(old), spans(new)
    assert [x for x in so if x[0] not in p] == [x for x in sn if x[0] not in p], f
    assert to == tn and old[:2] == new[:2], f
PY
declare -A N1; for f in "${N[@]}"; do N1[$f]="$(sha "$f")"; done
out="$("$PY" "$S" apply --host node17 "${N[@]}" 2>&1)"; rc=$?
same=1; for f in "${N[@]}"; do [ "$(sha "$f")" = "${N1[$f]}" ] || same=0; done
[ "$rc" = 0 ] && [ "$same" = 1 ] && echo "$out" | /usr/bin/grep -q 'changed 0, unchanged 6' && ok "5c: idempotent: a second apply changes no byte" || bad "5c: rc=$rc $out"
mkdir -p "$TMP/det2"; for f in "${N[@]}"; do GIT2 show "HEAD:${f#$R2/}" > "$TMP/det2/$(basename "$f")"; done
"$PY" "$S" apply --host node17 "$TMP/det2"/*.json > /dev/null 2>&1
same=1; for f in "${N[@]}"; do [ "$(sha "$TMP/det2/$(basename "$f")")" = "${N1[$f]}" ] || same=0; done
[ "$same" = 1 ] && ok "5d: deterministic: scrubbing copies of the originals elsewhere gives the same bytes" || bad "5d: copies differ"
out="$("$PY" "$S" verify --host node17 --against HEAD "${N[@]}" 2>&1)"; rc=$?
[ "$rc" = 0 ] && [ "$(echo "$out" | /usr/bin/grep -c '^ok ')" = 6 ] && ok "5e: verify --host node17 --against HEAD: 6 files ok" || bad "5e: rc=$rc $out"
out="$("$PY" "$S" verify --against HEAD "${N[@]}" 2>&1)"; rc=$?
[ "$rc" = 1 ] && [ "$(echo "$out" | /usr/bin/grep -c '^FAIL ')" = 2 ] && echo "$out" | /usr/bin/grep -q '^FAIL .*hipbone' && echo "$out" | /usr/bin/grep -q '^FAIL .*miniweather' \
    && ok "5f: verify without --host: hipBone / miniWeather FAIL (their host is named only in the note, so the same --host is part of the proof); NekBone / PENNANT / Branson ok" || bad "5f: rc=$rc $out"
GIT2 show HEAD:level2/pennant/reference/sedov-64.json | sed 's/^    \[114, 0.5\],$/    [114,  0.5],/' > "$TMP/pn-bad.json"
h="$(sha "$TMP/pn-bad.json")"; out="$("$PY" "$S" apply --host node17 "$TMP/pn-bad.json" 2>&1)"; rc=$?
[ "$rc" = 2 ] && [ "$(sha "$TMP/pn-bad.json")" = "$h" ] && echo "$out" | /usr/bin/grep -q 'not the exact json.dumps output' \
    && ok "5g: a PENNANT reference with one sample line not in check_xy.py's form -> refused, untouched" || bad "5g: rc=$rc $out"
"$PY" -c "import json;open('$TMP/hb-job.json','w').write(json.dumps({'schema':'hpcperf-hipbone-reference-1','note':'run in Slurm job 1397595 on cores 1-8','dofs':1},indent=1))"
h="$(sha "$TMP/hb-job.json")"; out="$("$PY" "$S" apply "$TMP/hb-job.json" 2>&1)"; rc=$?
[ "$rc" = 2 ] && [ "$(sha "$TMP/hb-job.json")" = "$h" ] && echo "$out" | /usr/bin/grep -q 'does not remove cleanly' \
    && ok "5h: a batch-job mention the tool cannot drop cleanly ('in Slurm job N') -> refused, untouched (never a half-redacted sentence)" || bad "5h: rc=$rc $out"
"$PY" -c "import json;open('$TMP/nk-bin.json','w').write(json.dumps({'schema':'hpcperf-nekbone-cg-reference-1','provenance':{'host':'node17','binary':42},'run':{}},indent=1))"
out="$("$PY" "$S" apply "$TMP/nk-bin.json" 2>&1)"; rc=$?
[ "$rc" = 2 ] && /usr/bin/grep -q node17 "$TMP/nk-bin.json" && echo "$out" | /usr/bin/grep -q "'provenance.binary' is not free text" && ok "5i: an in-block free-text field that is not a string -> refused" || bad "5i: rc=$rc $out"
"$PY" -c "import json;open('$TMP/hb-rec.json','w').write(json.dumps({'schema':'hpcperf-hipbone-reference-1','note':'run on node17','scrubbed':'by hand','dofs':1},indent=1))"
out="$("$PY" "$S" apply --host node17 "$TMP/hb-rec.json" 2>&1)"; rc=$?
[ "$rc" = 2 ] && /usr/bin/grep -q node17 "$TMP/hb-rec.json" && ok "5j: a foreign top-level 'scrubbed' (schema without a block) -> refused" || bad "5j: rc=$rc $out"

echo
echo "scrub tests: $pass passed, $fail failed"
[ "$fail" = 0 ]
