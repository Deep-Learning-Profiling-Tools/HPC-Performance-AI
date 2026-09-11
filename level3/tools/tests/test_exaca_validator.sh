#!/bin/bash
# Negative/positive tests of the ExaCA validator (CPU only, no GPU, no ExaCA binary): exaca_check.py stats /
# validate on synthetic GrainID fields and logs, plus the validate.sh stale-result gate.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../../.." && pwd)"
E="$R/level3/exaca"
pass=0; failn=0
ok()  { echo "ok   $*"; pass=$((pass+1)); }
bad() { echo "FAIL $*"; failn=$((failn+1)); }
TMP="$(mktemp -d /tmp/exaca-val-XXXXXX)"; trap 'rm -rf "$TMP"' EXIT
export L3_TOOLS="$R/level3/tools" PYTHONDONTWRITEBYTECODE=1
PY=python3
# --- synthetic field: 8x8x8 cells, 4 epitaxial grains + 2 nucleated grains, all solidified; orientation file with 10 rows
$PY - "$TMP" <<'PY'
import json, os, sys, numpy as np
T = sys.argv[1]
nx = ny = nz = 8
g = np.zeros((nz, ny, nx), dtype=np.int32)
g[:, :4, :4] = 1; g[:, :4, 4:] = 2; g[:, 4:, :4] = 3; g[:, 4:, 4:] = 4
g[5:, 2:4, 2:4] = -1; g[6:, 5:7, 5:7] = -2
def write_vtk(path, arr, binary=True):
    with open(path, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\nvtk output\n" + (b"BINARY\n" if binary else b"ASCII\n") + b"DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\nORIGIN 0 0 0\nSPACING 1 1 1\nPOINT_DATA {arr.size}\nSCALARS GrainID int 1\nLOOKUP_TABLE default\n".encode())
        f.write(arr.astype(">i4").tobytes() if binary else (" ".join(map(str, arr.ravel().tolist())) + "\n").encode())
write_vtk(os.path.join(T, "good.vtk"), g)
write_vtk(os.path.join(T, "good_ascii.vtk"), g, binary=False)
bad = g.copy(); bad[0, 0, 0] = 0; write_vtk(os.path.join(T, "unsolid.vtk"), bad)
with open(os.path.join(T, "orient.csv"), "w") as f:
    f.write("10\n")
    rng = np.random.default_rng(1)
    for i in range(10):
        q = rng.normal(size=(3, 3)); q, _ = np.linalg.qr(q); f.write(",".join(f"{x:.6f}" for x in q.ravel()) + "\n")
vf = float((g < 0).sum() / g.size)
def log(path, ranks, sizes, offsets, vfc=vf):
    json.dump({"ExaCAVersion": "2.1.0", "KokkosVersion": "4.7.4", "TimeStepOfOutput": 100, "NumberMPIRanks": ranks,
               "Nucleation": {"VolFractionNucleated": vfc}, "Domain": {"Nx": nx, "Ny": ny, "Nz": nz},
               "Decomposition": {"SubdomainYSize": sizes, "SubdomainYOffset": offsets}}, open(path, "w"))
log(os.path.join(T, "np1.json"), 1, [8], [0])
log(os.path.join(T, "np2.json"), 2, [5, 5], [0, 3])            # 8 cells, 1-cell halos: sizes sum 10 = 8 + 2
log(os.path.join(T, "np2_gapdup.json"), 2, [5, 5], [0, 5])     # halo-sum formula holds, but rows 3-4 duplicated / offset wrong -> gap
log(os.path.join(T, "np2_short.json"), 2, [4, 5], [0, 2])      # ends at 7, not 8
log(os.path.join(T, "np2_wrongranks.json"), 3, [5, 5], [0, 3])
log(os.path.join(T, "np1_badvf.json"), 1, [8], [0], vfc=vf + 0.05)
log(os.path.join(T, "np1_nanvf.json"), 1, [8], [0], vfc=float("nan"))
PY
ORI="$TMP/orient.csv"
$PY "$E/exaca_check.py" stats "$TMP/good.vtk" "$TMP/np1.json" "$ORI" --json "$TMP/good.stats.json" > /dev/null && ok "1a: statistics of a synthetic binary field" || bad "1a"
$PY "$E/exaca_check.py" stats "$TMP/good_ascii.vtk" "$TMP/np1.json" "$ORI" --json "$TMP/good_ascii.stats.json" > /dev/null && cmp -s <($PY -c "import json;d=json.load(open('$TMP/good.stats.json'));d.pop('vtk');print(d)") <($PY -c "import json;d=json.load(open('$TMP/good_ascii.stats.json'));d.pop('vtk');print(d)") && ok "1b: ASCII and binary fields give identical statistics" || bad "1b"
$PY "$E/exaca_check.py" stats "$TMP/unsolid.vtk" "$TMP/np1.json" "$ORI" --json "$TMP/unsolid.stats.json" > /dev/null; /usr/bin/grep -q '"unsolidified_cells": 1' "$TMP/unsolid.stats.json" && ok "1c: an unsolidified cell is counted" || bad "1c"
# reference + tolerances from the synthetic good run
$PY - "$TMP" <<'PY'
import json, sys; T = sys.argv[1]; st = json.load(open(f"{T}/good.stats.json"))
json.dump({"schema": "hpcperf-exaca-reference-1", "provenance": {"exaca_version": "synthetic"}, "stats": {k: v for k, v in st.items() if k != "log"}}, open(f"{T}/ref.json", "w"))
json.dump({"unsolidified_cells": 0, "vol_fraction_nucleated": 0.01, "n_grains_rel": 0.01, "n_nucleated_rel": 0.05, "top_layer_grains_rel": 0.25, "mean_misorientation_z_deg": 0.25, "mean_misorientation_z_top_deg": 0.7, "mean_grain_volume_cells_rel": 0.01}, open(f"{T}/tol.json", "w"))
PY
V="$PY $E/exaca_check.py validate"; D="--expect-dims 8,8,8"
$V "$TMP/good.stats.json" --ref "$TMP/ref.json" --tol "$TMP/tol.json" --ranks 1 $D > "$TMP/o.txt"; rc=$?; [ $rc -eq 0 ] && /usr/bin/grep -q ': PASS$' "$TMP/o.txt" && ok "2a: good 1-rank run PASSes" || bad "2a: rc=$rc $(tail -2 "$TMP/o.txt")"
$V "$TMP/unsolid.stats.json" --ref "$TMP/ref.json" --tol "$TMP/tol.json" --ranks 1 $D > "$TMP/o.txt"; rc=$?; [ $rc -ne 0 ] && /usr/bin/grep -q 'all cells solidified.*BAD' "$TMP/o.txt" && ok "2b: unsolidified cell -> FAIL" || bad "2b: rc=$rc"
$PY "$E/exaca_check.py" stats "$TMP/good.vtk" "$TMP/np2.json" "$ORI" --json "$TMP/np2.stats.json" > /dev/null
$V "$TMP/np2.stats.json" --ref "$TMP/ref.json" --tol "$TMP/tol.json" --ranks 2 --np1 "$TMP/good.stats.json" $D > "$TMP/o.txt"; rc=$?; [ $rc -eq 0 ] && /usr/bin/grep -q 'tile \[0, 8) once' "$TMP/o.txt" && ok "2c: correct 2-rank decomposition [5,5]/[0,3] tiles the box once -> PASS" || bad "2c: rc=$rc $(/usr/bin/grep -E 'BAD|tile' "$TMP/o.txt")"
for case in "np2_gapdup:gap or duplicate:2" "np2_short:ends at 7:2" "np2_wrongranks:ranks in the log:2"; do IFS=: read -r lg needle ranks <<< "$case"
  $PY "$E/exaca_check.py" stats "$TMP/good.vtk" "$TMP/$lg.json" "$ORI" --json "$TMP/$lg.stats.json" > /dev/null
  $V "$TMP/$lg.stats.json" --ref "$TMP/ref.json" --tol "$TMP/tol.json" --ranks $ranks --np1 "$TMP/good.stats.json" $D > "$TMP/o.txt"; rc=$?
  [ $rc -ne 0 ] && /usr/bin/grep -q "$needle" "$TMP/o.txt" && ok "2d: $lg -> FAIL ($needle)" || bad "2d: $lg rc=$rc $(/usr/bin/grep BAD "$TMP/o.txt" | head -2)"; done
$V "$TMP/np2.stats.json" --ref "$TMP/ref.json" --tol "$TMP/tol.json" --ranks 2 $D > "$TMP/o.txt"; rc=$?; [ $rc -ne 0 ] && /usr/bin/grep -q '1-GPU run of this build available: missing' "$TMP/o.txt" && ok "2e: N>1 without the 1-GPU run of this build -> FAIL" || bad "2e: rc=$rc"
$PY "$E/exaca_check.py" stats "$TMP/good.vtk" "$TMP/np1_badvf.json" "$ORI" --json "$TMP/badvf.stats.json" > /dev/null
$V "$TMP/badvf.stats.json" --ref "$TMP/ref.json" --tol "$TMP/tol.json" --ranks 1 $D > "$TMP/o.txt"; rc=$?; [ $rc -ne 0 ] && /usr/bin/grep -q 'VolFractionNucleated.*BAD' "$TMP/o.txt" && ok "2f: log/field inconsistency -> FAIL" || bad "2f: rc=$rc"
$PY "$E/exaca_check.py" stats "$TMP/good.vtk" "$TMP/np1_nanvf.json" "$ORI" --json "$TMP/nanvf.stats.json" > /dev/null
$V "$TMP/nanvf.stats.json" --ref "$TMP/ref.json" --tol "$TMP/tol.json" --ranks 1 $D > "$TMP/o.txt"; rc=$?; [ $rc -ne 0 ] && ok "2g: NaN VolFractionNucleated in the log -> FAIL" || bad "2g: rc=$rc"
$PY - "$TMP" <<'PY'
import json, sys; T = sys.argv[1]
for name, mut in (("nan", lambda s: s.update(mean_misorientation_z_deg=float("nan"))), ("inf", lambda s: s.update(vol_fraction_nucleated=float("inf"))),
                  ("outtol", lambda s: s.update(n_grains=s["n_grains"] + max(1, int(0.02 * s["n_grains"])) + 1)), ("dims", lambda s: s.update(nx=9)), ("missing", lambda s: s.pop("top_layer_grains"))):
    s = json.load(open(f"{T}/good.stats.json")); mut(s); json.dump(s, open(f"{T}/{name}.stats.json", "w"))
PY
for c in nan inf outtol dims missing; do $V "$TMP/$c.stats.json" --ref "$TMP/ref.json" --tol "$TMP/tol.json" --ranks 1 $D > "$TMP/o.txt" 2>&1; rc=$?; [ $rc -ne 0 ] && ! /usr/bin/grep -q ': PASS$' "$TMP/o.txt" && ok "2h: $c statistic -> FAIL" || bad "2h: $c rc=$rc"; done
$PY - "$TMP" <<'PY'
import json, sys; T = sys.argv[1]; r = json.load(open(f"{T}/ref.json")); r["stats"]["n_grains"] = float("nan"); json.dump(r, open(f"{T}/ref_nan.json", "w"))
PY
$V "$TMP/good.stats.json" --ref "$TMP/ref_nan.json" --tol "$TMP/tol.json" --ranks 1 $D > "$TMP/o.txt" 2>&1; rc=$?; [ $rc -ne 0 ] && ok "2i: a non-finite reference value never PASSes" || bad "2i"
# --- validate.sh stale-result gate: an old field/log in the run directory cannot stand in for a failed run
B="$TMP/bench"; mkdir -p "$B/references" "$B/src/examples/Substrate" "$B/inputs"; cp "$E/validate.sh" "$E/exaca_check.py" "$B/"; cp "$E/references/"*.json "$E/references/validation_protocol.yaml" "$B/references/"; cp "$ORI" "$B/src/examples/Substrate/GrainOrientationVectors.csv"; cp "$E/inputs/dirsolid.template.json" "$B/inputs/"; echo 'name: exaca' > "$B/benchmark.yaml"
printf '#!/bin/bash\necho "run.sh: simulated failure"; exit 1\n' > "$B/run.sh"; chmod +x "$B/run.sh"
mkdir -p "$TMP/fakeroot/level3/tools" "$TMP/fakeroot/level2/tools"; cp -a "$R/level3/tools/." "$TMP/fakeroot/level3/tools/"; cp -a "$R/level2/tools/." "$TMP/fakeroot/level2/tools/"; printf '#!/bin/bash\n' > "$TMP/fakeroot/hpcperf_env.sh"
mkdir -p "$TMP/fakeroot/level3/exaca"; cp -a "$B/." "$TMP/fakeroot/level3/exaca/"
RD="$TMP/fakeroot/build/level3/exaca/cuda/run/dirsolid.smoke.np1"; mkdir -p "$RD"; cp "$TMP/good.vtk" "$RD/dirsolid_smoke_np1.vtk"; cp "$TMP/np1.json" "$RD/dirsolid_smoke_np1.json"; cp "$TMP/good.stats.json" "$RD/stats.json"; echo "run_id=stale" > "$RD/run_manifest.txt"
out="$(cd "$TMP/fakeroot/level3/exaca" && HPCPERF_GPUS=1 bash ./validate.sh CUDA 2>&1)"; rc=$?
[ $rc -eq 1 ] && echo "$out" | /usr/bin/grep -q 'run.sh exited 1' && ! echo "$out" | /usr/bin/grep -q ': PASS$' && ok "3a: validate.sh FAILs when run.sh fails even though a previous field/log/stats exist in the run directory" || bad "3a: rc=$rc $(echo "$out" | tail -2)"
echo
echo "test_exaca_validator: $pass passed, $failn failed"
[ "$failn" -eq 0 ]
