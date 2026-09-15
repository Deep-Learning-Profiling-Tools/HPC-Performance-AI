#!/usr/bin/env bash
# Run the ExaCA directional-solidification benchmark (inputs/dirsolid.template.json) on N GPUs.
#
#   ./run.sh [CUDA|HIP] [extra ExaCA/Kokkos args...]
#
# Execution model (upstream): one MPI rank per GPU, Kokkos CUDA; the domain is decomposed in Y across the
# ranks (1-D), halo layers are exchanged through host-staged MPI buffers every time step. Ranks are launched
# through the common launcher with the per-rank GPU wrapper (each rank sees exactly one GPU; the launcher
# audits the rank->GPU mapping). One global simulation, never N independent copies.
#
# Resource / size controls (common Level 3 parameters):
#   HPCPERF_GPUS=N|all        ranks = GPUs (default 1)
#   HPCPERF_SCALE_MODE        smoke | strong | weak   (default smoke)
#     smoke  : 128 x 128 x 128 cells (2.1M), any rank count (Y is split 128/N); the GrainID field is written
#              (validate.sh reads it)
#     strong : ONE fixed global box 512 x 256 x 512 cells (67.1M), decomposed over the ranks in Y (a single rank
#              cannot hold 512^3: ExaCA indexes the 26-neighbour octahedron arrays with int, 134M x 26 overflows)
#     weak   : fixed work per rank 512 x 128 x 512 cells (33.5M/rank): the box is 512 x (128*N) x 512
#   HPCPERF_EXACA_NX/NY/NZ    override the cell counts of the selected mode (Ny in weak mode is per rank)
#   HPCPERF_EXACA_SEED        RandomSeed of the deck (default 0 = the validated deck)
#   HPCPERF_EXACA_PRINT=1     also write the GrainID field in strong/weak mode (default: smoke only -- the
#                             field is 4 B/cell, 512 MB for the strong box, and its write time is not the CA
#                             work being measured; ExaCA's log with VolFractionNucleated is always written)
# The deck is generated from the template with a JSON-aware substitution (domain size, output path/name,
# seed, printing policy); nothing else differs between modes. Extra args are appended to the command line
# (the input file stays the first argument); a second positional input file is rejected.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"; [ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
PROFILE="$(l3_backend_profile EXACA "$MODEL")"
l3_paths_profile exaca "$PROFILE" "$MODEL" || exit 2     # same derivation as build.sh: install and run tree of ONE profile
EXE="$L3_INSTALL/exaca/bin/ExaCA"
[ -x "$EXE" ] || { echo "run.sh: $EXE not found -- run ./build.sh $BACKEND first (profile $PROFILE)" >&2; exit 1; }
l3_fingerprint_expect_backend "$L3_INSTALL" "$MODEL" || exit 1
l3_require_materialized "$HERE" || exit 3
TEMPLATE="$HERE/inputs/dirsolid.template.json"
[ -f "$TEMPLATE" ] || { echo "run.sh: $TEMPLATE missing" >&2; exit 1; }
for a in "$@"; do case "$a" in *.json) echo "run.sh: a second input deck ($a) would replace the validated one -- rejected" >&2; exit 2;; esac; done

N_RANKS="$(hpcperf_ranks exaca yes)" || exit 2
MODE="$(l3_scale_mode exaca)" || exit 2
case "$MODE" in
    smoke)  NX=128; NY=128; NZ=128 ;;
    strong) NX=512; NY=256; NZ=512 ;;
    weak)   NX=512; NY=$(( 128 * N_RANKS )); NZ=512 ;;
esac
NX="${HPCPERF_EXACA_NX:-$NX}"; NZ="${HPCPERF_EXACA_NZ:-$NZ}"
if [ "$MODE" = weak ]; then NY=$(( ${HPCPERF_EXACA_NY:-128} * N_RANKS )); else NY="${HPCPERF_EXACA_NY:-$NY}"; fi
[ "$NY" -ge $(( 2 * N_RANKS )) ] || { echo "run.sh: Ny=$NY cannot be decomposed over $N_RANKS ranks (needs >= 2 cells per rank) -- refused" >&2; exit 2; }
SEED="${HPCPERF_EXACA_SEED:-0}"
CELLS=$(( NX * NY * NZ ))
RUN_DIR="$(l3_rundir "$L3_BUILD/$L3_RUN_SUBDIR/dirsolid.$MODE.np$N_RANKS")"
OUT="dirsolid_${MODE}_np${N_RANKS}"
IN="$RUN_DIR/dirsolid.$MODE.json"
PRINT_FIELD=$([ "$MODE" = smoke ] || [ -n "${HPCPERF_EXACA_PRINT:-}" ] && echo 1 || echo 0)
python3 - "$TEMPLATE" "$IN" "$NX" "$NY" "$NZ" "$RUN_DIR" "$OUT" "$SEED" "$PRINT_FIELD" <<'PY'
import json, sys
t, out, nx, ny, nz, rundir, name, seed, pf = sys.argv[1:10]
d = json.load(open(t)); d.pop("_comment", None)
d["Domain"]["Nx"], d["Domain"]["Ny"], d["Domain"]["Nz"] = int(nx), int(ny), int(nz)
d["RandomSeed"] = float(seed) if "." in seed else int(seed)
d["Printing"]["PathToOutput"] = rundir + "/"; d["Printing"]["OutputFile"] = name
if pf != "1":
    d["Printing"].pop("Interlayer", None)     # log file only (no GrainID field) for the performance modes
json.dump(d, open(out, "w"), indent=3)
PY
LOG="$RUN_DIR/stdout.log"

echo "# ExaCA $BACKEND profile=$PROFILE: mode=$MODE ranks=$N_RANKS box=${NX}x${NY}x${NZ} cells = $CELLS ($((CELLS / N_RANKS))/rank), seed=$SEED, field=$([ "$PRINT_FIELD" = 1 ] && echo GrainID || echo none), out=$RUN_DIR/$OUT.{vtk,json}"
RUN_ID="$(l3_run_id)"
rc=0
"$L3_LAUNCHER" --gpus "$N_RANKS" --bind wrapper -- "$EXE" "$IN" "$@" 2>&1 | tee "$LOG" || rc=${PIPESTATUS[0]}
if [ -z "${HPCPERF_DRY_RUN:-}" ]; then
    l3_manifest "$RUN_DIR" "run_id=$RUN_ID" "app=exaca" "backend=$BACKEND" "profile=$PROFILE" "mode=$MODE" "case=dirsolid" \
        "ranks=$N_RANKS" "nx=$NX" "ny=$NY" "nz=$NZ" "cells=$CELLS" "seed=$SEED" "exit_code=$rc" \
        "binary=$EXE" "binary_sha256=$(l3_sha_file "$EXE")" "input=$IN" "input_sha256=$(l3_sha_file "$IN")" \
        "fingerprint=$L3_INSTALL/.hpcperf-l3-fingerprint" "fingerprint_sha256=$(l3_sha_file "$L3_INSTALL/.hpcperf-l3-fingerprint")" \
        "template_sha256=$(l3_sha_file "$TEMPLATE")" "grainid_field=$PRINT_FIELD" "output_vtk=$RUN_DIR/$OUT.vtk" "output_log=$RUN_DIR/$OUT.json" "utc=$(date -u +%FT%TZ)"
fi
exit "$rc"
