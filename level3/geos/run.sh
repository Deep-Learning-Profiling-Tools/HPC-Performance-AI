#!/usr/bin/env bash
# Run GEOS (develop snapshot, CUDA sm_100, hypre on device) on N GPUs, one MPI
# rank per GPU (upstream's model: --ntasks-per-gpu=1 / jsrun -g 1; GEOS itself
# does not select a device -- the launcher's per-rank wrapper exposes one GPU).
#
#   ./run.sh [CUDA] [extra geos args...]
#
# Cases (HPCPERF_GEOS_CASE), upstream inputs under inputFiles/solidMechanics/:
#   beam (default)   Beam bending (quasi-static linear elasticity, SolidMechanicsLagrangianFEM,
#                    traction ramp over 10 steps, analytic Euler-Bernoulli reference shipped as
#                    beamBending_curve.py). Decks:
#     smoke  : beamBending_smoke.xml mesh (80x8x4 C3D8) with the benchmark deck's iterative solver
#              (GMRES + AMG on hypre, krylovTol 1e-6 -- the GPU linear-solver path); the shipped
#              smoke deck uses a serial direct solver (directParallel=0), kept for the 1-rank
#              official-baseline comparison in validate.sh (HPCPERF_GEOS_SOLVER=direct)
#     strong : beamBending_benchmark.xml as shipped (160x16x8, GMRES+AMG) -- fixed mesh over ranks
#     weak   : the beam refined so that elements/rank stays ~constant (nx,ny,nz scaled with the
#              partition grid; same physical problem, finer mesh) -- labelled "refinement weak"
#   Partitions come from the launcher's topology helper with the divisibility constraint
#   (nx % PX == 0 ...); GEOS receives -x PX -y PY -z PZ so requested == decomposed ranks.
#
# Controls: HPCPERF_GPUS, HPCPERF_SCALE_MODE, HPCPERF_GEOS_PROFILE, HPCPERF_GEOS_SOLVER=amg|direct
# (smoke only), HPCPERF_GEOS_WEAK_NX (elements per rank along x at PX=1, default 80).
# Output: <run_dir>/ (GEOS -o), displacement_history.hdf5 (TimeHistory), silo/, restart files
# (only for HPCPERF_GEOS_SOLVER=direct runs); stdout.log; run_manifest.txt.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"; [ $# -gt 0 ] && shift
[ "$BACKEND" = CUDA ] || { echo "run.sh: only CUDA is built for GEOS here" >&2; exit 2; }
GCC_MM="$(l3_version_mm "$(/usr/bin/gcc -dumpfullversion)")"; OMPI_V="$(mpirun --version | head -1 | /usr/bin/grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
PROFILE="${HPCPERF_GEOS_PROFILE:-cuda$(l3_version_mm "$(l3_cuda_version)")-gcc${GCC_MM}-ompi$(echo "$OMPI_V" | tr -d .)}"
l3_paths_profile geos "$PROFILE"
SRC="$R/_upstream/level3/GEOS"
EXE="$L3_INSTALL/geos/bin/geosx"; [ -x "$EXE" ] || EXE="$(find "$L3_INSTALL/geos/bin" -maxdepth 1 -type f -name 'geos*' 2>/dev/null | head -1)"
[ -n "$EXE" ] && [ -x "$EXE" ] || { echo "run.sh: GEOS executable not found under $L3_INSTALL/geos/bin -- run ./build.sh" >&2; exit 1; }
l3_binary_backend_check "$EXE" cuda || exit 1
CASE="${HPCPERF_GEOS_CASE:-beam}"; [ "$CASE" = beam ] || { echo "run.sh: HPCPERF_GEOS_CASE must be beam" >&2; exit 2; }
MODE="$(l3_scale_mode geos)" || exit 2
N_RANKS="$(hpcperf_ranks geos yes)" || exit 2
SOLVER="${HPCPERF_GEOS_SOLVER:-amg}"
hpcperf_forbid_args geos -x -y -z -i -o --input --output --x-partitions --y-partitions --z-partitions -- "$@" || exit 2
D="$SRC/inputFiles/solidMechanics"
case "$MODE" in
    smoke)  NX=80; NY=8; NZ=4; LABEL="beam80.$SOLVER" ;;
    strong) NX=160; NY=16; NZ=8; LABEL="beam160.amg"; SOLVER=amg ;;
    weak)   L="${HPCPERF_GEOS_WEAK_NX:-80}"; SOLVER=amg
            TOPO="$(hpcperf_topology geos "$N_RANKS")" || exit 2; read -r PX PY PZ <<< "$TOPO"
            NX=$((L * PX)); NY=$((L / 10 * PY)); NZ=$((L / 20 * PZ)); LABEL="beam.amg" ;;
esac
[ "$MODE" = smoke ] && case "$SOLVER" in amg|direct) ;; *) echo "run.sh: HPCPERF_GEOS_SOLVER must be amg or direct" >&2; exit 2;; esac
if [ "$MODE" != weak ]; then
    TOPO="$(hpcperf_topology geos "$N_RANKS" --divides "$NX,$NY,$NZ")" || { echo "run.sh: $N_RANKS ranks cannot partition the ${NX}x${NY}x${NZ} mesh (each factor must divide its extent)" >&2; exit 2; }
    read -r PX PY PZ <<< "$TOPO"
fi
ELEMS=$((NX * NY * NZ))
RUN_DIR="$(l3_rundir "$L3_BUILD/$L3_RUN_SUBDIR/$LABEL.$MODE.np$N_RANKS")" || exit 2
# Derived deck (class A): upstream smoke/benchmark deck with the mesh resolution and, for
# the smoke+amg variant, the benchmark's LinearSolverParameters; base file copied alongside
# so the <Included> relative reference resolves. Physics, BCs, material, time stepping and
# the analytic reference (beamBending_curve.py) are upstream's, untouched.
cp "$D/beamBending_base.xml" "$RUN_DIR/"
if [ "$MODE" = smoke ] && [ "$SOLVER" = direct ]; then
    cp "$D/beamBending_smoke.xml" "$RUN_DIR/input.xml"; DECK="$D/beamBending_smoke.xml"
else
    DECK="$D/beamBending_benchmark.xml"
    sed -e "s|nx=\"{ 160 }\"|nx=\"{ $NX }\"|; s|ny=\"{ 16 }\"|ny=\"{ $NY }\"|; s|nz=\"{ 8 }\"|nz=\"{ $NZ }\"|" "$DECK" > "$RUN_DIR/input.xml"
    /usr/bin/grep -q "nx=\"{ $NX }\"" "$RUN_DIR/input.xml" || { echo "run.sh: mesh substitution failed" >&2; exit 1; }
fi
echo "# GEOS $BACKEND profile=$PROFILE case=$CASE mode=$MODE ranks=$N_RANKS mesh=${NX}x${NY}x${NZ} ($ELEMS elements) partitions=${PX}x${PY}x${PZ} solver=$SOLVER deck=$(realpath --relative-to="$SRC" "$DECK") run_dir=$RUN_DIR"
cd "$RUN_DIR"
RUN_ID="$(l3_run_id)"
set +e; set -o pipefail
"$L3_LAUNCHER" --gpus "$N_RANKS" --bind wrapper -- "$EXE" -i input.xml -x "$PX" -y "$PY" -z "$PZ" -o "$RUN_DIR" "$@" 2>&1 | tee "$RUN_DIR/stdout.log"
rc=$?
set +o pipefail; set -e
if [ -z "${HPCPERF_DRY_RUN:-}" ]; then
    l3_manifest "$RUN_DIR" "run_id=$RUN_ID" "app=geos" "backend=$BACKEND" "profile=$PROFILE" "case=$CASE" "mode=$MODE" "solver=$SOLVER" \
        "ranks=$N_RANKS" "mesh=${NX}x${NY}x${NZ}" "elements=$ELEMS" "partitions=${PX}x${PY}x${PZ}" "exit_code=$rc" \
        "binary=$EXE" "binary_sha256=$(l3_sha_file "$EXE")" "deck=$DECK" "deck_sha256=$(l3_sha_file "$DECK")" "input_sha256=$(l3_sha_file "$RUN_DIR/input.xml")" \
        "fingerprint_sha256=$(l3_sha_file "$L3_INSTALL/.hpcperf-l3-fingerprint")" "stdout=$RUN_DIR/stdout.log" "utc=$(date -u +%FT%TZ)"
fi
exit "$rc"
