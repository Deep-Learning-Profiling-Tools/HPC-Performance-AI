#!/usr/bin/env bash
# Run WarpX (full 3D electromagnetic PIC: deposition, Maxwell/Yee solve,
# particle push, periodic halo/particle exchange) on N GPUs.
#
#   ./run.sh [CUDA|HIP] [extra WarpX inputs overrides...]
#
# Cases (HPCPERF_WARPX_CASE):
#   uniform_plasma (default) -- Examples/Physics_applications/uniform_plasma/
#       inputs_base_3d, upstream's "commonly used to study performance" case:
#       thermal electron plasma, 2 particles/cell. Sized by HPCPERF_SCALE_MODE.
#   langmuir -- Examples/Tests/langmuir/inputs_base_3d (test_3d_langmuir_multi):
#       electron/positron Langmuir wave with an analytic solution, 64^3 cells,
#       40 steps; the correctness case used by validate.sh (sizes fixed).
#
# Execution model (AMReX/WarpX docs): one MPI rank per GPU. The common
# launcher's per-rank wrapper gives each rank exactly one visible GPU (AMReX
# then binds device 0 = that GPU; mapping audited). `warpx.numprocs PX PY PZ`
# assigns exactly one box per rank, so the requested rank count is what is
# decomposed: PX*PY*PZ must equal HPCPERF_GPUS and every dimension of
# amr.n_cell must be divisible by PX*blocking_factor -- otherwise the run is
# refused with the nearby legal rank counts (nothing is silently changed).
# GPU-aware MPI: AMReX auto-detects it from the (CUDA-aware) Open MPI;
# HPCPERF_WARPX_GPU_AWARE=0 forces host-staged communication.
#
# Resource / size controls (uniform_plasma):
#   HPCPERF_GPUS=N|all        ranks = GPUs (default 1)
#   HPCPERF_SCALE_MODE        smoke | strong | weak   (default smoke)
#     smoke  : upstream grid 64x32x32 cells (131,072 macroparticles), 10 steps
#     strong : ONE fixed global grid G^3 (G=HPCPERF_WARPX_GLOBAL, default 256:
#              16.8M cells, 33.6M particles), decomposed over the ranks
#     weak   : fixed per-rank block L^3 (L=HPCPERF_WARPX_LOCAL, default 128:
#              2.1M cells, 4.2M particles per rank); grid L*PX x L*PY x L*PZ
#   HPCPERF_WARPX_STEPS       time steps (default 10; strong/weak 20)
#   HPCPERF_WARPX_GPU_AWARE   1|0 (default: AMReX auto-detect)
#
# Derived inputs (class A, written into the build tree; upstream files
# untouched): upstream physics/numerics lines are kept verbatim;
# uniform_plasma drops the plotfile/checkpoint diagnostics (I/O), sets
# amr.n_cell / amr.max_grid_size / max_step / warpx.numprocs per mode,
# warpx.random_seed = 1 (reproducible sampling) and adds reduced diagnostics
# (ParticleEnergy, FieldEnergy, ParticleNumber); langmuir keeps upstream's
# diag1 plotfile (the analysis input), drops only the openPMD diagnostic
# (openPMD is not built) and adds warpx.numprocs. stdout is copied to
# <run_dir>/stdout.log.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"; [ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level3/warpx/$MODEL"
EXE="$(find "$BUILD_DIR/bin" -maxdepth 1 -name 'warpx.3d*' -type f 2>/dev/null | head -1)"
[ -n "$EXE" ] && [ -x "$EXE" ] || { echo "run.sh: warpx.3d* not found under $BUILD_DIR/bin -- run ./build.sh $BACKEND first" >&2; exit 1; }
CASE="${HPCPERF_WARPX_CASE:-uniform_plasma}"
case "$CASE" in
    uniform_plasma) BASE="$R/_upstream/level3/WarpX/Examples/Physics_applications/uniform_plasma/inputs_base_3d" ;;
    langmuir)       BASE="$R/_upstream/level3/WarpX/Examples/Tests/langmuir/inputs_base_3d" ;;
    *) echo "run.sh: HPCPERF_WARPX_CASE must be uniform_plasma or langmuir" >&2; exit 2 ;;
esac
[ -f "$BASE" ] || { echo "run.sh: $BASE missing (run fetch.sh)" >&2; exit 1; }

N_RANKS="$(hpcperf_ranks warpx yes)" || exit 2
hpcperf_forbid_args warpx amr.n_cell amr.max_grid_size amr.blocking_factor warpx.numprocs max_step warpx.random_seed -- "$@" || exit 2

if [ "$CASE" = langmuir ]; then
    MODE=validate; NX=64; NY=64; NZ=64; STEPS=40; BF=8      # upstream test as shipped (default blocking factor)
    TOPO="$(hpcperf_topology warpx "$N_RANKS" --divides "$((NX / BF)),$((NY / BF)),$((NZ / BF))")" || exit 2
else
    MODE="$(l3_scale_mode warpx)" || exit 2
    BF=16
    case "$MODE" in
        smoke)  NX=64; NY=32; NZ=32; STEPS="${HPCPERF_WARPX_STEPS:-10}" ;;
        strong) G="${HPCPERF_WARPX_GLOBAL:-256}"; NX=$G; NY=$G; NZ=$G; STEPS="${HPCPERF_WARPX_STEPS:-20}" ;;
        weak)   L="${HPCPERF_WARPX_LOCAL:-128}"; STEPS="${HPCPERF_WARPX_STEPS:-20}"
                [ $((L % BF)) -eq 0 ] || { echo "run.sh: HPCPERF_WARPX_LOCAL=$L must be a multiple of the blocking factor $BF" >&2; exit 2; } ;;
    esac
    if [ "$MODE" = weak ]; then
        TOPO="$(hpcperf_topology warpx "$N_RANKS")" || exit 2
        read -r PX PY PZ <<< "$TOPO"; NX=$((L * PX)); NY=$((L * PY)); NZ=$((L * PZ))
    else
        TOPO="$(hpcperf_topology warpx "$N_RANKS" --divides "$((NX / BF)),$((NY / BF)),$((NZ / BF))")" || {
            echo "run.sh: $N_RANKS ranks cannot tile the ${NX}x${NY}x${NZ} grid into one ${BF}-aligned box per rank (see feasible counts above)" >&2; exit 2; }
    fi
fi
read -r PX PY PZ <<< "$TOPO"
BX=$((NX / PX)); BY=$((NY / PY)); BZ=$((NZ / PZ))
MGS=$BX; [ "$BY" -gt "$MGS" ] && MGS=$BY; [ "$BZ" -gt "$MGS" ] && MGS=$BZ
CELLS=$((NX * NY * NZ))
if [ "$CASE" = langmuir ]; then PARTS=$((2 * CELLS)); else PARTS=$((2 * CELLS)); fi

# l3_rundir: real runs get a fresh dir; a dry-run gets a throwaway .dryrun/ dir so it
# can never delete or overwrite a real result directory (the old code rm -rf'd the real
# dir before the launcher's dry-run check ever ran).
RUN_DIR="$(l3_rundir "$BUILD_DIR/$L3_RUN_SUBDIR/$CASE.$MODE.np$N_RANKS")" || exit 2
IN="$RUN_DIR/inputs"
{
    echo "# derived from upstream $(realpath --relative-to="$R/_upstream/level3/WarpX" "$BASE") (HPC-Performance-AI level3/warpx/run.sh)"
    if [ "$CASE" = langmuir ]; then
        grep -vE '^\s*(amr\.max_grid_size|diagnostics\.diags_names|openpmd\.)' "$BASE"
        echo "diagnostics.diags_names = diag1"
        echo "amr.max_grid_size = $MGS"
    else
        grep -vE '^\s*(max_step|amr\.n_cell|amr\.max_grid_size|amr\.blocking_factor|diagnostics\.|diag1\.|chk\.)' "$BASE"
        echo "max_step = $STEPS"
        echo "amr.n_cell = $NX $NY $NZ"
        echo "amr.max_grid_size = $MGS"
        echo "amr.blocking_factor = $BF"
        echo "warpx.random_seed = 1"
        # no plotfile/checkpoint diagnostics (the upstream diag1/chk lines were dropped above)
        echo "warpx.reduced_diags_names = EP EF NP"
        echo "EP.type = ParticleEnergy"; echo "EP.intervals = 1"
        echo "EF.type = FieldEnergy";    echo "EF.intervals = 1"
        echo "NP.type = ParticleNumber"; echo "NP.intervals = 1"
    fi
    echo "warpx.numprocs = $PX $PY $PZ"
    [ -n "${HPCPERF_WARPX_GPU_AWARE:-}" ] && echo "amrex.use_gpu_aware_mpi = $HPCPERF_WARPX_GPU_AWARE"
} > "$IN"

echo "# WarpX $BACKEND: case=$CASE mode=$MODE ranks=$N_RANKS grid=${NX}x${NY}x${NZ} ($CELLS cells, $PARTS particles, $((PARTS / N_RANKS))/rank) numprocs=${PX}x${PY}x${PZ} box=${BX}x${BY}x${BZ} steps=$STEPS run_dir=$RUN_DIR"
cd "$RUN_DIR"
RUN_ID="$(l3_run_id)"
"$L3_LAUNCHER" --gpus "$N_RANKS" --bind wrapper -- "$EXE" "$IN" "$@" 2>&1 | tee "$RUN_DIR/stdout.log"
rc=${PIPESTATUS[0]}
if [ -z "${HPCPERF_DRY_RUN:-}" ]; then
    l3_manifest "$RUN_DIR" "run_id=$RUN_ID" "app=warpx" "backend=$BACKEND" "case=$CASE" "mode=$MODE" \
        "ranks=$N_RANKS" "grid=${NX}x${NY}x${NZ}" "numprocs=${PX}x${PY}x${PZ}" "particles=$PARTS" "steps=$STEPS" \
        "exit_code=$rc" "binary=$EXE" "binary_sha256=$(l3_sha_file "$EXE")" "input=$IN" "input_sha256=$(l3_sha_file "$IN")" \
        "stdout=$RUN_DIR/stdout.log" "utc=$(date -u +%FT%TZ)"
fi
exit "$rc"
