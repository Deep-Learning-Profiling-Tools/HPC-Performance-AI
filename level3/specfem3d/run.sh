#!/usr/bin/env bash
# Run the full SPECFEM3D Cartesian workflow (mesh -> databases -> solver) for
# the homogeneous-halfspace problem on N GPUs.
#
#   ./run.sh [CUDA|HIP]
#
# Workflow (upstream EXAMPLES/*/run_this_example.sh): the mesh is partitioned
# into NPROC slices, `xgenerate_databases` (NPROC MPI ranks, CPU) builds the
# per-slice databases, `xspecfem3D` (NPROC MPI ranks, GPU_MODE) runs the
# spectral-element time loop. NPROC is fixed at mesh time, so every rank count
# gets its own run directory and its own mesh/databases.
#
# Execution model: one MPI rank per GPU (upstream: device = myrank %
# device_count; the common launcher's per-rank wrapper gives each rank one
# visible GPU, so device 0 is that rank's GPU; audited). Halo exchange in
# v4.1.1 is host-staged (no GPU-aware MPI requirement). All three stages are
# launched through the common launcher with the same rank count.
#
# Resource / size controls:
#   HPCPERF_GPUS=N|all        ranks = GPUs = NPROC (default 1)
#   HPCPERF_SCALE_MODE        smoke | strong | weak  (default smoke)
#     smoke  : upstream case as shipped: CUBIT mesh MESH-default (36x36x16 =
#              20,736 HEX8 elements, 134x134x60 km, Vp 2.8 km/s), partitioned
#              with the bundled SCOTCH by xdecompose_mesh (any NPROC), NSTEP
#              5000, DT 0.05 s -- the case whose reference seismograms ship
#     strong : in-house mesher xmeshfem3D on the SAME domain refined G times
#              (G=HPCPERF_SPECFEM_STRONG, default 2): (36G)x(36G)x(16G) =
#              165,888 elements, DT 0.05/G, NPROC = PX*PY from
#              hpcperf_topology.py (2-D grid; NEX must be divisible by PX/PY).
#              Larger G is legal but the serial per-slice neighbour search in
#              xgenerate_databases dominates (G=4, 1.33M elements on one rank:
#              > 20 min of CPU preprocessing before the GPU solver starts)
#     weak   : fixed per-rank block (36F)x(36F)x(16F) elements (F =
#              HPCPERF_SPECFEM_LOCAL, default 2: 165,888/rank), the DOMAIN is
#              extended PX x PY times at fixed resolution (element shape, DT
#              and per-step work per rank identical for every N); source and
#              stations stay in the first block
#   HPCPERF_SPECFEM_STEPS     NSTEP for strong/weak (default 1000; smoke keeps 5000)
#
# Derived inputs (class A): Par_file gets NPROC / GPU_MODE / NSTEP / DT (and
# SAVE_MESH_FILES=.false. for strong/weak to skip VTK mesh dumps); the mesher's
# Mesh_Par_file / interfaces.txt get the sizes above. Upstream files untouched.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
l3_paths specfem3d
BIN="$L3_INSTALL/bin"
for x in xdecompose_mesh xmeshfem3D xgenerate_databases xspecfem3D; do
    [ -x "$BIN/$x" ] || { echo "run.sh: $BIN/$x missing -- run ./build.sh $BACKEND first" >&2; exit 1; }
done
EX="$R/_upstream/level3/specfem3d/EXAMPLES/applications/homogeneous_halfspace"
[ -f "$EX/DATA/Par_file" ] || { echo "run.sh: $EX missing (run fetch.sh)" >&2; exit 1; }

N_RANKS="$(hpcperf_ranks specfem3d yes)" || exit 2
MODE="$(l3_scale_mode specfem3d)" || exit 2
BUILD_DIR="$R/build/level3/specfem3d/$MODEL"
RUN_DIR="$BUILD_DIR/run/$MODE.np$N_RANKS"; rm -rf "$RUN_DIR"; mkdir -p "$RUN_DIR/OUTPUT_FILES/DATABASES_MPI"
cp -r "$EX/DATA" "$RUN_DIR/DATA"
PAR="$RUN_DIR/DATA/Par_file"
sed -i -e "s/^NPROC  *=.*/NPROC                           = $N_RANKS/" -e "s/^GPU_MODE  *=.*/GPU_MODE                        = .true./" "$PAR"

case "$MODE" in
    smoke)
        NEX=36; NZ=16; STEPS=5000; DT=0.05; PX=1; PY=1; DESC="CUBIT mesh MESH-default, SCOTCH partition into $N_RANKS" ;;
    strong)
        G="${HPCPERF_SPECFEM_STRONG:-2}"; NEX=$((36 * G)); NZ=$((16 * G)); STEPS="${HPCPERF_SPECFEM_STEPS:-1000}"
        DT="$(python3 -c "print(0.05/$G)")"
        TOPO="$(hpcperf_topology specfem3d "$N_RANKS" --dims 2 --divides "$NEX,$NEX,0")" || exit 2
        read -r PX PY _ <<< "$TOPO"; NEX_XI=$NEX; NEX_ETA=$NEX; LX=134000.0; LY=134000.0
        DESC="xmeshfem3D ${NEX_XI}x${NEX_ETA}x${NZ} on 134x134x60 km, NPROC_XI x NPROC_ETA = ${PX}x${PY}" ;;
    weak)
        F="${HPCPERF_SPECFEM_LOCAL:-2}"; NZ=$((16 * F)); STEPS="${HPCPERF_SPECFEM_STEPS:-1000}"
        DT="$(python3 -c "print(0.05/$F)")"
        TOPO="$(hpcperf_topology specfem3d "$N_RANKS" --dims 2)" || exit 2
        read -r PX PY _ <<< "$TOPO"; NEX_XI=$((36 * F * PX)); NEX_ETA=$((36 * F * PY))
        LX="$(python3 -c "print(134000.0*$PX)")"; LY="$(python3 -c "print(134000.0*$PY)")"
        DESC="xmeshfem3D ${NEX_XI}x${NEX_ETA}x${NZ} on $((134 * PX))x$((134 * PY))x60 km, NPROC_XI x NPROC_ETA = ${PX}x${PY}" ;;
esac
if [ "$MODE" != smoke ]; then
    ELEMS=$((NEX_XI * NEX_ETA * NZ))
    sed -i -e "s/^NSTEP  *=.*/NSTEP                           = $STEPS/" -e "s/^DT  *=.*/DT                              = $DT/" \
           -e "s/^SAVE_MESH_FILES  *=.*/SAVE_MESH_FILES                 = .false./" "$PAR"
    mkdir -p "$RUN_DIR/DATA/meshfem3D_files"
    cp "$EX/meshfem3D_files/interface1.txt" "$RUN_DIR/DATA/meshfem3D_files/"
    sed -e "s/^ 16\$/ $NZ/" "$EX/meshfem3D_files/interfaces.txt" > "$RUN_DIR/DATA/meshfem3D_files/interfaces.txt"
    sed -e "s/^LATITUDE_MAX  *=.*/LATITUDE_MAX                    = $LY/" -e "s/^LONGITUDE_MAX  *=.*/LONGITUDE_MAX                   = $LX/" \
        -e "s/^NEX_XI  *=.*/NEX_XI                          = $NEX_XI/" -e "s/^NEX_ETA  *=.*/NEX_ETA                         = $NEX_ETA/" \
        -e "s/^NPROC_XI  *=.*/NPROC_XI                        = $PX/" -e "s/^NPROC_ETA  *=.*/NPROC_ETA                       = $PY/" \
        -e "s/^CREATE_VTK_FILES  *=.*/CREATE_VTK_FILES                = .false./" \
        -e "s/^1  *36  *1  *36  *1  *16  *1\$/1 $NEX_XI 1 $NEX_ETA 1 $NZ 1/" \
        "$EX/meshfem3D_files/Mesh_Par_file" > "$RUN_DIR/DATA/meshfem3D_files/Mesh_Par_file"
    grep -q "^1 $NEX_XI 1 $NEX_ETA 1 $NZ 1" "$RUN_DIR/DATA/meshfem3D_files/Mesh_Par_file" || { echo "run.sh: failed to rewrite the mesh region line" >&2; exit 1; }
else
    ELEMS=20736
fi

echo "# SPECFEM3D $BACKEND: mode=$MODE ranks=$N_RANKS elements=$ELEMS (~$((ELEMS / N_RANKS))/rank) NSTEP=$STEPS DT=$DT mesh: $DESC run_dir=$RUN_DIR"
cd "$RUN_DIR"
LAUNCH=("$L3_LAUNCHER" --gpus "$N_RANKS" --bind wrapper --)
if [ "$MODE" = smoke ]; then
    if [ -z "${HPCPERF_DRY_RUN:-}" ]; then
        echo "# stage 1/3: xdecompose_mesh $N_RANKS (serial, CPU, SCOTCH)"
        "$BIN/xdecompose_mesh" "$N_RANKS" "$EX/MESH-default" OUTPUT_FILES/DATABASES_MPI > OUTPUT_FILES/output_decompose_mesh.txt 2>&1 \
            || { tail -20 OUTPUT_FILES/output_decompose_mesh.txt; echo "run.sh: xdecompose_mesh failed" >&2; exit 1; }
    else
        echo "# stage 1/3: xdecompose_mesh $N_RANKS (serial, CPU) -- skipped in dry-run"
    fi
else
    echo "# stage 1/3: xmeshfem3D on $N_RANKS ranks (CPU)"
    "${LAUNCH[@]}" "$BIN/xmeshfem3D" > OUTPUT_FILES/output_meshfem3D.log 2>&1 || { tail -30 OUTPUT_FILES/output_meshfem3D.log; echo "run.sh: xmeshfem3D failed" >&2; exit 1; }
    grep -E 'hpcperf-launch: (actual|HYPOTHETICAL|launch|command|dry-run|NOTE)' OUTPUT_FILES/output_meshfem3D.log || true
fi
echo "# stage 2/3: xgenerate_databases on $N_RANKS ranks (CPU)"
"${LAUNCH[@]}" "$BIN/xgenerate_databases" > OUTPUT_FILES/output_generate_databases.log 2>&1 || { tail -30 OUTPUT_FILES/output_generate_databases.log; echo "run.sh: xgenerate_databases failed" >&2; exit 1; }
grep -E 'hpcperf-launch: (dry-run)' OUTPUT_FILES/output_generate_databases.log || true
echo "# stage 3/3: xspecfem3D on $N_RANKS ranks (GPU_MODE)"
t0=$(date +%s)
"${LAUNCH[@]}" "$BIN/xspecfem3D" 2>&1 | tee OUTPUT_FILES/output_specfem3D.log | grep -E 'hpcperf-launch|Error|ERROR|GPU|Time loop|Elapsed|End of' || true
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "run.sh: xspecfem3D exited $rc (see $RUN_DIR/OUTPUT_FILES/output_specfem3D.log)" >&2; exit "$rc"; }
[ -n "${HPCPERF_DRY_RUN:-}" ] && exit 0
echo "# solver wall time $(( $(date +%s)-t0 )) s; $(grep -E 'Total elapsed time in seconds|Time loop finished' OUTPUT_FILES/output_solver.txt 2>/dev/null | tr -s ' ' | tr '\n' ';')"
echo "# seismograms: $(ls OUTPUT_FILES/*.semd 2>/dev/null | wc -l) files in $RUN_DIR/OUTPUT_FILES"
