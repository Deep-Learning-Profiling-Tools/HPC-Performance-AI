#!/usr/bin/env bash
# Run the standard Laghos benchmark problem on the CUDA or HIP build.
#
#   ./run.sh [CUDA|HIP] [extra laghos args...]
#
# Distributed model: MFEM parallel FEM -- the serially-refined mesh is
# partitioned over the MPI ranks (METIS), one rank per GPU, halo/assembly
# communication through hypre. Multi-rank launches go through
# level2/tools/hpcperf_mpi_launch.sh, which also injects the per-rank GPU
# binding wrapper (MFEM does not map ranks to GPUs itself) -- no manual
# wrapping needed.
#
# Resource / size controls (common Level 2 parameters):
#   HPCPERF_GPUS=N|all     ranks = GPUs (default 1)
#   HPCPERF_SCALE_MODE     smoke | strong | weak    (default strong)
#     strong : ONE fixed global problem: 3D Sedov (-p 1) on cube01_hex.mesh
#              at -rs HPCPERF_LAGHOS_RS (default 4: 32,768 hexes, 823,875 H1
#              dofs, 2408 steps -- the historical single-GPU FOM deck),
#              partitioned over the ranks
#     weak   : per-rank work held EXACTLY constant with upstream's
#              -epm (elements per MPI rank) partitioner: every rank owns
#              HPCPERF_LAGHOS_EPM elements (default 32768 = the single-GPU
#              strong deck's zone count), any rank count
#     smoke  : -rs 2 (512 hexes), -tf 0.05 -- seconds-fast bring-up run
#   HPCPERF_LAGHOS_RS      serial refinements of cube01_hex.mesh (default 4;
#                          elements = 8^(rs+1) / 8 * 8 = 8 * 8^rs)
#   HPCPERF_LAGHOS_ARGS    replaces the whole problem line (except -d)
#   HPCPERF_LAGHOS_DEVICE  MFEM device string (default cuda/hip by backend)
#
# Constraint checked here: the serial mesh must have at least as many
# elements as ranks (METIS partition, one nonempty part per rank):
# elements(rs) = 8 * 8^rs  (cube01_hex.mesh has 8 hexes at rs=0).
#
# Compatibility patch note: laghos_solver.cpp carries a 13-line MFEM_UNROLL(1)
# Blackwell/CUDA 13.2 compiler workaround (see README, performance impact
# pending A/B validation); nothing here changes it.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true
set -euo pipefail
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
[ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level2/laghos/$MODEL"
EXE="$BUILD_DIR/laghos"
if [ ! -x "$EXE" ]; then
    echo "run.sh: $EXE not found -- run ./build.sh $BACKEND first" >&2
    exit 1
fi
case "$BACKEND" in
    CUDA) DEVICE="${HPCPERF_LAGHOS_DEVICE:-cuda}" ;;
    HIP)  DEVICE="${HPCPERF_LAGHOS_DEVICE:-hip}" ;;
    *) echo "usage: $0 [CUDA|HIP] [extra laghos args]" >&2; exit 2 ;;
esac

# shellcheck disable=SC1091
source "$R/level2/tools/hpcperf_launch_common.sh"
N_RANKS="$(hpcperf_ranks laghos yes)" || exit 2
hpcperf_forbid_args laghos -m -rs -rp -epm -nx -ny -nz -dev -d -- "$@" || exit 2   # mesh/size/device only via the checked variables
MODE="${HPCPERF_SCALE_MODE:-strong}"

if [ -n "${HPCPERF_LAGHOS_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    PROBLEM=(${HPCPERF_LAGHOS_ARGS})
else
    RS="${HPCPERF_LAGHOS_RS:-4}"
    case "$MODE" in
        smoke)  RS="${HPCPERF_LAGHOS_RS:-2}"
                PROBLEM=(-p 1 -m data/cube01_hex.mesh -E0 2 -rs "$RS" -tf 0.05 -pa) ;;
        strong) PROBLEM=(-p 1 -m data/cube01_hex.mesh -E0 2 -rs "$RS" -tf 0.6 -pa -f) ;;
        weak)   EPM="${HPCPERF_LAGHOS_EPM:-32768}"
                PROBLEM=(-p 1 -epm "$EPM" -E0 2 -tf 0.6 -pa -f) ;;
        *) echo "run.sh: HPCPERF_SCALE_MODE must be smoke|strong|weak (got '$MODE')" >&2; exit 2 ;;
    esac
    if [ "$MODE" != weak ]; then
        ELEMS=$((8 ** (RS + 1)))
        if [ "$ELEMS" -lt "$N_RANKS" ]; then
            echo "run.sh: serial mesh at -rs $RS has only $ELEMS elements for $N_RANKS ranks;" >&2
            echo "run.sh: raise HPCPERF_LAGHOS_RS (elements = 8^(rs+1)) or use fewer ranks." >&2
            exit 2
        fi
        echo "# Laghos $BACKEND: mode=$MODE ranks=$N_RANKS rs=$RS serial-elements=$ELEMS (~$((ELEMS / N_RANKS))/rank)"
    else
        echo "# Laghos $BACKEND: mode=weak ranks=$N_RANKS elements/rank=$EPM (global $((EPM * N_RANKS)))"
    fi
fi

cd "$HERE"
exec "$R/level2/tools/hpcperf_mpi_launch.sh" --gpus "$N_RANKS" --bind wrapper -- \
    "$EXE" "${PROBLEM[@]}" -d "$DEVICE" "$@"
