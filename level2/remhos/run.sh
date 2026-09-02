#!/usr/bin/env bash
# Run the Remhos GPU benchmark configuration on the CUDA or HIP build.
#
#   ./run.sh [CUDA|HIP] [extra remhos args...]
#
# Default problem: upstream README sample/verification run 11 (3D remap of a
# discontinuous field on data/cube01_hex.mesh, Taylor-Green mesh deformation),
#
#   remhos -m data/cube01_hex.mesh -p 10 -rs 1 -o 2 -dt 0.02 -tf 0.8 ...
#
# refined three more times (-rs 4: 32768 hexes, 884,736 unknowns) with the
# time step scaled with the mesh size (-dt 0.0025, 400 RK3 steps), and run in
# the only solver combination Remhos supports on a GPU and tracks its FOM for
# (README "Performance Timing and FOM"): high-order local inverse (-ho 3),
# mass-based low-order (-lo 5), clip-and-scale FCT (-fct 2), partial assembly
# (-pa), MFEM device "cuda" (or "hip"). One MPI rank. ~25 s wall clock on a
# B200 (~6 s of it in the timed kernels; the rest is serial mesh setup and
# untimed per-step work); ~1 GB of device memory.
#
# Remhos prints the per-phase kernel times and FOMs (megadofs x time steps /
# second) and the final mass / max value; validate.sh checks the latter.
#
# NOTE on -tf in remap mode (-p 10..19): the pseudo-time always runs 0..1, but
# -tf sets how long the Taylor-Green velocity is integrated to build the target
# mesh, i.e. how strongly the mesh is deformed. The default (-tf 4) tangles the
# refined cube01_hex meshes and the run blows up; the README uses -tf 0.8.
#
# NOTE on the FOM: dofs x (steps x RK stages) is accumulated in HYPRE_BigInt,
# a 32-bit int in this hypre build (no --enable-bigint), so configurations with
# unknowns x steps x 3 > 2^31 print a negative FOM (e.g. -rs 4 -o 3 -dt 0.00175).
# The default stays below that (884736 x 1200 = 1.06e9).
#
# Extra args are appended to the command line. MFEM's OptionsParser rejects an
# option given twice, so do not repeat -m/-p/-rs/-o/-dt/-tf/-ho/-lo/-fct/-pa/-d
# on the command line -- use the environment overrides instead.
#
# Environment overrides:
#   HPCPERF_REMHOS_RS     serial refinement levels (default 4)
#   HPCPERF_REMHOS_ORDER  polynomial order (default 2)
#   HPCPERF_REMHOS_DT     time step (default 0.0025; scale with 2^-RS)
#   HPCPERF_REMHOS_TF     mesh-deformation time (default 0.8)
#   HPCPERF_NP            number of MPI ranks (default 1)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true

set -euo pipefail

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
[ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level2/remhos/$MODEL"
EXE="$BUILD_DIR/remhos"

case "$BACKEND" in
    CUDA) DEVICE=cuda ;;
    HIP)  DEVICE=hip ;;
    *) echo "usage: $0 [CUDA|HIP] [extra remhos args]" >&2; exit 2 ;;
esac

if [ ! -x "$EXE" ]; then
    echo "run.sh: $EXE not found -- run ./build.sh $BACKEND first" >&2
    exit 1
fi

RS="${HPCPERF_REMHOS_RS:-4}"
ORDER="${HPCPERF_REMHOS_ORDER:-2}"
DT="${HPCPERF_REMHOS_DT:-0.0025}"
TF="${HPCPERF_REMHOS_TF:-0.8}"
NP="${HPCPERF_NP:-1}"

LAUNCH=(mpirun -np "$NP")
# mpirun inside this Slurm allocation needs --oversubscribe for >1 rank.
[ "$NP" -gt 1 ] && LAUNCH+=(--oversubscribe)

# Remhos writes any optional output files (-save, -visit, errors.txt, si_init.gf) to the
# current directory: run inside the build tree, not in level2/remhos.
RUN_DIR="$BUILD_DIR/run"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

ARGS=(-m "$HERE/data/cube01_hex.mesh" -p 10 -rs "$RS" -o "$ORDER" -dt "$DT" -tf "$TF"
      -ho 3 -lo 5 -fct 2 -pa -d "$DEVICE" -no-vis)

echo "# Remhos $BACKEND: ${LAUNCH[*]} $EXE ${ARGS[*]} $*"
exec "${LAUNCH[@]}" "$EXE" "${ARGS[@]}" "$@"
