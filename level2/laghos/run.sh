#!/usr/bin/env bash
# Run the standard Laghos benchmark problem on the CUDA or HIP build.
#
#   ./run.sh [CUDA|HIP] [extra laghos args...]
#
# Default problem: upstream's main problem of interest, the 3D Sedov blast
# wave (-p 1) with partial assembly (-pa) on data/cube01_hex.mesh, i.e. the
# README sample run / verification run 4
#     -p 1 -m data/cube01_hex.mesh -E0 2 -rs 2 -tf 0.6 -pa
# refined two more levels (-rs 4: 32768 Q2-Q1 hexahedra, 823,875 kinematic and
# 262,144 thermodynamic dofs, 2408 RK4 time steps) and run with `-f` so that
# Laghos prints its figure-of-merit table. About 2 min 20 s on a B200, of
# which ~121 s are in the timed "major kernels" (see README).
#
# Extra args are appended to the command line; MFEM's OptionsParser lets the
# last occurrence win, so e.g. `./run.sh CUDA -rs 3` or `./run.sh CUDA -ok 3
# -ot 2` override the defaults, and `./run.sh CUDA -ms 200` caps the number of
# time steps.
#
# Environment overrides:
#   HPCPERF_LAGHOS_RS    serial refinement levels of cube01_hex.mesh (default 4)
#   HPCPERF_LAGHOS_ARGS  replaces the whole default problem line (everything
#                        except -d <device>); e.g.
#                        HPCPERF_LAGHOS_ARGS="-p 3 -m data/box01_hex.mesh -rs 3 -tf 5.0 -pa -cgt 1e-12"
#   HPCPERF_NP           number of MPI ranks (default 1; >1 adds --oversubscribe,
#                        all ranks share the one visible GPU)
#   HPCPERF_LAGHOS_DEVICE  MFEM device string (default: cuda / hip by backend)
#
# The script cd's into the source directory so that the relative mesh paths
# (`-m data/...`) used by upstream's documentation resolve.

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

RS="${HPCPERF_LAGHOS_RS:-4}"
DEFAULT_ARGS="-p 1 -m data/cube01_hex.mesh -E0 2 -rs $RS -tf 0.6 -pa -f"
# shellcheck disable=SC2206
PROBLEM=(${HPCPERF_LAGHOS_ARGS:-$DEFAULT_ARGS})

NP="${HPCPERF_NP:-1}"
LAUNCH=(mpirun -np "$NP")
# mpirun inside this Slurm allocation needs --oversubscribe for >1 rank.
[ "$NP" -gt 1 ] && LAUNCH+=(--oversubscribe)

cd "$HERE"
echo "# Laghos $BACKEND: ${LAUNCH[*]} $EXE ${PROBLEM[*]} -d $DEVICE $*"
exec "${LAUNCH[@]}" "$EXE" "${PROBLEM[@]}" -d "$DEVICE" "$@"
