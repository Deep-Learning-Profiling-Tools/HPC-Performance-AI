#!/usr/bin/env bash
# Run Nekbone (CUDA: OpenACC + CUDA Fortran build) on one registered input.
#
#   ./run.sh [CUDA] [ID]
#   HPCPERF_NEKBONE_INPUT=<id>   the registered input (inputs.yaml); refused together with an ID argument
#   HPCPERF_GPUS=N|all           MPI ranks = GPUs (one rank per GPU through the common launcher, default 1)
#
# Default input: the registry's default (inputs.yaml default_input). An input is a directory inputs/<id>/ with
# the SIZE file compiled into build/level2/nekbone/cuda/<id>/nekbone (build.sh) and the data.rea the program
# reads from its working directory (src/driver.f read_param: open(file='data.rea')). Nekbone writes nothing
# but stdout, so the program runs with the registered input directory as its working directory and reads the
# registered data.rea itself -- no copy, no generated file. Its single argument ("data", the session name of
# the Nek5000 convention) is not read by this program; upstream runs `mpiexec -n 1 ./nekbone data`.
#
# data.rea gives the element count per rank and the polynomial order (one value each in the registered
# inputs), so HPCPERF_GPUS=N runs N ranks of that per-rank size (weak scaling, as upstream's `lp` allows);
# the registered workloads are the one-rank runs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$R" ] && [ -f "$R/hpcperf_env.sh" ]; then
    set +eu
    # shellcheck disable=SC1091
    source "$R/hpcperf_env.sh" 2>/dev/null
    set -eu
fi
# Registered inputs: HPCPERF_NEKBONE_INPUT=<id> (tools/inputs/README.md)
# shellcheck disable=SC1091
source "$R/tools/inputs/hpcperf_input_selector.sh"
hpcperf_apply_input "$HERE" HPCPERF_NEKBONE_INPUT || exit 2

BACKEND="CUDA"
case "$(printf '%s' "${1:-}" | tr '[:lower:]' '[:upper:]')" in
    CUDA) shift ;;
    CPU)  echo "run.sh: the CPU build is the reference build (validate.sh, check_cg.py --make-reference), not a measured backend" >&2; exit 2 ;;
    HIP)  echo "run.sh: no HIP build of Nekbone (OpenACC + CUDA Fortran port)" >&2; exit 1 ;;
esac
if [ -n "${HPCPERF_NEKBONE_INPUT:-}" ]; then
    [ $# -eq 0 ] || { echo "run.sh: HPCPERF_NEKBONE_INPUT=$HPCPERF_NEKBONE_INPUT is mutually exclusive with an ID argument" >&2; exit 2; }
    ID="$HPCPERF_NEKBONE_INPUT"
else
    case $# in
        0) ID="$(sed -n 's/^default_input:[[:space:]]*//p' "$HERE/inputs.yaml" | head -1)" ;;
        1) ID="$1" ;;
        *) echo "run.sh: at most one input ID (got $#: $*)" >&2; exit 2 ;;
    esac
fi
CASE="$HERE/inputs/$ID"
[ -f "$CASE/data.rea" ] && [ -f "$CASE/SIZE" ] || { echo "run.sh: no registered input directory $CASE (SIZE + data.rea)" >&2; exit 2; }
BUILD="$R/build/level2/nekbone/cuda/$ID"
EXE="$BUILD/nekbone"
[ -x "$EXE" ] || { echo "run.sh: $EXE not found -- run $HERE/build.sh CUDA $ID first" >&2; exit 1; }
# the executable must be the build of THIS input's SIZE (build.sh keeps the copy it compiled)
cmp -s "$CASE/SIZE" "$BUILD/SIZE" || { echo "run.sh: $BUILD/SIZE differs from $CASE/SIZE -- rebuild with build.sh CUDA $ID" >&2; exit 1; }

# shellcheck disable=SC1091
source "$R/level2/tools/hpcperf_launch_common.sh"
N_RANKS="$(hpcperf_ranks nekbone no)" || exit 2
LP="$(grep -o 'lp *= *[0-9]*' "$CASE/SIZE" | head -1 | tr -dc '0-9')"
if [ -n "$LP" ] && [ "$N_RANKS" -gt "$LP" ]; then
    echo "run.sh: HPCPERF_GPUS=$N_RANKS exceeds the compiled rank maximum lp=$LP of $CASE/SIZE" >&2; exit 2
fi
cd "$CASE"
echo "== Nekbone $BACKEND: input=$ID ranks=$N_RANKS exe=$EXE (SIZE $(grep -o 'lx1=[0-9]*' SIZE | head -1) $(grep -o 'lelt=[0-9]*' SIZE | head -1) compiled in) data.rea read from $CASE"
exec "$HPCPERF_LAUNCHER_BIN" --gpus "$N_RANKS" --bind wrapper -- "$EXE" data
