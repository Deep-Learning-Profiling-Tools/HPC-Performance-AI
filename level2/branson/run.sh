#!/usr/bin/env bash
# Run the standard Branson benchmark problem on the GPU.
#
#   ./run.sh [CUDA|HIP] [extra BRANSON args...]      (default: CUDA)
#
# Runs upstream's documented single-node problem
#   inputs/3D_hohlraum_single_node.xml
# (3D hohlraum, 591 500 cells, 10 M photons, 5 time steps, replicated mesh,
# event-based GPU transport) as `mpirun -np 1 BRANSON <deck>` and prints
# Branson's normal output (per-step conservation diagnostics, "Total
# transport:" time and the "Photons Per Second (FOM)" figure of merit).
# About 45 s wall on one B200.
#
# Extra arguments are appended to the BRANSON command line; Branson accepts
# overrides of the deck's <common> block as --<name> <value>, e.g.
#   ./run.sh CUDA --photons 1000000 --t-stop 0.5 --seed 42
# A different deck can be run directly:
#   mpirun -np 1 $R/build/level2/branson/cuda/BRANSON <deck.xml> [--opts]
#
# Environment knobs:
#   HPCPERF_NP       number of MPI ranks (default 1)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

BACKEND="$(printf '%s' "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
case "$BACKEND" in
    CUDA|HIP) [ $# -gt 0 ] && shift ;;
    *) echo "usage: $0 [CUDA|HIP] [extra BRANSON args]" >&2; exit 2 ;;
esac

if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$R" ] && [ -f "$R/hpcperf_env.sh" ]; then
    set +eu
    # shellcheck disable=SC1091
    source "$R/hpcperf_env.sh" 2>/dev/null
    set -eu
fi

BUILD="$R/build/level2/branson/$(printf '%s' "$BACKEND" | tr '[:upper:]' '[:lower:]')"
EXE="$BUILD/BRANSON"
DECK="$HERE/inputs/3D_hohlraum_single_node.xml"

if [ ! -x "$EXE" ]; then
    echo "error: $EXE not found -- run $HERE/build.sh $BACKEND first" >&2
    exit 1
fi
if ! command -v mpirun >/dev/null 2>&1; then
    echo "error: mpirun not on PATH (source $R/hpcperf_env.sh)" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../tools" && pwd)/hpcperf_launch_common.sh"
N_RANKS="$(hpcperf_ranks branson no)" || exit 2   # HPCPERF_GPUS (or legacy HPCPERF_NP); no scale modes yet
# One rank per GPU through the common launcher. Branson's own set_device_ID()
# uses the GLOBAL rank modulo the visible device count; with the wrapper each
# rank sees exactly one device, which makes that mapping correct under any
# rank placement (--bind wrapper).
MPIRUN=("$HPCPERF_LAUNCHER_BIN" --gpus "$N_RANKS" --bind wrapper --)

echo "== Branson $BACKEND: ${MPIRUN[*]} $EXE $DECK $*"
exec "${MPIRUN[@]}" "$EXE" "$DECK" "$@"
