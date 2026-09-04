#!/usr/bin/env bash
# Run the standard P3-miniapps heat3d benchmark problem (upstream docs/heat3d.md example).
#
#   ./run.sh [CUDA|HIP] [extra heat3d args]
#
# Default problem: --nx 512 --ny 512 --nz 512 --nbiter 1000 --freq_diag 0
# (512^3 cells, 1000 FTCS steps, no CSV diagnostics; ~4-5 s main loop on a B200).
# Any extra arguments are appended to the heat3d command line, e.g.
#   ./run.sh CUDA --nx 256 --ny 256 --nz 256 --nbiter 500
# The run happens in  $R/build/level2/p3_heat3d/<cuda|hip>/run/  so that the
# optional CSV output (--freq_diag N, written to ./data/heat3d/) stays out of
# the source tree.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$R" ] && [ -f "$R/hpcperf_env.sh" ]; then
    set +eu
    # shellcheck disable=SC1091
    source "$R/hpcperf_env.sh" 2>/dev/null
    set -eu
fi

BACKEND="$(printf '%s' "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
case "$BACKEND" in
    CUDA|HIP) shift || true ;;
    *) echo "usage: $0 [CUDA|HIP] [extra heat3d args]" >&2; exit 2 ;;
esac

BUILD_DIR="$R/build/level2/p3_heat3d/$(printf '%s' "$BACKEND" | tr '[:upper:]' '[:lower:]')"
EXE="$BUILD_DIR/miniapps/heat3d/thrust/heat3d"
if [ ! -x "$EXE" ]; then
    echo "error: $EXE not found; run ./build.sh $BACKEND first" >&2
    exit 1
fi

RUN_DIR="$BUILD_DIR/run"
mkdir -p "$RUN_DIR/data/heat3d"
cd "$RUN_DIR"

echo "== P3-miniapps heat3d ($BACKEND): $EXE --nx 512 --ny 512 --nz 512 --nbiter 1000 --freq_diag 0 $*"
exec "$EXE" --nx 512 --ny 512 --nz 512 --nbiter 1000 --freq_diag 0 "$@"
