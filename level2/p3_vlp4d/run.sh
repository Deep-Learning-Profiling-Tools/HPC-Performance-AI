#!/usr/bin/env bash
# Run the standard P3-miniapps vlp4d benchmark problem (upstream docs/vlp4d.md example).
#
#   ./run.sh [CUDA|HIP] [input deck] [extra vlp4d args]
#
# Default input deck: wk/SLD10_large.dat -- 2D Landau damping (test case 10),
# 128^4 phase-space points, dt = 0.125, 128 time steps (~11 s main loop on a B200).
# The smaller wk/SLD10.dat (dt = 0.01, 40 steps) can be given instead.
# vlp4d writes its diagnostics file nrj.out (time, log||E||_2, "mass") into the
# current directory, so the run happens in
#   $R/build/level2/p3_vlp4d/<cuda|hip>/run/
# (the optional data/vlp4d/ directory for the fxvx CSV dump is created there too).
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
    *) echo "usage: $0 [CUDA|HIP] [input deck] [extra vlp4d args]" >&2; exit 2 ;;
esac

BUILD_DIR="$R/build/level2/p3_vlp4d/$(printf '%s' "$BACKEND" | tr '[:upper:]' '[:lower:]')"
EXE="$BUILD_DIR/miniapps/vlp4d/thrust/vlp4d"
if [ ! -x "$EXE" ]; then
    echo "error: $EXE not found; run ./build.sh $BACKEND first" >&2
    exit 1
fi

# Input deck: first extra argument if given (resolved relative to the caller's
# cwd, then relative to wk/), else the standard SLD10_large.dat.
if [ $# -gt 0 ]; then
    DECK="$1"; shift
    if [ ! -f "$DECK" ] && [ -f "$HERE/wk/$DECK" ]; then DECK="$HERE/wk/$DECK"; fi
    DECK="$(cd "$(dirname "$DECK")" && pwd)/$(basename "$DECK")"
else
    DECK="$HERE/wk/SLD10_large.dat"
fi

RUN_DIR="$BUILD_DIR/run"
mkdir -p "$RUN_DIR/data/vlp4d"
cd "$RUN_DIR"

echo "== P3-miniapps vlp4d ($BACKEND): $EXE $DECK $*   (cwd: $RUN_DIR, diagnostics -> nrj.out)"
exec "$EXE" "$DECK" "$@"
