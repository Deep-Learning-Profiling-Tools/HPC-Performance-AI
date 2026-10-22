#!/usr/bin/env bash
# Validate the PENNANT CUDA build against upstream's gold-standard profiles.
#
#   ./validate.sh [CUDA]
#
# Runs the five upstream test decks that ship a gold standard file (test/<name>/<name>.xy.std: the per-zone
# density / energy / pressure profile after the deck's 10 cycles, 8 significant digits, doc/pennantdoc.tex
# "Test inputs and outputs") through run.sh and compares every zone of every column with check_xy.py:
#   sedovsmall  (9 x 9 rect, 81 zones)        sedov   (45 x 45 rect, 2025 zones)
#   nohsmall    (2 x 20 pie, 40 zones)         noh     (30 x 100 pie, 3000 zones)
#   leblanc     (10 x 90 rect, 900 zones)
# Rule (check_xy.py): populated zones (|ref| >= 1e-8 of the column's max) within 1e-6 relative -- the profile's
# 8 printed digits; zones below that floor (roundoff residues of the undisturbed near-vacuum region) within
# 1e-8 of the column's max absolute. Every run must exit 0, print "Run complete" and "hydro cycle run time",
# and report "Running Hydro on device..." (the GPU path). Prints PASS/FAIL at the end; exit 0/1.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$R" ] && [ -f "$R/hpcperf_env.sh" ]; then
    set +u
    # shellcheck disable=SC1091
    source "$R/hpcperf_env.sh" 2>/dev/null
    set -u
fi
BACKEND="$(printf '%s' "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
case "$BACKEND" in
    CUDA) ;;
    HIP) echo "validate.sh: no HIP build of PENNANT here"; echo "PENNANT HIP validation: FAIL"; exit 1 ;;
    *) echo "usage: $0 [CUDA]" >&2; exit 2 ;;
esac
BUILD_DIR="$R/build/level2/pennant/cuda"
EXE="$BUILD_DIR/build/pennant"
if [ ! -x "$EXE" ]; then
    echo "validate.sh: $EXE not found -- run $HERE/build.sh CUDA first" >&2
    echo "PENNANT CUDA validation: FAIL"
    exit 1
fi
unset HPCPERF_PENNANT_INPUT     # validate.sh names its decks itself

fail=0
for name in sedovsmall nohsmall sedov noh leblanc; do
    deck="$HERE/test/$name/$name.pnt"; gold="$HERE/test/$name/$name.xy.std"
    log="$BUILD_DIR/validate_$name.log"
    echo "== $name ($deck)"
    [ -f "$gold" ] || { echo "   -> FAILED: gold file $gold missing"; fail=1; continue; }
    "$HERE/run.sh" "$BACKEND" "$deck" > "$log" 2>&1
    rc=$?
    grep -E '^Running Hydro on device|^Zones:|^cycle =|^time  =|hydro cycle run time' "$log" | sed 's/^/   /'
    if [ "$rc" -ne 0 ]; then echo "   -> FAILED: run.sh exited $rc (see $log)"; fail=1; continue; fi
    if ! grep -q '^Run complete' "$log" || ! grep -q 'hydro cycle run time' "$log"; then
        echo "   -> FAILED: incomplete run (no 'Run complete' / run time; see $log)"; fail=1; continue
    fi
    if ! grep -q '^Running Hydro on device' "$log"; then
        echo "   -> FAILED: the GPU path banner 'Running Hydro on device...' is missing (see $log)"; fail=1; continue
    fi
    xy="$BUILD_DIR/run/$name/$name.xy"
    if python3 "$HERE/check_xy.py" --run "$xy" --reference "$gold" --label "$name vs upstream gold" | sed 's/^/   /' | tee -a "$log" | tail -n +1 | grep -q '^   PASS: PENNANT xy check'; then
        echo "   -> $name: PASSED (exit $rc)"
    else
        echo "   -> $name: FAILED (profile differs from the gold standard; see $log)"; fail=1
    fi
done

if [ "$fail" -eq 0 ]; then
    echo "PENNANT CUDA validation: PASS"
    exit 0
else
    echo "PENNANT CUDA validation: FAIL"
    exit 1
fi
