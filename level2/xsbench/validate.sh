#!/usr/bin/env bash
# Validate XSBench against upstream's built-in verification checksums.
#
#   ./validate.sh [CUDA|HIP]          (default: CUDA)
#
# XSBench always computes a verification hash over all lookup results
# (Simulation.cu: per-lookup hash of the 5 macroscopic XS values, reduced over
# all lookups, then % 999983 in Main.cu) and compares it to hard-coded
# reference values in io.cu:print_results() for the default problem sizes.
# Reference values for the event-based method (the only one the CUDA/HIP
# sources implement):
#
#   -m event -s large  (355 nuclides, 17,000,000 lookups) -> 952131
#   -m event -s small  ( 68 nuclides, 17,000,000 lookups) -> 945990
#
# This script runs both problems, requires exit code 0 and the line
# "Verification checksum: <ref> (Valid)" with the expected <ref>, and prints
# a final PASS/FAIL line (exit 0/1). Full program output is kept in
# $R/build/level2/xsbench/<backend>/validate_<size>.log.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

# Load the project environment first (see build.sh for why it comes first).
if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$R" ] && [ -f "$R/hpcperf_env.sh" ]; then
    set +u
    # shellcheck disable=SC1091
    source "$R/hpcperf_env.sh" 2>/dev/null
    set -u
fi

BACKEND="$(printf '%s' "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
case "$BACKEND" in
    CUDA|HIP) ;;
    *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac
BUILD_DIR="$R/build/level2/xsbench/$(printf '%s' "$BACKEND" | tr '[:upper:]' '[:lower:]')"
EXE="$BUILD_DIR/XSBench"

if [ ! -x "$EXE" ]; then
    echo "error: $EXE not found -- run $HERE/build.sh $BACKEND first" >&2
    echo "FAIL: xsbench $BACKEND (not built)"
    exit 1
fi

cd "$BUILD_DIR"

fail=0
check_case() { # $1 = -s size, $2 = expected checksum
    local size="$1" expect="$2" log="$BUILD_DIR/validate_${1}.log" rc line got
    echo "== XSBench -m event -s $size   (reference checksum $expect)"
    ./XSBench -m event -s "$size" > "$log" 2>&1
    rc=$?
    grep -E "^(Runtime|Lookups|Lookups/s):" "$log" || true
    line="$(grep -E '^Verification checksum:' "$log" || true)"
    echo "$line"
    got="$(printf '%s' "$line" | sed -nE 's/^Verification checksum: ([0-9]+) .*/\1/p')"
    if [ "$rc" -eq 0 ] && [ "$got" = "$expect" ] && printf '%s' "$line" | grep -q '(Valid)'; then
        echo "   ok  (exit $rc, checksum $got == $expect)"
    else
        echo "   MISMATCH  (exit $rc, checksum '${got:-none}' != $expect) -- see $log"
        fail=1
    fi
}

check_case large 952131
check_case small 945990

if [ "$fail" -eq 0 ]; then
    echo "PASS: xsbench $BACKEND (event-based large=952131, small=945990 verification checksums match upstream reference)"
    exit 0
else
    echo "FAIL: xsbench $BACKEND (verification checksum mismatch or non-zero exit)"
    exit 1
fi
