#!/usr/bin/env bash
# Validate the HACCabanaPM CUDA or HIP build.
#
#   ./validate.sh [CUDA|HIP]
#
# Runs upstream's demo configuration apps/demo/indat.params (NG = NP = 128,
# RL = 128 Mpc/h, z = 200 -> 50 in 5 DKD steps; ~5 s) through pm_ic + pm_run
# via run.sh, then checks the evolved snapshot against the initial one with
# apps/snapshot_check.cpp (an HPC-Performance-AI addition; upstream's only
# check of pm_run output is its GoogleTest end-to-end test, which needs GTest
# and asserts the first four items below). The checks are:
#
#   1. particle count conserved:  N(evolved) == N(ic) == attribute np
#   2. step == N_STEPS and a_out == 1/(1+Z_FIN) (1e-6 rel.), a_out > a_in
#   3. every position finite and in [0, rL) in all three components
#   4. velocities finite; masses finite, > 0 and all equal
#   5. particle ids are a permutation of [0, np) (none lost or duplicated)
#   6. total momentum: |sum_i v_i| / (N v_rms) <= 1e-6 per component
#      (antisymmetric CIC force pair; measured ~1e-10)
#   7. growth: rms density contrast of the CIC-deposited particles on
#      (ng/8)^3 cells, evolved/ic, must be within [0.5, 2.0] x the linear-theory
#      growth ratio D(a_out)/D(a_in) (flat LambdaCDM; measured 0.98 for this
#      window -- the 2 % deficit is the neglected radiation at z = 200)
#
# The GPU CIC deposit uses atomics, so runs are not bit-reproducible
# (h5diff shows ~1e-5-relative differences); a stored reference snapshot is
# therefore not used.
#
# Environment overrides:
#   HPCPERF_HACCABANAPM_INDAT  another indat (N_STEPS / Z_FIN are read from it);
#                              e.g. apps/demo/indat_bench_256.params validates
#                              the benchmark run itself (~25 s)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level2/haccabanapm/$MODEL"

export HPCPERF_HACCABANAPM_INDAT="${HPCPERF_HACCABANAPM_INDAT:-apps/demo/indat.params}"
INDAT="$HPCPERF_HACCABANAPM_INDAT"
case "$INDAT" in /*) ;; *) INDAT="$HERE/$INDAT" ;; esac
TAG="${HPCPERF_HACCABANAPM_TAG:-$(basename "${INDAT%.*}")}"
export HPCPERF_HACCABANAPM_TAG="$TAG"
export HPCPERF_HACCABANAPM_TIMING="${HPCPERF_HACCABANAPM_TIMING:-0}"

LOG_DIR="$BUILD_DIR/run"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/validate_$TAG.log"

"$HERE/run.sh" "$BACKEND" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

echo "----------------------------------------------------------------"
verdict() {
    if [ "$1" -eq 0 ]; then
        echo "HACCabanaPM $BACKEND validation ($HPCPERF_HACCABANAPM_INDAT): PASS"
        exit 0
    else
        echo "HACCabanaPM $BACKEND validation ($HPCPERF_HACCABANAPM_INDAT): FAIL ($2)"
        exit 1
    fi
}
[ "$rc" -eq 0 ] || verdict 1 "run.sh exit code $rc"
[ -x "$BUILD_DIR/snapshot_check" ] || verdict 1 "$BUILD_DIR/snapshot_check missing -- rebuild with ./build.sh $BACKEND"

NSTEPS="$(sed -n 's/^N_STEPS[[:space:]]\+\([0-9]\+\).*/\1/p' "$INDAT" | head -1)"
ZFIN="$(sed -n 's/^Z_FIN[[:space:]]\+\([0-9.eE+-]\+\).*/\1/p' "$INDAT" | head -1)"
[ -n "$NSTEPS" ] && [ -n "$ZFIN" ] || verdict 1 "could not read N_STEPS / Z_FIN from $INDAT"

"$BUILD_DIR/snapshot_check" "$LOG_DIR/ic_$TAG.h5" "$LOG_DIR/evolved_$TAG.h5" \
    --nsteps "$NSTEPS" --zfin "$ZFIN" 2>&1 | tee -a "$LOG"
crc=${PIPESTATUS[0]}
verdict "$crc" "snapshot_check exit code $crc"
