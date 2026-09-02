#!/usr/bin/env bash
# Validate the Laghos CUDA or HIP build against upstream's own reference data.
#
#   ./validate.sh [CUDA|HIP]
#
# Two upstream verification mechanisms are exercised, both on the GPU
# (`-d cuda` / `-d hip`) with one MPI rank:
#
# Phase 1 -- upstream `make checks` (built-in checks, laghos.cpp Checks()):
#   for every problem 0..7 in 2D (data/square01_quad.mesh) and 3D
#   (data/cube01_hex.mesh) run
#       laghos -cgt 1.e-14 -rs 0 --checks -p <P> -m data/<mesh> -pa -d <dev>
#   exactly as upstream's makefile `checks` target does for its "-pa -d cuda"
#   option. With --checks the code compares |e| at two prescribed time steps
#   against values hard-coded in laghos.cpp with a 1e-13 RELATIVE tolerance and
#   aborts (MFEM_VERIFY, non-zero exit) on a mismatch; it also verifies that
#   exactly two checks fired. We require exit code 0 and the MFEM device
#   banner "Device configuration: cuda" (resp. hip). 16 short runs.
#
# Phase 2 -- upstream README "Verification of Results" table:
#   the README lists final `step`, `dt` and `|e|` for 9 reference runs (run
#   with -np 8) and the equivalent `-d gpu` command lines for runs 1-4 and 6-9
#   (run 5 is 1D and not supported on the device). The results are independent
#   of the rank count up to round-off (the discretization is global; MPI only
#   changes the order of the reductions), so they are reproduced here with
#   `mpirun -np 1`. For each selected case the last "step ..." line of the
#   output is parsed (as upstream's `make tests` target does) and
#     - the final step count must match exactly,
#     - the final dt (printed with 6 decimals) must match exactly,
#     - |e| (printed with 11 significant digits) must match within a relative
#       tolerance of E_RTOL (default 1e-8).
#   Upstream's acceptance criterion is "within round-off distance". On the
#   B200 (CUDA 13.2, 1 rank) all eight GPU-capable rows reproduce the
#   reference |e| to all 11 printed digits (relative difference 0, i.e.
#   < 5e-11, the print resolution) -- see README. 1e-8 is therefore ~200x the
#   print resolution: it leaves room for reduction-order / FMA differences on
#   another GPU or rank count, while any real defect shows up as a step-count
#   mismatch or a change in the 3rd-6th significant digit of |e|.
#
#   Default cases: 3 (2D Sedov), 2, 4, 7 (the three 3D runs: Taylor-Green,
#   Sedov, triple point); ~2.5 min in total with phase 1. Override with
#   HPCPERF_LAGHOS_CASES="1 2 3 4 6 7 8 9" to run all eight GPU-capable rows
#   (~4 min; runs 6, 7 and 9 take ~45 s each).
#
# Environment overrides:
#   HPCPERF_LAGHOS_CASES   space-separated list of README rows (default "3 2 4 7")
#   HPCPERF_LAGHOS_E_RTOL  relative tolerance on |e| (default 1e-8)
#   HPCPERF_LAGHOS_SKIP_CHECKS=1  skip phase 1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true

set -uo pipefail

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level2/laghos/$MODEL"
EXE="$BUILD_DIR/laghos"
LOG_DIR="$BUILD_DIR/run"
mkdir -p "$LOG_DIR"

case "$BACKEND" in
    CUDA) DEVICE=cuda ;;
    HIP)  DEVICE=hip ;;
    *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac

if [ ! -x "$EXE" ]; then
    echo "validate.sh: $EXE not found -- run ./build.sh $BACKEND first" >&2
    exit 1
fi

E_RTOL="${HPCPERF_LAGHOS_E_RTOL:-1e-8}"
CASES="${HPCPERF_LAGHOS_CASES:-3 2 4 7}"
fail=0
cd "$HERE"   # relative mesh paths (data/...) as in upstream's documentation

check_device() { # $1 = log; verifies the MFEM device banner
    if ! grep -q "^Device configuration: .*\b${DEVICE}\b" "$1"; then
        echo "  -> FAIL: MFEM device banner does not list '${DEVICE}':"
        grep -m1 '^Device configuration' "$1" | sed 's/^/     /'
        return 1
    fi
    return 0
}

# ----------------------------------------------------------------------------
# Phase 1: upstream `make checks` equivalents on the device
# ----------------------------------------------------------------------------
if [ "${HPCPERF_LAGHOS_SKIP_CHECKS:-0}" != "1" ]; then
    echo "=== Phase 1: upstream built-in checks (--checks, 1e-13 relative, -pa -d $DEVICE, 1 rank)"
    for dim in 2 3; do
        [ "$dim" = 2 ] && mesh=data/square01_quad.mesh || mesh=data/cube01_hex.mesh
        for p in 0 1 2 3 4 5 6 7; do
            log="$LOG_DIR/checks-p${p}-${dim}D.log"
            cmd=(mpirun -np 1 "$EXE" -cgt 1.e-14 -rs 0 --checks -p "$p" -m "$mesh" -pa -d "$DEVICE")
            "${cmd[@]}" > "$log" 2>&1
            rc=$?
            steps="$(awk '/^step /{s=$2} END{sub(",","",s); print s}' "$log")"
            if [ "$rc" -eq 0 ] && check_device "$log" >/dev/null; then
                printf '  OK : p%d %dD %-24s (%s steps)\n' "$p" "$dim" "$mesh" "${steps:-?}"
            else
                printf '  KO : p%d %dD %-24s exit %d -- %s\n' "$p" "$dim" "$mesh" "$rc" "$log"
                grep -m2 -i 'verif\|error\|abort' "$log" | sed 's/^/       /'
                fail=1
            fi
        done
    done
fi

# ----------------------------------------------------------------------------
# Phase 2: README "Verification of Results" table (GPU variants, -np 1)
# ----------------------------------------------------------------------------
declare -A CMD REF_STEP REF_DT REF_E
CMD[1]="-p 0 -m data/square01_quad.mesh -rs 3 -tf 0.75 -pa";                REF_STEP[1]=339;  REF_DT[1]=0.000702; REF_E[1]=4.9695537349e+01
CMD[2]="-p 0 -m data/cube01_hex.mesh -rs 1 -tf 0.75 -pa";                   REF_STEP[2]=1041; REF_DT[2]=0.000121; REF_E[2]=3.3909635545e+03
CMD[3]="-p 1 -m data/square01_quad.mesh -rs 3 -tf 0.8 -pa";                 REF_STEP[3]=1154; REF_DT[3]=0.001655; REF_E[3]=4.6303396053e+01
CMD[4]="-p 1 -m data/cube01_hex.mesh -E0 2 -rs 2 -tf 0.6 -pa";              REF_STEP[4]=560;  REF_DT[4]=0.002449; REF_E[4]=1.3408616722e+02
CMD[6]="-p 3 -m data/rectangle01_quad.mesh -rs 2 -tf 3.0 -pa";              REF_STEP[6]=2872; REF_DT[6]=0.000064; REF_E[6]=5.6547039096e+01
CMD[7]="-p 3 -m data/box01_hex.mesh -rs 1 -tf 5.0 -pa -cgt 1e-12";          REF_STEP[7]=858;  REF_DT[7]=0.000474; REF_E[7]=5.6691500623e+01
CMD[8]="-p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.62831853 -s 7 -pa"; REF_STEP[8]=776; REF_DT[8]=0.000045; REF_E[8]=4.0982431726e+02
CMD[9]="-p 7 -m data/rt2D.mesh -tf 4 -rs 1 -ok 4 -ot 3 -pa";                REF_STEP[9]=2462; REF_DT[9]=0.000050; REF_E[9]=1.1792848680e+02

echo "=== Phase 2: README verification table (-d $DEVICE, 1 rank; |e| rel tol $E_RTOL)"
for c in $CASES; do
    if [ -z "${CMD[$c]:-}" ]; then
        echo "  run $c: no GPU-capable reference (run 5 is 1D) -- skipped"
        continue
    fi
    log="$LOG_DIR/verify-run${c}.log"
    # shellcheck disable=SC2206
    args=(${CMD[$c]})
    echo "--- run $c: mpirun -np 1 laghos ${CMD[$c]} -d $DEVICE"
    t0=$(date +%s)
    mpirun -np 1 "$EXE" "${args[@]}" -d "$DEVICE" > "$log" 2>&1
    rc=$?
    t1=$(date +%s)
    if [ "$rc" -ne 0 ]; then
        echo "  run failed (exit $rc); last lines of $log:"
        tail -5 "$log" | sed 's/^/    /'
        fail=1
        continue
    fi
    # last "step N, t = ..., dt = ..., |e| = ..." line, same fields as upstream's `make tests`
    read -r step dt e <<< "$(grep -E '^[[:space:]]*step[[:space:]]+[0-9]+' "$log" | tail -n 1 |
                             awk '{gsub(",","",$2); gsub(",","",$8); print $2, $8, $11}')"
    if [ -z "${e:-}" ]; then
        echo "  -> FAIL: no 'step ...' line in $log"
        fail=1
        continue
    fi
    rel="$(awk -v a="$e" -v r="${REF_E[$c]}" 'BEGIN{d=a-r; if(d<0)d=-d; printf "%.2e", d/r}')"
    printf '  got: step = %4d, dt = %s, |e| = %s   (%ds)\n' "$step" "$dt" "$e" "$((t1 - t0))"
    printf '  ref: step = %4d, dt = %s, |e| = %s   rel diff |e| = %s\n' "${REF_STEP[$c]}" "${REF_DT[$c]}" "${REF_E[$c]}" "$rel"
    ok=1
    check_device "$log" || ok=0
    if [ "$step" != "${REF_STEP[$c]}" ]; then
        echo "  -> FAIL: step count $step != reference ${REF_STEP[$c]}"; ok=0
    fi
    if [ "$dt" != "${REF_DT[$c]}" ]; then
        echo "  -> FAIL: final dt $dt != reference ${REF_DT[$c]}"; ok=0
    fi
    if ! awk -v x="$rel" -v t="$E_RTOL" 'BEGIN{exit !(x+0 <= t+0)}'; then
        echo "  -> FAIL: |e| relative difference $rel exceeds $E_RTOL"; ok=0
    fi
    if [ "$ok" -eq 1 ]; then
        echo "  -> ok"
    else
        fail=1
    fi
done

echo "----------------------------------------------------------------"
if [ "$fail" -eq 0 ]; then
    echo "PASS: laghos $BACKEND (built-in --checks for p0-7 in 2D/3D passed; README verification runs [$CASES] match: step and dt exact, |e| within $E_RTOL relative)"
    exit 0
else
    echo "FAIL: laghos $BACKEND (see $LOG_DIR/checks-*.log and $LOG_DIR/verify-run*.log)"
    exit 1
fi
