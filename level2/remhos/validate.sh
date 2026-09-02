#!/usr/bin/env bash
# Validate the Remhos CUDA or HIP build against upstream's documented
# reference results.
#
#   ./validate.sh [CUDA|HIP]
#
# Remhos' README ("Verification of Results") lists 13 runs with their final
# mass and maximum value; "an implementation is considered valid if the
# computed values are all within round-off distance from the reference
# values". None of those runs uses a solver combination that Remhos can
# execute on a GPU: remhos.cpp accepts a device only for
# -ho 3 (local inverse HO) -lo 5 (mass-based LO) -fct 2 (clip-and-scale FCT)
# (MFEM_VERIFY(..., "Wrong GPU setup.")), which is also the only configuration
# whose FOM upstream tracks. This script therefore checks two things:
#
# Part A -- the build reproduces upstream's reference table on the CPU
#   (-d cpu, full assembly). Rows 1, 3, 5, 7 and 13 (2D/3D transport, NURBS
#   mesh, remap, monolithic) are compared with the README values, relative
#   tolerance 1e-8: this build reproduces all five to the 10 printed digits
#   with 1 rank (the table was produced with 8 ranks; the results are
#   rank-independent). Rows 9 and 11 of the README at this commit are stale
#   (they differ from the current code by 1e-3 / 1e-4 in max while
#   autotest/out_baseline.dat, committed at the same time, matches) and are
#   not used. Rows 2, 4, 6, 8, 10, 12 are omitted only to keep the run short.
#
# Part B -- the GPU configuration -ho 3 -lo 5 -fct 2 -pa -d <cuda|hip>, one
#   MPI rank, on four README problems (rows 1/2, 5/6, 7 and 11 with the GPU
#   solver combination). For each case:
#     1. the MFEM banner shows "Device configuration: cuda" (resp. hip), i.e.
#        the mfem::forall kernels of Remhos and MFEM's PA operators ran on the
#        device (Remhos aborts if the solver combination cannot run there);
#     2. final mass and max agree with the SAME configuration run on the CPU
#        backend of the same binary (-pa -d cpu): mass rel. tol 1e-8, max rel.
#        tol 1e-6. Observed: identical to all 10 printed digits for three
#        cases; 9e-8 in max for periodic-hexagon after 2000 nonlinear FCT
#        steps (different reduction order on the device);
#     3. transport problems (rows 1/2, 5/6, 7): the final mass equals the
#        README reference mass within 1e-7 relative -- every Remhos solver
#        combination is conservative, so the table's mass applies to the GPU
#        combination too (the partial-assembly path loses ~1e-9..2e-8 relative
#        mass to quadrature, identically on CPU and GPU);
#     4. max <= 1 + 1e-12: the FCT solution stays within the bounds of the
#        initial data (all four problems start with 0 <= u <= 1).
#
# Everything runs with mpirun -np 1. Total ~1.5 min on a B200 node.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true

set -uo pipefail

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
case "$BACKEND" in
    CUDA) DEVICE=cuda ;;
    HIP)  DEVICE=hip ;;
    *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac
BUILD_DIR="$R/build/level2/remhos/$MODEL"
EXE="$BUILD_DIR/remhos"
LOG_DIR="$BUILD_DIR/run"
DATA="$HERE/data"

if [ ! -x "$EXE" ]; then
    echo "validate.sh: $EXE not found -- run ./build.sh $BACKEND first" >&2
    exit 1
fi
mkdir -p "$LOG_DIR"

TOL_REF=1e-8      # Part A: vs README table (reproduced exactly here)
TOL_MASS=1e-8     # Part B: GPU vs CPU mass
TOL_MAX=1e-6      # Part B: GPU vs CPU max
TOL_REFMASS=1e-7  # Part B: GPU mass vs README reference mass (transport)
fail=0

# rel_ok a b tol : |a-b| <= tol*max(|b|,1e-300)
rel_ok() {
    awk -v a="$1" -v b="$2" -v t="$3" 'BEGIN{
        d=a-b; if (d<0) d=-d; s=b; if (s<0) s=-s; if (s<1e-300) s=1;
        exit !(d<=t*s) }'
}
is_number() { [[ "$1" =~ ^-?[0-9]+(\.[0-9]*)?([eE][-+]?[0-9]+)?$ ]]; }

# run_remhos <log> <args...> : runs in LOG_DIR (Remhos may write files to cwd)
run_remhos() {
    local log="$1"; shift
    ( cd "$LOG_DIR" && mpirun -np 1 "$EXE" -no-vis "$@" ) > "$log" 2>&1
}
get_mass() { awk '/^Final mass u:/{print $4}' "$1" | tail -1; }
get_max()  { awk '/^Max value u:/{print $4}'  "$1" | tail -1; }

# ---------------------------------------------------------------------------
# Part A: README verification table, CPU backend (these solver combinations
# cannot run on a device).
check_ref() {
    local row="$1" ref_mass="$2" ref_max="$3"; shift 3
    local log="$LOG_DIR/validate-ref${row}.log"
    echo "=== README row $row (CPU): remhos $* -d cpu"
    echo "    reference: mass $ref_mass  max $ref_max"
    run_remhos "$log" "$@" -d cpu
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "  -> FAIL: run exited with $rc; last lines of $log:"
        tail -5 "$log" | sed 's/^/    /'; fail=1; return
    fi
    local mass max; mass="$(get_mass "$log")"; max="$(get_max "$log")"
    echo "    computed:  mass ${mass:-<missing>}  max ${max:-<missing>}"
    if ! is_number "$mass" || ! is_number "$max"; then
        echo "  -> FAIL: no finite final mass / max in $log"; fail=1; return
    fi
    if ! rel_ok "$mass" "$ref_mass" "$TOL_REF" || ! rel_ok "$max" "$ref_max" "$TOL_REF"; then
        echo "  -> FAIL: differs from the README reference by more than $TOL_REF (relative)"
        fail=1; return
    fi
    echo "  -> ok (matches the README reference within $TOL_REF)"
}

check_ref  1 0.3888354875 0.9333315791 -m "$DATA/periodic-hexagon.mesh" -p 0 -rs 2 -dt 0.005 -tf 10 -ho 1 -lo 2 -fct 2
check_ref  3 3.5982222    0.9995717563 -m "$DATA/disc-nurbs.mesh" -p 1 -rs 3 -dt 0.005 -tf 3 -ho 1 -lo 2 -fct 2
check_ref  5 0.1623263888 0.7676354393 -m "$DATA/periodic-square.mesh" -p 5 -rs 3 -dt 0.005 -tf 0.8 -ho 1 -lo 2 -fct 2
check_ref  7 0.9607429525 0.7678305756 -m "$DATA/periodic-cube.mesh" -p 0 -rs 1 -o 2 -dt 0.014 -tf 8 -ho 1 -lo 4 -fct 2
check_ref 13 0.3182739921 1            -m "$DATA/inline-quad.mesh" -p 6 -rs 2 -o 1 -dt 0.01 -tf 20 -mono 1 -si 1

# ---------------------------------------------------------------------------
# Part B: GPU configuration vs the CPU backend (+ reference mass for transport).
check_gpu() {
    local name="$1" ref_mass="$2"; shift 2   # ref_mass "-" for remap (no table value)
    local glog="$LOG_DIR/validate-${name}-${DEVICE}.log"
    local clog="$LOG_DIR/validate-${name}-cpu.log"
    echo "=== $name ($BACKEND): remhos $* -ho 3 -lo 5 -fct 2 -pa -d $DEVICE"
    run_remhos "$glog" "$@" -ho 3 -lo 5 -fct 2 -pa -d "$DEVICE"
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "  -> FAIL: $DEVICE run exited with $rc; last lines of $glog:"
        tail -5 "$glog" | sed 's/^/    /'; fail=1; return
    fi
    local banner; banner="$(grep -m1 '^Device configuration:' "$glog" || true)"
    echo "    ${banner:-<no MFEM device banner>}"
    if ! echo "$banner" | grep -q "Device configuration: *$DEVICE"; then
        echo "  -> FAIL: MFEM did not select the $DEVICE backend"; fail=1; return
    fi
    local gmass gmax; gmass="$(get_mass "$glog")"; gmax="$(get_max "$glog")"
    echo "    $DEVICE: mass ${gmass:-<missing>}  max ${gmax:-<missing>}  $(grep -m1 '^Total kernel time' "$glog" || true)  $(grep -m1 '^FOM:' "$glog" | tr -s ' ' || true)"
    if ! is_number "$gmass" || ! is_number "$gmax"; then
        echo "  -> FAIL: no finite final mass / max in $glog"; fail=1; return
    fi

    run_remhos "$clog" "$@" -ho 3 -lo 5 -fct 2 -pa -d cpu
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "  -> FAIL: cpu reference run exited with $rc; last lines of $clog:"
        tail -5 "$clog" | sed 's/^/    /'; fail=1; return
    fi
    local cmass cmax; cmass="$(get_mass "$clog")"; cmax="$(get_max "$clog")"
    echo "    cpu:  mass ${cmass:-<missing>}  max ${cmax:-<missing>}  $(grep -m1 '^Total kernel time' "$clog" || true)"
    if ! is_number "$cmass" || ! is_number "$cmax"; then
        echo "  -> FAIL: no finite final mass / max in $clog"; fail=1; return
    fi
    if ! rel_ok "$gmass" "$cmass" "$TOL_MASS"; then
        echo "  -> FAIL: $DEVICE mass differs from the cpu backend by more than $TOL_MASS"; fail=1; return
    fi
    if ! rel_ok "$gmax" "$cmax" "$TOL_MAX"; then
        echo "  -> FAIL: $DEVICE max differs from the cpu backend by more than $TOL_MAX"; fail=1; return
    fi
    if [ "$ref_mass" != "-" ] && ! rel_ok "$gmass" "$ref_mass" "$TOL_REFMASS"; then
        echo "  -> FAIL: $DEVICE mass $gmass differs from the README reference mass $ref_mass by more than $TOL_REFMASS"
        fail=1; return
    fi
    if ! awk -v m="$gmax" 'BEGIN{exit !(m <= 1 + 1e-12)}'; then
        echo "  -> FAIL: $DEVICE max $gmax exceeds the initial bound 1"; fail=1; return
    fi
    if [ "$ref_mass" != "-" ]; then
        echo "  -> ok ($DEVICE = cpu within $TOL_MASS/$TOL_MAX; mass = README $ref_mass within $TOL_REFMASS; max <= 1)"
    else
        echo "  -> ok ($DEVICE = cpu within $TOL_MASS/$TOL_MAX; max <= 1)"
    fi
}

# README rows 1/2 problem (2D transport, 3072 unknowns, 2000 steps)
check_gpu periodic-hexagon 0.3888354875 -m "$DATA/periodic-hexagon.mesh" -p 0 -rs 2 -dt 0.005 -tf 10
# README rows 5/6 problem (2D transport, 9216 unknowns, 160 steps)
check_gpu periodic-square  0.1623263888 -m "$DATA/periodic-square.mesh" -p 5 -rs 3 -dt 0.005 -tf 0.8
# README row 7 problem (3D transport, 5832 unknowns, 572 steps)
check_gpu periodic-cube    0.9607429525 -m "$DATA/periodic-cube.mesh" -p 0 -rs 1 -o 2 -dt 0.014 -tf 8
# README row 11 problem (3D remap, run.sh problem at README size; no table
# value for the GPU solver combination -- mass is not conserved through remap)
check_gpu cube01_hex-remap -              -m "$DATA/cube01_hex.mesh" -p 10 -rs 1 -o 2 -dt 0.02 -tf 0.8

echo "----------------------------------------------------------------"
if [ "$fail" -eq 0 ]; then
    echo "PASS: remhos $BACKEND (README rows 1,3,5,7,13 reproduced on the CPU; 4 GPU runs of -ho 3 -lo 5 -fct 2 -pa -d $DEVICE match the CPU backend and the README reference masses)"
    exit 0
else
    echo "FAIL: remhos $BACKEND (see $LOG_DIR/validate-*.log)"
    exit 1
fi
