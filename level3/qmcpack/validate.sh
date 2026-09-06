#!/usr/bin/env bash
# Correctness check for QMCPACK (real, OpenMP offload + CUDA) on N GPUs.
#
#   ./validate.sh [CUDA]      HPCPERF_GPUS=N (default 1), HPCPERF_CPUS_PER_RANK=T (default 8)
#
# [1] (N = 1 only, skip with HPCPERF_QMCPACK_SKIP_CTEST=1) upstream's own ctest suites on this binary:
#     the Catch2 unit tests (label `unit`, they exercise the offload/CUDA kernels, batched BLAS,
#     determinant updates, ...) and the deterministic integration tests of the diamond cases
#     (fixed seeds, exact scalar references with upstream tolerances). Every failure is a FAIL.
# [2] Science case (run.sh diamond2, smoke = upstream deck verbatim) on N GPUs:
#     completeness ("QMCPACK execution completed successfully", no "QMCPACK ERROR", exit 0,
#     25 DMC blocks in the s001 scalar file, every scalar finite), GPU evidence from QMCPACK's
#     own banner ("OpenMP target offload to accelerators build option is enabled" AND "CUDA
#     acceleration build option is enabled"; the run itself has OMP_TARGET_OFFLOAD=MANDATORY so a
#     host fallback would have aborted), and the upstream statistical check: check_scalars.py
#     --ns 3 --series 1 -e 2 --le "-21.844975 0.02" (DIAMOND2_DMC_SCALARS -- the very test upstream
#     runs on this deck at 1 and 4 ranks). For N > 1 additionally: the N-GPU DMC energy must agree
#     with the 1-GPU run of the same binary/deck within 3 sqrt(sigma_1^2 + sigma_N^2) (block error
#     bars from check_scalars -- a pre-fixed statistical consistency criterion; QMC results are
#     stochastic, bitwise agreement across rank counts is not expected).
# Every exit code is captured; timeout / nonzero / missing files -> FAIL; nothing is read from old runs
# except the 1-GPU reference, which is re-run when its binary hash differs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
N="${HPCPERF_GPUS:-1}"; T="${HPCPERF_CPUS_PER_RANK:-8}"
TIMEOUT="${HPCPERF_VALIDATE_TIMEOUT:-3600}"
PROFILE="${HPCPERF_QMCPACK_PROFILE:-clang231-cuda132-offload}"
l3_paths_profile qmcpack "$PROFILE"
SRC="$R/_upstream/level3/qmcpack"; RUNS="$L3_BUILD/run"; BLD="$L3_BUILD/real"
REF_LE="-21.844975 0.02"   # upstream DIAMOND2_DMC_SCALARS totenergy (mean sigma) for qmc_short_vmcbatch_dmcbatch, series 1
NSIGMA=3                    # upstream check_scalars default used by QMC_RUN_AND_CHECK
EQUIL=2                     # upstream: -e 2 blocks of equilibration
DET_REGEX='deterministic-diamondC_(1x1x1|2x1x1)_pp-(vmcbatch|dmcbatch|sdbatch|vmc_sdj|vmc_dmc)'
export HPCPERF_GPUS="$N" HPCPERF_CPUS_PER_RANK="$T" HPCPERF_SCALE_MODE=smoke
mkdir -p "$RUNS"; ok=1
fail() { echo "validate.sh: FAIL -- $*"; ok=0; }
manifest_val() { /usr/bin/grep -m1 "^$2=" "$1/run_manifest.txt" 2>/dev/null | cut -d= -f2- || true; }
LLVM="$L3_INSTALL/llvm"
export LD_LIBRARY_PATH="$LLVM/lib:$LLVM/lib/x86_64-unknown-linux-gnu:$L3_INSTALL/hdf5/lib:$L3_INSTALL/openblas/lib:${LD_LIBRARY_PATH:-}"

if [ "$N" -eq 1 ] && [ -z "${HPCPERF_QMCPACK_SKIP_CTEST:-}" ]; then
    echo "validate.sh: [1] QMCPACK $BACKEND upstream ctest suites on 1 GPU [profile $PROFILE, build $BLD]"
    [ -f "$BLD/CTestTestfile.cmake" ] || { fail "build tree $BLD has no CTestTestfile.cmake"; }
    if [ -f "$BLD/CTestTestfile.cmake" ]; then
        # ctest launches mpiexec itself for the MPI tests: same slot-accounting relaxation as the common launcher
        # (this Slurm allocation advertises 1 task slot), one GPU visible, no oversubscription of that GPU.
        export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe OMPI_MCA_rmaps_base_oversubscribe=true
        export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,sm,smcuda OMP_TARGET_OFFLOAD=MANDATORY CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
        for suite in "unit:-L unit" "deterministic:-R $DET_REGEX"; do
            name="${suite%%:*}"; sel="${suite#*:}"
            log="$RUNS/ctest.$name.np1.log"
            # shellcheck disable=SC2086
            ( cd "$BLD" && timeout "$TIMEOUT" ctest $sel --timeout 900 --output-on-failure -j "${HPCPERF_CTEST_JOBS:-4}" ) > "$log" 2>&1 || rc=$?
            rc=${rc:-0}
            summary="$(/usr/bin/grep -aE '^[0-9]+% tests passed|tests passed,' "$log" | tail -1)"
            ntests="$(/usr/bin/grep -aoE 'out of [0-9]+' "$log" | tail -1 | /usr/bin/grep -oE '[0-9]+' || echo 0)"
            echo "    ctest $name: ${summary:-no summary} (exit $rc, log $log)"
            [ "$rc" -eq 124 ] && fail "ctest $name timed out after ${TIMEOUT}s"
            [ "${ntests:-0}" -gt 0 ] || fail "ctest $name selected no tests"
            /usr/bin/grep -aq '^100% tests passed' "$log" || { fail "ctest $name did not pass 100%"; /usr/bin/grep -aE '\*\*\*Failed|Timeout|Not Run' "$log" | head -20 | sed 's/^/      /'; }
            unset rc
        done
        unset PRTE_MCA_rmaps_default_mapping_policy OMPI_MCA_rmaps_base_oversubscribe
    fi
fi

echo "validate.sh: [2] QMCPACK $BACKEND diamondC_2x1x1_pp VMC+DMC (batched drivers, upstream deck) on $N GPU(s) x $T threads"
D="$RUNS/diamond2.smoke.np$N.t$T"
rc=0; HPCPERF_QMCPACK_CASE=diamond2 timeout "$TIMEOUT" "$HERE/run.sh" "$BACKEND" > "$RUNS/validate.diamond2.np$N.stdout" 2>&1 || rc=$?
/usr/bin/grep -aE '^# QMCPACK|hpcperf-launch: audit summary' "$RUNS/validate.diamond2.np$N.stdout" || true
if [ "$rc" -eq 124 ]; then fail "diamond2 timed out after ${TIMEOUT}s"; elif [ "$rc" -ne 0 ]; then fail "diamond2 run.sh exited $rc (see $RUNS/validate.diamond2.np$N.stdout)"; fi
check_run() { # check_run <run_dir> -> writes <run_dir>/qmc_summary.txt; returns 1 on any violation
    local d=$1
    python3 "$HERE/qmc_check.py" "$d" "$(manifest_val "$d" prefix)" "$REF_LE" "$NSIGMA" "$EQUIL" "$SRC/tests/scripts/check_scalars.py" > "$d/qmc_summary.txt"
}
if [ -f "$D/qmc.out" ]; then
    check_run "$D" || fail "diamond2 np$N: completeness/GPU-evidence/statistical check failed"
    sed 's/^/    /' "$D/qmc_summary.txt"
    if [ "$N" -gt 1 ]; then
        REF="$RUNS/diamond2.smoke.np1.t$T"
        if [ ! -f "$REF/qmc_summary.txt" ] || [ "$(manifest_val "$REF" binary_sha256)" != "$(manifest_val "$D" binary_sha256)" ]; then
            echo "    (1-GPU reference missing/stale -- running diamond2 on 1 GPU now)"
            rc=0; HPCPERF_GPUS=1 HPCPERF_QMCPACK_CASE=diamond2 timeout "$TIMEOUT" "$HERE/run.sh" "$BACKEND" > "$RUNS/validate.diamond2.np1.stdout" 2>&1 || rc=$?
            [ "$rc" -eq 0 ] || fail "diamond2 1-GPU reference run exited $rc"
            if [ -f "$REF/qmc.out" ]; then check_run "$REF" || fail "diamond2 1-GPU reference run failed its own checks"; fi
        fi
        python3 - "$REF/qmc_summary.txt" "$D/qmc_summary.txt" "$NSIGMA" <<'PY' || fail "diamond2 np$N: DMC energy inconsistent with the 1-GPU run"
import math, re, sys
def load(p):
    d = {}
    for l in open(p):
        for k, v in re.findall(r"(\w+)=(-?[0-9.]+(?:e[-+]?\d+)?)", l): d[k] = float(v)
    return d
a, b, ns = load(sys.argv[1]), load(sys.argv[2]), float(sys.argv[3])
for k in ("dmc_mean", "dmc_err"):
    if k not in a or k not in b: print(f"    missing {k} in a summary"); sys.exit(1)
sig = math.sqrt(a["dmc_err"] ** 2 + b["dmc_err"] ** 2); d = abs(a["dmc_mean"] - b["dmc_mean"])
print(f"    cross-rank: 1-GPU {a['dmc_mean']:.6f} +- {a['dmc_err']:.6f}  N-GPU {b['dmc_mean']:.6f} +- {b['dmc_err']:.6f}  |diff| = {d:.6f} Ha = {d / sig if sig > 0 else float('inf'):.2f} sigma (limit {ns:.0f} sigma) {'ok' if d <= ns * sig else 'BAD'}")
sys.exit(0 if d <= ns * sig else 1)
PY
    fi
fi

if [ "$ok" -eq 1 ]; then echo "QMCPACK $BACKEND validation ($N GPU x $T threads; $( [ "$N" -eq 1 ] && [ -z "${HPCPERF_QMCPACK_SKIP_CTEST:-}" ] && echo "upstream unit + deterministic ctests 100%, " || true)diamondC_2x1x1 VMC+DMC complete/finite, offload+CUDA active, DMC energy within upstream ${NSIGMA}-sigma reference$( [ "$N" -gt 1 ] && echo ", consistent with 1-GPU" || true)): PASS"; exit 0; fi
echo "QMCPACK $BACKEND validation ($N GPU): FAIL"; exit 1
