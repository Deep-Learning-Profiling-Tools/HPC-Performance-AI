#!/usr/bin/env bash
# Validate the GPU Branson build.
#
#   ./validate.sh [CUDA|HIP]        (default: CUDA)
#
# Two modes (project decision: no CPU reference computation on a default path):
#   default  (HPCPERF_CPU_REFERENCE unset or 0) -- (A) and (B) below, unchanged, and (C) against a FROZEN
#     CPU-only reference of exactly this deck and these arguments (inputs/marshak_wave_replicated.xml
#     --t-stop 0.05 --seed 1234, 1 rank) instead of a live CPU-only build and run:
#       python3 check_reference.py validate_marshak_gpu.log --deck inputs/marshak_wave_replicated.xml \
#               --reference-dir reference/validate --expect-param "args=--t-stop 0.05 --seed 1234"
#     with the same quantities and margins as (C): final-step Post mat E, Absorption E, Exit E and transported
#     photons within 5 % relative, every cell's T_e within 0.02 absolute, the same number of steps (5).
#     reference/validate/marshak-t0.05-seed1234.json was frozen ONCE from the CPU-only run of one explicit
#     offline run of this script (check_log.py freeze, with the binary sha256, launch, dates and log sha256);
#     it is refused when its deck sha256, arguments or tolerances differ. Nothing is configured, built or run
#     on the CPU path: no cmake of $R/build/level2/branson/cpu_ref, no CPU-only Branson run.
#     (validate_marshak_frozen.log)
#   offline  (HPCPERF_CPU_REFERENCE set to 1: an explicit request only) -- (A), (B) and the original (C)
#     with the CPU-only reference build and run, unchanged.
#
# Three checks, all logged under $R/build/level2/branson/<cuda|hip>/:
#
#  (A) upstream unit tests: `ctest -E test_input_1pe` in the GPU build tree
#      (11 tests; 1- and 2-rank MPI tests plus the GPU warp-reduction test).
#      test_input_1pe is excluded because upstream's test expects
#      dd_batch_size/event_batch_size = 10000 while its own simple_input.xml
#      says 1000/777 -- an upstream test/data mismatch, unrelated to this port.
#      (validate_ctest.log)
#
#  (B) physics run on the GPU: inputs/marshak_wave_replicated.xml, 5 steps
#      (--t-stop 0.05, --seed 1234, 50k photons/step, 25 cells). Requires
#      exit code 0, exactly 5 completed steps, GPU transport actually used
#      ("Transferring ... cell(s) to the GPU" every step, no "GPU kernel not
#      available" fallback), and per step Branson's own energy balances
#         |Radiation conservation| <= 1e-9 * (Emission E + Source E + Pre census E)
#         |Material  conservation| <= 1e-9 * Pre mat E
#      (observed: ~1e-13 .. 1e-15 relative).  (validate_marshak_gpu.log)
#
#  (C) GPU vs CPU cross-check: the same deck and seed is run with a CPU-only
#      Branson (USE_GPU off) configured from the same sources into
#      $R/build/level2/branson/cpu_ref (built on first use, ~10 s). The
#      final step's Post mat E, Absorption E and Exit E must agree to 5 %
#      relative and every cell's T_e to 0.02 absolute. The two binaries use
#      different photon orderings, so the results agree only statistically;
#      the seed-to-seed scatter of these quantities is 0.2-0.8 % (energies)
#      and <= 0.005 (T_e), so 5 % / 0.02 is > 6 sigma yet catches a broken
#      transport kernel.  (validate_marshak_cpu.log; offline mode -- default: the frozen reference, above)
#
# Prints "PASS: branson <BACKEND> ..." / "FAIL: branson <BACKEND> ..." and
# exits 0/1.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# the sources include tools/timing/ROI markers (header-only, a no-op unless measured): the same
# include path build.sh exports, so the CPU-only reference configures from the same main.cc
export CPATH="$R/tools/timing/roi${CPATH:+:$CPATH}"

BACKEND="$(printf '%s' "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
case "$BACKEND" in
    CUDA|HIP) ;;
    *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac

if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$R" ] && [ -f "$R/hpcperf_env.sh" ]; then
    # conda's activate hooks are not `set -u`-clean
    set +u
    # shellcheck disable=SC1091
    source "$R/hpcperf_env.sh" 2>/dev/null
    set -u
fi

BUILD="$R/build/level2/branson/$(printf '%s' "$BACKEND" | tr '[:upper:]' '[:lower:]')"
EXE="$BUILD/BRANSON"
CPU_BUILD="$R/build/level2/branson/cpu_ref"
CPU_EXE="$CPU_BUILD/BRANSON"
DECK="$HERE/inputs/marshak_wave_replicated.xml"
DECK_ARGS=( --t-stop 0.05 --seed 1234 )
JOBS="${MAKE_JOBS:-4}"
FROZEN_DIR="$HERE/reference/validate"   # (C), default mode: the frozen CPU-only run of DECK + DECK_ARGS
case "${HPCPERF_CPU_REFERENCE:-}" in
    ''|0*) CPU_REF=0 ;;
    *)     CPU_REF=1 ;;
esac

fail() { echo "FAIL: branson $BACKEND -- $*"; exit 1; }

[ -x "$EXE" ] || fail "$EXE not found; run $HERE/build.sh $BACKEND first"
command -v mpirun >/dev/null 2>&1 || fail "mpirun not on PATH (source $R/hpcperf_env.sh)"
command -v python3 >/dev/null 2>&1 || fail "python3 not on PATH"

# ---------------------------------------------------------------- (A) ctest
# Upstream's 2-rank unit tests (test_imc_state_2pe, test_photon_2pe) are launched by ctest with a bare
# `mpirun -np 2`. Inside a Slurm allocation made with one task slot per node (SLURM_TASKS_PER_NODE=1) PRRTE
# refuses that launch ("not enough slots"); the project launcher relaxes the same limit per launch after its
# GPU checks (hpcperf_mpi_launch.sh, --map-by :OVERSUBSCRIBE). The same relaxation is applied here, for the
# ctest step only, and only when the allocation actually has fewer slots than the tests need (2 CPU ranks on
# one node; no GPU sharing is involved).
# The relaxation is a command-local environment of the ctest invocation only (never exported, so the caller's
# environment and the later 1-rank GPU run (B) are untouched).
SLOTS="${SLURM_TASKS_PER_NODE:-}"; SLOTS="${SLOTS%%[^0-9]*}"
CTEST_ENV=()
if [ -n "${SLURM_JOB_ID:-}" ] && [ -n "$SLOTS" ] && [ "$SLOTS" -lt 2 ]; then
    CTEST_ENV=(env PRTE_MCA_rmaps_default_mapping_policy=":OVERSUBSCRIBE")
    echo "== note: Slurm allocation has $SLOTS task slot(s)/node; PRRTE mapping relaxed for the 2-rank unit tests (ctest command only)"
fi
echo "== (A) upstream ctest in $BUILD (test_input_1pe excluded, see header)"
if ! "${CTEST_ENV[@]}" ctest --test-dir "$BUILD" -E test_input_1pe --output-on-failure > "$BUILD/validate_ctest.log" 2>&1; then
    tail -30 "$BUILD/validate_ctest.log"
    fail "ctest failed (log: $BUILD/validate_ctest.log)"
fi
CTEST_LINE="$(grep -E '^[0-9]+% tests passed' "$BUILD/validate_ctest.log" | tail -1)"
echo "   $CTEST_LINE"
case "$CTEST_LINE" in
    "100% tests passed, 0 tests failed out of "*) ;;
    *) fail "unexpected ctest summary: '$CTEST_LINE'" ;;
esac
NTESTS="${CTEST_LINE##*out of }"

# ------------------------------------------------ (B) GPU physics run + parser
echo "== (B) GPU run: mpirun -np 1 $EXE $DECK ${DECK_ARGS[*]}"
if ! mpirun -np 1 "$EXE" "$DECK" "${DECK_ARGS[@]}" > "$BUILD/validate_marshak_gpu.log" 2>&1; then
    tail -20 "$BUILD/validate_marshak_gpu.log"
    fail "GPU run exited non-zero (log: $BUILD/validate_marshak_gpu.log)"
fi

# The parser/checker lives in check_log.py (also usable on any deck's log by tools/inputs):
# mode gpu = check (B), mode cmp = check (C). Usage: check_py <mode> <gpu.log> [cpu.log]
check_py() {
    python3 "$HERE/check_log.py" "$@"
}

check_py gpu "$BUILD/validate_marshak_gpu.log" --steps 5 || fail "GPU physics checks failed (log: $BUILD/validate_marshak_gpu.log)"
echo "   $(grep 'Total transport:' "$BUILD/validate_marshak_gpu.log" | tail -1), $(grep 'FOM' "$BUILD/validate_marshak_gpu.log" | tail -1)"

# ------------------------------------------------ (C) CPU reference cross-check
if [ "$CPU_REF" -eq 0 ]; then
    echo "== CPU reference: skipped (default; (C) compares with the frozen CPU-only reference of this deck and arguments)"
    echo "== (C) GPU vs frozen CPU-only reference: check_reference.py --reference-dir $FROZEN_DIR --expect-param \"args=${DECK_ARGS[*]}\""
    python3 "$HERE/check_reference.py" "$BUILD/validate_marshak_gpu.log" --deck "$DECK" --reference-dir "$FROZEN_DIR" \
        --expect-param "args=${DECK_ARGS[*]}" > "$BUILD/validate_marshak_frozen.log" 2>&1
    rc=$?
    sed 's/^/   /' "$BUILD/validate_marshak_frozen.log"
    [ "$rc" -eq 0 ] || fail "GPU vs frozen CPU-only reference comparison failed, exit $rc (log: $BUILD/validate_marshak_frozen.log)"
    echo "PASS: branson $BACKEND (ctest $NTESTS/$NTESTS excl. test_input_1pe; Marshak 5 steps: rad/mat conservation <= 1e-9 rel; GPU vs frozen CPU-only reference final Post-mat/Absorption/Exit E within 5%, T_e within 0.02; CPU reference not run (default))"
    exit 0
fi
echo "== CPU reference: offline mode (HPCPERF_CPU_REFERENCE requested): (C) with the CPU-only reference build and run"
if [ ! -x "$CPU_EXE" ]; then
    echo "== (C) building CPU-only reference Branson into $CPU_BUILD"
    if ! cmake -S "$HERE/src" -B "$CPU_BUILD" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_COMPILER="${CC:-gcc}" -DCMAKE_CXX_COMPILER="${CXX:-g++}" \
            -DCMAKE_PREFIX_PATH="${CONDA_PREFIX:-}" \
            -DUSE_GPU=OFF -DUSE_CUDA=OFF -DUSE_HIP=OFF -DUSE_UMPIRE=OFF -DUSE_CALIPER=OFF \
            -DBUILD_TESTING=OFF > "$CPU_BUILD.cfg.log" 2>&1 \
       || ! cmake --build "$CPU_BUILD" -j"$JOBS" --target BRANSON >> "$CPU_BUILD.cfg.log" 2>&1; then
        tail -30 "$CPU_BUILD.cfg.log"
        fail "CPU reference build failed (log: $CPU_BUILD.cfg.log)"
    fi
fi
echo "== (C) CPU reference run: mpirun -np 1 $CPU_EXE $DECK ${DECK_ARGS[*]}"
if ! mpirun -np 1 --bind-to none "$CPU_EXE" "$DECK" "${DECK_ARGS[@]}" > "$BUILD/validate_marshak_cpu.log" 2>&1; then
    tail -20 "$BUILD/validate_marshak_cpu.log"
    fail "CPU reference run exited non-zero (log: $BUILD/validate_marshak_cpu.log)"
fi
check_py cmp "$BUILD/validate_marshak_gpu.log" "$BUILD/validate_marshak_cpu.log" \
    || fail "GPU vs CPU comparison failed (logs: $BUILD/validate_marshak_{gpu,cpu}.log)"

echo "PASS: branson $BACKEND (ctest $NTESTS/$NTESTS excl. test_input_1pe; Marshak 5 steps: rad/mat conservation <= 1e-9 rel; GPU vs CPU final Post-mat/Absorption/Exit E within 5%, T_e within 0.02)"
exit 0
