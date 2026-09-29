#!/usr/bin/env bash
# check_reference.sh -- Branson's GPU run of a deck against a CPU-only Branson built from the same
# sources (validate.sh check C, generalised to any deck): the correctness check of the registered
# inputs (level2/branson/inputs.yaml, `check:`).
#
#   check_reference.sh <gpu.log> <deck.xml> [--no-build]
#
# 1. builds build/level2/branson/cpu_ref once (USE_GPU=OFF, the options validate.sh uses);
# 2. runs `mpirun -np 1 cpu_ref/BRANSON <deck>` in the current directory -> cpu_ref.log
#    (same deck, same seed, one rank -- the same workload as the GPU run);
# 3. check_log.py cmp <gpu.log> cpu_ref.log: final Post-mat / Absorption / Exit energies within 5 %,
#    T_e of every cell within 0.02 absolute, transported photon count within 5 % (validate.sh's
#    criteria: > 6 sigma of the seed-to-seed scatter, catches a broken transport kernel).
# Prints "PASS: branson reference check ..." / "FAIL: branson reference check ..."; exit 0 / 1.
# The CPU run is single-threaded; for the 250 M-photon decks it takes hours (cost not established).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
GPU_LOG="${1:?usage: check_reference.sh <gpu.log> <deck.xml>}"
DECK="${2:?usage: check_reference.sh <gpu.log> <deck.xml>}"
NO_BUILD=0; [ "${3:-}" = "--no-build" ] && NO_BUILD=1
CPU_BUILD="$R/build/level2/branson/cpu_ref"
CPU_EXE="$CPU_BUILD/BRANSON"
JOBS="${MAKE_JOBS:-4}"
fail() { echo "FAIL: branson reference check -- $*"; exit 1; }

[ -f "$GPU_LOG" ] || fail "GPU log $GPU_LOG missing"
[ -f "$DECK" ] || fail "deck $DECK missing"
command -v mpirun >/dev/null 2>&1 || fail "mpirun not on PATH (source $R/hpcperf_env.sh)"
grep -q 'Photons Per Second (FOM)' "$GPU_LOG" || fail "the GPU run did not finish (no FOM line in $GPU_LOG)"

if [ ! -x "$CPU_EXE" ]; then
    [ "$NO_BUILD" = 0 ] || fail "$CPU_EXE not built (--no-build)"
    echo "== building the CPU-only reference Branson into $CPU_BUILD (validate.sh options)"
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
echo "== CPU reference run: mpirun -np 1 $CPU_EXE $DECK  (cwd $PWD, log cpu_ref.log)"
echo "   reference binary sha256: $(sha256sum "$CPU_EXE" | cut -d' ' -f1)"
t0=$(date +%s)
if ! mpirun -np 1 "$CPU_EXE" "$DECK" > cpu_ref.log 2>&1; then
    tail -20 cpu_ref.log
    fail "CPU reference run exited non-zero (log: $PWD/cpu_ref.log)"
fi
echo "   CPU reference run: $(( $(date +%s) - t0 )) s"
if python3 "$HERE/check_log.py" cmp "$GPU_LOG" cpu_ref.log; then
    echo "PASS: branson reference check ($(basename "$DECK")): GPU vs CPU-only Branson, same deck and seed -- final energies within 5 %, T_e within 0.02, transported photons within 5 %"
else
    fail "$(basename "$DECK"): GPU vs CPU reference comparison failed (see above; logs: $GPU_LOG, $PWD/cpu_ref.log)"
fi
