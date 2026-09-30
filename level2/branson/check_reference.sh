#!/usr/bin/env bash
# check_reference.sh -- Branson's GPU run of a deck against a CPU-only Branson built from the same
# sources (validate.sh check C, generalised to any deck): the correctness check of the registered
# inputs (level2/branson/inputs.yaml, `check:`).
#
#   check_reference.sh <gpu.log> <deck.xml> [--no-build] [--cpu-ref-ranks N]
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
# the sources include tools/timing/ROI markers (header-only, a no-op unless measured): the same
# include path build.sh exports, so the CPU-only reference configures from the same main.cc
export CPATH="$R/tools/timing/roi${CPATH:+:$CPATH}"
GPU_LOG="${1:?usage: check_reference.sh <gpu.log> <deck.xml> [--no-build] [--cpu-ref-ranks N]}"
DECK="${2:?usage: check_reference.sh <gpu.log> <deck.xml> [--no-build] [--cpu-ref-ranks N]}"
shift 2
# --cpu-ref-ranks N: the CPU reference with N MPI ranks (same deck, seed and global photon count; Branson's CPU
# build is MPI-parallel). The comparison criteria are statistical (5 % / 0.02), so the rank count does not enter
# them; it is reference provenance and is printed with the result. Used for the 250 M-photon lb-hohlraum deck,
# whose single-threaded reference did not finish its first step in 12 h.
NO_BUILD=0; REF_RANKS=1
while [ $# -gt 0 ]; do
    case "$1" in
        --no-build) NO_BUILD=1; shift ;;
        --cpu-ref-ranks) REF_RANKS="${2:?}"; shift 2 ;;
        *) echo "check_reference.sh: unknown option $1" >&2; exit 2 ;;
    esac
done
case "$REF_RANKS" in ''|*[!0-9]*|0) echo "check_reference.sh: --cpu-ref-ranks must be a positive integer" >&2; exit 2 ;; esac
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
# A finished CPU reference run of the same deck may be reused (HPCPERF_BRANSON_CPU_REF_LOG=<log>): the
# single-threaded references of the hohlraum decks take hours, and a re-run of the check must not have
# to repeat them. The reused log is named and hashed here, so check.log carries its provenance.
if [ -n "${HPCPERF_BRANSON_CPU_REF_LOG:-}" ]; then
    [ -f "$HPCPERF_BRANSON_CPU_REF_LOG" ] || fail "HPCPERF_BRANSON_CPU_REF_LOG=$HPCPERF_BRANSON_CPU_REF_LOG missing"
    grep -q 'Photons Per Second (FOM)' "$HPCPERF_BRANSON_CPU_REF_LOG" || fail "reused CPU reference log did not finish (no FOM line): $HPCPERF_BRANSON_CPU_REF_LOG"
    cp "$HPCPERF_BRANSON_CPU_REF_LOG" cpu_ref.log
    echo "== CPU reference: reusing the finished run $HPCPERF_BRANSON_CPU_REF_LOG (sha256 $(sha256sum cpu_ref.log | cut -d' ' -f1)); declared ranks: $REF_RANKS${HPCPERF_BRANSON_CPU_REF_NOTE:+; $HPCPERF_BRANSON_CPU_REF_NOTE}"
else
    MPI_MAP=()
    [ "$REF_RANKS" -gt 1 ] && MPI_MAP=(--map-by "ppr:$REF_RANKS:node:OVERSUBSCRIBE")   # one Slurm task slot, as the launcher does
    echo "== CPU reference run: mpirun -np $REF_RANKS ${MPI_MAP[*]} $CPU_EXE $DECK  (cwd $PWD, log cpu_ref.log)"
    echo "   reference binary sha256: $(sha256sum "$CPU_EXE" | cut -d' ' -f1)"
    t0=$(date +%s)
    if ! mpirun -np "$REF_RANKS" "${MPI_MAP[@]}" --bind-to none "$CPU_EXE" "$DECK" > cpu_ref.log 2>&1; then
        tail -20 cpu_ref.log
        fail "CPU reference run exited non-zero (log: $PWD/cpu_ref.log)"
    fi
    echo "   CPU reference run: $(( $(date +%s) - t0 )) s"
fi
if python3 "$HERE/check_log.py" cmp "$GPU_LOG" cpu_ref.log; then
    echo "PASS: branson reference check ($(basename "$DECK")): GPU vs CPU-only Branson ($REF_RANKS rank(s)), same deck and seed -- final energies within 5 %, T_e within 0.02, transported photons within 5 %"
else
    fail "$(basename "$DECK"): GPU vs CPU reference comparison failed (see above; logs: $GPU_LOG, $PWD/cpu_ref.log)"
fi
