#!/usr/bin/env bash
# generate_reference.sh -- CONSTRUCTION-TIME generation of a Branson frozen correctness reference
# (level2/branson/reference/<input-id>.json). It is never part of a candidate's correctness check: the registry
# check (check_reference.py) only reads the frozen file and errors out when it is missing.
#
#   generate_reference.sh <input-id> [--ranks N] [--note TEXT] [--force]
#       builds the CPU-only Branson (build/level2/branson/cpu_ref: -DUSE_GPU=OFF, the options validate.sh uses, the
#       same sources as the GPU build), runs the input's deck with N MPI ranks (default 1; same deck, seed and global
#       photon count -- the rank count is provenance, not part of the criteria) into
#       build/level2/branson/cpu_ref/runs/<input-id>/cpu_ref.log, then freezes the result with its provenance
#       (check_log.py freeze). Cost: seconds (marshak) to hours (the 250 M-photon decks: lb-hohlraum took 2 h 56 min
#       with 24 ranks; its 1-rank run did not finish the first of 5 steps in 12 h). Copy the log into the results
#       directory afterwards: the frozen JSON names it by sha256.
#   generate_reference.sh <input-id> --from-cpu-log LOG --ranks N --binary-sha256 SHA [--launch TEXT]
#                         [--allocation TEXT] [--started UTC] [--finished UTC] [--log-kept-at TEXT] [--note TEXT] [--force]
#       freezes an already finished CPU-only run (no build, no run): the four references of 2026-09-29..10-01 were
#       frozen this way from the logs that produced the PASS verdicts (the runs predate this script).
# Exit 0 with the JSON written; 2 on a usage, build or run error. An existing reference is kept unless --force.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# the sources include tools/timing/ROI markers (header-only, a no-op unless measured): the same include path
# build.sh exports, so the CPU-only reference configures from the same main.cc
export CPATH="$R/tools/timing/roi${CPATH:+:$CPATH}"
usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
[ $# -ge 1 ] || usage
ID="$1"; shift
RANKS=1; FROM_LOG=""; BIN_SHA=""; LAUNCH=""; ALLOC=""; STARTED=""; FINISHED=""; KEPT=""; NOTE=""; FORCE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --ranks) RANKS="${2:?}"; shift 2 ;;
        --from-cpu-log) FROM_LOG="${2:?}"; shift 2 ;;
        --binary-sha256) BIN_SHA="${2:?}"; shift 2 ;;
        --launch) LAUNCH="${2:?}"; shift 2 ;;
        --allocation) ALLOC="${2:?}"; shift 2 ;;
        --started) STARTED="${2:?}"; shift 2 ;;
        --finished) FINISHED="${2:?}"; shift 2 ;;
        --log-kept-at) KEPT="${2:?}"; shift 2 ;;
        --note) NOTE="${2:?}"; shift 2 ;;
        --force) FORCE=1; shift ;;
        *) echo "generate_reference.sh: unknown option $1" >&2; usage ;;
    esac
done
case "$RANKS" in ''|*[!0-9]*|0) echo "generate_reference.sh: --ranks must be a positive integer" >&2; exit 2 ;; esac
TOOL="$R/tools/inputs/hpcperf_inputs.py"
DECK_REL="$(python3 "$TOOL" param "$HERE" "$ID" deck 2>/dev/null)" || { echo "generate_reference.sh: input '$ID' is not registered in $HERE/inputs.yaml" >&2; exit 2; }
PHOTONS="$(python3 "$TOOL" param "$HERE" "$ID" photons 2>/dev/null)"
TSTOP="$(python3 "$TOOL" param "$HERE" "$ID" t_stop 2>/dev/null)"
DECK="$HERE/$DECK_REL"
[ -f "$DECK" ] || { echo "generate_reference.sh: deck $DECK missing" >&2; exit 2; }
OUT="$HERE/reference/$ID.json"
if [ -f "$OUT" ] && [ -z "$FORCE" ]; then
    echo "generate_reference.sh: $OUT exists -- a frozen reference is a construction-time artifact; --force regenerates it" >&2; exit 2
fi
UPSTREAM="$(sed -n 's/^Upstream commit: \([0-9a-f]\{40\}\).*/\1/p' "$HERE/README.md" | head -1)"
BUILD_OPTS="cmake -DCMAKE_BUILD_TYPE=Release -DUSE_GPU=OFF -DUSE_CUDA=OFF -DUSE_HIP=OFF -DUSE_UMPIRE=OFF -DUSE_CALIPER=OFF -DBUILD_TESTING=OFF on level2/branson/src (the GPU build's sources; project toolchain hpcperf_env.sh, Open MPI 5.0.10; CPATH tools/timing/roi)"
GEN_BY="level2/branson/generate_reference.sh (construction-time)"

if [ -z "$FROM_LOG" ]; then
    command -v mpirun >/dev/null 2>&1 || { echo "generate_reference.sh: mpirun not on PATH (source $R/hpcperf_env.sh)" >&2; exit 2; }
    CPU_BUILD="$R/build/level2/branson/cpu_ref"; CPU_EXE="$CPU_BUILD/BRANSON"
    if [ ! -x "$CPU_EXE" ]; then
        echo "== building the CPU-only reference Branson into $CPU_BUILD"
        if ! cmake -S "$HERE/src" -B "$CPU_BUILD" -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_C_COMPILER="${CC:-gcc}" -DCMAKE_CXX_COMPILER="${CXX:-g++}" -DCMAKE_PREFIX_PATH="${CONDA_PREFIX:-}" \
                -DUSE_GPU=OFF -DUSE_CUDA=OFF -DUSE_HIP=OFF -DUSE_UMPIRE=OFF -DUSE_CALIPER=OFF -DBUILD_TESTING=OFF \
                > "$CPU_BUILD.cfg.log" 2>&1 \
           || ! cmake --build "$CPU_BUILD" -j"${MAKE_JOBS:-4}" --target BRANSON >> "$CPU_BUILD.cfg.log" 2>&1; then
            tail -30 "$CPU_BUILD.cfg.log"; echo "generate_reference.sh: CPU reference build failed (log: $CPU_BUILD.cfg.log)" >&2; exit 2
        fi
    fi
    RUN_DIR="$CPU_BUILD/runs/$ID"; mkdir -p "$RUN_DIR"
    MPI_MAP=(); [ "$RANKS" -gt 1 ] && MPI_MAP=(--map-by "ppr:$RANKS:node:OVERSUBSCRIBE")   # one Slurm task slot, as the launcher does
    LAUNCH="mpirun -np $RANKS ${MPI_MAP[*]:-} --bind-to none"
    BIN_SHA="$(sha256sum "$CPU_EXE" | cut -d' ' -f1)"
    ALLOC="${ALLOC:-$(hostname -s)${SLURM_JOB_ID:+, Slurm job $SLURM_JOB_ID}}"
    echo "== CPU reference run: $LAUNCH $CPU_EXE $DECK  (cwd $RUN_DIR, log cpu_ref.log; binary sha256 $BIN_SHA)"
    STARTED="$(date -u +%FT%TZ)"; t0=$(date +%s)
    if ! (cd "$RUN_DIR" && mpirun -np "$RANKS" "${MPI_MAP[@]}" --bind-to none "$CPU_EXE" "$DECK" > cpu_ref.log 2>&1); then
        tail -20 "$RUN_DIR/cpu_ref.log"; echo "generate_reference.sh: CPU reference run exited non-zero (log: $RUN_DIR/cpu_ref.log)" >&2; exit 2
    fi
    FINISHED="$(date -u +%FT%TZ)"
    echo "   CPU reference run: $(( $(date +%s) - t0 )) s"
    FROM_LOG="$RUN_DIR/cpu_ref.log"
    KEPT="${KEPT:-build/level2/branson/cpu_ref/runs/$ID/cpu_ref.log (git-ignored build tree: copy it into the results directory)}"
    GEN_BY="level2/branson/generate_reference.sh --ranks $RANKS (construction-time; built and ran the CPU-only reference)"
else
    [ -f "$FROM_LOG" ] || { echo "generate_reference.sh: --from-cpu-log $FROM_LOG missing" >&2; exit 2; }
    [ -n "$BIN_SHA" ] || { echo "generate_reference.sh: --from-cpu-log needs --binary-sha256 (the CPU-only binary that produced the log)" >&2; exit 2; }
    GEN_BY="level2/branson/generate_reference.sh --from-cpu-log (construction-time; frozen from the finished run's log, no rerun)"
fi
python3 "$HERE/check_log.py" freeze --input-id "$ID" --deck "$DECK" --cpu-log "$FROM_LOG" --ranks "$RANKS" --out "$OUT" \
    --binary-sha256 "$BIN_SHA" --build-options "$BUILD_OPTS" ${LAUNCH:+--launch "$LAUNCH"} ${ALLOC:+--allocation "$ALLOC"} \
    ${STARTED:+--started "$STARTED"} ${FINISHED:+--finished "$FINISHED"} ${KEPT:+--log-kept-at "$KEPT"} \
    ${UPSTREAM:+--upstream-commit "$UPSTREAM"} ${PHOTONS:+--param "photons=$PHOTONS"} ${TSTOP:+--param "t_stop=$TSTOP"} \
    ${NOTE:+--note "$NOTE"} --generated-by "$GEN_BY" ${FORCE:+--force}
