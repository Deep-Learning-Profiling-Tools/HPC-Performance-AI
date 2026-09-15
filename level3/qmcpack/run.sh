#!/usr/bin/env bash
# Run QMCPACK (real build, OpenMP target offload + CUDA) on N GPUs, one MPI rank per GPU,
# OpenMP threads per rank = walker crowds (upstream's recommended GPU layout).
#
#   ./run.sh [CUDA]
#
# Case (HPCPERF_QMCPACK_CASE):
#   diamond2 (default)  tests/solids/diamondC_2x1x1_pp/qmc_short_vmcbatch_dmcbatch.in.xml -- upstream's
#       batched-driver VMC (256 walkers, 100 warm-up steps, 1 block) + DMC (256 walkers, 25 blocks x 100
#       steps, tau 0.005, no non-local moves) test for the 2x1x1 diamond supercell: 16 electrons, B-spline
#       single-particle orbitals from pwscf.pwscf.h5 (Quantum ESPRESSO), BFD pseudopotential, J1+J2
#       Jastrows. Upstream runs this deck on 1 rank x 4 threads and on 4 ranks (qmc_short_vmcbatch_dmcbatch_4r,
#       identical <qmc> blocks) and checks the DMC total energy against -21.844975 +- 0.02 Ha
#       (DIAMOND2_DMC_SCALARS, check_scalars.py --ns 3), which is exactly what validate.sh does here.
#     smoke : the deck verbatim (total_walkers 256 -> split over the N ranks)
#     strong: derived deck, VMC+DMC total_walkers = HPCPERF_QMCPACK_WALKERS (default 256 = the verbatim
#             population, fixed over N)
#     weak  : derived deck, VMC+DMC walkers_per_rank = HPCPERF_QMCPACK_WALKERS_PER_RANK (default 256),
#             i.e. total = W x N (QMC weak scaling = more statistics at constant work per GPU)
#   POPULATION LIMIT on this build (measured 2026-09-06, see README): device memory grows by ~320 MB per
#   walker (82 GB at 256 walkers, 165 GB at 512; 1024 walkers on one B200 exhaust the 183 GB and cuSOLVER
#   aborts with CUSOLVER_STATUS_INTERNAL_ERROR) although QMCPACK's own allocators report ~27 MiB -- so more
#   than ~300 walkers per GPU are refused by the run, not silently reduced. Larger requests are honoured
#   only when explicitly asked for and are expected to FAIL until the cause (LLVM 23.1 offload runtime on
#   CUDA 13.2/B200) is understood.
#   The derived decks change ONLY the walker-population parameter (recorded in the manifest with the diff);
#   the DMC energy estimate does not depend on the population beyond the population-control bias (which
#   shrinks with more walkers), so upstream's reference check applies in every mode.
#   NiO (tests/performance/NiO, S1-S256) is NOT used: its orbital files exist only behind an anl.box.com link.
#
# Controls: HPCPERF_GPUS=N|all, HPCPERF_CPUS_PER_RANK=T (OpenMP threads per rank, default 8),
#           HPCPERF_SCALE_MODE=smoke|strong|weak, HPCPERF_QMCPACK_PROFILE, HPCPERF_DRY_RUN=1,
#           HPCPERF_QMCPACK_MAX_WALKERS_PER_GPU (default 300), HPCPERF_QMCPACK_FORCE_POPULATION=1
# OMP_TARGET_OFFLOAD=MANDATORY: a failed offload aborts the run instead of silently falling back to the host.
# Output: <run_dir>/qmc.out (stdout+stderr incl. the launcher audit), <prefix>.s00N.scalar.dat, run_manifest.txt.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"; [ $# -gt 0 ] && shift
[ "$BACKEND" = CUDA ] || { echo "run.sh: only CUDA (OpenMP offload + CUDA) is built for QMCPACK here" >&2; exit 2; }
PROFILE="${HPCPERF_QMCPACK_PROFILE:-clang231-cuda132-offload}"
l3_paths_profile qmcpack "$PROFILE" cuda || exit 2
l3_require_materialized "$HERE" || exit 3
SRC="$HERE/src"      # frozen source bundle: tests/solids/diamondC_2x1x1_pp (deck, pseudopotential, orbitals) lives inside it
LLVM="${HPCPERF_QMCPACK_LLVM:-$L3_INSTALL/llvm}"
EXE="$L3_INSTALL/qmcpack-real/bin/qmcpack"
[ -x "$EXE" ] || { echo "run.sh: $EXE not found -- run ./build.sh first (profile $PROFILE)" >&2; exit 1; }
export LD_LIBRARY_PATH="$LLVM/lib:$LLVM/lib/x86_64-unknown-linux-gnu:$L3_INSTALL/hdf5/lib:$L3_INSTALL/openblas/lib:${LD_LIBRARY_PATH:-}"
l3_binary_backend_check "$EXE" cuda || exit 1
# the offload runtime must be the private LLVM's (never a stray system libomptarget)
ldd "$EXE" | /usr/bin/grep -q "libomptarget.so.*$LLVM" || { echo "run.sh: $EXE does not resolve libomptarget from $LLVM:"; ldd "$EXE" | /usr/bin/grep -E 'omptarget|libomp' >&2; exit 1; }
CASE="${HPCPERF_QMCPACK_CASE:-diamond2}"
MODE="$(l3_scale_mode qmcpack)" || exit 2
N_RANKS="$(hpcperf_ranks qmcpack yes)" || exit 2
THREADS="${HPCPERF_CPUS_PER_RANK:-8}"
case "$CASE" in
    diamond2) CASE_DIR="$SRC/tests/solids/diamondC_2x1x1_pp"; INP="$CASE_DIR/qmc_short_vmcbatch_dmcbatch.in.xml"; AUX="C.BFD.xml pwscf.pwscf.h5"; PREFIX=qmc_short_vmcbatch_dmcbatch ;;
    *) echo "run.sh: HPCPERF_QMCPACK_CASE must be diamond2" >&2; exit 2 ;;
esac
[ -f "$INP" ] || { echo "run.sh: $INP missing (run tools/prepare_benchmark.sh level3 qmcpack)" >&2; exit 1; }
case "$MODE" in
    smoke)  LABEL="$CASE"; DERIV="verbatim" ;;
    strong) W="${HPCPERF_QMCPACK_WALKERS:-256}"; LABEL="$CASE.w$W"; DERIV="total_walkers=$W" ;;
    weak)   W="${HPCPERF_QMCPACK_WALKERS_PER_RANK:-256}"; LABEL="$CASE.wpr$W"; DERIV="walkers_per_rank=$W" ;;
esac
# population guard (see the header): walkers per GPU above the measured device-memory limit are refused unless forced
MAXW="${HPCPERF_QMCPACK_MAX_WALKERS_PER_GPU:-300}"
case "$MODE" in smoke) WPR=$(( (256 + N_RANKS - 1) / N_RANKS )) ;; strong) WPR=$(( (W + N_RANKS - 1) / N_RANKS )) ;; weak) WPR=$W ;; esac
if [ "$WPR" -gt "$MAXW" ] && [ -z "${HPCPERF_QMCPACK_FORCE_POPULATION:-}" ]; then
    echo "run.sh: $WPR walkers per GPU requested, above the measured limit of $MAXW for this build (device memory ~320 MB/walker; 1024 walkers on one B200 abort in cuSOLVER) -- refusing; set HPCPERF_QMCPACK_FORCE_POPULATION=1 to try anyway" >&2; exit 2
fi
RUN_DIR="$(l3_rundir "$L3_BUILD/$L3_RUN_SUBDIR/$LABEL.$MODE.np$N_RANKS.t$THREADS")" || exit 2
DECK="$RUN_DIR/$(basename "$INP")"
if [ "$MODE" = smoke ]; then
    cp "$INP" "$DECK"
else
    python3 - "$INP" "$DECK" "$MODE" "$W" <<'PY'
import re, sys
src, dst, mode, w = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
txt = open(src).read()
pat = re.compile(r'<parameter name="total_walkers">\s*\d+\s*</parameter>')
if len(pat.findall(txt)) != 2:
    sys.exit("run.sh: expected exactly two total_walkers parameters (VMC, DMC) in the upstream deck")
name = "total_walkers" if mode == "strong" else "walkers_per_rank"
open(dst, "w").write(pat.sub(f'<parameter name="{name}"> {w} </parameter>', txt))
PY
    diff -u "$INP" "$DECK" > "$RUN_DIR/deck.diff" || true
fi
for f in $AUX; do ln -sfn "$CASE_DIR/$f" "$RUN_DIR/$f"; done
export OMP_NUM_THREADS="$THREADS" OMP_TARGET_OFFLOAD=MANDATORY OMP_PROC_BIND=false
echo "# QMCPACK $BACKEND profile=$PROFILE case=$CASE mode=$MODE ($DERIV) ranks=$N_RANKS threads/rank=$THREADS deck=$(realpath --relative-to="$SRC" "$INP") run_dir=$RUN_DIR"
cd "$RUN_DIR"
RUN_ID="$(l3_run_id)"
set +e; set -o pipefail
"$L3_LAUNCHER" --gpus "$N_RANKS" --cpus-per-rank "$THREADS" --bind wrapper -- "$EXE" "$(basename "$DECK")" 2>&1 | tee "$RUN_DIR/qmc.out"
rc=$?
set +o pipefail; set -e
if [ -z "${HPCPERF_DRY_RUN:-}" ]; then
    l3_manifest "$RUN_DIR" "run_id=$RUN_ID" "app=qmcpack" "backend=$BACKEND" "profile=$PROFILE" "case=$CASE" "label=$LABEL" "mode=$MODE" "deck_derivation=$DERIV" \
        "ranks=$N_RANKS" "threads_per_rank=$THREADS" "walkers_per_gpu=$WPR" "exit_code=$rc" "binary=$EXE" "binary_sha256=$(l3_sha_file "$EXE")" \
        "upstream_input=$INP" "upstream_input_sha256=$(l3_sha_file "$INP")" "deck_sha256=$(l3_sha_file "$DECK")" \
        "orbitals_h5_sha256=$(l3_sha_file "$CASE_DIR/pwscf.pwscf.h5")" "pseudo_sha256=$(l3_sha_file "$CASE_DIR/C.BFD.xml")" \
        "fingerprint_sha256=$(l3_sha_file "$L3_INSTALL/.hpcperf-l3-fingerprint")" "prefix=$PREFIX" "omp_target_offload=MANDATORY" "utc=$(date -u +%FT%TZ)"
    /usr/bin/grep -aE 'offload to accelerators|CUDA acceleration|OpenMP device|devices|MPI Nodes|MPI ranks|OMP 1st level|Rank.*device' "$RUN_DIR/qmc.out" 2>/dev/null | head -12 | sed 's/^/gpu_evidence: /' >> "$RUN_DIR/run_manifest.txt" || true
fi
exit "$rc"
