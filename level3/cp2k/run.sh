#!/usr/bin/env bash
# Run CP2K (cp2k.psmp: MPI + OpenMP + CUDA) on N GPUs, one MPI rank per GPU with
# OpenMP threads per rank -- the hybrid execution model upstream ships as
# "psmp" and uses for its GPU regression tests.
#
#   ./run.sh [CUDA]
#
# Cases (HPCPERF_CP2K_CASE), upstream inputs from the read-only checkout:
#   h2o (default)   benchmarks/QS/H2O-<S>.inp -- upstream's Quickstep GPW-DFT benchmark:
#                   Born-Oppenheimer NVE MD of liquid water (TZV2P, 280 Ry, LDA/Pade, OT/DIIS),
#                   10 MD steps, S water molecules (32/64/128/256/...):
#                     smoke : S = HPCPERF_CP2K_SYSTEM (default 64)      -- correctness case
#                     strong: S = HPCPERF_CP2K_SYSTEM (default 128), fixed across rank counts
#                     weak  : S = 32*N (N ranks) -- a SIZE SWEEP, not a strict weak-scaling series:
#                             the GPW/OT cost per electron is not constant with system size, so
#                             "equal work per GPU" is not well defined here (see README)
#   regtest         a single upstream regression-test input (HPCPERF_CP2K_REGTEST=<dir/file
#                   under tests/>, e.g. QS/regtest-gpw-1/Ar.inp); validate.sh uses this with the
#                   reference values/tolerances from the directory's TEST_FILES.toml
#
# Controls:
#   HPCPERF_GPUS=N|all             MPI ranks (= GPUs; default 1)
#   HPCPERF_CPUS_PER_RANK=T        OpenMP threads per rank (default 8; ranks*T <= CPUs/node enforced by the launcher)
#   HPCPERF_SCALE_MODE             smoke|strong|weak (default smoke)
#   HPCPERF_CP2K_SYSTEM            water molecules for h2o smoke/strong
#   HPCPERF_CP2K_PROFILE           build profile (default cuda<tk>-gcc<v>-ompi<v>)
# The input file is used VERBATIM (copied next to the output); CP2K_DATA_DIR points at
# the checkout's data/ (basis sets, potentials). GPU binding: the launcher's per-rank
# wrapper narrows CUDA_VISIBLE_DEVICES to one GPU; CP2K binds device (rank mod
# #visible) = 0. Output <run_dir>/cp2k.out (+ stdout.log with the launcher audit) and a
# run_manifest.txt with exit code and hashes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"; [ $# -gt 0 ] && shift
[ "$BACKEND" = CUDA ] || { echo "run.sh: only CUDA is built for CP2K here" >&2; exit 2; }
GCC_MM="$(l3_version_mm "$(/usr/bin/gcc -dumpfullversion)")"; OMPI_V="$(mpirun --version | head -1 | /usr/bin/grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
PROFILE="${HPCPERF_CP2K_PROFILE:-cuda$(l3_version_mm "$(l3_cuda_version)")-gcc${GCC_MM}-ompi$(echo "$OMPI_V" | tr -d .)}"
l3_paths_profile cp2k "$PROFILE"
l3_require_materialized "$HERE" || exit 3
SRC="$HERE/src"      # frozen source bundle: benchmarks/, tests/ and data/ (CP2K_DATA_DIR) live inside it
EXE="$L3_INSTALL/cp2k/bin/cp2k.psmp"
[ -x "$EXE" ] || { echo "run.sh: $EXE not found -- run ./build.sh first (profile $PROFILE)" >&2; exit 1; }
# runtime libraries: the toolchain's `setup` (OpenBLAS/ScaLAPACK/FFTW/libxc/libint/DBCSR/... lib dirs)
# plus libcp2k.so itself (cp2k.psmp is installed without an rpath to it)
[ -f "$L3_INSTALL/toolchain/setup" ] || { echo "run.sh: toolchain setup file missing under $L3_INSTALL/toolchain" >&2; exit 1; }
set +u; # shellcheck disable=SC1091
source "$L3_INSTALL/toolchain/setup"; set -u
export LD_LIBRARY_PATH="$L3_INSTALL/cp2k/lib:$L3_INSTALL/cp2k/lib64:${LD_LIBRARY_PATH:-}"
# the CUDA device code (GRID/DBM/PW kernels) is in libcp2k.so; DBCSR's libsmm_acc kernels are JIT-compiled (NVRTC)
LIBCP2K=""; for f in "$L3_INSTALL/cp2k/lib/libcp2k.so" "$L3_INSTALL/cp2k/lib64/libcp2k.so"; do if [ -f "$f" ]; then LIBCP2K="$f"; break; fi; done
[ -n "$LIBCP2K" ] || { echo "run.sh: libcp2k.so not found under $L3_INSTALL/cp2k" >&2; exit 1; }
l3_binary_backend_check "$LIBCP2K" cuda || exit 1
# the BLAS/LAPACK actually loaded must be the toolchain's OpenBLAS (not another libopenblas.so.0 on the loader path):
# refuse to run otherwise, and record the resolved library in the manifest
BLAS_RESOLVED="$(ldd "$LIBCP2K" 2>/dev/null | awk '/libopenblas|libblas|liblapack|libmkl/{print $3}' | head -1)"
if [ -z "$BLAS_RESOLVED" ]; then
    # no dynamic BLAS: it must be the toolchain's libopenblas.a linked statically into libcp2k.so (CP2K_BLAS_VENDOR=CUSTOM)
    nm -D "$LIBCP2K" 2>/dev/null | /usr/bin/grep -qE ' T (dgemm_|openblas_get_config)$' || { echo "run.sh: libcp2k.so has neither a dynamic BLAS nor statically linked OpenBLAS symbols -- refusing" >&2; exit 1; }
    BLAS_RESOLVED="static:$(/usr/bin/grep -oE 'blas=[^ ]+' "$L3_INSTALL/BUILD_INFO.txt" 2>/dev/null | head -1)"
else
    case "$BLAS_RESOLVED" in
        "$L3_INSTALL"/toolchain/openblas-*/lib/*) : ;;
        *) echo "run.sh: libcp2k.so resolves BLAS to '$BLAS_RESOLVED', not to the toolchain's OpenBLAS under $L3_INSTALL/toolchain -- refusing (rebuild with ./build.sh)" >&2; exit 1 ;;
    esac
fi
CASE="${HPCPERF_CP2K_CASE:-h2o}"
MODE="$(l3_scale_mode cp2k)" || exit 2
N_RANKS="$(hpcperf_ranks cp2k yes)" || exit 2
THREADS="${HPCPERF_CPUS_PER_RANK:-8}"
case "$CASE" in
    h2o)
        case "$MODE" in
            smoke)  S="${HPCPERF_CP2K_SYSTEM:-64}" ;;
            strong) S="${HPCPERF_CP2K_SYSTEM:-128}" ;;
            weak)   S=$((32 * N_RANKS)) ;;
        esac
        INP="$SRC/benchmarks/QS/H2O-$S.inp"; LABEL="h2o$S"
        [ -f "$INP" ] || { echo "run.sh: no upstream input for $S water molecules ($INP); available: $(ls "$SRC/benchmarks/QS" | /usr/bin/grep -oE 'H2O-[0-9]+\.inp' | tr '\n' ' ')" >&2; exit 2; } ;;
    regtest)
        [ "$MODE" = smoke ] || { echo "run.sh: regtest inputs are smoke-only" >&2; exit 2; }
        RT="${HPCPERF_CP2K_REGTEST:-}"; [ -n "$RT" ] || { echo "run.sh: set HPCPERF_CP2K_REGTEST=<dir/file under tests/>" >&2; exit 2; }
        INP="$SRC/tests/$RT"; [ -f "$INP" ] || { echo "run.sh: $INP not found" >&2; exit 2; }
        LABEL="regtest.$(echo "$RT" | tr '/' '_' | sed 's/\.inp$//')" ;;
    *) echo "run.sh: HPCPERF_CP2K_CASE must be h2o or regtest" >&2; exit 2 ;;
esac
RUN_DIR="$(l3_rundir "$L3_BUILD/$L3_RUN_SUBDIR/$LABEL.$MODE.np$N_RANKS.t$THREADS")" || exit 2
cp "$INP" "$RUN_DIR/input.inp"
# regtest inputs may reference sibling files (basis sets, restart files) by relative name
if [ "$CASE" = regtest ]; then for f in "$(dirname "$INP")"/*; do [ -f "$f" ] && [ "$(basename "$f")" != "$(basename "$INP")" ] && ln -sfn "$f" "$RUN_DIR/"; done; fi
export CP2K_DATA_DIR="$SRC/data" OMP_NUM_THREADS="$THREADS" OMP_PROC_BIND=close OMP_PLACES=cores
echo "# CP2K $BACKEND profile=$PROFILE case=$CASE ($LABEL) mode=$MODE ranks=$N_RANKS threads/rank=$THREADS input=$(realpath --relative-to="$SRC" "$INP") run_dir=$RUN_DIR"
cd "$RUN_DIR"
RUN_ID="$(l3_run_id)"
set +e; set -o pipefail
"$L3_LAUNCHER" --gpus "$N_RANKS" --cpus-per-rank "$THREADS" --bind wrapper -- "$EXE" -i input.inp -o cp2k.out 2>&1 | tee "$RUN_DIR/stdout.log"
rc=$?
set +o pipefail; set -e
if [ -z "${HPCPERF_DRY_RUN:-}" ]; then
    l3_manifest "$RUN_DIR" "run_id=$RUN_ID" "app=cp2k" "backend=$BACKEND" "profile=$PROFILE" "case=$CASE" "label=$LABEL" "mode=$MODE" \
        "ranks=$N_RANKS" "threads_per_rank=$THREADS" "exit_code=$rc" "binary=$EXE" "binary_sha256=$(l3_sha_file "$EXE")" \
        "input=$INP" "input_sha256=$(l3_sha_file "$INP")" "cp2k_data_dir=$CP2K_DATA_DIR" "blas_resolved=$BLAS_RESOLVED" \
        "fingerprint_sha256=$(l3_sha_file "$L3_INSTALL/.hpcperf-l3-fingerprint")" "output=$RUN_DIR/cp2k.out" "utc=$(date -u +%FT%TZ)"
    # GPU evidence lines from CP2K's own banner (DBCSR/GRID/DBM/PW backends) for the manifest
    /usr/bin/grep -aE 'DBCSR\| ACC|GRID\| .*(GPU|backend)|DBM\| .*(GPU|backend)|PW_GPU|ACC\| |offload' "$RUN_DIR/cp2k.out" 2>/dev/null | head -12 | sed 's/^/gpu_evidence: /' >> "$RUN_DIR/run_manifest.txt" || true
fi
exit "$rc"
