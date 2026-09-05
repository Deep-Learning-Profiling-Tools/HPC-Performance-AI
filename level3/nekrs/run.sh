#!/usr/bin/env bash
# Run nekRS' ethier case (examples/ethier: 3D incompressible Navier-Stokes with
# two passive scalars on the Ethier-Steinman exact solution; the full solver
# stack -- pressure Poisson with pMG+HYPRE coarse grid, velocity Helmholtz,
# subcycled advection, gather-scatter halo exchange) on N GPUs.
#
#   ./run.sh [CUDA|HIP] [extra nekrs args, e.g. --cimode 2]
#
# Execution model (upstream: "NekRS binds 1 GPU to 1 MPI rank"): one MPI rank
# per GPU. nekRS' own default is --device-id LOCAL-RANK; under the common
# launcher's per-rank wrapper each rank sees exactly one GPU, so the run passes
# --device-id 0 (the mapping is audited by the launcher). GPU-aware MPI is
# upstream-default OFF (RELEASE.md: enabling it "may cause a performance
# regression"); HPCPERF_NEKRS_GPU_MPI=1 turns it on (NEKRS_GPU_MPI env).
# Rank count is unconstrained (graph partitioning), but must stay well below
# the element count.
#
# Resource / size controls:
#   HPCPERF_GPUS=N|all       ranks = GPUs (default 1)
#   HPCPERF_SCALE_MODE       smoke | strong | weak  (default smoke)
#     smoke  : upstream ethier.par as shipped: 32 elements, N=9, 100 steps
#     strong : ethierRefine.par with hrefine=H (H=HPCPERF_NEKRS_HREFINE, default
#              10): 32*H^3 = 32,000 elements, N=7 (16.4M grid points), fixed
#     weak   : hrefine chosen per rank count so that elements/rank ~ 8000
#              (upstream's reference load, kershaw README "E/GPU=8000"):
#              H = round(cbrt(250 N)); integer H makes the per-rank count vary
#              between ~6.9k and ~8.5k -- printed and recorded
#   HPCPERF_NEKRS_STEPS      time steps (default: upstream 100)
#   HPCPERF_NEKRS_GPU_MPI    0|1 (default 0 = upstream default)
#
# The case directory (re2/usr/udf/oudf/par) is copied into the build tree;
# strong/weak write a derived ethier.par from upstream's ethierRefine.par with
# only `hrefine` and `numSteps` changed (class A). The OCCA JIT cache is shared
# per build (NEKRS_CACHE_DIR) -- the first run of a new kernel set compiles OKL
# kernels with nvcc, which takes minutes and is not part of the solve time.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"; [ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
l3_paths nekrs
# variant selection must match build.sh: the default 'hypregpu' uses the legacy layout; any other
# variant (e.g. cpucoarse = ENABLE_HYPRE_GPU=OFF) has its own install and JIT cache.
VARIANT="${HPCPERF_NEKRS_VARIANT:-$([ "${HPCPERF_NEKRS_HYPRE_GPU:-ON}" = ON ] && echo hypregpu || echo cpucoarse)}"
if [ "$VARIANT" = hypregpu ]; then
    BUILD_DIR="$R/build/level3/nekrs/$MODEL"
else
    L3_INSTALL="$L3_R/.deps/level3/nekrs/$VARIANT/install"
    BUILD_DIR="$R/build/level3/nekrs/$VARIANT.$MODEL"
fi
export NEKRS_HOME="$L3_INSTALL"
EXE="$NEKRS_HOME/bin/nekrs"
[ -x "$EXE" ] || { echo "run.sh: $EXE not found for variant '$VARIANT' -- run HPCPERF_NEKRS_VARIANT=$VARIANT ./build.sh $BACKEND first" >&2; exit 1; }
CASE_SRC="$R/_upstream/level3/nekRS/examples/ethier"
[ -f "$CASE_SRC/ethier.re2" ] || { echo "run.sh: $CASE_SRC missing (run fetch.sh)" >&2; exit 1; }

N_RANKS="$(hpcperf_ranks nekrs yes)" || exit 2
hpcperf_forbid_args nekrs --setup --device-id --backend -- "$@" || exit 2
MODE="$(l3_scale_mode nekrs)" || exit 2
STEPS="${HPCPERF_NEKRS_STEPS:-100}"
case "$MODE" in
    smoke)  H=0; ORDER=9 ;;
    strong) H="${HPCPERF_NEKRS_HREFINE:-10}"; ORDER=7 ;;
    weak)   H="$(python3 -c "import math; print(max(1, round((250*$N_RANKS)**(1/3))))")"; ORDER=7 ;;
esac
if [ "$H" -gt 0 ]; then ELEMS=$((32 * H * H * H)); else ELEMS=32; fi
POINTS=$((ELEMS * (ORDER + 1) * (ORDER + 1) * (ORDER + 1)))

# l3_rundir: dry-run gets a throwaway dir instead of rm -rf'ing the real run directory.
RUN_DIR="$(l3_rundir "$BUILD_DIR/run/$MODE.np$N_RANKS")" || exit 2
cp "$CASE_SRC"/* "$RUN_DIR"/     # complete upstream case directory (re2, usr, udf, CASEDATA include, ci.inc, par files)
if [ "$MODE" = smoke ]; then
    sed -e "s/^numSteps *=.*/numSteps = $STEPS/" "$CASE_SRC/ethier.par" > "$RUN_DIR/ethier.par"
else
    # derived from upstream ethierRefine.par: hrefine and numSteps only
    sed -e "s/^hrefine *=.*/hrefine = $H/" -e "s/^numSteps *=.*/numSteps = $STEPS/" "$CASE_SRC/ethierRefine.par" > "$RUN_DIR/ethier.par"
fi
export NEKRS_CACHE_DIR="$BUILD_DIR/cache"; mkdir -p "$NEKRS_CACHE_DIR"
export NEKRS_GPU_MPI="${HPCPERF_NEKRS_GPU_MPI:-0}"
# nekRS uses MPI one-sided operations (MPI_Win_lock); Open MPI's default one-sided component on this
# node is `osc ucx`, which goes through UCX/InfiniBand even on one node and aborts in uct_ib with 4 ranks
# ("'abort' is not implemented for protocol amo64/fetch"). The site profile already keeps point-to-point
# off UCX (pml ob1 / btl self,sm,smcuda); the same is done here for one-sided (class C, site transport).
export OMPI_MCA_osc="${HPCPERF_NEKRS_OSC:-^ucx}"
export OMPI_FC="${HPCPERF_SYSTEM_GFORTRAN:-/usr/bin/gfortran}"   # .usr file is compiled at run time with the MPI Fortran wrapper
unset CMAKE_GENERATOR   # nekRS' run-time UDF build configures with CMake and then calls `make okl.i`; the conda env's Ninja default breaks it
export CUDA_CACHE_DISABLE=1
# The Nek5000 side of the case (ethier.usr) keeps per-field work arrays of size lx1^3*lelt on the stack;
# upstream's job scripts raise the stack limit for that reason. With the default 8 MB the h-refined
# cases segfault in useric. Applies to mpirun's children (inherited rlimit). Class C.
ulimit -s unlimited 2>/dev/null || ulimit -s "$(ulimit -H -s)"

echo "# nekRS $BACKEND: mode=$MODE ranks=$N_RANKS case=ethier hrefine=$H elements=$ELEMS (~$((ELEMS / N_RANKS))/rank) N=$ORDER points=$POINTS steps=$STEPS gpu_mpi=$NEKRS_GPU_MPI run_dir=$RUN_DIR"
cd "$RUN_DIR"
RUN_ID="$(l3_run_id)"
"$L3_LAUNCHER" --gpus "$N_RANKS" --bind wrapper -- "$EXE" --setup ethier --backend "$BACKEND" --device-id 0 "$@" 2>&1 | tee "$RUN_DIR/stdout.log"
rc=${PIPESTATUS[0]}
if [ -z "${HPCPERF_DRY_RUN:-}" ]; then
    l3_manifest "$RUN_DIR" "run_id=$RUN_ID" "app=nekrs" "variant=$VARIANT" "backend=$BACKEND" "mode=$MODE" "ranks=$N_RANKS" \
        "elements=$ELEMS" "order=$ORDER" "points=$POINTS" "steps=$STEPS" "gpu_mpi=$NEKRS_GPU_MPI" \
        "osc=$OMPI_MCA_osc" "extra_args=$*" "exit_code=$rc" "binary=$EXE" "binary_sha256=$(l3_sha_file "$EXE")" \
        "par_sha256=$(l3_sha_file "$RUN_DIR/ethier.par")" "stdout=$RUN_DIR/stdout.log" "utc=$(date -u +%FT%TZ)"
fi
exit "$rc"
