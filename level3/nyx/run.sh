#!/usr/bin/env bash
# Run Nyx (full cosmological N-body + baryon hydrodynamics workflow: dark-matter
# particle IC read, Poisson gravity via AMReX MLMG, PPM hydro, particle push,
# periodic redistribution, plotfile I/O) on N GPUs, one MPI rank per GPU.
#
#   ./run.sh [CUDA|HIP|CPU] [extra Nyx inputs overrides...]
#
# Cases (HPCPERF_NYX_CASE), all upstream decks from the read-only checkout:
#   minisb (default)  Exec/MiniSB/inputs.32 -- Santa Barbara cluster, 32^3 cells,
#                     32,686 DM particles from the shipped ASCII IC, 10 steps;
#                     exactly upstream's nightly GPU regression test "MiniSB"
#                     (run there as `inputs.32 nyx.ppm_type=0`, 2 MPI ranks).
#                     Adiabatic (no heating/cooling in the deck). smoke only.
#   lya_adiabatic     smoke : Exec/LyA/inputs.rt.garuda -- upstream's GPU regression
#                             test "LyA-adiabatic" deck (heat_cool_type=0,
#                             strang_split=1, 32^3 cells, shipped 32.nyx IC, z=100).
#                     strong: Exec/LyA/inputs (the flagship 64^3 Lyman-alpha science
#                             deck with the shipped 64sssss_20mpc.nyx IC, z=159) with
#                             heating/cooling turned OFF (heat_cool_type=0, sdc_split=0,
#                             strang_split=1, UVB table unused). This is a NAMED
#                             ADIABATIC DERIVATIVE of LyA; it does not cover the
#                             heating/cooling LyA workload (see lya_heatcool).
#   lya_heatcool      Exec/LyA/inputs as shipped (heat_cool_type=11, CVODE via
#                     SUNDIALS) -- requires a *heatcool* profile (HPCPERF_NYX_HEATCOOL=YES build).
#   scaling_synthetic weak: Exec/Scaling/inputs physics with nyx.particle_init_type=RandomPerCell
#                     (upstream labels this initialisation as testing-only): a
#                     SYNTHETIC scaling/communication test, NOT a science IC.
#
# Decomposition: `amr.max_grid_size` fixes the box size so the BoxArray is the
# same for every rank count (amr.refine_grid_layout=0, as upstream's MiniSB deck);
# ranks are refused when boxes < ranks (a rank without a box is never launched
# silently) and reported when boxes % ranks != 0 (imbalance). Nothing changes N.
# Rank -> GPU: AMReX binds device 0 of what the launcher's per-rank wrapper
# exposes (one visible GPU per rank, audited by the launcher).
#
# Controls:
#   HPCPERF_GPUS=N|all            ranks (= GPUs, default 1)
#   HPCPERF_SCALE_MODE            smoke | strong | weak (default smoke)
#   HPCPERF_NYX_STEPS             max_step (default: 10 = upstream's MiniSB test length)
#   HPCPERF_NYX_PROFILE           build profile (default: cuda<tk>-gcc<v>-adiabatic|heatcool)
#   HPCPERF_NYX_HEATCOOL=YES      select the heatcool profile
#   HPCPERF_NYX_GPU_AWARE=0|1     amrex.use_gpu_aware_mpi (default: AMReX auto)
#   HPCPERF_NYX_MGS               override amr.max_grid_size (decomposition control only)
#   HPCPERF_NYX_WEAK_CELLS        weak: cells per rank per dimension (default 64)
# Derived deck (class A): upstream lines kept verbatim except the parameters listed
# in the header of the written file (I/O cadence, decomposition, and -- for the named
# adiabatic derivative -- heat_cool_type/sdc_split/strang_split); the official test
# command's `amrex.the_arena_init_size=0 amr.checkpoint_files_output=0` are applied.
# stdout is copied to <run_dir>/stdout.log; a run_manifest.txt records provenance.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"; [ $# -gt 0 ] && shift
HC="$(echo "${HPCPERF_NYX_HEATCOOL:-NO}" | tr '[:lower:]' '[:upper:]')"
VARIANT=adiabatic; [ "$HC" = YES ] && VARIANT=heatcool
GCC_MM="$(l3_version_mm "$("$CXX" -dumpfullversion 2>/dev/null || "$CXX" -dumpversion)")"
case "$BACKEND" in
    CUDA) MODEL=cuda; PROFILE_DEFAULT="cuda$(l3_version_mm "$(l3_cuda_version)")-gcc${GCC_MM}-${VARIANT}" ;;
    HIP)  MODEL=hip;  PROFILE_DEFAULT="hip-${HPCPERF_HIP_ARCH:-gfx950}-${VARIANT}" ;;
    CPU)  MODEL=cpu;  PROFILE_DEFAULT="cpu-gcc${GCC_MM}-${VARIANT}" ;;
    *) echo "usage: $0 [CUDA|HIP|CPU] [inputs overrides]" >&2; exit 2 ;;
esac
PROFILE="${HPCPERF_NYX_PROFILE:-$PROFILE_DEFAULT}"
l3_paths_profile nyx "$PROFILE"
SRC="$R/_upstream/level3/Nyx"
CASE="${HPCPERF_NYX_CASE:-minisb}"
MODE="$(l3_scale_mode nyx)" || exit 2
N_RANKS="$(hpcperf_ranks nyx yes)" || exit 2
STEPS="${HPCPERF_NYX_STEPS:-10}"
hpcperf_forbid_args nyx max_step amr.n_cell amr.max_grid_size amr.refine_grid_layout amr.plot_int amr.plot_file amr.check_int \
    amr.checkpoint_files_output nyx.heat_cool_type nyx.particle_init_type nyx.binary_particle_file nyx.ascii_particle_file \
    geometry.prob_hi nyx.particle_initrandom_mass_total -- "$@" || exit 2

# ----------------------------------------------------------------------------- case -> deck
EXTRA=()          # deck lines appended (documented in the derived file header)
DROP='^\s*(max_step|amr\.plot_int|amr\.check_int|amr\.checkpoint_files_output|amr\.plot_file|amr\.check_file|amr\.max_grid_size|amr\.refine_grid_layout)\s*='
LINKS=()          # upstream data files the deck references by relative name
case "$CASE" in
    minisb)
        [ "$MODE" = smoke ] || { echo "run.sh: case minisb is the 32^3 official regression deck: smoke only (use lya_adiabatic for strong, scaling_synthetic for weak)" >&2; exit 2; }
        EXE="$L3_INSTALL/bin/nyx_MiniSB"; DECK="$SRC/Exec/MiniSB/inputs.32"; LINKS=(ic_sb_32.ascii)
        NX=32; MGS="${HPCPERF_NYX_MGS:-16}"
        DROP="$DROP|^\s*nyx\.ppm_type\s*="; EXTRA+=("nyx.ppm_type = 0   # as upstream's nightly GPU test command") ;;
    lya_adiabatic)
        EXE="$L3_INSTALL/bin/nyx_LyA"
        [ "$HC" = NO ] || { echo "run.sh: lya_adiabatic is defined for the adiabatic profile (HPCPERF_NYX_HEATCOOL=NO)" >&2; exit 2; }
        case "$MODE" in
            smoke)  DECK="$SRC/Exec/LyA/inputs.rt.garuda"; LINKS=(32.nyx); NX=32; MGS="${HPCPERF_NYX_MGS:-16}" ;;
            strong) DECK="$SRC/Exec/LyA/inputs"; LINKS=(64sssss_20mpc.nyx); NX=64; MGS="${HPCPERF_NYX_MGS:-16}"
                    DROP="$DROP|^\s*nyx\.(heat_cool_type|sdc_split|strang_split|uvb_rates_file)\s*="
                    EXTRA+=("nyx.heat_cool_type = 0   # NAMED ADIABATIC DERIVATIVE of Exec/LyA/inputs (heating/cooling OFF)" \
                            "nyx.sdc_split = 0" "nyx.strang_split = 1   # required by Nyx without SDC/HEATCOOL (as inputs.rt.garuda)") ;;
            weak)   echo "run.sh: weak scaling with a science IC is not defined for Nyx (the shipped ICs are two different cosmologies); use HPCPERF_NYX_CASE=scaling_synthetic (labelled synthetic)" >&2; exit 2 ;;
        esac ;;
    lya_heatcool)
        EXE="$L3_INSTALL/bin/nyx_LyA"
        [ "$HC" = YES ] || { echo "run.sh: lya_heatcool needs the heatcool profile: HPCPERF_NYX_HEATCOOL=YES (build.sh + run.sh)" >&2; exit 2; }
        [ "$MODE" != weak ] || { echo "run.sh: weak not defined for lya_heatcool" >&2; exit 2; }
        case "$MODE" in
            smoke)  DECK="$SRC/Exec/LyA/inputs.rt"; LINKS=(32.nyx TREECOOL_middle); NX=32; MGS="${HPCPERF_NYX_MGS:-16}" ;;
            strong) DECK="$SRC/Exec/LyA/inputs"; LINKS=(64sssss_20mpc.nyx TREECOOL_middle); NX=64; MGS="${HPCPERF_NYX_MGS:-16}" ;;
        esac ;;
    scaling_synthetic)
        EXE="$L3_INSTALL/bin/nyx_LyA"; DECK="$SRC/Exec/Scaling/inputs"; LINKS=(TREECOOL_middle)
        [ "$MODE" != smoke ] || { echo "run.sh: scaling_synthetic is a scaling deck (HPCPERF_SCALE_MODE=strong|weak); correctness cases are minisb / lya_adiabatic" >&2; exit 2; }
        L="${HPCPERF_NYX_WEAK_CELLS:-64}"
        # upstream deck: 64^3 cells in a 28.49002849 Mpc box, RandomPerCell, total DM mass 869658119634944.0
        BOX0=28.49002849; MASS0=869658119634944.0
        if [ "$MODE" = weak ]; then
            TOPO="$(hpcperf_topology nyx "$N_RANKS")" || exit 2; read -r PX PY PZ <<< "$TOPO"
            NX=$((L * PX)); NY=$((L * PY)); NZ=$((L * PZ)); MGS="${HPCPERF_NYX_MGS:-$L}"; WHAT="weak: ${L}^3 cells per rank, ${PX}x${PY}x${PZ} tiles"
        else
            G="${HPCPERF_NYX_GLOBAL:-256}"; [ $((G % 64)) -eq 0 ] || { echo "run.sh: HPCPERF_NYX_GLOBAL=$G must be a multiple of 64" >&2; exit 2; }
            PX=$((G / 64)); PY=$PX; PZ=$PX; NX=$G; NY=$G; NZ=$G; MGS="${HPCPERF_NYX_MGS:-64}"; WHAT="strong: fixed ${G}^3 global grid (upstream 64^3 deck scaled x$PX per dimension)"
        fi
        PHX="$(python3 -c "print(f'{$BOX0*$PX:.8f}')")"; PHY="$(python3 -c "print(f'{$BOX0*$PY:.8f}')")"; PHZ="$(python3 -c "print(f'{$BOX0*$PZ:.8f}')")"
        MASS="$(python3 -c "print(f'{$MASS0*$PX*$PY*$PZ:.1f}')")"
        DROP="$DROP|^\s*(amr\.n_cell|geometry\.prob_hi|nyx\.particle_initrandom_mass_total)\s*="
        EXTRA+=("amr.n_cell = $NX $NY $NZ   # $WHAT" \
                "geometry.prob_hi = $PHX $PHY $PHZ   # box scaled with the tiles (mean density unchanged)" \
                "nyx.particle_initrandom_mass_total = $MASS   # total DM mass scaled with the volume")
        if [ "$HC" = NO ]; then
            DROP="$DROP|^\s*nyx\.(heat_cool_type|sdc_split|strang_split|uvb_rates_file)\s*="
            EXTRA+=("nyx.heat_cool_type = 0   # adiabatic profile" "nyx.sdc_split = 0" "nyx.strang_split = 1")
        fi ;;
    *) echo "run.sh: HPCPERF_NYX_CASE must be minisb | lya_adiabatic | lya_heatcool | scaling_synthetic (got '$CASE')" >&2; exit 2 ;;
esac
[ -x "$EXE" ] || { echo "run.sh: $EXE not found -- run ./build.sh $BACKEND (profile $PROFILE) first" >&2; exit 1; }
[ -f "$DECK" ] || { echo "run.sh: deck $DECK missing (run fetch.sh)" >&2; exit 1; }
[ "$MODEL" = cpu ] || l3_binary_backend_check "$EXE" "$MODEL" || exit 1
: "${NY:=$NX}"; : "${NZ:=$NX}"
# boxes with the fixed layout (n_cell / max_grid_size per dimension)
for d in "$NX" "$NY" "$NZ"; do [ $((d % MGS)) -eq 0 ] || { echo "run.sh: amr.n_cell $d not divisible by amr.max_grid_size $MGS" >&2; exit 2; }; done
BOXES=$(( (NX / MGS) * (NY / MGS) * (NZ / MGS) ))
if [ "$N_RANKS" -gt "$BOXES" ]; then
    echo "run.sh: $N_RANKS ranks requested but the ${NX}x${NY}x${NZ} grid with amr.max_grid_size=$MGS has only $BOXES boxes -- a rank without work is never launched silently; lower HPCPERF_GPUS or set HPCPERF_NYX_MGS (e.g. $((MGS / 2)) -> $((BOXES * 8)) boxes)" >&2; exit 2
fi
BALANCE=balanced; [ $((BOXES % N_RANKS)) -eq 0 ] || BALANCE="IMBALANCED ($BOXES boxes over $N_RANKS ranks)"

RUN_DIR="$(l3_rundir "$L3_BUILD/run/$CASE.$MODE.np$N_RANKS")" || exit 2
IN="$RUN_DIR/inputs"
{
    echo "# derived from upstream $(realpath --relative-to="$SRC" "$DECK") (HPC-Performance-AI level3/nyx/run.sh; profile $PROFILE)"
    echo "# upstream lines removed and replaced below: max_step, amr.plot_int/plot_file/check_int/check_file/checkpoint_files_output, amr.max_grid_size, amr.refine_grid_layout$( [ ${#EXTRA[@]} -gt 0 ] && echo ', plus the case-specific lines marked with a comment')"
    grep -vE "$DROP" "$DECK"
    echo ""
    echo "# --- level3/nyx/run.sh ---"
    echo "max_step = $STEPS"
    echo "amr.max_grid_size = $MGS   # fixed BoxArray ($BOXES boxes) for every rank count"
    echo "amr.refine_grid_layout = 0"
    echo "amr.plot_file = plt"
    echo "amr.plot_int = $STEPS      # plt00000 (initial) and plt$(printf '%05d' "$STEPS") (final)"
    echo "amr.plot_vars = ALL"
    echo "amr.check_file = chk"
    echo "amr.check_int = $STEPS      # chk00000 and chk$(printf '%05d' "$STEPS"): particle ids for the cross-rank-count"
    echo "amr.checkpoint_files_output = 1   # particle comparison (nyx_particle_compare.py); upstream's test command disables checkpoints"
    echo "amrex.the_arena_init_size = 0     # as upstream's nightly test command"
    for e in "${EXTRA[@]}"; do echo "$e"; done
    [ -n "${HPCPERF_NYX_GPU_AWARE:-}" ] && echo "amrex.use_gpu_aware_mpi = $HPCPERF_NYX_GPU_AWARE"
} > "$IN"
for f in "${LINKS[@]}"; do
    src="$(dirname "$DECK")/$f"; [ -f "$src" ] || { echo "run.sh: upstream data file $src missing" >&2; exit 1; }
    ln -sfn "$src" "$RUN_DIR/$f"
done

BIND=wrapper; [ "$MODEL" = cpu ] && BIND=none
echo "# Nyx $BACKEND profile=$PROFILE case=$CASE mode=$MODE ranks=$N_RANKS grid=${NX}x${NY}x${NZ} max_grid_size=$MGS boxes=$BOXES ($BALANCE) steps=$STEPS exe=$(basename "$EXE") deck=$(realpath --relative-to="$SRC" "$DECK") run_dir=$RUN_DIR"
cd "$RUN_DIR"
RUN_ID="$(l3_run_id)"
# real exit code of the launcher/application (pipefail: tee's 0 never masks it; set +e so it is recorded, not aborted on)
set +e; set -o pipefail
"$L3_LAUNCHER" --gpus "$N_RANKS" --bind "$BIND" -- "$EXE" "$IN" "$@" 2>&1 | tee "$RUN_DIR/stdout.log"
rc=$?
set +o pipefail; set -e
if [ -z "${HPCPERF_DRY_RUN:-}" ]; then
    ICS=""; for f in "${LINKS[@]}"; do ICS="$ICS $f=$(l3_sha_file "$(dirname "$DECK")/$f")"; done
    l3_manifest "$RUN_DIR" "run_id=$RUN_ID" "app=nyx" "backend=$BACKEND" "profile=$PROFILE" "case=$CASE" "mode=$MODE" \
        "ranks=$N_RANKS" "grid=${NX}x${NY}x${NZ}" "max_grid_size=$MGS" "boxes=$BOXES" "steps=$STEPS" \
        "exit_code=$rc" "binary=$EXE" "binary_sha256=$(l3_sha_file "$EXE")" "deck=$DECK" "deck_sha256=$(l3_sha_file "$DECK")" \
        "input=$IN" "input_sha256=$(l3_sha_file "$IN")" "ic_sha256=${ICS# }" \
        "fingerprint=$L3_INSTALL/.hpcperf-l3-fingerprint" "fingerprint_sha256=$(l3_sha_file "$L3_INSTALL/.hpcperf-l3-fingerprint")" \
        "stdout=$RUN_DIR/stdout.log" "utc=$(date -u +%FT%TZ)"
fi
exit "$rc"
