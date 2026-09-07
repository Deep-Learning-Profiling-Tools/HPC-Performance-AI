#!/usr/bin/env bash
# Run DFT-FE (dftfe_real, CUDA) on N GPUs, one MPI rank per GPU, 1 thread per rank (as upstream's GPU
# job scripts: OMP_NUM_THREADS=DFTFE_NUM_THREADS=DEAL_II_NUM_THREADS=1).
#
#   ./run.sh [CUDA]
#
# Cases (HPCPERF_DFTFE_CASE), inputs from the read-only checkout testsGPU/pseudopotential/real/:
#   al_md (default)  Input_MD_0.prm -- upstream's GPU regression case: 32-atom fcc Al supercell
#       (15.28 bohr cube, ONCV PBE, order-3 FE, 85 Kohn-Sham states, ANDERSON_WITH_KERKER mixing,
#       SCF TOLERANCE 1e-5), Born-Oppenheimer NVE MD at 1400 K, 4 steps of 1 fs, USE GPU = true,
#       REPRODUCIBLE OUTPUT = true; upstream reference accuracyBenchmarks/output_MD_0.
#         smoke / strong : the deck verbatim (fixed 32-atom problem; strong scaling of a SMALL problem)
#         weak           : derived Al supercell series, 32 atoms per GPU: N=1 -> 2x2x2 cells (32 atoms,
#                          verbatim), N=2 -> 4x2x2 (64), N=4 -> 4x4x2 (128); coordinates replicated,
#                          domain vectors scaled, NATOMS and NUMBER OF KOHN-SHAM WAVEFUNCTIONS (85 per
#                          32 atoms) scaled; everything else verbatim (diff recorded). SYNTHETIC series
#                          (same construction as upstream's dftfe-benchmarks Mo supercells).
#   llzo             parameterFile_LLZO.prm -- 192-atom Li7La3Zr2O12 ground state (720 states, USE ELPA,
#                    GPU); upstream reference accuracyBenchmarks/outputLLZO. smoke/strong verbatim only
#                    (a heavier fixed-size strong-scaling case).
# Controls: HPCPERF_GPUS=N|all, HPCPERF_CPUS_PER_RANK (CPU cores bound per rank, default 4; threads
#           stay 1), HPCPERF_SCALE_MODE=smoke|strong|weak, HPCPERF_DFTFE_PROFILE, HPCPERF_DRY_RUN=1
# Output: <run_dir>/dftfe.out (stdout+stderr incl. launcher audit), run_manifest.txt.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"; [ $# -gt 0 ] && shift
[ "$BACKEND" = CUDA ] || { echo "run.sh: only CUDA is built for DFT-FE here" >&2; exit 2; }
GCC_MM="$(l3_version_mm "$(/usr/bin/gcc -dumpfullversion)")"; OMPI_V="$(mpirun --version | head -1 | /usr/bin/grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
PROFILE="${HPCPERF_DFTFE_PROFILE:-cuda$(l3_version_mm "$(l3_cuda_version)")-gcc${GCC_MM}-ompi$(echo "$OMPI_V" | tr -d .)}"
l3_paths_profile dftfe "$PROFILE"
SRC="$R/_upstream/level3/dftfe"; T="$SRC/testsGPU/pseudopotential/real"; INST="$L3_INSTALL"
EXE="$INST/bin/dftfe_real"
[ -x "$EXE" ] || { echo "run.sh: $EXE not found -- run ./build.sh first (profile $PROFILE)" >&2; exit 1; }
export LD_LIBRARY_PATH="$INST/dealii/lib:$INST/dealii/lib64:$INST/elpa/lib:$INST/scalapack/lib:$INST/openblas/lib:$INST/libxc/lib:$INST/libxc/lib64:$INST/spglib/lib:$INST/spglib/lib64:$INST/kokkos/lib:$INST/kokkos/lib64:$INST/p4est/FAST/lib:$INST/alglib:$L3_BUILD/real:${LD_LIBRARY_PATH:-}"
l3_binary_backend_check "$EXE" cuda || exit 1
CASE="${HPCPERF_DFTFE_CASE:-al_md}"
MODE="$(l3_scale_mode dftfe)" || exit 2
N_RANKS="$(hpcperf_ranks dftfe yes)" || exit 2
CPUS="${HPCPERF_CPUS_PER_RANK:-4}"
case "$CASE" in
    al_md) PRM="$T/Input_MD_0.prm"; FILES="aluminumMD_coordinates.inp aluminumMD_domainBoundingVectors.inp aluminumMD_pseudo.inp Al.upf Mass.inp"; REF="$T/accuracyBenchmarks/output_MD_0" ;;
    llzo)  PRM="$T/parameterFile_LLZO.prm"; FILES="coordinates_LLZO.inp domainVectors_LLZO.inp pseudo_LLZO.inp Li.upf La.upf O.upf Mass.inp"; REF="$T/accuracyBenchmarks/outputLLZO"
           [ "$MODE" != weak ] || { echo "run.sh: llzo has no weak series" >&2; exit 2; }
           for f in $(awk '{print $2}' "$T/pseudo_LLZO.inp"); do FILES="$FILES $f"; done ;;
    *) echo "run.sh: HPCPERF_DFTFE_CASE must be al_md or llzo" >&2; exit 2 ;;
esac
[ -f "$PRM" ] || { echo "run.sh: $PRM missing (run fetch.sh)" >&2; exit 1; }
LABEL="$CASE"; DERIV="verbatim"
[ "$MODE" = weak ] && { LABEL="$CASE.x$N_RANKS"; DERIV="al-supercell x$N_RANKS (32 atoms/GPU)"; }
RUN_DIR="$(l3_rundir "$L3_BUILD/$L3_RUN_SUBDIR/$LABEL.$MODE.np$N_RANKS")" || exit 2
for f in $(echo "$FILES" | tr ' ' '\n' | sort -u); do [ -f "$T/$f" ] || { echo "run.sh: input $T/$f missing" >&2; exit 1; }; ln -sfn "$T/$f" "$RUN_DIR/$f"; done
if [ "$MODE" = weak ] && [ "$N_RANKS" -gt 1 ]; then
    case "$N_RANKS" in 2) REP="2 1 1" ;; 4) REP="2 2 1" ;; 8) REP="2 2 2" ;; *) echo "run.sh: weak series defined for 1/2/4/8 GPUs" >&2; exit 2 ;; esac
    python3 - "$T" "$RUN_DIR" "$PRM" $REP <<'PY'
import math, re, sys
T, out, prm = sys.argv[1], sys.argv[2], sys.argv[3]; rx, ry, rz = (int(v) for v in sys.argv[4:7])
coords = [l.split() for l in open(f"{T}/aluminumMD_coordinates.inp") if l.strip()]
dom = [[float(v) for v in l.split()] for l in open(f"{T}/aluminumMD_domainBoundingVectors.inp") if l.strip()]
new = []
for i in range(rx):
    for j in range(ry):
        for k in range(rz):
            for c in coords:
                x, y, z = (float(c[2]) + i) / rx, (float(c[3]) + j) / ry, (float(c[4]) + k) / rz
                new.append(f"{c[0]} {c[1]} {x:.12f} {y:.12f} {z:.12f}\n")
open(f"{out}/al_coordinates.inp", "w").writelines(new)
scale = [rx, ry, rz]
open(f"{out}/al_domainBoundingVectors.inp", "w").writelines(" ".join(f"{v * scale[i]:.10f}" for v in dom[i]) + "\n" for i in range(3))
n = len(new); nwf = math.ceil(85 * n / 32)
txt = open(prm).read()
txt, c1 = re.subn(r"(set NATOMS\s*=\s*)\d+", rf"\g<1>{n}", txt)
txt, c2 = re.subn(r"(set NUMBER OF KOHN-SHAM WAVEFUNCTIONS\s*=\s*)\d+", rf"\g<1>{nwf}", txt)
txt, c3 = re.subn(r"aluminumMD_coordinates\.inp", "al_coordinates.inp", txt)
txt, c4 = re.subn(r"aluminumMD_domainBoundingVectors\.inp", "al_domainBoundingVectors.inp", txt)
if not (c1 == c2 == c3 == c4 == 1): sys.exit(f"run.sh: deck derivation touched unexpected counts {c1, c2, c3, c4}")
open(f"{out}/parameters.prm", "w").write(txt)
print(f"# derived Al supercell: {rx}x{ry}x{rz} replication -> {n} atoms, {nwf} Kohn-Sham states")
PY
    diff -u "$PRM" "$RUN_DIR/parameters.prm" > "$RUN_DIR/deck.diff" || true
else
    cp "$PRM" "$RUN_DIR/parameters.prm"
fi
export OMP_NUM_THREADS=1 DFTFE_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
echo "# DFT-FE $BACKEND profile=$PROFILE case=$CASE mode=$MODE ($DERIV) ranks=$N_RANKS deck=$(realpath --relative-to="$SRC" "$PRM") run_dir=$RUN_DIR"
cd "$RUN_DIR"
RUN_ID="$(l3_run_id)"
set +e; set -o pipefail
"$L3_LAUNCHER" --gpus "$N_RANKS" --cpus-per-rank "$CPUS" --bind wrapper -- "$EXE" parameters.prm 2>&1 | tee "$RUN_DIR/dftfe.out"
rc=$?
set +o pipefail; set -e
if [ -z "${HPCPERF_DRY_RUN:-}" ]; then
    l3_manifest "$RUN_DIR" "run_id=$RUN_ID" "app=dftfe" "backend=$BACKEND" "profile=$PROFILE" "case=$CASE" "label=$LABEL" "mode=$MODE" "deck_derivation=$DERIV" \
        "ranks=$N_RANKS" "threads_per_rank=1" "exit_code=$rc" "binary=$EXE" "binary_sha256=$(l3_sha_file "$EXE")" \
        "upstream_prm=$PRM" "upstream_prm_sha256=$(l3_sha_file "$PRM")" "deck_sha256=$(l3_sha_file "$RUN_DIR/parameters.prm")" "reference_output=$REF" "reference_sha256=$(l3_sha_file "$REF")" \
        "fingerprint_sha256=$(l3_sha_file "$INST/.hpcperf-l3-fingerprint")" "elpa_gpu_probe=$(/usr/bin/grep -m1 '^RESULT' "$INST/elpa/ELPA_GPU_PROBE.txt" 2>/dev/null || echo NOT_RUN)" "utc=$(date -u +%FT%TZ)"
    /usr/bin/grep -aE 'Device|GPU|device' "$RUN_DIR/dftfe.out" 2>/dev/null | head -6 | sed 's/^/gpu_evidence: /' >> "$RUN_DIR/run_manifest.txt" || true
fi
exit "$rc"
