#!/usr/bin/env bash
# Correctness check for DFT-FE (dftfe_real, CUDA) on N GPUs.
#
#   ./validate.sh [CUDA]       HPCPERF_GPUS=N (default 1)
#
# [0] ELPA GPU kernels verified independently: <install>/elpa/ELPA_GPU_PROBE.txt must say RESULT PASS
#     (elpa_probe.sh: ELPA's own analytic-matrix test programs with the sm_100 GPU kernels on 1/2/4 GPUs,
#     ELPA's eigenvalue/eigenvector error limits, exit 0, GPU timers present, launcher audit 0 mismatch).
# [1] al_md (upstream GPU regression deck Input_MD_0.prm, verbatim) on N GPUs, dftfe_check.py --check
#     against upstream's own GPU reference accuracyBenchmarks/output_MD_0: SCF converged, MD completed
#     (4 steps), ground-state energy within 1e-5 Ha, per-step MD total energies within 2e-5 Ha,
#     temperatures within 0.1 K, ion forces within 2e-5 Ha/Bohr (pre-fixed, see dftfe_check.py), every
#     value finite; GPU evidence: the deck's USE GPU = true is honoured by a CUDA binary (backend check
#     in run.sh) and the launcher audit must report every rank verified on its own GPU with 0 mismatch.
#     For N > 1 the same quantities are ALSO compared with the 1-GPU run of the same binary/deck under
#     the same tolerances (rank-count independence).
# Exit codes are captured (timeout/nonzero -> FAIL); nothing is read from old runs except the 1-GPU
# reference, which is re-run when its binary hash differs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
N="${HPCPERF_GPUS:-1}"
TIMEOUT="${HPCPERF_VALIDATE_TIMEOUT:-3600}"
GCC_MM="$(l3_version_mm "$(/usr/bin/gcc -dumpfullversion)")"; OMPI_V="$(mpirun --version | head -1 | /usr/bin/grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
PROFILE="${HPCPERF_DFTFE_PROFILE:-cuda$(l3_version_mm "$(l3_cuda_version)")-gcc${GCC_MM}-ompi$(echo "$OMPI_V" | tr -d .)}"
l3_paths_profile dftfe "$PROFILE"
SRC="$R/_upstream/level3/dftfe"; RUNS="$L3_BUILD/run"; INST="$L3_INSTALL"
REF="$SRC/testsGPU/pseudopotential/real/accuracyBenchmarks/output_MD_0"
export HPCPERF_GPUS="$N" HPCPERF_SCALE_MODE=smoke HPCPERF_DFTFE_CASE=al_md
mkdir -p "$RUNS"; ok=1
fail() { echo "validate.sh: FAIL -- $*"; ok=0; }
manifest_val() { /usr/bin/grep -m1 "^$2=" "$1/run_manifest.txt" 2>/dev/null | cut -d= -f2- || true; }

echo "validate.sh: [0] ELPA GPU kernel probe record [profile $PROFILE]"
if [ -f "$INST/elpa/ELPA_GPU_PROBE.txt" ]; then
    /usr/bin/grep -E '^(PASS|FAIL|MISSING|RESULT)' "$INST/elpa/ELPA_GPU_PROBE.txt" | sed 's/^/    /' | cut -c1-200
    /usr/bin/grep -q '^RESULT PASS' "$INST/elpa/ELPA_GPU_PROBE.txt" || fail "ELPA GPU probe did not PASS"
else
    fail "ELPA GPU probe record missing -- run ./elpa_probe.sh first"
fi

echo "validate.sh: [1] DFT-FE $BACKEND al_md (Input_MD_0.prm: 32-atom Al BOMD, 4 steps) on $N GPU(s) vs upstream GPU reference"
D="$RUNS/al_md.smoke.np$N"
rc=0; timeout "$TIMEOUT" "$HERE/run.sh" "$BACKEND" > "$RUNS/validate.al_md.np$N.stdout" 2>&1 || rc=$?
/usr/bin/grep -aE '^# DFT-FE|hpcperf-launch: audit summary' "$RUNS/validate.al_md.np$N.stdout" || true
if [ "$rc" -eq 124 ]; then fail "al_md timed out after ${TIMEOUT}s"; elif [ "$rc" -ne 0 ]; then fail "al_md run.sh exited $rc (see $RUNS/validate.al_md.np$N.stdout)"; fi
audit="$(/usr/bin/grep -a 'audit summary' "$RUNS/validate.al_md.np$N.stdout" | tail -1)"
echo "$audit" | /usr/bin/grep -qE "audit summary: $N verified, 0 mismatch, 0 unverified" || fail "al_md np$N: launcher GPU audit is not '$N verified, 0 mismatch, 0 unverified' (${audit:-no audit line})"
if [ -f "$D/dftfe.out" ]; then
    python3 "$HERE/dftfe_check.py" "$D/dftfe.out" "$REF" --check --label "np$N vs upstream" > "$D/check_vs_upstream.txt" || fail "al_md np$N: mismatch against upstream reference / incomplete / non-finite"
    sed 's/^/    /' "$D/check_vs_upstream.txt"
    if [ "$N" -gt 1 ]; then
        R1="$RUNS/al_md.smoke.np1"
        if [ ! -f "$R1/dftfe.out" ] || [ "$(manifest_val "$R1" binary_sha256)" != "$(manifest_val "$D" binary_sha256)" ] || [ "$(manifest_val "$R1" exit_code)" != 0 ]; then
            echo "    (1-GPU reference missing/stale -- running al_md on 1 GPU now)"
            rc=0; HPCPERF_GPUS=1 timeout "$TIMEOUT" "$HERE/run.sh" "$BACKEND" > "$RUNS/validate.al_md.np1.stdout" 2>&1 || rc=$?
            [ "$rc" -eq 0 ] || fail "al_md 1-GPU reference run exited $rc"
        fi
        python3 "$HERE/dftfe_check.py" "$D/dftfe.out" "$R1/dftfe.out" --check --label "np$N vs np1" > "$D/check_vs_np1.txt" || fail "al_md np$N: mismatch against the 1-GPU run"
        sed 's/^/    /' "$D/check_vs_np1.txt"
    fi
fi

if [ "$ok" -eq 1 ]; then echo "DFT-FE $BACKEND validation ($N GPU; ELPA GPU probe PASS, al_md complete/finite within pre-fixed tolerances of upstream's GPU reference$( [ "$N" -gt 1 ] && echo " and of the 1-GPU run" || true), every rank verified on its own GPU): PASS"; exit 0; fi
echo "DFT-FE $BACKEND validation ($N GPU): FAIL"; exit 1
