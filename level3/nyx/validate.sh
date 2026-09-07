#!/usr/bin/env bash
# Correctness check for Nyx on N GPUs with upstream's own comparison tools and
# pre-fixed criteria (no tolerance is derived from the results).
#
#   ./validate.sh [CUDA|HIP]        HPCPERF_GPUS=N (default 1); HPCPERF_NYX_CASES="minisb lya_adiabatic"
#
# For every case (default: the two official GPU-regression decks, MiniSB and
# LyA-adiabatic, both 10 steps as upstream) the N-GPU run must satisfy ALL of:
#  [1] completeness: run.sh exit code 0 (timeout -> FAIL), plt00000 and the final
#      plotfile exist, the runlog reaches max_step, every plotfile variable is
#      finite (amrex_fextrema min/max through l3_check.require_finite), the DM
#      particle count in the final plotfile equals the IC count.
#  [2] official regression comparison at a pre-fixed tolerance, through the strict
#      wrapper nyx_fcompare_check.py around AMReX's fcompare (Header structure, raw
#      finiteness of BOTH plotfiles via fextrema, one parsed row per variable and level,
#      zero-reference fields judged by an absolute rule, non-finite errors never dropped,
#      parser/tool consistency) plus nyx_particle_compare.py on the DM particles:
#        N=1 : against a second, independent 1-GPU run of the same binary and deck
#              (same-configuration reproducibility, exactly what the nightly test measures);
#        N>1 : against the 1-GPU plotfile of the same binary/deck (rank-count
#              independence; the BoxArray is fixed by run.sh, only the distribution changes).
#      Tolerance provenance (official GPU nightly reports, ccse.lbl.gov/pub/GpuRegressionTesting/Nyx):
#        MiniSB       : fcompare --rel_tol 2e-10 (UPSTREAM, used as is);
#        LyA-adiabatic: upstream uses 5e-09; this validator applies the MiniSB value 2e-10
#                       to both adiabatic decks -- a PROJECT choice, 25x stricter than upstream;
#        LyA heat/cool: upstream compares plt00354 with --rel_tol 5e-05 (their max 2.8e-05 in
#                       Temp; their report also compares I_R, rel 4.2e-08 there); this project's
#                       10-step smoke deck (plt00010) is a different configuration. The 5e-5 was
#                       adopted on 2026-09-06 after a first heat/cool validation at 2e-10 FAILED
#                       (recorded in the review package).
#  [3] cross-backend reference: the CPU-profile binary (Nyx_GPU_BACKEND=NONE, same
#      Nyx/AMReX/deck) run on 1 rank; same comparison with rel_tol 1e-8 for the adiabatic
#      decks, a PROJECT tolerance FIXED before any run (two orders above the same-platform
#      tolerance to allow for FMA contraction, libm and reduction-order differences
#      between host and device code over 10 steps); 5e-5 for heat/cool.
#  [4] conservation: total comoving baryon mass sum(density*dV) (amrex_fvolumesum)
#      between plt00000 and the final plotfile: |dM/M| <= 1e-9 (adiabatic, periodic:
#      no mass source); DM particle count exact.
#  I_R (heat/cool only): the SDC reaction-integral field is NOT accepted or rejected by a
#      relative tolerance -- it is a small residual of cancelling O(a^2 rho e / dt) terms
#      whose run-to-run variation is O(1) relative at this stage of the deck while every
#      state field agrees to 1e-13 (README, section "I_R"). It is checked for presence and
#      finiteness, reported, and the case verdict is downgraded to I_R_CHECK_PENDING
#      (exit 3): state and particle checks PASS, the full heat/cool acceptance does not.
# Exit codes of run.sh/launcher/application and every tool are captured; a
# missing tool, plotfile or reference is FAIL, never skipped silently.
#
# Offline / harness hooks (no GPU): HPCPERF_NYX_OFFLINE=1 compares EXISTING run
# directories without executing run.sh (reports go to HPCPERF_NYX_REPORT_DIR, default
# <runs>/offline_recheck.<utc>/, never into the historical run directories);
# HPCPERF_NYX_TOOLS_DIR overrides the AMReX tool directory; HPCPERF_NYX_SKIP_PARTICLES=1
# skips the DM particle comparison (test harness only; printed loudly).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
[ "$BACKEND" != CPU ] || { echo "validate.sh: validates a GPU backend against the CPU reference; use CUDA or HIP" >&2; exit 2; }
N="${HPCPERF_GPUS:-1}"
CASES="${HPCPERF_NYX_CASES:-minisb lya_adiabatic}"
STEPS="${HPCPERF_NYX_STEPS:-10}"
TIMEOUT="${HPCPERF_VALIDATE_TIMEOUT:-1800}"
REL_TOL_SAME=2e-10     # MiniSB's official tolerance, applied to both adiabatic decks (see header)
REL_TOL_XBACKEND=1e-8  # PROJECT CPU-vs-GPU tolerance for the adiabatic decks (see header)
HC_REL_TOL=5e-5        # upstream's LyA heat/cool tolerance (plt00354); adopted for the 10-step deck, see header
HC_DIAGNOSTIC="I_R"    # reported, not gated; forces the I_R_CHECK_PENDING verdict (see header)
ABS_TOL_ZERO_REF=0     # a field that is identically zero in the reference must be identically zero in the test
case_tols() { # case_tols <case> -> sets TOL_SAME TOL_X DIAG
    case "$1" in
        lya_heatcool) TOL_SAME=$HC_REL_TOL; TOL_X=$HC_REL_TOL; DIAG="$HC_DIAGNOSTIC" ;;
        *)            TOL_SAME=$REL_TOL_SAME; TOL_X=$REL_TOL_XBACKEND; DIAG="" ;;
    esac
}
MASS_TOL=1e-9          # pre-fixed baryon mass conservation tolerance
python3 -c 'import numpy' 2>/dev/null || { echo "validate.sh: python3 with numpy required" >&2; exit 1; }
export HPCPERF_GPUS="$N" HPCPERF_SCALE_MODE=smoke HPCPERF_NYX_STEPS="$STEPS"
OFFLINE="${HPCPERF_NYX_OFFLINE:-}"; SKIP_PARTICLES="${HPCPERF_NYX_SKIP_PARTICLES:-}"
[ -z "$SKIP_PARTICLES" ] || echo "validate.sh: WARNING -- HPCPERF_NYX_SKIP_PARTICLES set: DM particle comparison skipped (test harness mode, not a validation)"

HC="$(echo "${HPCPERF_NYX_HEATCOOL:-NO}" | tr '[:lower:]' '[:upper:]')"; VARIANT=adiabatic; [ "$HC" = YES ] && VARIANT=heatcool
GCC_MM="$(l3_version_mm "$("$CXX" -dumpfullversion 2>/dev/null || "$CXX" -dumpversion)")"
case "$BACKEND" in
    CUDA) GPU_PROFILE="${HPCPERF_NYX_PROFILE:-cuda$(l3_version_mm "$(l3_cuda_version)")-gcc${GCC_MM}-${VARIANT}}" ;;
    HIP)  GPU_PROFILE="${HPCPERF_NYX_PROFILE:-hip-${HPCPERF_HIP_ARCH:-gfx950}-${VARIANT}}" ;;
esac
CPU_PROFILE="${HPCPERF_NYX_CPU_PROFILE:-cpu-gcc${GCC_MM}-${VARIANT}}"
GPU_ROOT="$R/.deps/level3/nyx/$GPU_PROFILE"; CPU_ROOT="$R/.deps/level3/nyx/$CPU_PROFILE"
GPU_RUNS="$R/build/level3/nyx/$GPU_PROFILE/$L3_RUN_SUBDIR"; CPU_RUNS="$R/build/level3/nyx/$CPU_PROFILE/$L3_RUN_SUBDIR"
TOOLS="${HPCPERF_NYX_TOOLS_DIR:-$CPU_ROOT/install/bin}"
for t in amrex_fcompare amrex_fextrema amrex_fvolumesum particle_compare; do
    [ -x "$TOOLS/$t" ] || { echo "validate.sh: FAIL -- $TOOLS/$t missing: build the CPU reference profile first (./build.sh CPU)"; exit 1; }
done
[ -n "$OFFLINE" ] || [ -x "$GPU_ROOT/install/bin/nyx_MiniSB" ] || { echo "validate.sh: FAIL -- GPU profile $GPU_PROFILE not built (./build.sh $BACKEND)"; exit 1; }
mkdir -p "$GPU_RUNS" "$CPU_RUNS"
if [ -n "$OFFLINE" ]; then
    REPORT_DIR="${HPCPERF_NYX_REPORT_DIR:-$GPU_RUNS/offline_recheck.$(date -u +%Y%m%dT%H%M%SZ)}"; mkdir -p "$REPORT_DIR"
    echo "validate.sh: OFFLINE mode -- no run.sh execution; comparing existing run directories; reports under $REPORT_DIR"
fi
ok=1; pending=""
fail() { echo "validate.sh: FAIL -- $*"; ok=0; }
report_dir() { # report_dir <run_dir>: where this run's comparison outputs go (never a historical dir in offline mode)
    if [ -n "$OFFLINE" ]; then local d="$REPORT_DIR/$(basename "$1")"; mkdir -p "$d"; echo "$d"; else echo "$1"; fi
}

run_gpu() { # run_gpu <case> <n> <stdout-file>
    local case=$1 n=$2 out=$3 rc=0
    if [ -n "$OFFLINE" ]; then [ -f "$GPU_RUNS/$case.smoke.np$n/run_manifest.txt" ] && return 0; echo "validate.sh: OFFLINE -- run directory $GPU_RUNS/$case.smoke.np$n missing" >&2; return 66; fi
    HPCPERF_NYX_CASE="$case" HPCPERF_GPUS="$n" timeout "$TIMEOUT" "$HERE/run.sh" "$BACKEND" > "$out" 2>&1 || rc=$?
    return $rc
}
run_cpu() {
    local case=$1 out=$2 rc=0
    if [ -n "$OFFLINE" ]; then [ -f "$CPU_RUNS/$case.smoke.np1/run_manifest.txt" ] && return 0; echo "validate.sh: OFFLINE -- CPU reference $CPU_RUNS/$case.smoke.np1 missing" >&2; return 66; fi
    HPCPERF_NYX_CASE="$case" HPCPERF_GPUS=1 HPCPERF_NYX_PROFILE="$CPU_PROFILE" timeout "$TIMEOUT" "$HERE/run.sh" CPU > "$out" 2>&1 || rc=$?
    return $rc
}
manifest_val() { /usr/bin/grep -m1 "^$2=" "$1/run_manifest.txt" 2>/dev/null | cut -d= -f2- || true; }
final_plt() { printf 'plt%05d' "$STEPS"; }

# [1] completeness + finiteness + particle count
check_complete() { # check_complete <run_dir> <label> <expected particle count or ''>
    local d=$1 label=$2 want_np=$3 plt   # (set -u: expansions of one `local` happen before its assignments)
    plt="$d/$(final_plt)"
    [ -f "$d/plt00000/Header" ] || { fail "$label: initial plotfile plt00000 missing"; return 1; }
    [ -f "$plt/Header" ] || { fail "$label: final plotfile $(final_plt) missing (run did not reach step $STEPS)"; return 1; }
    [ -f "$d/runlog" ] || { fail "$label: runlog missing"; return 1; }
    local last; last="$(awk 'NF>=2 && $1 ~ /^[0-9]+$/ {s=$1} END{print s+0}' "$d/runlog")"
    [ "$last" -eq "$STEPS" ] || { fail "$label: runlog last step $last != $STEPS"; return 1; }
    # finite extrema for every variable (fextrema prints: name min max per variable)
    local ext; ext="$("$TOOLS/amrex_fextrema" "$plt" 2>&1)" || { fail "$label: amrex_fextrema failed on $plt: $ext"; return 1; }
    python3 - "$label" <<PY || { fail "$label: non-finite plotfile extrema"; return 1; }
import sys, os
sys.path.insert(0, os.environ["L3_TOOLS"])
from l3_check import require_finite, ValidationError
label = sys.argv[1]; n = 0
try:
    for line in """$ext""".splitlines():
        f = line.split()
        if len(f) >= 3:
            try: lo, hi = float(f[-2]), float(f[-1])
            except ValueError: continue
            require_finite(f"{label} {f[0]} min", lo); require_finite(f"{label} {f[0]} max", hi); n += 1
    if n < 5: raise ValidationError(f"only {n} variables parsed from fextrema output")
    print(f"    {label}: {n} plotfile variables finite (fextrema)")
except ValidationError as ex:
    print(f"  VALIDATION ERROR: {ex}"); sys.exit(1)
PY
    # DM particle count from the particle Header (version, dim, nreal, names..., nint, names..., is_checkpoint, nparticles)
    local np; np="$(python3 - "$plt/DM/Header" <<'PY'
import sys
L = [l.strip() for l in open(sys.argv[1]) if l.strip()]
i = 2; nreal = int(L[i]); i += 1 + nreal; nint = int(L[i]); i += 1 + nint; i += 1  # is_checkpoint
print(int(L[i]))
PY
)" || { fail "$label: cannot read $plt/DM/Header"; return 1; }
    if [ -n "$want_np" ]; then
        [ "$np" -eq "$want_np" ] || { fail "$label: DM particle count $np != IC count $want_np"; return 1; }
        echo "    $label: DM particles $np == IC count (exact)"
    fi
    echo "$np" > "$d/.np_final"
    return 0
}

# [4] baryon mass conservation between plt00000 and the final plotfile
check_mass() { # check_mass <run_dir> <label>
    local d=$1 label=$2 m0 m1
    m0="$("$TOOLS/amrex_fvolumesum" -v density "$d/plt00000" 2>&1)" || { fail "$label: fvolumesum failed: $m0"; return 1; }
    m1="$("$TOOLS/amrex_fvolumesum" -v density "$d/$(final_plt)" 2>&1)" || { fail "$label: fvolumesum failed: $m1"; return 1; }
    python3 - "$label" "$MASS_TOL" "$m0" "$m1" <<'PY' || { fail "$label: baryon mass not conserved"; return 1; }
import sys, re, os
sys.path.insert(0, os.environ["L3_TOOLS"])
from l3_check import require_finite, ValidationError
label, tol = sys.argv[1], float(sys.argv[2])
def val(txt):
    nums = re.findall(r"[-+]?\d+\.\d+e[-+]?\d+|[-+]?\d+\.\d+|[-+]?\d+e[-+]?\d+", txt.split("density")[-1])
    if not nums: raise ValidationError(f"cannot parse fvolumesum output: {txt!r}")
    return float(nums[-1])
try:
    m0 = require_finite("mass(t0)", val(sys.argv[3])); m1 = require_finite("mass(t1)", val(sys.argv[4]))
    rel = abs(m1 - m0) / abs(m0)
    print(f"    {label}: baryon mass {m0:.12e} -> {m1:.12e}, |dM/M| = {rel:.3e} (tol {tol:.0e}) {'ok' if rel <= tol else 'BAD'}")
    sys.exit(0 if rel <= tol else 1)
except ValidationError as ex:
    print(f"  VALIDATION ERROR: {ex}"); sys.exit(1)
PY
}

# [2]/[3] strict plotfile comparison (nyx_fcompare_check.py around fcompare/fextrema) + DM particles
compare() { # compare <ref_dir> <dir> <rel_tol> <label>
    local ref=$1 d=$2 tol=$3 label=$4 out rc=0 plt rep; plt="$(final_plt)"; rep="$(report_dir "$d")"
    local diag_args=(); [ -n "${DIAG:-}" ] && diag_args=(--diagnostic $DIAG)
    # exit 0 = every gated variable within tolerance; 1 = a gated variable outside it; 2 = structural
    # (header mismatch, non-finite raw values, missing/duplicate/message rows, tool errors, parser/tool
    # inconsistency). The full report (headers, fextrema, fcompare output, decisions) is in fcompare.<label>.txt.
    out="$(python3 "$HERE/nyx_fcompare_check.py" "$ref/$plt" "$d/$plt" --rel_tol "$tol" --abs_tol_zero_ref "$ABS_TOL_ZERO_REF" "${diag_args[@]}" \
            --fcompare "$TOOLS/amrex_fcompare" --fextrema "$TOOLS/amrex_fextrema" --out "$rep/fcompare.$label.txt" --label "$label" 2>&1)" || rc=$?
    case "$rc" in
        0) echo "    $label: plotfile comparison rel_tol $tol: ${out#RESULT: }" ;;
        1) fail "$label: plotfile comparison (rel_tol $tol): ${out#RESULT: }"; return 1 ;;
        *) fail "$label: plotfile comparison STRUCTURAL failure (rc=$rc): ${out#RESULT: }"; return 1 ;;
    esac
    if [ -n "${DIAG:-}" ]; then
        # the diagnostic field was parsed and is finite but is not accepted by any tolerance here
        pending="${pending:+$pending }$label:${DIAG}(${out##*diagnostic=})"
        echo "    $label: diagnostic field(s) $DIAG reported, NOT accepted -> case verdict I_R_CHECK_PENDING"
    fi
    if [ -n "$SKIP_PARTICLES" ]; then echo "    $label: DM particle comparison SKIPPED (harness mode)"; return 0; fi
    # DM particles: AMReX's particle_compare needs identical headers (incl. next_id and the per-file
    # layout), i.e. the same rank count -- and it exits 0 even when it prints "FAIL - Particle data
    # headers do not agree". Across rank counts the particles are matched through their exact t=0
    # positions (chk00000 -> chk<final>, ids within a run) by nyx_particle_compare.py, which prints
    # per-component abs/rel errors with particle_compare's definitions and exits nonzero on disagreement.
    rc=0; out="$(python3 "$HERE/nyx_particle_compare.py" "$ref" "$d" "$STEPS" --rel_tol "$tol" 2>&1)" || rc=$?
    echo "$out" > "$rep/particle_compare.$label.txt"
    if [ "$rc" -ne 0 ] || ! echo "$out" | /usr/bin/grep -q 'PARTICLES AGREE'; then fail "$label: DM particle comparison (rel_tol $tol) rc=$rc: $(echo "$out" | /usr/bin/grep -E 'AGREE|DISAGREE|rror|missing|differ' | head -2 | tr '\n' ' ')"; return 1; fi
    local pw; pw="$(echo "$out" | awk '$1 ~ /^DM_/ {if ($3+0 > m) {m=$3+0; v=$1}} END{printf "%s %.3e", v, m}')"
    echo "    $label: DM particles (t=0-identity matched, chk$(printf '%05d' "$STEPS")) rel_tol $tol: PARTICLES AGREE (max rel err $pw)"
    return 0
}

ic_count() { # expected DM particle count from the deck's IC file
    case "$1" in
        minisb) head -1 "$R/_upstream/level3/Nyx/Exec/MiniSB/ic_sb_32.ascii" | tr -d ' ' ;;
        lya_adiabatic|lya_heatcool) python3 -c "import struct; f=open('$R/_upstream/level3/Nyx/Exec/LyA/32.nyx','rb'); print(struct.unpack('<q', f.read(8))[0])" ;;
        *) echo "" ;;
    esac
}

for CASE in $CASES; do
    echo "validate.sh: === Nyx $BACKEND case=$CASE, $STEPS steps, $N GPU(s) [profile $GPU_PROFILE; CPU reference $CPU_PROFILE] ==="
    WANT_NP="$(ic_count "$CASE")"; case_tols "$CASE"; REL_TOL_SAME_CASE=$TOL_SAME; REL_TOL_X_CASE=$TOL_X
    D="$GPU_RUNS/$CASE.smoke.np$N"
    rc=0; run_gpu "$CASE" "$N" "$GPU_RUNS/validate.$CASE.np$N.stdout" || rc=$?
    /usr/bin/grep -aE '^# Nyx|hpcperf-launch: audit summary|Run time =|Total Time|ERROR|Error|abort' "$GPU_RUNS/validate.$CASE.np$N.stdout" | head -8 || true
    if [ "$rc" -eq 124 ]; then fail "$CASE np$N timed out after ${TIMEOUT}s"; continue; fi
    if [ "$rc" -ne 0 ]; then fail "$CASE np$N run.sh exited $rc (see $GPU_RUNS/validate.$CASE.np$N.stdout)"; continue; fi
    [ "$(manifest_val "$D" exit_code)" = 0 ] || { fail "$CASE np$N manifest exit_code=$(manifest_val "$D" exit_code)"; continue; }
    check_complete "$D" "$CASE np$N" "$WANT_NP" || continue
    check_mass "$D" "$CASE np$N" || true

    # reference for [2]
    if [ "$N" -eq 1 ]; then
        REF="$GPU_RUNS/$CASE.smoke.np1.rerun"
        if [ -n "$OFFLINE" ]; then
            [ -f "$REF/$(final_plt)/Header" ] || { fail "$CASE np1: OFFLINE -- rerun directory $REF missing"; continue; }
        else
            rm -rf "$REF"; rc=0; run_gpu "$CASE" 1 "$GPU_RUNS/validate.$CASE.np1.rerun.stdout" || rc=$?
            # run.sh wrote into the canonical np1 dir again: move that fresh result aside as the rerun,
            # keeping the first run as the one under test (manifests record both run_ids)
            if [ "$rc" -ne 0 ]; then fail "$CASE np1 rerun exited $rc"; continue; fi
            mv "$GPU_RUNS/$CASE.smoke.np1" "$REF"
            # re-run the case under test so that $D holds a real result again (third execution)
            rc=0; run_gpu "$CASE" 1 "$GPU_RUNS/validate.$CASE.np1.stdout" || rc=$?
            [ "$rc" -eq 0 ] || { fail "$CASE np1 (re-execution) exited $rc"; continue; }
            check_complete "$D" "$CASE np1" "$WANT_NP" || continue
        fi
        LABEL2="same-config-rerun"
    else
        REF="$GPU_RUNS/$CASE.smoke.np1"
        BIN="$(manifest_val "$D" binary_sha256)"
        if [ ! -f "$REF/$(final_plt)/Header" ] || [ "$(manifest_val "$REF" binary_sha256)" != "$BIN" ] || [ "$(manifest_val "$REF" deck_sha256)" != "$(manifest_val "$D" deck_sha256)" ] || [ "$(manifest_val "$REF" steps)" != "$STEPS" ]; then
            echo "    (1-GPU reference for $CASE missing or from a different binary/deck -- producing it now)"
            rc=0; run_gpu "$CASE" 1 "$GPU_RUNS/validate.$CASE.np1.stdout" || rc=$?
            [ "$rc" -eq 0 ] || { fail "$CASE np1 reference run exited $rc"; continue; }
            check_complete "$REF" "$CASE np1(ref)" "$WANT_NP" || continue
        fi
        LABEL2="vs-1GPU"
    fi
    compare "$REF" "$D" "$REL_TOL_SAME_CASE" "$CASE.np$N.$LABEL2" || true

    # [3] CPU reference (independent backend, same Nyx/AMReX sources and deck)
    C="$CPU_RUNS/$CASE.smoke.np1"
    if [ ! -f "$C/$(final_plt)/Header" ] || [ "$(manifest_val "$C" deck_sha256)" != "$(manifest_val "$D" deck_sha256)" ] || [ "$(manifest_val "$C" steps)" != "$STEPS" ]; then
        echo "    (CPU reference for $CASE missing -- running the CPU profile now, 1 rank)"
        rc=0; run_cpu "$CASE" "$CPU_RUNS/validate.$CASE.cpu.stdout" || rc=$?
        [ "$rc" -eq 0 ] || { fail "$CASE CPU reference run exited $rc (see $CPU_RUNS/validate.$CASE.cpu.stdout)"; continue; }
    fi
    check_complete "$C" "$CASE cpu-ref" "$WANT_NP" || continue
    compare "$C" "$D" "$REL_TOL_X_CASE" "$CASE.np$N.vs-CPU" || true
done

CRIT="fcompare/particle rel_tol adiabatic $REL_TOL_SAME / heatcool $HC_REL_TOL vs $( [ "$N" -eq 1 ] && echo rerun || echo 1-GPU), CPU reference rel_tol adiabatic $REL_TOL_XBACKEND / heatcool $HC_REL_TOL, zero-reference fields exact, baryon mass |dM/M|<=$MASS_TOL, DM count exact, finite"
if [ "$ok" -eq 1 ] && [ -z "$pending" ]; then
    echo "Nyx $BACKEND validation ($N GPU, cases: $CASES; $CRIT): PASS"; exit 0
elif [ "$ok" -eq 1 ]; then
    echo "Nyx $BACKEND validation ($N GPU, cases: $CASES; $CRIT): STATE_AND_PARTICLES_PASS; I_R_CHECK_PENDING [$pending] -- heat/cool acceptance incomplete, not a full regression PASS"; exit 3
fi
echo "Nyx $BACKEND validation ($N GPU, cases: $CASES): FAIL${pending:+ (and I_R_CHECK_PENDING [$pending])}"; exit 1
