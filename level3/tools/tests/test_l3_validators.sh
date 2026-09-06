#!/bin/bash
# Negative/positive tests for the second-batch application checkers (CPU only, no GPU, no build):
# each checker must FAIL on a truncated run, a NaN/Inf, a non-converged/aborted solve, a missing GPU
# banner and a nonzero exit code, and PASS on the genuine output it was written for. The genuine
# outputs are taken from real runs under build/level3/<app>/<profile>/run when present; the CP2K
# case falls back to a synthetic minimal output so the harness runs everywhere.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../../.." && pwd)"
export L3_TOOLS="$R/level3/tools"
pass=0; failn=0
ok()  { echo "ok   $*"; pass=$((pass+1)); }
bad() { echo "FAIL $*"; failn=$((failn+1)); }
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

# ---------------------------------------------------------------- CP2K: cp2k_md_summary.py --check
CP2K_CHK="$R/level3/cp2k/cp2k_md_summary.py"
synth_cp2k() { # synth_cp2k <out> : minimal output with the features the checker gates on
    local out=$1; {
    echo " CP2K|            m dbcsr_acc spglib offload_cuda"
    echo " GLOBAL| Run type                                                             MD"
    echo " DBCSR| ACC: Number of devices/node                                            1"
    echo "  Leaving inner SCF loop after reaching    50 steps."
    echo " *** WARNING in qs_scf.F:700 :: SCF run NOT converged ***"
    echo " ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]          -1101.031005888854452"
    echo " MD_INI| MD initialization"
    for i in $(seq 1 10); do
        echo "  *** SCF run converged in     5 steps ***"
        echo " ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]          -1101.0392895677$i"
        echo " MD| Step number                                                         $i"
        echo " MD| Potential energy [hartree]         -0.110103928957E+04  -0.110103928957E+04"
        echo " MD| Conserved quantity [hartree]                            -0.110076554067E+04"
    done
    echo " GRID| ..."
    echo " 1     collocate ortho    GPU                               15972172      21.34%"
    echo " 1     integrate ortho    GPU                               15311000      20.46%"
    echo "  **** **** ******  **  PROGRAM ENDED AT                2026-09-05 21:00:00.000"; } > "$out"
}
INP="$TMP/input.inp"; printf '&FORCE_EVAL\n  &DFT\n    &SCF\n      IGNORE_CONVERGENCE_FAILURE\n    &END SCF\n  &END DFT\n&END FORCE_EVAL\n' > "$INP"
INP_STRICT="$TMP/input_strict.inp"; printf '&FORCE_EVAL\n  &DFT\n    &SCF\n      MAX_SCF 50\n    &END SCF\n  &END DFT\n&END FORCE_EVAL\n' > "$INP_STRICT"
GOOD="$TMP/good.out"
real="$(ls -d "$R"/build/level3/cp2k/*/run/h2o64.smoke.np1.t*/cp2k.out 2>/dev/null | head -1)"
if [ -n "$real" ] && python3 "$CP2K_CHK" "$real" "$(dirname "$real")/input.inp" --check >/dev/null 2>&1; then cp "$real" "$GOOD"; cp "$(dirname "$real")/input.inp" "$INP"; echo "# cp2k: using real output $real"; else synth_cp2k "$GOOD"; echo "# cp2k: using synthetic output"; fi
cp2k_check() { python3 "$CP2K_CHK" "$1" "${2:-$INP}" --check >/dev/null 2>&1; }
cp2k_check "$GOOD" && ok "cp2k 1: genuine complete output accepted" || bad "cp2k 1: genuine output rejected"
head -n $(( $(wc -l < "$GOOD") - 5 )) "$GOOD" | /usr/bin/grep -v 'PROGRAM ENDED' > "$TMP/trunc.out"
cp2k_check "$TMP/trunc.out" && bad "cp2k 2: truncated output (no PROGRAM ENDED) accepted" || ok "cp2k 2: truncated output rejected"
sed '0,/ENERGY| Total FORCE_EVAL/s/ENERGY| Total FORCE_EVAL ( QS ) energy \[hartree\] *-[0-9.]*/ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]          NaN/' "$GOOD" > "$TMP/nan.out"
cp2k_check "$TMP/nan.out" && bad "cp2k 3: NaN energy accepted" || ok "cp2k 3: NaN energy rejected"
awk '/MD_INI\| MD initialization/{ini=1} ini && /SCF run converged in/ && !done {sub(/SCF run converged in +[0-9]+ steps/, "SCF run NOT converged"); done=1} {print}' "$GOOD" > "$TMP/nonconv.out"
cp2k_check "$TMP/nonconv.out" && bad "cp2k 4: non-converged MD-step SCF accepted" || ok "cp2k 4: non-converged MD-step SCF rejected"
cp2k_check "$GOOD" "$INP_STRICT" && bad "cp2k 5: non-converged initial SCF accepted although the deck lacks IGNORE_CONVERGENCE_FAILURE" || ok "cp2k 5: non-converged initial SCF rejected without IGNORE_CONVERGENCE_FAILURE"
sed 's/DBCSR| ACC: Number of devices\/node *[0-9]*/DBCSR| ACC: Number of devices\/node                                            0/' "$GOOD" > "$TMP/nogpu.out"
cp2k_check "$TMP/nogpu.out" && bad "cp2k 6: zero accelerator devices accepted" || ok "cp2k 6: zero accelerator devices rejected"
sed 's/offload_cuda//' "$GOOD" > "$TMP/noflag.out"
cp2k_check "$TMP/noflag.out" && bad "cp2k 7: missing offload_cuda cp2kflag accepted" || ok "cp2k 7: missing offload_cuda cp2kflag rejected"
sed 's/ortho    GPU /ortho    CPU /' "$GOOD" > "$TMP/gridcpu.out"
cp2k_check "$TMP/gridcpu.out" && bad "cp2k 8: GRID tasks all on CPU accepted" || ok "cp2k 8: GRID tasks all on CPU rejected"
: > "$TMP/empty.out"; cp2k_check "$TMP/empty.out" && bad "cp2k 9: empty output accepted" || ok "cp2k 9: empty output rejected"

# ---------------------------------------------------------------- QMCPACK: qmc_check.py
QMC_CHK="$R/level3/qmcpack/qmc_check.py"; CS="$R/_upstream/level3/qmcpack/tests/scripts/check_scalars.py"
realq="$(ls -d "$R"/build/level3/qmcpack/*/run/diamond2.smoke.np1.t*/qmc.out 2>/dev/null | head -1)"
if [ -n "$realq" ] && [ -f "$CS" ]; then
    RD="$(dirname "$realq")"; prefix="$(/usr/bin/grep -m1 '^prefix=' "$RD/run_manifest.txt" | cut -d= -f2)"
    qmc_check() { python3 "$QMC_CHK" "$1" "$prefix" "-21.844975 0.02" 3 2 "$CS" >/dev/null 2>&1; }
    mk() { rm -rf "$TMP/q"; cp -r "$RD" "$TMP/q"; }
    mk; qmc_check "$TMP/q" && ok "qmc 1: genuine run accepted" || bad "qmc 1: genuine run rejected"
    mk; sed -i '/QMCPACK execution completed successfully/d' "$TMP/q/qmc.out"; qmc_check "$TMP/q" && bad "qmc 2: incomplete run accepted" || ok "qmc 2: incomplete run rejected"
    mk; sed -i 's/^exit_code=0$/exit_code=1/' "$TMP/q/run_manifest.txt"; qmc_check "$TMP/q" && bad "qmc 3: nonzero exit code accepted" || ok "qmc 3: nonzero exit code rejected"
    mk; f="$TMP/q/$prefix.s001.scalar.dat"; awk '!/^#/ && ++n==2 {$2="nan"} {print}' "$f" > "$f.tmp" && mv "$f.tmp" "$f"; qmc_check "$TMP/q" && bad "qmc 4: NaN in DMC scalars accepted" || ok "qmc 4: NaN in DMC scalars rejected"
    mk; f="$TMP/q/$prefix.s001.scalar.dat"; head -n 12 "$f" > "$f.tmp" && mv "$f.tmp" "$f"; qmc_check "$TMP/q" && bad "qmc 5: truncated DMC scalar file accepted" || ok "qmc 5: truncated DMC scalar file rejected"
    mk; sed -i '/OpenMP target offload to accelerators build option is enabled/d' "$TMP/q/qmc.out"; qmc_check "$TMP/q" && bad "qmc 6: missing offload banner accepted" || ok "qmc 6: missing offload banner rejected"
    mk; f="$TMP/q/$prefix.s001.scalar.dat"; awk '!/^#/ && NF>2 {$2=$2+1.0} {print}' "$f" > "$f.tmp" && mv "$f.tmp" "$f"; qmc_check "$TMP/q" && bad "qmc 7: energies shifted by +1 Ha accepted" || ok "qmc 7: energies shifted by +1 Ha rejected (statistical test)"
    mk; echo "QMCPACK ERROR synthetic" >> "$TMP/q/qmc.out"; qmc_check "$TMP/q" && bad "qmc 8: 'QMCPACK ERROR' accepted" || ok "qmc 8: 'QMCPACK ERROR' rejected"
else
    echo "# qmcpack: no genuine run under build/level3/qmcpack/*/run yet -- qmc_check.py tests skipped"
fi

# ---------------------------------------------------------------- DFT-FE: dftfe_check.py
DFT_CHK="$R/level3/dftfe/dftfe_check.py"
reald="$(ls -d "$R"/build/level3/dftfe/*/run/al_md.smoke.np1*/dftfe.out 2>/dev/null | head -1)"
if [ -f "$DFT_CHK" ] && [ -n "$reald" ]; then
    RD="$(dirname "$reald")"; REFD="$R/_upstream/level3/dftfe/testsGPU/pseudopotential/real/accuracyBenchmarks/output_MD_0"
    dft_check() { python3 "$DFT_CHK" "$1/dftfe.out" "$REFD" --check >/dev/null 2>&1; }
    mk() { rm -rf "$TMP/d"; cp -r "$RD" "$TMP/d"; }
    mk; dft_check "$TMP/d" && ok "dftfe 1: genuine run accepted" || bad "dftfe 1: genuine run rejected"
    mk; sed -i '/MD run completed successfully/d' "$TMP/d/dftfe.out"; dft_check "$TMP/d" && bad "dftfe 2: incomplete MD accepted" || ok "dftfe 2: incomplete MD rejected"
    mk; sed -i '0,/Total Energy in Ha at timeIndex/s/\(Total Energy in Ha at timeIndex\) *-[0-9.]*/\1 nan/' "$TMP/d/dftfe.out"; dft_check "$TMP/d" && bad "dftfe 3: NaN energy accepted" || ok "dftfe 3: NaN energy rejected"
    mk; sed -i '0,/Total Energy in Ha at timeIndex/s/\(Total Energy in Ha at timeIndex\) *\(-[0-9]*\)\./\1 \2.9/' "$TMP/d/dftfe.out"; dft_check "$TMP/d" && bad "dftfe 4: energy off by ~0.5 Ha accepted" || ok "dftfe 4: energy off by ~0.5 Ha rejected"
else
    echo "# dftfe: checker or genuine run not present yet -- tests skipped"
fi

echo "== $pass ok, $failn FAIL =="; [ "$failn" -eq 0 ]
