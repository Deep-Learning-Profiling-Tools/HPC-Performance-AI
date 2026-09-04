#!/usr/bin/env bash
# Validate the GPU Branson build.
#
#   ./validate.sh [CUDA|HIP]        (default: CUDA)
#
# Three checks, all logged under $R/build/level2/branson/<cuda|hip>/:
#
#  (A) upstream unit tests: `ctest -E test_input_1pe` in the GPU build tree
#      (11 tests; 1- and 2-rank MPI tests plus the GPU warp-reduction test).
#      test_input_1pe is excluded because upstream's test expects
#      dd_batch_size/event_batch_size = 10000 while its own simple_input.xml
#      says 1000/777 -- an upstream test/data mismatch, unrelated to this port.
#      (validate_ctest.log)
#
#  (B) physics run on the GPU: inputs/marshak_wave_replicated.xml, 5 steps
#      (--t-stop 0.05, --seed 1234, 50k photons/step, 25 cells). Requires
#      exit code 0, exactly 5 completed steps, GPU transport actually used
#      ("Transferring ... cell(s) to the GPU" every step, no "GPU kernel not
#      available" fallback), and per step Branson's own energy balances
#         |Radiation conservation| <= 1e-9 * (Emission E + Source E + Pre census E)
#         |Material  conservation| <= 1e-9 * Pre mat E
#      (observed: ~1e-13 .. 1e-15 relative).  (validate_marshak_gpu.log)
#
#  (C) GPU vs CPU cross-check: the same deck and seed is run with a CPU-only
#      Branson (USE_GPU off) configured from the same sources into
#      $R/build/level2/branson/cpu_ref (built on first use, ~10 s). The
#      final step's Post mat E, Absorption E and Exit E must agree to 5 %
#      relative and every cell's T_e to 0.02 absolute. The two binaries use
#      different photon orderings, so the results agree only statistically;
#      the seed-to-seed scatter of these quantities is 0.2-0.8 % (energies)
#      and <= 0.005 (T_e), so 5 % / 0.02 is > 6 sigma yet catches a broken
#      transport kernel.  (validate_marshak_cpu.log)
#
# Prints "PASS: branson <BACKEND> ..." / "FAIL: branson <BACKEND> ..." and
# exits 0/1.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"

BACKEND="$(printf '%s' "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
case "$BACKEND" in
    CUDA|HIP) ;;
    *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac

if [ "${HPC_PERFORMANCE_AI_ROOT:-}" != "$R" ] && [ -f "$R/hpcperf_env.sh" ]; then
    # conda's activate hooks are not `set -u`-clean
    set +u
    # shellcheck disable=SC1091
    source "$R/hpcperf_env.sh" 2>/dev/null
    set -u
fi

BUILD="$R/build/level2/branson/$(printf '%s' "$BACKEND" | tr '[:upper:]' '[:lower:]')"
EXE="$BUILD/BRANSON"
CPU_BUILD="$R/build/level2/branson/cpu_ref"
CPU_EXE="$CPU_BUILD/BRANSON"
DECK="$HERE/inputs/marshak_wave_replicated.xml"
DECK_ARGS=( --t-stop 0.05 --seed 1234 )
JOBS="${MAKE_JOBS:-4}"

fail() { echo "FAIL: branson $BACKEND -- $*"; exit 1; }

[ -x "$EXE" ] || fail "$EXE not found; run $HERE/build.sh $BACKEND first"
command -v mpirun >/dev/null 2>&1 || fail "mpirun not on PATH (source $R/hpcperf_env.sh)"
command -v python3 >/dev/null 2>&1 || fail "python3 not on PATH"

# ---------------------------------------------------------------- (A) ctest
echo "== (A) upstream ctest in $BUILD (test_input_1pe excluded, see header)"
if ! ctest --test-dir "$BUILD" -E test_input_1pe --output-on-failure > "$BUILD/validate_ctest.log" 2>&1; then
    tail -30 "$BUILD/validate_ctest.log"
    fail "ctest failed (log: $BUILD/validate_ctest.log)"
fi
CTEST_LINE="$(grep -E '^[0-9]+% tests passed' "$BUILD/validate_ctest.log" | tail -1)"
echo "   $CTEST_LINE"
case "$CTEST_LINE" in
    "100% tests passed, 0 tests failed out of "*) ;;
    *) fail "unexpected ctest summary: '$CTEST_LINE'" ;;
esac
NTESTS="${CTEST_LINE##*out of }"

# ------------------------------------------------ (B) GPU physics run + parser
echo "== (B) GPU run: mpirun -np 1 $EXE $DECK ${DECK_ARGS[*]}"
if ! mpirun -np 1 "$EXE" "$DECK" "${DECK_ARGS[@]}" > "$BUILD/validate_marshak_gpu.log" 2>&1; then
    tail -20 "$BUILD/validate_marshak_gpu.log"
    fail "GPU run exited non-zero (log: $BUILD/validate_marshak_gpu.log)"
fi

# Python helper: parses a Branson log into per-step records and does the
# checks. Usage: python3 - <mode> <gpu.log> [cpu.log]
check_py() {
python3 - "$@" <<'PY'
import re, sys
mode, logs = sys.argv[1], sys.argv[2:]
NUM = r'([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)'

def parse(path):
    steps, cur = [], None
    text = open(path, errors='replace').read()
    for line in text.splitlines():
        if line.startswith('Step:'):
            cur = {'Te': [], 'gpu': False}
            steps.append(cur)
            continue
        if cur is None:
            continue
        if 'cell(s) to the GPU' in line:
            cur['gpu'] = True
        m = re.match(r'\s*(\d+)\s+' + NUM + r'\s+' + NUM + r'\s+' + NUM + r'\s*$', line)
        if m:
            cur['Te'].append(float(m.group(2)))
        for key, pat in (('Emission', r'Emission E: ' + NUM), ('Source', r'Source E: ' + NUM),
                         ('Absorption', r'Absorption E: ' + NUM), ('Exit', r'Exit E: ' + NUM),
                         ('PreCensus', r'Pre census E: ' + NUM), ('PreMat', r'Pre mat E: ' + NUM),
                         ('PostMat', r'Post mat E: ' + NUM),
                         ('RadCons', r'Radiation conservation: ' + NUM),
                         ('MatCons', r'Material conservation: ' + NUM)):
            m = re.search(pat, line)
            if m:
                cur[key] = float(m.group(1))
    return steps, text

errors = []
gpu_steps, gpu_text = parse(logs[0])
if mode == 'gpu':
    if len(gpu_steps) != 5:
        errors.append(f'expected 5 time steps, found {len(gpu_steps)}')
    if 'GPU kernel not available' in gpu_text:
        errors.append('transport fell back to the CPU ("GPU kernel not available")')
    if 'Photons Per Second (FOM)' not in gpu_text:
        errors.append('no final "Photons Per Second (FOM)" line -- run did not finish')
    for i, s in enumerate(gpu_steps, 1):
        need = ('Emission', 'Source', 'PreCensus', 'PreMat', 'RadCons', 'MatCons')
        if any(k not in s for k in need):
            errors.append(f'step {i}: conservation block incomplete'); continue
        if not s['gpu']:
            errors.append(f'step {i}: no "cell(s) to the GPU" transfer -> GPU transport not used')
        rad_scale = s['Emission'] + s['Source'] + s['PreCensus']
        rad_rel = abs(s['RadCons']) / rad_scale
        mat_rel = abs(s['MatCons']) / s['PreMat']
        print(f'   step {i}: |rad cons| = {abs(s["RadCons"]):.3e} ({rad_rel:.2e} rel), '
              f'|mat cons| = {abs(s["MatCons"]):.3e} ({mat_rel:.2e} rel)')
        if rad_rel > 1e-9:
            errors.append(f'step {i}: radiation conservation {rad_rel:.3e} rel > 1e-9')
        if mat_rel > 1e-9:
            errors.append(f'step {i}: material conservation {mat_rel:.3e} rel > 1e-9')
else:  # compare final step of gpu vs cpu
    cpu_steps, _ = parse(logs[1])
    if len(cpu_steps) != len(gpu_steps) or not gpu_steps:
        errors.append(f'step count differs: GPU {len(gpu_steps)} vs CPU {len(cpu_steps)}')
    else:
        g, c = gpu_steps[-1], cpu_steps[-1]
        for key in ('PostMat', 'Absorption', 'Exit'):
            rel = abs(g[key] - c[key]) / abs(c[key])
            print(f'   final {key:<10s} E: GPU {g[key]:.6e}  CPU {c[key]:.6e}  rel diff {rel:.3e}')
            if rel > 0.05:
                errors.append(f'{key} E differs by {rel:.3e} (> 5e-2)')
        if len(g['Te']) != len(c['Te']) or not g['Te']:
            errors.append(f'T_e cell count differs: GPU {len(g["Te"])} vs CPU {len(c["Te"])}')
        else:
            dmax = max(abs(a - b) for a, b in zip(g['Te'], c['Te']))
            imax = max(range(len(g['Te'])), key=lambda i: abs(g['Te'][i] - c['Te'][i]))
            print(f'   final T_e: {len(g["Te"])} cells, max |GPU-CPU| = {dmax:.4f} at cell {imax} '
                  f'(GPU {g["Te"][imax]:.5f}, CPU {c["Te"][imax]:.5f}); '
                  f'front cells GPU {[round(x,4) for x in g["Te"][:3]]} CPU {[round(x,4) for x in c["Te"][:3]]}')
            if dmax > 0.02:
                errors.append(f'T_e differs by {dmax:.4f} (> 0.02) at cell {imax}')
for e in errors:
    print('   ERROR:', e)
sys.exit(1 if errors else 0)
PY
}

check_py gpu "$BUILD/validate_marshak_gpu.log" || fail "GPU physics checks failed (log: $BUILD/validate_marshak_gpu.log)"
echo "   $(grep 'Total transport:' "$BUILD/validate_marshak_gpu.log" | tail -1), $(grep 'FOM' "$BUILD/validate_marshak_gpu.log" | tail -1)"

# ------------------------------------------------ (C) CPU reference cross-check
if [ ! -x "$CPU_EXE" ]; then
    echo "== (C) building CPU-only reference Branson into $CPU_BUILD"
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
echo "== (C) CPU reference run: mpirun -np 1 $CPU_EXE $DECK ${DECK_ARGS[*]}"
if ! mpirun -np 1 "$CPU_EXE" "$DECK" "${DECK_ARGS[@]}" > "$BUILD/validate_marshak_cpu.log" 2>&1; then
    tail -20 "$BUILD/validate_marshak_cpu.log"
    fail "CPU reference run exited non-zero (log: $BUILD/validate_marshak_cpu.log)"
fi
check_py cmp "$BUILD/validate_marshak_gpu.log" "$BUILD/validate_marshak_cpu.log" \
    || fail "GPU vs CPU comparison failed (logs: $BUILD/validate_marshak_{gpu,cpu}.log)"

echo "PASS: branson $BACKEND (ctest $NTESTS/$NTESTS excl. test_input_1pe; Marshak 5 steps: rad/mat conservation <= 1e-9 rel; GPU vs CPU final Post-mat/Absorption/Exit E within 5%, T_e within 0.02)"
exit 0
