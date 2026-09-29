#!/usr/bin/env python3
"""Branson log checks (level2/branson): the per-step energy balances Branson prints, and the
GPU-vs-CPU-reference comparison. This is the parser/checker validate.sh uses (mode gpu = its
check B, mode cmp = its check C), usable on the log of any deck.

  check_log.py gpu <gpu.log> [--steps N]
      every completed step must print a complete conservation block and satisfy
        |Radiation conservation| <= 1e-9 * (Emission E + Source E + Pre census E)
        |Material conservation|  <= 1e-9 * Pre mat E
      (Branson's own diagnostics; observed 1e-13 .. 1e-15 relative), GPU transport must have
      been used in every step ("cell(s) to the GPU", no "GPU kernel not available" fallback)
      and the run must have finished (final "Photons Per Second (FOM)" line). --steps N requires
      exactly N steps (validate.sh: 5); without it at least one step.
      This is a consistency check of the transport, not a comparison with a reference: it
      does not verify the transported photon count against an independent result.
  check_log.py cmp <gpu.log> <cpu.log>
      the final step's Post mat E, Absorption E and Exit E must agree to 5 % relative and every
      cell's T_e to 0.02 absolute between the GPU run and a CPU-only Branson built from the
      same sources (validate.sh: > 6 sigma of the seed-to-seed scatter, yet catches a broken
      transport kernel); the "Total Photons transported" count of the two runs must agree to
      5 % relative as well (the same deck and seed source the same photons; the count is a
      tally of the transport, compared with the same margin as the energies).
Prints the per-step / per-quantity figures and "   ERROR: ..." lines, then
"PASS: branson log check ..." / "FAIL: branson log check ..."; exit 0/1.
"""
import re, sys

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


def main(argv):
    if len(argv) < 3 or argv[1] not in ('gpu', 'cmp'):
        print(__doc__); return 2
    mode, logs = argv[1], [a for a in argv[2:] if not a.startswith('--')]
    want_steps = None
    if '--steps' in argv:
        want_steps = int(argv[argv.index('--steps') + 1])
        logs = [a for a in logs if a != str(want_steps)]
    errors = []
    gpu_steps, gpu_text = parse(logs[0])
    if mode == 'gpu':
        if want_steps is not None and len(gpu_steps) != want_steps:
            errors.append(f'expected {want_steps} time steps, found {len(gpu_steps)}')
        if not gpu_steps:
            errors.append('no "Step:" block in the log')
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
        what = f'{len(gpu_steps)} steps: rad/mat conservation <= 1e-9 rel, GPU transport used, run finished'
    else:  # compare final step of gpu vs cpu
        if len(logs) < 2:
            print('FAIL: branson log check: cmp needs <gpu.log> <cpu.log>'); return 2
        cpu_steps, cpu_text = parse(logs[1])
        if len(cpu_steps) != len(gpu_steps) or not gpu_steps:
            errors.append(f'step count differs: GPU {len(gpu_steps)} vs CPU {len(cpu_steps)}')
        else:
            g, c = gpu_steps[-1], cpu_steps[-1]
            for key in ('PostMat', 'Absorption', 'Exit'):
                if key not in g or key not in c:
                    errors.append(f'final step: {key} E missing in a log'); continue
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
        gp = re.search(r'Total Photons transported: (\d+)', gpu_text)
        cp = re.search(r'Total Photons transported: (\d+)', cpu_text)
        if not gp or not cp:
            errors.append('"Total Photons transported" line missing in a log')
        else:
            g_n, c_n = int(gp.group(1)), int(cp.group(1))
            rel = abs(g_n - c_n) / c_n if c_n else float('inf')
            print(f'   total photons transported: GPU {g_n}  CPU {c_n}  rel diff {rel:.3e}')
            if rel > 0.05:
                errors.append(f'transported photon count differs by {rel:.3e} (> 5e-2)')
        what = 'final Post-mat/Absorption/Exit E within 5 %, T_e within 0.02 and transported photons within 5 % of the CPU reference'
    for e in errors:
        print('   ERROR:', e)
    print(('FAIL' if errors else 'PASS') + f': branson log check ({mode}): {what}')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
