#!/usr/bin/env python3
"""Branson log parser and checks (level2/branson): the per-step energy balances Branson prints, the
GPU-vs-CPU-reference comparison of two logs, and the extraction of a frozen reference from a finished
CPU-only run. This is the parser/checker validate.sh uses (mode gpu = its check B, mode cmp = its
check C); check_reference.py (candidate validation against reference/<id>.json) imports parse().

Parser semantics (one rule for every mode):
  * a "Step:" line opens a step block; every quantity printed until the next "Step:" belongs to it;
  * energies (Emission / Source / Pre census / Pre mat / Post mat / Absorption / Exit E) and the
    conservation lines are per step; the T_e table (rows "cell T_e ...") is per step;
  * "Total Photons transported: N" is per step as well (Branson prints one per step);
  * "final" always means the FINAL COMPLETED STEP (the last "Step:" block): final energies, final
    T_e table and the final step's transported photon count. The first step's count is never the
    compared value (before 2026-10-01 the cmp mode took the first "Total Photons transported" line
    of the log through re.search -- fixed: it now takes the final step's count).

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
      cell's final T_e to 0.02 absolute between the GPU run and a CPU-only Branson built from the
      same sources (validate.sh: > 6 sigma of the seed-to-seed scatter, yet catches a broken
      transport kernel); the final step's "Total Photons transported" count of the two runs must
      agree to 5 % relative as well (the same deck and seed source the same photons; the count is
      a tally of the transport, compared with the same margin as the energies).
  check_log.py freeze --input-id ID --deck DECK --cpu-log LOG --ranks N --out FILE [provenance options]
      construction-time only (generate_reference.sh): extracts the frozen reference
      reference/<id>.json from a FINISHED CPU-only run's log -- deck sha256 and seed, the final step's
      energies, T_e table and transported photon count, the per-step counts, the run's provenance
      (binary sha256, rank count, launch, allocation, start/finish, log sha256 and where the log is
      kept) and the comparison rule with its basis. Refuses a GPU log, an unfinished run and an
      existing file (--force overwrites).
Prints the per-step / per-quantity figures and "   ERROR: ..." lines, then
"PASS: branson log check ..." / "FAIL: branson log check ..."; exit 0/1 (freeze: the file, exit 0/2).
"""
import hashlib, json, os, re, sys, time

NUM = r'([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)'

# the comparison rule of validate.sh check C (reference/<id>.json carries the same numbers; check_reference.py
# refuses a reference whose tolerances differ from these, so the artifact cannot loosen the rule)
ENERGY_REL_TOL = 0.05
PHOTONS_REL_TOL = 0.05
TE_ABS_TOL = 0.02
TOLERANCE_BASIS = ("validate.sh check C: the seed-to-seed scatter of the final energies is 0.2-0.8 % and of the "
                   "front-cell T_e <= 0.006, so 5 % relative / 0.02 absolute is > 6 sigma of the Monte Carlo scatter "
                   "yet far below any transport error; the transported photon count (a tally of the transport) uses "
                   "the energies' margin. Fixed before any optimized candidate existed; the reference's rank count "
                   "does not enter the criteria (same deck, seed and global photon count).")
REFERENCE_SCHEMA = "hpcperf-branson-reference-1"


def parse(path):
    """-> (steps, text): one dict per "Step:" block (energies, conservation, 'Te' rows, 'gpu' transfer seen,
    'Photons' = the step's "Total Photons transported" count) and the whole text."""
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
        m = re.search(r'Total Photons transported: (\d+)', line)
        if m:
            cur['Photons'] = int(m.group(1))
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


def finished(text):
    return 'Photons Per Second (FOM)' in text


def totals(text):
    """The run's 'Total transport' seconds and FOM (photons per second), or None each."""
    t = re.search(r'^Total transport: ' + NUM, text, re.M)
    f = re.search(r'Photons Per Second \(FOM\): ' + NUM, text)
    return (float(t.group(1)) if t else None, float(f.group(1)) if f else None)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def compare_final(g, c, g_label='GPU', c_label='CPU'):
    """Compare two final-step dicts (parse() blocks, or a reference's final_step mapped to the parse keys).
    Prints the per-quantity lines; returns the list of error strings."""
    errors = []
    for key, name in (('PostMat', 'Post mat E'), ('Absorption', 'Absorption E'), ('Exit', 'Exit E')):
        if key not in g or key not in c:
            errors.append(f'final step: {name} missing in a log'); continue
        rel = abs(g[key] - c[key]) / abs(c[key]) if c[key] else (0.0 if g[key] == c[key] else float('inf'))
        print(f'   final {name:<12s}: {g_label} {g[key]:.6e}  {c_label} {c[key]:.6e}  rel diff {rel:.3e} (rule <= {ENERGY_REL_TOL:g})')
        if not rel <= ENERGY_REL_TOL:
            errors.append(f'{name} differs by {rel:.3e} (> {ENERGY_REL_TOL:g})')
    gte, cte = g.get('Te') or [], c.get('Te') or []
    if len(gte) != len(cte):
        errors.append(f'T_e cell count differs: {g_label} {len(gte)} vs {c_label} {len(cte)}')
    elif not gte:
        # the hohlraum decks print no per-cell temperature table (only the marshak deck does):
        # the T_e criterion does not apply; energies and the photon count are compared
        print('   final T_e: no per-cell temperature table in this deck (criterion not applicable)')
    else:
        dmax = max(abs(a - b) for a, b in zip(gte, cte))
        imax = max(range(len(gte)), key=lambda i: abs(gte[i] - cte[i]))
        print(f'   final T_e: {len(gte)} cells, max |{g_label}-{c_label}| = {dmax:.4f} at cell {imax} '
              f'({g_label} {gte[imax]:.5f}, {c_label} {cte[imax]:.5f}); '
              f'front cells {g_label} {[round(x, 4) for x in gte[:3]]} {c_label} {[round(x, 4) for x in cte[:3]]} (rule <= {TE_ABS_TOL:g})')
        if not dmax <= TE_ABS_TOL:
            errors.append(f'T_e differs by {dmax:.4f} (> {TE_ABS_TOL:g}) at cell {imax}')
    if 'Photons' not in g or 'Photons' not in c:
        errors.append('final step: "Total Photons transported" line missing in a log')
    else:
        g_n, c_n = g['Photons'], c['Photons']
        rel = abs(g_n - c_n) / c_n if c_n else (0.0 if g_n == c_n else float('inf'))
        print(f'   final-step photons transported: {g_label} {g_n}  {c_label} {c_n}  rel diff {rel:.3e} (rule <= {PHOTONS_REL_TOL:g})')
        if not rel <= PHOTONS_REL_TOL:
            errors.append(f'final-step transported photon count differs by {rel:.3e} (> {PHOTONS_REL_TOL:g})')
    return errors


def freeze(argv):
    """freeze --input-id ID --deck DECK --cpu-log LOG --ranks N --out FILE [--binary-sha256 S] [--build-options S]
    [--launch S] [--allocation S] [--started UTC] [--finished UTC] [--log-kept-at REL] [--upstream-commit C]
    [--param k=v ...] [--note S] [--generated-by S] [--generated-utc UTC] [--force]"""
    opts, params, i = {}, {}, 0
    while i < len(argv):
        a = argv[i]
        if a == '--force':
            opts['force'] = True; i += 1; continue
        if a == '--param':
            k, _, v = argv[i + 1].partition('='); params[k] = v; i += 2; continue
        if a.startswith('--') and i + 1 < len(argv):
            opts[a[2:].replace('-', '_')] = argv[i + 1]; i += 2; continue
        sys.exit(f'check_log.py freeze: unexpected argument {a!r}')
    for need in ('input_id', 'deck', 'cpu_log', 'ranks', 'out'):
        if need not in opts:
            sys.exit(f'check_log.py freeze: --{need.replace("_", "-")} is required')
    if not re.fullmatch(r'[1-9][0-9]*', opts['ranks']):
        sys.exit('check_log.py freeze: --ranks must be a positive integer')
    out = opts['out']
    if os.path.exists(out) and not opts.get('force'):
        sys.exit(f'check_log.py freeze: {out} exists (construction-time artifact; --force overwrites)')
    deck = opts['deck']
    if not os.path.isfile(deck):
        sys.exit(f'check_log.py freeze: deck {deck} missing')
    deck_text = open(deck, errors='replace').read()
    seed = re.search(r'<seed>\s*(\d+)\s*</seed>', deck_text)
    steps, text = parse(opts['cpu_log'])
    if not finished(text):
        sys.exit('check_log.py freeze: the CPU run did not finish (no "Photons Per Second (FOM)" line) -- not a reference')
    if not steps:
        sys.exit('check_log.py freeze: no "Step:" block in the CPU log')
    if any(s['gpu'] for s in steps):
        sys.exit('check_log.py freeze: the log shows GPU transport ("cell(s) to the GPU"); a reference must come from the CPU-only build')
    fin = steps[-1]
    for key in ('PostMat', 'Absorption', 'Exit', 'Photons'):
        if key not in fin:
            sys.exit(f'check_log.py freeze: the final step lacks {key}')
    if any('Photons' not in s for s in steps):
        sys.exit('check_log.py freeze: a step lacks its "Total Photons transported" line')
    transport_s, fom = totals(text)
    here = os.path.dirname(os.path.abspath(__file__))
    deck_rel = os.path.relpath(os.path.abspath(deck), here)
    ref = {
        'schema': REFERENCE_SCHEMA, 'benchmark': 'branson', 'input_id': opts['input_id'],
        'deck': deck_rel, 'deck_sha256': sha256_file(deck), 'seed': int(seed.group(1)) if seed else None,
        'params': params,
        'upstream': {'repo': 'lanl/branson', 'commit': opts.get('upstream_commit')},
        'reference_run': {
            'implementation': ('CPU-only Branson built from the same sources with -DUSE_GPU=OFF: Branson\'s CPU transport, '
                               'independent of the GPU transport kernel (the deck\'s use_gpu_transporter is ignored by that build)'),
            'binary_sha256': opts.get('binary_sha256'), 'build_options': opts.get('build_options'),
            'mpi_ranks': int(opts['ranks']), 'launch': opts.get('launch'), 'allocation': opts.get('allocation'),
            'started_utc': opts.get('started'), 'finished_utc': opts.get('finished'),
            'total_transport_s': transport_s, 'fom_photons_per_s': fom,
            'log_sha256': sha256_file(opts['cpu_log']), 'log_bytes': os.path.getsize(opts['cpu_log']),
            'log_kept_at': opts.get('log_kept_at')},
        'steps': len(steps),
        'final_step': {'post_mat_e': fin['PostMat'], 'absorption_e': fin['Absorption'], 'exit_e': fin['Exit'],
                       'photons_transported': fin['Photons'], 't_e': fin['Te'] or None},
        'per_step_photons_transported': [s['Photons'] for s in steps],
        'rule': {'post_mat_e': f'rel <= {ENERGY_REL_TOL:g} (final step)', 'absorption_e': f'rel <= {ENERGY_REL_TOL:g} (final step)',
                 'exit_e': f'rel <= {ENERGY_REL_TOL:g} (final step)',
                 'photons_transported': f'rel <= {PHOTONS_REL_TOL:g} (final step, the last "Total Photons transported" line)',
                 't_e': f'abs <= {TE_ABS_TOL:g} per cell of the final step (when the deck prints the table)',
                 'steps': 'the candidate run must complete the same number of steps and print the FOM line'},
        'tolerances': {'energy_rel': ENERGY_REL_TOL, 'photons_rel': PHOTONS_REL_TOL, 't_e_abs': TE_ABS_TOL},
        'tolerance_basis': TOLERANCE_BASIS,
        'generated_utc': opts.get('generated_utc') or time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'generated_by': opts.get('generated_by') or 'level2/branson/check_log.py freeze (construction-time)',
        'note': opts.get('note'),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w') as fh:
        json.dump(ref, fh, indent=1, sort_keys=True)
        fh.write('\n')
    print(f'wrote {out}: {len(steps)} steps, final step Post mat E {fin["PostMat"]:.6e}, Absorption E {fin["Absorption"]:.6e}, '
          f'Exit E {fin["Exit"]:.6e}, photons transported {fin["Photons"]}, T_e cells {len(fin["Te"])}; '
          f'CPU log sha256 {ref["reference_run"]["log_sha256"][:12]}..., {int(opts["ranks"])} rank(s)')
    return 0


def main(argv):
    if len(argv) >= 2 and argv[1] == 'freeze':
        return freeze(argv[2:])
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
        if not finished(gpu_text):
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
    else:  # compare the final completed step of gpu vs cpu
        if len(logs) < 2:
            print('FAIL: branson log check: cmp needs <gpu.log> <cpu.log>'); return 2
        cpu_steps, cpu_text = parse(logs[1])
        if len(cpu_steps) != len(gpu_steps) or not gpu_steps:
            errors.append(f'step count differs: GPU {len(gpu_steps)} vs CPU {len(cpu_steps)}')
        else:
            print(f'   final completed step: {len(gpu_steps)} of {len(gpu_steps)} in both logs')
            errors += compare_final(gpu_steps[-1], cpu_steps[-1])
        what = ('final-step Post-mat/Absorption/Exit E within 5 %, T_e within 0.02 and the final-step transported '
                'photons within 5 % of the CPU reference')
    for e in errors:
        print('   ERROR:', e)
    print(('FAIL' if errors else 'PASS') + f': branson log check ({mode}): {what}')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
