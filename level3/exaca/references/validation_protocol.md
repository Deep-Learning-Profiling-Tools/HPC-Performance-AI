# ExaCA validation protocol (v2, frozen 2026-09-11 before the holdout runs)

Scope: the `dirsolid` smoke case only (128x128x128 cells, `inputs/dirsolid.template.json`, seed 0, nucleation
density 10). Strong/weak modes are performance modes with completion checks only (log statistics recorded, no
acceptance claim). Nothing here is an upstream-provided oracle: ExaCA ships no reference output for this problem,
so this is a **project-defined statistical validation**.

## Inputs, metrics, decision rule

- Input: the frozen deck template (sha256 recorded in every `run_manifest.txt`), the frozen binary of the
  benchmark build, `GrainOrientationVectors.csv` from the artifact.
- Metrics (`exaca_check.py stats`): `unsolidified_cells`, `n_grains`, `n_nucleated`, `vol_fraction_nucleated`,
  `top_layer_grains`, `mean_misorientation_z_deg`, `mean_misorientation_z_top_deg`, `mean_grain_volume_cells`.
- Decision (`exaca_check.py validate`, each criterion must hold):
  1. completeness / one global decomposition: dimensions == deck; every cell solidified; the log reports N ranks;
     the Y subdomains **tile the box exactly once** (first offset 0, `offset[i+1] == offset[i] + size[i] - 2`,
     last end == Ny, every size >= 2, N entries). The halo-sum formula alone is not used: it would accept a
     missing subdomain paired with a duplicated one.
  2. self-consistency: ExaCA's own `VolFractionNucleated` == the value recomputed from the field (1e-3).
  3. the run's statistics vs the frozen reference (one calibration run) within the frozen tolerances.
  4. N > 1: the N-rank statistics vs the same build's 1-rank run within the same tolerances.
- Tolerance semantics: applied to **a single run** vs the reference, absolute or relative per metric; not to a
  mean of runs. The reference is itself one sample of the same distribution; its uncertainty is covered because
  the tolerance is derived from the observed range between samples (below), multiplied by 3.

## Spread, calibration and the tolerance rule

- Spread = **range (max - min)** of each metric over the 8 calibration runs (4 x 1 GPU, 1 x 2 GPU, 3 x 4 GPU,
  2026-09-10, `references/calibration.json`, binary `f2a8875c...`); the population standard deviation is
  recorded for information. With n = 8 this is an empirical range rule -- **no 3-sigma confidence claim is made**.
- Tolerance = max(3 x range, floor): `vol_fraction_nucleated` 0.01 abs (range 0.0012), `n_grains` 1 % rel
  (range 10 = 0.28 %), `n_nucleated` 5 % rel (= +-1 of 20; range 0), `top_layer_grains` 25 % rel (range 2 of
  29 = 6.9 %; 3 x = 20.7 %), `mean_misorientation_z_deg` 0.25 deg (range 0.033), `mean_misorientation_z_top_deg`
  0.7 deg (range 0.226; 3 x = 0.68), `mean_grain_volume_cells` 1 % rel (range 1.6 = 0.28 %), `unsolidified_cells`
  exactly 0.
- Protocol v1 (2026-09-10) used tolerances `top_layer_grains` 15 % and `mean_misorientation_z_top_deg` 0.5 deg
  (only 2.2 x the calibration range) and its 1/2/4-GPU "validations" were run against the same data that set the
  thresholds. Those results are kept as **CALIBRATION**, not as acceptance. v2 widens exactly those two
  tolerances to the 3 x rule and is frozen before any holdout run.

## Holdout (independent acceptance)

- Plan: 3 independent sets, each = fresh 1-, 2- and 4-GPU runs (9 runs, new run ids, separate run directories
  `run.holdout-001-{1,2,3}`), evaluated by `validate.sh` with the frozen v2 rule. Budget: ~12 s per run.
- Acceptance: every holdout run passes every criterion. A failure is analysed and kept; the tolerances are
  **not** widened on the same holdout -- a revised rule (v3) would need a new holdout set.
- Record: `references/holdout.json` (run ids, per-run metrics, verdicts).

## Non-determinism: evidence, not assumption

- Observation: identical 1-GPU runs differ in 2.2 % of cells (46,499 of 2,097,152, s1a vs s1b), 1- vs 4-rank runs
  in 6.1 %; the differing cells carry thousands of distinct (id_a, id_b) pairs and the grain-ID sets differ
  slightly, so these are **real spatial differences of competitive growth, not a renumbering**. The single-grain
  analytic case (`Inp_SmallEquiaxedGrain`, 64^3) is not bitwise reproducible either (see the calibration file).
- Code path consistent with this (not an upstream statement): active cells are collected into the steering
  vector with `Kokkos::atomic_fetch_add` (`src/CAupdate.hpp` lines 41/97/108/132: order of processing varies
  between launches) and a liquid cell contested by several growing octahedra in the same step is claimed by
  `Kokkos::atomic_compare_exchange` on `cell_type` (`src/CAupdate.hpp` line 208): the winner sets the new
  octahedron's centre and diagonal length, so the race changes the subsequent growth geometry. Upstream
  documents no determinism guarantee; the comment at line 432 mentions a race the code avoids (a different one).
- Consequence: bitwise comparison is not a valid criterion; upstream's GoogleTest unit tests are the only
  deterministic upstream checks (kernel-level; not built here because GoogleTest is not part of the environment
  or the artifact) -- cross-rank statistical agreement is therefore combined with the invariants of criterion 1/2
  and the frozen reference, and this remains a project-defined validation.
