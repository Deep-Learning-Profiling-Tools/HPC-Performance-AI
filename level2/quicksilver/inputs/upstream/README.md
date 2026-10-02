# Upstream Quicksilver decks (verbatim)

Copied byte-for-byte from LLNL/Quicksilver at the pinned CUDA revision
`eb68bb8d6fc53de1f65011d4e79ff2ed0dd60f3b` (the same commit `source/cuda/` was
taken from), path `Examples/`:

| file | upstream path | what it is |
|---|---|---|
| `Coral2_P1_1.inp` | `Examples/CORAL2_Benchmark/Problem1/Coral2_P1_1.inp` | CORAL-2 Problem 1 (`coralBenchmark: 1`), single-rank size: 16^3 mesh, 163,840 particles, 100 steps, `xDom=yDom=zDom=1` |
| `Coral2_P2_1.inp` | `Examples/CORAL2_Benchmark/Problem2/Coral2_P2_1.inp` | CORAL-2 Problem 2 (`coralBenchmark: 2`, 3 cross-section tables, `bTally/fTally/cTally`): 11^3 mesh, 53,240 particles, 100 steps, 1 rank |
| `CTS2_1.inp` | `Examples/CTS2_Benchmark/CTS2_1.inp` | CTS-2 problem (`coralBenchmark: 2` physics with the CTS-2 cross sections): 16^3 mesh, 40,960 particles, 100 steps, 1 rank |

`SHA256SUMS` records the hashes of the copies; `../inputs.yaml` registers them as
input ids. These decks carry their own mesh/particle/step counts, so `run.sh`
passes them **verbatim** (`-i <deck>` only) when selected through
`HPCPERF_QUICKSILVER_INPUT`; the pre-existing `coral2_p1_profile.inp` path
(physics block only, sizes supplied by `run.sh`) is unchanged.
