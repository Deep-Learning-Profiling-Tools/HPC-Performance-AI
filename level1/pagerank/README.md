# PageRank

Power-iteration PageRank on a CSR link matrix.

## Source

Source suite: Hetero-Mark (src/pr (benchmark + cuda + hip))
Upstream repository: https://github.com/NUCAR-DEV/Hetero-Mark
Upstream commit: aa685e02705f4be380cba215c6df10b42262a9cd

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++20 | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
HIP/ROCm: source + build config present where noted, unverified (no AMD GPU available)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Backends

CUDA: working (configure + build + run + validation verified on B200)
HIP: extracted, untested (no AMD GPU / ROCm on the development machine)

## Build

```bash
source hpcperf_env.sh
cmake -S level1/pagerank -B build/pagerank/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/pagerank/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/pagerank -B build/pagerank/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/pagerank/hip
```

## Run

```bash
./build/pagerank/cuda/pagerank_cuda -i build/pagerank/cuda/data/16384.data
```

## Validation

Correctness = the PageRank vector of the last timed run (captured by `PrBenchmark::CaptureGoldenOutputs`, called by `common/benchmark_runner.cc` after the timed repetitions) within max |a - g| / |g| <= (3e-5 - d0) / (1 + d0) of the frozen B200 golden reference `reference/<input_id>.json` (d0 = the golden's own deviation from the CPU implementation; rule in `inputs.yaml` `golden:`). The metric is upstream Verify's (|gpu - cpu| <= cpu/1000 per node), the tolerance is tightened from 1e-3 on the evidence of 2026-10-28 (B200, 500 iterations): 16 legitimate implementations -- fmad off, 4/8/32 lanes per row with shuffle trees, merge-style nonzero chunks and push atomics, cuSPARSE CSR ALG1/ALG2, reverse / 4-accumulator / Kahan / f64 accumulation, f64 vectors, block sizes -- stay <= 1.4e-6, while a rank vector stored in f16 (6.2e-4..7.7e-4, which passes 1e-3) or bf16 (4.5e-3) and matrix values rounded to f16 / bf16 / tf32 (>= 6.7e-2) fail; 3e-5 is the geometric centre, >= 20x on both sides. Renormalising the vector every iteration or re-deriving the matrix values as exact 1/outdegree in f64 changes the output by 1.0e-5..1.1e-5 (the f32 matrix's column sums are not exactly 1: the defined iteration drifts by ~2.1e-8 per step) and passes; no tolerance with sound headroom can see the iteration count after ~12 iterations (that is the probe's job). The upstream Verify (|gpu - cpu| <= cpu/1000 per node, prints 'Passed.') validated that reference once and runs only on request (below). The CSR matrix is generated deterministically at configure time (data/gen_input.py) because the upstream data server is offline.

```bash
ctest --test-dir build/pagerank/cuda --output-on-failure   # golden run of the registry's default input
```

**Correctness probe (`nodes16384-m5`).** After ~12 power iterations the rank vector is converged to f32 rounding, so the 500-iteration performance inputs cannot see how many iterations ran or late partial work: 8 of 500 iterations (3.5e-5..5.1e-5), 12 or more (<= 1.1e-5), an in-place single-buffer update (<= 4.2e-5) and one row, one 64-row block or 5 % of the rows left stale in the last iteration (<= 2.4e-7) pass the 3e-5 limit or exceed it by less than 1.8x (measured 2026-10-28). The registry's correctness-probe input (`inputs.yaml` `purpose: correctness-probe`: never timed, never the default) runs the default matrix with upstream's `-m 5` and has its own golden spec, max_rel <= 1e-5 (the same metric centred on the probe's envelope): 16 legitimate implementations stay <= 4.2e-7 (23x headroom), while one iteration fewer deviates by 6.5e-3, 5 % / a block / one stale row by 4.5e-3 / 3.3e-3 / 4.9e-4, the in-place update by 1.7e-4..3.5e-4 (nondeterministic), an f16 vector by 6.1e-4 and f16 / tf32 matrix values by 8.7e-4 / 1.1e-3 (>= 17x). ctest `pagerank_probe` runs it against its frozen reference. An early exit once the vector stops changing remains unobservable on the 500-iteration inputs (it would only show through the 2.1e-8-per-step drift) and does not trigger on the 5-iteration probe. The benchmark has no damping term and no dangling nodes (every node has out-degree 8-24, data/gen_input.py), so there is no damping order or dangling-node handling to validate.

**Correctness switch.** By default the host-side CPU check is skipped and the benchmark prints `SKIP_VERIFY` (tools/timing relies on it); a plain run, ctest, `tools/inputs` and `tools/timing` never run the CPU reference and nothing falls back to it. `HPCPERF_CPU_REFERENCE=1` runs the original CPU check (offline only: `tools/golden/hpcperf_golden.py make` uses it once to validate the golden reference); `HPCPERF_SKIP_VERIFY=1` still forces it off. `HPCPERF_GOLDEN_DUMP=<dir>` makes the run capture its outputs. See [tools/golden/README.md](../../tools/golden/README.md) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `common/benchmark_runner.cc`: the timed repetitions, not WarmUp/Verify. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 635 (20 source files, cloc, cuda/ + common/)
HIP: 562
