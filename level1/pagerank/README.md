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

Correctness = the PageRank vector of the last timed run (captured by `PrBenchmark::CaptureGoldenOutputs`, called by `common/benchmark_runner.cc` after the timed repetitions) within max |a - g| / |g| <= (3e-5 - d0) / (1 + d0) of the frozen B200 golden reference `reference/<input_id>.json` (d0 = the golden's own deviation from the CPU implementation; rule in `inputs.yaml` `golden:`). The metric is upstream Verify's (|gpu - cpu| <= cpu/1000 per node), the tolerance is tightened from 1e-3 on the evidence of 2026-10-28 (B200, 500 iterations): 16 legitimate implementations -- fmad off, 4/8/32 lanes per row with shuffle trees, merge-style nonzero chunks and push atomics, cuSPARSE CSR ALG1/ALG2, reverse / 4-accumulator / Kahan / f64 accumulation, f64 vectors, block sizes -- stay <= 1.4e-6, while a rank vector stored in f16 (6.2e-4..7.7e-4, which passes 1e-3) or bf16 (4.5e-3) and matrix values rounded to f16 / bf16 / tf32 (>= 6.7e-2) fail; 3e-5 is the geometric centre, >= 20x on both sides. Renormalising the vector every iteration or re-deriving the matrix values as exact 1/outdegree in f64 changes the output by 1.0e-5..1.1e-5 (the f32 matrix's column sums are not exactly 1: the defined iteration drifts by ~2.1e-8 per step) and passes -- a uniform scale factor, accepted by design ("Scale-only variants" below); no tolerance with sound headroom can see the iteration count after ~12 iterations (that is the probe's job). The upstream Verify (|gpu - cpu| <= cpu/1000 per node, prints 'Passed.') validated that reference once and runs only on request (below). The CSR matrix is generated deterministically at configure time (data/gen_input.py) because the upstream data server is offline.

```bash
ctest --test-dir build/pagerank/cuda --output-on-failure   # golden run of the registry's default input
```

**Correctness probe (`nodes16384-m5`).** After ~12 power iterations the rank vector is converged to f32 rounding, so the 500-iteration performance inputs cannot see how many iterations ran or late partial work: 8 of 500 iterations (3.5e-5..5.1e-5), 12 or more (<= 1.1e-5), an in-place single-buffer update (<= 4.2e-5) and one row, one 64-row block or 5 % of the rows left stale in the last iteration (<= 2.4e-7) pass the 3e-5 limit or exceed it by less than 1.8x (measured 2026-10-28). The registry's correctness-probe input (`inputs.yaml` `purpose: correctness-probe`: never timed, never the default) runs the default matrix with upstream's `-m 5` and has its own golden spec, max_rel <= 1e-5 (the same metric centred on the probe's envelope): 16 legitimate implementations stay <= 4.2e-7 (23x headroom), while one iteration fewer deviates by 6.5e-3, 5 % / a block / one stale row by 4.5e-3 / 3.3e-3 / 4.9e-4, the in-place update by 1.7e-4..3.5e-4 (nondeterministic), an f16 vector by 6.1e-4 and f16 / tf32 matrix values by 8.7e-4 / 1.1e-3 (>= 17x). ctest `pagerank_probe` runs it against its frozen reference. An early exit once the vector stops changing remains unobservable on the 500-iteration inputs (it would only show through the 2.1e-8-per-step drift) and does not trigger on the 5-iteration probe. The benchmark has no damping term and no dangling nodes (every node has out-degree 8-24, data/gen_input.py), so there is no damping order or dangling-node handling to validate.

**Scale-only variants: renormalised iteration and exact 1/outdegree values (accepted; analysed 2026-10-28).** What the benchmark defines (`common/pr_benchmark.cc` Verify / `CpuPageRankUpdate`, `cuda/pr_cuda_benchmark.cu`): x0 = 1/n (exact in f32 for these power-of-two n, so sum(x0) = 1), then x(k+1) = M x(k) with M the f32 values read from the file, `-m` (default 500) times; no damping, no teleportation, no normalisation, no convergence test; the output is the unnormalised x(500), which Verify compares node by node with the same CPU iteration (|g - c| <= c/1000). M is column-stochastic by construction (M[i][j] = 1/outdegree(j) for each edge j -> i, `data/gen_input.py`; out-degree 8..24, no dangling column), so in exact arithmetic sum(x(k)) = 1 for every k and x(k) tends to the stationary distribution. The stored values f32(1/d) round up for every out-degree that is not a power of two (d = 8 and 16 are exact), so the column sums of the stored matrix are 1 + [0, 5.2e-8] (mean 2.0e-8) and its Perron root is 1 + 2.04e-8..2.05e-8 on the three matrices (offline f64 model; equal to the stationary-weighted column-sum excess): the defined iterates grow by that factor per step, sum(x(500)) = 1 + 1.02e-5 in f64 arithmetic (the frozen B200 goldens: 1 + 1.00e-5..1.08e-5). The two variants:

* *Renormalising every iteration* (x <- M x / sum(M x)): the same input; in exact arithmetic the same iterates up to one scalar per step (exactly x(k) / sum(x(k)), the scaling commutes with the linear map; f64 model 8e-16), i.e. the same direction with the defined output's scale removed -- with an exactly stochastic matrix it is the identical iteration. Its deviation from the defined output is the accumulated drift, 1.02e-5 (f64 model); measured on the B200 against the goldens (round 4): 1.03e-5 / 1.11e-5 / 1.12e-5 (nodes1024 / 4096 / 16384).
* *Exact 1/outdegree values in f64* (each stored value replaced by 1/round(1/v)): different input data -- the exact rational matrix the generator meant and the file approximates; the reconstruction relies on how `gen_input.py` built the values (the file format carries arbitrary f32 weights). Its iterates keep sum = 1 and agree with the renormalised ones to 2.3e-8 (the rounding of M barely moves the Perron vector's direction); deviation from the defined output again the drift: 1.03e-5 (f64 model), 1.02e-5 / 1.10e-5 / 1.11e-5 measured.

Both compute the defined iteration's direction (PageRank's object, the stationary distribution normalised to 1) and differ from the defined output only by a uniform factor 1 + ~1.0e-5 that stems from the f32 rounding of the input, not from the algorithm; neither does less work (one extra reduction per iteration / f64 traffic). Why the current criterion accepts them: the rule is upstream Verify's per-node relative metric with 3e-5 instead of 1e-3; a uniform scale error shows as the same relative error at every node, 1.1e-5 = 2.6x below the effective limit 2.9e-5 (and 90x below upstream's own 1e-3). The limit was centred between measured legitimate (<= 1.4e-6) and wrong (>= 6.2e-4) implementations, and these two were classified as neither: they are equivalent for the intended exactly stochastic matrix and differ only in how the input's rounding is carried. A limit that rejects them would have to sit at ~4e-6 (the geometric centre of the legitimate maximum 1.4e-6 and the variants' 1.0e-5: 2.7x on each side, against >= 20x for the current limit), and the only signal it would add is the same drift: 250 instead of 500 iterations (5.2e-6..5.9e-6 measured) would then fail, 400 would still pass (~2e-6, predicted from the drift) -- so the iteration count stays the probe's job, on which the drift is only 5 x 2.05e-8 and the variants deviate by 4.9e-7 / 4.2e-7 (limit 9.7e-6) while one iteration fewer deviates by 6.5e-3. No validation rule was changed for them; whether the unnormalised scale of the stored-f32 iteration is part of the contract is an owners' decision, recorded here.

**Correctness switch.** By default the host-side CPU check is skipped and the benchmark prints `SKIP_VERIFY` (tools/timing relies on it); a plain run, ctest, `tools/inputs` and `tools/timing` never run the CPU reference and nothing falls back to it. `HPCPERF_CPU_REFERENCE=1` runs the original CPU check (offline only: `tools/golden/hpcperf_golden.py make` uses it once to validate the golden reference); `HPCPERF_SKIP_VERIFY=1` still forces it off. `HPCPERF_GOLDEN_DUMP=<dir>` makes the run capture its outputs. See [tools/golden/README.md](../../tools/golden/README.md) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `common/benchmark_runner.cc`: the timed repetitions, not WarmUp/Verify. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 635 (20 source files, cloc, cuda/ + common/)
HIP: 562
