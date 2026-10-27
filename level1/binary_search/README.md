# Binary Search

Batched binary searches over a sorted array with four kernel variants.

## Source

Source suite: HeCBench (src/bsearch-cuda, src/bsearch-hip)
Upstream repository: https://github.com/ORNL/HeCBench
Upstream commit: 23714d9980070bc5543e99c11fd93ee6f79c6947

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
cmake -S level1/binary_search -B build/binary_search/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/binary_search/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/binary_search -B build/binary_search/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/binary_search/hip
```

## Run

```bash
./build/binary_search/cuda/binary_search_cuda 8388608 100
```

## Validation

Correctness = the results of the four variants (`r` of bs1..bs4, `size_t[2 x elements]`, captured after each variant's timed region and copy-back as `r.bs1`..`r.bs4`, u64; the buffer the four variants share, `d_r` and its captured host copy `r`, is poisoned (all-ones bytes, SIZE_MAX: never a valid index) before each variant's timed region, outside it and the timer, so a variant that does not run fails the check instead of passing on the previous variant's identical result) byte-identical (sha256; mode hash: 134 MB per variant for the default input, 2.1 GB for n134217728) to the frozen B200 golden reference `reference/<input_id>.json` (rule in `inputs.yaml` `golden:`). Upstream's own check `verify()` -- every result satisfies `a[r] <= z < a[r+1]` with `r+1 < aSize`, a host-side pass over the GPU output, not a recomputation -- validated that reference once and, since 2026-10-27, also runs in every default run (below), independently of the reference. The property has at most one solution for the non-decreasing `a[]` (also where float rounding repeats values of `a[]` above 2^24), so a correct result equals the golden bit for bit.

Upstream compiles `verify()` and the copy-back of the results only with `-DDEBUG` (Makefile `VERIFY=yes`), which the build never defines: the unmodified default build checked nothing (the old ctest only grepped for an "incorrect result" that could not be printed). Now the copy-back always happens (a device-to-host copy outside every timed region / ROI) and `verify()` runs by default after each variant's copy-back, outside the ROI and the benchmark's own timer; it returns its verdict (upstream: `break`), prints `PASS: verify: ...` / `FAIL: verify: ...` after the four variants and a failure makes the process exit 1 (upstream: exit 0 either way), so the golden run (ctest) and the registry check FAIL ("the benchmark run exited 1") even where no reference exists. Its bound test is `r < aSize - 1` (upstream `r+1 < aSize`): identical except for `r = SIZE_MAX`, the poison a variant leaves where it wrote no result, for which `r+1` wraps to 0 and upstream then read `a[SIZE_MAX]`, before the allocation (ASan heap-buffer-overflow, reproduced with a mutant that skips one write); its diagnostic no longer reads `a[r]`, `a[r+1]` out of range and names a query that has no valid `r` at all.

```bash
ctest --test-dir build/binary_search/cuda --output-on-failure   # golden run of the registry's default input
python3 tools/golden/hpcperf_golden.py run level1/binary_search [<input_id>]   # the same for any registered input
```

**Correctness switch.** `verify()` runs by default (a plain run, ctest, `tools/inputs check`, `tools/golden make`); `HPCPERF_SKIP_VERIFY=1` turns it off and the benchmark prints `SKIP_VERIFY` -- `tools/timing/measure_level1.sh` sets it for timing (`tools/inputs check` and `tools/golden run` remove it); a `-DDEBUG` build runs it regardless, as upstream. `HPCPERF_CPU_REFERENCE` no longer matters here (verify() is a property check of the GPU output, not a CPU reference). Measured cost (B200 node, 2026-10-27; verify() timed in a scratch copy, plus the process wall of default vs. `HPCPERF_SKIP_VERIFY=1` runs, ROI unchanged): n8388608 ~0.1 s per variant, 0.46-0.64 s in all (wall +0.5 s on 1.8 s); n33554432 ~0.43 s per variant, 1.7-1.9 s (wall +1.9 s on 5.3 s); n134217728 ~0.7 s per variant up to its first failing query, 3.0 s (wall +3.6 s on 31.7 s). `HPCPERF_GOLDEN_DUMP=<dir>` (with `HPCPERF_GOLDEN_MODE=hash`) makes the binary record the four results. See [tools/golden/README.md](../../tools/golden/README.md).

**Input n134217728.** With `Real_t = float`, `a[i] = i` and `z = rand() % N` are rounded above 2^24; at 2^27 elements 6 of the 268435456 queries round to `a[aSize-1]` (e.g. query 91320694: z = 134217728), for which no `r` with `r+1 < aSize` exists: `verify()` reports "incorrect result" for every variant (bs1/bs2 return N-1, bs3/bs4 return N), so no golden reference can be made for this input until its registration is decided -- and since verify() runs by default, every plain run, golden run and registry check of this input FAILs with exit 1 (with `HPCPERF_SKIP_VERIFY=1` it still runs to completion for timing). n8388608 and n33554432 pass `verify()`.

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/main.cu` and the HIP port: each region the benchmark's own timer measures around its kernel launches (4 marked regions; their times add up). Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 254 (1 source files, cloc, cuda/ + common/)
HIP: 254
