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

Correctness = the results of the four variants (`r` of bs1..bs4, `size_t[2 x elements]`, captured after each variant's timed region and copy-back as `r.bs1`..`r.bs4`, u64; the buffer the four variants share, `d_r` and its captured host copy `r`, is poisoned (all-ones bytes, SIZE_MAX: never a valid index) before each variant's timed region, outside it and the timer, so a variant that does not run fails the check instead of passing on the previous variant's identical result) byte-identical (sha256; mode hash: 134 MB per variant for the default input, 2.1 GB for n134217728) to the frozen B200 golden reference `reference/<input_id>.json` (rule in `inputs.yaml` `golden:`). Upstream's own check `verify()` -- every result satisfies `a[r] <= z < a[r+1]` with `r+1 < aSize`, a host-side pass over the GPU output, not a recomputation -- validated that reference once and, since 2026-10-27, also runs in every default run (below), independently of the reference. The property has at most one solution for the non-decreasing `a[]` (also where float rounding repeats values of `a[]` above 2^24) and, with the query-range correction ("Input n134217728" below), exactly one for every generated query, so a correct result equals the golden bit for bit.

Upstream compiles `verify()` and the copy-back of the results only with `-DDEBUG` (Makefile `VERIFY=yes`), which the build never defines: the unmodified default build checked nothing (the old ctest only grepped for an "incorrect result" that could not be printed). Now the copy-back always happens (a device-to-host copy outside every timed region / ROI) and `verify()` runs by default after each variant's copy-back, outside the ROI and the benchmark's own timer; it returns its verdict (upstream: `break`), prints `PASS: verify: ...` / `FAIL: verify: ...` after the four variants and a failure makes the process exit 1 (upstream: exit 0 either way), so the golden run (ctest) and the registry check FAIL ("the benchmark run exited 1") even where no reference exists. Its bound test is `r < aSize - 1` (upstream `r+1 < aSize`): identical except for `r = SIZE_MAX`, the poison a variant leaves where it wrote no result, for which `r+1` wraps to 0 and upstream then read `a[SIZE_MAX]`, before the allocation (ASan heap-buffer-overflow, reproduced with a mutant that skips one write); its diagnostic no longer reads `a[r]`, `a[r+1]` out of range, prints the bracket exactly (upstream's 6 significant digits showed a violated bracket above 10^6 as satisfied: `1.34218e+08 <= 1.34218e+08 < 1.34218e+08`) and names a query that has no valid `r` at all. Fewer than 2 elements are refused (exit 1): upstream divided by zero (`rand() % 0`) for 1 and, for 0, wrapped `N` to SIZE_MAX and checked zero queries (a vacuous PASS).

```bash
ctest --test-dir build/binary_search/cuda --output-on-failure   # golden run of the registry's default input
python3 tools/golden/hpcperf_golden.py run level1/binary_search [<input_id>]   # the same for any registered input
```

**Correctness switch.** `verify()` runs by default (a plain run, ctest, `tools/inputs check`, `tools/golden make`); `HPCPERF_SKIP_VERIFY=1` turns it off and the benchmark prints `SKIP_VERIFY` -- `tools/timing/measure_level1.sh` sets it for timing (`tools/inputs check` and `tools/golden run` remove it); a `-DDEBUG` build runs it regardless, as upstream. `HPCPERF_CPU_REFERENCE` no longer matters here (verify() is a property check of the GPU output, not a CPU reference). Measured cost (B200 node, 2026-10-27; verify() timed in a scratch copy, plus the process wall of default vs. `HPCPERF_SKIP_VERIFY=1` runs, ROI unchanged): n8388608 ~0.1 s per variant, 0.46-0.64 s in all (wall +0.5 s on 1.8 s); n33554432 ~0.43 s per variant, 1.7-1.9 s (wall +1.9 s on 5.3 s); n134217728 (corrected queries, 2026-10-28, shared node) 2.0-2.4 s per variant over all 268435456 queries, 8.9 s in all (wall 58.0 s against 52.2-56.1 s with `HPCPERF_SKIP_VERIFY=1`). `HPCPERF_GOLDEN_DUMP=<dir>` (with `HPCPERF_GOLDEN_MODE=hash`) makes the binary record the four results. See [tools/golden/README.md](../../tools/golden/README.md).

**Input n134217728 (query-range correction, 2026-10-28).** *Root cause.* Upstream builds `a[i] = i` ("strictly ascending") and the queries `z[i] = rand() % N` ("lower = 0, upper = n-1"; `N = aSize-1`, `srand(2)`, glibc `rand()` in [0, 2^31)) in `Real_t = float`. In exact arithmetic every query integer is at most N-1 < N = `a[N]`, so `verify()`'s `a[r] <= z < a[r+1]`, `r+1 < aSize` has exactly one solution. In float, integers above 2^24 are rounded to nearest (ties to even) on both sides, with two consequences:

* `a[]` is only non-decreasing: runs of up to 3 equal values in n33554432 and up to 9 in n134217728. This is consistent with `verify()` and with the kernels: for every z in [`a[0]`, `a[N]`) the half-open bracket still has exactly one solution, the LAST index of z's run (r1 < r2 would need `a[r1+1] > z >= a[r2] >= a[r1+1]`), and that is what all four kernels compute -- bs1 (bisection keeping `a[low] <= z`, `high` starting at n, never read) and bs2 (bitwise search, steps taken only for `r < n`) return the last r < n with `a[r] <= z`, bs3/bs4 the same with `a[n]` read as a sentinel in place of indices >= n. They are upper-bound-minus-one searches, so `verify()`'s half-open bracket is the right specification also for repeated values; a lower-bound result (the first index of a run) violates it.
* A query integer can round to `a[N]` itself whenever aSize-2 and aSize-1 round to the same float (from aSize = 2^24+2 on, 218103807 of the sizes in [3, 2^28]; not 2^23, exact, and not 2^25, where (float)(2^25-2) = 2^25-2 < (float)(2^25-1) = 2^25 -- for any `rand()` stream). Such a query has no valid r: bs1/bs2 return N-1, bs3/bs4 N (the sentinel path takes every bit). At 2^27 elements `a[2^27-4 .. 2^27-1]` = 2^27 = `a[N]`, and the query integers 2^27-4 (a tie, rounded to even), 2^27-3 and 2^27-2 round to 2^27: probability 48/2^31 per query, expected exactly 6 of the 268435456 queries, and the `srand(2)` stream has 6 (indices 91320694, 134025290, 146221282, 170476487, 201230603, 232400441). `verify()` failed all four variants and no golden reference could be made.

*Correction (`cuda/main.cu`, `hip/main.cu`).* A pure insertion after upstream's generator line, outside every timed region / ROI: a query with `!(z < a[N])` -- only such a query -- is replaced by `std::nextafter(a[N], 0)`, the largest `Real_t` below `a[N]`: the smallest change that gives it its one valid answer, i.e. upstream's "upper = n-1" realised in float. At 2^27: z = 134217728 becomes 134217720 (2^27-8, the value the three integers round down to, which about 14 other queries already have), its result r = N-4 = 134217723 instead of N-1 / N. The element count, `a[]`, the `rand()` stream and the other 268435450 queries are unchanged (bit-identical, host comparison of `z[]`), so the input is upstream's except for these 6 queries; n8388608 and n33554432 are byte-identical to upstream's (their frozen references PASS bit for bit with the corrected binary). With `-DReal_t=double` (exact below 2^53) it is a no-op.

*Alternatives (built and run in scratch, 2026-10-28).* Redraw in a second pass after the stream (rejection sampling of `rand() % N`): keeps the query distribution exactly and also PASSes, but the 6 replacements become arbitrary new queries (up to 7.6e7 away) taken from the `rand()` continuation. Round-toward-zero conversion of every query: changes 8386728 queries of n33554432 and 92269579 of n134217728 (rejected). Round-toward-zero of the affected queries only: equal to the correction at 2^27 but not general (at aSize = 2^27+8, `a[N]` = 2^27 < N and 7 of the 11 affected integers truncate to `a[N]` again). 2^27-3 elements (another size) or `-DReal_t=double` (8-byte keys: another workload).

*Evidence (B200, 2026-10-28).* With the correction `verify()` PASSes for all four variants; the results of bs1..bs4 are identical (sha256 `319b33ea...`) and equal, bit for bit, to an independent host recomputation (`upper_bound - 1` per query); the same host recomputation reproduces the committed n8388608 / n33554432 references and, for upstream's queries, the failing results (bs1/bs2 `584b92ee...`, bs3/bs4 `3699701c...`). ROI per variant unchanged within the run-to-run noise (n134217728, `HPCPERF_SKIP_VERIFY=1`, `HPCPERF_ROI_LOG`, alternating runs on one B200 of a shared node): best run per variant uncorrected / corrected bs1 10.707 / 10.702 s, bs2 10.700 / 10.711 s, bs3 10.724 / 10.730 s, bs4 10.789 / 10.789 s; four-variant total median 43.69 s (6 runs, 43.31-44.72) / 44.12 s (9 runs, 42.93-49.43). The 6 moved queries are 2e-8 of the searches. (Absolute times are above the 25.3 s of the 2026-09-23 timing campaign, which ran on another B200, PCI 0000:52:00.0; these runs on 0000:C3:00.0 read an SM clock of 1155 MHz under load. Only the paired comparison is evidence here.)

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/main.cu` and the HIP port: each region the benchmark's own timer measures around its kernel launches (4 marked regions; their times add up). Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 254 (1 source files, cloc, cuda/ + common/)
HIP: 254
