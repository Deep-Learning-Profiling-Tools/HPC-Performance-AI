# Block Scan

Block-level exclusive prefix sum (`cub::BlockScan::ExclusiveSum` plus the tile aggregate) of one 1024-int tile, in 18
variants (3 BlockScan algorithms x 6 threads/items splits); every one of the `grid_size` thread blocks scans the same
tile into the same output (a throughput replication, as in CUB's example_block_scan).

## Source

Source suite: HeCBench (src/blockScan-cuda, src/blockScan-hip)
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
cmake -S level1/block_scan -B build/block_scan/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/block_scan/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/block_scan -B build/block_scan/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/block_scan/hip
```

## Run

```bash
./build/block_scan/cuda/block_scan_cuda 2227104 100
```

## Validation

Golden reference (tools/golden): a plain run captures the scan output of each of the 18 variants after its timed launches (`scan.<algorithm>.t<threads>.i<items>`, 1024 prefix sums + the aggregate, int, compared exactly; bit-identical to the first warm-up launch's output that upstream's check compares and from which the frozen references were made). Between the warm-up launches and the timed region `d_out` (and its host copy) is poisoned (all-ones bytes, -1), outside the ROI and the timer, so timed launches that do not run fail the check instead of passing on the warm-up result. `tools/golden/hpcperf_golden.py check` compares the capture with the frozen B200 reference `reference/<input_id>.json` (inputs.yaml `golden:` / `check:`). ctest runs the registry's default input this way (`hpcperf_golden.py run`). The upstream CPU reference (`memcmp` with the host prefix sums from `Initialize`) stays in the source and runs only with `HPCPERF_CPU_REFERENCE=1`, which `hpcperf_golden.py make` uses once to validate a new reference.

Run it via:

```bash
ctest --test-dir build/block_scan/cuda --output-on-failure
```

**What the benchmark computes.** As in CUB's `example_block_scan.cu` (from which HeCBench's blockScan is derived), each of the 18 variants computes the exclusive prefix sum of one 1024-int tile plus the tile's total (the aggregate), with one thread block; `grid_size` only replicates that identical block computation (every block loads the same `d_in` tile and stores the same 1025 ints to the same `d_out`) to saturate the GPU for the throughput figure `TILE_SIZE * grid_size / time` (CUB's own default grid size is 1). Upstream checks, after the first launch only, `memcmp` of `d_out` with the host prefix sums and the aggregate from `Initialize`. Consequently the two performance inputs have byte-identical references, and no output-based check can tell how many of the `grid_size` blocks -- or of the timed launches -- ran: measured 2026-10-28, a candidate that launches one block passes all inputs. Under the benchmark's mathematical semantics (one tile scan per variant) the replicated blocks are redundant work, so their number is not a correctness property and no grid-size requirement is imposed; it only makes the throughput figure of such a candidate meaningless. Making it observable needs per-block distinct tiles and outputs (`grid_size x 1025` ints, 9.1 GB each way at 2227104 blocks), which turns the on-chip block-scan throughput test into a DRAM-bandwidth test -- a workload change (class E), reported, not implemented.

**Correctness probe (`probe-randsigned`).** Upstream's tile `h_in[i] = i % 17` is a fixed pattern of small non-negative values (prefix sums <= 8166), scanned by all 18 variants: a scan that never reads `d_in` (the closed form of the pattern), the 1025 frozen output values written as constants, a scan in f32 (exact up to 2^24) or in 16-bit integers (exact up to 32767), and one scan copied into all 18 outputs reproduce every output of both performance inputs byte for byte (measured 2026-10-28). The registry's correctness-probe input `probe-randsigned` (`block_scan_cuda 2227104 100 --probe`; `inputs.yaml` `purpose: correctness-probe`: never timed, never the default) runs the same 18 kernels and launches on tiles selected in set-up only (`InitializeProbe`): each variant gets its own tile of signed values uniform in [-2^21, 2^21) from a splitmix64 stream with a fixed per-variant seed, so the 18 expected outputs differ, the prefix sums span -6.2e7 .. 6.6e7 (33 to 961 of the 1024 beyond 2^24 per variant) and no partial sum can overflow `int`. Measured on the B200 against a scratch reference: those five candidates fail (the copy in 17 of the 18 variants, the others in all 18), and so do an inclusive or shifted scan, a missing or short aggregate, a half tile and a dropped last item (all of which the performance inputs also catch); warp-local stitching fails the 15 variants with more than one warp and per-thread stitching the 15 with more than one item per thread (on every input); the legitimate alternatives -- the other CUB scan algorithm, an inclusive scan minus the item, a 64-bit scan, a serial and a Hillis-Steele shared-memory scan, `cub::DeviceScan` on the tile, one block instead of `grid_size` -- pass. The rule stays `exact`: integer addition is associative, so every correct evaluation order on any architecture gives the same bits. ctest `block_scan_probe` runs it against its frozen reference `reference/probe-randsigned.json` (about 60 s, like the default input).


**Measurement switch.** By default the host-side CPU reference is skipped and the benchmark prints `SKIP_VERIFY` (exit 0), so a plain run, ctest, `tools/inputs` and `tools/timing` never execute it; `HPCPERF_CPU_REFERENCE=1` (offline, golden make only) turns it on, `HPCPERF_SKIP_VERIFY=1` keeps it off regardless. Output capture (`HPCPERF_GOLDEN_DUMP`) is outside the timed region. See [tools/golden/hpcperf_golden.h](../../tools/golden/hpcperf_golden.h) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/main.cu` and the HIP port: each region the benchmark's own timer measures around its kernel launches. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 162 (1 source files, cloc, cuda/ + common/)
HIP: 160
