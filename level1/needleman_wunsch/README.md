# Needleman-Wunsch

Needleman-Wunsch global sequence alignment (dynamic programming).

## Source

Source suite: Rodinia (cuda/nw)
Upstream repository: https://github.com/HPC-FAIR/rodinia_3.1
Upstream commit: 366b283456506ef2fe2c2b7dc5e83e941ffc5524

## Verified Environment

GCC/G++ 13.3.0 (conda, pinned) | C++20 | CMake 3.28.4 | Ninja 1.13.2 | Python 3.12.3
CUDA Toolkit 13.2 (nvcc 13.2.78, /usr/local/cuda) | NVIDIA B200 (sm_100), driver 595.58.03
HIP/ROCm: source + build config present where noted, unverified (no AMD GPU available)

Reproduce the toolchain from the repository root: `./setup_env.sh` then
`source hpcperf_env.sh` (all user-space versions are pinned in environment.yml).

## Backends

CUDA: working (configure + build + run + validation verified on B200)
HIP: pending (upstream provides no HIP implementation)

## Build

```bash
source hpcperf_env.sh
cmake -S level1/needleman_wunsch -B build/needleman_wunsch/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/needleman_wunsch/cuda
```

## Run

```bash
cd <run dir> && ./build/needleman_wunsch/cuda/needleman_wunsch_cuda 2048 10     # writes result.txt (TRACEBACK build)
```

## Validation

Correctness = the traceback values (written to result.txt by the TRACEBACK code path enabled at build time, captured as `traceback`, i32 in file order) and the global alignment score (the DP cell (dim, dim) of the copied-back matrix, captured as `score`, i32) byte-identical to the frozen B200 golden reference `reference/<input_id>.json` (rules in `inputs.yaml` `golden:`). The added check `verify.py` (reproduces the glibc rand() input, recomputes the DP matrix on the CPU, walks the upstream traceback comparing all values, and compares the score with the binary's capture) validated that reference once and runs only on request (below). As in verify.py the DP matrix itself is not compared, only the outputs derived from it.

**What is compared, and why (2026-10-28).** Needleman-Wunsch's result is the optimal global alignment of the two sequences: its score and an optimal alignment. Upstream writes only the traceback (with `TRACEBACK`), and that traceback starts at the cell (dim - 1, dim - 1) (`max_rows - 2`), so it covers the alignment of the first dim - 1 symbols of each sequence; the values along it (2,124 of the 4.2M DP cells at n2048) determine the path through upstream's own tie rule (diagonal, then left, then up). The DP matrix is an implementation intermediate (a linear-space or score-only implementation would not produce it) and is not compared. What the traceback does not contain is the score of the full sequences, the cell (dim, dim): the kernels compute the last DP row and column (2 dim + 1 cells), but before 2026-10-28 no compared output depended on them -- a sweep that skips them, or leaves the last row stale, passed (the traceback is identical, the score is then 0 / wrong instead of 21 / 210 / 325). `score` closes that. Measured 2026-10-28 (B200, `hpcperf_golden.py run` against scratch references of this spec, all three inputs): legitimate variants -- 32x32 and 64x64 tiles (`RD_WG_SIZE`), a plain one-thread-per-cell anti-diagonal wavefront in global memory, 32x32 tiles on a tile wavefront with a sequential row-major schedule inside each tile -- pass; wrong ones fail: gap penalty 11 instead of 10, boundary row/column 0 or -(i - 1) p, the substitution score of the neighbouring column, a local-alignment floor at 0, the bottom-right tile not computed, upstream's traceback with left preferred over diagonal (traceback), and the last row and column skipped or the last row left stale (score only: the traceback still matched). A skipped top-right tile (at n2048 its cells left at 0 instead of -20,073..-20,462) passes: it reaches no optimal path, so traceback and score are those of the correct computation.

```bash
ctest --test-dir build/needleman_wunsch/cuda --output-on-failure   # golden run of the registry's default input
python3 tools/golden/hpcperf_golden.py run level1/needleman_wunsch [<input_id>]   # the same for any registered input
```

**Correctness switch.** The binary has no built-in CPU check (nothing to skip, so it prints no `SKIP_VERIFY`); its CPU reference is `verify.py`, which a plain run, ctest, `tools/inputs` and `tools/timing` never run and nothing falls back to. `tools/golden/hpcperf_golden.py make` runs it once, offline (`HPCPERF_CPU_REFERENCE=1`, `HPCPERF_GOLDEN_DUMP`), to validate the golden reference. `HPCPERF_GOLDEN_DUMP=<dir>` makes the binary capture its output. See [tools/golden/README.md](../../tools/golden/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/needle.cu`: both anti-diagonal sweeps. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 305 (3 source files, cloc, cuda/ + common/)
HIP: - (pending)
