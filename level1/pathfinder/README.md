# PathFinder

Dynamic-programming minimum path through a 2D grid.

## Source

Source suite: Rodinia (cuda/pathfinder)
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
cmake -S level1/pathfinder -B build/pathfinder/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/pathfinder/cuda
```

## Run

```bash
./build/pathfinder/cuda/pathfinder_cuda 100000 100 20     # BENCH_PRINT: prints the whole wall, then the first row and the result row
```

## Validation

Correctness = the final DP row (`result` after the final copy-back, captured as `result`, i32; the last line BENCH_PRINT prints) byte-identical to the frozen B200 golden reference `reference/<input_id>.json` (rule in `inputs.yaml` `golden:`). The added check `verify.py` (reproduces the glibc rand() wall, checks the printed first row, recomputes the DP result row on the CPU in pure Python, ~10-13 s for the default input) validated that reference once and runs only on request (below).

```bash
ctest --test-dir build/pathfinder/cuda --output-on-failure   # golden run of the registry's default input
python3 tools/golden/hpcperf_golden.py run level1/pathfinder [<input_id>]   # the same for any registered input
```

**Correctness switch.** The binary has no built-in CPU check (nothing to skip, so it prints no `SKIP_VERIFY`); its CPU reference is `verify.py`, which a plain run, ctest, `tools/inputs` and `tools/timing` never run and nothing falls back to. `tools/golden/hpcperf_golden.py make` runs it once, offline (`HPCPERF_CPU_REFERENCE=1`, `HPCPERF_GOLDEN_DUMP`), to validate the golden reference. `HPCPERF_GOLDEN_DUMP=<dir>` makes the binary capture its output. See [tools/golden/README.md](../../tools/golden/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/pathfinder.cu`: calc_path (all pyramid steps). Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 172 (1 source files, cloc, cuda/ + common/)
HIP: - (pending)
