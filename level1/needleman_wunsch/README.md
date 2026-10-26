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

Correctness = the traceback values (written to result.txt by the TRACEBACK code path enabled at build time, captured as `traceback`, i32 in file order) byte-identical to the frozen B200 golden reference `reference/<input_id>.json` (rule in `inputs.yaml` `golden:`). The added check `verify.py` (reproduces the glibc rand() input, recomputes the DP matrix on the CPU and walks the upstream traceback, comparing all values) validated that reference once and runs only on request (below). As in verify.py the DP matrix itself is not compared, only the traceback derived from it.

```bash
ctest --test-dir build/needleman_wunsch/cuda --output-on-failure   # golden run of the registry's default input
python3 tools/golden/hpcperf_golden.py run level1/needleman_wunsch [<input_id>]   # the same for any registered input
```

**Correctness switch.** The binary has no built-in CPU check (nothing to skip, so it prints no `SKIP_VERIFY`); its CPU reference is `verify.py`, which a plain run, ctest, `tools/inputs` and `tools/timing` never run and nothing falls back to. `tools/golden/hpcperf_golden.py make` runs it once, offline (`HPCPERF_CPU_REFERENCE=1`, `HPCPERF_GOLDEN_DUMP`), to validate the golden reference. `HPCPERF_GOLDEN_DUMP=<dir>` makes the binary capture its output. See [tools/golden/README.md](../../tools/golden/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/needle.cu`: both anti-diagonal sweeps. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 305 (3 source files, cloc, cuda/ + common/)
HIP: - (pending)
