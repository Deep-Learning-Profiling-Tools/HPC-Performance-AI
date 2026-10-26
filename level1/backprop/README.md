# Backprop

One training step of a fully-connected neural network (face recognition workload shape).

## Source

Source suite: Rodinia (cuda/backprop)
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
cmake -S level1/backprop -B build/backprop/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/backprop/cuda
```

## Run

```bash
./build/backprop/cuda/backprop_cuda 65536
```

## Validation

Golden reference (tools/golden): a plain run captures the trained weights `input_weights` ((in + 1) x (hid + 1)) and `hidden_weights` ((hid + 1) x (out + 1)), f32, streamed row by row from the row-pointer matrices (`HPCPERF_GOLDEN_DUMP`), and `hpcperf_golden.py check` compares them with the frozen B200 reference `reference/<input>.json`: max |a - g| / (1 + |g|) <= 1e-3 - d0 per matrix, i.e. the added check's tolerance against an identically-initialized network trained with the upstream CPU `bpnn_train`, minus the golden's own deviation d0 from it (measured once when the reference was frozen). That CPU training step (prints PASS/FAIL) runs only on the explicit offline request below. **Known defect (not fixed):** `bpnn_train_cuda` never writes its `*eo`/`*eh` arguments, so the output/hidden error scalars the added check also compares are uninitialized on the GPU path (they read 0 on dgx003 and dominate the printed "Max relative difference"); the golden check does not compare them. ctest runs `hpcperf_golden.py run` on the registry's default input (`n65536`, the former ctest argument); the `check:` block of `inputs.yaml` does the same for every registered input.

Run it via:

```bash
ctest --test-dir build/backprop/cuda --output-on-failure
```


**CPU reference switch.** By default the host-side CPU check is skipped: the benchmark prints `SKIP_VERIFY` (never PASS/FAIL) and its correctness is the golden check above. `HPCPERF_CPU_REFERENCE=1` runs the CPU reference (offline only: `hpcperf_golden.py make` uses it once to validate a new golden reference; no measurement, check or ctest path sets it, and nothing falls back to it); `HPCPERF_SKIP_VERIFY=1` (tools/timing) keeps it off as well. See [tools/golden](../../tools/golden/) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/backprop_cuda.cu`: one training step: GPU forward, host output layer/errors, GPU weight adjust. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 694 (6 source files, cloc, cuda/ + common/)
HIP: - (pending)
