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

Golden reference (tools/golden): a plain run captures the trained weights `input_weights` ((in + 1) x (hid + 1)) and `hidden_weights` ((hid + 1) x (out + 1)), f32, streamed row by row from the row-pointer matrices (`HPCPERF_GOLDEN_DUMP`), and `hpcperf_golden.py check` compares them with the frozen B200 reference `reference/<input>.json`: max |a - g| / (1 + |g|) <= 1e-3 - d0 per matrix, i.e. the added check's tolerance against an identically-initialized network trained with the upstream CPU `bpnn_train`, minus the golden's own deviation d0 from it (measured once when the reference was frozen). That CPU training step (prints PASS/FAIL) runs only on the explicit offline request below. **Defect fixed 2026-10-27:** upstream's `bpnn_train_cuda` never wrote its `*eo`/`*eh` arguments, so the output/hidden error scalars the added check also compares were uninitialized on the GPU path (they read 0 on dgx003 and dominated the printed "Max relative difference"); it now returns them, as the CPU `bpnn_train` does (`cuda/backprop_cuda.cu`, host scalars after the ROI; no weight, kernel or timed work changes, the captured weights of every performance input are bit-identical). The golden check does not compare them (the frozen spec is unchanged). So the fix affects only the offline CPU check's error-scalar comparison: before it, the printed "Max relative difference" at `n65536` was 5.5e-5, i.e. the real output error 5.5e-5 against an uninitialized 0 (4.3e-4 at `n262144`, 2.3x below the 1e-3 limit). Re-checked 2026-10-28 (B200): a default `n65536` run's captured `input_weights` / `hidden_weights` have exactly the sha256 of the committed reference (golden PASS); with `HPCPERF_CPU_REFERENCE=1` it prints output error 0.000055 / hidden error 0.000000 on both the GPU and the CPU side and a max relative difference of 0; `probe-dpn1-n32` prints 0.119931 / 0.176782 on both sides, max relative difference 3.8e-8. ctest runs `hpcperf_golden.py run` on the registry's default input (`n65536`, the former ctest argument); the `check:` block of `inputs.yaml` does the same for every registered input.

Run it via:

```bash
ctest --test-dir build/backprop/cuda --output-on-failure
```


**Correctness probes (`probe-dpn1-n32`, `probe-dpn1-n1024`).** With upstream's weights in [0, 1] every hidden sum is ~in/4 (16384 for `n65536`), all 16 hidden units saturate to exactly 1.0f, `hidden_delta = 0` and the input-weight update is exactly 0: skipping `bpnn_adjust_weights_cuda`, replacing the forward kernel by hidden units = 1, or skipping the whole training step passes every performance input. The registry's two correctness-probe inputs (`backprop_cuda 32 --probe`, `backprop_cuda 1024 --probe`; `inputs.yaml` `purpose: correctness-probe`: never timed, never the default) run the same kernels and host code with set-up data selected in `setup()` / `bpnn_randomize_weights` only: input and hidden weights from upstream's own commented-out alternative `dpn1()` (symmetric, [-1, 1]; same seed 7 and number of `rand()` calls) and a small input layer, so the hidden units do not saturate (32 units, 2 thread blocks: 14 of 16 hidden units unsaturated; 1024 units, 64 thread blocks: 5 of 16). Measured 2026-10-27 on the B200 (max |a - g| / (1 + |g|), tolerance 1e-3): the skipped weight adjustment fails both (7.2e-3), hidden units = 1 fails both (1.8e-2 / 2.5e-2 in `hidden_weights`), no training fails both (3.2e-2 / 3.5e-2), `--fmad=false` passes (0); the CPU reference's deviation is <= 4e-8. Sizes 256 and 4096 were tried and rejected: at 256 the output unit nearly reaches its target 0.1 (`delta_o` = -4e-3), at 4096 only 2 of 16 hidden units are unsaturated, and a skipped update stays within 1e-3 (2.7e-4 / 4.8e-4). The probes' CPU validation needs the `*eo`/`*eh` fix above: their error scalars are far from 0. ctest `backprop_probe_n32` / `backprop_probe_n1024` run them against their frozen references.

**CPU reference switch.** By default the host-side CPU check is skipped: the benchmark prints `SKIP_VERIFY` (never PASS/FAIL) and its correctness is the golden check above. `HPCPERF_CPU_REFERENCE=1` runs the CPU reference (offline only: `hpcperf_golden.py make` uses it once to validate a new golden reference; no measurement, check or ctest path sets it, and nothing falls back to it); `HPCPERF_SKIP_VERIFY=1` (tools/timing) keeps it off as well. See [tools/golden](../../tools/golden/) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/backprop_cuda.cu`: one training step: GPU forward, host output layer/errors, GPU weight adjust. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 694 (6 source files, cloc, cuda/ + common/)
HIP: - (pending)
