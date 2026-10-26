# SRAD v1

Speckle-reducing anisotropic diffusion on an ultrasound image.

## Source

Source suite: Rodinia (cuda/srad/srad_v1, data/srad)
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
cmake -S level1/srad_v1 -B build/srad_v1/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/srad_v1/cuda
```

## Run

```bash
cd <run dir> && ./build/srad_v1/cuda/srad_v1_cuda 100 0.5 502 458 level1/srad_v1/data/image.pgm   # writes image_out.pgm
```

## Validation

Correctness = the float output image (`image` after the copy-back, captured as `image`, column-major Nr x Nc, i.e. before write_graphics truncates it to int in image_out.pgm) against the frozen B200 golden reference `reference/<input_id>.json`: the truncated pixels differ by at most 3 - d0 and at most 0.1 % - f0 of them differ by >= 1 (d0 / f0 = the golden's own deviation from the CPU re-implementation; rule `trunc_int` in `inputs.yaml` `golden:`). The added check `verify.py` (full numpy float32 re-implementation of the SRAD pipeline compared against image_out.pgm: max pixel diff <= 3, < 0.1 % pixels off by more than 1 -- the golden rule counts pixels off by >= 1, i.e. it is stricter) validated that reference once and runs only on request (below). The input image path was made a command-line argument (upstream hardcoded a relative path).

**Upstream reduction issue (not fixed; the golden freezes the current behaviour).** In `cuda/reduce_kernel.cu` the remainder loop of a partially filled last block (lines 93-96) reads `d_sums[i]` / `d_sums2[i]`; from the second reduction level on (`d_mul` = 512) the partial sums live at `d_sums[i*d_mul]`. For 502 x 458 (450 first-level blocks -> one second-level block with nf = 450, df = 256) the image statistics therefore add 194 raw pixel values instead of 194 block sums: q0sqr is ~0.85 instead of ~0.05. An emulation of that indexing reproduces the B200 image to 1e-4 grey levels, the exact statistics differ from it by up to 0.033; verify.py's tolerance does not detect it (it passes with max pixel diff 1).

```bash
ctest --test-dir build/srad_v1/cuda --output-on-failure   # golden run of the registry's default input
python3 tools/golden/hpcperf_golden.py run level1/srad_v1 [<input_id>]   # the same for any registered input
```

**Correctness switch.** The binary has no built-in CPU check (nothing to skip, so it prints no `SKIP_VERIFY`); its CPU reference is `verify.py`, which a plain run, ctest, `tools/inputs` and `tools/timing` never run and nothing falls back to. `tools/golden/hpcperf_golden.py make` runs it once, offline (`HPCPERF_CPU_REFERENCE=1`, `HPCPERF_GOLDEN_DUMP`), to validate the golden reference; verify.py then also captures its CPU reference array (`image.cpu`, same order) so make can measure d0. `HPCPERF_GOLDEN_DUMP=<dir>` makes the binary capture its output. See [tools/golden/README.md](../../tools/golden/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/main.cu`: extract, the SRAD iterations, compress. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 628 (12 source files, cloc, cuda/ + common/)
HIP: - (pending)
