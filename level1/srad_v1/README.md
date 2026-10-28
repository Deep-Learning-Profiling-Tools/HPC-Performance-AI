# SRAD v1

Speckle-reducing anisotropic diffusion on an ultrasound image.

## Source

Source suite: Rodinia (cuda/srad/srad_v1, data/srad)
Upstream repository: https://github.com/HPC-FAIR/rodinia_3.1
Upstream commit: 366b283456506ef2fe2c2b7dc5e83e941ffc5524

## Changes relative to upstream

| Change | Class | Where |
|---|---|---|
| Input image path as an optional 5th argument (upstream hardcodes `../../../data/srad/image.pgm`) | packaging | `cuda/main.cu:146-161` |
| tools/timing ROI markers and the tools/golden output capture | pure insertions | `cuda/main.cu:24-25, 299, 431, 442` |
| Reduction remainder index (below) | D (bug fix, 2026-10) | `cuda/reduce_kernel.cu:93-99` |

**What.** In the branch for a partially filled last block of `reduce` the remainder loop read `d_sums[i]` / `d_sums2[i]` (upstream `cuda/srad/srad_v1/reduce_kernel.cu:93-96`); it now reads `d_sums[i*d_mul]` / `d_sums2[i*d_mul]`, where the kernel itself loads (`reduce_kernel.cu:25-26`) and stores (`:48-49, :68-69, :102-103`) element i of a reduction level. Nothing else changed: same kernels, launches, grid and work (two loads and two adds per remainder element, at other addresses).

**Why.** From the second reduction level on (`d_mul` = 512) upstream added first-level values (single pixels left in `d_sums` by `prepare`) instead of block sums. For 502 x 458 (Ne = 229916: 450 first-level blocks, then one block with nf = 450, df = 256) the image statistics summed 256 block sums plus 194 single pixels: total 206242 instead of 357030 at iteration 0, mean 0.897 (impossible: every pixel is exp(g/255) >= 1), q0sqr 0.851 instead of 0.0513 (16.6-28 times too large over the run). With that q0sqr the diffusion coefficient exceeds 1 at every pixel and is saturated to 1, so the unfixed program computed plain isotropic diffusion: its output is bit-identical to a build with c = 1 at all four registered inputs (B200, 2026-10-28).

**Upstream status** (fetched and cited 2026-10-28; nothing fixed, nothing reported). The same three lines are in Rodinia CUDA srad_v1 2.0.1, 2.4, 3.0 and 3.1 (UVA tarballs; HPC-FAIR/rodinia_3.1@366b283 `cuda/srad/srad_v1/reduce_kernel.cu:93-96`), Rodinia 3.1 OpenCL (`opencl/srad/kernel/kernel_gpu_opencl.cl:161-164`, 256-thread work-groups), yuhc/gpu-rodinia@9c10d3e (same lines), ROCm HIP-Examples (`rodinia_3.0/hip/srad/srad_v1/reduce_kernel.cu:90-93` at 77e50d4; the directory was removed in a75ae53, 2022), HeCBench (ORNL/HeCBench@23714d9: `src/srad-cuda/reduce_kernel.cu:93-96`, `src/srad-hip/reduce_kernel.cu:93-96`, `src/srad-sycl/kernel_reduce.sycl:80-83`, `src/srad-omp/main.c:272-275`) and the SYCL/DPC++ ports derived from them (zjin-lcf/Rodinia_SYCL, artecs-group/rodinia-dpct-dpcpp, silviaralcaraz/syclomatic-eval). With 256-thread blocks (Rodinia OpenCL, HeCBench) level 2 has nf = 131, df = 128: 3 pixels replace 3 block sums and q0sqr is about 5 % too large at iteration 0 instead of 17 times (CPU model of the kernel's indexing on the registered image). Not affected: Rodinia OpenMP srad_v1 and CUDA srad_v2, which sum on the host. GitHub issue/PR searches of HPC-FAIR/rodinia_3.1, yuhc/gpu-rodinia, ROCm/HIP-Examples, ORNL/HeCBench and zjin-lcf/HeCBench found no report; the remainder loops of HeCBench and HIP-Examples are unchanged since their import (later commits only reformat, rename or move the files). The 3.1 `__syncthreads()` ("Lingjie Zhang", `reduce_kernel.cu:30-32`) fixes a different defect, a missing barrier after the shared-memory load. Upstream has no HIP version of this benchmark; a HIP port must carry the same fix.

**Evidence** (B200, 2026-10-28; executed).
* Upstream's own CPU implementation, Rodinia 3.1 OpenMP srad_v1 (HPC-FAIR/rodinia_3.1@366b283 `openmp/srad/srad_v1/main.c`, sha256 bdaf2398...; built unmodified with GCC 13.3 `-O3 -fopenmp`, and once more with a pure-insertion dump of its float image before write_graphics), on exactly the registered image and parameters: the fixed CUDA float image is within 4.1e-4 / 7.6e-5 / 7.6e-5 grey levels (iter100 / iter500 / iter2000; the written int images differ by 1 at 2 / 0 / 1 pixels). Metric: max |difference| of the float output images in grey levels (before the int truncation of write_graphics) -- the strictest per-pixel comparison, and the probe's own metric; the registered `trunc_int` rule would accept much more. The unfixed CUDA build is within 0.034 / 0.0026 / 3.1e-4 as well: on the registered image c < 1 only at 124 pixels at iteration 0 and nowhere after the first iterations, so these inputs cannot tell SRAD from plain diffusion. On the correctness probe (below) the unfixed build differs from Rodinia OpenMP by 130 grey levels and the fixed build by 0.099 (`check --dump` of Rodinia's image against the probe reference: PASS); the remaining difference is Rodinia's statistic: `main.c:241-247` sums the 229916 pixels as a float32 recursive sum in row order (q0sqr 0.92 % too small at iteration 0 on the probe, 0.52 % on the registered image). The same program with `double sum, sum2` (one declaration changed) is within 1.0e-3 of the fixed CUDA image (q0sqr within 1.5e-4 relative), and a CUDA build that sums in Rodinia's order on one GPU thread reproduces Rodinia OpenMP within 9.1e-4 (probe) / 7.6e-5 (registered image): the CUDA kernels compute Rodinia's SRAD and the fix makes the statistic exact. On the round-3 probe image (texture +-8) Rodinia's row-order float32 sum was 64.5 (2.2e-4) too large, q0sqr 4.4-8.1 % off and the image 5.3 grey levels away -- one reason the probe texture was changed (below).
* The SRAD equations of Yu & Acton (IEEE TIP 11(11):1260-1270, 2002) -- (36) q0^2 = var/mean^2, (52)-(54) one-sided differences and Laplacian with h = 1, (55)-(56) and (59)-(60) replicated borders, (35)/(57) q^2 = [G2/2 - L^2/16] / (1 + L/4)^2 with G2 = (|grad_R I|^2 + |grad_L I|^2)/I^2 and L = lap(I)/I, (33) c = 1/(1 + (q^2 - q0^2)/(q0^2 (1 + q0^2))), (58) divergence with c_{i+1,j} and c_{i,j+1} on the south/east terms, (61) I += (dt/4) d with dt = lambda -- recomputed in float64 from the GPU's image at iterations 0 / 10 / 19 of the probe and 0 / 50 / 99 of iter100: the kernels' c is within 4.1e-7 and the updated image within 1.5e-7 (7e-8 relative) of the equations, q0sqr within 9e-5 relative of (36). Mapping: `cuda/main.cu:206-212, 375-378` (q0sqr), `prepare_kernel.cu`, `reduce_kernel.cu` (sums), `srad_kernel.cu:47-65` (differences, G2, L, q^2, c), `srad2_kernel.cu:41-50` (divergence, update), borders `main.cu:233-236`. Rodinia (CPU and CUDA alike) departs from the paper in four documented ways that the fix does not touch: c is saturated to [0, 1] (`srad_kernel.cu:67-73`, Rodinia's 2006.04 change; (33) alone gives c > 1 wherever q^2 < q0^2 -- 42-100 % of the pixels here, up to 1 + 1/q0^2 = 21-121 -- which with lambda = 0.5 puts lambda/4 times the sum of the four coefficients far above the explicit scheme's stability limit of 1: a build without the upper saturation turns iter100's output range 34..251 into -34..293 grey levels); q0 is taken from the whole image (`main.cu:206-212`) instead of a homogeneous region (36) or the analytic q0 exp(-rho t) (37); dt = lambda = 0.5 (the paper uses 0.05); and SRAD runs on exp(g/255) of the 8-bit image (`extract` / `compress`) rather than on raw intensities.
* verify.py (this repository's numpy re-implementation; float64 sums): fixed build within 6.1e-5 (registered image) / 9.7e-4 (probe); unfixed build 0.0334 / 0.00264 / 2.8e-4 / 130. q0sqr of the fixed build is within 3.3e-5 / 6.1e-5 / 8.3e-5 / 1.4e-4 (relative, worst iteration) of a float64 computation on the GPU's own image.
* Timed region: `cuda/main.cu` is unchanged -- the same kernel sequence per iteration (prepare, two reduce levels, two device-to-host copies, srad, srad2) and the same work; the fix only changes the addresses of the 2 x 194 remainder loads that one thread of the second-level reduce block performs per iteration. Measured interleaved on the B200 (two sessions, 5 + 10 runs per input, GPU 0 shared with another job): the reduce kernels (cudaEvents around the reduction loop) are 3-7 % faster with the fix (1.49 vs 1.57-1.60 ms over iter100, 7.38-7.41 vs 7.73-7.76 ms over iter500, 29.6-29.7 vs 30.7-30.8 ms over iter2000); the synchronized ROI shows no systematic change (fix/unfixed ratio of the minima 0.87-1.01, medians noisy) -- it is dominated by srad / srad2 and the per-iteration host round trip. The three performance inputs' registry entries, arguments and image are byte-identical; the probe is a new entry that is never timed.

**References.** iter100 / iter500 / iter2000 were frozen on 2026-10-26 from the unfixed build (image sha256 fedd6c91... / f4f62ce6... / 6ae1a0ba..., i.e. plain diffusion) and are replaced from the fixed build (`hpcperf_golden.py make --replace`; fixed B200 images sha256 76868d23... / 5fe05607... / 4d9d3d24..., identical in two independent scratch freezes); `probe-edges-n20` is frozen from the fixed build only (scratch freeze 2026-10-28: 3a5425c4...). The rule of the three performance inputs is unchanged.

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
