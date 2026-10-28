# N-Body

All-pairs gravitational N-body simulation (GSimulation).

## Source

Source suite: HeCBench (src/nbody-cuda, src/nbody-hip)
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
cmake -S level1/nbody -B build/nbody/cuda -DBACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
cmake --build build/nbody/cuda
```

HIP (build configuration present, unverified without ROCm):

```bash
cmake -S level1/nbody -B build/nbody/hip -DBACKEND=HIP -DCMAKE_BUILD_TYPE=Release
cmake --build build/nbody/hip
```

## Run

```bash
./build/nbody/cuda/nbody_cuda 16000 10
```

## Validation

Golden reference (tools/golden): a plain run captures, after the timed loop, the final kinetic energy `kenergy` (f32 scalar -- the only quantity upstream's check compares) and the final particle state `pos` / `vel` (positions / velocities, 3 x n f32 each, particle-major x0 y0 z0 x1 ...; copied back from the device only when `HPCPERF_GOLDEN_DUMP` is set, `common/GSimulationCapture.hpp`). `hpcperf_golden.py check` compares them with the frozen B200 reference `reference/<input>.json`; each rule allows its tolerance minus d0, the golden's own deviation from the CPU reference (GSimulationReference.hpp, which captures `kenergy.cpu` / `pos.cpu` / `vel.cpu`), measured once when the reference was frozen (`tol - d0` for max_abs, `(tol - d0) / (1 + d0)` for max_rel). Every input has its own golden block in `inputs.yaml`:

| output | metric | n16000-s10 | n32000-s10 | n64000-s10 | d0 (B200 vs CPU), measured 2026-10-28 |
|---|---|---|---|---|---|
| `kenergy` | max_rel | 1e-4 | 1e-4 | 1e-4 | 0 / 0 / 0 |
| `pos` | max_abs | 1e-4 | 1e-4 | 1e-4 | 5.96e-8 / 1.19e-7 / 1.79e-7 |
| `vel` | max_abs | 2.0e-6 | 8.5e-6 | 4.1e-5 | 4.66e-9 / 2.24e-8 / 1.19e-7 |

The CPU reference (prints PASS/FAIL) runs only on the explicit offline request below. ctest runs `hpcperf_golden.py run` on the registry's default input (`n16000-s10`, the former ctest arguments); the `check:` block of `inputs.yaml` does the same for every registered input.

**Why the state, and these tolerances** (measured 2026-10-28 on a B200 with CUDA 13.2: 38 legitimate implementations, 27 wrong ones and 10 probes built from a scratch copy of the kernels, each run on all three inputs; the float-atomic ones and the race repeated up to 41 times). Upstream compares only the final kinetic energy, `|a - c| < 1e-3`: 1 f32 ulp at n16000 (KE 14571.5, ulp 9.8e-4) and bit equality at n32000 / n64000 (KE 1.10e6 / 1.03e8, ulp 0.125 / 8). That rule passes wrong programs with a bit-identical energy (the last position update skipped: positions off by 2.1e-3 / 8.6e-3 / 4.1e-2; every position and velocity negated; nothing computed and the energy printed) and fails 16 / 18 / 29 of the 38 legitimate implementations (any other order of the force or energy sums: at n64000 a reversed partner loop is 4 ulp off, a tree reduction of the energy 42, `--fmad=false` 3); the unmodified build fails it itself at n16000 with 6 or 8 steps (3 and 2 ulp; d0 = 0 at 10 steps is luck). The rules are scaled to each field: positions within 1e-4 of the unit box they start in (they stay inside [0.004, 0.996]), velocities within 1e-4 of the input's max |v| (golden 0.02057 / 0.08552 / 0.41395, rounded down), the energy within 1e-4 relative.
* Legitimate (largest value/limit of the check, `hpcperf_golden.py run` against scratch references: 0.038 / 0.059 / 0.10 at n16000 / n32000 / n64000; envelope: pos <= 1.3e-6, vel <= 5.8e-6 of max |v|, KE <= 1.0e-5 relative -- the last from a float-atomic energy sum, whose order varies run to run: 41 runs at n64000, mean 4.0e-6, standard deviation 2.2e-6): reversed, lane-split (4/8/32 lanes + shuffle), four-accumulator, tiled and float4-tiled partner loops (G factored out), each pair once with Newton's third law and float atomics, one fused kernel per step (ping-pong buffers), double / Kahan / mixed (f32 per tile, f64 across tiles) force accumulation, the softening added first, r^-3 as 1/(d2 sqrt(d2)), `1/sqrtf`, `rsqrtf` with a Newton step, `--fmad=false`, fast-math, an f64 integration state, a full double-precision run, tree / double / Kahan / atomic energy reductions, block sizes 128 / 1024 and combinations. A GPU build with the CPU's expression order, `1/sqrtf` and `--fmad=false` reproduces the CPU state and energy bit for bit, i.e. d0 is FMA contraction, `rsqrtf` and association order.
* Wrong (smallest value/limit 7.1 / 5.4 / 4.9): positions rounded to fp16 in the force (vel 4.8e-4 to 6.9e-4 of max |v| while the energy is only 4e-7 to 4e-6 off); the last position update skipped (pos, >= 20.6); all others >= 21: 0.1 % or 50 % wrong time step, one step fewer or more, no position update, explicit Euler, acceleration not reset between steps, only partners j > i, half the partners with doubled mass, own instead of partner mass, 1 % or 0.1 % of the partners skipped, cutoff at r = 1, softening 1e-4, G 1 % off, partners' positions from the previous step, fp16 positions / bf16 velocities stored, last particle never updated, one-Newton-step rsqrt, mirrored state, nothing computed, the energy summed over half the particles or from the previous step's velocities (KE 2e3 to 5e3). A fused kernel updating in place (a data race) fails by 6 to 50 when the race changes the result; in 8 of 27 runs it did not, and the output was bit-identical to the golden (nothing an output check can see).
* Probes (where the rules draw the line): positions rounded to 12 mantissa bits in the force fail (1.2 to 1.6), 14 / 16 / 19 bits pass (<= 0.42 / 0.094 / 0.011), i.e. the force must be accurate to ~1e-5 relative; an rsqrt of two Newton steps from the 0x5f3759df guess (~40 ulp) passes (<= 0.15); a 0.01 % wrong time step, G = 6.67430e-11 (CODATA 2018) instead of the benchmark's 6.67259e-11, softening 1.01e-3 and one partner skipped fail (>= 2.2 / 5.5 / 6.6 / 28); dropping one of the n energy terms passes (0.93 / 0.43 / 0.026: below the f32 reduction noise at these sizes).

The legitimate side keeps a 17x headroom on the state (the wrong side >= 4.9x) because the envelope is dominated by the order of the n-term partner sums (~sqrt(n) ulp per force, then amplified by the dynamics), the same class of rounding every architecture's compiler, FMA contraction and rsqrt produces; the 40-ulp rsqrt probe stays at 0.15. The energy keeps a 10x headroom over float-atomic reductions (the limit is ~40 standard deviations of their spread) and >= 2000x over the energy-only bugs (half the terms 0.5, stale velocities 0.2).

**Physical invariants (measured; reference-free, not part of the check).** The kick-drift update with pairwise antisymmetric softened forces conserves in exact arithmetic the total momentum P = sum m v (Newton's third law; the self term is 0), the uniform motion of the centre of mass X(t) = X0 + t P/M, and the angular momentum L = sum m x x v (central forces, and x_new x v_new = x x v_new); not the energy KE + PE (Plummer potential): the integrator's own O(dt) error moves it by -0.116 / -0.114 / -0.172 of the final KE, identically (to 3e-7) for every legitimate implementation. The f32 noise floors of the legitimate set are |dP| / sum m|v| <= 1.9e-8, COM <= 2.3e-9 (box units), |dL| / sum m|x x v| <= 2.0e-8, |KE printed - KE of the state| / KE <= 5.7e-6. At 10x those floors the invariants catch broken antisymmetry (only partners j > i, half partners, own mass, partial partner sets, last particle not updated, stale partner positions, bf16 velocities, the race when it manifests), the mirrored state (P, COM), wrong step counts, skipped drifts and nothing computed (COM -- only because the random initial velocities carry a net momentum: T P0/M = 1.0e-6 / 4.8e-6 / 3.3e-6), fp16 positions (L, 13x to 36x) and the energy-reduction bugs (KE consistency); they cannot see wrong constants (time step, G, softening, cutoff, a one-Newton-step rsqrt). Every candidate an invariant catches also fails the state rules, so the check does not use them; they stay the reference-free option for inputs whose chaotic state can no longer be compared with a golden (more steps). The potential energy is an O(n^2) host pass: 75 s per state with blocked numpy at n64000, 0.47 s with a 16-thread C loop; a 256 / 2048-row subset estimate is 0.9 % / 0.24 % off.

**Scope of the tolerances.** They hold for the registered 10-step inputs only: round-off grows chaotically (at n64000 a legitimate reordering differs by 5.8e-6 of max |v| after 10 steps, 1.9e-5 after 15, 2.3e-4 after 20 and 6.4e-2 after 40; at n16000 by <= 2e-6 after 40), so there is no benchmark-level golden spec -- an input added later has none, and `make` refuses it, until its tolerances are measured.

Run it via:

```bash
ctest --test-dir build/nbody/cuda --output-on-failure
```


**CPU reference switch.** By default the host-side CPU check is skipped: the benchmark prints `SKIP_VERIFY` (never PASS/FAIL) and its correctness is the golden check above. `HPCPERF_CPU_REFERENCE=1` runs the CPU reference and captures its own state for d0 (offline only: `hpcperf_golden.py make` uses it once to validate a new golden reference; no measurement, check or ctest path sets it, and nothing falls back to it); `HPCPERF_SKIP_VERIFY=1` (tools/timing) keeps it off as well. See [tools/golden](../../tools/golden/) and [tools/timing/README.md](../../tools/timing/README.md).

**Measurement markers.** The region of interest `tools/timing` measures is marked with `hpcperf_roi.h` in `cuda/GSimulation.cu` and the HIP port: all integration steps. Pure insertions; a no-op unless measuring, so ctest is unaffected. Placement rule: [tools/timing/roi/README.md](../../tools/timing/roi/README.md).

## LOC

CUDA: 362 (7 source files, cloc, cuda/ + common/)
HIP: 362
