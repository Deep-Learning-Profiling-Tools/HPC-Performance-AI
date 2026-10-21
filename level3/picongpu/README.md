# PIConGPU (dev) -- relativistic electromagnetic particle-in-cell on alpaka/CUDA

PIConGPU (Helmholtz-Zentrum Dresden-Rossendorf) is a 3D3V explicit electromagnetic particle-in-cell code written
against the alpaka abstraction (CUDA here): Yee-grid FDTD field solver, Boris/Vay-type pushers, Esirkepov
charge-conserving current deposition with configurable particle shapes, and a plugin system for in-situ
diagnostics. Physics, species, shapes, pusher and precision are compile-time template parameters (an "input set"
of `.param` files); grid, step count, boundaries and plugins are runtime arguments.

## Provenance

- Upstream: https://github.com/ComputationalRadiationPhysics/picongpu, branch `dev`, commit
  `93cba5b61b628659a6ec8181f5e2dd28b8006f29` (0.9.0-dev, 2026-10-20; the last release 0.8.0 predates the
  CUDA 13 / current alpaka support, so the identity is this commit). License GPLv3+ (PMacc LGPLv3+; `LICENSE.md`).
- Frozen source bundle (scheme 3, `provenance/source.lock.yaml`): `src/` = the upstream tree with its vendored
  `thirdParty/` (alpaka, mallocMC, nlohmann_json, toml11, cmake-modules, cuda_memtest; the `catch2` submodule of the
  unit tests and `docs/` excluded), **no patches**; `deps/boost_1_86_0.tar.gz` and `deps/gsl-2.8.tar.gz` = the
  unmodified upstream release tarballs of the two libraries the conda environment lacks (sha256-pinned `kind: file`
  components). Artifact `picongpu-hpcperf-l3-v1.tar.zst` (153 MB, unpublished until the maintainer publishes it);
  `tools/prepare_benchmark.sh level3 picongpu [--artifact FILE]` materializes it. `fetch.sh` (freeze time only)
  clones the pinned commit into `_upstream/level3/picongpu`.

## Build strategy: NATIVE (upstream pic-create / pic-build, one build per input set)

`build.sh [CUDA] [ID ...]`: (1) Boost (program_options, atomic; headers) with `b2` and GSL with autotools from the
bundled tarballs into `.deps/level3/picongpu/<profile>/install/{boost,gsl}` (`dependency_installs`); (2) for every
registered input set `inputs/<set>/` (the `.param` files and `cmakeFlags` of an upstream test / example),
`pic-create` (upstream's merge of the default parameters with the set's) into `build/level3/picongpu/<profile>/<set>`
and `pic-build -b cuda:<arch>` (= cmake on `src/include/picongpu` with `-DPIC_EXTENSION_PATH=<set>`,
`alpaka_ACC_GPU_CUDA_ENABLE`/`CUDA_ONLY_MODE`, extended lambdas, conda g++ 13.3 as the CUDA host compiler, the built
Boost through `Boost_DIR`), installed as `install/<set>/bin/picongpu` next to the set's content hash that `run.sh`
checks. Modification class A (cmake cache entries only). Build time on dgx003: Boost ~3 min, GSL ~1 min, each
input set ~10 min at 12 jobs.

## Execution model

One MPI rank per GPU through the common launcher (`--gpus 1 --bind wrapper`); every registered input fixes
`-d 1 1 1` (one device), so `run.sh` refuses `HPCPERF_GPUS` other than 1. The registered `args` are the program's
entire command line (device decomposition, grid `-g`, steps `-s`, periodicity, plugin periods); `run.sh` runs in
`build/level3/picongpu/<profile>/<run subdir>/<id>.np1/`, where the plugins write `fields_energy.dat`,
`chargeConservation.dat`, `e_energy_all.dat`; `run_manifest.txt` records the input set and its hash, the grid, the
step count, the executable and its hash.

## Registered inputs (`inputs.yaml`, `HPCPERF_PICONGPU_INPUT`)

| id | input set (upstream) | grid | steps | timed loop (B200) | correctness |
|---|---|---|---|---|---|
| khi-192x512 | tests/KHI_growthRate (electron-scale Kelvin-Helmholtz, gamma = 1.021 shear, 2-D with 12 cells in z) | 192 x 512 x 12 | 3000 | 76.4 s | growth rate of the Bx energy vs 1/(sqrt(8) gamma), 20 % (upstream criterion) |
| khi-384x1024 (default) | same set, 4 x the cells | 384 x 1024 x 12 | 3000 | 173.0 s | same |
| khi-768x2048 | same set, 16 x the cells | 768 x 2048 x 12 | 3000 | 611.3 s | same |
| lwfa-192x1024x192 | examples/LaserWakefield (3-D laser-driven wakefield; upstream's 1.cfg: 1 device, sliding window off) | 192 x 1024 x 192 | 2000 | 53.8 s | charge conservation + energy histories vs this build's frozen first run (self-baseline) |

The KHI test's physics is per cell (cell size, time step, shear velocity and density from the `.param` files), so
a larger grid is a larger box of the same plasma with the same fastest-growing mode: upstream's analytic criterion
applies at every size. The laser-wakefield example is upstream's documented single-GPU 3-D setup; its png /
openPMD / phase-space plugins (PNGwriter, openPMD-api: not available here) are replaced by the energy and
charge-conservation plugins.

## Timed region (`tools/timing/apptimers.py: picongpu`)

`calculation  simulation time: ... = <s>` -- `include/picongpu/simulation/control/SimulationStarter.hpp` times
`Simulation::startSimulation` (the time-step loop: particle push, current deposition, field solve, boundaries /
communication, the plugins at their periods) and prints it after the loop; the following `full simulation time`
line adds initialization (species loading, memory set-up) and shut-down and is not used. Steps from the final
progress line (`100 % = <step> | ...`). No NVTX range: the profiled run's device picture is whole-process context.

## Validation (`validate.sh`, upstream mechanism)

Upstream's `tests/KHI_growthRate/bin/ci.sh` builds the test input set, runs it (`-g 192 512 12 -s 3000
--fields_energy.period 10`) and judges `fields_energy.dat` with `lib/python/test/setups/ESKHI` (numpy /
matplotlib, absent here). `validate.sh` runs the registered `khi-192x512` input (that command line on that input
set) through `run.sh` and applies the ESKHI criterion through `check_khi_growth.py` (standard library only; the
physical parameters gamma = 1.021, BASE_DENSITY_SI = 1e25, DELTA_T_SI = 1.79e-16 x 0.86 from the set's param
files; theory 1/(sqrt(8) gamma), acceptance 0.2 as `config.py`). Result (2026-10-21, 1 GPU): PASS, growth-rate
deviation 0.070 (run 2026-10-21 09:45 UTC alongside another application's smoke run on the same GPU: the result is not a timing).

### Correctness of the registered inputs

KHI inputs: `check_khi_growth.py` = the upstream criterion (above) on a fresh run (`run.check.<id>`): the maximum
of Gamma(t) = 0.5 ln(Bx(t_{k+1}) / Bx(t_{k-1})) / (t_{k+1} - t_{k-1}) in units of the relativistic plasma
frequency within 20 % of 1/(sqrt(8) gamma). LWFA: `check_lwfa.py` against `reference/lwfa-192x1024x192.json`,
this unmodified build's own first run (a self-baseline, stated as such): the charge-conservation plugin's maximal
|div E - rho| (in macro-particle charges) at every sampled step within a factor of the reference and the total
field / electron kinetic energies within a relative tolerance recorded from the observed run-to-run spread
(charge_factor 1.1 and energy_rel 1e-3 from the observed spread of two realisations: 0.1 % in the charge deviation,
9.6e-5 in the energies -- 100 x and 10 x that spread; the campaign day's check run deviated by 1.001 and 2.1e-4). Three
results per input: timing SUCCESS (the timer line of a clean run), workload verification PASS
(`verify_registry_runs.py`: manifest input set hash, grid and steps, the registered argv, the `100 % = <steps>`
line, the launcher audit), scientific correctness PASS (`hpcperf_inputs.py check` + `verdict`).

## Results on dgx003 (B200, CUDA 13.2.78)

Campaign `l3-ext-2026-10-20` (2026-10-21 06:45-09:22 UTC, Slurm job 2732030, GPU `0000:9D:00.0` by UUID, final
Level 3 protocol: 0 warm-up + 3 clean + 1 nsys-profiled run per input, headline = median of the application timer,
spread = (max - min) / median with 10 % as the UNSTABLE threshold, no clock locking, strictly serial;
`docs/timing/index.html` has the current rows):

| input | timer median (s) | min / max (s) | spread | stable | profiled run (s) | run verification | correctness |
|---|---|---|---|---|---|---|---|
| khi-192x512 | 76.421 | 75.863 / 77.081 | 1.6 % | yes | 88.175 | PASS | PASS (growth-rate deviation 0.080 of the 0.2 acceptance) |
| khi-384x1024 (default) | 172.954 | 172.942 / 173.319 | 0.2 % | yes | 196.232 | PASS | PASS (0.037) |
| khi-768x2048 | 611.310 | 607.404 / 613.463 | 1.0 % | yes | 631.635 | PASS | PASS (0.107) |
| lwfa-192x1024x192 | 53.842 | 53.811 / 54.093 | 0.5 % | yes | 68.573 | PASS | PASS (charge-deviation ratio 1.001 of 1.1; energies 2.1e-4 of 1e-3) |

- Timing SUCCESS for all four (`calculation  simulation time` of every clean run; `verify_registry_runs.py` PASS 4/4:
  manifest input-set hash, grid and steps, the registered argv, the `100 % = <steps>` line, the launcher audit). The
  nsys trace shows the device busy 80-98 % of the loop (the LWFA run 80 %: many short kernels per step).
- The profiled run is 3-27 % slower than the clean median (khi-768x2048 3 %, lwfa 27 %); its device picture is the
  whole-process one (no NVTX range).
- Correctness (`run_checks.sh picongpu` = one fresh run of every input, then `hpcperf_inputs.py verdict`): all four
  PASS (2026-10-21 09:23-09:38 UTC). The KHI growth rate grows from the simulation's own noise seed and varies between
  realisations of the same input (deviations 0.04-0.13 across the smoke, campaign-day check and earlier runs);
  upstream's 20 % acceptance covers that, a tighter per-run bound would have no basis. The LWFA self-baseline
  rule (above) is used to 0.1 % / 21 %.

## Limitations

- One device per registered input; multi-GPU decompositions would be other registered inputs; multi-node
  BLOCKED/UNVERIFIED; no HIP build validated (upstream supports `pic-build -b hip`).
- The dev branch moves daily: the frozen commit is a snapshot; 0.8.0 (the last release) does not build with this
  toolchain (CUDA 13 / alpaka).
- The LWFA input has no analytic reference upstream (a demonstration setup by upstream's own description); its
  correctness is a self-baseline.
- Boost and GSL are built from the bundled tarballs (~4 min) because the conda environment ships neither.

## Source distribution

Scheme 3 (`level3/EXTERNAL_ARTIFACT_DESIGN.md`): `provenance/source.lock.yaml`, artifact
`picongpu-hpcperf-l3-v1.tar.zst` in the maintainer's local staging, `publish_status: unpublished`. LOC:
`provenance/LOC.md`.
