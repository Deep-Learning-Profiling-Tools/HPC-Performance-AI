# Level 2 MPI helper tools

Shared helpers for the multi-rank / multi-GPU Level 2 runs. See the "MPI and
multi-GPU" section of [../README.md](../README.md) for the policy and
[../SCALEOUT_AUDIT.md](../SCALEOUT_AUDIT.md) for the per-app audit.

## `hpcperf_mpi_launch.sh` — GPU-count-aware launcher

```bash
hpcperf_mpi_launch.sh [--gpus N|all] [--launcher auto|mpirun|srun]
                      [--cpus-per-rank C] [--bind wrapper|app|none]
                      [--dry-run] -- <exe> [args...]
```

One MPI rank per GPU. Three resource views are kept apart:

- **actual** — the scheduler's allocation (nodes from Slurm, GPUs per node
  verified across node groups with `scontrol show job -d`, CPUs and task
  slots per node). Heterogeneous allocations are refused explicitly; no
  per-node uniformity is assumed unproven. Outside Slurm: the local node.
- **requested** — `--gpus`, plus an optional *subset* via `HPCPERF_NODES` /
  `HPCPERF_GPUS_PER_NODE`. Outside dry-run these can only shrink the
  allocation, and a node subset is enforced in the placement (`mpirun --host
  node:ranks,...`, `srun --nodelist`), not merely printed.
- **hypothetical** — dry-run only: larger `HPCPERF_NODES` /
  `HPCPERF_GPUS_PER_NODE` are accepted to plan a run elsewhere and the plan is
  labelled HYPOTHETICAL.

`--gpus N` launches exactly N ranks even if more GPUs are allocated, `--gpus
all` uses every requested GPU; ranks > GPUs fails fast (only
`HPCPERF_ALLOW_OVERSUBSCRIBE=1` overrides, with a "DEBUG ONLY" warning).
`HPCPERF_NP` is a compatibility alias compared *after* `all` is resolved; a
mismatch is an error. `HPCPERF_CPUS_PER_RANK=C` binds C cores per rank
(`--map-by ppr:R:node:PE=C --bind-to core`, or `--cpus-per-task=C` with srun)
after checking R x C <= CPUs per node; without it the runtime's default
binding applies and the per-rank CPU affinity is reported. Slurm task slots
smaller than the rank count (allocation requested with `-n 1`) are relaxed per
launch (`--map-by ...:OVERSUBSCRIBE`) only after both the GPU and the CPU
checks passed — that is slot bookkeeping, never GPU sharing, and it is
logged. Transport and mpirun-vs-srun come from the site profile
(`site/<profile>.sh`; `gmu-hopper` = mpirun + single-node `self,sm,smcuda`,
multi-node BLOCKED/UNVERIFIED). Environment defaults: `HPCPERF_GPUS`,
`HPCPERF_LAUNCHER`, `HPCPERF_CPUS_PER_RANK`, `HPCPERF_SITE_PROFILE`,
`HPCPERF_DRY_RUN=1`, `HPCPERF_BIND_OBSERVE=0` (skip sampling).

**GPU binding audit.** Each rank prints (via `mpi_gpu_bind.sh`)
`hpcperf-bind: host= grank= lrank= pid= mode= expected_gpu= expected_uuid=
observed=unverified cpus=`. The launcher samples `nvidia-smi
--query-compute-apps=pid,gpu_uuid` while the job runs and then prints per
rank `expected=<uuid> observed=<uuid|none> -> verified | MISMATCH |
unverified`, plus a summary. Observation is local to the launching node, so
ranks on other hosts stay `unverified`; very short runs can also end
`unverified` — that is reported as such, never as verified.

## `mpi_gpu_bind.sh` — scheduler-safe one-GPU-per-rank wrapper

```bash
mpirun -np 4 level2/tools/mpi_gpu_bind.sh <exe> <args...>   # normally injected by the launcher
```

Policy: local rank from `OMPI_COMM_WORLD_LOCAL_RANK`, else `SLURM_LOCALID`
(with neither, exec unchanged). Scheduler per-task binding
(`SLURM_GPUS_PER_TASK`) is respected untouched. If every rank inherits the
full `CUDA_VISIBLE_DEVICES` list (e.g. `2,3,6,7`), the rank takes its
local-rank-th entry — never an id outside the allocation. A local rank with
no dedicated GPU **fails (exit 12)** instead of silently sharing; only
`HPCPERF_ALLOW_OVERSUBSCRIBE=1` downgrades that to modulo reuse (loud
warning). GPU mode with no usable GPU (no devices, or an explicitly empty
`CUDA_VISIBLE_DEVICES`) **fails (exit 13)** unless `HPCPERF_ALLOW_NO_GPU=1`.
`CUDA_DEVICE_ORDER=PCI_BUS_ID` is exported so a numeric ordinal means the
same device to CUDA and to nvidia-smi; the expected UUID comes from
nvidia-smi's index/uuid table (a UUID entry is used as is). Apps that bind by
themselves (hypre, Kokkos/Cabana, hipBone) use `HPCPERF_BIND_REPORT_ONLY=1`
(audit line only, no narrowing).

## `hpcperf_topology.py` — balanced process grids

```bash
hpcperf_topology.py N [--dims 1|2|3] [--divides GX,GY,GZ] [--power-of-two] [--max-aspect R]
hpcperf_topology.py --self-test
```

Prints the most balanced `PX PY PZ` with `PX*PY*PZ == N` (40 -> `5 4 2`,
80 -> `5 4 4`). With `--divides` every axis permutation of each factorization
is tried (N=6 on `8,6,4` -> `2 3 1`). An infeasible N FAILS with nearby
feasible rank counts listed — it never silently changes N.

## `hpcperf_launch_common.sh` — run.sh helpers

Sourced by every `run.sh`: `hpcperf_ranks <app> <yes|no>` resolves
`HPCPERF_GPUS` (legacy alias `HPCPERF_NP`; disagreement is an error) and
rejects `HPCPERF_SCALE_MODE` for apps without size policies;
`hpcperf_forbid_args` rejects extra arguments that would override validated
parameters; `hpcperf_topology` propagates the helper's exit status.

## `mpi_cuda_check/` — CUDA-aware MPI runtime smoke test

`level2/tools/mpi_cuda_check/run.sh [NP]` (or `./check_env.sh --mpi-cuda`)
reports `MPIX_Query_cuda_support()` (the *requested* capability) and performs
a numerically checked device-buffer ring `Sendrecv` + `Allreduce` (the
*confirmed* runtime capability). It sets the single-node shared-memory
transport itself because the site-default UCX hangs on CUDA device buffers
on the dev node (`HPCPERF_MPI_CUDA_MCA` overrides).

## `tests/` — regression tests without GPU execution

`level2/tools/tests/run_all.sh` runs: the topology self-test; sandboxed
`setup_level2_deps.sh` marker/fingerprint tests (no patch / with patch /
unwritable dir / fingerprint mismatch / legacy / post-hoc stamp); launcher
resource-parsing dry-run tests; and the run.sh interface guards (needs built
apps; nothing is launched).
