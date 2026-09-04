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

One MPI rank per GPU. Detects the Slurm allocation (nodes, GPUs/node, task
slots, `CUDA_VISIBLE_DEVICES`); `--gpus N` launches exactly N ranks even if
more GPUs are allocated, `--gpus all` uses every allocated GPU; ranks >
allocated GPUs fails fast (only `HPCPERF_ALLOW_OVERSUBSCRIBE=1` overrides,
with a "DEBUG ONLY" warning). Environment defaults: `HPCPERF_GPUS`,
`HPCPERF_LAUNCHER`, `HPCPERF_CPUS_PER_RANK`, `HPCPERF_SITE_PROFILE`,
`HPCPERF_DRY_RUN=1` (plan only), `HPCPERF_NODES`/`HPCPERF_GPUS_PER_NODE`
(hypothetical-allocation planning). `HPCPERF_NP` is a compatibility alias and
must match `HPCPERF_GPUS` when both are set.

Transport and mpirun-vs-srun come from the site profile
(`site/<profile>.sh`, auto-selected; `gmu-hopper` uses mpirun with the
single-node `self,sm,smcuda` transport and marks multi-node
BLOCKED/UNVERIFIED). Slurm task-slot counts smaller than the rank count
(allocation requested with `-n 1`) are relaxed per launch via
`--map-by ...:OVERSUBSCRIBE` ONLY after the GPU check passed — that is slot
bookkeeping, never GPU sharing, and it is logged. Every rank prints an
auditable `hpcperf-bind: host= grank= lrank= gpu= uuid=` line.

## `hpcperf_topology.py` — balanced process grids

```bash
hpcperf_topology.py N [--dims 1|2|3] [--divides GX,GY,GZ] [--power-of-two] [--max-aspect R]
```

Prints the most balanced `PX PY PZ` with `PX*PY*PZ == N` (40 -> `5 4 2`,
80 -> `5 4 4`). Application constraints are checked; an infeasible N FAILS
with the nearby feasible rank counts listed — it never silently changes N.

## `mpi_gpu_bind.sh` — scheduler-safe one-GPU-per-rank wrapper

```bash
mpirun --oversubscribe -np 4 level2/tools/mpi_gpu_bind.sh <exe> <args...>
```

Binds each MPI rank to a distinct GPU for programs that do not do it
themselves (the MFEM apps Laghos/Remhos, and flag-selected backends like
CloverLeaf/TeaLeaf). hypre (AMG2023), the Kokkos/Cabana apps and Branson bind
by rank on their own and do not need it.

Policy: the local rank comes from `OMPI_COMM_WORLD_LOCAL_RANK`, falling back
to `SLURM_LOCALID`; with neither, the program is exec'd unchanged. If the
scheduler already bound GPUs per task (`SLURM_GPUS_PER_TASK`), that result is
respected untouched. If every rank inherits the full `CUDA_VISIBLE_DEVICES`
list (e.g. `2,3,6,7`), the rank takes its local-rank-th entry — never an id
outside the allocation. A local rank with no dedicated GPU **fails (exit
12)** instead of silently sharing; only `HPCPERF_ALLOW_OVERSUBSCRIBE=1`
downgrades that to modulo reuse (loud warning). Each rank emits an audit line
(`hpcperf-bind: host= grank= lrank= gpu= uuid=`, also appended to
`$HPCPERF_BIND_LOG` if set); `HPCPERF_BIND_REPORT_ONLY=1` audits without
narrowing (for apps that bind by themselves: hypre, Kokkos/Cabana, Branson).

## `mpi_cuda_check/` — CUDA-aware MPI runtime smoke test

```bash
level2/tools/mpi_cuda_check/run.sh [NP]     # or: ./check_env.sh --mpi-cuda [NP]
```

Builds a tiny MPI program (`make`) and runs it on `NP` ranks (default 2, each
bound to its own GPU). It reports `MPIX_Query_cuda_support()` — the compiled
capability that `OMPI_MCA_opal_cuda_support=true` *requests* — and then
performs an actual device-buffer ring `Sendrecv` plus an `Allreduce` over GPU
memory, checked numerically on the host: this is the part that *confirms* the
runtime capability. Exit 0 only if the query is 1 and both device-buffer
checks pass on every rank.

Transport note: it explicitly uses the single-node shared-memory profile
(`pml ob1`, `btl self,sm,smcuda`) with a hard timeout, because on the B200 dev
node the site-default UCX transport hangs on CUDA device buffers (host-side
MPI over UCX is fine). Set `HPCPERF_MPI_CUDA_MCA=""` to test the default stack
instead, or `HPCPERF_MPI_CUDA_MCA="--mca ..."` for a custom transport.
