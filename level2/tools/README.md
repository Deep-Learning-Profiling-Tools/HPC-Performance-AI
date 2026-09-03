# Level 2 MPI helper tools

Small shared helpers used by the multi-rank / multi-GPU Level 2 runs. See the
"MPI and multi-GPU" section of [../README.md](../README.md) for the policy.

## `mpi_gpu_bind.sh` — scheduler-safe one-GPU-per-rank wrapper

```bash
mpirun --oversubscribe -np 4 level2/tools/mpi_gpu_bind.sh <exe> <args...>
```

Binds each MPI rank to a distinct GPU for programs that do not do it
themselves (the MFEM apps Laghos/Remhos, and flag-selected backends like
CloverLeaf/TeaLeaf). hypre (AMG2023), the Kokkos/Cabana apps and Branson bind
by rank on their own and do not need it.

Policy: the local rank comes from `OMPI_COMM_WORLD_LOCAL_RANK`, falling back to
`SLURM_LOCALID`. If the scheduler already set `CUDA_VISIBLE_DEVICES` (e.g. a
Slurm allocation of `2,3,6,7`), it selects the *(local rank mod K)*-th entry of
that list — it never invents ids outside the allocation. Only when no
`CUDA_VISIBLE_DEVICES` is set does it fall back to physical index = local rank.
With no MPI-rank variable at all it execs the program unchanged.

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
