# Shared runtime tools -- commonization proposal (not yet moved)

The GPU-count-aware launcher and its helpers currently live under
`level2/tools/` and are validated there:

| Tool | Role |
|---|---|
| `level2/tools/hpcperf_mpi_launch.sh` | allocation parser, `HPCPERF_GPUS` selection, CPU binding, site profiles, GPU binding audit |
| `level2/tools/mpi_gpu_bind.sh` | scheduler-safe one-GPU-per-rank wrapper |
| `level2/tools/hpcperf_topology.py` | process-grid helper with constraint checking |
| `level2/tools/hpcperf_launch_common.sh` | `run.sh` helpers (rank resolution, argument guards) |
| `level2/tools/site/*.sh` | site profiles (transport, launcher choice) |
| `level2/tools/tests/` | regression tests |

Level 3 uses the same semantics (`HPCPERF_GPUS=N|all`, `HPCPERF_NODES`,
`HPCPERF_GPUS_PER_NODE`, `HPCPERF_CPUS_PER_RANK`, `HPCPERF_SCALE_MODE`,
`HPCPERF_SITE_PROFILE`, `HPCPERF_DRY_RUN=1`) and MUST not fork them.

## Minimal commonization plan

1. **Now (this round):** Level 3 scripts reference the tools through one
   variable, `HPCPERF_RUNTIME_DIR` (default `level2/tools`), set in
   `level3/tools/l3_common.sh`. No file is moved or copied; Level 2 is not
   disturbed.
2. **Next (separate PR):** `git mv level2/tools/{hpcperf_mpi_launch.sh,
   mpi_gpu_bind.sh,hpcperf_topology.py,hpcperf_launch_common.sh,site,tests}
   tools/runtime/`, leave thin forwarding shims at the old paths for one
   release (`exec "$(dirname "$0")/../../tools/runtime/<name>" "$@"`), switch
   the `HPCPERF_RUNTIME_DIR` default to `tools/runtime`, re-run
   `tools/runtime/tests/run_all.sh` plus the Level 2 4-GPU smoke set before
   merging.
3. Only then remove the shims.

Until step 2 lands, `tools/runtime/` holds this note only.
