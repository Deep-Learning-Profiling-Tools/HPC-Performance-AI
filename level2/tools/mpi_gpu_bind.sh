#!/bin/bash
# mpi_gpu_bind.sh -- scheduler-safe one-GPU-per-rank binding wrapper.
#
# Usage:  mpirun -np N .../level2/tools/mpi_gpu_bind.sh <exe> [args...]
#
# Needed for programs that do NOT map ranks to GPUs themselves (MFEM apps such
# as Laghos/Remhos default every rank to device 0; CloverLeaf/TeaLeaf select
# the device with a flag). hypre (AMG2023) and Kokkos/Cabana apps bind by
# local rank on their own and do not need this wrapper.
#
# Policy:
#   1. Local rank comes from OMPI_COMM_WORLD_LOCAL_RANK, falling back to
#      SLURM_LOCALID. Without either (not an MPI child), exec unchanged.
#   2. If the scheduler already set CUDA_VISIBLE_DEVICES (possibly a subset
#      like "2,3,6,7"), NEVER invent new ids: pick the (local rank mod K)-th
#      entry of that list, preserving the scheduler's allocation.
#   3. Only if CUDA_VISIBLE_DEVICES is unset fall back to the physical index
#      local_rank mod <device count> (count via nvidia-smi, else rank as-is).

lr="${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-}}"
if [ -z "$lr" ]; then
    exec "$@"
fi

if [ "${CUDA_VISIBLE_DEVICES+set}" = "set" ]; then
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        # scheduler explicitly hid all GPUs -- respect it, bind nothing
        exec "$@"
    fi
    IFS=',' read -r -a _devs <<< "$CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES="${_devs[$(( lr % ${#_devs[@]} ))]}"
else
    _n="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
    if [ "${_n:-0}" -gt 0 ]; then
        export CUDA_VISIBLE_DEVICES=$(( lr % _n ))
    else
        export CUDA_VISIBLE_DEVICES="$lr"
    fi
fi
exec "$@"
