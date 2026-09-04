# Site profile: GMU Hopper, dgx003 B200 node (RHEL 10, conda Open MPI 5.0.10).
# Sourced by hpcperf_mpi_launch.sh; defines launcher + transport policy.

SITE_NAME="gmu-hopper"
# srun on dgx003 regularly fails with "Socket timed out"; mpirun (PRRTE) works.
SITE_LAUNCHER_DEFAULT="mpirun"

# Multi-node MPI on this site is BLOCKED/UNVERIFIED: the conda Open MPI's
# device-buffer traffic over the site UCX hangs even single-node, the working
# sm/smcuda transport is single-node only, and no cross-node GPU-aware
# transport has been validated. Needs a site-provided (or purpose-built)
# GPU-aware UCX/Open MPI, plus rebuilds of hypre/MFEM/Cabana/heFFTe on top.
SITE_MULTINODE_STATUS="blocked"
SITE_MULTINODE_MSG="multi-node MPI is BLOCKED/UNVERIFIED on gmu-hopper (validated transport self,sm,smcuda is single-node only; site UCX hangs on CUDA device buffers)"

# site_mpi_args <nnodes>: extra launcher arguments for the transport.
site_mpi_args() {
    if [ "$1" -le 1 ]; then
        # Validated single-node profile: shared-memory + CUDA IPC. The site
        # default (UCX) passes host traffic but hangs on CUDA device buffers.
        echo "--mca pml ob1 --mca btl self,sm,smcuda"
    else
        echo ""  # never inherit the single-node smcuda profile across nodes
    fi
}
