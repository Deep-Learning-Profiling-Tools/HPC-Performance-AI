# Site profile: generic fallback. No transport is forced (the MPI/site stack
# chooses); multi-node is permitted but has not been validated by this repo.
SITE_NAME="generic"
SITE_LAUNCHER_DEFAULT="auto"   # srun inside Slurm, mpirun otherwise
SITE_MULTINODE_STATUS="unverified"
SITE_MULTINODE_MSG="multi-node MPI has not been validated by this repository on this site"
site_mpi_args() { echo ""; }
