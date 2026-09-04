#!/bin/bash
# run.sh -- build (if needed) and run the CUDA-aware MPI runtime smoke test.
#
#   ./run.sh [NP]          NP ranks (default 2; each binds its own GPU)
#
# Reports MPIX_Query_cuda_support() (the compiled/"requested" capability) and
# a numerically-checked device-buffer MPI exchange (the "runtime capability
# confirmed" part). Exit 0 on PASS.
#
# Transport: this uses the single-node shared-memory profile
# (pml=ob1, btl=self,sm,smcuda) explicitly, because on the B200 dev node the
# site-default UCX transport HANGS on CUDA device buffers (host-side MPI over
# UCX is fine; device-buffer Sendrecv over UCX does not complete). Override
# with HPCPERF_MPI_CUDA_MCA="" to test whatever the default stack selects, or
# set it to your own --mca string. A hard timeout guards against a hang.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NP="${1:-2}"
TIMEOUT="${HPCPERF_MPI_CUDA_TIMEOUT:-90}"
MCA_DEFAULT="--mca pml ob1 --mca btl self,sm,smcuda"
MCA="${HPCPERF_MPI_CUDA_MCA-$MCA_DEFAULT}"

command -v mpicc >/dev/null || { echo "mpi_cuda_check: mpicc not found -- source hpcperf_env.sh first" >&2; exit 3; }
make -C "$HERE" >/dev/null 2>&1 || { echo "mpi_cuda_check: build failed" >&2; make -C "$HERE"; exit 3; }

OS=""; [ "$NP" -gt 1 ] && OS="--oversubscribe"
echo "mpi_cuda_check: mpirun $OS -np $NP ${MCA:-<default transport>} (timeout ${TIMEOUT}s)"
# shellcheck disable=SC2086
timeout "$TIMEOUT" mpirun $OS -np "$NP" $MCA "$HERE/mpi_cuda_check"
rc=$?
if [ "$rc" -eq 124 ]; then
  echo "mpi_cuda_check: TIMED OUT after ${TIMEOUT}s (device-buffer MPI did not complete on this transport)" >&2
fi
exit "$rc"
