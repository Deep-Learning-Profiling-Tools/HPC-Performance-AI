#!/usr/bin/env bash
# Run the standard Kripke GPU benchmark problem.
#
#   ./run.sh [CUDA|HIP] [extra kripke.exe arguments...]      (default: CUDA)
#
# Default problem (single MPI rank, one GPU): the upstream-style GPU configuration
#   --arch <BACKEND> --layout GDZ --groups 64 --legendre 4 --quad 128 --zones 32,32,32
#   --gset 1 --dset 8 --zset 1,1,1 --niter 10
# i.e. 32^3 spatial zones x 64 energy groups x 128 directions = 268,435,456 unknowns,
# 10 source iterations of the Kobayashi 3i sweep problem.  Kripke prints per-iteration
# particle counts, a timing table and Figures of Merit (Throughput, Grind time, Sweep
# efficiency).  Solve ~0.9 s, ~3 s wall on an idle B200 (see README.md).
#
# Environment overrides (extra command-line arguments are appended last and override these):
#   KRIPKE_LAYOUT   DGZ | DZG | GDZ | GZD | ZDG | ZGD   (default: GDZ)
#   KRIPKE_ZONES    x,y,z zones                        (default: 32,32,32)
#   KRIPKE_GROUPS   energy groups                      (default: 64)
#   KRIPKE_QUAD     quadrature points (multiple of 8)  (default: 128)
#   KRIPKE_NITER    solver iterations                  (default: 10)
#   KRIPKE_NP       MPI ranks (needs --procs for >1; launch via level2/tools/hpcperf_mpi_launch.sh) (default: 1)
#   KRIPKE_MPIRUN   launcher                           (default: mpirun -np $KRIPKE_NP)
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SRC_DIR}/../.." && pwd)"
BACKEND="${1:-CUDA}"
if [[ $# -gt 0 ]]; then shift; fi
BACKEND_UPPER="$(echo "${BACKEND}" | tr '[:lower:]' '[:upper:]')"

case "${BACKEND_UPPER}" in
  CUDA) EXE="${REPO_ROOT}/build/level2/kripke/cuda/kripke.exe" ;;
  HIP)  EXE="${REPO_ROOT}/build/level2/kripke/hip/kripke.exe" ;;
  *) echo "usage: $0 [CUDA|HIP] [extra kripke.exe arguments...]" >&2; exit 2 ;;
esac

if [[ ! -x "${EXE}" ]]; then
  echo "run.sh: ${EXE} not found -- run ./build.sh ${BACKEND_UPPER} first" >&2
  exit 1
fi

LAYOUT="${KRIPKE_LAYOUT:-GDZ}"
ZONES="${KRIPKE_ZONES:-32,32,32}"
NGROUPS="${KRIPKE_GROUPS:-64}"   # (GROUPS is a bash builtin)
QUAD="${KRIPKE_QUAD:-128}"
NITER="${KRIPKE_NITER:-10}"
NP="${KRIPKE_NP:-1}"
MPIRUN="${KRIPKE_MPIRUN:-mpirun -np ${NP}}"

ARGS=( --arch "${BACKEND_UPPER}" --layout "${LAYOUT}"
       --groups "${NGROUPS}" --legendre 4 --quad "${QUAD}" --zones "${ZONES}"
       --gset 1 --dset 8 --zset 1,1,1 --niter "${NITER}" )

echo "== Kripke ${BACKEND_UPPER}: ${EXE}"
echo "== ${MPIRUN} kripke.exe ${ARGS[*]} $*"
exec ${MPIRUN} "${EXE}" "${ARGS[@]}" "$@"
