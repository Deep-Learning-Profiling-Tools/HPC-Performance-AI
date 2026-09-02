#!/usr/bin/env bash
# Run the ExaCMech orientation_evolution miniapp on the GPU (level2/exacmech).
#
#   ./run.sh [CUDA|HIP] [option_file]        (default: CUDA, generated standard problem)
#
# The miniapp takes a single "option file" (see miniapp/orientation_evolution.cxx):
#   line 1  quaternion file      ("#random N" -> N reproducible random orientations, seed 42)
#   line 2  material model       (evptn_FCC_A, evptn_BCC_A, evptn_BCC_MD, evptn_BCC_D, evptn_BCC_E, ...)
#   line 3  material property file
#   line 4  device               (CPU | OpenMP | GPU)
#   line 5  dt                   (optional, default 0.00025)
#   line 6  number of steps      (optional, default 60)
#   line 7  velocity gradient    (optional, default [[-0.5 0 0], [0 -0.5 0], [0 0 1.0]])
#
# Standard problem (no option_file given): upstream's miniapp_script.bash / cases/option_gpu.txt
# problem -- evptn_FCC_A crystal plasticity with cases/props.txt, axisymmetric velocity gradient
# diag(-0.5,-0.5,1.0), dt = 0.00025 -- scaled from 350,000 points x 60 steps to
# 1,000,000 random orientations x 1000 steps (25 % axial strain) so that it runs for
# ~10 s on a B200.  The option file is written to $R/build/level2/exacmech/run/.
# Every step prints the volume-averaged Cauchy stress ("Step# n Stress: s11 s22 s33 s23 s13 s12").
#
# Overrides for the generated problem:
#   EXACMECH_NQPTS   number of random orientations (default 1000000)
#   EXACMECH_NSTEPS  number of time steps          (default 1000)
#   EXACMECH_DT      time step                     (default 0.00025)
#   EXACMECH_MODEL   material model                (default evptn_FCC_A)
#   EXACMECH_PROPS   property file                 (default miniapp/cases/props.txt)
#   EXACMECH_DEVICE  CPU | OpenMP | GPU            (default GPU)
#
# An explicit option_file is run as-is with the working directory set to miniapp/, so
# upstream's own decks work unchanged:  ./run.sh CUDA cases/option_gpu.txt
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SRC_DIR}/../.." && pwd)"
BACKEND="${1:-CUDA}"
BACKEND_UPPER="$(echo "${BACKEND}" | tr '[:lower:]' '[:upper:]')"
shift || true

case "${BACKEND_UPPER}" in
  CUDA) EXE="${REPO_ROOT}/build/level2/exacmech/cuda/bin/orientation_evolution" ;;
  HIP)  EXE="${REPO_ROOT}/build/level2/exacmech/hip/bin/orientation_evolution" ;;
  *) echo "usage: $0 [CUDA|HIP] [option_file]" >&2; exit 2 ;;
esac

if [[ ! -x "${EXE}" ]]; then
  echo "run.sh: ${EXE} not found -- run ./build.sh ${BACKEND_UPPER} first" >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  if [[ -f "$1" ]]; then
    OPTION_FILE="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
  elif [[ -f "${SRC_DIR}/miniapp/$1" ]]; then
    # allow paths relative to miniapp/ (e.g. cases/option_gpu.txt) from any cwd
    OPTION_FILE="${SRC_DIR}/miniapp/$1"
  else
    echo "run.sh: option file '$1' not found" >&2
    exit 1
  fi
else
  RUN_DIR="${REPO_ROOT}/build/level2/exacmech/run"
  mkdir -p "${RUN_DIR}"
  NQPTS="${EXACMECH_NQPTS:-1000000}"
  QUAT_FILE="${RUN_DIR}/rand_quats_${NQPTS}.txt"
  OPTION_FILE="${RUN_DIR}/option_${BACKEND_UPPER,,}.txt"
  echo "#random ${NQPTS}" > "${QUAT_FILE}"
  {
    echo "${QUAT_FILE}"
    echo "${EXACMECH_MODEL:-evptn_FCC_A}"
    echo "${EXACMECH_PROPS:-${SRC_DIR}/miniapp/cases/props.txt}"
    echo "${EXACMECH_DEVICE:-GPU}"
    echo "${EXACMECH_DT:-0.00025}"
    echo "${EXACMECH_NSTEPS:-1000}"
    echo "[[-0.5 0 0], [0 -0.5 0], [0 0 1.0]]"
  } > "${OPTION_FILE}"
fi

echo "== ExaCMech ${BACKEND_UPPER}: ${EXE} ${OPTION_FILE}"
echo "== option file:"
sed 's/^/     /' "${OPTION_FILE}"
cd "${SRC_DIR}/miniapp"
exec "${EXE}" "${OPTION_FILE}"
