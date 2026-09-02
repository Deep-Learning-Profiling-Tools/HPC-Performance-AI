#!/usr/bin/env bash
# Run the standard miniBUDE benchmark problem.
#
#   ./run.sh [CUDA|HIP] [extra bude arguments...]      (default: CUDA)
#
# Default problem: the upstream `bm1` deck (65536 poses, 26 ligand / 938 protein atoms), 16 timed
# iterations (+2 warm-up) per configuration, work-group size 64, auto-tuning over all compiled
# poses-per-work-item values (PPWI = 1,2,4,8,16,32,64,128); miniBUDE prints one result block per
# configuration and a final `best:` line. Takes ~5 s on an idle B200 (the PPWI=64,128 configurations
# launch too few blocks to fill the GPU and dominate the total).
#
# Larger problem: MINIBUDE_DECK=bm2 (2672 ligand / 2672 protein atoms, ~300x more interactions;
# 1.5-2.1 s/iteration at PPWI=2 on a B200 -- combine with MINIBUDE_PPWI=2 MINIBUDE_ITERS=8 or so).
#
# Environment overrides (extra command-line arguments are appended last and override these):
#   MINIBUDE_DECK    bm1 | bm2 | /path/to/deck   (default: bm1)
#   MINIBUDE_ITERS   timed iterations           (default: 16)
#   MINIBUDE_WGSIZE  CSV list of work-group sizes (default: 64)
#   MINIBUDE_PPWI    CSV list of PPWI or `all`  (default: all)
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SRC_DIR}/../.." && pwd)"
BACKEND="${1:-CUDA}"
if [[ $# -gt 0 ]]; then shift; fi
BACKEND_UPPER="$(echo "${BACKEND}" | tr '[:lower:]' '[:upper:]')"

case "${BACKEND_UPPER}" in
  CUDA) EXE="${REPO_ROOT}/build/level2/minibude/cuda/cuda-bude" ;;
  HIP)  EXE="${REPO_ROOT}/build/level2/minibude/hip/hip-bude" ;;
  *) echo "usage: $0 [CUDA|HIP] [extra bude arguments...]" >&2; exit 2 ;;
esac

if [[ ! -x "${EXE}" ]]; then
  echo "run.sh: ${EXE} not found -- run ./build.sh ${BACKEND_UPPER} first" >&2
  exit 1
fi

DECK="${MINIBUDE_DECK:-bm1}"
case "${DECK}" in
  /*) DECK_DIR="${DECK}" ;;
  *)  DECK_DIR="${SRC_DIR}/data/${DECK}" ;;
esac
if [[ ! -f "${DECK_DIR}/poses.in" ]]; then
  echo "run.sh: input deck '${DECK_DIR}' not found (expected poses.in etc.)" >&2
  exit 1
fi

ITERS="${MINIBUDE_ITERS:-16}"
WGSIZE="${MINIBUDE_WGSIZE:-64}"
PPWI="${MINIBUDE_PPWI:-all}"

echo "== miniBUDE ${BACKEND_UPPER}: ${EXE}"
echo "== deck ${DECK_DIR}, iterations ${ITERS}, wgsize ${WGSIZE}, ppwi ${PPWI} $*"
exec "${EXE}" --deck "${DECK_DIR}" -i "${ITERS}" -w "${WGSIZE}" -p "${PPWI}" "$@"
