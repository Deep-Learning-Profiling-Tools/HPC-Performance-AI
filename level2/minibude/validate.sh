#!/usr/bin/env bash
# Validate miniBUDE against the reference energies shipped with the input decks.
#
#   ./validate.sh [CUDA|HIP]        (default: CUDA)
#
# miniBUDE validates itself: after each configuration it compares every computed pose energy with
# data/<deck>/ref_energies.out (produced by the full BUDE code) and reports
#   `outcome: { valid: true|false, max_diff_%: X }`
# where valid requires the maximum relative difference to be below DIFF_TOLERANCE_PCT = 0.025 %
# (entries with |ref| < 1 and |computed| < 1 are skipped, see src/main.cpp validate()). The process
# exit code is non-zero if any configuration is invalid.
#
# This script runs
#   1. deck bm1, wgsize 64, every compiled PPWI (1,2,4,8,16,32,64,128)  -> 8 kernel instantiations
#   2. deck bm2, wgsize 64, PPWI 2                                       -> the large ligand
# with a minimal iteration count and requires: exit code 0, no `valid: false`, and exactly one
# `valid: true` per configuration. Prints PASS/FAIL, exit code 0/1.
#
#   MINIBUDE_VALIDATE_BM2=0   skip the bm2 run (~7 s on a B200)
set -uo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SRC_DIR}/../.." && pwd)"
BACKEND="${1:-CUDA}"
BACKEND_UPPER="$(echo "${BACKEND}" | tr '[:lower:]' '[:upper:]')"

case "${BACKEND_UPPER}" in
  CUDA) EXE="${REPO_ROOT}/build/level2/minibude/cuda/cuda-bude" ;;
  HIP)  EXE="${REPO_ROOT}/build/level2/minibude/hip/hip-bude" ;;
  *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac

if [[ ! -x "${EXE}" ]]; then
  echo "validate.sh: ${EXE} not found -- run ./build.sh ${BACKEND_UPPER} first" >&2
  echo "miniBUDE ${BACKEND_UPPER} validation: FAIL"
  exit 1
fi

fail=0

# check <label> <expected-config-count> <bude args...>
check() {
  local label="$1" expected="$2"; shift 2
  local out rc n_true n_false
  echo "== ${label}: ${EXE} $*"
  out="$("${EXE}" "$@" 2>&1)"
  rc=$?
  echo "${out}" | grep -E -e '^# \(ppwi=' -e 'outcome:' -e '^best:' -e 'Verification failed' -e 'error' -e 'Error'
  n_true="$(echo "${out}" | grep -c 'valid: true')"
  n_false="$(echo "${out}" | grep -c 'valid: false')"
  if [[ ${rc} -eq 0 && ${n_false} -eq 0 && ${n_true} -eq ${expected} ]]; then
    echo "== ${label}: ok (${n_true}/${expected} configurations valid, max_diff_% within 0.025)"
  else
    echo "== ${label}: FAILED (exit=${rc}, valid=${n_true}/${expected}, invalid=${n_false})"
    fail=1
  fi
}

check "bm1 all PPWI" 8 --deck "${SRC_DIR}/data/bm1" -i 2 -w 64 -p all
if [[ "${MINIBUDE_VALIDATE_BM2:-1}" != "0" ]]; then
  check "bm2 PPWI 2" 1 --deck "${SRC_DIR}/data/bm2" -i 1 -w 64 -p 2
fi

if [[ ${fail} -eq 0 ]]; then
  echo "miniBUDE ${BACKEND_UPPER} validation: PASS"
  exit 0
else
  echo "miniBUDE ${BACKEND_UPPER} validation: FAIL"
  exit 1
fi
