#!/usr/bin/env bash
# Cross-check one Kripke GPU run against the Sequential architecture of the same binary
# (level2/kripke), the comparison validate.sh makes, for a registered configuration.
#
#   ./check_sequential.sh <gpu_run.log> --zones X,Y,Z --groups G --quad Q --layout L --niter N [CUDA|HIP]
#
# <gpu_run.log> is the stdout of the GPU run of that configuration (run.sh through the
# registered input). The script runs `kripke.exe --arch Sequential` with the same problem
# arguments run.sh uses (--legendre 4 --gset 1 --dset 8 --zset 1,1,1) and compares the
# "iter N: particle count=<P_N>" line of every one of the N source iterations:
#   1. the GPU log and the Sequential run must contain "Solver terminated" and "END",
#   2. both must hold exactly N particle counts, all finite and > 0,
#   3. for every iteration  |P(gpu) - P(seq)| / |P(seq)| <= KRIPKE_VALIDATE_RTOL  (default 1e-6,
#      one unit in the last of the 7 printed significant digits; validate.sh).
# The particle count is the global sum of the scalar flux weighted by zone volume, so it
# exercises sweep, LTimes/LPlusTimes, scattering and population end to end.
# Prints PASS/FAIL, exit 0/1. The Sequential run is a single-core run of the whole problem:
# its cost grows with zones x groups x quadrature (not established for the registered sizes).
set -uo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SRC_DIR}/../.." && pwd)"
GPU_LOG="${1:?usage: $0 <gpu_run.log> --zones X,Y,Z --groups G --quad Q --layout L --niter N [CUDA|HIP]}"
shift
ZONES=""; NGROUPS=""; QUAD=""; LAYOUT=""; NITER=""; BACKEND_UPPER=CUDA
while [[ $# -gt 0 ]]; do
  case "$1" in
    --zones)  ZONES="$2"; shift 2 ;;
    --groups) NGROUPS="$2"; shift 2 ;;
    --quad)   QUAD="$2"; shift 2 ;;
    --layout) LAYOUT="$2"; shift 2 ;;
    --niter)  NITER="$2"; shift 2 ;;
    CUDA|cuda|HIP|hip) BACKEND_UPPER="$(echo "$1" | tr '[:lower:]' '[:upper:]')"; shift ;;
    *) echo "check_sequential.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
for v in "$ZONES" "$NGROUPS" "$QUAD" "$LAYOUT" "$NITER"; do
  [[ -n "$v" ]] || { echo "check_sequential.sh: --zones --groups --quad --layout --niter are all required" >&2; exit 2; }
done
[[ -f "$GPU_LOG" ]] || { echo "check_sequential.sh: GPU log $GPU_LOG not found" >&2; exit 2; }
RTOL="${KRIPKE_VALIDATE_RTOL:-1e-6}"
MPIRUN="${KRIPKE_MPIRUN:-mpirun -np 1}"
case "${BACKEND_UPPER}" in
  CUDA) EXE="${REPO_ROOT}/build/level2/kripke/cuda/kripke.exe" ;;
  HIP)  EXE="${REPO_ROOT}/build/level2/kripke/hip/kripke.exe" ;;
esac
LABEL="Kripke ${BACKEND_UPPER} vs Sequential (zones ${ZONES}, groups ${NGROUPS}, quad ${QUAD}, layout ${LAYOUT}, ${NITER} iterations)"
if [[ ! -x "${EXE}" ]]; then
  echo "check_sequential.sh: ${EXE} not found -- run ./build.sh ${BACKEND_UPPER} first" >&2
  echo "${LABEL}: FAIL"; exit 1
fi
OUT_DIR="${REPO_ROOT}/build/level2/kripke/check"
mkdir -p "${OUT_DIR}"
SEQ_LOG="${OUT_DIR}/sequential_${LAYOUT}_${ZONES//,/x}_g${NGROUPS}_q${QUAD}_n${NITER}.log"
PROBLEM=( --layout "${LAYOUT}" --groups "${NGROUPS}" --legendre 4 --quad "${QUAD}" --zones "${ZONES}"
          --gset 1 --dset 8 --zset 1,1,1 --niter "${NITER}" )

counts_of() {  # counts_of <log> <label> -> particle counts on stdout; sets fail on a bad log
  local log="$1" label="$2" counts n
  counts="$(sed -n 's/^ *iter [0-9]*: particle count=\([^,]*\),.*/\1/p' "${log}")"
  n="$(echo "${counts}" | grep -c .)"
  if ! grep -q "Solver terminated" "${log}" || ! grep -q "^END" "${log}" || [[ "${n}" -ne ${NITER} ]]; then
    echo "== ${label}: incomplete (${n}/${NITER} particle counts, Solver terminated / END present: $(grep -c 'Solver terminated' "${log}")/$(grep -c '^END' "${log}"))" >&2
    fail=1
  fi
  echo "${counts}"
}

fail=0
echo "== ${MPIRUN} kripke.exe --arch Sequential ${PROBLEM[*]}   (log: ${SEQ_LOG})"
${MPIRUN} "${EXE}" --arch Sequential "${PROBLEM[@]}" > "${SEQ_LOG}" 2>&1
rc=$?
grep -E "iter $((NITER - 1)):|Solver terminated|Throughput|error|Error" "${SEQ_LOG}" | sed 's/^/   /'
if [[ ${rc} -ne 0 ]]; then
  echo "== Sequential run exited ${rc}"; fail=1
fi
P_GPU="$(counts_of "${GPU_LOG}" "GPU log")"
P_SEQ="$(counts_of "${SEQ_LOG}" "Sequential run")"

if [[ ${fail} -eq 0 ]]; then
  if ! paste <(echo "${P_GPU}") <(echo "${P_SEQ}") | awk -v tol="${RTOL}" -v name="${BACKEND_UPPER}" '
      {
        a = $1 + 0; b = $2 + 0; n++;
        if (!(a > 0) || !(b > 0)) { printf("== iter %d: non-positive/non-finite particle count (%s vs %s)\n", n - 1, $1, $2); bad++; next }
        rel = (a > b ? a - b : b - a) / b;
        if (rel > maxrel) maxrel = rel;
        if (rel > tol) { printf("== iter %d: %s=%s Sequential=%s rel=%.3e > %s\n", n - 1, name, $1, $2, rel, tol); bad++ }
      }
      END {
        printf("== final particle count: %s=%s Sequential=%s\n", name, $1, $2);
        printf("== %d iterations compared, max relative difference = %.3e (tolerance %s)\n", n, maxrel, tol);
        exit (bad > 0 ? 1 : 0)
      }'; then
    fail=1
  fi
fi

if [[ ${fail} -eq 0 ]]; then
  echo "${LABEL}: PASS"; exit 0
else
  echo "${LABEL}: FAIL"; exit 1
fi
