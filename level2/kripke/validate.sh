#!/usr/bin/env bash
# Validate the Kripke GPU build against its own CPU (Sequential) code path.
#
#   ./validate.sh [CUDA|HIP]        (default: CUDA)
#
# Kripke 1.2.8 has no built-in self-test / reference answer.  Every kripke.exe binary
# always contains the Sequential (host) RAJA architecture in addition to the GPU one
# (selected at run time via --arch), so this script runs the SAME small problem twice
#   --layout GDZ --groups 32 --legendre 4 --quad 32 --zones 16,16,16
#   --gset 1 --dset 8 --zset 1,1,1 --niter 10          (4,194,304 unknowns)
# once with --arch <BACKEND> and once with --arch Sequential, and compares the
# "iter N: particle count=<P_N>" line of every one of the 10 source iterations:
#   1. both runs must exit 0 and print "Solver terminated" and "END",
#   2. both must print exactly 10 particle counts, all finite and > 0,
#   3. for every N:  |P_N(gpu) - P_N(seq)| / |P_N(seq)| <= KRIPKE_VALIDATE_RTOL.
# The particle count is the global sum of the scalar flux (phi) weighted by zone volume,
# so it exercises the sweep, LTimes/LPlusTimes, scattering and population kernels end
# to end.  Kripke prints it with printf("%e"), i.e. 7 significant digits, so the
# default tolerance is 1e-6 (one unit in the last printed digit); on a B200 vs. GCC 13
# -O3 host code all 10 values agree to all printed digits (relative difference 0).
# Prints PASS/FAIL, exit code 0/1.
set -uo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SRC_DIR}/../.." && pwd)"
BACKEND="${1:-CUDA}"
BACKEND_UPPER="$(echo "${BACKEND}" | tr '[:lower:]' '[:upper:]')"

case "${BACKEND_UPPER}" in
  CUDA) EXE="${REPO_ROOT}/build/level2/kripke/cuda/kripke.exe" ;;
  HIP)  EXE="${REPO_ROOT}/build/level2/kripke/hip/kripke.exe" ;;
  *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac

if [[ ! -x "${EXE}" ]]; then
  echo "validate.sh: ${EXE} not found -- run ./build.sh ${BACKEND_UPPER} first" >&2
  echo "Kripke ${BACKEND_UPPER} validation: FAIL"
  exit 1
fi

RTOL="${KRIPKE_VALIDATE_RTOL:-1e-6}"
MPIRUN="${KRIPKE_MPIRUN:-mpirun -np 1}"
NITER=10
PROBLEM=( --layout GDZ --groups 32 --legendre 4 --quad 32 --zones 16,16,16
          --gset 1 --dset 8 --zset 1,1,1 --niter "${NITER}" )

fail=0

# run_case <arch>  -> prints the NITER particle counts (one per line) on stdout
run_case() {
  local arch="$1" out rc counts n
  echo "== ${MPIRUN} kripke.exe --arch ${arch} ${PROBLEM[*]}" >&2
  out="$(${MPIRUN} "${EXE}" --arch "${arch}" "${PROBLEM[@]}" 2>&1)"
  rc=$?
  echo "${out}" | grep -E "iter $((NITER - 1)):|Solver terminated|Throughput|error|Error" >&2
  counts="$(echo "${out}" | sed -n 's/^ *iter [0-9]*: particle count=\([^,]*\),.*/\1/p')"
  n="$(echo "${counts}" | grep -c .)"
  if [[ ${rc} -ne 0 ]] || ! echo "${out}" | grep -q "Solver terminated" \
     || ! echo "${out}" | grep -q "^END" || [[ "${n}" -ne ${NITER} ]]; then
    echo "== --arch ${arch}: FAILED (exit=${rc}, ${n}/${NITER} particle counts found)" >&2
    fail=1
  fi
  echo "${counts}"
}

P_GPU="$(run_case "${BACKEND_UPPER}")"
P_SEQ="$(run_case Sequential)"

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
  echo "Kripke ${BACKEND_UPPER} validation: PASS"
  exit 0
else
  echo "Kripke ${BACKEND_UPPER} validation: FAIL"
  exit 1
fi
