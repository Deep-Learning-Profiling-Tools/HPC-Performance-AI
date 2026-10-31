#!/usr/bin/env bash
# Validate the ExaCMech GPU build (level2/exacmech).
#
#   ./validate.sh [CUDA|HIP]        (default: CUDA)
#
# Two modes (project decision: no CPU reference computation on a default path):
#
#   default  (HPCPERF_CPU_REFERENCE unset or 0) -- runs NOTHING on the CPU path: the GPU copy of the
#     deck below only (all 60 steps, exit 0), then the frozen-reference check of the registered
#     default input fcc-a-1m-1000:
#       python3 tools/inputs/hpcperf_inputs.py check level2/exacmech fcc-a-1m-1000 \
#               --out build/level2/exacmech/validate/frozen-reference-check
#     i.e. the GPU run of that input (run.sh) compared with its frozen B200 golden reference
#     (tools/golden, reference/<id>.json, inputs.yaml check:). Exit 0 of the check = PASS; anything
#     else (including "no frozen reference") is a FAIL -- never a fallback to a CPU/OpenMP run. The
#     registered check covers the CUDA build only.
#   offline  (HPCPERF_CPU_REFERENCE set to 1: an explicit request only, e.g. by
#     tools/golden/hpcperf_golden.py make) -- the original GPU-vs-CPU comparison, unchanged:
#
# Upstream ships no reference outputs, so the check is GPU-vs-CPU on upstream's own
# CPU deck, miniapp/cases/option_cpu.txt:
#   50,000 random orientations (seed 42, generated identically on the host for every device),
#   evptn_FCC_A, cases/props.txt, dt = 0.00025, 60 steps, velocity gradient diag(-0.5,-0.5,1.0).
# The same binary runs the deck once with device "CPU" (RAJA sequential, ~20-26 s on this
# machine's cores) and once with device "GPU" (only line 4 of the deck differs; the GPU copy is
# written under $R/build/level2/exacmech/validate/).  Every step prints the volume-averaged
# Cauchy stress, "Step# n Stress: s11 s22 s33 s23 s13 s12"; all 60 x 6 values are compared:
#   |gpu - cpu| <= RTOL * |cpu| + ATOL   with RTOL = 1e-5, ATOL = 1e-6 (MPa).
# The output is printed with 6 significant digits, so the tolerance amounts to "agrees to
# print precision"; the full-precision GPU and CPU results differ only by summation order.
# Both runs must also complete all 60 steps and exit 0.  Prints PASS/FAIL, exit code 0/1.
#
#   EXACMECH_REF_DEVICE=OpenMP   use the OpenMP execution path as the reference (faster; offline mode)
#   EXACMECH_RTOL / EXACMECH_ATOL  override the tolerances (offline mode)
set -uo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SRC_DIR}/../.." && pwd)"
BACKEND="${1:-CUDA}"
BACKEND_UPPER="$(echo "${BACKEND}" | tr '[:lower:]' '[:upper:]')"
RTOL="${EXACMECH_RTOL:-1e-5}"
ATOL="${EXACMECH_ATOL:-1e-6}"
REF_DEVICE="${EXACMECH_REF_DEVICE:-CPU}"
FROZEN_INPUT="fcc-a-1m-1000"     # inputs.yaml default_input
case "${HPCPERF_CPU_REFERENCE:-}" in
  ''|0*) CPU_REF=0 ;;
  *)     CPU_REF=1 ;;
esac

case "${BACKEND_UPPER}" in
  CUDA) EXE="${REPO_ROOT}/build/level2/exacmech/cuda/bin/orientation_evolution" ;;
  HIP)  EXE="${REPO_ROOT}/build/level2/exacmech/hip/bin/orientation_evolution" ;;
  *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2 ;;
esac

if [[ ! -x "${EXE}" ]]; then
  echo "validate.sh: ${EXE} not found -- run ./build.sh ${BACKEND_UPPER} first" >&2
  echo "ExaCMech ${BACKEND_UPPER} validation: FAIL"
  exit 1
fi

VAL_DIR="${REPO_ROOT}/build/level2/exacmech/validate"
mkdir -p "${VAL_DIR}"
DECK="${SRC_DIR}/miniapp/cases/option_cpu.txt"
REF_DECK="${VAL_DIR}/option_ref_${REF_DEVICE}.txt"
GPU_DECK="${VAL_DIR}/option_gpu.txt"
# only the device line (line 4) differs between the reference and GPU decks
[[ ${CPU_REF} -eq 1 ]] && sed "4s/^CPU\$/${REF_DEVICE}/" "${DECK}" > "${REF_DECK}"
sed '4s/^CPU$/GPU/' "${DECK}" > "${GPU_DECK}"

fail=0
cd "${SRC_DIR}/miniapp"   # the deck references ./cases/... relative to miniapp/

run_case() {  # run_case <label> <deck> <output>
  local label="$1" deck="$2" out="$3" rc nsteps
  echo "== ${label}: ${EXE} ${deck}"
  "${EXE}" "${deck}" > "${out}" 2>&1
  rc=$?
  grep -E "^(Execution Strategy|Number of qpts|Number of steps|Step# 1 |Step# 60 |Run time)" "${out}"
  nsteps="$(grep -c '^Step# ' "${out}")"
  if [[ ${rc} -ne 0 || ${nsteps} -ne 60 ]]; then
    echo "== ${label}: FAILED (exit=${rc}, step lines=${nsteps}/60)"
    tail -20 "${out}"
    fail=1
  fi
}

run_case "GPU run"                    "${GPU_DECK}" "${VAL_DIR}/gpu.out"

if [[ ${CPU_REF} -eq 1 ]]; then
echo "== CPU reference: offline mode (HPCPERF_CPU_REFERENCE requested): reference run with device ${REF_DEVICE} + GPU-vs-${REF_DEVICE} comparison"
run_case "reference run (${REF_DEVICE})" "${REF_DECK}" "${VAL_DIR}/ref.out"

if [[ ${fail} -eq 0 ]]; then
  echo "== comparing 60 steps x 6 stress components (|gpu-ref| <= ${RTOL}*|ref| + ${ATOL})"
  paste -d' ' <(grep '^Step# ' "${VAL_DIR}/ref.out") <(grep '^Step# ' "${VAL_DIR}/gpu.out") |
  awk -v rtol="${RTOL}" -v atol="${ATOL}" '
    function abs(x) { return x < 0 ? -x : x }
    {
      # fields: Step# n Stress: r1..r6 Step# n Stress: g1..g6
      if ($2 != $11) { printf("step mismatch: %s vs %s\n", $2, $11); bad++ }
      for (i = 0; i < 6; i++) {
        r = $(4+i); g = $(13+i); d = abs(g - r); tol = rtol*abs(r) + atol
        if (d >= maxd) { maxd = d; maxstep = $2; maxcomp = i+1 }
        if (d > tol) { printf("step %s component %d: ref %s gpu %s |diff| %g > %g\n", $2, i+1, r, g, d, tol); bad++ }
      }
      n++
    }
    END {
      printf("compared %d steps; max |gpu-ref| = %g (step %s, component %d)\n", n, maxd+0, maxstep, maxcomp)
      if (n != 60 || bad > 0) exit 1
    }'
  [[ $? -eq 0 ]] || fail=1
fi
else
  echo "== CPU reference: skipped (default; frozen reference check instead)"
  if [[ "${BACKEND_UPPER}" != CUDA ]]; then
    echo "== frozen reference check: the registered check (inputs.yaml, ${FROZEN_INPUT}) covers the CUDA build only; no frozen reference for ${BACKEND_UPPER} -- FAIL (the GPU-vs-CPU comparison runs only on an explicit offline request)"
    fail=1
  else
    CHECK_OUT="${VAL_DIR}/frozen-reference-check"
    echo "== frozen reference check: python3 tools/inputs/hpcperf_inputs.py check level2/exacmech ${FROZEN_INPUT} --out ${CHECK_OUT}"
    python3 "${REPO_ROOT}/tools/inputs/hpcperf_inputs.py" check "${SRC_DIR}" "${FROZEN_INPUT}" --out "${CHECK_OUT}"
    rc=$?
    if [[ ${rc} -ne 0 ]]; then
      echo "== frozen reference check ${FROZEN_INPUT}: FAIL (exit ${rc}; ${CHECK_OUT}/check.json)"
      fail=1
    else
      echo "== frozen reference check ${FROZEN_INPUT}: PASS"
    fi
  fi
fi

if [[ ${fail} -eq 0 ]]; then
  echo "ExaCMech ${BACKEND_UPPER} validation: PASS"
  exit 0
else
  echo "ExaCMech ${BACKEND_UPPER} validation: FAIL"
  exit 1
fi
