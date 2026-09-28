#!/usr/bin/env bash
# GPU-vs-reference check of ExaCMech for one registered configuration (level2/exacmech).
#
#   ./check_reference.sh <nqpts> <nsteps> <model> [CUDA|HIP]
#
# The same comparison validate.sh makes on upstream's CPU deck, for the deck run.sh generates
# from a registered input (EXACMECH_NQPTS / EXACMECH_NSTEPS / EXACMECH_MODEL): the miniapp runs
# the configuration once with device GPU and once with the reference execution path
# (EXACMECH_REF_DEVICE, default OpenMP; CPU = RAJA sequential), from decks that differ only in
# the device line. "#random <nqpts>" orientations are generated identically on the host for
# every device (seed 42). Every step prints the volume-averaged Cauchy stress,
# "Step# n Stress: s11 s22 s33 s23 s13 s12"; all nsteps x 6 values must satisfy
#   |gpu - ref| <= RTOL * |ref| + ATOL      RTOL = 1e-5, ATOL = 1e-6 MPa   (validate.sh)
# i.e. agree to the printed 6 significant digits (the two paths differ only by summation
# order). Both runs must complete all nsteps steps and exit 0. Prints PASS/FAIL, exit 0/1.
# Outputs: $R/build/level2/exacmech/check/<model>_<nqpts>_<nsteps>/{gpu,ref}.out
set -uo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SRC_DIR}/../.." && pwd)"
NQPTS="${1:?usage: $0 <nqpts> <nsteps> <model> [CUDA|HIP]}"
NSTEPS="${2:?usage: $0 <nqpts> <nsteps> <model> [CUDA|HIP]}"
MODEL="${3:?usage: $0 <nqpts> <nsteps> <model> [CUDA|HIP]}"
BACKEND_UPPER="$(echo "${4:-CUDA}" | tr '[:lower:]' '[:upper:]')"
RTOL="${EXACMECH_RTOL:-1e-5}"
ATOL="${EXACMECH_ATOL:-1e-6}"
REF_DEVICE="${EXACMECH_REF_DEVICE:-OpenMP}"
DT="${EXACMECH_DT:-0.00025}"
for v in "$NQPTS" "$NSTEPS"; do
  case "$v" in ''|*[!0-9]*|0) echo "check_reference.sh: nqpts/nsteps must be positive integers" >&2; exit 2 ;; esac
done

case "${BACKEND_UPPER}" in
  CUDA) EXE="${REPO_ROOT}/build/level2/exacmech/cuda/bin/orientation_evolution" ;;
  HIP)  EXE="${REPO_ROOT}/build/level2/exacmech/hip/bin/orientation_evolution" ;;
  *) echo "usage: $0 <nqpts> <nsteps> <model> [CUDA|HIP]" >&2; exit 2 ;;
esac
if [[ ! -x "${EXE}" ]]; then
  echo "check_reference.sh: ${EXE} not found -- run ./build.sh ${BACKEND_UPPER} first" >&2
  echo "ExaCMech ${BACKEND_UPPER} reference check (${MODEL}, ${NQPTS} points, ${NSTEPS} steps, ref ${REF_DEVICE}): FAIL"
  exit 1
fi

VAL_DIR="${REPO_ROOT}/build/level2/exacmech/check/${MODEL}_${NQPTS}_${NSTEPS}"
mkdir -p "${VAL_DIR}"
QUAT_FILE="${VAL_DIR}/rand_quats_${NQPTS}.txt"
echo "#random ${NQPTS}" > "${QUAT_FILE}"
write_deck() {   # write_deck <device> <file> -- the deck run.sh generates, with the device line replaced
  {
    echo "${QUAT_FILE}"
    echo "${MODEL}"
    echo "${EXACMECH_PROPS:-${SRC_DIR}/miniapp/cases/props.txt}"
    echo "$1"
    echo "${DT}"
    echo "${NSTEPS}"
    echo "[[-0.5 0 0], [0 -0.5 0], [0 0 1.0]]"
  } > "$2"
}
write_deck GPU "${VAL_DIR}/option_gpu.txt"
write_deck "${REF_DEVICE}" "${VAL_DIR}/option_ref_${REF_DEVICE}.txt"

fail=0
cd "${SRC_DIR}/miniapp"   # the deck references ./cases/... relative to miniapp/

run_case() {  # run_case <label> <deck> <output>
  local label="$1" deck="$2" out="$3" rc nsteps
  echo "== ${label}: ${EXE} ${deck}"
  "${EXE}" "${deck}" > "${out}" 2>&1
  rc=$?
  grep -E "^(Execution Strategy|Number of qpts|Number of steps|Step# 1 |Step# ${NSTEPS} |Run time)" "${out}"
  nsteps="$(grep -c '^Step# ' "${out}")"
  if [[ ${rc} -ne 0 || ${nsteps} -ne ${NSTEPS} ]]; then
    echo "== ${label}: FAILED (exit=${rc}, step lines=${nsteps}/${NSTEPS})"
    tail -20 "${out}"
    fail=1
  fi
}

run_case "GPU run"                       "${VAL_DIR}/option_gpu.txt"              "${VAL_DIR}/gpu.out"
run_case "reference run (${REF_DEVICE})" "${VAL_DIR}/option_ref_${REF_DEVICE}.txt" "${VAL_DIR}/ref.out"

if [[ ${fail} -eq 0 ]]; then
  echo "== comparing ${NSTEPS} steps x 6 stress components (|gpu-ref| <= ${RTOL}*|ref| + ${ATOL})"
  paste -d' ' <(grep '^Step# ' "${VAL_DIR}/ref.out") <(grep '^Step# ' "${VAL_DIR}/gpu.out") |
  awk -v rtol="${RTOL}" -v atol="${ATOL}" -v want="${NSTEPS}" '
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
      if (n != want || bad > 0) exit 1
    }'
  [[ $? -eq 0 ]] || fail=1
fi

if [[ ${fail} -eq 0 ]]; then
  echo "ExaCMech ${BACKEND_UPPER} reference check (${MODEL}, ${NQPTS} points, ${NSTEPS} steps, ref ${REF_DEVICE}): PASS"
  exit 0
else
  echo "ExaCMech ${BACKEND_UPPER} reference check (${MODEL}, ${NQPTS} points, ${NSTEPS} steps, ref ${REF_DEVICE}): FAIL"
  exit 1
fi
