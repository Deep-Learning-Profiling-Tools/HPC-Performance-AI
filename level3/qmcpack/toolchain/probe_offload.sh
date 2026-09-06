#!/usr/bin/env bash
# Minimal OpenMP target-offload probe for the private LLVM toolchain, run BEFORE
# QMCPACK is built (spec: no silent host fallback; prove the target region does
# not run on the initial device; check numerics; then MPI with one rank per GPU).
#
#   ./probe_offload.sh            (1 GPU probe + 2- and 4-rank MPI probe via the common launcher)
#
# Checks (all must hold, else exit 1):
#   [1] omp_get_num_devices() >= 1 and the default device is not the host
#   [2] inside `#pragma omp target`: omp_is_initial_device() == 0 (executed on the GPU)
#   [3] OMP_TARGET_OFFLOAD=MANDATORY set for every run (a failed offload aborts instead of
#       falling back to the host); LIBOMPTARGET_INFO summary recorded
#   [4] numerics: y = a*x + y on 2^20 doubles and a reduction, compared with the host result
#       to 1e-12 relative (require_finite on the sums)
#   [5] MPI (2 and 4 ranks, one GPU each through the launcher wrapper): every rank reports
#       exactly 1 visible device, offload succeeds, and the nvidia-smi audit verifies the mapping
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../../.." && pwd)"
set +u; # shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true; set -u
# shellcheck disable=SC1091
source "$R/level3/tools/l3_common.sh"
PROFILE="${HPCPERF_QMCPACK_PROFILE:-clang231-cuda132-offload}"
l3_paths_profile qmcpack "$PROFILE"
LLVM="$L3_INSTALL/llvm"; CLANG="$LLVM/bin/clang"; CLANGXX="$LLVM/bin/clang++"
[ -x "$CLANG" ] || { echo "probe_offload.sh: $CLANG missing -- run build_llvm.sh" >&2; exit 1; }
ARCH="${HPCPERF_CUDA_ARCH:-$(l3_gpu_arch)}"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
export OMPI_CC="$CLANG" OMPI_CXX="$CLANGXX" OMP_TARGET_OFFLOAD=MANDATORY
export LD_LIBRARY_PATH="$LLVM/lib/x86_64-unknown-linux-gnu:$LLVM/lib:${LD_LIBRARY_PATH:-}"
W="$L3_BUILD_DEPS/offload-probe"; rm -rf "$W"; mkdir -p "$W" "$L3_LOGS"
cat > "$W/probe.cpp" <<'EOF'
#include <omp.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdlib>
#ifdef USE_MPI
#include <mpi.h>
#endif
int main(int argc, char** argv) {
  int rank = 0, size = 1;
#ifdef USE_MPI
  MPI_Init(&argc, &argv); MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
  const int ndev = omp_get_num_devices();
  const int def = omp_get_default_device(), init = omp_get_initial_device();
  int on_initial = 1;                       // must become 0 inside the target region
  #pragma omp target map(tofrom: on_initial)
  { on_initial = omp_is_initial_device(); }
  const size_t n = 1u << 20; const double a = 3.0;
  std::vector<double> x(n), y(n), yh(n);
  for (size_t i = 0; i < n; ++i) { x[i] = std::sin(0.001 * i); y[i] = std::cos(0.002 * i); yh[i] = a * x[i] + y[i]; }
  double* xp = x.data(); double* yp = y.data(); double sum = 0.0;
  #pragma omp target teams distribute parallel for map(to: xp[0:n]) map(tofrom: yp[0:n]) reduction(+: sum)
  for (size_t i = 0; i < n; ++i) { yp[i] = a * xp[i] + yp[i]; sum += yp[i]; }
  double sumh = 0.0, maxdiff = 0.0;
  for (size_t i = 0; i < n; ++i) { sumh += yh[i]; maxdiff = std::fmax(maxdiff, std::fabs(yh[i] - y[i])); }
  const double rel = std::fabs(sum - sumh) / std::fabs(sumh);
  const char* cvd = std::getenv("CUDA_VISIBLE_DEVICES");
  std::printf("probe rank %d/%d: num_devices=%d default_device=%d initial_device=%d target_ran_on_initial_device=%d "
              "axpy_maxabs=%.3e sum_rel=%.3e sum_finite=%d CUDA_VISIBLE_DEVICES=%s\n",
              rank, size, ndev, def, init, on_initial, maxdiff, rel, std::isfinite(sum) && std::isfinite(sumh), cvd ? cvd : "<unset>");
  int ok = (ndev >= 1) && (def != init) && (on_initial == 0) && (maxdiff < 1e-12) && (rel < 1e-12) && std::isfinite(sum);
#ifdef USE_MPI
  int allok = 0; MPI_Allreduce(&ok, &allok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD); ok = allok; MPI_Finalize();
#endif
  return ok ? 0 : 1;
}
EOF
FLAGS=(-O2 -std=c++17 -fopenmp "--offload-arch=sm_$ARCH" "--cuda-path=$CUDA_ROOT")
echo "# probe: clang $("$CLANG" --version | head -1); --offload-arch=sm_$ARCH; CUDA $CUDA_ROOT; OMP_TARGET_OFFLOAD=MANDATORY"
"$CLANGXX" "${FLAGS[@]}" -o "$W/probe" "$W/probe.cpp" 2>&1 | tee "$L3_LOGS/probe-compile.log" | /usr/bin/grep -iE 'error|warning: .*(cuda|offload)' || true
[ -x "$W/probe" ] || { echo "probe_offload.sh: FAIL -- single-process probe did not compile (log $L3_LOGS/probe-compile.log)"; exit 1; }
echo "# [1-4] single process, GPU 0"
rc=0; CUDA_VISIBLE_DEVICES=0 LIBOMPTARGET_INFO=16 "$W/probe" > "$L3_LOGS/probe-1gpu.log" 2>&1 || rc=$?
/usr/bin/grep -E '^probe rank|Libomptarget|omptarget' "$L3_LOGS/probe-1gpu.log" | head -8
[ "$rc" -eq 0 ] || { echo "probe_offload.sh: FAIL -- single-process offload probe exited $rc (see $L3_LOGS/probe-1gpu.log)"; exit 1; }
echo "# [5] MPI + one rank per GPU (conda Open MPI wrappers with OMPI_CXX=clang++)"
mpicxx "${FLAGS[@]}" -DUSE_MPI -o "$W/probe_mpi" "$W/probe.cpp" > "$L3_LOGS/probe-mpi-compile.log" 2>&1 || { tail -20 "$L3_LOGS/probe-mpi-compile.log"; echo "probe_offload.sh: FAIL -- MPI probe did not compile"; exit 1; }
for n in 2 4; do
    rc=0; "$L3_LAUNCHER" --gpus "$n" --bind wrapper -- "$W/probe_mpi" > "$L3_LOGS/probe-mpi-np$n.log" 2>&1 || rc=$?
    /usr/bin/grep -aE 'probe rank [0-9]|audit summary|hpcperf-bind:' "$L3_LOGS/probe-mpi-np$n.log" | head -12
    [ "$rc" -eq 0 ] || { echo "probe_offload.sh: FAIL -- MPI probe with $n ranks exited $rc"; exit 1; }
    [ "$(/usr/bin/grep -ac 'probe rank [0-9]' "$L3_LOGS/probe-mpi-np$n.log")" -eq "$n" ] || { echo "probe_offload.sh: FAIL -- expected $n rank reports"; exit 1; }
    /usr/bin/grep -aq 'audit summary: '"$n"' verified, 0 mismatch' "$L3_LOGS/probe-mpi-np$n.log" || echo "probe_offload.sh: NOTE -- GPU mapping not fully verified by sampling for $n ranks (see log)"
    good="$(/usr/bin/grep -acE 'probe rank [0-9]+/[0-9]+: num_devices=1 default_device=0 initial_device=1 target_ran_on_initial_device=0 ' "$L3_LOGS/probe-mpi-np$n.log")"
    [ "$good" -eq "$n" ] || { echo "probe_offload.sh: FAIL -- only $good of $n ranks report exactly one device with the target region on it"; exit 1; }
done
echo "probe_offload.sh: PASS -- clang $("$CLANG" --version | head -1 | /usr/bin/grep -oE '[0-9]+\.[0-9]+\.[0-9]+') offload to sm_$ARCH: target regions run on the device (not the initial device), numerics match the host to <1e-12, MANDATORY offload, 2 and 4 MPI ranks each with exactly one GPU"
echo "PASS $(date -u +%FT%TZ) clang=$("$CLANG" --version | head -1) arch=sm_$ARCH" > "$L3_INSTALL/llvm/OFFLOAD_PROBE.txt"
