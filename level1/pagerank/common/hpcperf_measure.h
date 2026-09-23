#ifndef HPCPERF_MEASURE_H
#define HPCPERF_MEASURE_H
//
// HPC-Performance-AI measurement switch (default OFF, nothing changes without it).
//
// This benchmark validates by recomputing the GPU result on the CPU
// (BenchmarkRunner::Run -> Verify(), gated upstream by verification_mode_).
// That recomputation is correctness machinery, not part of the workload being
// timed, so tools/timing/measure_level1.sh sets HPCPERF_SKIP_VERIFY=1 and this
// driver then leaves verification off. ctest never sets the variable, so the
// default path -- and every correctness result -- is unchanged.
//
// Upstream code (common/benchmark_runner.*, common/*_benchmark.cc) is untouched.
//
#include <cstdlib>

inline bool hpcperf_skip_verify() {
  static const bool skip = []() {
    const char* e = std::getenv("HPCPERF_SKIP_VERIFY");
    return e != nullptr && *e != '\0' && *e != '0';
  }();
  return skip;
}

#endif  // HPCPERF_MEASURE_H
