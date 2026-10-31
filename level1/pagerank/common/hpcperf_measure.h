#ifndef HPCPERF_MEASURE_H
#define HPCPERF_MEASURE_H
//
// HPC-Performance-AI correctness switch.
//
// This benchmark validates by recomputing the GPU result on the CPU
// (BenchmarkRunner::Run -> Verify(), gated upstream by verification_mode_).
// That CPU reference runs ONLY on an explicit offline request
// (HPCPERF_CPU_REFERENCE=1; tools/golden/hpcperf_golden.py make uses it once to
// validate the frozen golden reference); by default verification stays off, the
// driver prints SKIP_VERIFY, and correctness is judged by comparing the outputs
// captured by CaptureGoldenOutputs() (hpcperf_golden_output, HPCPERF_GOLDEN_DUMP)
// with the frozen B200 golden reference (tools/golden).
//
// No kernel, data initialization, tolerance or algorithm is touched.
//
#include <cstdlib>
#include "hpcperf_golden.h"

inline bool hpcperf_skip_verify() {
  return !hpcperf_cpu_reference_enabled();
}

#endif  // HPCPERF_MEASURE_H
