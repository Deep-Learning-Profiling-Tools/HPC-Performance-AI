// Minimal standalone driver for the Hetero-Mark FIR filter benchmark (CUDA).
// Replaces the upstream command-line-option framework; the benchmark class,
// its CPU reference implementation, and Verify() are unchanged upstream code.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "fir_cuda_benchmark.h"
#include "benchmark_runner.h"
#include "hpcperf_measure.h"
#include "time_measurement_impl.h"

static void usage(const char *prog) {
  fprintf(stderr, "Usage: %s [-x <num data per block>] [-y <num blocks>]\n", prog);
  exit(1);
}

int main(int argc, char **argv) {
  uint32_t num_data_per_block = 1024;  // upstream default
  uint32_t num_blocks = 1024;          // upstream default
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-x") && i + 1 < argc) { num_data_per_block = strtoul(argv[++i], nullptr, 10); continue; }
    if (!strcmp(argv[i], "-y") && i + 1 < argc) { num_blocks = strtoul(argv[++i], nullptr, 10); continue; }
    usage(argv[0]);
  }

  FirCudaBenchmark benchmark;
  TimeMeasurementImpl timer;
  BenchmarkRunner runner(&benchmark, &timer);
  benchmark.SetNumDataPerBlock(num_data_per_block);
  benchmark.SetNumBlock(num_blocks);
  benchmark.SetQuietMode(true);
  runner.SetQuietMode(true);
  runner.SetVerificationMode(!hpcperf_skip_verify());  // off in measurement mode
  if (hpcperf_skip_verify()) printf("SKIP_VERIFY\n");
  // HPCPERF_L1_PHASE_TIMING=1 prints the six-phase summary the upstream
  // BenchmarkRunner already computes (to stderr). Off by default.
  runner.SetTimingMode(getenv("HPCPERF_L1_PHASE_TIMING") != nullptr);
  runner.Run();
  return 0;
}
