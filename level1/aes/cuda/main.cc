// Minimal standalone driver for the Hetero-Mark AES-256 ECB encryption benchmark (CUDA).
// Replaces the upstream command-line-option framework; the benchmark class,
// its CPU reference implementation, and Verify() are unchanged upstream code.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "aes_cuda_benchmark.h"
#include "benchmark_runner.h"
#include "hpcperf_measure.h"
#include "time_measurement_impl.h"

static void usage(const char *prog) {
  fprintf(stderr, "Usage: %s -i <plaintext hex file> -k <key hex file>\n", prog);
  exit(1);
}

int main(int argc, char **argv) {
  const char *input_file = nullptr;
  const char *key_file = nullptr;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-i") && i + 1 < argc) { input_file = argv[++i]; continue; }
    if (!strcmp(argv[i], "-k") && i + 1 < argc) { key_file = argv[++i]; continue; }
    usage(argv[0]);
  }
  if (!input_file || !key_file) usage(argv[0]);
  AesCudaBenchmark benchmark;
  TimeMeasurementImpl timer;
  BenchmarkRunner runner(&benchmark, &timer);
  benchmark.SetInputFileName(input_file);
  benchmark.SetKeyFileName(key_file);
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
