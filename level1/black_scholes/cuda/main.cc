// Minimal standalone driver for the Hetero-Mark Black-Scholes option pricing benchmark (CUDA).
// Replaces the upstream command-line-option framework; the benchmark class,
// its CPU reference implementation, and Verify() are unchanged upstream code.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "bs_cuda_benchmark.h"
#include "benchmark_runner.h"
#include "time_measurement_impl.h"

static void usage(const char *prog) {
  fprintf(stderr, "Usage: %s [-x <num elements>]\n", prog);
  exit(1);
}

int main(int argc, char **argv) {
  uint32_t num_elements = 1024;  // upstream default
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-x") && i + 1 < argc) { num_elements = strtoul(argv[++i], nullptr, 10); continue; }
    usage(argv[0]);
  }

  BsCudaBenchmark benchmark;
  TimeMeasurementImpl timer;
  BenchmarkRunner runner(&benchmark, &timer);
  benchmark.SetNumElements(num_elements);
  benchmark.SetActiveCPU(false);
  benchmark.SetGpuChunk(0);
  benchmark.SetQuietMode(true);
  runner.SetQuietMode(true);
  runner.SetVerificationMode(true);  // always verify GPU result against CPU reference
  runner.Run();
  return 0;
}
