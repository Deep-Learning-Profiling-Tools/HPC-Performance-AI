// Minimal standalone driver for the Hetero-Mark PageRank benchmark (CUDA).
// Replaces the upstream command-line-option framework; the benchmark class,
// its CPU reference implementation, and Verify() are unchanged upstream code.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "pr_cuda_benchmark.h"
#include "benchmark_runner.h"
#include "time_measurement_impl.h"

static void usage(const char *prog) {
  fprintf(stderr, "Usage: %s -i <csr matrix file> [-m <max iterations>]\n", prog);
  exit(1);
}

int main(int argc, char **argv) {
  const char *input_file = nullptr;
  uint32_t max_iterations = 500;  // upstream default
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-i") && i + 1 < argc) { input_file = argv[++i]; continue; }
    if (!strcmp(argv[i], "-m") && i + 1 < argc) { max_iterations = strtoul(argv[++i], nullptr, 10); continue; }
    usage(argv[0]);
  }
  if (!input_file) usage(argv[0]);
  PrCudaBenchmark benchmark;
  TimeMeasurementImpl timer;
  BenchmarkRunner runner(&benchmark, &timer);
  benchmark.SetMaxIteration(max_iterations);
  benchmark.SetInputFileName(input_file);
  benchmark.SetQuietMode(true);
  runner.SetQuietMode(true);
  runner.SetVerificationMode(true);  // always verify GPU result against CPU reference
  runner.Run();
  return 0;
}
