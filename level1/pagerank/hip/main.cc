// Minimal standalone driver for the Hetero-Mark PageRank benchmark (HIP).
// Replaces the upstream command-line-option framework; the benchmark class,
// its CPU reference implementation, and Verify() are unchanged upstream code (plus the
// tools/golden capture calls; the CPU reference runs only with HPCPERF_CPU_REFERENCE=1).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "pr_hip_benchmark.h"
#include "benchmark_runner.h"
#include "hpcperf_measure.h"
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
  PrHipBenchmark benchmark;
  TimeMeasurementImpl timer;
  BenchmarkRunner runner(&benchmark, &timer);
  benchmark.SetMaxIteration(max_iterations);
  benchmark.SetInputFileName(input_file);
  benchmark.SetQuietMode(true);
  runner.SetQuietMode(true);
  // default: the CPU reference (Verify) is skipped; correctness = the captured outputs against the
  // frozen golden reference (tools/golden). HPCPERF_CPU_REFERENCE=1 runs it (offline only).
  runner.SetVerificationMode(!hpcperf_skip_verify());
  if (hpcperf_skip_verify()) printf("SKIP_VERIFY\n");
  runner.Run();
  return 0;
}
