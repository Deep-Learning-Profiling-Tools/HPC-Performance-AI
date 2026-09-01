// Minimal standalone driver for the Hetero-Mark color histogram benchmark (HIP).
// Replaces the upstream command-line-option framework; the benchmark class,
// its CPU reference implementation, and Verify() are unchanged upstream code.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "hist_hip_benchmark.h"
#include "benchmark_runner.h"
#include "time_measurement_impl.h"

static void usage(const char *prog) {
  fprintf(stderr, "Usage: %s [-n <num colors>] [-x <num pixels>]\n", prog);
  exit(1);
}

int main(int argc, char **argv) {
  uint32_t num_colors = 256;   // upstream default
  uint32_t num_pixels = 65536; // upstream default
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-n") && i + 1 < argc) { num_colors = strtoul(argv[++i], nullptr, 10); continue; }
    if (!strcmp(argv[i], "-x") && i + 1 < argc) { num_pixels = strtoul(argv[++i], nullptr, 10); continue; }
    usage(argv[0]);
  }

  HistHipBenchmark benchmark;
  TimeMeasurementImpl timer;
  BenchmarkRunner runner(&benchmark, &timer);
  benchmark.SetNumColor(num_colors);
  benchmark.SetNumPixel(num_pixels);
  benchmark.SetQuietMode(true);
  runner.SetQuietMode(true);
  runner.SetVerificationMode(true);  // always verify GPU result against CPU reference
  runner.Run();
  return 0;
}
