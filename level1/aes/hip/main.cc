// Minimal standalone driver for the Hetero-Mark AES-256 ECB encryption benchmark (HIP).
// Replaces the upstream command-line-option framework; the benchmark class,
// its CPU reference implementation, and Verify() are unchanged upstream code (plus the
// tools/golden capture calls; the CPU reference runs only with HPCPERF_CPU_REFERENCE=1).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "aes_hip_benchmark.h"
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
  AesHipBenchmark benchmark;
  TimeMeasurementImpl timer;
  BenchmarkRunner runner(&benchmark, &timer);
  benchmark.SetInputFileName(input_file);
  benchmark.SetKeyFileName(key_file);
  benchmark.SetQuietMode(true);
  runner.SetQuietMode(true);
  // default: the CPU reference (Verify) is skipped; correctness = the captured outputs against the
  // frozen golden reference (tools/golden). HPCPERF_CPU_REFERENCE=1 runs it (offline only).
  runner.SetVerificationMode(!hpcperf_skip_verify());
  if (hpcperf_skip_verify()) printf("SKIP_VERIFY\n");
  runner.Run();
  return 0;
}
