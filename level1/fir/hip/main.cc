// Minimal standalone driver for the Hetero-Mark FIR filter benchmark (HIP).
// Replaces the upstream command-line-option framework; the benchmark class,
// its CPU reference implementation, and Verify() are unchanged upstream code (plus the
// tools/golden capture calls; the CPU reference runs only with HPCPERF_CPU_REFERENCE=1; the
// input seed is fixed to 1 unless HPCPERF_RANDOM_SEED asks otherwise, common/hpcperf_measure.h).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "fir_hip_benchmark.h"
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

  FirHipBenchmark benchmark;
  TimeMeasurementImpl timer;
  BenchmarkRunner runner(&benchmark, &timer);
  benchmark.SetNumDataPerBlock(num_data_per_block);
  benchmark.SetNumBlock(num_blocks);
  benchmark.SetQuietMode(true);
  runner.SetQuietMode(true);
  // default: the CPU reference (Verify) is skipped; correctness = the captured outputs against the
  // frozen golden reference (tools/golden). HPCPERF_CPU_REFERENCE=1 runs it (offline only).
  runner.SetVerificationMode(!hpcperf_skip_verify());
  if (hpcperf_skip_verify()) printf("SKIP_VERIFY\n");
  runner.Run();
  return 0;
}
