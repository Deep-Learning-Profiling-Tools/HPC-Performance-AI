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
// HPC-Performance-AI input seed (approved by the project owners, 2026-10-27).
//
// Upstream seeds the input generator from the clock (`unsigned int seed =
// time(NULL);` in FirBenchmark::Initialize), so every run filtered another random
// signal and no output could be frozen. The seed is now FIXED by default:
// kHpcperfDefaultSeed = 1 -- Hetero-Mark's own fixed seed (src/ep/ep_benchmark.h:
// kSeedInitValue = 1; upstream ships no seed for fir) and the C library's default
// rand() seed. The registered inputs (inputs.yaml params.seed) and their golden
// references are defined by it.
//   HPCPERF_RANDOM_SEED unset or empty   -> seed 1 (the registered inputs)
//   HPCPERF_RANDOM_SEED=time             -> seed = time(NULL) (upstream's behaviour)
//   HPCPERF_RANDOM_SEED=<0..4294967295>  -> that seed
//   anything else                        -> refused (exit 2), never replaced silently
// Every run prints the seed it used ("fir: input seed N (...)"). A seed other than
// 1 is marked "NON-DEFAULT SEED": its outputs differ from the golden references, so a
// golden check of such a run FAILs, and tools/golden make refuses to freeze one
// (inputs.yaml golden.cpu_check.fail_regex). Only the seed source changed; the
// generator (rand_r), its call order and the value mapping are upstream's.
//
// No kernel, tolerance or algorithm is touched.
//
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include "hpcperf_golden.h"

inline bool hpcperf_skip_verify() {
  return !hpcperf_cpu_reference_enabled();
}

constexpr unsigned int kHpcperfDefaultSeed = 1;

inline unsigned int hpcperf_input_seed(const char *bench) {
  const char *e = std::getenv("HPCPERF_RANDOM_SEED");
  const bool set = e != nullptr && *e != '\0';
  unsigned int seed = kHpcperfDefaultSeed;
  if (set) {
    if (std::strcmp(e, "time") == 0) {
      seed = static_cast<unsigned int>(time(NULL));
    } else {
      char *end = nullptr;
      errno = 0;
      const unsigned long long v = std::strtoull(e, &end, 10);
      if (e[0] < '0' || e[0] > '9' || *end != '\0' || errno != 0 || v > 0xFFFFFFFFull) {
        std::fprintf(stderr, "%s: HPCPERF_RANDOM_SEED='%s' is neither 'time' nor an unsigned 32-bit integer: refused\n",
                     bench, e);
        std::exit(2);
      }
      seed = static_cast<unsigned int>(v);
    }
  }
  if (seed == kHpcperfDefaultSeed)
    std::printf("%s: input seed %u (default%s%s)\n", bench, seed, set ? "; HPCPERF_RANDOM_SEED=" : "", set ? e : "");
  else
    std::printf("%s: input seed %u (NON-DEFAULT SEED, HPCPERF_RANDOM_SEED=%s: the registered inputs and their "
                "golden references use seed %u)\n", bench, seed, e, kHpcperfDefaultSeed);
  std::fflush(stdout);
  return seed;
}

#endif  // HPCPERF_MEASURE_H
