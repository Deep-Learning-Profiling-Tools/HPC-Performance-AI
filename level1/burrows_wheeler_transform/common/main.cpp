#include <string>
#include <vector>
#include <list>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include "bwt.hpp"
#include "hpcperf_golden.h"

// HPC-Performance-AI correctness switch. The host-side CPU reference below runs ONLY on an
// explicit offline request (HPCPERF_CPU_REFERENCE=1; tools/golden/hpcperf_golden.py make uses it
// once to validate the frozen golden reference); by default it is skipped and correctness is
// judged by comparing the captured output (hpcperf_golden_output, HPCPERF_GOLDEN_DUMP) with the
// frozen B200 golden reference (tools/golden). No kernel, data initialization, tolerance or
// algorithm is touched.
static bool hpcperf_skip_verify() {
  return !hpcperf_cpu_reference_enabled();
}


#define NOW std::chrono::high_resolution_clock::now()

std::string bwt_cpu(const std::string sequence) {
  const int n = sequence.size();
  const char* sequence_cstr = sequence.c_str();

  std::vector<int> table(n);

  for (int i = 0; i < n; i++){
    table[i] = i;
  }

  std::list<int> sorted_table(table.begin(), table.end());
  sorted_table.sort([sequence_cstr,n](const int& a, const int& b) -> bool {
    for(int i = 0; i < n; i++) {
      if(sequence_cstr[(a + i) % n] != sequence_cstr[(b + i) % n]) {
        return sequence_cstr[(a + i) % n] < sequence_cstr[(b + i) % n];
      }
    }
    return false;
  });

  std::string transformed_sequence;

  for(auto r = sorted_table.begin(); r != sorted_table.end(); ++r) {
    transformed_sequence += sequence_cstr[(n + *r - 1) % n];
  }
  return transformed_sequence;
}

int main(int argc, char const *argv[])
{
  const int N = (argc > 1) ? atoi(argv[1]) : 1E6;
  std::cout << "running a sample sequence of length " << N << std::endl;

  // initialize the alphabet used in bioinformatics
  std::string alphabet("ATCG");

  srand(123);
  char* sequence = (char*) malloc((N+1) * sizeof(char));

  for (int i = 0; i < N; i++) {
    sequence[i] = alphabet[rand() % alphabet.size()];
  }
  sequence[N] = ETX;

  const bool skip_verify = hpcperf_skip_verify();

  // host reference (may take a while); skipped in measurement mode
  auto start = NOW;
  std::string cpu_seq;
  if (!skip_verify) cpu_seq = bwt_cpu(sequence);
  auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(NOW - start);

  // device run
  start = NOW;
  auto gpu_seq = bwt(sequence);
  auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(NOW - start);
  // tools/golden: the checked output (the n transformed characters), after the timed device run
  hpcperf_golden_output("bwt", gpu_seq.data(), gpu_seq.size(), "u8");

  if (skip_verify) {
    // default: the host transform (bwt_cpu) is skipped; correctness = the captured output against
    // the frozen golden reference (tools/golden)
    std::cout << "Device time: " << gpu_time.count() << " ms" << std::endl;
    std::cout << "SKIP_VERIFY\n";
  } else {
  std::cout << "Host time: " << cpu_time.count() << " ms" << std::endl;
  std::cout << "Device time: " << gpu_time.count() << " ms" << std::endl;

  if(cpu_seq.compare(gpu_seq) == 0) {
    std::cout << "PASS\n";
  } else {
    std::cout << "FAIL\n";
  }
  }

  free(sequence);
  return 0;
}
