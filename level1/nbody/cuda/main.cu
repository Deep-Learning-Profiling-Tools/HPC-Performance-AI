//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <iostream>
#include "GSimulation.hpp"
#include "GSimulationReference.hpp"
#include "hpcperf_golden.h"

#include <cstdlib>

// HPC-Performance-AI correctness switch. The host-side CPU reference (Verify(), common/
// GSimulationReference.hpp) runs ONLY on an explicit offline request (HPCPERF_CPU_REFERENCE=1;
// tools/golden/hpcperf_golden.py make uses it once to validate the frozen golden reference); by
// default it is skipped and correctness is judged by comparing the captured output
// (hpcperf_golden_output, HPCPERF_GOLDEN_DUMP) with the frozen B200 golden reference
// (tools/golden). No kernel, data initialization, tolerance or algorithm is touched.
static bool hpcperf_skip_verify() {
  return !hpcperf_cpu_reference_enabled();
}


int main(int argc, char** argv) {
  int n;      // number of particles
  int nstep;  // number ot integration steps

  GSimulation sim;

  if (argc > 1) {
    n = std::atoi(argv[1]);
    sim.SetNumberOfParticles(n);
    if (argc == 3) {
      nstep = std::atoi(argv[2]);
      if (nstep < 3) {
        std::cerr << "The number of integration steps should be at least 3.\n";
        return 1;
      }
      sim.SetNumberOfSteps(nstep);
    }
  }

  sim.Start();
  if (hpcperf_skip_verify())
    printf("SKIP_VERIFY\n");   // default: Verify() (it re-runs nsteps on the host) is skipped
  else
    sim.Verify();

  return 0;
}
