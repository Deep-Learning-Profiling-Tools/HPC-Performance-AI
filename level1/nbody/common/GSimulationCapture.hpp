// HPC-Performance-AI (tools/golden): capture of the final particle state. Not part of upstream HeCBench.
//
// Writes the positions and the velocities of a HOST copy of the Particle array as two f32 outputs of
// 3 x n elements each, particle-major (x0 y0 z0 x1 y1 z1 ...). A no-op unless HPCPERF_GOLDEN_DUMP is set
// (hpcperf_golden.h), so a plain run, ctest and the timed region are unaffected. acc is not captured
// (update_particles zeroes it every step) nor is mass (never modified).
#ifndef _GSIMULATION_CAPTURE_HPP
#define _GSIMULATION_CAPTURE_HPP

#include <cstddef>
#include <vector>
#include "hpcperf_golden.h"

static inline void hpcperf_capture_particle_state(const Particle* ps, int n, const char* pos_name,
                                                  const char* vel_name) {
  if (!hpcperf_golden_active()) return;
  std::vector<RealType> pos(3 * (std::size_t)n), vel(3 * (std::size_t)n);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < 3; ++k) {
      pos[3 * (std::size_t)i + k] = ps[i].pos[k];
      vel[3 * (std::size_t)i + k] = ps[i].vel[k];
    }
  hpcperf_golden_output(pos_name, pos.data(), pos.size(), "f32");
  hpcperf_golden_output(vel_name, vel.data(), vel.size(), "f32");
}

#endif
