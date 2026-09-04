#pragma once
// src/ic/WhiteNoise.hpp
// Gaussian white-noise field generation for the IC pipeline.
//
// HACC reference:  nbody/initializer/Initializer.cxx::indens_whitenoise
//                  (lines ~242-358, CBRNG branch).  CBRNG plumbing comes
//                  from external/cbrng/{CBRNG_Random,CBRNG_Uniform}.h plus
//                  external/Random123/threefry.h.
//
// Reproducibility (the acceptance criterion for prompt 10a):
//   - Same seed + same global ng yields a bit-identical real-space field
//     regardless of MPI rank count or topology.  This is delivered by
//     keying the per-cell CBRNG draw on the GLOBAL (jg*ng + kg) index
//     and the per-x-slab key derived from (seed, global x-index).
//   - Same seed yields a bit-identical field across CPU / OpenMP / SYCL
//     backends, because the draws are computed on the host (option (A)
//     in the prompt: host staging).  The device only stores the result.
//   - The legacy MT19937 (`UseCBRNG=false`) branch is intentionally not
//     ported — its rank-count-dependent seeding violates the property
//     above (PORTING_GAP_ANALYSIS.md §2 Gap 2).

#include "Grid.hpp"

namespace pmk {

// Fill grid.phi with a unit-variance, zero-mean Gaussian random field in
// REAL space.  phi(i,j,k,0) receives the Gaussian draw; phi(i,j,k,1) is
// set to 0.0.  After this returns, callers can run a forward FFT to move
// the field to k-space — see white_noise_to_kspace below.
//
// `seed`  global RNG seed (HACC indat.iseed()).  Bit-identical output
//         across all decompositions for the same (seed, grid.ng()).
// `phase_shift`  if true, multiply the Gaussian by -1.0 (HACC's pi
//                phase flip; Initializer.cxx:244-245).
void generate_white_noise_real_space(Grid& grid,
                                     unsigned long seed,
                                     bool phase_shift = false);

// Forward FFT phi (with FFTScaleNone — match HACC's symmetric
// ng^(-1.5) scaling, which is delivered by the √P(k) multiply in 10b)
// and zero the k=0 mode.  After this returns, grid.phi holds the
// unit-variance Gaussian field in k-space, ready for multiplication
// by sqrt(P(k)).
void white_noise_to_kspace(Grid& grid);

} // namespace pmk
