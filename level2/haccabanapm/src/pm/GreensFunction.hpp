#pragma once
// src/pm/GreensFunction.hpp
// 2nd-order discrete (finite-difference) Green's function multiply in k-space.
//
// HACC reference: nbody/dfft/solver.hpp:355-414 (SolverDiscrete) — specifically
// the formula at solver.hpp:403:
//     G(k) = coeff / (cos(2π k0/N) + cos(2π k1/N) + cos(2π k2/N) − 3)
// with coeff = 0.5 / N^3 in HACC, where the 1/N^3 factor compensates for
// HACC's unnormalized FFTW (no scaling on either forward or backward).
//
// This pmkokkos port uses the convention:
//     fft.forward(*phi, FFTScaleNone()) → in-place k-space data
//     apply_greens_function(*phi)        → multiply by G(k) with coeff = 0.5
//     fft.reverse(*phi, FFTScaleFull()) → 1/N^3 normalization here
// so the 1/N^3 is provided by the FFT scaling rather than baked into G,
// matching HACC end-to-end while keeping the kernel pure physics.
//
// The pole G(0,0,0) = 0 (HACC solver.hpp:411-413) is enforced explicitly.
// Wavenumber wrap-around is implicit in cos(2π k/N): cos((N-k)·2π/N) =
// cos(k·2π/N), so positive indices in [N/2, N-1] are the negative frequencies.

#include "../Grid.hpp"

namespace pmk {

// Multiply phi (FP64 complex, post-forward-FFT) by the discrete Green's
// function in place over the owned k-space cells of this rank.
//
// LEGACY DIAGNOSTIC PATH.  This produces the k-space potential phi_hat(k) =
// G(k) * rho_hat(k); a subsequent inverse FFT yields the real-space potential
// from which the legacy ScalarGradientG2P force gather takes an analytic
// gradient of the CIC basis.  That scheme exhibits the ~1%/decade low bias
// documented in analysis/references/pk_discrepancy_diagnosis.md.  Used by
// test_pm_step_two_body (the original prompt-08 force test that pinned the
// diagnostic correctness of the pipeline); production force gathers go
// through apply_greens_and_gradient + gather_force_value instead.
void apply_greens_function(Grid::phi_array_t& phi);

// HACC-equivalent spectral gradient applied to the rho_hat field in k-space.
// In one fused kernel: multiply by G(k) (the discrete Green's function above)
// AND by the per-axis derivative multiplier `-i · sin(2π k_axis / N)`.
// After this call, an inverse FFT yields the real-space gradient component
// along `axis`; a CIC value gather then produces the force contribution.
//
// Algorithm matches HACC nbody/dfft/solver.hpp:286-315
// (kspace_solve_gradient) and the m_gradient[k] = sin(kk*kstep) table built
// at solver.hpp:386-392.  The DC mode is set to (0, 0).
//
// The k-space multiplier expanded out (a + i b)·(−i sin(2πk/N))·G(k)
//   c   = -sin(2π k_axis / N) · G(k)
//   re' = -c · b
//   im' =  c · a
// matches HACC solver.hpp:305-308 exactly (with our G(k) absorbing 1/N^3 via
// the reverse-FFT scaling instead of baking it into the coefficient).
//
// axis: 0, 1, or 2 (selects the gradient-component direction).
void apply_greens_and_gradient(Grid::phi_array_t& phi, int axis);

} // namespace pmk
