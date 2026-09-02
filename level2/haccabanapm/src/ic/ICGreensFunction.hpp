#pragma once
// src/ic/ICGreensFunction.hpp
// Continuum (spectral) Green's function with ik-gradient, applied in k-space
// to the unit-variance Gaussian field.  This is DISTINCT from the prompt-07
// discrete Green's function in src/pm/GreensFunction.* — the IC uses a
// continuum operator because the goal is to reproduce the input P(k), not to
// emulate a finite-difference Laplacian.
//
// HACC reference:
//   - solve_gravity (per-axis):     Initializer.cxx:550-622
//   - Nyquist wrap for k_i_signed:  Initializer.cxx:566-583
//   - phase factor & k=0 pole:      Initializer.cxx:584-608
//
// Spec: ../analysis/prompts/10b_implement_zeldovich_ic.md (Deliverable 3)
//
// For axis ∈ {0, 1, 2} and at each k-mode:
//     factor = (i · k_axis) · (-1/k²) · sqrt(P_cb(k_phys)) · ng^(-1.5)
// where k_axis is the signed wavenumber after Nyquist wrapping, k² is the
// sum of squared signed wavenumbers, and k_phys = (2π/rL) · sqrt(k²) in h/Mpc.
// Multiplication by i swaps real and imaginary components with a sign flip on
// the new imag part:  (re, im) → (-im, re), then scaled by the real factor
// |k_axis| · (-1/k²) · sqrt(P) · ng^(-3/2) (signs combined in the kernel).
//
// For axis == 3 (potential, is_potential=true) the i·k_axis prefactor is
// dropped, leaving:
//     factor = (-1/k²) · sqrt(P_cb(k_phys)) · ng^(-1.5)
// applied to both real and imaginary components (no swap).  HACC additionally
// scales the inverse-FFT'd potential by d_z·1.5·omega_cb in real space; that
// is left to the caller (InitialConditions.cpp) to keep this kernel pure
// k-space per the prompt's deliverable boundary.
//
// The k=0 (DC) global mode is set to (0, 0).

#include "../Grid.hpp"

namespace pmk {

class PowerSpectrum;

void apply_ic_kernel(Grid& grid, int axis,
                     const PowerSpectrum& pk,
                     double rL, double ng,
                     double d_z = 1.0,
                     double omega_cb = 0.0,
                     bool is_potential = false);

} // namespace pmk
