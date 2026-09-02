#pragma once
// src/pm/PoissonSolve.hpp
// k-space Poisson solver: Green's function application and gradient operator.
//
// ============================================================================
// STATUS: UNUSED SCAFFOLDING — NOT PART OF THE BUILT CODE.
//
// There is no PoissonSolve.cpp, this header is in no target's source list,
// and nothing in src/, apps/, or tests/ includes it.  The method bodies below
// were never written; their descriptions are followed by TODO markers.
//
// The Poisson solve that actually runs lives in src/pm/GreensFunction.{hpp,cpp}
// as two free functions over the Cabana::Grid phi array:
//   apply_greens_function()       — legacy diagnostic potential solve
//   apply_greens_and_gradient()   — production fused G(k) + spectral gradient
// and is orchestrated by src/pm/PMStep.cpp.  Prefer those.
//
// This file is retained only as a record of the original class-based design
// sketch (note it also assumes a flat k-space buffer rather than the
// Cabana::Grid Array the rest of the code settled on).  Do not extend it
// without first deciding whether it should exist at all.
// ============================================================================
//
// HACC reference:
//   - SolverDiscrete: nbody/dfft/solver.hpp:355-414
//   - 2nd-order discrete Green's function:
//       G(k) = 0.5 / (N^3 * (cos(2*pi*k0/n) + cos(2*pi*k1/n) + cos(2*pi*k2/n) - 3))
//       solver.hpp:396-413
//   - Gradient (imaginary part): sin(2*pi*k/n)  solver.hpp:391
//   - kspace_solve_gradient(): solver.hpp:287-315
//       grad_phi[idx] = complex_t(-gradient[k_axis]*green[idx]*imag(rho),
//                                  gradient[k_axis]*green[idx]*real(rho))
//   - k=0 handled: G(0) = 0  solver.hpp:411-413
//
// Spec: HACC_PM_IC_PORTING_SPEC.md §4 (Poisson Green's Function)
//       PORTING_GAP_ANALYSIS.md §4 Gap 2
//
// Design:
//   - Stores pre-computed m_green (G(k) per k-space flat index) and
//     m_gradient (sin(2*pi*k/n) per 1D wavenumber) as Kokkos::View.
//   - apply_green() and apply_gradient(axis) are device kernels.
//   - Operates on the heFFTe output buffer (complex<double>* or complex<float>*).
//   - No CIC deconvolution (matches HACC); TODO comment marks the extension point.

#include "../Types.hpp"
#include <Kokkos_Core.hpp>

namespace pmk {

class PoissonSolve {
public:
    // Initialize Green's function tables for an ng^3 grid.
    // local_lo[3] and local_dim[3] describe this rank's k-space pencil box
    // (obtained from heFFTe plan after construction).
    // HACC analogue: SolverDiscrete::initialize_greens_function() — solver.hpp:369-413
    PoissonSolve(int ng,
                 const int local_lo[3],
                 const int local_dim[3]);

    // Apply G(k) to k-space density buffer (potential solve).
    // phi_k[i] = G(k_i) * rho_k[i]
    // HACC: solver.hpp:261-277 (kspace_solve)
    // TODO §4: implement as Kokkos::parallel_for over k-space pencil.
    void apply_green(Kokkos::View<Kokkos::complex<double>*, MemorySpace> kbuf) const;

    // Apply G(k) and gradient operator for axis (0=x, 1=y, 2=z).
    // kbuf[i] = -i * sin(2*pi*k_axis/n) * G(k) * rho_k[i]
    // HACC: solver.hpp:287-315 (kspace_solve_gradient)
    // TODO §4: implement as Kokkos::parallel_for.
    void apply_green_gradient(int axis,
                               Kokkos::View<Kokkos::complex<double>*, MemorySpace> kbuf) const;

    // TODO §4: add optional CIC deconvolution factor W(k) = sinc(pi*k/n)^p
    //   (p=1 for CIC, p=2 for TSC). HACC omits this; scaffold lists it.

private:
    int m_ng;
    int m_local_lo[3];
    int m_local_dim[3];

    // G(k) for each k-space point in local pencil (flat index)
    // HACC: solver.hpp:329  std::vector<double> m_green;
    Kokkos::View<double*, MemorySpace> m_green;

    // sin(2*pi*k/n) for k = 0..ng-1 (gradient imaginary part, 1D)
    // HACC: solver.hpp:330  std::vector<double> m_gradient;
    Kokkos::View<double*, MemorySpace> m_gradient;

    void initialize_tables();
};

} // namespace pmk
