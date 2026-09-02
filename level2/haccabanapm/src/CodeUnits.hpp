#pragma once
// src/CodeUnits.hpp
// Unit system, conversion factors, and Newton's G in code units.
//
// HACC reference:
//   - nbody/common/Domain.cxx:43-121          (initialize, all conversions)
//   - cosmotools/algorithms/halofinder/BasicDefinition.h:111  (RHO_C)
//   - nbody/simulation/TimeStepper.cxx:106-111 (phiscal = 1.5*omega_cb)
//
// Spec: HACC_PM_IC_PORTING_SPEC.md §1 (Code Units & Constants)
//       PORTING_GAP_ANALYSIS.md §1
//
// Design (vs HACC):
//   - HACC's `Domain` is a static singleton.  Here, a plain struct passed by
//     const-ref.  All factors are pure arithmetic of (rL, ng, oL, omega_cb) —
//     ported verbatim from Domain.cxx:54-119.
//   - Mixed precision: this struct is double; HACC stores them as float.  The
//     downstream code casts to POSVEL_T when it captures them in kernels.
//
// Newton's G in code units (the value most likely to be wrong):
//   The Poisson equation in HACC code units is
//       ∇²φ_grid = (3/2) Ω_cb (ρ/ρ_bar − 1)         (TimeStepper.cxx:109)
//   so the "effective G" the solver applies is the prefactor `phiscal`.
//   In dimensional terms this comes from
//       4πG ρ_crit Ω_cb a / H0²  →  (3/2) Ω_cb / a       (matter-only term)
//   The factor 3/2 is the dimensionless gravitational coupling in code units;
//   the entire scale dependence is absorbed into the rL/ng grid spacing and
//   H0=100·h km/s/Mpc velocity unit.  See newton_G_code() below.

#include <cmath>
#include <cstdint>

namespace pmk {

// Critical density today, M_sun · h^2 / Mpc^3.
// HACC: cosmotools/algorithms/halofinder/BasicDefinition.h:111
inline constexpr double RHO_C = 2.77536627e11;

// H0 in km/s/Mpc when h=1 (HACC convention; all velocities scale with this).
inline constexpr double H0_KM_S_MPC = 100.0;

// Dimensionless gravitational coupling in HACC code units.
// Poisson:  ∇²φ_grid = NEWTON_G_CODE · Ω_cb · (ρ − 1)
// Source:   TimeStepper.cxx:109  (m_phiscal = 1.5*m_omega_cb)
inline constexpr double NEWTON_G_CODE = 1.5;

struct CodeUnits {
    // Inputs
    double rL;        // box side length [Mpc/h]
    int    ng;        // PM grid dimension (ng^3 cells)
    double oL;        // overload width [Mpc/h]  (Domain.cxx:64)
    double omega_cb;  // Omega_cdm + Omega_baryon

    // Position
    double phys2grid_pos;  // cells per Mpc/h        (ng/rL,  Domain.cxx:54)
    double grid2phys_pos;  // Mpc/h per cell         (rL/ng)

    // Velocity (H0=100 km/s/Mpc)
    double phys2grid_vel;  // 1/(100*rL/ng)          Domain.cxx:58
    double grid2phys_vel;  // 100*rL/ng [km/s]       Domain.cxx:59

    // Potential
    double phys2grid_phi;  // (ng/rL)^2 / 100^2      Domain.cxx:61
    double grid2phys_phi;  // inverse

    // Mass
    double grid2phys_mass; // M_sun/h per particle   Domain.cxx:116-119
    double phys2grid_mass; // 1/grid2phys_mass

    // Overload snapped to integer grid cells (Domain.cxx:65-66)
    int    ng_overload;
    double oL_snapped;

    // Poisson normalization phiscal = 1.5 * omega_cb  (TimeStepper.cxx:109)
    double phiscal;

    explicit CodeUnits(double rL_, int ng_, double oL_, double omega_cb_);

    // Per-rank live grid count along one dimension (uniform decomposition).
    int ng_local_alive(int ng_global, int decomp_size) const {
        return ng_global / decomp_size;
    }

    // Per-rank total grid count: alive + ghost layers + 1.  Domain.cxx:89
    int ng_local_total(int ng_alive) const {
        return ng_alive + 2 * ng_overload + 1;
    }
};

// Convenience: Newton's G in code units (the prefactor on Ω_cb in the Poisson
// source).  Equal to NEWTON_G_CODE = 3/2.  Provided as a function so test code
// has a single, named entry point matching the HACC value.
inline double newton_G_code() { return NEWTON_G_CODE; }

} // namespace pmk
