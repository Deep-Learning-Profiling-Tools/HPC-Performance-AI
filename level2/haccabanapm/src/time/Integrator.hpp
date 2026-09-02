#pragma once
// src/time/Integrator.hpp
// DKD (drift-kick-drift) symplectic time integrator with the p = a^α stepping
// variable.  Owns the per-step cosmological scalars (aa, pp, adot, phiscal,
// fscal) and orchestrates one full step:
//
//   half_drift(τ/2) → advance_half_step → kick(τ) → advance_half_step
//                  → half_drift(τ/2)    → migrate
//
// HACC reference:
//   - State + initialize:     TimeStepper.cxx:46-74
//   - set_adot:               TimeStepper.cxx:93-103
//   - set_scal:               TimeStepper.cxx:106-111
//   - advanceHalfStep:        TimeStepper.cxx:114-122
//   - DKD step ordering:      driver_pm.cxx:449-500
//
// Habib 2011 HACC code-units note (analysis/references/hacc_units.pdf):
//   - Eq. 16: Poisson source coefficient is 3/2 · Ω_cb  → m_phiscal.
//   - Eq. 22: drift dx̃/dy = P̃ / (α·ȧ·p^(1+1/α))      → Drift kernel.
//   - Eq. 23: kick  dP̃/dy = − ∇φ̃ · m_phiscal · dt/dy / a → m_fscal.
//   - Eq. 25: α=1 specialization (the case used by all prompt 09 tests).

#include "../Cosmology.hpp"
#include "../Particles.hpp"
#include "../Grid.hpp"

namespace pmk {

class Integrator
{
  public:
    Integrator(const Cosmology& cosmo,
               double alpha,
               double a_in,
               double a_fin,
               int    nsteps);

    // Initialize stepping-variable state at a = a_in.  Computes pin, pfin,
    // tau, tau2 and resets aa/pp/adot/phiscal/fscal to their initial values.
    // Mirrors the trailing block of HACC TimeStepper ctor (lines 61-73).
    void initialize();

    // Advance pp by tau2; recompute aa, adot, fscal.
    // HACC TimeStepper.cxx:114-122  (advanceHalfStep).
    void advance_half_step();

    // One full DKD step, updating particles in place.
    //   half_drift(τ/2) → migrate → advance_half_step → kick(τ)
    //                  → advance_half_step → half_drift(τ/2) → migrate.
    // HACC driver_pm.cxx:449-500.
    //
    // Note the PRE-KICK migrate (the first of two).  It is required for
    // correctness, not defensive: this code uses strict ownership with no
    // overload buffers, so a particle that crosses a rank boundary during the
    // first half-drift would otherwise deposit its CIC stencil onto the wrong
    // rank's local mesh and receive a wrong kick force.  See
    // src/comm/Migrator.hpp and tests/test_migration_protects_scatter.cpp.
    //
    // `uniform_mass` is the per-particle mass passed to the PM mass deposit
    // (UniformMassP2G; see src/pm/MassAssign.hpp).
    //
    // STABILITY: τ is fixed at construction as (pfin - pin)/nsteps, and this
    // integrator has no sub-cycling and no CFL check.  Too few steps over a
    // wide a-window does not merely lose accuracy — the kick velocity can
    // reach box scale and the NEXT deposit indexes off the mesh (observed as
    // a GPU access fault).  Cosmological z=200 → z=0 evolution needs
    // O(100s) of steps; the validated reference run uses 500.
    void full_step(Particles& parts, Grid& grid, double uniform_mass);

    // Read-only accessors for tests, IO, and the PM driver.
    double a()      const { return aa; }
    double p()      const { return pp; }
    double z()      const { return 1.0 / aa - 1.0; }
    double a_dot()  const { return adot; }
    double f_scal() const { return fscal; }
    double phi_scal() const { return phiscal; }
    double tau_full() const { return tau; }
    double tau_half() const { return tau2; }
    int    nsteps_total() const { return nsteps; }
    int    step()  const { return current_step; }

    // Stepping-variable parameters.
    double alpha;
    double pin, pfin;
    double tau, tau2;
    int    nsteps;
    int    current_step;

    // Time-dependent state, updated each half-step.
    double aa;
    double pp;
    double adot;
    double phiscal;     // 1.5 · Ω_cb  (constant; computed once in initialize)
    double fscal;       // m_phiscal · dtdy / a   — recomputed each step.

    const Cosmology& cosmo;

  private:
    void set_adot();    // recompute adot from cosmology at current aa
    void set_scal();    // recompute phiscal/fscal at current aa, pp, adot
};

} // namespace pmk
