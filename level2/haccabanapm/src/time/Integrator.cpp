// src/time/Integrator.cpp
// DKD time-integrator state and orchestration.  See Integrator.hpp for the
// equation numbering (Habib 2011 HACC code-units note, Eqs. 16, 22, 23, 25).

#include "Integrator.hpp"

#include "Drift.hpp"
#include "../comm/Migrator.hpp"
#include "../pm/PMStep.hpp"

#include <cmath>

namespace pmk {

Integrator::Integrator(const Cosmology& cosmo_,
                       double alpha_,
                       double a_in_,
                       double a_fin_,
                       int    nsteps_)
    : alpha(alpha_),
      pin(0.0), pfin(0.0),
      tau(0.0), tau2(0.0),
      nsteps(nsteps_),
      current_step(0),
      aa(a_in_),
      pp(0.0),
      adot(0.0),
      phiscal(0.0),
      fscal(0.0),
      cosmo(cosmo_)
{
    // pin/pfin and the rest of the derived state are populated by initialize().
    pin  = std::pow(a_in_,  alpha);
    pfin = std::pow(a_fin_, alpha);
}

void Integrator::initialize()
{
    // HACC TimeStepper.cxx:61-73 — set up Δp and seed the stepping state.
    tau   = (pfin - pin) / static_cast<double>(nsteps);
    tau2  = 0.5 * tau;
    pp    = pin;
    aa    = std::pow(pin, 1.0 / alpha);
    current_step = 0;
    set_adot();
    set_scal();
}

void Integrator::set_adot()
{
    // HACC TimeStepper.cxx:93-103 — adot = a · H(a)/H0 = a · E(a) in code units.
    adot = cosmo.adot(aa);
}

void Integrator::set_scal()
{
    // HACC TimeStepper.cxx:106-111.
    //   m_phiscal = 1.5 · Ω_cb       (Habib 2011 Eq. 16).
    //   dtdy      = a / (α · ȧ · p)  (Eq. 23 derivation).
    //   m_fscal   = m_phiscal · dtdy / a   = 1.5·Ω_cb / (α · ȧ · p · a).
    phiscal = 1.5 * cosmo.omega_cb();
    const double dtdy = aa / (alpha * adot * pp);
    fscal = phiscal * dtdy / aa;
}

void Integrator::advance_half_step()
{
    // HACC TimeStepper.cxx:114-122.  pp advances by τ/2; everything else
    // follows from the new pp (and hence the new aa).
    pp += tau2;
    aa  = std::pow(pp, 1.0 / alpha);
    set_adot();
    set_scal();
}

void Integrator::full_step(Particles& parts, Grid& grid, double uniform_mass)
{
    // HACC driver_pm.cxx:449-500 canonical DKD ordering.
    //
    //   half-drift at start scalars
    drift(parts, tau2, alpha, pp, adot);

    // Migrate after the first half-drift so the kick's CIC scatter
    // operates on correctly-distributed particles. Required for the
    // no-overload scheme; see analysis/references/pmkokkos_migration_audit.md
    // for the diagnosis. The post-second-half-drift migrate below
    // protects the NEXT step's kick.
    migrate(parts, grid);

    //   advance to mid-point scalars (HACC driver_pm.cxx:461)
    advance_half_step();

    //   full kick.  The PMStep applies vel += (-grad(phi)) * dt; HACC's
    //   set_scal puts the cosmological prefactor in fscal, so the integrator's
    //   "dt" passed to pm_kick is τ · fscal.  This is the discrete form of
    //   Eq. 23: ΔP̃ = -∇φ̃ · m_phiscal · dtdy/a · Δp.
    pm_kick(parts, grid, uniform_mass, tau * fscal);

    //   advance to end-of-step scalars (HACC driver_pm.cxx:492)
    advance_half_step();

    //   half-drift at end-of-step scalars
    drift(parts, tau2, alpha, pp, adot);

    //   migration: positions may have crossed rank boundaries during either
    //   half-drift.  Cabana::Distributor reassigns ownership; the Migrator
    //   also wraps positions into the periodic [0, ng)^3 box first.
    migrate(parts, grid);

    ++current_step;
}

} // namespace pmk
