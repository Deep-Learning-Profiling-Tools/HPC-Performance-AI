// src/time/PMDriver.cpp
// Thin wrapper around Integrator::full_step.  Kept as a separate module so
// the eventual apps/pm_run.cpp executable has a single entry point and so
// orbit/regression tests can exercise the same loop the production app uses.

#include "PMDriver.hpp"

#include "../Particles.hpp"
#include "../Grid.hpp"
#include "../comm/Migrator.hpp"

namespace pmk {

void run(Particles& parts, Grid& grid, Integrator& integ, double uniform_mass)
{
    // Pre-loop migration: IC code may have placed particles outside the local
    // Cartesian block (Zel'dovich displacement crosses rank boundaries
    // routinely).  Cheap if there's nothing to do.
    migrate(parts, grid);

    for (int s = 0; s < integ.nsteps_total(); ++s) {
        integ.full_step(parts, grid, uniform_mass);
    }
}

} // namespace pmk
