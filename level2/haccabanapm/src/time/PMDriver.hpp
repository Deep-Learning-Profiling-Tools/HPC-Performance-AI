#pragma once
// src/time/PMDriver.hpp
// Top-level PM driver: pre-loop migration in case the IC put particles on
// the wrong ranks, then nsteps DKD steps.  Mirrors the run() loop in
// HACC's driver_pm.cxx but holds no state of its own — every per-step thing
// lives in the Integrator.
//
// NOTE: run() currently has no callers.  apps/pm_run_lib.cpp open-codes the
// same loop (pre-loop migrate, then nsteps * full_step) because it needs to
// interleave per-step timing and FULL_ALIVE_DUMP snapshot writes inside the
// loop body.  This module is kept as the minimal reference form of the
// evolution loop; if you change the step sequence here, change it there too,
// or collapse the two.

#include "Integrator.hpp"

namespace pmk {

class Particles;
class Grid;

// Run the cosmological PM evolution from integ.a() to integ.a()*… for
// integ.nsteps_total() full steps.  `uniform_mass` is the per-particle mass
// passed to the PM mass deposit (UniformMassP2G).
void run(Particles& parts, Grid& grid, Integrator& integ, double uniform_mass);

} // namespace pmk
