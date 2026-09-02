// src/CodeUnits.cpp
// Definitions for CodeUnits constructor (math is straight from HACC Domain.cxx).
// HACC reference: nbody/common/Domain.cxx:43-121

#include "CodeUnits.hpp"

namespace pmk {

CodeUnits::CodeUnits(double rL_, int ng_, double oL_, double omega_cb_)
    : rL(rL_), ng(ng_), oL(oL_), omega_cb(omega_cb_)
{
    const double dng = static_cast<double>(ng);

    // Position (Domain.cxx:54-55)
    phys2grid_pos = dng / rL;
    grid2phys_pos = rL / dng;

    // Velocity (Domain.cxx:58-59)
    phys2grid_vel = dng / (H0_KM_S_MPC * rL);
    grid2phys_vel = H0_KM_S_MPC * rL / dng;

    // Potential (Domain.cxx:61)
    phys2grid_phi = (dng / rL) * (dng / rL) / (H0_KM_S_MPC * H0_KM_S_MPC);
    grid2phys_phi = 1.0 / phys2grid_phi;

    // Overload (Domain.cxx:65-66) — snap to integer grid cells.
    ng_overload = static_cast<int>(std::ceil(oL * phys2grid_pos));
    oL_snapped  = ng_overload * grid2phys_pos;

    // Particle mass (Domain.cxx:116-119)
    const double ng3 = dng * dng * dng;
    grid2phys_mass = RHO_C * omega_cb * rL * rL * rL / ng3;
    phys2grid_mass = 1.0 / grid2phys_mass;

    // Poisson (TimeStepper.cxx:109)
    phiscal = NEWTON_G_CODE * omega_cb;
}

} // namespace pmk
