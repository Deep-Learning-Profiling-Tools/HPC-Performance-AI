// src/time/Drift.cpp
// Drift kernel implementation.  See Drift.hpp for the full equation reference
// (Habib 2011 HACC units, Eq. 22 / 25).

#include "Drift.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>

namespace pmk {

void drift(Particles& parts,
           double tau_step,
           double alpha,
           double pp,
           double adot)
{
    // Eq. 22 prefactor 1/(α · ȧ · p^(1+1/α)).  Computed once on host so the
    // device kernel does only multiply-add — and so a future α=1 specialization
    // can short-circuit the std::pow without affecting the kernel signature.
    const double pf       = std::pow(pp, 1.0 + 1.0 / alpha);
    const double prefac   = 1.0 / (alpha * adot * pf);
    const float  coeff    = static_cast<float>(tau_step * prefac);

    auto pos = parts.pos();
    auto vel = parts.vel();
    const std::size_t n = parts.num_local();

    Kokkos::parallel_for(
        "pmk::drift",
        Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const int p) {
            pos(p, 0) += coeff * vel(p, 0);
            pos(p, 1) += coeff * vel(p, 1);
            pos(p, 2) += coeff * vel(p, 2);
        });
}

} // namespace pmk
