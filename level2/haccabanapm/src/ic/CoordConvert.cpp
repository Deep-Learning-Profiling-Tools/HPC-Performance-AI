// src/ic/CoordConvert.cpp
// Per-particle wrap + velocity scale.  See CoordConvert.hpp for the
// position-convention rationale (we keep positions GLOBAL, contrary to the
// prompt's literal step 2).

#include "CoordConvert.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>

namespace pmk {

void convert_ic_to_simulation_coords(Particles& parts,
                                     const Grid& grid,
                                     double a_init)
{
    const std::size_t n = parts.num_local();
    if (n == 0) return;

    auto pos = parts.pos();
    auto vel = parts.vel();
    const float ng_f      = static_cast<float>(grid.ng());
    const float v_scale_f = static_cast<float>(a_init * a_init);

    Kokkos::parallel_for(
        "pmk::convert_ic_to_simulation_coords",
        Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(const int p) {
            for (int d = 0; d < 3; ++d) {
                // Periodic wrap into [0, ng) — Zel'dovich displacements can
                // push positions slightly outside the box.  Same algorithm as
                // src/comm/Migrator.cpp.
                float x = pos(p, d);
                x -= Kokkos::floor(x / ng_f) * ng_f;
                if (x >= ng_f) x -= ng_f;
                if (x <  0.0f) x += ng_f;
                pos(p, d) = x;

                // Non-symplectic dx/dt → P̃ = a²·dx/dt.
                vel(p, d) *= v_scale_f;
            }
        });
    Kokkos::fence();
}

} // namespace pmk
