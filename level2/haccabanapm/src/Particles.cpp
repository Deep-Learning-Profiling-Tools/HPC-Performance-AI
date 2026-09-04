// src/Particles.cpp
// Particles wrapper: trivial RAII over a Cabana::AoSoA on device memory.

#include "Particles.hpp"

namespace pmk {

Particles::Particles(MPI_Comm comm, std::size_t n_max)
    : m_comm(comm), m_particles("particles", 0)
{
    if (n_max > 0)
        reserve(n_max);
}

void Particles::resize(std::size_t n)
{
    m_particles.resize(n);
    m_n_local = n;
}

void Particles::reserve(std::size_t n_max)
{
    m_particles.reserve(n_max);
}

} // namespace pmk
