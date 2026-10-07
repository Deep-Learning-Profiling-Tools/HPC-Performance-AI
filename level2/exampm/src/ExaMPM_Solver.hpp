/****************************************************************************
 * Copyright (c) 2018-2020 by the ExaMPM authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ExaMPM library. ExaMPM is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef EXAMPM_SOLVER_HPP
#define EXAMPM_SOLVER_HPP

#include "hpcperf_roi.h"
#include <ExaMPM_BoundaryConditions.hpp>
#include <ExaMPM_Mesh.hpp>
#include <ExaMPM_ProblemManager.hpp>
#include <ExaMPM_TimeIntegrator.hpp>
#include <ExaMPM_TimeStepControl.hpp>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <memory>
#include <string>

#include <mpi.h>

namespace ExaMPM
{
//---------------------------------------------------------------------------//
class SolverBase
{
  public:
    virtual ~SolverBase() = default;
    virtual void solve( const double t_final, const int write_freq ) = 0;
};

//---------------------------------------------------------------------------//
template <class MemorySpace, class ExecutionSpace>
class Solver : public SolverBase
{
  public:
    template <class InitFunc>
    Solver( MPI_Comm comm, const Kokkos::Array<double, 6>& global_bounding_box,
            const std::array<int, 3>& global_num_cell,
            const std::array<bool, 3>& periodic,
            const Cabana::Grid::BlockPartitioner<3>& partitioner,
            const int halo_cell_width, const InitFunc& create_functor,
            const int particles_per_cell, const double bulk_modulus,
            const double density, const double gamma, const double kappa,
            const double delta_t, const double gravity,
            const BoundaryCondition& bc )
        : _dt( delta_t )
        , _time( 0.0 )
        , _step( 0 )
        , _gravity( gravity )
        , _bc( bc )
        , _halo_min( 3 )
    {
        _mesh = std::make_shared<Mesh<MemorySpace>>(
            global_bounding_box, global_num_cell, periodic, partitioner,
            halo_cell_width, _halo_min, comm );

        _bc.min = _mesh->minDomainGlobalNodeIndex();
        _bc.max = _mesh->maxDomainGlobalNodeIndex();

        _pm = std::make_shared<ProblemManager<MemorySpace>>(
            ExecutionSpace(), _mesh, create_functor, particles_per_cell,
            bulk_modulus, density, gamma, kappa );

        MPI_Comm_rank( comm, &_rank );

        // tools/inputs correctness signature: the global particle count at the start (see printFinalState)
        unsigned long long n_local = _pm->numParticle();
        MPI_Allreduce( &n_local, &_num_particles_initial, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm );
    }

    void solve( const double t_final, const int write_freq ) override
    {
        // Output initial state.
        outputParticles();

        // tools/timing ROI: the time loop; the initial and the periodic particle output are excluded
        HPCPERF_ROI_BEGIN_SYNC();
        while ( _time < t_final )
        {
            if ( 0 == _rank && 0 == _step % write_freq )
                printf( "Time %f / %f\n", _time, t_final );

            // Fixed timestep is guaranteed only when sufficently low dt
            // does not violate the CFL condition (otherwise user-set dt is
            // really a max_dt).
            _dt = timeStepControl( _mesh->localGrid()->globalGrid().comm(),
                                   ExecutionSpace(), *_pm, _dt );

            TimeIntegrator::step( ExecutionSpace(), *_pm, _dt, _gravity, _bc );

            _pm->communicateParticles( _halo_min );

            _time += _dt;
            _step++;

            // Output particles periodically.
            const bool hpcperf_output = ( 0 == ( _step ) % write_freq );
            if ( hpcperf_output ) HPCPERF_ROI_EXCLUDE_BEGIN_SYNC();
            if ( 0 == ( _step ) % write_freq )
                outputParticles();
            if ( hpcperf_output ) HPCPERF_ROI_EXCLUDE_END();
        }
        HPCPERF_ROI_END_SYNC();
        // tools/inputs correctness signature: conserved quantities of the final state, after the ROI
        printFinalState();
    }

    // Final-state summary for the correctness check (level2/exampm/inputs.yaml `baseline.quantities`):
    // the global particle count (must equal the initial count), the position bounds (the fluid stays in
    // the unit cube), the total volume sum(J)/N_0 (the deformation-gradient determinant summed over the
    // particles, 1 for the incompressible-ish fluid) and, as diagnostics, the mean velocity and centre of
    // mass. Computed with device reductions after the time loop -- never inside the ROI -- and printed
    // by rank 0 with full precision. The computation of the run is not touched.
    void printFinalState()
    {
        auto x_p = _pm->get( Location::Particle(), Field::Position() );
        auto u_p = _pm->get( Location::Particle(), Field::Velocity() );
        auto j_p = _pm->get( Location::Particle(), Field::J() );
        const int n = _pm->numParticle();
        Kokkos::RangePolicy<ExecutionSpace> policy( 0, n );
        double sum_j = 0.0, pos_min = 1.0e300, pos_max = -1.0e300;
        double sum_u[3] = { 0.0, 0.0, 0.0 }, sum_x[3] = { 0.0, 0.0, 0.0 };
        Kokkos::parallel_reduce(
            "hpcperf_final_sum_j", policy,
            KOKKOS_LAMBDA( const int p, double& s ) { s += j_p( p ); }, sum_j );
        Kokkos::parallel_reduce(
            "hpcperf_final_pos_min", policy,
            KOKKOS_LAMBDA( const int p, double& m ) {
                for ( int d = 0; d < 3; ++d )
                    m = ( x_p( p, d ) < m ) ? x_p( p, d ) : m;
            },
            Kokkos::Min<double>( pos_min ) );
        Kokkos::parallel_reduce(
            "hpcperf_final_pos_max", policy,
            KOKKOS_LAMBDA( const int p, double& m ) {
                for ( int d = 0; d < 3; ++d )
                    m = ( x_p( p, d ) > m ) ? x_p( p, d ) : m;
            },
            Kokkos::Max<double>( pos_max ) );
        for ( int d = 0; d < 3; ++d )
        {
            Kokkos::parallel_reduce(
                "hpcperf_final_sum_u", policy,
                KOKKOS_LAMBDA( const int p, double& s ) { s += u_p( p, d ); }, sum_u[d] );
            Kokkos::parallel_reduce(
                "hpcperf_final_sum_x", policy,
                KOKKOS_LAMBDA( const int p, double& s ) { s += x_p( p, d ); }, sum_x[d] );
        }
        Kokkos::fence();
        MPI_Comm comm = _mesh->localGrid()->globalGrid().comm();
        unsigned long long n_local = n, n_global = 0;
        double g_sum_j, g_min, g_max, g_sum_u[3], g_sum_x[3];
        MPI_Allreduce( &n_local, &n_global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm );
        MPI_Allreduce( &sum_j, &g_sum_j, 1, MPI_DOUBLE, MPI_SUM, comm );
        MPI_Allreduce( &pos_min, &g_min, 1, MPI_DOUBLE, MPI_MIN, comm );
        MPI_Allreduce( &pos_max, &g_max, 1, MPI_DOUBLE, MPI_MAX, comm );
        MPI_Allreduce( sum_u, g_sum_u, 3, MPI_DOUBLE, MPI_SUM, comm );
        MPI_Allreduce( sum_x, g_sum_x, 3, MPI_DOUBLE, MPI_SUM, comm );
        if ( 0 == _rank )
        {
            const double n0 = static_cast<double>( _num_particles_initial );
            const double ng = static_cast<double>( n_global );
            printf( "ExaMPM final state: step %d time %.17g particles %llu initial %llu "
                    "pos_min %.17g pos_max %.17g volume_ratio %.17g "
                    "v_mean %.17g %.17g %.17g x_mean %.17g %.17g %.17g\n",
                    _step, _time, n_global, _num_particles_initial, g_min, g_max,
                    ( n0 > 0.0 ) ? g_sum_j / n0 : 0.0,
                    ( ng > 0.0 ) ? g_sum_u[0] / ng : 0.0, ( ng > 0.0 ) ? g_sum_u[1] / ng : 0.0,
                    ( ng > 0.0 ) ? g_sum_u[2] / ng : 0.0, ( ng > 0.0 ) ? g_sum_x[0] / ng : 0.0,
                    ( ng > 0.0 ) ? g_sum_x[1] / ng : 0.0, ( ng > 0.0 ) ? g_sum_x[2] / ng : 0.0 );
            fflush( stdout );
        }
    }

    void outputParticles()
    {
        // Prefer HDF5 output over Silo. Only output if one is enabled.
#ifdef Cabana_ENABLE_HDF5
        Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
        Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
            h5_config, "particles", _mesh->localGrid()->globalGrid().comm(),
            _step, _time, _pm->numParticle(),
            _pm->get( Location::Particle(), Field::Position() ),
            _pm->get( Location::Particle(), Field::Velocity() ),
            _pm->get( Location::Particle(), Field::J() ) );
#else
#ifdef Cabana_ENABLE_SILO
        Cabana::Grid::Experimental::SiloParticleOutput::writeTimeStep(
            "particles", _mesh->localGrid()->globalGrid(), _step, _time,
            _pm->get( Location::Particle(), Field::Position() ),
            _pm->get( Location::Particle(), Field::Velocity() ),
            _pm->get( Location::Particle(), Field::J() ) );
#else
        if ( _rank == 0 )
            std::cout << "No particle output enabled in Cabana. Add "
                         "Cabana_REQUIRE_HDF5=ON or Cabana_REQUIRE_SILO=ON to "
                         "the Cabana build if needed.";
#endif
#endif
    }

  private:
    double _dt;
    double _time;
    int _step;
    double _gravity;
    BoundaryCondition _bc;
    int _halo_min;
    std::shared_ptr<Mesh<MemorySpace>> _mesh;
    std::shared_ptr<ProblemManager<MemorySpace>> _pm;
    int _rank;
    unsigned long long _num_particles_initial = 0;
};

//---------------------------------------------------------------------------//
// Creation method.
template <class InitFunc>
std::shared_ptr<SolverBase>
createSolver( const std::string& exec_space, MPI_Comm comm,
              const Kokkos::Array<double, 6>& global_bounding_box,
              const std::array<int, 3>& global_num_cell,
              const std::array<bool, 3>& periodic,
              const Cabana::Grid::BlockPartitioner<3>& partitioner,
              const int halo_cell_width, const InitFunc& create_functor,
              const int particles_per_cell, const double bulk_modulus,
              const double density, const double gamma, const double kappa,
              const double delta_t, const double gravity,
              const BoundaryCondition& bc )
{
    if ( 0 == exec_space.compare( "serial" ) ||
         0 == exec_space.compare( "Serial" ) ||
         0 == exec_space.compare( "SERIAL" ) )
    {
#ifdef KOKKOS_ENABLE_SERIAL
        return std::make_shared<
            ExaMPM::Solver<Kokkos::HostSpace, Kokkos::Serial>>(
            comm, global_bounding_box, global_num_cell, periodic, partitioner,
            halo_cell_width, create_functor, particles_per_cell, bulk_modulus,
            density, gamma, kappa, delta_t, gravity, bc );
#else
        throw std::runtime_error( "Serial Backend Not Enabled" );
#endif
    }
    else if ( 0 == exec_space.compare( "openmp" ) ||
              0 == exec_space.compare( "OpenMP" ) ||
              0 == exec_space.compare( "OPENMP" ) )
    {
#ifdef KOKKOS_ENABLE_OPENMP
        return std::make_shared<
            ExaMPM::Solver<Kokkos::HostSpace, Kokkos::OpenMP>>(
            comm, global_bounding_box, global_num_cell, periodic, partitioner,
            halo_cell_width, create_functor, particles_per_cell, bulk_modulus,
            density, gamma, kappa, delta_t, gravity, bc );
#else
        throw std::runtime_error( "OpenMP Backend Not Enabled" );
#endif
    }
    else if ( 0 == exec_space.compare( "cuda" ) ||
              0 == exec_space.compare( "Cuda" ) ||
              0 == exec_space.compare( "CUDA" ) )
    {
#ifdef KOKKOS_ENABLE_CUDA
        return std::make_shared<
            ExaMPM::Solver<Kokkos::CudaSpace, Kokkos::Cuda>>(
            comm, global_bounding_box, global_num_cell, periodic, partitioner,
            halo_cell_width, create_functor, particles_per_cell, bulk_modulus,
            density, gamma, kappa, delta_t, gravity, bc );
#else
        throw std::runtime_error( "CUDA Backend Not Enabled" );
#endif
    }
    else if ( 0 == exec_space.compare( "hip" ) ||
              0 == exec_space.compare( "Hip" ) ||
              0 == exec_space.compare( "HIP" ) )
    {
#ifdef KOKKOS_ENABLE_HIP
        return std::make_shared<ExaMPM::Solver<Kokkos::Experimental::HIPSpace,
                                               Kokkos::Experimental::HIP>>(
            comm, global_bounding_box, global_num_cell, periodic, partitioner,
            halo_cell_width, create_functor, particles_per_cell, bulk_modulus,
            density, gamma, kappa, delta_t, gravity, bc );
#else
        throw std::runtime_error( "HIP Backend Not Enabled" );
#endif
    }
    else
    {
        throw std::runtime_error( "invalid backend" );
        return nullptr;
    }
}

//---------------------------------------------------------------------------//

} // end namespace ExaMPM

#endif // end EXAMPM_SOLVER_HPP
