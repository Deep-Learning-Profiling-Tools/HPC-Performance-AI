#pragma once
// apps/pm_run_lib.hpp
// The pm_run evolution pipeline as a callable function so tests can invoke
// it without spawning a subprocess.  apps/pm_run.cpp is a thin main() shell
// around this; tests/test_pm_run_indat_driven.cpp exercises it directly.

#include <mpi.h>
#include <string>

namespace pmk {

// Run the pm_run pipeline.  Returns 0 on success, non-zero on error.
// The MPI/Kokkos environment must already be initialized by the caller.
//
// Reads <input.h5>, parses <indat.params> for Z_FIN/N_STEPS/T_CMB/
// FULL_ALIVE_DUMP, advances the integrator from a_in (snapshot) to
// a_fin = 1/(1+Z_FIN) in N_STEPS DKD steps, writes <output.h5> at the
// end and (optionally) <output>.step.<i>.h5 after each step in the
// FULL_ALIVE_DUMP list.
//
// Validates a_in < a_fin and N_STEPS > 0 (throws std::runtime_error
// internally; this function catches and prints to stderr on rank 0).
int pm_run_main(const std::string& input_path,
                const std::string& output_path,
                const std::string& indat_path,
                MPI_Comm           comm);

} // namespace pmk
