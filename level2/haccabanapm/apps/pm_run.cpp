// apps/pm_run.cpp
//
// Evolution binary.  Thin main() shell over pm_run_main; the actual
// pipeline lives in apps/pm_run_lib.cpp so tests can invoke it directly
// without spawning a subprocess.
//
// Usage:
//   mpiexec -n N pm_run <input_snapshot.h5> <output_snapshot.h5> <indat.params>
//
// All three arguments are positional and required (argc != 4 → usage, exit 2).
// Note the indat comes LAST here and FIRST in pm_ic.
//
// Cosmology comes from the SNAPSHOT, not the indat — see the precedence
// block at the top of apps/pm_run_lib.cpp, and docs/RUNNING.md.
//
// Opt-in per-step wall timing: PMKOKKOS_TIMING=1 (with optional
// PMKOKKOS_TIMING_WARMUP, default 5).  Topology override: PMKOKKOS_TOPOLOGY
// or an indat TOPOLOGY key.  See docs/RUNTIME_CONTROL.md.
//
// Reads the input snapshot's a_in and the indat's Z_FIN / N_STEPS, evolves
// the particles for N_STEPS DKD steps from a_in to a_fin = 1/(1+Z_FIN),
// writes the resulting state.  If FULL_ALIVE_DUMP is set in the indat to a
// space-separated list of step indices in [0, N_STEPS) the integrator also
// writes <output>.step.<i>.h5 after each requested step.
//
// See apps/pm_run_lib.cpp for the full pipeline documentation.

#include "pm_run_lib.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int exit_code = 0;
    {
        Kokkos::ScopeGuard guard(argc, argv);

        int my_rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

        if (argc != 4) {
            if (my_rank == 0) {
                std::cerr <<
                    "Usage: pm_run <input.h5> <output.h5> <indat.params>\n"
                    "\n"
                    "Reads the input snapshot and the indat (Z_FIN, N_STEPS,\n"
                    "T_CMB) and evolves from a_in (snapshot metadata) to\n"
                    "a_fin = 1/(1+Z_FIN) in N_STEPS DKD steps.  If\n"
                    "FULL_ALIVE_DUMP is a space-separated list of step\n"
                    "indices in [0, N_STEPS) the integrator writes\n"
                    "<output>.step.<i>.h5 after each requested step.\n";
            }
            MPI_Finalize();
            return 2;
        }

        exit_code = pmk::pm_run_main(argv[1], argv[2], argv[3], MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return exit_code;
}
