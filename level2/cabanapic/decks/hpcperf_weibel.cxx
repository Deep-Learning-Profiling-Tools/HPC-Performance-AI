// HPC-Performance-AI addition (not an upstream file).
//
// The upstream Weibel / filamentation-instability problem -- identical to the
// built-in default deck in src/input/deck.h and to the deck of the upstream
// energy-comparison test (tests/energy_comparison/2stream-em.cxx): two
// counter-streaming electron beams (+/- v0 along x) whose drift is perturbed
// along y, on a 1 x ny x 1 periodic grid. The only difference is that the
// three size parameters can be overridden at run time through environment
// variables, so one build serves both a quick smoke run and a GPU-sized run:
//
//   HPCPERF_NY         cells along y       (upstream: 32)
//   HPCPERF_NPPC       particles per cell  (upstream: 100)
//   HPCPERF_NUM_STEPS  time steps          (upstream: 6000)
//
// The defaults reproduce the upstream problem exactly. Everything else (v0,
// box lengths, n0, boundary, the CFL time step dt = 0.99*courant/c) is left
// as upstream; dt therefore shrinks as 1/ny, so keeping the same physical
// duration as upstream needs num_steps = 6000*ny/32. Because the default
// particle initialiser hard-codes the cell layout for nx = nz = 1, only ny
// may be scaled.
#include "src/input/deck.h"

#include <cstdlib>

namespace {
    long env_or(const char* name, long fallback)
    {
        const char* v = std::getenv(name);
        if (v == nullptr || *v == '\0') return fallback;
        char* end = nullptr;
        long r = std::strtol(v, &end, 10);
        if (end == v || *end != '\0' || r <= 0) {
            std::cerr << "hpcperf_weibel deck: ignoring invalid " << name
                      << "=\"" << v << "\" (need a positive integer)" << std::endl;
            return fallback;
        }
        return r;
    }
}

Input_Deck::Input_Deck()
{
    nx = 1;
    ny = env_or("HPCPERF_NY", 32);
    nz = 1;

    num_steps = env_or("HPCPERF_NUM_STEPS", 6000);
    nppc = env_or("HPCPERF_NPPC", 100);

    v0 = 0.0866025403784439;

    real_ gam = 1.0 / sqrt(1.0 - v0*v0);

    const real_ default_grid_len = 1.0;

    len_x_global = default_grid_len;
    len_y_global = 0.628318530717959*(gam*sqrt(gam));
    len_z_global = default_grid_len;

    dt = 0.99*courant_length(
            len_x_global, len_y_global, len_z_global,
            nx, ny, nz
            ) / c;

    n0 = 2.0; //for 2stream, for 2 species, making sure omega_p of each species is 1

    std::cout << "hpcperf_weibel deck: ny=" << ny << " nppc=" << nppc
              << " num_steps=" << num_steps
              << " (HPCPERF_NY / HPCPERF_NPPC / HPCPERF_NUM_STEPS)" << std::endl;
}
