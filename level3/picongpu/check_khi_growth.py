#!/usr/bin/env python3
"""PIConGPU KHI growth-rate check (upstream's KHI_growthRate test re-implemented without its plotting stack).

    check_khi_growth.py --energy <fields_energy.dat> --gamma G --density N_SI --dt DT_SI [--acceptance 0.2] [--label TEXT]

Upstream (share/picongpu/tests/KHI_growthRate, lib/python/test/setups/ESKHI/config.py + testsuite/Math): the
magnetic field energy of the Bx component (fields_energy.dat, column Bx[Joule], written every
--fields_energy.period steps) grows exponentially in the linear phase of the electron-scale Kelvin-Helmholtz
instability; the amplitude growth rate Gamma(t_k) = 0.5 * ln(f(t_{k+1}) / f(t_{k-1})) / (t_{k+1} - t_{k-1}) in
units of the relativistic electron plasma frequency omega_pe = sqrt(n e^2 / (eps0 gamma m_e)) is compared with
the theoretical maximum growth rate of the ESKHI, Gamma_th = 1 / (sqrt(8) gamma) (Alves 2012, Grismayer 2013);
the test passes when the maximum of Gamma(t) over the run lies within `acceptance` (upstream config: 0.2) of
Gamma_th. gamma = the Lorentz factor of the shear flow and the density come from the input set's .param files
(speciesInitialization / density params); the run's time step from simulation.param (DELTA_T_SI).
"""
import argparse, math, sys

E_CHARGE, EPS0, M_E = 1.602176634e-19, 8.8541878128e-12, 9.1093837015e-31


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--energy", required=True); ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--density", type=float, required=True, help="BASE_DENSITY_SI (1/m^3)")
    ap.add_argument("--dt", type=float, required=True, help="DELTA_T_SI (s)")
    ap.add_argument("--acceptance", type=float, default=0.2); ap.add_argument("--label", default="")
    a = ap.parse_args()
    label = a.label or a.energy
    steps, bx = [], []
    try:
        for ln in open(a.energy):
            if ln.startswith("#") or not ln.strip():
                continue
            p = ln.split(); steps.append(int(p[0])); bx.append(float(p[2]))
    except (OSError, ValueError, IndexError) as ex:
        print(f"FAIL: PIConGPU KHI growth-rate check ({label}): cannot read {a.energy}: {ex}"); return 1
    if len(steps) < 5:
        print(f"FAIL: PIConGPU KHI growth-rate check ({label}): only {len(steps)} rows"); return 1
    wpe = math.sqrt(a.density * E_CHARGE ** 2 / (EPS0 * a.gamma * M_E))
    t = [s * a.dt * wpe for s in steps]               # time in 1/omega_pe
    # upstream testsuite.Math.math.growthRate: the first (zero) value is skipped, central differences over two intervals
    g = []
    for k in range(2, len(bx) - 1):
        if bx[k + 1] > 0 and bx[k - 1] > 0:
            g.append(0.5 * math.log(bx[k + 1] / bx[k - 1]) / (t[k + 1] - t[k - 1]))
    if not g or not all(math.isfinite(x) for x in g):
        print(f"FAIL: PIConGPU KHI growth-rate check ({label}): no finite growth rate (Bx energy zero or non-finite)"); return 1
    gmax, gth = max(g), 1.0 / (math.sqrt(8.0) * a.gamma)
    dev = abs(gmax - gth) / gth
    ok = dev <= a.acceptance
    print(f"PIConGPU KHI growth-rate check ({label}): rows={len(steps)} omega_pe={wpe:.4e} 1/s, max Gamma_sim={gmax:.4f} "
          f"Gamma_theory={gth:.4f} (1/(sqrt(8) gamma), gamma={a.gamma}) deviation={dev:.3f} (acceptance {a.acceptance})")
    print(f"{'PASS' if ok else 'FAIL'}: PIConGPU KHI growth-rate check ({label})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
