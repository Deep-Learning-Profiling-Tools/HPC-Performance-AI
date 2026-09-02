#!/usr/bin/env python3
"""Check an ExaMiniMD Lennard-Jones melt log against independent references.

Used by validate.sh. Reads ExaMiniMD's stdout (the rows after the
"#Timestep Temperature PotE ETot Time Atomsteps/s" header, values are per
atom) and checks:

  1. step-0 temperature      == T0 requested by `velocity all create T0 ...`
  2. step-0 potential energy == analytic lattice sum for the perfect fcc
                                lattice the deck creates, INCLUDING the LJ
                                cutoff-energy shift ExaMiniMD always applies
                                (shift_flag = true in force_lj_neigh_impl.h)
  3. step-0 total energy     == that PE + KE/atom, KE/atom = 1.5*T0*(N-1)/N
                                (ExaMiniMD, like LAMMPS, uses dof = 3N-3)
  4. energy conservation     max_t |ETot(t) - ETot(0)| / |ETot(0)| <= drift_tol
  5. (optional) LAMMPS bench/in.lj reference values (LAMMPS uses the unshifted
     pair energy, so the constant shift 0.5*n_cut*(-phi(rc)) per atom is
     removed from ExaMiniMD's PE/ETot before comparing):
       PE(0)-shift   == LAMMPS E_pair(0)
       ETot(0)-shift == LAMMPS TotEng(0)
       T(100)        == LAMMPS Temp(100)   (forces are shift-independent, so
                                            the trajectories are identical)

The references are computed here from first principles (reduced density,
cutoff, lattice size and temperature of the deck); nothing is copied from an
ExaMiniMD run. Exit status 0 if all checks pass, 1 otherwise.
"""
import argparse
import itertools
import math
import re
import sys


def lattice_sum(rho, rc, eps=1.0, sigma=1.0):
    """Per-atom LJ energy of an atom in an infinite fcc lattice of reduced
    density rho with cutoff rc. Returns (pe_unshifted, pe_shifted, n_cut,
    shift_per_atom)."""
    a = (4.0 / rho) ** (1.0 / 3.0)          # conventional cubic cell edge
    sites = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    m = int(math.ceil(rc / a)) + 1
    rc2 = rc * rc
    pe = 0.0
    n_cut = 0
    for nx, ny, nz in itertools.product(range(-m, m + 1), repeat=3):
        for sx, sy, sz in sites:
            x, y, z = (nx + sx) * a, (ny + sy) * a, (nz + sz) * a
            r2 = x * x + y * y + z * z
            if r2 == 0.0 or r2 >= rc2:   # ExaMiniMD/LAMMPS use rsq < cutsq
                continue
            s6 = (sigma * sigma / r2) ** 3
            pe += 4.0 * eps * (s6 * s6 - s6)
            n_cut += 1
    phi_rc = 4.0 * eps * ((sigma / rc) ** 12 - (sigma / rc) ** 6)
    pe_unshifted = 0.5 * pe                 # every pair counted twice
    shift = -0.5 * n_cut * phi_rc           # ExaMiniMD subtracts phi(rc) per pair
    return pe_unshifted, pe_unshifted + shift, n_cut, shift


def parse_log(path):
    """Return {step: (T, PE, ETot)} from ExaMiniMD stdout."""
    rows = {}
    pat = re.compile(r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$")
    with open(path) as f:
        for line in f:
            if "PERFORMANCE" in line or "Atomsteps" in line:
                continue
            m = pat.match(line)
            if m:
                try:
                    rows[int(m.group(1))] = tuple(float(m.group(i)) for i in (2, 3, 4))
                except ValueError:
                    pass
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, help="ExaMiniMD stdout")
    ap.add_argument("--lattice", type=int, required=True, help="fcc unit cells per box edge (N = 4 L^3)")
    ap.add_argument("--temp", type=float, required=True, help="T0 of `velocity all create`")
    ap.add_argument("--rho", type=float, default=0.8442, help="reduced density (lattice fcc RHO)")
    ap.add_argument("--rc", type=float, default=2.5, help="LJ cutoff")
    ap.add_argument("--drift-tol", type=float, default=1e-3, help="relative tolerance on ETot drift")
    ap.add_argument("--step0-rtol", type=float, default=1e-5, help="relative tolerance on step-0 PE/ETot")
    ap.add_argument("--temp-atol", type=float, default=1e-5, help="absolute tolerance on T(0)")
    ap.add_argument("--lammps-epair0", type=float, help="LAMMPS E_pair at step 0 (unshifted)")
    ap.add_argument("--lammps-etot0", type=float, help="LAMMPS TotEng at step 0 (unshifted)")
    ap.add_argument("--lammps-temp", type=float, nargs=2, metavar=("STEP", "TEMP"),
                    help="LAMMPS Temp at a later step")
    ap.add_argument("--lammps-rtol", type=float, default=1e-3, help="relative tolerance vs LAMMPS T(step)")
    args = ap.parse_args()

    N = 4 * args.lattice ** 3
    pe_u, pe_s, n_cut, shift = lattice_sum(args.rho, args.rc)
    ke = 1.5 * args.temp * (N - 1) / N
    etot_ref = pe_s + ke
    rows = parse_log(args.log)

    print(f"reference: fcc rho={args.rho} rc={args.rc} L={args.lattice} N={N} T0={args.temp}")
    print(f"           {n_cut} neighbours within rc; PE/atom unshifted={pe_u:.10f} "
          f"shift={shift:.10f} shifted={pe_s:.10f}; KE/atom={ke:.10f}; ETot/atom={etot_ref:.10f}")

    ok = True

    def check(name, got, ref, tol, rel):
        nonlocal ok
        err = abs(got - ref) / (abs(ref) if rel else 1.0)
        passed = err <= tol
        ok &= passed
        kind = "rel" if rel else "abs"
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<28s} got {got:.7f}  ref {ref:.7f}  "
              f"{kind} err {err:.2e}  tol {tol:.0e}")

    if 0 not in rows:
        print("  FAIL  no step-0 thermo row found in", args.log)
        print("ExaMiniMD LJ check: FAIL")
        return 1
    T0, PE0, E0 = rows[0]
    check("T(0)", T0, args.temp, args.temp_atol, rel=False)
    check("PE(0)/atom (shifted)", PE0, pe_s, args.step0_rtol, rel=True)
    check("ETot(0)/atom (shifted)", E0, etot_ref, args.step0_rtol, rel=True)

    later = [s for s in rows if s > 0]
    if not later:
        print("  FAIL  no thermo rows after step 0 -- cannot check energy conservation")
        ok = False
    else:
        drift = max(abs(rows[s][2] - E0) for s in later) / abs(E0)
        passed = drift <= args.drift_tol
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {'ETot drift, steps 0..' + str(max(later)):<28s} "
              f"max rel |dE/E| {drift:.2e}  tol {args.drift_tol:.0e}  "
              f"(ETot range [{min(rows[s][2] for s in rows):.6f}, {max(rows[s][2] for s in rows):.6f}])")

    if args.lammps_epair0 is not None:
        check("PE(0)-shift vs LAMMPS E_pair", PE0 - shift, args.lammps_epair0, args.step0_rtol, rel=True)
    if args.lammps_etot0 is not None:
        check("ETot(0)-shift vs LAMMPS TotEng", E0 - shift, args.lammps_etot0, args.step0_rtol, rel=True)
    if args.lammps_temp is not None:
        step = int(args.lammps_temp[0])
        if step in rows:
            check(f"T({step}) vs LAMMPS Temp", rows[step][0], args.lammps_temp[1], args.lammps_rtol, rel=True)
        else:
            print(f"  FAIL  no thermo row for step {step} in {args.log}")
            ok = False

    print("ExaMiniMD LJ check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
