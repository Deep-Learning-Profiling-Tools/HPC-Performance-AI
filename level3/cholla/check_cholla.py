#!/usr/bin/env python3
"""Cholla correctness check of a registered input's run directory: the problem's physical / analytic criterion plus
a frozen self-baseline digest of the final fields (h5dump binary dumps + numpy; no h5py needed).

    check_cholla.py <problem> --run-dir DIR --params inputs/<id>.txt [--reference reference/<id>.json] [--label TEXT]
    check_cholla.py <problem> --run-dir DIR --params inputs/<id>.txt --make-reference <id> --out reference/<id>.json
                    --basis TEXT [--binary TEXT] [--l1-factor X] [--digest-rel X]          (never overwrites)

DIR is run.sh's run directory: DIR/out/<n>/<n>.h5.0 (upstream's HDF5 output of rank 0: datasets density,
momentum_x/y/z, Energy on dims (nx, ny, nz), x the slowest index, attributes t, n_step, dims, dx, bounds, domain,
gamma). <n> = 0 is the initial state, the last <n> the final state. Every check first asserts that the final output's
time equals the parameter file's tout (relative 1e-9: a run that stopped early is no result), then:

  sod        the planar Sod shock tube (init=Riemann, upstream's examples/3D/sod*.txt): the exact solution of the
             Riemann problem (Toro 1999, iterative star-state; left/right states, diaph, gamma from the parameter
             file) at the final time, against the x profiles of density, velocity and pressure averaged over y and
             z: L1 errors (mean |numerical - exact| over the cells). With a reference, each L1 error must not exceed
             l1_factor x the unmodified build's L1 error frozen in it (a Riemann-solver or reconstruction defect
             raises the L1 error by an order of magnitude: first-order ~ 4 x, a wrong wave speed ~ 10 x).
  soundwave  the linear sound wave (init=Sound_Wave: density, momentum_x perturbed by A sin(2 pi x) with the
             energy of a right-moving sound wave, c_s = sqrt(gamma P / rho) = 1 for the registered states): at the
             final time density = rho + A sin(2 pi (x - c_s t)) and momentum_x = A sin(2 pi (x - c_s t)): the L1 and
             the maximum error of both fields must not exceed sine_tol x A. Upstream's criteria: the one-period
             linear-wave tests (tHYDROtMHDSYSTEMLinearWavesParameterizedMpi) allow an L1 and a maximum error of
             1e-7 for A = 1e-6 with PLMC, i.e. 10 % of the amplitude after one period (sine_tol 0.1, the registered
             one-period input); the short tHYDROtMHDSYSTEMSoundWave3D test (t = 0.05, 64^3) allows 1e-7 for
             A = 1e-5 cell by cell, i.e. 1 % (sine_tol 0.01, validate.sh). The transverse momenta are not checked
             (upstream does not either: they are not part of the sound-wave eigenvector).
  noh        the 3-D Noh implosion (init=Noh_3D, custom_bcnd=noh: uniform inflow at |v| = 1 towards the origin,
             gamma = 5/3): the analytic post-shock state is density 64 inside the shock radius t/3 and the
             pre-shock density (1 + t/r)^2 outside. The mean density of the cells inside 0.8 x t/3 must be within
             noh_tol of 64 (default 0.10: the wall-heating dip of second-order Godunov codes is ~5 % of the plateau,
             Stone et al. 2008, Athena test 11; the Noh tests of other hydrodynamics codes accept 15 %) and the mean relative error
             of the density in the shell 0.80 <= r <= 0.95 (pre-shock, inside the unit cube) is reported.
  blast      the spherical over-pressure blast in a periodic box (init=Spherical_Overpressure_3D): no analytic
             solution; the conservative scheme must conserve total mass, momentum and energy between the initial
             and the final output to roundoff (|delta| <= cons_tol x the initial total, the momenta relative to
             M sqrt(E/M); cons_tol default 1e-9, double-precision sums over 1.7e7 cells), and the fields are checked
             against the frozen digest.
  all        self-baseline digest (with --reference): per field the sum, min, max and the block means on an
             8 x 8 x 8 coarsening of the grid, within digest_rel relative (of the field's max |value|) of the
             unmodified build's own run frozen in the reference (upstream's system tests compare whole fields to
             fiducial files at a fixed epsilon of 5e-12: Cholla's hydro step is deterministic).
Exit 0 with a 'PASS: Cholla check' line, 1 with 'FAIL: ...' (and the measured numbers either way).
"""
import argparse, glob, hashlib, json, math, os, re, subprocess, sys, tempfile, time

import numpy as np

NAME = "Cholla check"
SCHEMA = "hpcperf-cholla-reference-1"
FIELDS = ("density", "momentum_x", "momentum_y", "momentum_z", "Energy")


def h5attr(path, name):
    r = subprocess.run(["h5dump", "-a", name, path], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"h5dump -a {name}: {r.stderr.strip()[:200]}")
    m = re.search(r"DATA\s*\{\s*\(0\):\s*([^\n}]+)", r.stdout)
    if not m:
        raise RuntimeError(f"attribute {name}: no DATA block")
    return [float(x) for x in m.group(1).replace(",", " ").split()]


def h5field(path, name, dims):
    with tempfile.NamedTemporaryFile(delete=False) as t:
        tmp = t.name
    try:
        r = subprocess.run(["h5dump", "-d", "/" + name, "-b", "LE", "-o", tmp, path], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"h5dump {name}: {r.stderr.strip()[:200]}")
        a = np.fromfile(tmp, dtype="<f8")
    finally:
        os.unlink(tmp)
    n = int(np.prod(dims))
    if a.size != n:
        raise RuntimeError(f"{name}: {a.size} values, dims {dims} need {n}")
    return a.reshape(dims)


def read_params(path):
    p = {}
    with open(path) as f:
        for ln in f:
            ln = ln.split("#", 1)[0].strip()
            if "=" in ln:
                k, v = ln.split("=", 1)
                p[k.strip()] = v.strip()
    return p


def outputs(run_dir):
    outs = []
    for d in glob.glob(os.path.join(run_dir, "out", "*")):
        b = os.path.basename(d)
        if b.isdigit():
            f = os.path.join(d, f"{b}.h5.0")
            if os.path.isfile(f):
                outs.append((int(b), f))
    outs.sort()
    return [f for _, f in outs]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cell_centers(path, dims):
    bounds = h5attr(path, "bounds")
    dx = h5attr(path, "dx")
    return [bounds[a] + (np.arange(dims[a]) + 0.5) * dx[a] for a in range(3)]


# ----------------------------------------------------------------------------------------------------------- Sod
def riemann_exact(rho_l, u_l, p_l, rho_r, u_r, p_r, g, xs, t):
    """Exact solution of the 1-D Riemann problem (Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics,
    ch. 4): star-state pressure by Newton iteration, then sampling at x/t. Returns (rho, u, p) arrays on xs."""
    c_l, c_r = math.sqrt(g * p_l / rho_l), math.sqrt(g * p_r / rho_r)

    def f_k(p, rho_k, p_k, c_k):
        if p > p_k:  # shock
            a, b = 2.0 / ((g + 1) * rho_k), (g - 1) / (g + 1) * p_k
            return (p - p_k) * math.sqrt(a / (p + b)), math.sqrt(a / (p + b)) * (1 - (p - p_k) / (2 * (p + b)))
        return (2 * c_k / (g - 1)) * ((p / p_k) ** ((g - 1) / (2 * g)) - 1), (1 / (rho_k * c_k)) * (p / p_k) ** (-(g + 1) / (2 * g))

    p = max(1e-8, 0.5 * (p_l + p_r))
    for _ in range(100):
        fl, dl = f_k(p, rho_l, p_l, c_l); fr, dr = f_k(p, rho_r, p_r, c_r)
        f = fl + fr + (u_r - u_l)
        step = f / (dl + dr)
        p_new = max(1e-10, p - step)
        if abs(p_new - p) < 1e-14 * max(p_new, 1e-30):
            p = p_new; break
        p = p_new
    u = 0.5 * (u_l + u_r) + 0.5 * (f_k(p, rho_r, p_r, c_r)[0] - f_k(p, rho_l, p_l, c_l)[0])
    rho, vel, pre = np.empty_like(xs), np.empty_like(xs), np.empty_like(xs)
    s = xs / t if t > 0 else np.sign(xs) * np.inf
    for i, si in enumerate(s):
        if si <= u:  # left of the contact
            if p > p_l:  # left shock
                sl = u_l - c_l * math.sqrt((g + 1) / (2 * g) * p / p_l + (g - 1) / (2 * g))
                if si < sl:
                    rho[i], vel[i], pre[i] = rho_l, u_l, p_l
                else:
                    rho[i] = rho_l * ((p / p_l + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * p / p_l + 1)); vel[i], pre[i] = u, p
            else:  # left rarefaction
                shl, c_star = u_l - c_l, c_l * (p / p_l) ** ((g - 1) / (2 * g))
                stl = u - c_star
                if si < shl:
                    rho[i], vel[i], pre[i] = rho_l, u_l, p_l
                elif si > stl:
                    rho[i], vel[i], pre[i] = rho_l * (p / p_l) ** (1 / g), u, p
                else:
                    cf = 2 / (g + 1) * (c_l + (g - 1) / 2 * (u_l - si))
                    rho[i] = rho_l * (cf / c_l) ** (2 / (g - 1)); vel[i] = 2 / (g + 1) * (c_l + (g - 1) / 2 * u_l + si); pre[i] = p_l * (cf / c_l) ** (2 * g / (g - 1))
        else:  # right of the contact
            if p > p_r:  # right shock
                sr = u_r + c_r * math.sqrt((g + 1) / (2 * g) * p / p_r + (g - 1) / (2 * g))
                if si > sr:
                    rho[i], vel[i], pre[i] = rho_r, u_r, p_r
                else:
                    rho[i] = rho_r * ((p / p_r + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * p / p_r + 1)); vel[i], pre[i] = u, p
            else:  # right rarefaction
                shr, c_star = u_r + c_r, c_r * (p / p_r) ** ((g - 1) / (2 * g))
                str_ = u + c_star
                if si > shr:
                    rho[i], vel[i], pre[i] = rho_r, u_r, p_r
                elif si < str_:
                    rho[i], vel[i], pre[i] = rho_r * (p / p_r) ** (1 / g), u, p
                else:
                    cf = 2 / (g + 1) * (c_r - (g - 1) / 2 * (u_r - si))
                    rho[i] = rho_r * (cf / c_r) ** (2 / (g - 1)); vel[i] = 2 / (g + 1) * (-c_r + (g - 1) / 2 * u_r + si); pre[i] = p_r * (cf / c_r) ** (2 * g / (g - 1))
    return rho, vel, pre


def digest(fields, dims):
    """per-field sum / min / max and the block means of an 8 x 8 x 8 coarsening (dims must divide by 8)"""
    out = {}
    for name, a in fields.items():
        bx, by, bz = [max(1, d // 8) for d in dims]
        c = a[: (dims[0] // bx) * bx, : (dims[1] // by) * by, : (dims[2] // bz) * bz]
        blocks = c.reshape(dims[0] // bx, bx, dims[1] // by, by, dims[2] // bz, bz).mean(axis=(1, 3, 5))
        out[name] = {"sum": float(a.sum(dtype=np.float64)), "min": float(a.min()), "max": float(a.max()),
                     "absmax": float(np.abs(a).max()), "blocks": [round(float(x), 17) for x in blocks.ravel()]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problem", choices=["sod", "soundwave", "noh", "blast"])
    ap.add_argument("--run-dir", required=True); ap.add_argument("--params", required=True); ap.add_argument("--label", default="")
    ap.add_argument("--reference"); ap.add_argument("--make-reference", metavar="INPUT_ID"); ap.add_argument("--out")
    ap.add_argument("--basis", default=""); ap.add_argument("--binary", default="")
    ap.add_argument("--l1-factor", type=float, default=1.5, help="sod: allowed L1 error = this x the frozen one (recorded in the reference at freeze time)")
    ap.add_argument("--digest-rel", type=float, default=5e-12, help="self-baseline digest tolerance, relative to the field's max |value| (recorded at freeze time)")
    ap.add_argument("--sine-tol", type=float, default=0.1, help="soundwave: allowed L1 and max |numerical - analytic| as a fraction of the amplitude A (upstream one-period linear-wave tests: 1e-7 / 1e-6 = 0.1; the short SoundWave3D test: 1e-7 / 1e-5 = 0.01)")
    ap.add_argument("--noh-tol", type=float, default=0.10, help="noh: allowed relative deviation of the post-shock mean density from 64")
    ap.add_argument("--cons-tol", type=float, default=1e-9, help="blast: allowed relative change of the conserved totals")
    a = ap.parse_args()
    label = a.label or a.problem
    try:
        P = read_params(a.params)
        outs = outputs(a.run_dir)
        if len(outs) < 2:
            print(f"FAIL: {NAME} ({label}): {len(outs)} output(s) under {a.run_dir}/out (the initial and the final state are needed)"); return 1
        first, last = outs[0], outs[-1]
        dims = [int(x) for x in h5attr(last, "dims")]
        t = h5attr(last, "t")[0]; n_step = int(h5attr(last, "n_step")[0]); g = h5attr(last, "gamma")[0]
        tout = float(P["tout"])
        lines, ok = [], True
        good = abs(t - tout) <= 1e-9 * max(abs(tout), 1e-30)
        lines.append(f"  final output {os.path.relpath(last, a.run_dir)}: t = {t:.10g} vs tout {tout:g} {'ok' if good else 'BAD'}; n_step = {n_step}; dims = {dims}"); ok &= good
        xs, ys, zs = cell_centers(last, dims)
        F = {n: h5field(last, n, dims) for n in FIELDS}
        rho = F["density"]
        if a.problem == "sod":
            rho_l, u_l, p_l = float(P["rho_l"]), float(P["vx_l"]), float(P["P_l"])
            rho_r, u_r, p_r = float(P["rho_r"]), float(P["vx_r"]), float(P["P_r"])
            diaph = float(P["diaph"])
            prof_rho = rho.mean(axis=(1, 2)); vx = F["momentum_x"] / rho
            prof_u = vx.mean(axis=(1, 2))
            pres = (g - 1) * (F["Energy"] - 0.5 * (F["momentum_x"] ** 2 + F["momentum_y"] ** 2 + F["momentum_z"] ** 2) / rho)
            prof_p = pres.mean(axis=(1, 2))
            ex_rho, ex_u, ex_p = riemann_exact(rho_l, u_l, p_l, rho_r, u_r, p_r, g, xs - diaph, t)
            l1 = {"density": float(np.mean(np.abs(prof_rho - ex_rho))), "velocity": float(np.mean(np.abs(prof_u - ex_u))), "pressure": float(np.mean(np.abs(prof_p - ex_p)))}
            spread = {"density": float(np.max(rho.std(axis=(1, 2)))), "velocity": float(np.max(vx.std(axis=(1, 2))))}
            lines.append(f"  sod: exact Riemann solution (gamma {g:g}, t {t:g}, diaph {diaph:g}); L1 errors of the y,z-averaged profiles: density {l1['density']:.4e}, velocity {l1['velocity']:.4e}, pressure {l1['pressure']:.4e}; max std over y,z (planarity): density {spread['density']:.2e}, velocity {spread['velocity']:.2e}")
            crit = {"l1": l1, "planarity": spread}
        elif a.problem == "soundwave":
            rho0, p0, A, vx0 = float(P["rho"]), float(P["P"]), float(P["A"]), float(P.get("vx", 0))
            c = math.sqrt(g * p0 / rho0)
            if abs(c - 1.0) > 1e-6 or vx0 != 0:
                raise RuntimeError(f"soundwave: the IC's eigenvector assumes c_s = 1 and vx = 0 (c_s = {c:.12g}, vx = {vx0:g})")
            phase = 2 * math.pi * (xs - c * t)
            an = (A * np.sin(phase))[:, None, None]
            e_rho = np.abs(rho - (rho0 + an)); e_mx = np.abs(F["momentum_x"] - (rho0 * vx0 + an))
            dev = {"density_max": float(e_rho.max()), "density_l1": float(e_rho.mean()), "momentum_x_max": float(e_mx.max()), "momentum_x_l1": float(e_mx.mean())}
            amp = float(0.5 * (rho.max() - rho.min()))
            worst = max(dev.values()); good = worst <= a.sine_tol * A; ok &= good
            lines.append(f"  soundwave: c_s = {c:g}, {t / (1.0 / c):.4g} period(s); |density - analytic| max {dev['density_max']:.3e} L1 {dev['density_l1']:.3e}, |momentum_x - analytic| max {dev['momentum_x_max']:.3e} L1 {dev['momentum_x_l1']:.3e} (all <= {a.sine_tol:g} x A = {a.sine_tol * A:.3e}: usage {worst / (a.sine_tol * A):.3f}); numerical amplitude {amp:.6e} vs A {A:g} {'ok' if good else 'BAD'}")
            crit = {"deviations": dev, "amplitude": amp}
        elif a.problem == "noh":
            r = np.sqrt(xs[:, None, None] ** 2 + ys[None, :, None] ** 2 + zs[None, None, :] ** 2)
            r_s = t / 3.0
            inside = rho[r < 0.8 * r_s]
            if inside.size < 10:
                raise RuntimeError(f"noh: only {inside.size} cells inside 0.8 r_s")
            post = float(inside.mean()); dev = abs(post - 64.0) / 64.0
            shell = (r >= 0.80) & (r <= 0.95)
            pre_err = float(np.mean(np.abs(rho[shell] - (1 + t / r[shell]) ** 2) / (1 + t / r[shell]) ** 2))
            good = dev <= a.noh_tol; ok &= good
            lines.append(f"  noh: t = {t:g}, r_shock = {r_s:.4f}; post-shock mean density ({inside.size} cells inside 0.8 r_s) {post:.3f} vs 64 (deviation {dev:.3f} <= {a.noh_tol:g}) {'ok' if good else 'BAD'}; pre-shock shell 0.80-0.95: mean relative error vs (1 + t/r)^2 = {pre_err:.3e}")
            crit = {"post_shock_mean": post, "post_shock_dev": dev, "pre_shock_rel_err": pre_err}
        elif a.problem == "blast":
            F0 = {n: h5field(first, n, dims) for n in FIELDS}
            cons = {}
            m_tot, e_tot = float(F0["density"].sum(dtype=np.float64)), float(F0["Energy"].sum(dtype=np.float64))
            p_scale = m_tot * math.sqrt(e_tot / m_tot)      # momentum scale: total mass x the energy velocity scale sqrt(E/M)
            for n in FIELDS:
                s0, s1 = float(F0[n].sum(dtype=np.float64)), float(F[n].sum(dtype=np.float64))
                scale = p_scale if n.startswith("momentum") else abs(s0)
                cons[n] = {"initial": s0, "final": s1, "rel": abs(s1 - s0) / scale}
            worst = max(v["rel"] for v in cons.values())
            good = worst <= a.cons_tol; ok &= good
            lines.append("  blast: conserved totals initial -> final: " + ", ".join(f"{n} {v['initial']:.10e} -> {v['final']:.10e} (rel {v['rel']:.2e})" for n, v in cons.items()))
            lines.append(f"  blast: worst relative change {worst:.2e} (allowed {a.cons_tol:g}; momenta relative to M sqrt(E/M) = {p_scale:.3e}) {'ok' if good else 'BAD'}")
            crit = {"conservation": cons}
        dg = digest(F, dims)
        if a.make_reference:
            if not (a.out and a.basis):
                print(f"FAIL: {NAME} ({label}): --make-reference needs --out and --basis"); return 1
            if os.path.exists(a.out):
                print(f"FAIL: {NAME} ({label}): {a.out} exists (references are frozen once; move it aside explicitly)"); return 1
            ref = {"schema": SCHEMA, "input_id": a.make_reference, "problem": a.problem,
                   "provenance": {"kind": "self-baseline (digest) + the problem's analytic criterion", "run_dir": os.path.relpath(a.run_dir),
                                  "final_output_sha256": sha256(last), "params_sha256": sha256(a.params), "binary": a.binary,
                                  "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "basis": a.basis},
                   "rule": {"digest_rel": a.digest_rel, "l1_factor": a.l1_factor, "sine_tol": a.sine_tol, "noh_tol": a.noh_tol, "cons_tol": a.cons_tol,
                            "text": "digest: every field's sum/min/max/8^3 block means within digest_rel x max|field| of the reference; sod: L1 <= l1_factor x the frozen L1; soundwave / noh / blast: the analytic / conservation criteria at the stated tolerances"},
                   "time": t, "n_step": n_step, "dims": dims, "criterion": crit, "digest": dg}
            with open(a.out, "w") as f:
                json.dump(ref, f, indent=1, sort_keys=True); f.write("\n")
            lines.append(f"  reference written: {a.out}")
        elif a.reference:
            ref = json.load(open(a.reference))
            if ref.get("schema") != SCHEMA or ref.get("problem") != a.problem:
                raise RuntimeError(f"{a.reference}: schema/problem {ref.get('schema')}/{ref.get('problem')} do not match")
            rule = ref["rule"]
            if ref["dims"] != dims or abs(ref["time"] - t) > 1e-9 * max(abs(t), 1e-30):
                lines.append(f"  reference: dims {ref['dims']} / t {ref['time']:g} vs run {dims} / {t:g} BAD"); ok = False
            if ref["n_step"] != n_step:
                lines.append(f"  reference: n_step {ref['n_step']} vs run {n_step} (the time-step sequence changed) BAD"); ok = False
            if a.problem == "sod":
                rl1 = ref["criterion"]["l1"]
                for k in ("density", "velocity", "pressure"):
                    lim = rule["l1_factor"] * rl1[k]; good = crit["l1"][k] <= lim; ok &= good
                    lines.append(f"  reference sod L1 {k}: run {crit['l1'][k]:.4e} vs frozen {rl1[k]:.4e} (allowed {rule['l1_factor']:g} x = {lim:.4e}) {'ok' if good else 'BAD'}")
            worst = 0.0
            for n in FIELDS:
                rd, d = ref["digest"][n], dg[n]; scale = max(abs(rd["absmax"]), 1e-300)
                dev = max(abs(rd["sum"] - d["sum"]) / (abs(rd["sum"]) if rd["sum"] != 0 else scale * np.prod(dims)), abs(rd["min"] - d["min"]) / scale, abs(rd["max"] - d["max"]) / scale,
                          float(np.max(np.abs(np.array(rd["blocks"]) - np.array(d["blocks"])))) / scale)
                worst = max(worst, dev)
            good = worst <= rule["digest_rel"]; ok &= good
            lines.append(f"  reference digest ({os.path.basename(a.reference)}, self-baseline of the unmodified build): worst relative deviation {worst:.3e} (allowed {rule['digest_rel']:g}) {'ok' if good else 'BAD'}")
    except (RuntimeError, OSError, ValueError, KeyError, ZeroDivisionError) as ex:
        print(f"FAIL: {NAME} ({label}): {type(ex).__name__}: {ex}"); return 1
    print(f"{NAME} ({label}): {len(outs)} outputs under {a.run_dir}")
    print("\n".join(lines))
    print(f"{'PASS' if ok else 'FAIL'}: {NAME} ({label})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
