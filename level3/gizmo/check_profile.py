#!/usr/bin/env python3
"""GIZMO correctness check: upstream's own test acceptance per problem, re-implemented without h5py / scipy / matplotlib
(h5dump binary dumps of the final snapshot + the Python standard library), on a registered input's run directory.

    check_profile.py evrard --run-dir DIR --exact src/test/evrard/evrard_exact.txt [--label TEXT]
    check_profile.py sedov  --run-dir DIR --exact deps/exact/sedov_exact.txt
    check_profile.py noh    --run-dir DIR
    check_profile.py blob   --run-dir DIR

DIR is run.sh's run directory (DIR/output/snapshot_NNN.hdf5 and DIR/<id>.params). Every check first asserts, as upstream's
assert_final_time does, that the last snapshot's Header/Time equals the parameter file's TimeMax (relative 1e-6: a run that
stopped early is no result). Then (src/test/<name>/test_<name>.py of the frozen source, same binning and thresholds):
  evrard  radius from the origin; 30 log-spaced radial bins between the reference's smallest and largest radius, per-bin
          median density; mean |log10 rho_sim - log10 rho_exact| over the bins with a positive reference < 0.3
          (evrard_exact.txt: radius, density, entropy, velocity at t = 0.8, Hopkins 2015)
  sedov   radius from the box centre (BoxSize/2); 30 linear bins from 0 to 0.9 x the reference's largest radius, per-bin
          median density; over the bins where the interpolated reference density > 0.01:
          mean |rho_sim - rho_exact| / mean rho_exact < 0.3 (sedov_exact.txt: radius, temperature, density, radial velocity).
          --exact-time T0 (the time of the reference table, 0.03 for sedov_exact.txt) with a run ending at another
          time t: the Sedov-Taylor solution is self-similar, rho(r, t) = rho0 f(r / R(t)) with R proportional to
          t^(2/5), so the table is rescaled exactly: radius x (t / T0)^(2/5), density unchanged (velocity would be
          x (t / T0)^(-3/5), temperature x (t / T0)^(-6/5); only the density is checked, as upstream does)
  noh     radius from the box centre; the median density of the particles inside 0.8 x the analytic shock radius
          r_shock = t / 3 (at least 10 particles) within 15 % of the analytic post-shock density 64 (gamma = 5/3)
  blob    total gas mass of the final snapshot equal to the first snapshot's to 1e-3 relative; the mass of the blob
          (particles above 5 x the mean density) at the end at least 10 % of its initial mass
Exit 0 with a 'PASS: GIZMO profile check' line, 1 with 'FAIL: ...' (and the measured numbers either way).
"""
import argparse, array, glob, math, os, re, statistics, subprocess, sys, tempfile

NAME = "GIZMO profile check"


def h5dump(snap, dataset, dtype_code="d"):
    """A dataset as a flat array.array via h5dump binary little-endian output (double datasets of GIZMO snapshots)."""
    with tempfile.NamedTemporaryFile(delete=False) as t:
        tmp = t.name
    try:
        r = subprocess.run(["h5dump", "-d", dataset, "-b", "LE", "-o", tmp, snap], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"h5dump {dataset}: {r.stderr.strip()[:200]}")
        a = array.array(dtype_code)
        with open(tmp, "rb") as f:
            a.frombytes(f.read())
        return a
    finally:
        os.unlink(tmp)


def h5attr(snap, attr):
    r = subprocess.run(["h5dump", "-a", attr, snap], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"h5dump -a {attr}: {r.stderr.strip()[:200]}")
    m = re.search(r"DATA\s*\{\s*\(0\):\s*([^\n}]+)", r.stdout)
    if not m:
        raise RuntimeError(f"attribute {attr}: no DATA block in h5dump output")
    return [float(x) for x in m.group(1).replace(",", " ").split()]


def dataset_type(snap, dataset):
    r = subprocess.run(["h5dump", "-H", "-d", dataset, snap], capture_output=True, text=True)
    return "d" if "H5T_IEEE_F64" in r.stdout else "f"


def load(snap, names):
    out = {}
    for n in names:
        code = dataset_type(snap, f"/PartType0/{n}")
        out[n] = h5dump(snap, f"/PartType0/{n}", code)
    return out


def radii(coords, center):
    n = len(coords) // 3
    return [math.sqrt((coords[3 * i] - center) ** 2 + (coords[3 * i + 1] - center) ** 2 + (coords[3 * i + 2] - center) ** 2) for i in range(n)]


def interp(x, xs, ys):
    """linear interpolation with linear extrapolation (scipy interp1d fill_value='extrapolate')"""
    if x <= xs[0]:
        i = 0
    elif x >= xs[-1]:
        i = len(xs) - 2
    else:
        lo, hi = 0, len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid] <= x:
                lo = mid
            else:
                hi = mid
        i = lo
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] + t * (ys[i + 1] - ys[i])


def binned_median(r, v, edges):
    """per-bin median (scipy binned_statistic 'median'; empty bin -> None)"""
    buckets = [[] for _ in range(len(edges) - 1)]
    lo, hi, nb = edges[0], edges[-1], len(edges) - 1
    for ri, vi in zip(r, v):
        if ri < lo or ri > hi:
            continue
        k = min(int((ri - lo) / (hi - lo) * nb), nb - 1) if hi > lo else 0
        # exact edge semantics as binned_statistic: find the bin by edges (bins may be non-uniform)
        while k > 0 and ri < edges[k]:
            k -= 1
        while k < nb - 1 and ri >= edges[k + 1]:
            k += 1
        buckets[k].append(vi)
    return [statistics.median(b) if b else None for b in buckets]


def read_exact(path, cols):
    rows = []
    with open(path) as f:
        for ln in f:
            if ln.startswith("#") or not ln.strip():
                continue
            p = ln.split()
            rows.append([float(p[c]) for c in cols])
    rows.sort()
    return [list(c) for c in zip(*rows)]


def final_time_ok(snap, params, lines):
    t = h5attr(snap, "/Header/Time")[0]
    tmax = None
    with open(params) as f:
        for ln in f:
            p = ln.split()
            if p and p[0] == "TimeMax":
                tmax = float(p[1])
    if tmax is None:
        lines.append("  TimeMax not found in the parameter file BAD"); return False
    good = abs(t - tmax) <= 1e-6 * max(abs(tmax), 1e-30)
    lines.append(f"  final snapshot time {t:g} vs TimeMax {tmax:g} {'ok' if good else 'BAD'}")
    return good


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problem", choices=["evrard", "sedov", "noh", "blob"])
    ap.add_argument("--run-dir", required=True); ap.add_argument("--exact"); ap.add_argument("--label", default="")
    ap.add_argument("--exact-time", type=float, help="sedov only: the time of the reference table; the table is rescaled to the run's final time (self-similar solution)")
    a = ap.parse_args()
    label = a.label or a.problem
    snaps = sorted(glob.glob(os.path.join(a.run_dir, "output", "snapshot_*.hdf5")))
    params = sorted(glob.glob(os.path.join(a.run_dir, "*.params")))
    if len(snaps) < 2 or not params:
        print(f"FAIL: {NAME} ({label}): {len(snaps)} snapshot(s) and {len(params)} parameter file(s) under {a.run_dir}"); return 1
    lines, ok = [], True
    try:
        ok &= final_time_ok(snaps[-1], params[0], lines)
        if a.problem == "evrard":
            d = load(snaps[-1], ["Coordinates", "Density"])
            r = radii(d["Coordinates"], 0.0)
            re_, rhoe = read_exact(a.exact, [0, 1])
            lo, hi = math.log10(min(re_)), math.log10(max(re_))
            edges = [10 ** (lo + (hi - lo) * i / 29) for i in range(30)]
            med = binned_median(r, d["Density"], edges)
            l1 = []
            for b in range(29):
                rc = 0.5 * (edges[b] + edges[b + 1]); rx = interp(rc, re_, rhoe)
                if med[b] is not None and med[b] > 0 and rx > 0:
                    l1.append(abs(math.log10(med[b]) - math.log10(rx)))
            val = sum(l1) / len(l1) if l1 else float("inf"); good = val < 0.3; ok &= good
            lines.append(f"  evrard: {len(r)} particles, {len(l1)} bins; mean |log10 rho_sim - log10 rho_exact| = {val:.4f} (< 0.3) {'ok' if good else 'BAD'}")
        elif a.problem == "sedov":
            d = load(snaps[-1], ["Coordinates", "Density"])
            box = h5attr(snaps[-1], "/Header/BoxSize")[0]
            r = radii(d["Coordinates"], box / 2)
            re_, rhoe = read_exact(a.exact, [0, 2])
            if a.exact_time is not None:
                t = h5attr(snaps[-1], "/Header/Time")[0]; scale = (t / a.exact_time) ** 0.4
                re_ = [x * scale for x in re_]
                lines.append(f"  sedov: reference table at t = {a.exact_time:g} rescaled to the run's t = {t:g} (radius x {scale:.5f}, self-similar)")
            edges = [max(re_) * 0.9 * i / 29 for i in range(30)]
            med = binned_median(r, d["Density"], edges)
            num, den, nb = 0.0, 0.0, 0
            for b in range(29):
                rc = 0.5 * (edges[b] + edges[b + 1]); rx = interp(rc, re_, rhoe)
                if rx > 0.01 and med[b] is not None:
                    num += abs(med[b] - rx); den += abs(rx); nb += 1
            val = (num / nb) / (den / nb) if nb else float("inf"); good = val < 0.3; ok &= good
            lines.append(f"  sedov: {len(r)} particles, box {box:g}, {nb} post-shock bins; L1 relative density error = {val:.4f} (< 0.3) {'ok' if good else 'BAD'}")
        elif a.problem == "noh":
            d = load(snaps[-1], ["Coordinates", "Density"])
            box = h5attr(snaps[-1], "/Header/BoxSize")[0]; t = h5attr(snaps[-1], "/Header/Time")[0]
            r = radii(d["Coordinates"], box / 2)
            r_shock = t / 3.0
            inside = [rho for ri, rho in zip(r, d["Density"]) if ri < 0.8 * r_shock]
            if len(inside) <= 10:
                lines.append(f"  noh: only {len(inside)} particles inside 0.8 x r_shock = {0.8 * r_shock:g} BAD"); ok = False
            else:
                med = statistics.median(inside); dev = abs(med - 64.0) / 64.0; good = dev < 0.15; ok &= good
                lines.append(f"  noh: {len(r)} particles, t = {t:g}, r_shock = {r_shock:g}, {len(inside)} post-shock particles; median density {med:.2f} vs 64 (deviation {dev:.3f} < 0.15) {'ok' if good else 'BAD'}")
        elif a.problem == "blob":
            d0 = load(snaps[0], ["Density", "Masses"]); df = load(snaps[-1], ["Density", "Masses"])
            m0, mf = sum(d0["Masses"]), sum(df["Masses"])
            err = abs(mf - m0) / m0; good1 = err < 1e-3
            mean0 = sum(d0["Density"]) / len(d0["Density"]); meanf = sum(df["Density"]) / len(df["Density"])
            b0 = sum(m for rho, m in zip(d0["Density"], d0["Masses"]) if rho > 5 * mean0)
            bf = sum(m for rho, m in zip(df["Density"], df["Masses"]) if rho > 5 * meanf)
            frac = bf / b0 if b0 > 0 else float("inf"); good2 = frac > 0.1; ok &= good1 and good2
            lines.append(f"  blob: total mass {m0:.6e} -> {mf:.6e} (relative change {err:.2e} < 1e-3) {'ok' if good1 else 'BAD'}; blob mass fraction remaining {frac:.3f} (> 0.1) {'ok' if good2 else 'BAD'}")
    except (RuntimeError, OSError, ValueError, ZeroDivisionError) as ex:
        print(f"FAIL: {NAME} ({label}): {ex}"); return 1
    print(f"{NAME} ({label}): {os.path.basename(snaps[-1])} ({len(snaps)} snapshots) under {a.run_dir}")
    print("\n".join(lines))
    print(f"{'PASS' if ok else 'FAIL'}: {NAME} ({label})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
