#!/usr/bin/env python3
"""GEOS beam-bending checks on the TimeHistory output (displacement_history.hdf5).

  geos_beam_check.py analytic <run_dir> --tol T --curve beamBending_curve.py
        geos-ats "script" curve check: the displacement trace vs upstream's analytic reference
        (inputFiles/solidMechanics/beamBending_curve.py, imported and called exactly as geos-ats does);
        gate = geos-ats check_diff() metric ||u - u_script||_2 / N <= T over the whole (ntime, nnodes, 3)
        array (T = 0.0002 in beamBending.ats). The per-output-time relative L-inf of uy is printed as
        information (it is the mesh's discretisation error, ~9e-4 for 80x8x4 in upstream's own baseline).
  geos_beam_check.py compare <ref_dir> <run_dir> [--tol REL [--info-tol REL2]] [--l2n-tol T]
        two histories (trace nodes matched by reference position when partitions order them
        differently): --tol gates on the relative L-inf max|u_ref-u|/max|u_ref| (cross-rank-count
        consistency), --l2n-tol gates on the geos-ats "baseline" curve metric ||u - u_ref||_2 / N.
  geos_beam_check.py restart <run_dir> <baseline_dir> --rtol R --atol A
        GEOS restart HDF5 files vs upstream's integrated-test baseline (same rank count), geos-ats
        restart_check.py rules (see RESTART_EXCLUDE / restart()).

Every value is required finite. Exit 0 on agreement, 1 on disagreement, 2 on structural problems
(missing file/dataset, NaN, shape). Run with the profile's venv python (h5py) -- see validate.sh.
"""
import argparse, glob, importlib.util, math, os, sys
import numpy as np
import h5py


def die(msg, code=2):
    print(f"  VALIDATION ERROR: {msg}"); sys.exit(code)


def load_history(run_dir):
    f = os.path.join(run_dir, "displacement_history.hdf5")
    if not os.path.isfile(f): die(f"{f} missing (TimeHistory output not produced)")
    with h5py.File(f, "r") as h:
        keys = list(h.keys())
        need = ["totalDisplacement trace", "totalDisplacement Time", "totalDisplacement ReferencePosition trace"]
        for k in need:
            if k not in h: die(f"dataset {k!r} missing in {f} (have {keys})")
        d = {k: np.array(h[k]) for k in need}
    for k, v in d.items():
        if not np.isfinite(v).all(): die(f"non-finite values in {k}")
    return d


def geos_ats_diff(target, baseline):
    """geos-ats curve_check.py check_diff(): ||t-b||_2 / N over the whole (ntime, nnodes, 3) array [m]."""
    dx = target - baseline
    return math.sqrt(float(np.sum(dx * dx))) / dx.size


def analytic(run_dir, tol, curve_py):
    d = load_history(run_dir)
    spec = importlib.util.spec_from_file_location("beam_curve", curve_py)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    ref = mod.curve(**d)                          # shape (ntime, nnodes, 3), y-component filled, x/z zero
    disp = d["totalDisplacement trace"]
    if ref.shape != disp.shape: die(f"reference shape {ref.shape} != data shape {disp.shape}")
    t = np.squeeze(d["totalDisplacement Time"][:, 0])
    worst = 0.0
    for i, tb in enumerate(t):
        num = np.abs(disp[i, :, 1] - ref[i, :, 1]).max(); den = np.abs(ref[i, :, 1]).max()
        rel = num / den if den > 0 else num
        worst = max(worst, rel)
        print(f"    t={tb:5.1f}: max|uy_geos - uy_analytic| = {num:.4e}, max|uy_analytic| = {den:.4e}, rel = {rel:.3e}")
    diff = geos_ats_diff(disp, ref)
    print(f"    (info) per-time relative L-inf of uy vs the Euler-Bernoulli curve: worst {worst:.3e} over {len(t)} output times (discretisation error of the mesh)")
    print(f"    analytic beam check (geos-ats curve_check metric ||u - u_script||_2 / N over {disp.size} values): {diff:.4e} (tol {tol:.0e}) {'ok' if diff <= tol else 'BAD'}")
    return 0 if diff <= tol else 1


def aligned_traces(ref_dir, run_dir):
    a, b = load_history(ref_dir), load_history(run_dir)
    ua, ub = a["totalDisplacement trace"], b["totalDisplacement trace"]
    if ua.shape != ub.shape: die(f"totalDisplacement trace: shape {ua.shape} vs {ub.shape}")
    ta, tb = np.squeeze(a["totalDisplacement Time"][:, 0]), np.squeeze(b["totalDisplacement Time"][:, 0])
    if not np.allclose(ta, tb, rtol=0, atol=1e-12): die(f"output times differ: {ta} vs {tb}")
    pa = a["totalDisplacement ReferencePosition trace"][0]; pb = b["totalDisplacement ReferencePosition trace"][0]
    if pa.shape != pb.shape: die(f"trace node count differs: {pa.shape} vs {pb.shape}")
    if not np.array_equal(pa, pb):
        # node ordering of the trace set differs across partitions: match nodes by reference position
        ia = np.lexsort(pa.T[::-1]); ib = np.lexsort(pb.T[::-1])
        if not np.allclose(pa[ia], pb[ib], rtol=0, atol=1e-12): die("trace node positions differ between the runs")
        ua, ub = ua[:, ia, :], ub[:, ib, :]
        print(f"    (trace nodes re-ordered by reference position: {len(ia)} nodes)")
    return ua, ub


def compare(ref_dir, run_dir, tol, info_tol, l2n_tol):
    ua, ub = aligned_traces(ref_dir, run_dir)
    num = np.abs(ua - ub).max(); den = np.abs(ua).max(); rel = num / den if den > 0 else num
    diff = geos_ats_diff(ub, ua)
    ok = True
    if tol is not None:
        ok &= rel <= tol
        print(f"    cross-run displacement history: max|u_ref - u| = {num:.4e}, max|u_ref| = {den:.4e}, rel L-inf = {rel:.3e} (tol {tol:.0e}) {'ok' if rel <= tol else 'BAD'}"
              + (f"; {'also within' if rel <= info_tol else 'not within'} {info_tol:.0e} (info)" if info_tol is not None else ""))
    if l2n_tol is not None:
        ok &= diff <= l2n_tol
        print(f"    geos-ats baseline curve metric ||u - u_ref||_2 / N = {diff:.4e} (tol {l2n_tol:.0e}) {'ok' if diff <= l2n_tol else 'BAD'}")
    elif tol is not None:
        print(f"    (info) geos-ats curve metric ||u - u_ref||_2 / N = {diff:.4e}")
    return 0 if ok else 1


# Same rules as geos-ats restart_check.py (geosPythonPackages, geos-ats/src/geos/ats/helpers/
# restart_check.py + permute_array.py): default exclusions EXCLUDE_DEFAULT; groups walked
# recursively with their attributes; an LvArray group (__dimensions__/__permutation__/__values__)
# is compared as the logical array -- values reshaped through the stored permutation on each side
# (a device build stores 2-D node/element fields in a different memory layout than a host build,
# so the raw __values__ and __permutation__ datasets differ while the arrays are equal); float
# entries agree iff |x-b| <= atol OR |x-b| <= rtol*|b| (the "(1+max|b|)" scaling is commented out
# upstream), integer/boolean and string data must be identical; a child present on one side only
# is an error.
import re
RESTART_EXCLUDE = [re.compile(p) for p in (".*/commandLine", ".*/schema$", ".*/globalToLocalMap", ".*/timeHistoryOutput.*/restart")]
# Compile-time defaults of GEOS's LinearSolverParameters that differ between device and host builds
# (src/coreComponents/linearAlgebra/utilities/LinearSolverParameters.hpp under GEOS_USE_HYPRE_DEVICE:
# krylov.maxRestart 100 vs 200, amg.coarseningType PMIS vs HMIS, amg.smootherType l1jacobi vs l1sgs).
# The public baseline was produced by a host build, so a GPU build legitimately stores other values
# for these three; they are reported but not gating (the direct-solver deck compared here does not
# use them, and every other dataset must still agree).
DEVICE_DEFAULTS = re.compile(r".*/LinearSolverParameters/(krylovMaxRestart|amgCoarseningType|amgSmootherType)(/__values__)?$")
LVARRAY_KEYS = ("__dimensions__", "__permutation__", "__values__")
INT_KINDS = set("?bBiumMV"); FLOAT_KINDS = set("fc"); STR_KINDS = set("SaU")


def permute_array(data, shape, perm):
    """geos-ats permute_array.permuteArray: stored (memory-order) values -> logical array of `shape`."""
    shape, perm = np.asarray(shape), np.asarray(perm)
    if shape.ndim != 1 or perm.ndim != 1 or shape.size != perm.size: return None, "shape/permutation are not 1-D of equal length"
    if np.prod(shape) != data.size: return None, f"shape {shape} (size {np.prod(shape)}) vs stored size {data.size}"
    if np.any(np.sort(perm) != np.arange(shape.size)): return None, f"invalid permutation {perm}"
    data = data.reshape(shape[perm])
    rev = np.empty_like(perm); rev[perm] = np.arange(perm.size)
    data = np.transpose(data, rev)
    if tuple(data.shape) != tuple(shape): return None, "reshape failed"
    return data, None


def restart(run_dir, baseline_dir, rtol, atol):
    def find_root(d):
        roots = sorted(glob.glob(os.path.join(d, "**", "*restart_*.root"), recursive=True))
        if not roots: die(f"no restart .root file under {d}")
        return roots[-1]
    ra, rb = find_root(run_dir), find_root(baseline_dir)
    print(f"    restart files: {os.path.basename(ra)} vs baseline {os.path.basename(rb)}")
    da, db = ra[:-5], rb[:-5]          # <name>.root -> <name>/ directory with rank_*.hdf5
    fa, fb = sorted(glob.glob(os.path.join(da, "rank_*.hdf5"))), sorted(glob.glob(os.path.join(db, "rank_*.hdf5")))
    if len(fa) != len(fb) or not fa: die(f"rank file count differs or empty: {len(fa)} vs {len(fb)} (baseline is for a fixed rank count)")
    stats = {"ncmp": 0, "nfail": 0, "nexcl": 0, "nlv": 0, "nperm": 0, "ndev": 0, "worst": 0.0, "worst_name": ""}

    def fail(msg): stats["nfail"] += 1; print(f"    {msg}")

    def cmp(path, A, B):
        A, B = np.asarray(A), np.asarray(B)
        if A.shape == (): A = A.reshape(1)
        if B.shape == (): B = B.reshape(1)
        stats["ncmp"] += 1
        ka, kb = A.dtype.kind, B.dtype.kind
        if A.size == 0 and B.size == 0: return
        if A.shape != B.shape: fail(f"{path}: shape {A.shape} vs {B.shape}"); return
        exact = (ka in INT_KINDS and kb in INT_KINDS) or (ka in STR_KINDS and kb in STR_KINDS) or ka == "O" or kb == "O"
        if exact:
            if not np.array_equal(A, B):
                what = f"{path}: exact ({A.dtype}) content differs" + (f": {A.flat[0]!r} (run) vs {B.flat[0]!r} (baseline)" if A.size == 1 else "")
                if DEVICE_DEFAULTS.match(path): stats["ndev"] += 1; print(f"    [device-build default, not gating] {what}")
                else: fail(what)
            return
        if ka in FLOAT_KINDS | INT_KINDS and kb in FLOAT_KINDS | INT_KINDS:
            A, B = A.astype(float), B.astype(float)
            if not (np.isfinite(A).all() and np.isfinite(B).all()): fail(f"{path}: non-finite values"); return
            diff = np.abs(A - B); bad = (diff > atol) & (diff > rtol * np.abs(B))
            if bad.any(): fail(f"{path}: {bad.sum()} of {bad.size} values outside atol {atol:g} / rtol {rtol:g} (max diff {diff.max():.3e})")
            if np.abs(B).max() > atol:       # relative figure only meaningful for non-vanishing fields
                m = diff.max() / np.abs(B).max()
                if m > stats["worst"]: stats["worst"], stats["worst_name"] = m, path
            return
        fail(f"{path}: unrecognised type combination {A.dtype} / {B.dtype}")

    def cmp_attrs(path, oa, ob):
        for k in sorted(set(oa.attrs) | set(ob.attrs)):
            if k not in oa.attrs or k not in ob.attrs: fail(f"{path}.attrs[{k}]: present on one side only"); continue
            cmp(f"{path}.attrs[{k}]", oa.attrs[k], ob.attrs[k])

    def walk(ga, gb):
        path = ga.name
        cmp_attrs(path, ga, gb)
        children = set(ga.keys()) | set(gb.keys())
        if all(k in ga and k in gb for k in LVARRAY_KEYS):
            for k in LVARRAY_KEYS: children.discard(k)
            stats["nlv"] += 1
            dims_a, dims_b = np.asarray(ga["__dimensions__"][()]), np.asarray(gb["__dimensions__"][()])
            perm_a, perm_b = np.asarray(ga["__permutation__"][()]), np.asarray(gb["__permutation__"][()])
            if dims_a.shape != dims_b.shape or np.any(dims_a != dims_b): fail(f"{path}: LvArray dimensions differ: {dims_a} vs {dims_b}")
            else:
                if not np.array_equal(perm_a, perm_b): stats["nperm"] += 1
                va, ea = permute_array(np.asarray(ga["__values__"][()]), dims_a, perm_a)
                vb, eb = permute_array(np.asarray(gb["__values__"][()]), dims_b, perm_b)
                if va is None: fail(f"{path}: cannot permute the run's LvArray: {ea}")
                elif vb is None: fail(f"{path}: cannot permute the baseline's LvArray: {eb}")
                else: cmp(path, va, vb)
        for name in sorted(children):
            p = path.rstrip("/") + "/" + name
            if any(r.match(p) for r in RESTART_EXCLUDE): stats["nexcl"] += 1; continue
            if name not in gb: fail(f"{p}: in the run but not in the baseline"); continue
            if name not in ga: fail(f"{p}: in the baseline but not in the run"); continue
            a, b = ga[name], gb[name]
            if isinstance(a, h5py.Group) != isinstance(b, h5py.Group): fail(f"{p}: group on one side, dataset on the other"); continue
            if isinstance(a, h5py.Group): walk(a, b)
            else: cmp(p, a[()], b[()]); cmp_attrs(p, a, b)

    for x, y in zip(fa, fb):
        with h5py.File(x, "r") as ha, h5py.File(y, "r") as hb:
            walk(ha, hb)
    print(f"    restart vs baseline: {stats['ncmp']} arrays/datasets/attributes compared over {len(fa)} rank file(s) "
          f"({stats['nlv']} LvArrays, {stats['nperm']} of them stored with a different memory permutation; {stats['nexcl']} children excluded by the "
          f"geos-ats default patterns; {stats['ndev']} device-build solver defaults reported, not gating), {stats['nfail']} disagree; "
          f"worst relative diff {stats['worst']:.3e} ({stats['worst_name']})")
    return 0 if stats["nfail"] == 0 else 1


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("analytic"); p.add_argument("run_dir"); p.add_argument("--tol", type=float, required=True); p.add_argument("--curve", required=True)
    p = sub.add_parser("compare"); p.add_argument("ref_dir"); p.add_argument("run_dir")
    p.add_argument("--tol", type=float, help="gate: relative L-inf of the displacement history vs ref")
    p.add_argument("--info-tol", type=float, help="also report whether this tighter relative L-inf is met (not gating)")
    p.add_argument("--l2n-tol", type=float, help="gate: geos-ats curve metric ||u - u_ref||_2 / N (baseline curve check)")
    p = sub.add_parser("restart"); p.add_argument("run_dir"); p.add_argument("baseline_dir"); p.add_argument("--rtol", type=float, required=True); p.add_argument("--atol", type=float, required=True)
    a = ap.parse_args()
    if a.cmd == "analytic": return analytic(a.run_dir, a.tol, a.curve)
    if a.cmd == "compare":
        if a.tol is None and a.l2n_tol is None: die("compare: give --tol and/or --l2n-tol")
        return compare(a.ref_dir, a.run_dir, a.tol, a.info_tol, a.l2n_tol)
    return restart(a.run_dir, a.baseline_dir, a.rtol, a.atol)


if __name__ == "__main__":
    sys.exit(main())
