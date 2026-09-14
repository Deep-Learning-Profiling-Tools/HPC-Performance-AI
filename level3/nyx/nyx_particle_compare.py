#!/usr/bin/env python3
"""Compare Nyx dark-matter particles between two runs that may have used a
DIFFERENT number of MPI ranks.

AMReX's own particle_compare (Tools/Postprocessing/C_Src) compares particle
files chunk by chunk and requires identical headers, including `next_id` and the
per-file layout -- i.e. the same process count. Across rank counts Nyx assigns
particle (id, cpu) pairs per reading rank, so ids differ as well. Particle
identity is therefore established through the initial checkpoint: every particle
of a run is keyed by its exact (bit-identical) t=0 position, read from the
checkpoint written at step 0 with the same (id, cpu) as the final checkpoint.

    nyx_particle_compare.py <run_dir_A> <run_dir_B> <final_step> [--rel_tol R] [--abs_tol A]

Reads <run>/chk00000/DM and <run>/chk<final>/DM (checkpoint format: ids present).
Reports, per real component (position_x/y/z, mass, xvel, yvel, zvel), the maximum
absolute difference and the relative difference max|a-b| / max|a| (the same
definition AMReX's particle_compare prints), and exits 0 only if every component
satisfies abs <= abs_tol or rel <= rel_tol (particle_compare semantics: with
abs_tol 0 the relative tolerance alone decides). Exit 2 on any structural problem
(count mismatch, unmatched particle, unreadable file, non-finite data).
"""
import argparse, math, os, struct, sys
import numpy as np


def read_particle_dir(pdir):
    """Return (header dict, ints[n, num_int] int32, reals[n, num_real] float64)."""
    lines = [l.strip() for l in open(os.path.join(pdir, "Header")) if l.strip()]
    i = 0
    version = lines[i]; i += 1
    ndim = int(lines[i]); i += 1
    nreal_extra = int(lines[i]); i += 1
    real_names = lines[i:i + nreal_extra]; i += nreal_extra
    nint_extra = int(lines[i]); i += 1
    int_names = lines[i:i + nint_extra]; i += nint_extra
    is_chk = int(lines[i]); i += 1
    nparticles = int(lines[i]); i += 1
    next_id = int(lines[i]); i += 1
    finest = int(lines[i]); i += 1
    ngrids = [int(lines[i + l]) for l in range(finest + 1)]; i += finest + 1
    grids = []  # (level, which, count, where)
    for lev in range(finest + 1):
        for g in range(ngrids[lev]):
            w, c, o = lines[i].split(); i += 1
            grids.append((lev, int(w), int(c), int(o)))
    if "single" in version:
        raise SystemExit(f"{pdir}: single-precision particle files are not expected here")
    num_int = 2 * is_chk + nint_extra
    num_real = ndim + nreal_extra
    ints = np.zeros((nparticles, num_int), dtype=np.int32)
    reals = np.zeros((nparticles, num_real), dtype=np.float64)
    pos = 0
    for lev, which, count, where in grids:
        if count == 0:
            continue
        fn = os.path.join(pdir, f"Level_{lev}", f"DATA_{which:05d}")
        with open(fn, "rb") as f:
            f.seek(where)
            ib = f.read(4 * num_int * count)
            rb = f.read(8 * num_real * count)
        if len(ib) != 4 * num_int * count or len(rb) != 8 * num_real * count:
            raise SystemExit(f"{fn}: short read at offset {where} (grid count {count})")
        if num_int:
            ints[pos:pos + count] = np.frombuffer(ib, dtype="<i4").reshape(count, num_int)
        reals[pos:pos + count] = np.frombuffer(rb, dtype="<f8").reshape(count, num_real)
        pos += count
    if pos != nparticles:
        raise SystemExit(f"{pdir}: header says {nparticles} particles, grids hold {pos}")
    names = [f"DM_position_{'xyz'[d]}" for d in range(ndim)] + [f"DM_{n}" for n in real_names]
    hdr = dict(version=version, ndim=ndim, is_checkpoint=is_chk, nparticles=nparticles, next_id=next_id,
               num_int=num_int, num_real=num_real, names=names, int_names=(["id", "cpu"] if is_chk else []) + int_names)
    return hdr, ints, reals


def keyed_final(run, final_step):
    """Map each particle's exact initial position -> its final real components (via id,cpu)."""
    h0, i0, r0 = read_particle_dir(os.path.join(run, "chk00000", "DM"))
    h1, i1, r1 = read_particle_dir(os.path.join(run, f"chk{final_step:05d}", "DM"))
    for h, tag in ((h0, "chk00000"), (h1, f"chk{final_step:05d}")):
        if not h["is_checkpoint"]:
            raise SystemExit(f"{run}/{tag}: not a checkpoint (no particle ids) -- run.sh must write checkpoints")
    if h0["nparticles"] != h1["nparticles"]:
        raise SystemExit(f"{run}: particle count changed {h0['nparticles']} -> {h1['nparticles']}")
    if not (np.isfinite(r0).all() and np.isfinite(r1).all()):
        raise SystemExit(f"{run}: non-finite particle data")
    # (id, cpu) -> row, then order the final data by the initial-position key
    key0 = i0[:, 0].astype(np.int64) * (1 << 32) + i0[:, 1].astype(np.int64)
    key1 = i1[:, 0].astype(np.int64) * (1 << 32) + i1[:, 1].astype(np.int64)
    if len(np.unique(key0)) != len(key0) or len(np.unique(key1)) != len(key1):
        raise SystemExit(f"{run}: duplicate (id,cpu) pairs")
    order1 = np.argsort(key1); k1s = key1[order1]
    idx = np.searchsorted(k1s, key0)
    if not np.array_equal(k1s[idx], key0):
        raise SystemExit(f"{run}: a particle of chk00000 is missing in the final checkpoint")
    final_by_initial_row = r1[order1[idx]]          # row j: final data of the particle that was row j initially
    init_pos = r0[:, :h0["ndim"]]
    # canonical order: lexicographic on the exact initial position (identical across runs by construction)
    canon = np.lexsort([init_pos[:, d] for d in reversed(range(h0["ndim"]))])
    return h1, init_pos[canon], final_by_initial_row[canon]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a"); ap.add_argument("run_b"); ap.add_argument("final_step", type=int)
    ap.add_argument("--rel_tol", type=float, default=0.0); ap.add_argument("--abs_tol", type=float, default=0.0)
    a = ap.parse_args()
    ha, pa, fa = keyed_final(a.run_a, a.final_step)
    hb, pb, fb = keyed_final(a.run_b, a.final_step)
    if ha["names"] != hb["names"] or ha["nparticles"] != hb["nparticles"]:
        raise SystemExit(f"component/count mismatch: {ha['names']} ({ha['nparticles']}) vs {hb['names']} ({hb['nparticles']})")
    if not np.array_equal(pa, pb):
        raise SystemExit("initial (t=0) particle positions differ between the runs -- cannot establish identity")
    print(f" {ha['nparticles']} particles matched through their exact initial positions; comparing chk{a.final_step:05d}")
    print(f" {'component':<20} {'abs error':>16} {'rel error':>16}")
    ok = True
    for j, name in enumerate(ha["names"]):
        d = np.abs(fa[:, j] - fb[:, j]).max()
        ref = np.abs(fa[:, j]).max()
        rel = d / ref if ref > 0 else d
        print(f" {name:<20} {d:16.8e} {rel:16.8e}")
        if d > a.abs_tol and rel > a.rel_tol:
            ok = False
    if ok:
        print(f" PARTICLES AGREE to relative tolerance {a.rel_tol:g}" + (f" and/or absolute tolerance {a.abs_tol:g}" if a.abs_tol > 0 else ""))
        return 0
    print(f" PARTICLES DISAGREE to relative tolerance {a.rel_tol:g}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
