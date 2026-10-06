#!/usr/bin/env python3
"""Compare a refined-mesh homogeneous_halfspace run with upstream's reference seismograms (REF_SEIS).

The refined registered inputs (run.sh strong mode, G = 2: xmeshfem3D on the same meshfem3D_files with NEX x2, DT / 2)
solve the SAME physical problem as the shipped example -- same domain, medium, CMTSOLUTION, STATIONS, Stacey
boundaries and t0 -- on a finer discretisation; REF_SEIS was produced by the same internal mesher on the default mesh,
which already resolves periods below the source half duration (REF_SEIS/output_solver.txt: minimum period resolved
3.125 s vs half duration 5 s). Upstream's own criterion, utils/scripts/compare_seismogram_correlations.py, is applied
UNCHANGED (correlation >= 0.8, L2 misfit <= 0.01, time shift <= 0.01 s, every one of the 12 reference traces compared).

One deterministic preprocessing step is needed because the upstream script truncates by sample INDEX and only warns on
a dt mismatch: every trace of the run is decimated by k = dt_ref / dt_run (an integer, checked) after checking that its
samples fall exactly on the reference time grid (same t0), and both the run and the reference are truncated to their
common time window. The window covered is printed; a run shorter than the reference is compared over its own record
length only (refine2-1000: the first 25 s), which the verdict line states.

usage: check_refined_vs_ref_seis.py <OUTPUT_FILES dir> <REF_SEIS dir> <compare_seismogram_correlations.py> <work dir>
Prints 'PASS: ...' or 'VALIDATION ERROR: ...'; exit 0 only on PASS."""
import glob, math, os, re, subprocess, sys

def load(p):
    t, v = [], []
    for ln in open(p):
        parts = ln.split()
        if len(parts) >= 2:
            a, b = float(parts[0]), float(parts[1])
            if not (math.isfinite(a) and math.isfinite(b)):
                raise ValueError(f"{os.path.basename(p)}: non-finite sample")
            t.append(a); v.append(b)
    if len(t) < 2:
        raise ValueError(f"{os.path.basename(p)}: fewer than 2 samples")
    return t, v

def main():
    out, ref, cmp_script, work = sys.argv[1:5]
    try:
        refs = sorted(glob.glob(os.path.join(ref, "*.semd")))
        if not refs:
            raise ValueError(f"no reference traces in {ref}")
        dsyn, dref = os.path.join(work, "syn"), os.path.join(work, "ref")
        os.makedirs(dsyn, exist_ok=True); os.makedirs(dref, exist_ok=True)
        windows = set()
        for rp in refs:
            name = os.path.basename(rp); sp = os.path.join(out, name)
            if not os.path.isfile(sp):
                raise ValueError(f"run produced no trace {name}")
            tr, vr = load(rp); ts, vs = load(sp)
            # dt over the whole record: the time column is printed with limited precision (e.g. 0.0249996 for 0.025)
            dt_r, dt_s = (tr[-1] - tr[0]) / (len(tr) - 1), (ts[-1] - ts[0]) / (len(ts) - 1)
            k = dt_r / dt_s
            if abs(k - round(k)) > 1e-4 or round(k) < 1:
                raise ValueError(f"{name}: run dt {dt_s:g} does not divide reference dt {dt_r:g}")
            k = int(round(k))
            if abs(ts[0] - tr[0]) > 1e-2 * dt_r:
                raise ValueError(f"{name}: run starts at {ts[0]:g}, reference at {tr[0]:g}")
            ts, vs = ts[::k], vs[::k]
            n = min(len(ts), len(tr))
            for i in range(n):
                if abs(ts[i] - tr[i]) > 1e-2 * dt_r:      # exact grid alignment up to the printed precision
                    raise ValueError(f"{name}: decimated sample {i} at t={ts[i]:g} is not on the reference grid ({tr[i]:g})")
            windows.add((n, tr[0], tr[n - 1], k))
            with open(os.path.join(dsyn, name), "w") as f:
                f.writelines(f"{ts[i]:.6f} {vs[i]:.9e}\n" for i in range(n))
            with open(os.path.join(dref, name), "w") as f:
                f.writelines(f"{tr[i]:.6f} {vr[i]:.9e}\n" for i in range(n))
        if len(windows) != 1:
            raise ValueError(f"traces cover different windows: {sorted(windows)}")
        n, t0, t1, k = windows.pop()
        print(f"    {len(refs)} traces decimated by {k} onto the REF_SEIS grid; common window t = {t0:g} .. {t1:g} s ({n} samples)")
        r = subprocess.run([sys.executable, cmp_script, dsyn + "/", dref + "/"], capture_output=True, text=True)
        log = r.stdout + r.stderr
        open(os.path.join(work, "compare_ref_seis.log"), "w").write(log)
        for ln in log.splitlines():
            if ln.startswith("|") or re.search(r"seismograms compared|poor correlation|poor match|significant time shift|no poor|no significant", ln):
                print("    " + ln)
        m = re.search(r"^(\d+) seismograms compared", log, re.M)
        ncmp = int(m.group(1)) if m else 0
        ok = (ncmp == len(refs) and "no poor correlations found" in log and "no poor matches found" in log
              and "no significant time shifts found" in log)
        if not ok:
            raise ValueError(f"upstream comparison failed ({ncmp} of {len(refs)} traces compared; see {work}/compare_ref_seis.log)")
        print(f"PASS: SPECFEM3D refined mesh vs REF_SEIS ({ncmp}/{len(refs)} traces, window {t0:g}..{t1:g} s, corr>=0.8 err<=1% shift<=0.01s, upstream script unchanged)")
        return 0
    except (OSError, ValueError) as ex:
        print(f"VALIDATION ERROR: {ex}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
