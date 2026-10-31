#!/usr/bin/env python3
"""Post-run correctness check of lud: upstream's lud_verify criterion on the factors the run captured.

    lud_residual_check.py --dump <capture dir> -- <lud arguments>     a run's capture (registry `check:`, verify.py)
    lud_residual_check.py --lu <file.f32> -- <lud arguments>          a raw float32 factor file (offline analysis)

The run captures its result after the timed region and the final device-to-host copy (cuda/lud.cu:
hpcperf_golden_output("lu", m, n*n, "f32") with HPCPERF_GOLDEN_DUMP=<dir>, tools/golden/hpcperf_golden.h):
the packed n x n factor matrix, unit-lower L strictly below the diagonal and U on and above it.

Criterion -- upstream's own (common/common.c lud_verify): tmp = L*U is recomposed with the unit diagonal of
L, and every one of the n*n elements with fabs(A_ij - tmp_ij) > 0.0001 prints "dismatch". Here:

    PASS  <=>  every factor is finite  and  max_ij |A_ij - (L U)_ij| <= 1e-4      (all n*n elements)

* A is regenerated from create_matrix (common/common.c), not read from the run: the same float32
  operations in the same order (coe_i = (float)(10 * exp((double)(lamda * (float)i))), lamda = -0.001f,
  libm exp, A_ij = coe[|i - j|]), bit-identical to the program's matrix (verified for the registered
  sizes, README). The check therefore judges the factors of the REGISTERED matrix.
* L*U is evaluated in float64 with BLAS (numpy matmul = dgemm), the difference and the comparison in
  float64; 1e-4 is the same double constant. Upstream accumulates L*U in float32 in a
  single thread (O(n^3), 922-969 s at -s 8192); its own rounding error (measured up to 1.9e-5) is part of its
  verdict, this one's (< 1e-11) is not. Both residuals are quantified per input in README.md ("Validation").
* A non-finite factor is a FAIL. Upstream's comparison is false for NaN (fabs(NaN) > 1e-4 is false), so
  upstream's check passes NaN factors. That is the one intended difference in verdict; the other possible one
  is an element whose exact residual lies within upstream's own float32 accumulation error of 1e-4 (README).
* -i <matrix file> is not supported (no registered input uses it).

Upstream's single-thread host check itself stays available offline: HPCPERF_CPU_REFERENCE=1 lud_cuda ... -v.
Exit 0 only with the PASS line; 1 FAIL; 2 the check could not be performed (bad arguments / capture).
"""
import hashlib
import json
import math
import os
import resource
import sys
import time

import numpy as np

T_START = time.time()     # reset by main(): the in-process time of the check (interpreter start-up excluded)

TOL = 0.0001              # common.c lud_verify: fabs(m - tmp) > 0.0001 (a double constant)
CAPTURE_NAME = "lu"       # cuda/lud.cu hpcperf_golden_output("lu", ...)
MAX_LISTED = 10           # offending elements listed (upstream prints every one, row-major)
NO_FILE = "the -i <matrix file> form is not supported by this check (no registered input uses it)"


class CheckError(Exception):
    """The check cannot be performed (exit 2; no PASS line is printed)."""


def parse_lud_args(args):
    """The matrix size of the lud command line, parsed like its getopt_long("::vs:i:") loop: -s N / -sN /
    --size N / --size=N, -v / --verify, clustered short options (-vs 256). Default size 32 (cuda/lud.cu)."""
    n, i, verify = 32, 0, False
    args = [str(a) for a in args]
    while i < len(args):
        a = args[i]
        if a in ("--verify",):
            verify = True
        elif a.startswith("--size"):
            if a == "--size":
                i += 1
                if i >= len(args):
                    raise CheckError("--size needs a value")
                v = args[i]
            elif a.startswith("--size="):
                v = a[len("--size="):]
            else:
                raise CheckError(f"unknown lud argument {a!r}")
            n = _size(v)
        elif a.startswith("--input"):
            raise CheckError(NO_FILE)
        elif a.startswith("-") and len(a) > 1 and not a.startswith("--"):
            k = 1
            while k < len(a):
                c = a[k]
                if c == "v":
                    verify = True; k += 1
                elif c == "i":
                    raise CheckError(NO_FILE)
                elif c == "s":
                    v = a[k + 1:]
                    if not v:
                        i += 1
                        if i >= len(args):
                            raise CheckError("-s needs a value")
                        v = args[i]
                    n = _size(v)
                    break
                else:
                    raise CheckError(f"unknown lud option -{c} in {a!r}")
        else:
            raise CheckError(f"unexpected lud argument {a!r} (lud takes no positional arguments)")
        i += 1
    return n, verify


def _size(v):
    try:
        n = int(v)
    except ValueError:
        raise CheckError(f"matrix size {v!r} is not an integer")
    if n <= 0:
        raise CheckError(f"matrix size {n} must be positive")
    return n


def regenerate_matrix(n):
    """create_matrix(&m, n), common/common.c, bit for bit: float lamda = -0.001; coe_i = 10*exp(lamda*i) with
    lamda*i a float product, exp the C library's double exp (Python's math.exp calls the same libm), the
    double 10*exp(..) rounded to float on assignment; coe[n-1+i] = coe[n-1-i] = coe_i; m[i*n + j] = coe[n-1-i+j]
    (a symmetric Toeplitz matrix). Returned as a read-only strided view of coe with exactly that indexing."""
    lam = np.float32(-0.001)
    coe = np.empty(2 * n - 1, dtype=np.float32)
    for i in range(n):
        x = float(lam * np.float32(i))           # float32 product, as in C (FLT_EVAL_METHOD 0)
        c = np.float32(10.0 * math.exp(x))       # double 10*exp(x), rounded to nearest float
        coe[n - 1 + i] = c
        coe[n - 1 - i] = c
    s = coe.itemsize                             # row i starts at coe[n-1-i]: stride -1 element per row
    return np.lib.stride_tricks.as_strided(coe[n - 1:], shape=(n, n), strides=(-s, s), writeable=False)


def load_capture(dump, n):
    """The factor matrix from a capture directory (manifest.jsonl + lu.bin), its integrity verified."""
    man = os.path.join(dump, "manifest.jsonl")
    if not os.path.isfile(man):
        raise CheckError(f"no capture manifest {man} (was the run made with HPCPERF_GOLDEN_DUMP={dump}?)")
    recs = []
    with open(man) as fh:
        for ln in fh:
            if ln.strip():
                try:
                    recs.append(json.loads(ln))
                except ValueError:
                    raise CheckError(f"{man}: unreadable line {ln.strip()[:80]!r}")
    mine = [r for r in recs if r.get("name") == CAPTURE_NAME]
    if len(mine) != 1:
        raise CheckError(f"{man}: {len(mine)} records named {CAPTURE_NAME!r} (exactly one expected)")
    r = mine[0]
    if r.get("dtype") != "f32" or r.get("data") is not True:
        raise CheckError(f"{man}: {CAPTURE_NAME!r} must be f32 data (got dtype {r.get('dtype')}, data {r.get('data')}; HPCPERF_GOLDEN_MODE=hash is not checkable)")
    if r.get("count") != n * n or r.get("bytes") != 4 * n * n:
        raise CheckError(f"{man}: {CAPTURE_NAME!r} has {r.get('count')} elements; the input (-s {n}) needs {n * n}")
    path = os.path.join(dump, CAPTURE_NAME + ".bin")
    lu, sha = load_raw(path, n)
    if sha != r.get("sha256"):
        raise CheckError(f"{path}: sha256 {sha} differs from the manifest's {r.get('sha256')} (partial or altered capture)")
    return lu, path, sha


def load_raw(path, n):
    if not os.path.isfile(path):
        raise CheckError(f"no factor file {path}")
    if os.path.getsize(path) != 4 * n * n:
        raise CheckError(f"{path}: {os.path.getsize(path)} bytes; the input (-s {n}) needs {4 * n * n}")
    with open(path, "rb") as fh:
        raw = fh.read()
    return np.frombuffer(raw, dtype="<f4").reshape(n, n), hashlib.sha256(raw).hexdigest()


def blas_info():
    """Best effort: the BLAS library numpy/scipy loaded and its thread count (OpenBLAS only)."""
    try:
        import ctypes
        libs = sorted({ln.split()[-1] for ln in open("/proc/self/maps") if "openblas" in ln.lower() and ".so" in ln})
        if libs:
            lib = ctypes.CDLL(libs[0])
            lib.openblas_get_config.restype = ctypes.c_char_p
            return f"{lib.openblas_get_config().decode()} threads={lib.openblas_get_num_threads()}"
    except Exception:  # noqa: BLE001 - informational only
        pass
    return "unknown BLAS (thread count not reported)"


def recompose(lu):
    """L*U in float64 (numpy matmul = BLAS dgemm): L the unit-lower factor stored strictly below the
    diagonal, U the factor on and above it. The float32 factors are exact in float64."""
    lo = np.tril(lu, -1).astype(np.float64)
    np.fill_diagonal(lo, 1.0)
    up = np.triu(lu).astype(np.float64)
    return lo @ up, "dgemm"


def check(lu, n, source):
    t0 = time.time()
    print(f"lud residual check: n={n}, upstream lud_verify criterion |A_ij - (L U)_ij| <= {TOL:g} for all {n * n} "
          f"elements (common/common.c), evaluated in float64; factors: {source}")
    bad = ~np.isfinite(lu)
    nbad = int(bad.sum())
    if nbad:
        i, j = (int(v) for v in np.argwhere(bad)[0])
        print(f"FAIL: lud residual: {nbad} non-finite value(s) in the captured factors (first at ({i}, {j}): {lu[i, j]}) (n={n})")
        return 1
    a = regenerate_matrix(n)
    t1 = time.time()
    r, how = recompose(lu)
    t2 = time.time()
    np.subtract(a, r, out=r)          # r := A - L U (float64; A's float32 values are exact in float64)
    np.abs(r, out=r)
    if not np.isfinite(r).all():
        k = int((~np.isfinite(r)).sum())
        print(f"FAIL: lud residual: A - L*U overflows at {k} element(s) (n={n})")
        return 1
    flat = int(np.argmax(r))
    worst, (wi, wj) = float(r.flat[flat]), divmod(flat, n)
    over = r > TOL
    nover = int(over.sum())
    rms = math.sqrt(float(np.vdot(r.ravel(), r.ravel())) / r.size)
    t3 = time.time()
    print(f"  max |A - L U| = {worst:.3e} at ({wi}, {wj}); RMS {rms:.3e}; elements > {TOL:g}: {nover}")
    print(f"  time: regenerate A {t1 - t0:.3f} s, L*U ({how}, {blas_info()}) {t2 - t1:.3f} s, compare {t3 - t2:.3f} s, "
          f"in-process total {t3 - T_START:.3f} s; peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
    if nover:
        for i, j in np.argwhere(over)[:MAX_LISTED]:   # upstream's diagnostic, in its row-major order and format
            i, j = int(i), int(j)
            print(f"  dismatch at ({i}, {j}): (o){float(a[i, j]):f} (n){_recomposed(lu, i, j):f}")
        if nover > MAX_LISTED:
            print(f"  ... {nover - MAX_LISTED} more")
        print(f"FAIL: lud residual max|A - L*U| = {worst:.3e} > {TOL:g} at {nover} of {n * n} elements (n={n})")
        return 1
    print(f"PASS: lud residual max|A - L*U| = {worst:.3e} <= {TOL:g} (n={n}, all {n * n} elements)")
    return 0


def _recomposed(lu, i, j):
    """(L U)_ij in float64 for one element (the diagnostic lines)."""
    k = min(i, j)
    l = lu[i, :k + 1].astype(np.float64)
    if i <= j:
        l[i] = 1.0
    return float(l @ lu[:k + 1, j].astype(np.float64))


def main(argv):
    global T_START
    T_START = time.time()
    if "--" in argv:
        cut = argv.index("--")
        opts, lud_args = argv[:cut], argv[cut + 1:]
    else:
        opts, lud_args = argv, []
    if len(opts) != 2 or opts[0] not in ("--dump", "--lu"):
        print(__doc__)
        return 2
    try:
        n, _verify = parse_lud_args(lud_args)
        t = time.time()
        if opts[0] == "--dump":
            lu, path, sha = load_capture(opts[1], n)
            src = f"{path} (sha256 {sha[:16]}.. = manifest; read + verified in {time.time() - t:.3f} s)"
        else:
            lu, sha = load_raw(opts[1], n)
            src = f"{opts[1]} (sha256 {sha[:16]}..; read in {time.time() - t:.3f} s)"
    except CheckError as ex:
        print(f"lud residual check: ERROR: {ex}")
        return 2
    return check(lu, n, src)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
