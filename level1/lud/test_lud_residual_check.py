#!/usr/bin/env python3
"""CPU-only tests of lud_residual_check.py (no GPU, no build): python3 level1/lud/test_lud_residual_check.py

Exact pivot-free factors of the registered matrix (float64 LU, rounded to float32) must PASS; a perturbed
element above the threshold, a non-finite factor, the Crout convention and an unfactored matrix must FAIL; a
missing, partial, hash-only or wrongly sized capture and unsupported arguments must be refused (exit 2)."""
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lud_residual_check as lrc  # noqa: E402

FAILED = []


def expect(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


def lu_nopiv(a):
    m = a.astype(np.float64).copy()
    for k in range(m.shape[0] - 1):
        m[k + 1:, k] /= m[k, k]
        m[k + 1:, k + 1:] -= np.outer(m[k + 1:, k], m[k, k + 1:])
    return m


def capture(d, arr, name="lu", dtype="f32", data=True, sha=None, count=None):
    os.makedirs(d, exist_ok=True)
    raw = np.ascontiguousarray(arr, dtype="<f4").tobytes()
    if data:
        open(os.path.join(d, name + ".bin"), "wb").write(raw)
    rec = {"name": name, "dtype": dtype, "count": count if count is not None else arr.size, "bytes": len(raw),
           "sha256": sha or hashlib.sha256(raw).hexdigest(), "data": data}
    with open(os.path.join(d, "manifest.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def run(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = lrc.main(argv)
    return rc, out.getvalue()


def main():
    n = 128
    a = np.ascontiguousarray(lrc.regenerate_matrix(n))
    # create_matrix's own definition, independently: coe_|i-j| = (float)(10 exp((float)(-0.001f * i)))
    c = np.array([np.float32(10.0 * np.exp(np.float64(np.float32(-0.001) * np.float32(i)))) for i in range(n)], np.float32)
    expect("A is the symmetric Toeplitz matrix of create_matrix", np.array_equal(a, c[np.abs(np.subtract.outer(np.arange(n), np.arange(n)))]))
    good = lu_nopiv(a).astype(np.float32)
    with tempfile.TemporaryDirectory() as t:
        def case(label, arr, want_rc, want=None, args=("-s", str(n), "-v"), **kw):
            d = os.path.join(t, label)
            capture(d, arr, **kw)
            rc, txt = run(["--dump", d, "--", *args])
            expect(f"{label}: exit {want_rc}", rc == want_rc and (want is None or want in txt), txt.strip().splitlines()[-1] if txt.strip() else "no output")
            return txt
        txt = case("exact factors", good, 0, "PASS: lud residual")
        r = float(txt.split("max|A - L*U| = ")[1].split()[0])
        expect("exact factors: residual far below 1e-4", r < 1e-5, r)
        for (i, j, dlt, rc, label) in ((40, 90, 2e-4, 1, "U element +2e-4"), (40, 90, 5e-5, 0, "U element +5e-5"),
                                       (90, 40, 2e-4 / float(np.abs(good[40, 40:]).max()), 1, "L element above"),
                                       (90, 40, 5e-5 / float(np.abs(good[40, 40:]).max()), 0, "L element below")):
            p = good.copy(); p[i, j] += np.float32(dlt)
            case(label, p, rc, "PASS" if rc == 0 else "FAIL: lud residual max")
        for v in (np.nan, np.inf, -np.inf):
            p = good.copy(); p[7, 9] = v
            case(f"factor {v}", p, 1, "non-finite")
        d64 = np.diag(lu_nopiv(a)).copy(); f = lu_nopiv(a)
        crout = np.tril(f * d64[None, :], -1) + np.diag(d64) + np.triu(f / d64[:, None], 1)
        case("Crout convention", crout.astype(np.float32), 1, "FAIL: lud residual max")
        case("unfactored matrix", a, 1, "FAIL: lud residual max")
        p = good.copy(); p[-16:, -16:] = a[-16:, -16:]
        case("last diagonal block left unfactored", p, 1, "FAIL")
        case("another size than the input", good, 2, "needs", args=("-s", str(n * 2)))
        case("sha256 differs from the manifest", good, 2, "partial or altered", sha="0" * 64)
        case("hash-only capture", good, 2, "hash", data=False)
        case("another output name", good, 2, "exactly one", name="lux")
        case("-i file input", good, 2, "not supported", args=("-i", "m.dat"))
        case("clustered -vs N", good, 0, "PASS", args=("-vs", str(n)))
        case("--size=N", good, 0, "PASS", args=(f"--size={n}", "--verify"))
        case("unknown option", good, 2, "unknown", args=("-s", str(n), "-x"))
        rc, txt = run(["--dump", os.path.join(t, "nothing"), "--", "-s", str(n)])
        expect("missing capture: exit 2", rc == 2 and "no capture manifest" in txt)
        d = os.path.join(t, "twice"); capture(d, good); capture(d, good)
        rc, txt = run(["--dump", d, "--", "-s", str(n)])
        expect("two records of the output: exit 2", rc == 2)
    print(f"{'ALL PASSED' if not FAILED else str(len(FAILED)) + ' FAILED: ' + ', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
