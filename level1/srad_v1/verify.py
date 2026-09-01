#!/usr/bin/env python3
"""Standalone correctness check for Rodinia srad_v1 (upstream provides none).

Runs the benchmark, then recomputes the full SRAD pipeline (extract ->
niter x [statistics, srad, srad2] -> compress) in numpy float32, ported
directly from the extracted kernels, and compares image_out.pgm.

Usage: verify.py <exe> <niter> <lambda> <rows> <cols> <input_image>
"""
import subprocess, sys
import numpy as np

exe, niter, lam, Nr, Nc, img = (sys.argv[1], int(sys.argv[2]), float(sys.argv[3]),
                                int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
r = subprocess.run([exe, str(niter), str(lam), str(Nr), str(Nc), img],
                   capture_output=True, text=True)
if r.returncode != 0:
    print(f"FAIL: benchmark exited with {r.returncode}\n{r.stdout}{r.stderr}")
    sys.exit(1)

def read_pgm(path):
    tok = []
    with open(path) as f:
        for line in f:
            line = line.split("#")[0]
            tok.extend(line.split())
    assert tok[0] == "P2"
    w, h, _ = int(tok[1]), int(tok[2]), int(tok[3])
    return np.array(tok[4:4+w*h], dtype=np.float64).reshape(h, w), w, h

orig, ow, oh = read_pgm(img)
# resize.c: tiling copy (identity when sizes match); srad stores column-major
inp = np.empty((Nr, Nc), dtype=np.float32)
for i in range(Nr):
    for j in range(Nc):
        inp[i, j] = orig[i % oh, j % ow]

lam = np.float32(lam)
I = np.exp(inp.astype(np.float32) / np.float32(255))          # extract kernel
iN = np.maximum(np.arange(Nr) - 1, 0)
iS = np.minimum(np.arange(Nr) + 1, Nr - 1)
jW = np.maximum(np.arange(Nc) - 1, 0)
jE = np.minimum(np.arange(Nc) + 1, Nc - 1)
NeROI = Nr * Nc
for _ in range(niter):
    total = np.float32(I.sum(dtype=np.float64))
    total2 = np.float32((I.astype(np.float64)**2).sum())
    meanROI = total / np.float32(NeROI)
    varROI = total2 / np.float32(NeROI) - meanROI * meanROI
    q0sqr = np.float32(varROI / (meanROI * meanROI))
    Jc = I
    dN = I[iN, :] - Jc
    dS = I[iS, :] - Jc
    dW = I[:, jW] - Jc
    dE = I[:, jE] - Jc
    G2 = (dN*dN + dS*dS + dW*dW + dE*dE) / (Jc*Jc)
    L = (dN + dS + dW + dE) / Jc
    num = np.float32(0.5)*G2 - np.float32(1.0/16.0)*(L*L)
    den = np.float32(1) + np.float32(0.25)*L
    qsqr = num / (den*den)
    den = (qsqr - q0sqr) / (q0sqr * (np.float32(1) + q0sqr))
    c = np.float32(1.0) / (np.float32(1.0) + den)
    c = np.clip(c, np.float32(0), np.float32(1))
    cN = c
    cS = c[iS, :]
    cW = c
    cE = c[:, jE]
    D = cN*dN + cS*dS + cW*dW + cE*dE
    I = (I + np.float32(0.25)*lam*D).astype(np.float32)
I = np.log(I) * np.float32(255)                                # compress kernel

out, w, h = read_pgm("image_out.pgm")
ref = np.trunc(I).astype(np.int64)      # write_graphics casts to (int)
diff = np.abs(out.astype(np.int64).reshape(Nr, Nc) - ref)
frac_gt1 = (diff > 1).mean()
if diff.max() > 3 or frac_gt1 > 0.001:
    print(f"FAIL: image differs (max {diff.max()}, frac>1 {frac_gt1:.4%})")
    sys.exit(1)
print(f"PASS: srad output image matches CPU reference "
      f"(max pixel diff {diff.max()}, frac>1 {frac_gt1:.4%})")
