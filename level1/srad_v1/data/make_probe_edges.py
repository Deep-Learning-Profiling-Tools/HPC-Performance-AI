#!/usr/bin/env python3
"""Generates data/probe_edges.pgm, the input of the srad_v1 correctness probe `probe-edges-n20` (inputs.yaml).

Deterministic (integer hash, no random-number library), so the file is reproducible bit for bit:
    python3 level1/srad_v1/data/make_probe_edges.py level1/srad_v1/data/probe_edges.pgm
502 rows x 458 columns -- the size cuda/main.cu reads (image_ori_rows / image_ori_cols) -- ASCII P2 with the same
3-line header as data/image.pgm. A textured dark background (grey 50..70 with a +-40 texture: every pixel evolves, there
is no flat region) carries small high-contrast objects (grey 200 +-40; squares of 8..16 px, discs, lines of 1..3 px;
1.9 % of the pixels). The SRAD diffusion coefficient c stays far below 1 at the object edges for the whole run while
the texture is diffused, and the diffusing texture makes the image statistic q0sqr change quickly (0.0187 -> 0.0083 in
20 iterations, 24 % in the first): plain diffusion (c = 1), a constant, stale or once-computed q0sqr or a wrong
reduction change the image by grey levels to tens of grey levels, while every float32 summation order of the statistic
(including Rodinia 3.1's OpenMP host loop) stays within about 0.1. The texture amplitude was raised from 8 to 40 on
2026-10-28: with +-8 q0sqr changed only 1-4 % per iteration and a float32 recursive sum of the statistic was off by up
to 8 %, so the two could not be told apart. Background, measurements and rationale: README.md "Correctness probe".
"""
import sys
import numpy as np

R, C = 502, 458


def hash_noise(i, j, amp, salt=0):
    """Integer texture in [-amp, amp]: a multiplicative/xor-shift hash of the pixel coordinates."""
    h = ((i.astype(np.uint64) + np.uint64(salt)) * np.uint64(73856093)) ^ (j.astype(np.uint64) * np.uint64(19349663))
    h = (h ^ (h >> np.uint64(13))) * np.uint64(0x5bd1e995) & np.uint64(0xffffffff)
    h = h ^ (h >> np.uint64(15))
    return (h % np.uint64(2 * amp + 1)).astype(np.int64) - amp


def make(bg=50, fg=200, amp=40):
    i, j = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
    img = bg + (20 * i) // (R - 1) + hash_noise(i, j, amp)                     # background bg..bg+20 + texture +-amp
    obj = np.zeros((R, C), bool)
    for k, (ci, cj) in enumerate([(60, 60), (60, 200), (60, 380), (180, 120), (180, 300), (300, 60), (300, 220),
                                  (300, 400), (420, 140), (420, 330)]):
        s = 8 + 2 * (k % 5)                                                        # squares of 8..16 px
        obj |= (np.abs(i - ci) <= s // 2) & (np.abs(j - cj) <= s // 2)
    for (ci, cj, r) in [(120, 250, 10), (240, 380, 7), (360, 300, 12), (470, 60, 6)]:
        obj |= (i - ci) ** 2 + (j - cj) ** 2 <= r * r                             # discs
    obj |= (np.abs(i - 2 * j + 100) <= 1) & (i > 10) & (i < 490)                  # 3 px diagonal line
    obj |= (j == 430) & (i > 30) & (i < 470)                                       # 1 px vertical line
    obj |= (i >= 240) & (i < 242) & (j > 20) & (j < 200)                           # 2 px horizontal line
    img = np.where(obj, fg + hash_noise(i, j, amp, salt=7), img)
    return np.clip(img, 0, 255).astype(np.int64)


def main(out):
    a = make()
    with open(out, "w") as f:
        f.write(f"P2\n{C} {R}\n255\n")
        for row in a:
            f.write(" ".join(str(int(v)) for v in row) + " \n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "probe_edges.pgm")
