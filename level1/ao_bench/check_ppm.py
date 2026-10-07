#!/usr/bin/env python3
"""Compare aobench's rendered image with the trusted reference image of the same input.

    check_ppm.py <ao.ppm> --reference <reference.sha256>

The reference file holds the sha256 of ao.ppm rendered by the unoptimized reference build of
this repository for the same iteration count (its provenance -- build, commit, date -- is in the
sidecar <reference.sha256>.txt written when it was captured). The render is deterministic
(per-pixel RNG seeds, verify.py), so the comparison is exact: the image must be a well-formed
P6 PPM of the expected size and byte-identical to the reference. A candidate that reorders
floating-point work will differ; any tolerance for that needs a stated basis and is not set here.
Prints "PASS: aobench image check ..." / "FAIL: aobench image check ..."; exit 0 / 1.
"""
import hashlib, os, sys


def main(argv):
    if len(argv) < 2 or "--reference" not in argv:
        print(__doc__); return 2
    img, ref = argv[1], argv[argv.index("--reference") + 1]
    if not os.path.isfile(img):
        print(f"FAIL: aobench image check: {img} missing"); return 1
    data = open(img, "rb").read()
    if not data.startswith(b"P6"):
        print("FAIL: aobench image check: not a P6 PPM"); return 1
    parts = data.split(b"\n", 3)
    try:
        w, h = map(int, parts[1].split())
    except (ValueError, IndexError):
        print("FAIL: aobench image check: malformed PPM header"); return 1
    if len(parts) < 4 or len(parts[3]) != w * h * 3:
        print(f"FAIL: aobench image check: pixel payload {len(parts[3]) if len(parts) > 3 else 0} != {w}*{h}*3"); return 1
    got = hashlib.sha256(data).hexdigest()
    print(f"   {img}: {w}x{h} P6, sha256 {got}")
    if not os.path.isfile(ref):
        print(f"FAIL: aobench image check: no reference image hash at {ref} (capture it from the reference build first)"); return 1
    want = open(ref).read().split()[0].strip()
    if got != want:
        print(f"   ERROR: sha256 differs from the reference {want}")
        print("FAIL: aobench image check (byte-identical to the reference image)"); return 1
    print(f"PASS: aobench image check ({w}x{h} P6, byte-identical to the reference image {os.path.basename(ref)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
