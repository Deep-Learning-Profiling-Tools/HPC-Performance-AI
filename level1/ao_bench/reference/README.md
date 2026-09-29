# Reference image of the registered aobench input

`iter100.sha256` (not yet captured) holds the sha256 of `ao.ppm` rendered by the unoptimized
reference build of this repository for the registered input `iter100` (`./main 100`, 256x256),
one line: `<sha256>  ao.ppm`. `iter100.sha256.txt` is its provenance: build directory, binary
sha256, git commit, host/GPU, date, and the run's stdout log.

`check_ppm.py` compares a run's `ao.ppm` byte for byte with this reference (the render is
deterministic: per-pixel RNG seeds, `verify.py`). Capturing the reference is a step of the
correctness evidence plan (`correctness_runs2.sh`, results directory); the hash is copied here
afterwards and committed with its provenance. Any tolerance for a candidate that reorders
floating-point work is a separate decision (`correctness_decisions_needed.md`).
