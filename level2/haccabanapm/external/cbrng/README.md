# Vendored CBRNG / Random123

These files are mirrored **verbatim** from HACC for bit-identical
initial-condition reproducibility.  The white-noise field that seeds
the IC pipeline must produce the same Gaussian draws as a HACC run with
the same global seed and grid size, regardless of MPI rank count or
execution backend.  Any local modification of the source breaks that
guarantee.

## Sources

- HACC initializer:
  `~/ClaudeCode/HACC/nbody/initializer/`
  - `CBRNG_Random.h`     →  `external/cbrng/CBRNG_Random.h`
  - `CBRNG_Random.cxx`   →  `external/cbrng/CBRNG_Random.cxx`
  - `CBRNG_Uniform.h`    →  `external/cbrng/CBRNG_Uniform.h`
- Random123 library:
  `~/ClaudeCode/HACC/nbody/initializer/Random123/`
  →  `external/Random123/`  (entire directory, including
  `features/`, `conventional/`, and the `*.h`/`*.hpp` headers)

## Mirror policy

- **DO NOT** edit any file in `external/cbrng/` or `external/Random123/`.
- If pmkokkos needs different behaviour, write a wrapper in `src/ic/`
  (see `WhiteNoise.{hpp,cpp}`) — never patch the vendored sources.
- If HACC's CBRNG implementation changes upstream, re-mirror the entire
  directory.  Do not cherry-pick individual files.
- The Random123 sub-library is header-only; only an include path is
  required.  `CBRNG_Random.cxx` is compiled into `pmkokkos_core` so the
  `GetSlabKeys` / `GetRandomDoublesWhiteNoise` symbols link.

## How the wrapper validates the mirror

`tests/test_white_noise_reproducibility.cpp` is the primary check that
the vendored CBRNG behaves correctly.  It generates the same white-noise
field on 1, 2, 4, and 8 MPI ranks and asserts that the global sum and
sum-of-squares are bit-identical across rank counts.  The
rank-independence property is what the per-mode CBRNG seeding gives us;
if the mirror were ever silently corrupted, this test would fail.

`tests/test_white_noise_against_hacc.cpp` is the optional secondary
check: if a HACC reference run with `INIT_DEBUG_OUTPUTS=1` is available,
a bit-comparison against `cbrng/cbrng.gwn.r.<rank>.txt` validates that
pmkokkos's draws agree with HACC's draws for the same seed.  At present
this test is `GTEST_SKIP`'ed pending a HACC validation harness.
