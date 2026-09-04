# HACCabana (pmkokkos) documentation

User guide for the HACCabana particle-mesh cosmology code: how it works,
how to build it, and how to run it in each supported configuration.

| Guide | Covers |
|---|---|
| [OVERVIEW.md](OVERVIEW.md) | What the code does and how the pieces fit: IC pipeline → PM gravity step → DKD integrator → migration → snapshot I/O, mapped onto the `src/` module tree. |
| [BUILD.md](BUILD.md) | Dependencies, the CMake option table, GPU vs CPU builds, and how the Kokkos / heFFTe backends are selected. |
| [RUNNING.md](RUNNING.md) | The driver applications, their command-line interfaces, the full `indat.params` key reference, and dump schedules. |
| [RUNTIME_CONTROL.md](RUNTIME_CONTROL.md) | Environment-variable reference, GPU vs CPU selection, MPI topology control, GPU-aware MPI, tile affinity, and per-step timing. |
| [SYSTEMS.md](SYSTEMS.md) | Per-machine recipes. Aurora is complete; Frontier, Perlmutter, and CPU+FFTW are stubs. |
| [PROFILING_GUIDE.md](PROFILING_GUIDE.md) | Profiling `pm_run` on Aurora PVC with unitrace / onetrace / vtune / advisor. |

## Where to start

- **"I want to run the reference simulation."** [BUILD.md](BUILD.md) →
  [SYSTEMS.md § Aurora](SYSTEMS.md#aurora-alcf) → [RUNNING.md](RUNNING.md).
- **"I want to understand the physics and the code structure."**
  [OVERVIEW.md](OVERVIEW.md).
- **"Something is misbehaving at multi-rank."**
  [RUNTIME_CONTROL.md](RUNTIME_CONTROL.md) — most multi-rank failures on
  Aurora trace back to `MPIR_CVAR_ENABLE_GPU` or to a topology mismatch.
- **"I want to reproduce the paper's results."** See
  [Reproducing the paper results](#reproducing-the-paper-results) below.

## Scope

pmkokkos is a **PM-only** code: long-range gravity on a uniform mesh. There
are no short-range/tree forces, no sub-cycling, and no hydrodynamics. That
bounds what a run means physically — see
[OVERVIEW.md § What this code does not do](OVERVIEW.md#what-this-code-does-not-do).

## Reproducing the paper results

The companion paper (*Human-in-the-Loop Agentic AI for Trustworthy
Scientific HPC Code Development: The HACCabana PM/IC Case Study*) has a
separate artifact repository holding the prompt corpus, the development
worklog, validation baselines, and claim-verification scripts. Its
`reproduce/` directory contains the entry points:

- `reproduce/reproduce.sh` — top-level driver.
- `reproduce/verify_claims.py` — re-checks the paper's numerical claims
  against the recorded baselines.
- `reproduce/make_*_figure.py` — regenerates the paper figures.

The git tag `agenticai4hpc-paper` marks the exact pmkokkos state whose
behavior the paper validates. Commits after that tag add performance
instrumentation and documentation for follow-up studies; **check out the
tag** if you need the validated-era behavior rather than current `master`.

## A note on accuracy

The guides in this directory were written against the source at the time of
writing and state which file each behavior comes from, so claims can be
re-checked. Where a documented feature exists only on an experiment branch
rather than on `master`, that is called out explicitly rather than
described as if it were available.
