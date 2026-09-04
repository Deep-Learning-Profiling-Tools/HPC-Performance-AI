# Overview — how HACCabana works

HACCabana (`pmkokkos`) is a particle-mesh (PM) cosmology code in the HACC
software lineage, rebuilt on performance-portable libraries: Cabana for
particle and particle-grid data structures, Kokkos for execution backends,
heFFTe for distributed FFTs, parallel HDF5 for snapshots, and MPI.

It does two things:

1. **Generate initial conditions** (`pm_ic`) — HACC-compatible Zel'dovich
   ICs from a CAMB transfer function.
2. **Evolve them** (`pm_run`) — drift–kick–drift PM steps from `z_in` to
   `z_final`.

Everything below maps onto the module tree in `src/`.

---

## Module tree

```
src/            Types, CodeUnits, Cosmology, Grid, Particles      core
src/ic/         TransferFunction, PowerSpectrum, WhiteNoise,
                ICGreensFunction, InitialConditions, CoordConvert  IC pipeline
src/pm/         PackUnpack, GreensFunction, MassAssign,
                ForceInterp, PMStep                                PM gravity
src/time/       Drift, Integrator, PMDriver                        DKD integrator
src/comm/       Migrator                                           migration
src/io/         IndatParser, SnapshotSchema,
                SnapshotWriter, SnapshotReader                     I/O
apps/           pm_ic, pm_run (+pm_run_lib), snapshot_inspect,
                diag/pm_single_kick                                drivers
```

Three files in the tree are **not part of the built code**; see
[Unwired modules](#unwired-modules) at the end.

---

## Core data structures

### Particles — `src/Particles.hpp`, `src/Types.hpp`

Particles live in a `Cabana::AoSoA` (array-of-structs-of-arrays) with six
members, mirroring HACC's particle layout:

| Member | Type | Meaning |
|---|---|---|
| `FIELD_POS` | `float[3]` | position, grid units |
| `FIELD_VEL` | `float[3]` | velocity — the canonical momentum `P̃`, not `dx/dt` |
| `FIELD_MASS` | `float` | per-particle mass (always 1 in practice) |
| `FIELD_ID` | `int64_t` | unique particle ID |
| `FIELD_MASK` | `uint16_t` | status/overload bitmask |
| `FIELD_PHI` | `float` | gravitational potential |

The AoSoA's inner SIMD width is `PMKOKKOS_VECTOR_LENGTH` (default 16,
matching the SYCL/PVC backend default). Position and velocity are **FP32**,
matching HACC's `POSVEL_T`; IDs are `int64_t`, matching HACC's `ID_64`
build.

Two conventions matter and are easy to get wrong:

- **Velocity stores `P̃ = a²·dx/dt`**, not a physical velocity. The
  integrator's per-step coefficients absorb every coordinate transform, so
  there is no "convert coordinates" state machine as in HACC. Conversion
  to and from physical units happens only at the I/O boundary.
- **Positions are stored GLOBAL, in `[0, ng)` per dimension** — not
  rank-local. This is deliberate and documented in
  `src/ic/CoordConvert.hpp`: `Migrator` finds a particle's destination rank
  by integer-floor of the *global* cell index against per-rank boxes, and
  `Cabana::Grid::p2g`/`g2p` subtract the rank's own mesh offset internally.
  HACC by contrast stores LOCAL rank-relative coordinates. The two
  representations differ in FP32 rounding behavior across subdomains; a
  source audit measured the effect and found it inconsequential relative to
  the observed cross-code residual.

`FIELD_MASS` exists for HACC layout compatibility and I/O round-trip, but
the PM deposit never reads it — mass enters as a uniform runtime constant
(see `UniformMassP2G` in `src/pm/MassAssign.hpp`), so the scatter doesn't
spend bandwidth on a value that is constant.

### Grid — `src/Grid.hpp`

A wrapper over `Cabana::Grid` holding a periodic uniform `ng³` cell-centered
mesh plus the FFT plan:

- `rho` — FP32, 1 degree of freedom per cell. The mass mesh.
- `phi` — FP64, 2 dofs per cell. The **complex** working buffer for FFTs.
- `phi_real` — FP32, 1 dof. Real component extracted after an inverse FFT;
  this is the field the force gather interpolates.
- Halos for each, and a heFFTe plan built over the `phi` layout.

The decomposition is a 3D Cartesian block partition. Passing explicit
`(Px, Py, Pz)` selects `ManualBlockPartitioner`, which is what guarantees
the topology matches HACC's; the default path uses `MPI_Dims_create`
followed by a descending sort.

### Cosmology and units — `src/Cosmology.hpp`, `src/CodeUnits.hpp`

`Cosmology` is pure host-side FP64: `E(a) = H(a)/H0`, `adot(a) = a·E(a)`,
and growth factors from an adaptive Cash-Karp ODE solve ported from HACC.
`CodeUnits` holds the physical↔grid conversion factors and the Poisson
normalization `phiscal = 1.5·Ω_cb`.

One parameter deserves attention: **`omega_radiation` must be nonzero.**
It is computed from `T_CMB` and `h` as `2.471e-5·(T_CMB/2.725)⁴/h²`.
Leaving it at zero underestimates expansion at high redshift and produced a
~10% velocity error at z=200 against HACC — a real bug that was found and
fixed during development, now guarded by a regression test.

---

## Stage 1 — initial conditions (`pm_ic`)

Driver: `apps/pm_ic.cpp`. Orchestration: `src/ic/InitialConditions.cpp`.

```
indat.params
    │
    ├─► Cosmology            (src/Cosmology.cpp)
    ├─► TransferFunction     (src/ic/TransferFunction.cpp)   CAMB 7-column file
    └─► PowerSpectrum        (src/ic/PowerSpectrum.cpp)      P(k)=A·k^ns·T²(k), σ₈-normalized
                                     │
    per axis (x, y, z):              ▼
      WhiteNoise ──► forward FFT ──► ICGreensFunction ──► reverse FFT ──► Zel'dovich move
    (src/ic/WhiteNoise.cpp)          (src/ic/ICGreensFunction.cpp)
                                     │
                                     ▼
                    INIT→PHYSICAL unit conversion, migrate, periodic wrap
                                     │
                                     ▼
                            schema-v2 HDF5 snapshot
```

**White noise** (`src/ic/WhiteNoise.cpp`) is the reproducibility-critical
piece. It uses HACC's counter-based RNG (CBRNG/Random123, vendored verbatim
under `external/`) keyed on the **global** cell index and a per-x-slab key
derived from `(seed, global x-index)`. The consequence: the same seed and
the same `ng` give a **bit-identical** real-space field regardless of rank
count, topology, or backend. Draws are computed on the host and copied to
device, which is what makes the result backend-independent. HACC's legacy
MT19937 path is deliberately not ported — its seeding depends on rank
count, which would break exactly this property.

**Power spectrum** normalization integrates σ₈ by composite Simpson's rule
in `log k`, with a doubling-trick convergence check.

**The IC Green's function** (`src/ic/ICGreensFunction.cpp`) is a *continuum*
spectral operator `(i·k_axis)·(−1/k²)·√P(k)·ng^(−3/2)`, deliberately
**distinct** from the discrete finite-difference Green's function used in
the PM step. The IC's goal is to reproduce the input P(k); the PM step's
goal is to emulate a finite-difference Laplacian. Using the wrong one in
either place is a correctness bug, not a tuning choice.

**Output state** is `(INIT, GLOBAL, periodic, non-symplectic)`. `pm_ic` then
converts INIT→PHYSICAL (`pos *= rL/np`, `vel *= 100·rL/np`), migrates each
particle to its owning rank, wraps into `[0, rL)`, and writes the snapshot.

---

## Stage 2 — the PM gravity step

Entry point: `pm_kick()` in `src/pm/PMStep.cpp`.

```
particles ──► deposit ──► pack ──► forward FFT ──► save rho_k
              (CIC)      FP32→FP64                     │
                          complex                      │
                                    ┌──────────────────┴──────────────────┐
                                    │  for each axis 0,1,2:               │
                                    │    restore rho_k                    │
                                    │    G(k)·(−i·sin(2πk_axis/N))        │
                                    │    reverse FFT                      │
                                    │    unpack real part                 │
                                    │    CIC value gather ──► vel[axis]   │
                                    └─────────────────────────────────────┘
```

**Deposit** (`src/pm/MassAssign.cpp`) is `Cabana::Grid::p2g` with
`Spline<1>()` — cloud-in-cell. A `Kokkos::ScatterView` accumulates, then
`halo.scatter()` reduces ghost-cell contributions back to owning ranks.

**Pack/unpack** (`src/pm/PackUnpack.cpp`) exists because Cabana::Grid's FFT
is **complex-to-complex only** — there is no real-to-complex path. So FP32
`rho` is promoted into the real component of the FP64 complex `phi` before
each forward transform, and the real part is extracted afterward. Note the
unpack is a lossy FP64→FP32 narrowing: the FFT itself round-trips to ~1e-10,
but anything downstream of `unpack_phi_real` is bounded by FP32 epsilon
(~1e-7).

**The Green's function** (`src/pm/GreensFunction.cpp`) is HACC's 2nd-order
discrete operator:

```
G(k) = coeff / (cos(2πk₀/N) + cos(2πk₁/N) + cos(2πk₂/N) − 3),   G(0,0,0) = 0
```

HACC uses `coeff = 0.5/N³`. **pmkokkos uses `coeff = 0.5`** and delivers the
missing `1/N³` through the FFT scaling convention instead —
`forward(FFTScaleNone)` then `reverse(FFTScaleFull)`. This is end-to-end
equivalent to HACC (HACC's FFTW is unnormalized on both directions) while
keeping the kernel pure physics. If you compare this kernel against HACC's
line by line, that factor is the expected difference.

The production path fuses the Green's function with a **spectral gradient**
`−i·sin(2πk_axis/N)` in one kernel (`apply_greens_and_gradient`), so the
inverse FFT yields the real-space force component directly. Three FFTs per
kick (one forward, three reverse) — matching HACC's structure. The saved
`rho_k` shadow buffer costs one extra `ng³·2·FP64` array per rank.

A **legacy diagnostic path** also exists (`apply_greens_function` +
`compute_forces`, taking an analytic gradient of the CIC basis). It is
retained because the original acceptance test was written against it, and
it is documented as lower-fidelity at intermediate `k`. Production gravity
does not use it.

---

## Stage 3 — the DKD integrator

`src/time/Integrator.cpp`. The stepping variable is `p = a^α`; all
production runs use `α = 1`.

One `full_step()`:

```
half-drift(τ/2)              positions advance at start-of-step scalars
migrate                      ← pre-kick migration (see below)
advance_half_step            scalars → midpoint
kick(τ · fscal)              the full PM gravity step
advance_half_step            scalars → end-of-step
half-drift(τ/2)
migrate
```

**The pre-kick migrate is load-bearing.** It was missing originally, and
its absence was a real defect. Under this code's strict-ownership scheme
(no overload/ghost particle buffers, unlike HACC), a particle that crosses
a rank boundary during the first half-drift would have its CIC stencil land
on the *wrong rank's* local mesh — giving it a wrong kick force. Adding the
pre-kick migrate fixed it; the regression guard is
`tests/test_migration_protects_scatter.cpp`, whose 2-rank run is the
decisive case. The cost is one extra `Cabana::Distributor` exchange per
step, which measured within launcher noise on the reference run.

The per-step cosmological coefficients follow HACC's `TimeStepper`:

- `phiscal = 1.5 · Ω_cb` — the Poisson source coefficient.
- `fscal = phiscal · dtdy / a`, with `dtdy = a/(α·ȧ·p)`, recomputed each
  half-step. The kick passes `dt = τ · fscal` into `pm_kick`.

`src/time/Drift.cpp` is a single `parallel_for`; the divisions are done
once on the host and folded into a scalar coefficient so the device kernel
does none.

### Step count is a correctness constraint, not a performance knob

Evolving `z=200 → z=0` in a handful of steps does not merely lose accuracy
— it **crashes**. One full DKD step with a large `τ` produces a kick
velocity comparable to the box size, and the next deposit indexes unmapped
memory. Production cosmological evolution needs hundreds of steps; the
reference configuration uses 500. There is no PM sub-cycling in this code.

---

## Stage 4 — migration

`src/comm/Migrator.cpp`. Wraps `Cabana::Distributor`:

1. One `MPI_Allgather` collects every rank's `(globalOffset, ownedNumCell)`
   box into a table.
2. A per-particle kernel wraps positions into `[0, ng)³` and scans the box
   table for the destination rank.
3. `Cabana::Distributor` + `Cabana::migrate` redistribute the AoSoA in
   place; `Particles::set_num_local` picks up the new owned count.

Because the distributor posts `MPI_Isend`/`MPI_Irecv` with **device**
pointers, multi-rank runs require GPU-aware MPI. On Aurora that means
`MPIR_CVAR_ENABLE_GPU=1`. This is the single most common cause of
multi-rank failure — and note HACC's own jobscripts set it to `0`, because
HACC stages MPI buffers through host memory. pmkokkos cannot. Small test
cases can pass without it, so the failure often first appears at scale.

---

## Stage 5 — snapshot I/O

`src/io/`. Parallel HDF5, schema version 2:

```
/particles/position   float32 (N,3)      /particles/mass    float32 (N,)
/particles/velocity   float32 (N,3)      /particles/phi     float32 (N,)   [v2]
/particles/id         int64   (N,)       /particles/status  int32   (N,)   [v2]
/particles/mask       uint16  (N,)
```

Metadata (`a`, `step`, `ng`, `np`, `rL`, `z_in`, `omega_m`, `omega_cb`,
`omega_b`, `h`, `n_s`, `sigma_8`, `topology_dims`, `schema_version`) lives
as scalar attributes on the root group.

Each rank writes its owned particles as a hyperslab into the global
dataset, in rank order, with collective MPI-IO. Ranks holding zero
particles still participate in every collective call — mandatory for the
collective contract, and a common source of hangs if skipped.

The reader **range-partitions** the global list (rank `r` gets its share by
index) rather than by position — so `read_snapshot` must be followed by
`migrate()` if you need particles on their spatially-correct ranks. Both
drivers do this.

Schema v1 files are read transparently: `phi` is backfilled to 0, `status`
to the local rank, and a warning is logged.

> **Layout pitfall.** The SYCL backend's default Kokkos View layout is
> `LayoutLeft`, but HDF5 hyperslabs for `(N,3)` datasets are row-major. The
> staging views are therefore declared explicit `LayoutRight`. A single-rank
> round-trip masks a mistake here because the same scrambling happens on
> read; the multi-rank round-trip is the canary.

---

## What this code does not do

- **No short-range forces.** PM only — long-range gravity on the mesh.
  Sub-grid-scale clustering is not resolved.
- **No PM sub-cycling.**
- **No hydrodynamics, no halo finding.**
- **No GenericIO.** HDF5 is the only on-disk format; GIO compatibility was
  explicitly out of scope. Conversion scripts live in the analysis tree.
- **`W_DE` / `WA_DE` / `OMEGA_NU` are parsed but not validated.** `pm_ic`
  honors them; `pm_run` **ignores them and hardcodes `w=-1, wa=0`** (see
  [RUNNING.md](RUNNING.md#cosmology-precedence-in-pm_run)). Only the ΛCDM
  region has been exercised.

---

## Unwired modules

Three files sit in the tree but are **not compiled into anything**. They are
not part of the working code and should not be assumed functional:

| File | Status |
|---|---|
| `src/pm/PoissonSolve.hpp` | Header-only scaffolding whose method bodies are `TODO` comments. No `.cpp` exists, nothing includes it. The real k-space solve lives in `src/pm/GreensFunction.cpp`. |
| `src/pm/SwfftBackend.{hpp,cpp}` | An FFT backend wrapping HACC's SWFFT, single-rank only. Its header comment claims a `PMKOKKOS_FFT_BACKEND_SWFFT` CMake option enables it — **that option does not exist**, the source is not in `CMakeLists.txt`, and `PMStep.cpp` does not reference it. Untracked working copy. |
| `src/comm/OverloadReplica.{hpp,cpp}` | HACC-style overload-buffer particle replication for an experiment branch. Not in `CMakeLists.txt`, not included anywhere. Untracked working copy. |

`src/time/PMDriver.cpp` *is* compiled, but its `run()` has no callers —
`pm_run` runs its own step loop so it can interleave timing and dump logic.
