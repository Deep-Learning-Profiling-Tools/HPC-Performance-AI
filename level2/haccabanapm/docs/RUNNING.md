# Running guide

Covers the driver applications, their command-line interfaces, the
`indat.params` reference, and dump schedules. For environment variables and
topology control see [RUNTIME_CONTROL.md](RUNTIME_CONTROL.md).

## The applications

All arguments are **positional**. There are no `--flags` on the production
binaries, and no argument is optional — a wrong argument count prints usage
and exits with status 2.

| Binary | Invocation | Ranks |
|---|---|---|
| `pm_ic` | `pm_ic <indat.params> <output.h5>` | any |
| `pm_run` | `pm_run <input.h5> <output.h5> <indat.params>` | any |
| `snapshot_inspect` | `snapshot_inspect <snapshot.h5>` | 1 only |
| `pm_single_kick` | diagnostic — see [below](#pm_single_kick-diagnostic) | any |

Note the argument order differs between `pm_ic` and `pm_run`: `pm_ic` takes
the indat **first**, `pm_run` takes it **last**.

### A minimal end-to-end run

```sh
# 1. Generate initial conditions.
mpiexec -n 8 ./pm_ic indat.params ic.h5

# 2. Evolve.
mpiexec -n 8 ./pm_run ic.h5 evolved.h5 indat.params

# 3. Look at the result.
mpiexec -n 1 ./snapshot_inspect evolved.h5
```

The transfer-function file named by `INPUT_BASE_NAME` is resolved relative
to the working directory, so run from the directory holding it (or give a
path).

### `pm_ic`

Generates Zel'dovich initial conditions and writes a schema-v2 HDF5
snapshot. On rank 0 it echoes the resolved topology, box and grid
parameters, cosmology, and transfer-function path, then reports any indat
keys present in the file but never read — useful for catching typos.

Its cosmology comes **entirely from the indat**.

### `pm_run`

Reads a snapshot, evolves it through `N_STEPS` DKD steps, and writes the
final state.

#### Cosmology precedence in `pm_run`

This is the subtlest thing in the whole interface and worth stating
plainly:

> **`pm_run` takes cosmology from the *snapshot*, not from the indat.**

From the indat it reads only `Z_FIN`, `N_STEPS`, `T_CMB`,
`FULL_ALIVE_DUMP`, and `TOPOLOGY`. Everything else — `rL`, `ng`, `np`,
`omega_m`, `omega_cb`, `omega_b`, `h`, `n_s`, `sigma_8`, and the starting
scale factor `a_in` — comes from the input snapshot's metadata attributes.

Two consequences:

- `T_CMB` is read from the indat because the v2 schema does not carry it.
  It defaults to 2.725. The difference between 2.725 and 2.726 is ~0.15% on
  `omega_radiation`, well under the FP32 noise floor.
- **`W_DE` and `WA_DE` are ignored by `pm_run`.** It hardcodes `w = -1`,
  `wa = 0`. Setting them in the indat affects `pm_ic` only. Do not assume a
  non-ΛCDM dark-energy run works end to end.

Similarly `OMEGA_NU` is only read by `pm_ic`; `pm_run` reconstructs
`omega_nu` as `omega_m − omega_cb − omega_b` from snapshot attributes,
clamped at 0.

#### Validation

`pm_run` fails fast on:

- `N_STEPS <= 0`
- `a_in >= a_fin` — i.e. the snapshot's scale factor is not before
  `1/(1+Z_FIN)`. You cannot run backwards.

Errors are printed on rank 0 and the process returns 1.

### `snapshot_inspect`

Single-rank human-readable dump: schema version, all metadata attributes,
global particle count, and the first and last 5 particles' position,
velocity, and ID. It deliberately goes through the normal `SnapshotReader`
path, so it doubles as a smoke check that the read side works on any file
`SnapshotWriter` produced.

### `pm_single_kick` (diagnostic)

Built from `apps/diag/`. Not a production driver — it exists for kernel
microbenchmarking and cross-code comparison, and its interface is unlike
the others. Three modes:

```sh
pm_single_kick <ic.h5> <output_basename> [a_fin] [nsteps] [t_cmb]
pm_single_kick --full-step  <ic.h5> <output_basename> [a_fin] [nsteps] [t_cmb]
pm_single_kick --two-particle <output_basename>
```

The optional trailing arguments are **numbers, not an indat file**
(`a_fin` a double, `nsteps` an int, `t_cmb` a double defaulting to 2.725).
Default mode performs one PM kick — deposit, FFT, Green's function, three
inverse FFTs, three gathers — and exits, which isolates kernel cost without
DKD overhead. `--full-step` runs one complete `Integrator::full_step`.

---

## `indat.params` reference

HACC-style plain-text configuration: one `KEY VALUE` pair per line,
whitespace-separated, `#` starts a comment, blank lines ignored, **keys are
case-sensitive**. Parser: `src/io/IndatParser.cpp`.

The file format is shared with HACC deliberately, so the same indat can
drive both codes for comparison. pmkokkos reads only the keys it needs and
ignores the rest — both drivers print a list of unread keys on completion.

### Keys read by `pm_ic`

**Required** — a missing key throws and aborts the run:

| Key | Type | Meaning |
|---|---|---|
| `RL` | double | Box side length, Mpc/h |
| `NG` | int | PM mesh cells per dimension (`ng³` total) |
| `NP` | int | Particles per dimension (`np³` total) |
| `HUBBLE` | double | `h` |
| `OMEGA_CDM` | double | CDM density parameter |
| `DEUT` | double | Baryon density `ω_b = Ω_b·h²` — divided by `h²` internally |
| `SS8` | double | σ₈ normalization target |
| `NS` | double | Primordial spectral index `n_s` |
| `Z_IN` | double | Starting redshift |
| `I_SEED` | int | White-noise RNG seed |
| `INPUT_BASE_NAME` | string | Path to the CAMB transfer-function file |

**Optional** — used if present, otherwise defaulted:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `OMEGA_NU` | double | `0.0` | Massive-neutrino density |
| `W_DE` | double | `-1.0` | CPL dark energy `w₀` |
| `WA_DE` | double | `0.0` | CPL dark energy `wₐ` |
| `T_CMB` | double | `2.725` | CMB temperature, K — sets `omega_radiation` |
| `TOPOLOGY` | string | auto | `"Px Py Pz"` or `"Px,Py,Pz"` MPI decomposition override |

`Ω_m` is assembled as `OMEGA_CDM + DEUT/h² + OMEGA_NU`.

### Keys read by `pm_run`

**Required:**

| Key | Type | Meaning |
|---|---|---|
| `Z_FIN` | double | Final redshift; `a_fin = 1/(1+Z_FIN)` |
| `N_STEPS` | int | Number of DKD steps. Must be > 0 |

**Optional:**

| Key | Type | Default | Meaning |
|---|---|---|---|
| `T_CMB` | double | `2.725` | CMB temperature, K |
| `FULL_ALIVE_DUMP` | string | none | Space-separated step indices to dump — see [below](#dump-schedules) |
| `TOPOLOGY` | string | snapshot | `"Px Py Pz"` decomposition override |

Everything else `pm_run` needs comes from the snapshot.

### Keys read by neither

A HACC indat carries many keys pmkokkos does not implement — `OL`,
`N_SUB`, `RSM`, `PM_SUBCYCLE_REDSHIFT`, `REFRESH_ALWAYS`, `DISTRIBUTE_TYPE`,
`USE_MPI_IO`, `OUTPUT_BASE_NAME`, `GLASS_START_*`, `CM_SIZE`, `ALPHA`, and
others. They are silently ignored and reported as unused. Leaving them in
is harmless and is what makes a shared indat possible.

Two are worth flagging because their names suggest they do something here:

- **`OL`** (overload width) has no effect. pmkokkos uses strict ownership
  with no overload buffers; see
  [OVERVIEW.md § Stage 4](OVERVIEW.md#stage-4--migration).
- **`ALPHA`** does not set the integrator's `α`. `pm_run` hardcodes
  `alpha = 1.0`.

### Example

`apps/demo/indat.params` is a working example that mirrors a HACC
configuration. The pmkokkos-relevant subset of it:

```
# Cosmology
OMEGA_CDM        0.22
DEUT             0.02258
OMEGA_NU         0.0
HUBBLE           0.71
SS8              0.8
NS               0.963
W_DE             -1.0
WA_DE            0.0
T_CMB            2.726

# Box and resolution
RL               128.0
NG               128
NP               128

# Initial conditions
Z_IN             200.0
I_SEED           5126873
INPUT_BASE_NAME  cmbM000.tf

# Evolution
Z_FIN            50.0
N_STEPS          5
```

The validated reference configuration is `NG=NP=128`, `RL=128` Mpc/h,
z=200 → z=0 in **500** steps, on 1 Aurora node with 8 ranks in a 2×2×2
topology.

---

## Dump schedules

`FULL_ALIVE_DUMP` is a space-separated list of **0-indexed step numbers**
at which `pm_run` writes an intermediate snapshot, in addition to the final
output:

```
FULL_ALIVE_DUMP 0 100 250 499
```

Output naming inserts `.step.<i>` before the final extension:

| Output argument | Dump at step 100 |
|---|---|
| `evolved.h5` | `evolved.step.100.h5` |
| `out/run_a.h5` | `out/run_a.step.100.h5` |
| `evolved` (no extension) | `evolved.step.100` |

Behavior details:

- The dump happens **after** the step completes, so `FULL_ALIVE_DUMP 0`
  gives you the state after one full DKD step, not the initial conditions.
- Indices outside `[0, N_STEPS)` are **warned about on rank 0 and skipped**
  — they do not abort the run.
- The list is sorted and de-duplicated, so order and repeats in the file do
  not matter.
- Each intermediate write round-trips the particles through
  simulation→physical units, a periodic wrap, and a migrate, then converts
  back to continue evolving. The dump is therefore not free; dumping every
  step will dominate the wall time of a small run.

To capture the final state only, omit `FULL_ALIVE_DUMP` entirely — the
final snapshot is always written to the `<output.h5>` path.
