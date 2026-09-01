# Contributing to HPC-Performance-AI

This document describes the basic workflow for contributing benchmarks,
applications, environment updates, and documentation to HPC-Performance-AI.

The goal is to keep branches and pull requests easy to identify without
introducing unnecessary process overhead.

## Branch Naming

Create all development branches from the latest `main`.

Use the following format:

```text
<scope>/<short-description>
```

Use lowercase letters and hyphens in the description.

### Scopes

| Scope | Use |
|---|---|
| `level1` | Level 1 benchmark additions or benchmark-specific changes |
| `level2` | Level 2 mini-app additions or changes |
| `level3` | Level 3 application additions or changes |
| `env` | Toolchain, dependency, or environment changes |
| `build` | CMake or build-system changes |
| `fix` | Bug fixes that are not specific to one level |
| `docs` | Documentation-only changes |
| `misc` | Small changes that do not fit the categories above |

### Examples

Adding or updating a Level 1 benchmark:

```text
level1/bfs
level1/spmv
level1/murmurhash3
level1/hotspot-hip
```

Adding a Level 2 mini-app:

```text
level2/minibude
level2/miniweather
```

Adding a Level 3 application:

```text
level3/<application-name>
```

Environment or toolchain changes:

```text
env/rocm-support
env/cuda-update
env/amd-environment
```

Build-system changes:

```text
build/level1-cmake
build/hip-detection
```

Bug fixes:

```text
fix/bfs-validation
fix/spgemm-input
```

Documentation:

```text
docs/level1-catalog
docs/environment-guide
```

## Naming Rules

Keep branch names short and descriptive.

Prefer:

```text
level1/bfs-hip
fix/cg-validation
env/rocm-support
```

Avoid:

```text
my-branch
test
new-code
update
final
final-v2
john-work
```

Do not include spaces or underscores in branch names.

Use:

```text
hotspot-3d
```

instead of:

```text
hotspot_3d
```

The repository directory may still be named `hotspot_3d`; this rule applies
only to Git branch names.

## Typical Workflow

Start from the latest `main`:

```bash
git checkout main
git pull origin main
```

Create a branch:

```bash
git checkout -b level1/bfs-hip
```

Make and test the changes, then push the branch:

```bash
git push -u origin level1/bfs-hip
```

Open a pull request from the new branch into:

```text
main
```

Do not push development work directly to `main`.

## Pull Requests

A pull request should represent one logical change.

For example, prefer:

```text
level1/bfs-hip
```

for adding HIP support to BFS rather than combining unrelated changes to BFS,
SpMV, the environment setup, and documentation in the same pull request.

A pull request should briefly describe:

- what was added or changed;
- which benchmark, mini-app, or application is affected;
- how the change was validated;
- any known limitations or unverified backends.

For benchmark changes, include the build and validation result when applicable.

Example:

```text
Title:
[Level 1] Add HIP support for BFS

Summary:
- Added the HIP implementation for BFS.
- Preserved the existing benchmark input and correctness semantics.
- CUDA remains unchanged.
- HIP build/run was validated on <GPU/system>.
```

## Before Opening a Pull Request

Make sure your branch is based on a reasonably recent `main` and that the
affected code builds successfully.

For Level 1 CUDA benchmarks, the usual validation is:

```bash
source hpcperf_env.sh

cmake -S level1/<benchmark> \
      -B build/<benchmark>/cuda \
      -DBACKEND=CUDA \
      -DCMAKE_BUILD_TYPE=Release

cmake --build build/<benchmark>/cuda

ctest --test-dir build/<benchmark>/cuda --output-on-failure
```

If a backend cannot be tested because the required hardware is unavailable,
state that clearly in the pull request.

## After Merge

Once a pull request is merged, the development branch can be deleted.

Keep `main` as the shared stable branch.
