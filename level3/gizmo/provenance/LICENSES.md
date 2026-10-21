# Licenses of the GIZMO (gpu_kokkos_unified branch) source bundle

Declared components (path in the bundle, project, license, upstream, commit/version):

| path | project | license | upstream | commit / version |
|---|---|---|---|---|
| `src` | GIZMO | GPL-2.0-or-later (README.md section 2 "free software, distributed under the GNU General Public License"; the repository ships no separate license file) | https://github.com/pfhopkins/gizmo | e9d2d0bea23b08872f747b780cdcedb5847a2bff |
| `deps/ics` | GIZMO test initial conditions (P. F. Hopkins, tapir.caltech.edu/~phopkins/sims) | distributed with the GIZMO test suite (the files its harness downloads); redistributed unchanged |  |  |
| `deps/exact` | GIZMO test reference solution (sedov_exact.txt, same site) | distributed with the GIZMO test suite (the file its harness downloads); redistributed unchanged |  |  |

Redistribution notes:

- GIZMO is GPL (README.md); the patched source is redistributed as a source bundle with the two local patches stated in patch_series.txt, as the license requires modifications to be indicated.
- The initial-condition files are the ones upstream's own tests download; they are bundled unchanged so that the benchmark's workloads are reproducible without network access.

License/notice files present in the bundle (0):

