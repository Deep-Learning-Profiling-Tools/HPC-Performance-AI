# Vendored NVTX v3 (header-only)

Copied verbatim from the CUDA Toolkit 13.2.78 install on dgx003
(`/usr/local/cuda/include/nvtx3/nvToolsExt.h` and `nvtxDetail/`); `SHA256SUMS`
pins every file against that source. Licence: Apache-2.0 WITH LLVM-exception
(SPDX header in every file; `LICENSE.txt` is the Apache-2.0 text).

Why vendored: CUDA 13 no longer ships `libnvToolsExt`, so NVTX is header-only,
and `tools/timing/roi/hpcperf_roi.h` must compile in every build of the suite --
including plain C/Fortran hosts and upstream build systems that never add
`/usr/local/cuda/include` -- with a single include path. NVTX v3 is designed to be
copied into projects: its symbols are versioned, so this copy coexists with an
application's own NVTX include of the same version.

Only the core API is copied (`nvToolsExt.h` + `nvtxDetail/`); nothing here is
modified. Update by re-copying from a newer toolkit and regenerating `SHA256SUMS`.
