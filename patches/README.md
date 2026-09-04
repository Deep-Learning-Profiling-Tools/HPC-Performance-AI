# Patches applied to Level 2 framework dependencies

`setup_level2_deps.sh` applies every `patches/<dep>-*.patch` to the pinned
checkout in `.deps/src/<dep>` right after cloning it (idempotently, with
`git apply`). These are **dependency build workarounds**, not benchmark
optimizations: they are kept to a minimum, touch build systems only, and none
changes numerical results of the mini-apps.

| Patch | Dependency | Why |
|---|---|---|
| `mfem-v4.10-blackwell-cicc-O1.patch` | MFEM v4.10 | On Blackwell (sm_100) with CUDA 13.2, nvcc's device front end (`cicc`) at its default `-O3` takes hours on three translation units -- measured on the B200 machine at `-j4`: `fem/integ/bilininteg_convection_ea.cpp` > 11 h (killed unfinished; `cicc` alone ran 7 h and produced 140 MB of PTX), `fem/lor/lor_batched.cpp` 5.2 h, `fem/integ/bilininteg_diffusion_ea.cpp` 1.6 h. The same file compiles in ~10 s with `-Xcicc -O1`. The patch lowers the *device* optimization level to `-O1` for exactly these three files, only when a Blackwell 10x CUDA architecture is requested (`CMAKE_CUDA_ARCHITECTURES` matches `10x` -- deliberately NOT every future arch >= 100, since the pathology is only verified for MFEM v4.10 + CUDA 13.2 + sm_100). They contain element-assembly (`AssemblyLevel::ELEMENT`) and batched low-order-refined kernels that neither Laghos nor Remhos execute (both use partial/legacy full assembly), so the mini-app performance is unaffected. The same pathology is reported upstream for sm_120 / CUDA 12.8 in [mfem/mfem#5363](https://github.com/mfem/mfem/issues/5363) (closed "not planned"). `-Xcicc` is an undocumented but long-standing nvcc pass-through option. **After any CUDA toolkit upgrade, first retry the vanilla (unpatched) MFEM build** -- if the three files compile in reasonable time, drop this patch. Two further files remain slow but finish at `-O3` and are left untouched (`fem/qinterp/eval_hdiv.cpp` ~40 min, `fem/integ/bilininteg_vectorfemass_pa.cpp` ~26 min); they dominate a fresh MFEM build (~40 min at high `-j`). |
