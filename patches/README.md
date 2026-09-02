# Patches applied to Level 2 framework dependencies

`setup_level2_deps.sh` applies every `patches/<dep>-*.patch` to the pinned
checkout in `.deps/src/<dep>` right after cloning it (idempotently, with
`git apply`). Patches are kept to a minimum and touch build systems only;
none changes numerical results of the mini-apps.

| Patch | Dependency | Why |
|---|---|---|
| `mfem-v4.10-blackwell-cicc-O1.patch` | MFEM v4.10 | On Blackwell (sm_100) with CUDA 13.2, nvcc's device front end (`cicc`) at its default `-O3` takes hours on three translation units -- measured on the B200 machine at `-j4`: `fem/integ/bilininteg_convection_ea.cpp` > 11 h (killed unfinished; `cicc` alone ran 7 h and produced 140 MB of PTX), `fem/lor/lor_batched.cpp` 5.2 h, `fem/integ/bilininteg_diffusion_ea.cpp` 1.6 h. The same file compiles in ~10 s with `-Xcicc -O1`. The patch lowers the *device* optimization level to `-O1` for exactly these three files, only when a CUDA architecture >= 100 is requested (`CMAKE_CUDA_ARCHITECTURES`). They contain element-assembly (`AssemblyLevel::ELEMENT`) and batched low-order-refined kernels that neither Laghos nor Remhos execute (both use partial/legacy full assembly), so the mini-app performance is unaffected. The same pathology is reported upstream for sm_120 / CUDA 12.8 in [mfem/mfem#5363](https://github.com/mfem/mfem/issues/5363) (closed "not planned"). `-Xcicc` is an undocumented but long-standing nvcc pass-through option. |
