#!/usr/bin/env bash
# Fetch Nyx (AMReX-Astro) at the pinned release plus the AMReX release it is
# built against here. Read-only checkouts under _upstream/level3/; nothing is
# committed to this repository.
#
#   Nyx 26.09            e06eabc1b9dbcad5612db9529aced682402daede  (tag 26.09, 2026-08-26)
#     subprojects/amrex    6e875b7cc1a4eec78e22ae4cdaa79f88acf5169e  (development 2026-08-12; the pin)
#     subprojects/sundials 5c53be85c88f63c5201c130b8cb2c686615cfb03  (v7.2.1; used by the heatcool profile)
#   AMReX 26.09          a52ca73324ac2c7b65ec04f131e6df99eec9c576  (tag 26.09, 2026-09-01; 21 commits
#                        ahead of / 0 behind the Nyx pin -- a strict descendant)
#
# Why AMReX 26.09 and not the submodule pin: the pinned AMReX resolves CUDA
# architectures through CMake's legacy cuda_select_nvcc_arch_flags and its
# convert_cuda_archs() drops every SM >= 10.0 ("CMake 3.30 does not support SM
# 10.0+"), then autodetects 8.6+PTX on this B200 node -> an sm_86 binary (first
# attempt, removed). AMReX 26.09 resolves the architecture through nvcc
# --list-gpu-arch into CMAKE_CUDA_ARCHITECTURES (the path WarpX 26.09 already
# verified here with sm_100). Nyx's own minimum is AMREX_MINIMUM_VERSION 20.11
# and it consumes an external AMReX through find_package(AMReX CONFIG). This is a
# private build for Nyx: WarpX's AMReX *binary* is not reused (WarpX's superbuild
# builds AMReX with a different component set: EB, no linear solvers, FFT ...).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
UP="$R/_upstream/level3"
NYX_TAG=26.09;   NYX_SHA=e06eabc1b9dbcad5612db9529aced682402daede
AMREX_PIN_SHA=6e875b7cc1a4eec78e22ae4cdaa79f88acf5169e
SUNDIALS_SHA=5c53be85c88f63c5201c130b8cb2c686615cfb03
AMREX_TAG=26.09; AMREX_SHA=a52ca73324ac2c7b65ec04f131e6df99eec9c576

mkdir -p "$UP"
if [ ! -d "$UP/Nyx/.git" ]; then git clone -q --branch "$NYX_TAG" --depth 1 https://github.com/AMReX-Astro/Nyx.git "$UP/Nyx"; fi
got="$(git -C "$UP/Nyx" rev-parse HEAD)"
[ "$got" = "$NYX_SHA" ] || { echo "fetch.sh: $UP/Nyx is at $got, expected $NYX_SHA (tag $NYX_TAG)" >&2; exit 1; }
git -C "$UP/Nyx" submodule update --init --depth 1 subprojects/amrex subprojects/sundials
a="$(git -C "$UP/Nyx/subprojects/amrex" rev-parse HEAD)"; s="$(git -C "$UP/Nyx/subprojects/sundials" rev-parse HEAD)"
[ "$a" = "$AMREX_PIN_SHA" ] || { echo "fetch.sh: Nyx amrex submodule at $a, expected $AMREX_PIN_SHA" >&2; exit 1; }
[ "$s" = "$SUNDIALS_SHA" ] || { echo "fetch.sh: Nyx sundials submodule at $s, expected $SUNDIALS_SHA" >&2; exit 1; }

if [ ! -d "$UP/amrex/.git" ]; then git clone -q --branch "$AMREX_TAG" --depth 1 https://github.com/AMReX-Codes/amrex.git "$UP/amrex"; fi
x="$(git -C "$UP/amrex" rev-parse HEAD)"
[ "$x" = "$AMREX_SHA" ] || { echo "fetch.sh: $UP/amrex is at $x, expected $AMREX_SHA (tag $AMREX_TAG)" >&2; exit 1; }
echo "# Nyx $NYX_TAG $got (submodules amrex $a, sundials $s); AMReX $AMREX_TAG $x -> $UP"
