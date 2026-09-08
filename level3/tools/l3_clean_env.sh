#!/bin/bash
# l3_clean_env.sh -- run a command under an allow-listed environment.
#
#   l3_clean_env.sh [--list] [--show] [--] <cmd> [args...]
#
# Why: upstream tools record the process environment (CP2K's install_cp2k_toolchain.sh
# writes `declare -x` of the whole environment into toolchain.env; nsys/ncu store it in
# their reports; some build systems log it). A login shell carries credentials and
# agent/session variables that must never end up in such files. This wrapper re-executes
# the command with `env -i` and only the variables an HPC build/run needs.
#
# Allow-list (exact names or prefixes; nothing else survives):
#   session/locale : PATH HOME USER LOGNAME SHELL TERM TMPDIR TZ LANG LANGUAGE LC_* XDG_RUNTIME_DIR
#   scheduler      : SLURM_* SLURMD_NODENAME SLURM_TOPOLOGY_ADDR* (Slurm), PBS_* LSB_* (other sites)
#   MPI/PMIx       : OMPI_* PRTE_* PMIX_* OPAL_* MPI_* HWLOC_* UCX_* NCCL_* FI_*
#   GPU            : CUDA_HOME CUDA_PATH CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER CUDA_CACHE_* NVCC_* CUDACXX CUDAHOSTCXX CUDAARCHS CUDA_LAUNCH_BLOCKING
#                    ROCM_PATH ROCM_HOME HIP_* HSA_* HCC_* ROCR_*
#   compilers/build: CC CXX FC F77 F90 F95 CPP LD AR RANLIB NM STRIP CFLAGS CXXFLAGS FFLAGS FCFLAGS LDFLAGS CPPFLAGS
#                    CMAKE_* NINJA_* MAKEFLAGS MAKELEVEL CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH
#                    LD_RUN_PATH PKG_CONFIG_PATH MANPATH INFOPATH
#   conda/python   : CONDA_* _CONDA_* PYTHONPATH PYTHONNOUSERSITE PYTHONHOME VIRTUAL_ENV
#   runtime knobs  : OMP_* KMP_* GOMP_* MKL_* OPENBLAS_* BLIS_* KOKKOS_* RAJA_* AMREX_* CP2K_DATA_DIR
#   project        : HPCPERF_* HPC_PERFORMANCE_AI_ROOT _HPCPERF_* L3_*
# Everything else (API keys, tokens, CLAUDE_*, ANTHROPIC_*, OPENAI_*, GITHUB_TOKEN, HUGGING_FACE*,
# SSH_*, GPG_*, DBUS_*, ...) is dropped. A deny-list is applied AFTER the allow-list, so a
# credential-looking name never survives through an allow-listed prefix (e.g. HPCPERF_*_TOKEN).
# No value is ever printed by this script.
set -euo pipefail
DENY_REGEX='(^|_)(API_KEY|KEY|TOKEN|SECRET|SECRETS|PASSWORD|PASSWD|CREDENTIAL|CREDENTIALS)(_|$)|^(CLAUDE|ANTHROPIC|OPENAI|DEEPSEEK|HUGGING_FACE|HF|GITHUB|GH|AWS|AZURE|GOOGLE|SSH|GPG)(_|$)'
ALLOW_EXACT="PATH HOME USER LOGNAME SHELL TERM TMPDIR TZ LANG LANGUAGE XDG_RUNTIME_DIR SLURMD_NODENAME \
CUDA_HOME CUDA_PATH CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER CUDACXX CUDAHOSTCXX CUDAARCHS CUDA_LAUNCH_BLOCKING ROCM_PATH ROCM_HOME \
CC CXX FC F77 F90 F95 CPP LD AR RANLIB NM STRIP CFLAGS CXXFLAGS FFLAGS FCFLAGS LDFLAGS CPPFLAGS MAKEFLAGS MAKELEVEL \
CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH LD_RUN_PATH PKG_CONFIG_PATH MANPATH INFOPATH \
PYTHONPATH PYTHONNOUSERSITE PYTHONHOME VIRTUAL_ENV CP2K_DATA_DIR HPC_PERFORMANCE_AI_ROOT"
ALLOW_PREFIX="LC_ SLURM_ PBS_ LSB_ OMPI_ PRTE_ PMIX_ OPAL_ MPI_ HWLOC_ UCX_ NCCL_ FI_ CUDA_CACHE_ NVCC_ HIP_ HSA_ HCC_ ROCR_ \
CMAKE_ NINJA_ CONDA_ _CONDA_ OMP_ KMP_ GOMP_ MKL_ OPENBLAS_ BLIS_ KOKKOS_ RAJA_ AMREX_ HPCPERF_ _HPCPERF_ L3_"
list=0; show=0
while [ $# -gt 0 ]; do case "$1" in --list) list=1; shift;; --show) show=1; shift;; --) shift; break;; -*) echo "l3_clean_env.sh: unknown option $1" >&2; exit 2;; *) break;; esac; done
if [ "$list" -eq 1 ]; then echo "exact: $ALLOW_EXACT"; echo "prefixes: $ALLOW_PREFIX"; exit 0; fi
[ $# -gt 0 ] || { echo "usage: l3_clean_env.sh [--list] [--show] [--] <cmd> [args...]" >&2; exit 2; }
keep=(); denied=()
while IFS= read -r name; do
    ok=0
    for e in $ALLOW_EXACT; do [ "$name" = "$e" ] && { ok=1; break; }; done
    if [ "$ok" -eq 0 ]; then for p in $ALLOW_PREFIX; do case "$name" in "$p"*) ok=1; break;; esac; done; fi
    if [ "$ok" -eq 1 ] && [[ "$name" =~ $DENY_REGEX ]]; then ok=0; denied+=("$name"); fi
    [ "$ok" -eq 1 ] && keep+=("$name")
done < <(env | sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' | sort -u)
args=()
for name in "${keep[@]}"; do args+=("$name=${!name}"); done
if [ "$show" -eq 1 ]; then
    printf 'l3_clean_env: %d variables kept (names only): %s\n' "${#keep[@]}" "${keep[*]}" >&2
    printf 'l3_clean_env: %d variables dropped' "$(( $(env | /usr/bin/grep -c '^[A-Za-z_][A-Za-z0-9_]*=') - ${#keep[@]} ))" >&2
    [ "${#denied[@]}" -eq 0 ] && printf '\n' >&2 || printf ' (%d allow-listed names denied by the credential rule: %s)\n' "${#denied[@]}" "${denied[*]}" >&2
fi
exec env -i "${args[@]}" "$@"
