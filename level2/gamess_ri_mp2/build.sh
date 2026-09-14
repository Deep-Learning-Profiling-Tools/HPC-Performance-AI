#!/usr/bin/env bash
# Build the upstream cuBLAS or hipBLAS/hipfort implementation.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null
set -euo pipefail

BACKEND="${1:-CUDA}"; BACKEND="${BACKEND^^}"
MPIFC="${MPIFC:-$(command -v mpifort || true)}"

case "$BACKEND" in
  CUDA)
    [ -n "$MPIFC" ] || { echo "build.sh: MPI Fortran compiler wrapper not found (mpifort)" >&2; exit 1; }
    NVFORTRAN="${NVFORTRAN:-$(command -v nvfortran || true)}"
    [ -n "$NVFORTRAN" ] || { echo "build.sh: nvfortran not found; the native CUDA path requires NVIDIA HPC SDK Fortran OpenMP offload" >&2; exit 1; }
    [ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ] || { echo "build.sh: nvcc not found under CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}" >&2; exit 1; }
    [ -f "${CUDA_HOME:-/usr/local/cuda}/include/cublas_v2.h" ] || { echo "build.sh: cuBLAS headers unavailable under CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}" >&2; exit 1; }
    ARCH="${HPCPERF_CUDA_ARCH:-${CUDA_ARCH:-}}"
    [ -n "$ARCH" ] || ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
    ARCH="${ARCH#sm_}"; ARCH="${ARCH//./}"
    case "$ARCH" in ''|*[!0-9]*) echo "build.sh: set HPCPERF_CUDA_ARCH (for example 90)" >&2; exit 1;; esac
    BUILD="$R/build/level2/gamess_ri_mp2/cuda"; mkdir -p "$BUILD"
    read -r -a MPI_COMPILE <<< "$($MPIFC --showme:compile 2>/dev/null)"
    read -r -a MPI_LINK_RAW <<< "$($MPIFC --showme:link 2>/dev/null)"
    MPI_LINK=()
    for flag in "${MPI_LINK_RAW[@]}"; do
        case "$flag" in -lmpi_usempif08|-lmpi_usempi_ignore_tkr) ;; *) MPI_LINK+=("$flag");; esac
    done
    CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda}"
    cd "$BUILD"
    echo "== GAMESS RI-MP2 CUDA build: cc${ARCH} -> $BUILD"
    "$NVFORTRAN" -cpp -mp=gpu -gpu="cc${ARCH}" -O3 -DCUBLAS \
      -c "$HERE/source/cublasf.f90" -o cublasf.o
    "$NVFORTRAN" -cpp -mp=gpu -gpu="cc${ARCH}" -O3 -DCUBLAS -DHPCPERF_MINIMAL_MPIF \
      "${MPI_COMPILE[@]}" "$HERE/source/rimp2_energy_whole_KERN.f90" cublasf.o \
      "${MPI_LINK[@]}" -L"$CUDA_ROOT/lib64" -Wl,-rpath,"$CUDA_ROOT/lib64" -lcublas -lcudart \
      -o rimp2-cublas
    EXE="$BUILD/rimp2-cublas"
    ;;
  HIP)
    command -v hipcc >/dev/null 2>&1 || { echo "build.sh: hipcc not found; install/source ROCm" >&2; exit 1; }
    HIPFC="${HIPFORT_FC:-${HIPFC:-$(command -v hipfc || true)}}"
    [ -n "$HIPFC" ] || { echo "build.sh: hipfc not found; install hipfort and set HIPFORT_FC or HIPFC" >&2; exit 1; }
    ARCH="${HPCPERF_HIP_ARCH:-${HIP_ARCH:-}}"
    [ -n "$ARCH" ] || ARCH="$(rocminfo 2>/dev/null | sed -n 's/^[[:space:]]*Name:[[:space:]]*\(gfx[0-9a-fA-F]*\).*/\1/p' | head -1)"
    case "$ARCH" in gfx[0-9a-fA-F]*) ;; *) echo "build.sh: set HPCPERF_HIP_ARCH (for example gfx942)" >&2; exit 1;; esac
    ROCM_ROOT="${ROCM_PATH:-${ROCM_ROOT:-/opt/rocm}}"
    HIPBLAS_LIB="$(compgen -G "$ROCM_ROOT/lib*/libhipblas.so*" | head -1)"
    [ -n "$HIPBLAS_LIB" ] || { echo "build.sh: hipBLAS unavailable under ROCM_PATH=$ROCM_ROOT" >&2; exit 1; }
    HIPBLAS_LIBDIR="$(dirname "$HIPBLAS_LIB")"
    HIPFC_VERSION="$($HIPFC --version 2>&1 | head -4)"
    MPI_COMPILE=(); MPI_LINK=()
    if [ -n "$MPIFC" ]; then
      read -r -a MPI_COMPILE <<< "$($MPIFC --showme:compile 2>/dev/null || true)"
      read -r -a MPI_LINK <<< "$($MPIFC --showme:link 2>/dev/null || true)"
    elif [[ "$HIPFC_VERSION" != *Cray* ]]; then
      echo "build.sh: MPI Fortran compiler wrapper not found (mpifort); set MPIFC or use a Cray hipfc configured through the Cray MPI wrapper" >&2
      exit 1
    fi
    HIP_ARCH_FLAGS=()
    if [ "${HPCPERF_HIP_ARCH_FLAG+x}" = x ]; then
      read -r -a HIP_ARCH_FLAGS <<< "$HPCPERF_HIP_ARCH_FLAG"
    elif [[ "$HIPFC_VERSION" == *Cray* ]]; then
      ACCEL_TARGET="${PE_ACCEL_TARGET:-${CRAY_ACCEL_TARGET:-}}"
      [[ "$ACCEL_TARGET" == *"$ARCH"* ]] || {
        echo "build.sh: Cray hipfc requires the $ARCH accelerator target; load craype-accel-amd-$ARCH or set HPCPERF_HIP_ARCH_FLAG" >&2
        exit 1
      }
    else
      HIP_ARCH_FLAGS=("--offload-arch=$ARCH")
    fi
    BUILD="$R/build/level2/gamess_ri_mp2/hip"; mkdir -p "$BUILD"; cd "$BUILD"
    echo "== GAMESS RI-MP2 HIP build: $ARCH -> $BUILD"
    # -homp is the authoritative Crusher/CCE route. Sites using another
    # hipfort compiler can override only this compiler-specific flag.
    "$HIPFC" ${HPCPERF_HIP_OPENMP_FLAG:--homp} -O3 -DHIPBLAS "${HIP_ARCH_FLAGS[@]}" \
      "${MPI_COMPILE[@]}" "$HERE/source/rimp2_energy_whole_KERN.f90" "${MPI_LINK[@]}" \
      -L"$HIPBLAS_LIBDIR" -Wl,-rpath,"$HIPBLAS_LIBDIR" -lhipblas -o rimp2-hipblas
    EXE="$BUILD/rimp2-hipblas"
    ;;
  *) echo "usage: $0 [CUDA|HIP]" >&2; exit 2;;
esac
echo "== built: $EXE"
