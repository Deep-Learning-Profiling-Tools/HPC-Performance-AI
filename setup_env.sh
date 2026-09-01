#!/usr/bin/env bash
# HPC-Performance-AI one-time environment setup.
#
# Usage (after git clone, from the repository root):
#   ./setup_env.sh
#
# Installs everything user-space and project-local so a fresh clone gets the
# exact validated toolchain:
#   - bootstraps Miniforge into .tools/miniforge3 if no conda exists,
#   - creates/updates .conda_env from environment.yml (all versions pinned),
#   - installs cloc v2.06 into .tools/bin,
#   - validates the GCC <-> NVCC toolchain and records the choice.
# It never touches shell startup files, sudo, /usr/local/cuda, the NVIDIA
# driver, conda base, or anything in $HOME.
#
# NOT installed here (system prerequisites, see README): NVIDIA driver,
# CUDA Toolkit (expected at /usr/local/cuda or $CUDA_HOME), ROCm.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
CONDA_ENV_PREFIX="$ROOT/.conda_env"
CLOC_VERSION=2.06
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname -s)-$(uname -m).sh"
FAILURES=0

say()  { printf '%s\n' "$*"; }
ok()   { printf '[OK]   %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*"; FAILURES=$((FAILURES+1)); }

fetch() { # $1 = url, $2 = output file
    if command -v curl >/dev/null 2>&1; then curl -fsSL -o "$2" "$1";
    elif command -v wget >/dev/null 2>&1; then wget -qO "$2" "$1";
    else return 1; fi
}

cd "$ROOT" || { echo "cannot cd to $ROOT"; exit 1; }
mkdir -p "$ROOT/.tools/bin"

# ------------------------------------------------------------------ 1. Conda
# Order: conda already on PATH -> project-local Miniforge -> known site
# install (dev machine) -> bootstrap a project-local Miniforge.
CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
fi
for cand in "$ROOT/.tools/miniforge3" "/projects/kzhou6/bcui2/env_software/miniconda3"; do
    [ -n "$CONDA_BASE" ] && break
    [ -x "$cand/bin/conda" ] && CONDA_BASE="$cand"
done
if [ -z "$CONDA_BASE" ]; then
    say "No conda found -- bootstrapping project-local Miniforge into .tools/miniforge3"
    if fetch "$MINIFORGE_URL" "$ROOT/.tools/miniforge.sh"; then
        bash "$ROOT/.tools/miniforge.sh" -b -p "$ROOT/.tools/miniforge3" \
            && CONDA_BASE="$ROOT/.tools/miniforge3"
        rm -f "$ROOT/.tools/miniforge.sh"
    fi
    [ -n "$CONDA_BASE" ] || { fail "could not bootstrap Miniforge (no network or no curl/wget?)"; exit 1; }
fi
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    . "$CONDA_BASE/etc/profile.d/conda.sh"
    ok "Conda: $(conda --version 2>/dev/null) (base: $CONDA_BASE)"
else
    fail "conda base at $CONDA_BASE has no profile.d/conda.sh"
    exit 1
fi

# ------------------------------------------- 2. project-local Conda environment
if [ -d "$CONDA_ENV_PREFIX" ]; then
    say "Updating existing project env: $CONDA_ENV_PREFIX"
    conda env update --prefix "$CONDA_ENV_PREFIX" -f "$ROOT/environment.yml" \
        || fail "conda env update failed"
else
    say "Creating project env: $CONDA_ENV_PREFIX (all versions pinned in environment.yml)"
    conda env create --prefix "$CONDA_ENV_PREFIX" -f "$ROOT/environment.yml" \
        || fail "conda env create failed"
fi
[ -x "$CONDA_ENV_PREFIX/bin/python" ] || { fail "project env has no python"; exit 1; }
set +u; conda activate "$CONDA_ENV_PREFIX"; set -u  # conda activation scripts are not set-u clean

# ----------------------------------------------------------- 3. .tools / cloc
if [ -x "$ROOT/.tools/bin/cloc" ]; then
    ok "cloc already installed"
else
    say "Installing cloc v$CLOC_VERSION into .tools/bin (cloc is not on conda-forge)"
    if fetch "https://github.com/AlDanial/cloc/releases/download/v$CLOC_VERSION/cloc-$CLOC_VERSION.pl" \
             "$ROOT/.tools/bin/cloc"; then
        chmod +x "$ROOT/.tools/bin/cloc"
    else
        fail "cloc download failed"
    fi
fi
export PATH="$ROOT/.tools/bin:$PATH"

# --------------------------------------------------------- 4. version report
report() { # $1 label, $2 command...
    local label="$1"; shift
    local out
    if out="$("$@" 2>/dev/null | head -1)"; then
        ok "$label: $out"
    else
        fail "$label: not available ($1)"
    fi
}
report "Python"  python --version
report "CMake"   cmake --version
report "Ninja"   ninja --version
report "cloc"    cloc --version
report "numpy"   python -c "import numpy; print(numpy.__version__)"
CONDA_GCC="$CONDA_ENV_PREFIX/bin/x86_64-conda-linux-gnu-cc"
CONDA_GXX="$CONDA_ENV_PREFIX/bin/x86_64-conda-linux-gnu-c++"
if [ -x "$CONDA_GCC" ]; then
    ok "Conda GCC: $("$CONDA_GCC" -dumpfullversion)"
else
    warn "Conda GCC not present in project env"
fi
[ -x /usr/bin/gcc ] && ok "System GCC: $(/usr/bin/gcc -dumpfullversion)" || warn "no system gcc"

# ------------------------------------------------------------- 5. CUDA / GPU
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [ -x "$CUDA_HOME/bin/nvcc" ]; then
    ok "CUDA_HOME=$CUDA_HOME -> $(readlink -f "$CUDA_HOME")"
    ok "NVCC: $("$CUDA_HOME/bin/nvcc" --version | grep release)"
else
    fail "nvcc not found under CUDA_HOME=$CUDA_HOME (install the CUDA toolkit or export CUDA_HOME)"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    ok "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
else
    warn "nvidia-smi not available (no GPU visible -- benchmarks will build but not run)"
fi
if [ -d /opt/nvidia/nsight-compute/2026.1.1/extras/python ]; then
    ok "Nsight Compute python extras present"
else
    say "[INFO] Nsight Compute python extras not found (optional, profiling only)"
fi
if command -v hipcc >/dev/null 2>&1 || [ -d /opt/rocm ]; then
    ok "ROCm present"
else
    say "[INFO] ROCm unavailable (expected on NVIDIA machines; HIP backends stay unverified)"
fi

# ----------------------------------- 6. GCC + NVCC host-compiler compatibility
if [ -x "$CUDA_HOME/bin/nvcc" ]; then
    TESTDIR="$ROOT/.tools/nvcc_host_test"
    mkdir -p "$TESTDIR"
    cat > "$TESTDIR/saxpy.cu" <<'CU'
#include <cstdio>
#include <cmath>
#include <vector>
__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
int main() {
    const int n = 1 << 20;
    std::vector<float> hx(n, 1.0f), hy(n, 2.0f);
    float *dx, *dy;
    if (cudaMalloc(&dx, n * sizeof(float)) != cudaSuccess) { printf("SKIP no GPU\n"); return 0; }
    cudaMalloc(&dy, n * sizeof(float));
    cudaMemcpy(dx, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    saxpy<<<(n + 255) / 256, 256>>>(n, 3.0f, dx, dy);
    cudaMemcpy(hy.data(), dy, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
        if (std::fabs(hy[i] - 5.0f) > 1e-6f) { printf("FAIL at %d: %f\n", i, hy[i]); return 1; }
    printf("PASS\n");
    return 0;
}
CU
    try_host_compiler() { # $1 label, $2 c++ path, $3 marker
        local label="$1" cxx="$2" marker="$3"
        [ -x "$cxx" ] || { warn "$label host compiler missing: $cxx"; return 1; }
        if "$CUDA_HOME/bin/nvcc" -std=c++20 -ccbin "$cxx" \
             -o "$TESTDIR/saxpy_$marker" "$TESTDIR/saxpy.cu" 2> "$TESTDIR/build_$marker.log"; then
            local run
            run="$("$TESTDIR/saxpy_$marker" 2>&1)"
            if [ "$run" = "PASS" ]; then
                ok "NVCC + $label GCC ($("$cxx" -dumpfullversion)): compile+run PASS"
                return 0
            elif [ "$run" = "SKIP no GPU" ]; then
                warn "NVCC + $label GCC: compiled, but no GPU visible to run"
                return 0
            else
                fail "NVCC + $label GCC: runtime check failed: $run"
                return 1
            fi
        else
            fail "NVCC + $label GCC: compile failed (see $TESTDIR/build_$marker.log)"
            return 1
        fi
    }
    COMPILER_SOURCE=""
    if try_host_compiler "Conda" "$CONDA_GXX" conda; then
        COMPILER_SOURCE="conda"
    elif try_host_compiler "system" /usr/bin/g++ system; then
        COMPILER_SOURCE="system"
    fi
    if [ -n "$COMPILER_SOURCE" ]; then
        echo "$COMPILER_SOURCE" > "$ROOT/.tools/compiler_source"
        ok "Selected NVCC host compiler source: $COMPILER_SOURCE (recorded in .tools/compiler_source)"
    else
        fail "No working NVCC host compiler found"
    fi
else
    warn "Skipping the NVCC host-compiler test (no nvcc)."
fi

say ""
if [ "$FAILURES" -eq 0 ]; then
    say "setup_env.sh finished with no failures."
else
    say "setup_env.sh finished with $FAILURES failure(s) -- see [FAIL] lines above."
fi
say "Next: source hpcperf_env.sh && ./check_env.sh"
exit "$FAILURES"
