#!/usr/bin/env bash
# Run the standard SHAW benchmark problem on the CUDA or HIP build.
#
#   ./run.sh [CUDA|HIP] [extra shawExe/Kokkos args, e.g. --kokkos-num-threads=1]
#
# Benchmark problem: upstream demo1 (docs/demos/demo1, demos/demo1/input.yaml)
# scaled to a GPU-sized grid. demo1 is the full-mantle axisymmetric shear-wave
# problem: PREM material, Ricker-wavelet source at 640 km depth (period 65 s,
# delay 180 s), 2000 s of simulated propagation, CFL and numerical-dispersion
# checks enabled. demo1 uses a 200 (radial) x 1000 (angular) grid with
# dt = 0.25 s; the default here is a 1000 x 5000 grid (5.0e6 velocity +
# 1.0e7 stress dofs) with dt scaled to keep the same CFL number (0.05 s,
# 40000 leapfrog steps). The mesh is generated once with the upstream python
# mesher (meshing/create_single_mesh.py, needs numpy + scipy; ~2.5 min and
# 1.3 GB for 1000x5000) and cached under $R/build/level2/shaw/mesh/.
#
# I/O: by default no io section is written (kernel-only time loop; the
# upstream template documents the io section as optional). demo1 also
# collects a seismogram at 3 surface receivers and 80 binary snapshots of the
# full state; SHAW implements data collection with a full device->host copy
# of the state at every time step, which on a GPU costs far more than the
# kernels themselves (see README.md, "Run"). HPCPERF_SHAW_SEISMOGRAM=1 adds
# demo1's seismogram section (upstream's normal output "seismogram_0").
#
# Upstream's normal output is printed. `loopTime` is the wall time of the time
# loop; NOTE that the per-step kernel timers behind totTime/aveTime/
# aveBandwidth/aveGFlop are taken before the fence and therefore measure only
# kernel-launch time on a GPU (upstream comment: "need to fix timers for
# async launch").
#
# Environment overrides:
#   HPCPERF_SHAW_NR, HPCPERF_SHAW_NTH  radial / angular grid points (1000 / 5000)
#   HPCPERF_SHAW_DT           time step [s]; default keeps demo1's CFL number
#                             (0.25 * hRef/hRef_demo1, hRef = min(dr, dtheta*rCMB))
#   HPCPERF_SHAW_FINAL_TIME   final time [s] (2000)
#   HPCPERF_SHAW_SEISMOGRAM   1 = add demo1's seismogram section (default 0)
#   HPCPERF_SHAW_INPUT        use this yaml instead of generating one (the mesh
#                             directory it references must exist; no mesh is
#                             generated); relative paths are resolved from cwd
#   HPCPERF_SHAW_MESH_DIR     mesh cache directory ($R/build/level2/shaw/mesh)
#   HPCPERF_PYTHON            python interpreter for the mesher (python3)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$R/hpcperf_env.sh" 2>/dev/null || true

set -euo pipefail

BACKEND="$(echo "${1:-CUDA}" | tr '[:lower:]' '[:upper:]')"
[ $# -gt 0 ] && shift
MODEL="$(echo "$BACKEND" | tr '[:upper:]' '[:lower:]')"
BUILD_DIR="$R/build/level2/shaw/$MODEL"
EXE="$BUILD_DIR/shawExe"

if [ ! -x "$EXE" ]; then
    echo "run.sh: $EXE not found -- run ./build.sh $BACKEND first" >&2
    exit 1
fi

NR="${HPCPERF_SHAW_NR:-1000}"
NTH="${HPCPERF_SHAW_NTH:-5000}"
FINAL_TIME="${HPCPERF_SHAW_FINAL_TIME:-2000.0}"
MESH_CACHE="${HPCPERF_SHAW_MESH_DIR:-$R/build/level2/shaw/mesh}"
PYTHON="${HPCPERF_PYTHON:-python3}"

# Output (input yaml, seismogram, log) goes under the build tree, not the caller's cwd.
OUT_DIR="$BUILD_DIR/run"
mkdir -p "$OUT_DIR"

if [ -n "${HPCPERF_SHAW_INPUT:-}" ]; then
    INPUT="$(cd "$(dirname "$HPCPERF_SHAW_INPUT")" && pwd)/$(basename "$HPCPERF_SHAW_INPUT")"
    [ -f "$INPUT" ] || { echo "run.sh: HPCPERF_SHAW_INPUT=$HPCPERF_SHAW_INPUT not found" >&2; exit 1; }
else
    # ---- 1. mesh (generated once, cached) --------------------------------
    MESH="$MESH_CACHE/mesh${NR}x${NTH}"
    if [ ! -f "$MESH/mesh_info.dat" ] || [ ! -f "$MESH/graph_vp.dat" ] || \
       [ ! -f "$MESH/graph_sp.dat" ] || [ ! -f "$MESH/coeff_vp.dat" ]; then
        echo "# SHAW: generating mesh ${NR}x${NTH} into $MESH (once; python + numpy + scipy)"
        mkdir -p "$MESH_CACHE/.work"
        rm -rf "$MESH"
        # The mesher writes *.dat into the cwd and then moves them into
        # <working-dir>/mesh<nr>x<nth>, so run it from a scratch directory.
        ( cd "$MESH_CACHE/.work" && PYTHONDONTWRITEBYTECODE=1 \
          "$PYTHON" "$HERE/meshing/create_single_mesh.py" -nr "$NR" -nth "$NTH" -working-dir "$MESH_CACHE" )
        [ -f "$MESH/mesh_info.dat" ] || { echo "run.sh: mesh generation failed" >&2; exit 1; }
    fi

    # ---- 2. time step: keep demo1's CFL number unless overridden ----------
    # SHAW's CFL check (check_cfl.hpp): dt*sqrt(2)*maxVs/hRef <= 0.28 with
    # hRef = min(dr, minArc, maxArc); minArc = dtheta*rCMB governs for these
    # aspect ratios. demo1: 200x1000, dt 0.25 -> cfl 0.235. Nominal spacings
    # (mantle thickness / NR, pi*rCMB / NTH) are used so that dt comes out as
    # a round number (0.1 for 500x2500, 0.05 for 1000x5000). SHAW takes
    # numSteps = (size_t)(finalTime/dt) and requires the seismogram frequency
    # (4) to divide it, so the step count is fixed first (rounded up to a
    # multiple of 4, i.e. dt is never larger than the CFL-scaled value) and dt
    # is then chosen so that finalTime/dt truncates to exactly that count.
    if [ -n "${HPCPERF_SHAW_DT:-}" ]; then
        DT="$HPCPERF_SHAW_DT"
    else
        DT="$(awk -v nr="$NR" -v nth="$NTH" -v T="$FINAL_TIME" -v freq=4 'BEGIN{
                pi=3.14159265358979; dr=2891.0/nr; arc=3480.0*pi/nth;
                h=(dr<arc)?dr:arc; h0=3480.0*pi/1000.0; dtc=0.25*h/h0;
                n=int(T/dtc); if (n*dtc < T*(1-1e-12)) n++;
                n=freq*int((n+freq-1)/freq);
                dt=T/n; s=sprintf("%.15g", dt);
                while (int(T/(s+0)) < n) { dt=dt*(1-1e-12); s=sprintf("%.15g", dt) }
                print s }')"
    fi

    # ---- 3. input yaml (sections as in input_template.yaml / demo1) -------
    INPUT="$OUT_DIR/input.yaml"
    {
        echo "# generated by level2/shaw/run.sh from upstream demo1 (demos/demo1/input.yaml)"
        echo "general:"
        echo "  meshDir: $MESH"
        echo "  dt: $DT"
        echo "  finalTime: $FINAL_TIME"
        echo "  checkNumericalDispersion: true"
        echo "  checkCfl: true"
        echo
        if [ "${HPCPERF_SHAW_SEISMOGRAM:-0}" = "1" ]; then
            echo "io:"
            echo "  seismogram:"
            echo "    binary: false"
            echo "    freq: 4"
            echo "    receivers: [5, 30, 80]"
            echo
        fi
        echo "source:"
        echo "  signal: {kind: ricker, depth: 640.0, period: 65.0, delay: 180.0}"
        echo
        echo "material:"
        echo "  kind: prem"
    } > "$INPUT"
fi

echo "# SHAW $BACKEND: (cd $OUT_DIR && $EXE $INPUT $*)"
echo "# ---- input.yaml ----"
sed 's/^/# /' "$INPUT"
echo "# --------------------"
cd "$OUT_DIR"
exec "$EXE" "$INPUT" "$@"
