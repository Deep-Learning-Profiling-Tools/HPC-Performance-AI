# tools/timing/lib/collectors.sh -- the measurement-side half of each collector.
# (The analysis-side half is tools/timing/collectors/<name>.py.) Sourced by engine.sh.
#
# A collector here provides three functions:
#   collector_available <name>                 exit 0 if its tool is on PATH
#   collector_wrap <name> <outbase>            set the array COLLECTOR_ARGV (prefix for the command)
#   collector_export <name> <outbase> <dir>    turn the tool's output into what the adapter reads
# and the platform (backend) side provides
#   backend_env_allow <backend>                variable NAMES a run on that backend needs

collector_available() {
    case "$1" in
        nvidia_nsys) command -v nsys >/dev/null 2>&1 ;;
        none)        return 0 ;;
        *)           return 1 ;;
    esac
}

collector_version() {
    case "$1" in
        nvidia_nsys) nsys --version 2>/dev/null | sed -n 's/.*version //p' | head -1 ;;
        none)        echo "-" ;;
        *)           echo "-" ;;
    esac
}

# Only what the analysis reads is collected: CUDA API + device activity + NVTX ranges.
# No CPU sampling (-s none --cpuctxsw=none): no metric uses it and it only adds
# overhead (measured on quicksilver: -s none 15.4 s vs process-tree 16.3 s vs +cpuctxsw
# 17.7 s). No --cuda-memory-usage: it segfaults MiniEM inside cudaFreeAsync and feeds
# no metric. --cuda-graph-trace=node: kernels launched from a CUDA graph are traced one
# by one, so they are counted and clipped like any other launch.
collector_wrap() {
    case "$1" in
        nvidia_nsys)
            COLLECTOR_ARGV=(nsys profile -o "$2" -f true -t cuda,nvtx -s none --cpuctxsw=none
                            --cuda-graph-trace=node --stats=false --) ;;
        none)
            COLLECTOR_ARGV=() ;;
        amd_rocprofv3|tpu_xprof)
            echo "collector $1 is interface-only and UNVERIFIED (see tools/timing/collectors/$1.py)" >&2
            return 2 ;;
        *)
            echo "unknown collector '$1'" >&2
            return 2 ;;
    esac
}

collector_export() {
    case "$1" in
        nvidia_nsys)
            [ -f "$2.nsys-rep" ] || return 1
            nsys export --type sqlite -f true -o "$3/trace.sqlite" "$2.nsys-rep" > "$3/export.log" 2>&1 ;;
        none)
            return 0 ;;
        *)
            return 2 ;;
    esac
}

# Device visibility and runtime variables. Everything else is dropped by `env -i`.
backend_env_allow() {
    case "$1" in
        CUDA) echo "CUDA_HOME CUDA_PATH CUDA_VISIBLE_DEVICES CUDA_CACHE_PATH CUDA_DEVICE_ORDER
                    CUDA_MODULE_LOADING GPU_DEVICE_ORDINAL NVIDIA_VISIBLE_DEVICES NVIDIA_DRIVER_CAPABILITIES" ;;
        HIP)  echo "HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL HSA_OVERRIDE_GFX_VERSION
                    ROCM_PATH HIP_PATH" ;;                                   # UNVERIFIED
        XLA)  echo "TPU_NAME TPU_VISIBLE_CHIPS XLA_FLAGS JAX_PLATFORMS" ;;   # interface only
        *)    echo "" ;;
    esac
}
