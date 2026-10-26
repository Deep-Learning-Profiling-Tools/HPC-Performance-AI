"""Python twin of hpcperf_golden_output() (tools/golden/hpcperf_golden.h) for checkers written in Python.

A verify.py that recomputes a benchmark's result on the CPU is run only by `hpcperf_golden.py make`
(the one-time validation of a golden reference); when HPCPERF_GOLDEN_DUMP is set it captures its CPU
reference array here, under the name the golden spec gives it (`cpu:`), so make can measure how far
the B200 output is from the CPU reference. Same files and manifest format as the C header.

    import sys, os; sys.path.insert(0, os.path.join(<repo>, "tools", "golden"))
    import hpcperf_golden_capture as gc
    gc.capture("image.cpu", ref_array, "f32")      # a no-op unless HPCPERF_GOLDEN_DUMP is set
"""
import hashlib, json, os

DTYPES = {"f32": "<f4", "f64": "<f8", "i8": "i1", "u8": "u1", "i16": "<i2", "u16": "<u2",
          "i32": "<i4", "u32": "<u4", "i64": "<i8", "u64": "<u8"}


def enabled():
    return bool(os.environ.get("HPCPERF_GOLDEN_DUMP"))


def capture(name, array, dtype):
    d = os.environ.get("HPCPERF_GOLDEN_DUMP")
    if not d:
        return
    capture_to(d, name, array, dtype)


def capture_to(d, name, array, dtype):
    """capture() into an explicit directory (hpcperf_golden.py: values read from a run's log)."""
    import numpy as np
    if dtype not in DTYPES:
        raise ValueError(f"unknown dtype {dtype}")
    data = np.ascontiguousarray(np.asarray(array).reshape(-1).astype(DTYPES[dtype])).tobytes()
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{name}.bin"), "wb") as fh:
        fh.write(data)
    rec = {"name": name, "dtype": dtype, "count": len(data) // np.dtype(DTYPES[dtype]).itemsize, "bytes": len(data),
           "sha256": hashlib.sha256(data).hexdigest(), "data": True}
    with open(os.path.join(d, "manifest.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
