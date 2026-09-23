"""hpcperf_roi -- region-of-interest markers for Python workloads (JAX, PyTorch, ...).

The Python twin of hpcperf_roi.h, for accelerators whose programs are not C/C++:
TPUs run XLA programs, so a TPU workload marks its ROI from Python. Same contract,
same log format (tools/timing/summarize.py reads both):

    import hpcperf_roi as roi
    roi.set_device_sync(lambda: jax.effects_barrier())    # how to drain the device
    roi.add_annotator(*roi.jax_annotator())               # optional profiler ranges

    ... set-up, warm-up ...
    with roi.region(sync=True):
        ... the computation ...
        with roi.excluded():
            ... a check inside the ROI ...
    ... verification, output ...

OFF BY DEFAULT: unless HPCPERF_ROI_LOG is set (clean timing) or an annotator is
registered (profiling), every call is a cheap no-op.

Status: the clean-timing half is complete and tested CPU-only. The profiler half is an
INTERFACE: jax_annotator()/torch_annotator() adapt the frameworks' own trace
annotations but are UNVERIFIED -- no TPU, JAX or PyTorch on the node this was written
on. A TPU collector (tools/timing/collectors/tpu_xprof.py) is likewise interface-only.
"""

import atexit
import contextlib
import json
import os
import socket
import sys
import time

LOG_VERSION = 2
_FLUSH_AT = 4096          # events buffered before a flush (always outside an ROI)
_RANK_VARS = ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "SLURM_PROCID",
              "JAX_PROCESS_ID", "RANK")


class _State:
    def __init__(self):
        prefix = os.environ.get("HPCPERF_ROI_LOG", "")
        self.enabled = bool(prefix)
        self.path = f"{prefix}.{os.getpid()}" if prefix else ""
        self.depth = 0
        self.xdepth = 0
        self.events = []            # (kind, a, b): B/E/U -> monotonic_ns, realtime_ns; x -> excluded_ns, count
        self.xstart = 0
        self.xsum = 0
        self.xcount = 0
        self.header_written = False
        self.unmatched = 0
        self.sync = None
        self.annotators = []        # [(push(name), pop())]
        if self.enabled:
            atexit.register(self._atexit)

    @property
    def measuring(self):
        return self.enabled or bool(self.annotators)

    def record(self, kind):
        if self.enabled:
            self.events.append((kind, time.monotonic_ns(), time.time_ns()))

    def flush(self):
        if not self.enabled:
            return
        if not self.events and self.header_written and not self.unmatched:
            return
        with open(self.path, "a") as f:
            if not self.header_written:
                rank = next((os.environ[v] for v in _RANK_VARS if os.environ.get(v)), "-")
                f.write(f"# hpcperf-roi-log {LOG_VERSION}\n")
                f.write(f"pid {os.getpid()}\nrank {rank}\nclock CLOCK_MONOTONIC CLOCK_REALTIME\n")
                f.write(f"host {socket.gethostname()}\nexe {sys.executable}\ncwd {os.getcwd()}\n")
                f.write(f"argv {json.dumps(list(sys.argv))}\n")
                self.header_written = True
            for kind, mono, real in self.events:
                f.write(f"{kind} {mono} {real}\n")
            if self.unmatched:
                f.write(f"unmatched_end {self.unmatched}\n")
        self.events = []
        self.unmatched = 0

    def _atexit(self):
        if self.depth > 0:
            self.record("U")        # an ROI that never ended
        self.flush()


_S = _State()


def set_device_sync(fn):
    """fn() must block until all queued device work is complete (JAX: jax.effects_barrier
    or block_until_ready on the outputs; PyTorch: torch.cuda.synchronize / xm.wait_device_ops)."""
    _S.sync = fn


def add_annotator(push, pop):
    """Register a profiler annotation: push(name) opens a named range, pop() closes it."""
    _S.annotators.append((push, pop))


def _sync():
    if _S.sync is not None and _S.measuring:
        _S.sync()


def begin(sync=False):
    if sync:
        _sync()
    _S.depth += 1
    if _S.depth > 1:
        return
    for push, _ in _S.annotators:
        push("hpcperf:roi")
    _S.record("B")


def end(sync=False):
    if sync:
        _sync()
    if _S.depth <= 0:
        _S.unmatched += 1
        return
    _S.depth -= 1
    if _S.depth > 0:
        return
    if _S.xdepth > 0:
        _S.xdepth = 0
        for _, pop in reversed(_S.annotators):
            pop()
        if _S.enabled:
            _S.xsum += time.monotonic_ns() - _S.xstart
            _S.xcount += 1
    _S.record("E")
    if _S.enabled and _S.xcount:
        _S.events.append(("x", _S.xsum, _S.xcount))
    _S.xsum = _S.xcount = 0
    for _, pop in reversed(_S.annotators):
        pop()
    if len(_S.events) >= _FLUSH_AT:
        _S.flush()


def exclude_begin(sync=False):
    if _S.depth <= 0:
        return
    if sync:
        _sync()
    _S.xdepth += 1
    if _S.xdepth > 1:
        return
    if _S.enabled:
        _S.xstart = time.monotonic_ns()
    for push, _ in _S.annotators:
        push("hpcperf:exclude")


def exclude_end():
    if _S.depth <= 0 or _S.xdepth <= 0:
        return
    _S.xdepth -= 1
    if _S.xdepth > 0:
        return
    for _, pop in reversed(_S.annotators):
        pop()
    if _S.enabled:
        _S.xsum += time.monotonic_ns() - _S.xstart
        _S.xcount += 1


@contextlib.contextmanager
def region(sync=True):
    begin(sync=sync)
    try:
        yield
    finally:
        end(sync=sync)


@contextlib.contextmanager
def excluded(sync=False):
    exclude_begin(sync=sync)
    try:
        yield
    finally:
        exclude_end()


def _stack_annotator(make_ctx):
    stack = []

    def push(name):
        ctx = make_ctx(name)
        ctx.__enter__()
        stack.append(ctx)

    def pop():
        if stack:
            stack.pop().__exit__(None, None, None)
    return push, pop


def jax_annotator():
    """(push, pop) emitting jax.profiler.TraceAnnotation ranges. UNVERIFIED (no JAX here)."""
    import jax  # noqa: F401 -- imported only when asked for
    return _stack_annotator(lambda name: jax.profiler.TraceAnnotation(name))


def torch_annotator():
    """(push, pop) emitting torch.profiler.record_function ranges. UNVERIFIED."""
    import torch
    return _stack_annotator(lambda name: torch.profiler.record_function(name))
