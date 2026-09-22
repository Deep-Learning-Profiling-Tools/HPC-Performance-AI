"""TPU adapter (XLA profiler / XProf trace) -- INTERFACE ONLY, by project decision.

None of the Level 1/2/3 applications can run on a TPU: TPUs execute XLA programs
(JAX, TensorFlow, PyTorch/XLA), not CUDA or HIP C++. A TPU workload therefore marks its
ROI from Python (tools/timing/roi/hpcperf_roi.py) and is profiled with the framework's
own profiler. This module fixes the contract such an adapter must meet; it is not
implemented.

  measurement (collector "tpu_xprof"):
      the workload itself starts/stops the profiler (jax.profiler.start_trace /
      stop_trace into <raw>/prof); hpcperf_roi.jax_annotator() emits the
      "hpcperf:roi" / "hpcperf:exclude" ranges as TraceAnnotation events.

  mapping to the canonical model:
      device ops on the TPU timeline (HLO ops / fusions)  -> "compute"
      infeed / outfeed / host transfers                    -> "copy_h2d" / "copy_d2h"
      collectives (all-reduce, all-gather, ...)            -> "collective"
      TraceAnnotation "hpcperf:roi" / ":exclude"           -> Marker
      runtime_calls()                                      -> None unless the trace exposes
                                                              per-dispatch host calls; there is
                                                              no CUDA-style launch per op
  CAPABILITIES must list only what the trace really exposes; everything else is null.
  "launches" does not carry over from GPUs -- compute_ops counts device ops as recorded.

  Before VERIFIED = True: a JAX version of the conformance probe (same phases as
  probe.cu: warm-up, ROI with an excluded check, work after) must reproduce its
  expected split exactly.
"""

NAME = "tpu_xprof"
RUNTIME = "xla"
VERIFIED = False
CAPABILITIES = frozenset({"compute", "copy_h2d", "copy_d2h", "collective"})


def open(raw_dir):  # noqa: A001
    raise NotImplementedError("collector tpu_xprof is interface-only (project decision); see its docstring")
